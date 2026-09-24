"""Command-line interface.

soundakira init                       # write a commented config.yaml
soundakira process INPUTS... -c cfg   # per-source stages (resumable, shardable)
soundakira build -c cfg               # speakers + references + export
soundakira run INPUTS... -c cfg       # process + build
soundakira status -c cfg              # per-source stage status
soundakira doctor                     # check system + optional dependencies
soundakira components                 # list available components
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from soundakira import __version__
from soundakira.config import PipelineConfig, default_config_text, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = logging.getLogger("soundakira")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file.")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a config value: key.path=value (repeatable)."),
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


def _setup(config: Path | None, overrides: list[str] | None, verbose: bool) -> PipelineConfig:
    cfg = load_config(config, overrides)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "logs").mkdir(exist_ok=True)
    from soundakira.pipeline.workspace import now_iso

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(cfg.work_dir / "logs" / f"run-{now_iso().replace(':', '')}.log"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("urllib3", "httpx", "filelock", "numba", "matplotlib", "speechbrain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return cfg


def _parse_shard(shard: str | None) -> tuple[int, int] | None:
    if not shard:
        return None
    i, _, n = shard.partition("/")
    idx, total = int(i), int(n)
    if not 0 <= idx < total:
        raise typer.BadParameter("shard must be i/N with 0 <= i < N")
    return idx, total


def _process(
    cfg: PipelineConfig,
    inputs: list[str],
    shard: str | None,
    until: str | None,
    force: list[str] | None,
) -> int:
    from soundakira.pipeline.runner import Runner
    from soundakira.sources.resolve import resolve_inputs
    from soundakira.utils.hashing import shard_of

    sources = resolve_inputs(
        inputs, cfg.asr.language, cfg.fetch.expand_playlists, cfg.fetch.yt_dlp_options
    )
    parsed = _parse_shard(shard)
    if parsed:
        idx, total = parsed
        sources = [s for s in sources if shard_of(s.source_id, total) == idx]
    log.info("%d source(s) to process", len(sources))
    runner = Runner(cfg)
    summary = runner.run(runner.prepare(sources), until=until, force=force or [])
    for stage, counts in summary.counts.items():
        log.info("  %-10s %s", stage, dict(counts))
    if summary.failed:
        log.warning("%d stage failure(s); see `soundakira status` for details", summary.failed)
    return summary.failed


@app.command()
def init(
    path: Annotated[Path, typer.Argument()] = Path("config.yaml"),
    force: bool = typer.Option(False, help="Overwrite an existing file."),
) -> None:
    """Write the default, fully commented config file."""
    if path.exists() and not force:
        raise typer.BadParameter(f"{path} exists (use --force)")
    path.write_text(default_config_text(), encoding="utf-8")
    typer.echo(f"wrote {path}")


@app.command()
def process(
    inputs: Annotated[list[str], typer.Argument(help="Files, folders, URLs or manifest files.")],
    config: ConfigOpt = None,
    set_: SetOpt = None,
    shard: Annotated[str | None, typer.Option(help="Process shard i/N only.")] = None,
    until: Annotated[str | None, typer.Option(help="Stop after this stage.")] = None,
    force: Annotated[
        list[str] | None, typer.Option(help="Re-run this stage (and downstream).")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Run the per-source stages (fetch -> ... -> embed). Safe to interrupt and re-run."""
    cfg = _setup(config, set_, verbose)
    failed = _process(cfg, inputs, shard, until, force)
    raise typer.Exit(1 if failed else 0)


@app.command()
def build(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    clean: bool = typer.Option(False, help="Delete previously exported audio first."),
    verbose: VerboseOpt = False,
) -> None:
    """Cluster global speakers, pick references, and export the dataset."""
    from soundakira.dataset.build import build_dataset

    cfg = _setup(config, set_, verbose)
    report = build_dataset(cfg, clean=clean)
    typer.echo(json.dumps(report.to_dict(), indent=2))


@app.command()
def run(
    inputs: Annotated[list[str], typer.Argument(help="Files, folders, URLs or manifest files.")],
    config: ConfigOpt = None,
    set_: SetOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """process + build in one go."""
    from soundakira.dataset.build import build_dataset

    cfg = _setup(config, set_, verbose)
    _process(cfg, inputs, None, None, None)
    typer.echo(json.dumps(build_dataset(cfg).to_dict(), indent=2))


@app.command()
def status(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    errors: bool = typer.Option(False, help="Print failure messages."),
) -> None:
    """Show per-stage progress over all sources in work_dir."""
    from soundakira.pipeline.stages import STAGE_NAMES
    from soundakira.pipeline.workspace import list_workspaces

    cfg = load_config(config, set_)
    workspaces = list_workspaces(cfg.work_dir)
    counts: dict[str, Counter[str]] = {s: Counter() for s in STAGE_NAMES}
    failures = []
    for ws in workspaces:
        stages = ws.manifest()["stages"]
        for s in STAGE_NAMES:
            rec = stages.get(s)
            state = rec["status"] if rec else "pending"
            counts[s][state] += 1
            if state == "failed":
                failures.append((ws.source_id, s, rec.get("error", "")))
    typer.echo(f"{len(workspaces)} source(s) in {cfg.work_dir}")
    for s in STAGE_NAMES:
        c = counts[s]
        typer.echo(f"  {s:<11} done={c['done']:<5} failed={c['failed']:<5} pending={c['pending']}")
    if errors:
        for sid, stage, err in failures:
            typer.echo(f"  [{sid}] {stage}: {err}")


@app.command()
def components() -> None:
    """List registered components (built-in and plugins)."""
    from soundakira import registry

    for kind in registry.KINDS:
        typer.echo(f"{kind:<9} {', '.join(registry.available(kind))}")


@app.command()
def doctor() -> None:
    """Check ffmpeg, GPU, Hugging Face token and optional backends."""
    import os

    def line(ok: bool, label: str, detail: str = "") -> None:
        typer.echo(f"  [{'ok' if ok else '--'}] {label}{(': ' + detail) if detail else ''}")

    typer.echo(f"soundakira {__version__} on Python {sys.version.split()[0]}")
    for binary in ("ffmpeg", "ffprobe"):
        line(
            shutil.which(binary) is not None, binary, shutil.which(binary) or "not found (required)"
        )
    try:
        import torch

        cuda = torch.cuda.is_available()
        line(
            True,
            "torch",
            f"{torch.__version__}, cuda={cuda}"
            + (f" ({torch.cuda.device_count()} GPU)" if cuda else ""),
        )
    except ImportError:
        line(False, "torch", "not installed (needed by all model backends)")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    line(bool(token), "HF_TOKEN", "set" if token else "unset (pyannote models are gated)")
    for module, extra in [
        ("yt_dlp", "fetch"),
        ("audio_separator", "separation"),
        ("demucs", "demucs"),
        ("df", "denoise"),
        ("silero_vad", "vad"),
        ("pyannote.audio", "diarization"),
        ("faster_whisper", "asr"),
        ("nemo", "parakeet"),
        ("whisperx", "align"),
        ("torchmetrics", "quality"),
    ]:
        root = module.split(".")[0]
        found = importlib.util.find_spec(root) is not None
        line(found, module, "installed" if found else f"pip install 'soundakira[{extra}]'")


@app.callback(invoke_without_command=True)
def _version(version: bool = typer.Option(False, "--version", help="Show version.")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
