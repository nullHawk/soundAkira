"""Publish and incrementally maintain the dataset on the Hugging Face Hub.

The Hub repo is the accumulated dataset. Every `soundakira push`:

1. reads the Hub's current `metadata.jsonl` and `manifest.json`;
2. merges: rows of sources that were built locally replace that source's old
   rows, and rows of all other sources (pushed earlier, maybe from another
   machine) are kept;
3. uploads only audio whose content hash changed, and deletes audio no row
   references any more;
4. rewrites `metadata.{jsonl,csv}`, `speakers.csv`, `dataset.json`,
   `speaker_registry.json`, `manifest.json` and the dataset card over the
   *merged* rows, and appends a history entry.

Speaker IDs stay consistent because `build` first pulls
`speaker_registry.json` from the Hub (`hub.sync_registry`). Clusters built
locally are then matched against every speaker ever pushed. A push refuses to
run if the local build was not based on the Hub registry.

Commits are chained with `parent_commit`, so a concurrent push fails cleanly
instead of overwriting.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from soundakira import __version__
from soundakira.config import PipelineConfig
from soundakira.dataset import summary
from soundakira.dataset.build import BASE_COLUMNS, choose_test_speakers
from soundakira.pipeline.workspace import now_iso
from soundakira.utils.io import read_json, read_jsonl

log = logging.getLogger(__name__)


class HubError(RuntimeError):
    pass


class HubBackend(Protocol):
    def ensure_repo(self, private: bool) -> None: ...
    def head(self) -> str | None: ...
    def read(self, path: str, revision: str | None) -> bytes | None: ...
    def commit(
        self, adds: dict[str, Path | bytes], deletes: list[str], message: str, parent: str | None
    ) -> str: ...


class HfBackend:
    """huggingface_hub implementation of `HubBackend`."""

    def __init__(self, repo_id: str, token: str | None, branch: str = "main"):
        from huggingface_hub import HfApi

        self.repo_id, self.branch, self.token = repo_id, branch, token
        self.api = HfApi(token=token)

    def ensure_repo(self, private: bool) -> None:
        self.api.create_repo(self.repo_id, repo_type="dataset", private=private, exist_ok=True)

    def whoami(self) -> str:
        return str(self.api.whoami()["name"])

    def head(self) -> str | None:
        try:
            info = self.api.repo_info(self.repo_id, repo_type="dataset", revision=self.branch)
            if info.id != self.repo_id:  # renamed on the Hub: follow it
                log.warning(
                    "Hub repo %s was renamed to %s; using the new name", self.repo_id, info.id
                )
                self.repo_id = info.id
            return info.sha
        except Exception as e:
            if type(e).__name__ in ("RepositoryNotFoundError", "RevisionNotFoundError"):
                return None
            raise

    def read(self, path: str, revision: str | None) -> bytes | None:
        from huggingface_hub import hf_hub_download

        try:
            local = hf_hub_download(
                self.repo_id,
                path,
                repo_type="dataset",
                revision=revision or self.branch,
                token=self.token,
            )
        except Exception as e:
            if "NotFound" in type(e).__name__:
                return None
            raise
        return Path(local).read_bytes()

    def commit(
        self, adds: dict[str, Path | bytes], deletes: list[str], message: str, parent: str | None
    ) -> str:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        ops: list[Any] = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(v) if isinstance(v, Path) else v)
            for p, v in adds.items()
        ]
        ops += [CommitOperationDelete(path_in_repo=p) for p in deletes]
        info = self.api.create_commit(
            self.repo_id,
            operations=ops,
            commit_message=message,
            repo_type="dataset",
            revision=self.branch,
            parent_commit=parent,
        )
        return str(info.oid)


def backend_for(cfg: PipelineConfig) -> HfBackend:
    repo = cfg.hub.resolved_repo_id()
    if not repo:
        raise HubError("no Hub repo configured: set SOUNDAKIRA_HF_REPO or hub.repo_id")
    return HfBackend(repo, cfg.hub.resolved_token(cfg.resolved_hf_token()), cfg.hub.branch)


# -- pure planning logic ---------------------------------------------------------
def merge_rows(
    remote: list[dict[str, Any]], local: list[dict[str, Any]], local_sources: set[str]
) -> list[dict[str, Any]]:
    """Local rows replace every remote row of the sources that were built locally."""
    kept = [r for r in remote if r["source_id"] not in local_sources]
    return sorted(kept + local, key=lambda r: (int(r["speaker_id"]), r["utt_id"]))


def referenced_files(rows: Iterable[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for r in rows:
        out.add(r["audio_path"])
        if r.get("ref_audio_path"):
            out.add(r["ref_audio_path"])
    return out


def plan_paths(
    needed: set[str], local: dict[str, str], remote: dict[str, str]
) -> tuple[list[str], list[str], dict[str, str]]:
    """(paths to upload, paths to delete, new manifest {path: sha1})."""
    missing = sorted(p for p in needed if p not in local and p not in remote)
    if missing:
        raise HubError(
            f"{len(missing)} referenced file(s) exist neither locally nor on the Hub, "
            f"e.g. {missing[0]}. Rebuild those sources locally and push again."
        )
    upload = sorted(p for p in needed if p in local and remote.get(p) != local[p])
    delete = sorted(p for p in remote if p not in needed)
    manifest = {p: local.get(p) or remote[p] for p in sorted(needed)}
    return upload, delete, manifest


def plan_files(
    rows: list[dict[str, Any]], local: dict[str, str], remote: dict[str, str]
) -> tuple[list[str], list[str], dict[str, str]]:
    """File layout: every clip and reference is its own audio file."""
    return plan_paths(referenced_files(rows), local, remote)


def check_registry_lineage(
    remote: dict[str, Any] | None, local: dict[str, Any], local_sources: set[str]
) -> None:
    """The local registry must have been built on top of the Hub's: every Hub
    speaker must still exist locally with the same ID, and keep its members
    from sources this build didn't touch. Otherwise IDs would collide."""
    if not remote:
        return
    lspk = local.get("speakers", {})
    bad = []
    for sid, entry in remote.get("speakers", {}).items():
        foreign = {m for m in entry.get("members", {}) if m.rsplit(":", 1)[0] not in local_sources}
        if sid not in lspk or not foreign <= set(lspk[sid].get("members", {})):
            bad.append(sid)
    if bad or local.get("next_id", 0) < remote.get("next_id", 0):
        raise HubError(
            "the local build was not based on the Hub's speaker registry (speaker IDs would "
            f"collide, e.g. speaker {bad[0] if bad else '?'}). Run `soundakira build` with "
            "hub.sync_registry enabled, then push again."
        )


# -- IO helpers ------------------------------------------------------------------
def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    import csv

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    return buf.getvalue().encode()


def _card(repo: str, totals: dict[str, Any], sample_rate: int, layout: str) -> bytes:
    name = repo.split("/")[-1]
    if layout == "parquet":
        data_files = "  data_files:\n  - split: train\n    path: data/*.parquet"
        usage = f"""```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train")
row = ds[0]
row["audio"]       # {{"array": ..., "sampling_rate": {sample_rate}}}: the utterance
row["ref_audio"]   # a different clip of the same speaker (voice prompt)
row["text"], row["speaker_id"], row["ref_text"]
```"""
        layout_note = (
            "Audio is stored in Parquet shards under `data/` (one per source) with `audio` "
            "and `ref_audio` columns."
        )
    else:
        data_files = "  data_files: metadata.csv"
        usage = f"""```python
from huggingface_hub import snapshot_download
import pandas as pd, soundfile as sf

root = snapshot_download("{repo}", repo_type="dataset")
meta = pd.read_csv(f"{{root}}/metadata.csv")
audio, sr = sf.read(f"{{root}}/" + meta.audio_path[0])
```"""
        layout_note = "Audio files live under `wavs/` (utterances) and `refs/` (prompts)."
    return f"""---
pretty_name: {name}
task_categories:
- text-to-speech
tags:
- audio
- speech
- voice-cloning
configs:
- config_name: default
{data_files}
---

# {name}

Speaker-labelled speech for zero-shot TTS and voice cloning. Every example is a
single-speaker utterance with its transcript, plus a reference prompt: a
different clip of the same speaker.

| | |
|---|---|
| utterances | {totals["total_utterances"]} |
| hours | {totals["total_hours"]} |
| speakers | {totals["total_speakers"]} |
| sources | {totals["num_sources"]} |
| sample rate | {sample_rate} Hz |

{layout_note} Speaker IDs are stable across updates (`speaker_registry.json`),
`speakers.csv` summarises the speakers, `metadata.jsonl` holds per-utterance
metadata and quality metrics, and `manifest.json` records the update history.

{usage}
""".encode()


PARQUET_FIELDS: list[tuple[str, str]] = [
    ("utt_id", "string"),
    ("audio", "audio"),
    ("text", "string"),
    ("text_tagged", "string"),
    ("speaker_id", "int64"),
    ("speaker_name", "string"),
    ("language", "string"),
    ("duration", "float64"),
    ("split", "string"),
    ("source_id", "string"),
    ("source_url", "string"),
    ("source_start", "float64"),
    ("source_end", "float64"),
    ("num_speakers_in_source", "int64"),
    ("ref_id", "string"),
    ("ref_audio", "audio"),
    ("ref_text", "string"),
    ("ref_duration", "float64"),
    ("ref_source_id", "string"),
    ("metrics", "string"),
    ("words", "string"),
]
_ROW_KEYS = set(BASE_COLUMNS) | {"words", "shard", "split"}


def write_parquet_shard(
    rows: list[dict[str, Any]], out_dir: Path, sample_rate: int, dest: Path
) -> None:
    """One shard with Hugging Face `Audio` columns (the Hub viewer renders them as
    players, and `load_dataset` decodes them). The schema is fixed; variable
    metrics go in a JSON column, so shards written by different versions still
    concatenate."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    arrow = {
        "string": pa.string(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "audio": audio_type,
    }

    def audio(rel: str | None) -> dict[str, Any] | None:
        if not rel:
            return None
        return {"bytes": (out_dir / rel).read_bytes(), "path": Path(rel).name}

    def value(r: dict[str, Any], name: str) -> Any:
        if name == "audio":
            return audio(r["audio_path"])
        if name == "ref_audio":
            return audio(r.get("ref_audio_path"))
        if name == "source_url":
            uri = r.get("source_uri") or ""
            return uri if uri.startswith(("http://", "https://")) else None
        if name == "metrics":
            return json.dumps({k: v for k, v in r.items() if k not in _ROW_KEYS}, sort_keys=True)
        if name == "words":
            return json.dumps(r.get("words", []), ensure_ascii=False)
        v = r.get(name)
        return None if v in ("", None) else v

    features = {
        name: {"sampling_rate": sample_rate, "_type": "Audio"}
        if kind == "audio"
        else {"dtype": kind, "_type": "Value"}
        for name, kind in PARQUET_FIELDS
    }
    schema = pa.schema(
        [pa.field(name, arrow[kind]) for name, kind in PARQUET_FIELDS],
        metadata={"huggingface": json.dumps({"info": {"features": features}})},
    )
    table = pa.table(
        {name: [value(r, name) for r in rows] for name, _ in PARQUET_FIELDS}, schema=schema
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest, compression="zstd")


def public_uri(uri: str | None) -> str | None:
    """URLs are published as-is; local paths are reduced to the file name, since
    they reveal machine/filesystem details."""
    if not uri or uri.startswith(("http://", "https://")):
        return uri
    return Path(uri).name


def shard_path(source_id: str) -> str:
    return f"data/{source_id}.parquet"


@dataclass
class PushReport:
    repo_id: str
    uploaded: int
    deleted: int
    total_utterances: int
    total_speakers: int
    commits: int
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- operations -------------------------------------------------------------------
def pull_registry(cfg: PipelineConfig, dest: Path, backend: HubBackend | None = None) -> bool:
    """Replace the local speaker registry with the Hub's (if the Hub has one)."""
    backend = backend or backend_for(cfg)
    head = backend.head()
    data = backend.read("speaker_registry.json", head) if head else None
    if data is None:
        log.info("Hub has no speaker registry yet; using the local one")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    log.info("pulled speaker registry from the Hub (%s)", cfg.hub.resolved_repo_id())
    return True


def push(
    cfg: PipelineConfig,
    backend: HubBackend | None = None,
    dry_run: bool = False,
    message: str | None = None,
) -> PushReport:
    backend = backend or backend_for(cfg)
    out = cfg.output_dir
    layout = cfg.hub.audio_layout
    meta_path = out / "metadata.jsonl"
    if not meta_path.exists():
        raise HubError(f"{meta_path} not found: run `soundakira build` (with export.write_jsonl)")
    local_rows = read_jsonl(meta_path)
    local_sources = set(read_json(out / "dataset.json").get("source_ids", []))
    local_sources |= {r["source_id"] for r in local_rows}
    registry_path = cfg.work_dir / "speakers" / "registry.json"
    local_registry = read_json(registry_path) if registry_path.exists() else {}

    # Resolve first (this also follows a rename): creating the repo up front
    # would make a new empty repo under the old, now-free name.
    head = backend.head()
    if head is None and not dry_run:
        backend.ensure_repo(cfg.hub.private)
        head = backend.head()
    repo = getattr(backend, "repo_id", None) or cfg.hub.resolved_repo_id() or ""
    who = getattr(backend, "whoami", None)
    if who is not None:
        log.info("Hub account: %s -> %s", who(), repo)
    remote_meta = backend.read("metadata.jsonl", head) if head else None
    remote_rows = (
        [json.loads(x) for x in remote_meta.decode().splitlines() if x.strip()]
        if remote_meta
        else []
    )
    remote_manifest_b = backend.read("manifest.json", head) if head else None
    remote_manifest = (
        json.loads(remote_manifest_b) if remote_manifest_b else {"files": {}, "history": []}
    )
    remote_reg_b = backend.read("speaker_registry.json", head) if head else None
    check_registry_lineage(
        json.loads(remote_reg_b) if remote_reg_b else None, local_registry, local_sources
    )

    merged = merge_rows(remote_rows, local_rows, local_sources)
    for r in merged:  # also cleans rows published by older versions
        r["source_uri"] = public_uri(r.get("source_uri"))
    split = cfg.export.split
    test = choose_test_speakers(
        sorted({int(r["speaker_id"]) for r in merged}), split.test_speaker_fraction, split.seed
    )
    for r in merged:
        r["split"] = "test" if int(r["speaker_id"]) in test else "train"

    # What the repo should contain, and the local files that provide it.
    staged: dict[str, Path] = {}
    if layout == "parquet":
        for r in merged:
            r["shard"] = shard_path(r["source_id"])
        needed = {r["shard"] for r in merged}
        stage_dir = out / ".hub_shards"
        shutil.rmtree(stage_dir, ignore_errors=True)
        for sid in sorted(local_sources):
            rows_s = [r for r in merged if r["source_id"] == sid]
            if rows_s:
                dest = stage_dir / f"{sid}.parquet"
                write_parquet_shard(rows_s, out, cfg.export.sample_rate, dest)
                staged[shard_path(sid)] = dest
    else:
        needed = referenced_files(merged)
        staged = {p: out / p for p in referenced_files(local_rows) if (out / p).exists()}
    local_hashes = {p: _sha1(f) for p, f in staged.items()}
    upload, delete, files = plan_paths(needed, local_hashes, remote_manifest.get("files", {}))

    speakers = summary.speaker_rows(merged)
    totals = summary.totals(merged, speakers)
    report = PushReport(repo, len(upload), len(delete), len(merged), len(speakers), 0, dry_run)
    log.info(
        "push plan (%s layout): %d rows (%d local), upload %d files, delete %d",
        layout,
        len(merged),
        len(local_rows),
        len(upload),
        len(delete),
    )
    if dry_run:
        return report

    metric_cols = sorted({k for r in merged for k in r} - set(BASE_COLUMNS) - {"words", "shard"})
    history = [
        *remote_manifest.get("history", []),
        {
            "time": now_iso(),
            "pipeline_version": __version__,
            "layout": layout,
            "sources": sorted(local_sources),
            "local_utterances": len(local_rows),
            "uploaded": len(upload),
            "deleted": len(delete),
            "total_utterances": len(merged),
        },
    ]
    dataset_json = {
        "updated_at": now_iso(),
        "pipeline_version": __version__,
        **totals,
        "sample_rate": cfg.export.sample_rate,
        "audio_layout": layout,
    }
    csv_cols = BASE_COLUMNS + (["shard"] if layout == "parquet" else []) + metric_cols
    final: dict[str, Path | bytes] = {
        "metadata.jsonl": _jsonl(merged),
        "metadata.csv": _csv_bytes(merged, csv_cols),
        "speakers.csv": _csv_bytes(speakers, summary.SPEAKER_COLUMNS),
        "dataset.json": json.dumps(dataset_json, indent=2).encode(),
        "manifest.json": json.dumps({"files": files, "history": history}, indent=1).encode(),
        "README.md": _card(repo, totals, cfg.export.sample_rate, layout),
    }
    if registry_path.exists():
        final["speaker_registry.json"] = registry_path.read_bytes()

    parent, n = head, cfg.hub.commit_batch_size
    batches = [upload[i : i + n] for i in range(0, len(upload), n)]
    for k, batch in enumerate(batches, 1):
        parent = backend.commit(
            {p: staged[p] for p in batch}, [], f"Add audio ({k}/{len(batches)})", parent
        )
        report.commits += 1
    parent = backend.commit(
        final,
        delete,
        message
        or f"Update dataset: +{len(local_rows)} utterances from {len(local_sources)} "
        f"source(s), {len(merged)} total",
        parent,
    )
    report.commits += 1
    return report
