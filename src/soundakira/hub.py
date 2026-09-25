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
            return self.api.repo_info(self.repo_id, repo_type="dataset", revision=self.branch).sha
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


def plan_files(
    rows: list[dict[str, Any]], local: dict[str, str], remote: dict[str, str]
) -> tuple[list[str], list[str], dict[str, str]]:
    """(paths to upload, paths to delete, new manifest {path: sha1})."""
    needed = referenced_files(rows)
    missing = sorted(p for p in needed if p not in local and p not in remote)
    if missing:
        raise HubError(
            f"{len(missing)} referenced audio file(s) exist neither locally nor on "
            f"the Hub, e.g. {missing[0]}"
        )
    upload = sorted(p for p in needed if p in local and remote.get(p) != local[p])
    delete = sorted(p for p in remote if p not in needed)
    manifest = {p: local.get(p) or remote[p] for p in sorted(needed)}
    return upload, delete, manifest


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


def _card(repo: str, totals: dict[str, Any], sample_rate: int) -> bytes:
    return f"""---
pretty_name: {repo.split("/")[-1]}
task_categories:
- text-to-speech
tags:
- audio
- speech
- voice-cloning
- soundakira
configs:
- config_name: default
  data_files: metadata.csv
---

# {repo.split("/")[-1]}

Speaker-labelled speech for zero-shot TTS / voice cloning, built and maintained
with [soundAkira](https://github.com/nullHawk/soundAkira).

| | |
|---|---|
| utterances | {totals["total_utterances"]} |
| hours | {totals["total_hours"]} |
| speakers | {totals["total_speakers"]} |
| sources | {totals["num_sources"]} |
| sample rate | {sample_rate} Hz |

Each row of `metadata.csv` / `metadata.jsonl` is one single-speaker utterance
(`audio_path`, `text`, `speaker_id`) with a reference prompt of the same voice
(`ref_audio_path`, `ref_text`) and quality metrics. Speaker IDs are stable
across updates (`speaker_registry.json`); `speakers.csv` summarises them and
`manifest.json` records the update history.

```python
from huggingface_hub import snapshot_download
import pandas as pd, soundfile as sf

root = snapshot_download("{repo}", repo_type="dataset")
meta = pd.read_csv(f"{{root}}/metadata.csv")
audio, sr = sf.read(f"{{root}}/" + meta.audio_path[0])
```
""".encode()


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
    repo = cfg.hub.resolved_repo_id() or ""
    backend = backend or backend_for(cfg)
    out = cfg.output_dir
    meta_path = out / "metadata.jsonl"
    if not meta_path.exists():
        raise HubError(f"{meta_path} not found: run `soundakira build` (with export.write_jsonl)")
    local_rows = read_jsonl(meta_path)
    local_sources = set(read_json(out / "dataset.json").get("source_ids", []))
    local_sources |= {r["source_id"] for r in local_rows}
    registry_path = cfg.work_dir / "speakers" / "registry.json"
    local_registry = read_json(registry_path) if registry_path.exists() else {}

    who = getattr(backend, "whoami", None)
    if who is not None:
        log.info(
            "Hub account: %s -> %s (%s)", who(), repo, "private" if cfg.hub.private else "public"
        )
    if not dry_run:
        backend.ensure_repo(cfg.hub.private)
    head = backend.head()
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
    split = cfg.export.split
    test = choose_test_speakers(
        sorted({int(r["speaker_id"]) for r in merged}), split.test_speaker_fraction, split.seed
    )
    for r in merged:
        r["split"] = "test" if int(r["speaker_id"]) in test else "train"
    local_files = {p: _sha1(out / p) for p in referenced_files(local_rows) if (out / p).exists()}
    upload, delete, files = plan_files(merged, local_files, remote_manifest.get("files", {}))

    speakers = summary.speaker_rows(merged)
    totals = summary.totals(merged, speakers)
    report = PushReport(repo, len(upload), len(delete), len(merged), len(speakers), 0, dry_run)
    log.info(
        "push plan: %d rows (%d local), upload %d files, delete %d",
        len(merged),
        len(local_rows),
        len(upload),
        len(delete),
    )
    if dry_run:
        return report

    metric_cols = sorted({k for r in merged for k in r} - set(BASE_COLUMNS) - {"words"})
    history = [
        *remote_manifest.get("history", []),
        {
            "time": now_iso(),
            "soundakira_version": __version__,
            "sources": sorted(local_sources),
            "local_utterances": len(local_rows),
            "uploaded": len(upload),
            "deleted": len(delete),
            "total_utterances": len(merged),
        },
    ]
    dataset_json = {
        "updated_at": now_iso(),
        "soundakira_version": __version__,
        **totals,
        "sample_rate": cfg.export.sample_rate,
    }
    final: dict[str, Path | bytes] = {
        "metadata.jsonl": _jsonl(merged),
        "metadata.csv": _csv_bytes(merged, BASE_COLUMNS + metric_cols),
        "speakers.csv": _csv_bytes(speakers, summary.SPEAKER_COLUMNS),
        "dataset.json": json.dumps(dataset_json, indent=2).encode(),
        "manifest.json": json.dumps({"files": files, "history": history}, indent=1).encode(),
        "README.md": _card(repo, totals, cfg.export.sample_rate),
    }
    if registry_path.exists():
        final["speaker_registry.json"] = registry_path.read_bytes()

    parent, n = head, cfg.hub.commit_batch_size
    batches = [upload[i : i + n] for i in range(0, len(upload), n)]
    for k, batch in enumerate(batches, 1):
        parent = backend.commit(
            {p: out / p for p in batch}, [], f"soundakira: audio {k}/{len(batches)}", parent
        )
        report.commits += 1
    parent = backend.commit(
        final,
        delete,
        message
        or f"soundakira: +{len(local_rows)} utterances from {len(local_sources)} "
        f"source(s), {len(merged)} total",
        parent,
    )
    report.commits += 1
    return report
