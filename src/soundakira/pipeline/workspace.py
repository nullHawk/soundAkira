"""On-disk layout for per-source intermediate artifacts.

<work_dir>/
  sources/<source_id>/
    manifest.json        stage status, fingerprints, stats, errors
    source.json          the resolved Source
    media.json           fetched media path + metadata
    extract.json         chosen audio stream, duration
    audio/source.flac    decoded dialogue track (44.1 kHz)
    audio/clean.flac     enhanced mono (vocals, denoised)
    audio/analysis.flac  16 kHz mono for VAD / diarization / ASR / embeddings
    vad.json  diarization.json  transcript.json
    segments.jsonl  scores.json  embeddings.npz
  speakers/registry.json  global speaker registry (stable IDs across builds)
  logs/
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soundakira.sources.resolve import Source
from soundakira.utils.io import read_json, write_json

_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SourceWorkspace:
    def __init__(self, work_dir: Path, source: Source):
        self.source = source
        self.dir = work_dir / "sources" / source.source_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.dir / "audio"
        self.download_dir = self.dir / "download"
        self.manifest_path = self.dir / "manifest.json"
        self.source_json = self.dir / "source.json"
        self.media_json = self.dir / "media.json"
        self.extract_json = self.dir / "extract.json"
        self.source_audio = self.audio_dir / "source.flac"
        self.clean_audio = self.audio_dir / "clean.flac"
        self.analysis_audio = self.audio_dir / "analysis.flac"
        self.vad_json = self.dir / "vad.json"
        self.diarization_json = self.dir / "diarization.json"
        self.transcript_json = self.dir / "transcript.json"
        self.segments_jsonl = self.dir / "segments.jsonl"
        self.scores_json = self.dir / "scores.json"
        self.embeddings_npz = self.dir / "embeddings.npz"
        if not self.source_json.exists():
            write_json(self.source_json, source.to_dict())

    @classmethod
    def open(cls, source_dir: Path) -> SourceWorkspace:
        source = Source.from_dict(read_json(source_dir / "source.json"))
        return cls(source_dir.parent.parent, source)

    @property
    def source_id(self) -> str:
        return self.source.source_id

    # -- manifest -------------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return read_json(self.manifest_path)
        return {"source_id": self.source_id, "stages": {}}

    def stage_record(self, stage: str) -> dict[str, Any] | None:
        return self.manifest()["stages"].get(stage)

    def is_done(self, stage: str) -> bool:
        rec = self.stage_record(stage)
        return bool(rec and rec.get("status") == "done")

    def update_stage(self, stage: str, record: dict[str, Any]) -> None:
        with _lock:
            m = self.manifest()
            m["stages"][stage] = record
            write_json(self.manifest_path, m)

    def invalidate(self, stages: list[str]) -> None:
        with _lock:
            m = self.manifest()
            for s in stages:
                m["stages"].pop(s, None)
            write_json(self.manifest_path, m)

    def media_path(self) -> Path:
        return Path(read_json(self.media_json)["path"])


def list_workspaces(work_dir: Path) -> list[SourceWorkspace]:
    root = work_dir / "sources"
    if not root.exists():
        return []
    return [
        SourceWorkspace.open(d)
        for d in sorted(root.iterdir())
        if d.is_dir() and (d / "source.json").exists()
    ]
