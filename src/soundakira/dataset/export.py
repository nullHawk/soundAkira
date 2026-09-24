"""Clip cutting and metadata writing."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from soundakira.audio.io import apply_fade, normalize, read_audio, resample, write_audio


@dataclass(frozen=True)
class ClipJob:
    src: str
    start: float
    end: float
    dst: str
    sample_rate: int
    fade_ms: float
    normalize_mode: str
    normalize_db: float


def export_clip(job: ClipJob) -> str:
    dst = Path(job.dst)
    if dst.exists():
        return "skipped"
    audio, sr = read_audio(Path(job.src), job.start, job.end)
    audio = resample(audio, sr, job.sample_rate)
    audio = normalize(
        apply_fade(audio, job.sample_rate, job.fade_ms), job.normalize_mode, job.normalize_db
    )
    write_audio(dst, audio, job.sample_rate)
    return "written"


def run_clip_jobs(jobs: Sequence[ClipJob], workers: int) -> None:
    if workers <= 1:
        for job in tqdm(jobs, desc="export", unit="clip", leave=False):
            export_clip(job)
        return
    with ProcessPoolExecutor(workers) as pool:
        for _ in tqdm(
            pool.map(export_clip, jobs, chunksize=16),
            total=len(jobs),
            desc="export",
            unit="clip",
            leave=False,
        ):
            pass


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    """Fully quoted where needed: transcripts contain commas, quotes and newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    tmp.replace(path)
