"""Constant-memory processing of arbitrarily long audio.

Enhancement models (source separation, denoising) can't hold a 3-hour movie
in memory. We stream the file in chunks, give each chunk `overlap` seconds of
context on both sides, and linearly crossfade neighbouring chunks across each
boundary, so there are no clicks or level jumps at chunk edges.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from soundakira.audio.io import fit_length, to_mono

ChunkFn = Callable[[np.ndarray, int], np.ndarray]


def chunk_boundaries(total: int, chunk: int, overlap: int) -> list[int]:
    """Core chunk edges [0, b1, ..., total]; every core is > 2*overlap samples
    long so that crossfade zones never collide."""
    if total <= 0:
        return [0, 0]
    bounds = list(range(0, total, chunk)) + [total]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] <= 2 * overlap:
        del bounds[-2]
    return bounds


def process_file_in_chunks(
    src: Path,
    dst: Path,
    fn: ChunkFn,
    chunk_seconds: float,
    overlap_seconds: float,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Apply `fn((C, T) audio, sr) -> (C', T) audio` over `src`, writing mono to `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    with sf.SoundFile(str(src)) as fin:
        sr, total = fin.samplerate, fin.frames
        chunk = max(1, int(chunk_seconds * sr))
        ov = max(0, int(overlap_seconds * sr))
        if chunk <= 2 * ov:
            raise ValueError("chunk_seconds must be more than twice overlap_seconds")
        bounds = chunk_boundaries(total, chunk, ov)
        fade_in = np.linspace(0.0, 1.0, 2 * ov, dtype=np.float32)
        with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=1, subtype="PCM_16") as fout:
            prev_tail: np.ndarray | None = None
            n_chunks = len(bounds) - 1
            for k in range(n_chunks):
                core_start, core_end = bounds[k], bounds[k + 1]
                read_start = max(0, core_start - ov)
                read_end = min(total, core_end + ov)
                fin.seek(read_start)
                x = fin.read(read_end - read_start, dtype="float32", always_2d=True).T
                y = to_mono(np.asarray(fn(x, sr), dtype=np.float32))
                y = fit_length(y, read_end - read_start)

                pos = 0  # position in y
                if k > 0 and prev_tail is not None:
                    head = y[: len(prev_tail)]
                    ramp = fade_in[: len(prev_tail)] if len(prev_tail) == 2 * ov else (
                        np.linspace(0.0, 1.0, len(prev_tail), dtype=np.float32)
                    )
                    fout.write(np.clip(prev_tail * (1 - ramp) + head * ramp, -1, 1))
                    pos = len(prev_tail)
                if k < n_chunks - 1:
                    tail_start = (core_end - ov) - read_start
                    fout.write(np.clip(y[pos:tail_start], -1, 1))
                    prev_tail = y[tail_start:]
                else:
                    fout.write(np.clip(y[pos:], -1, 1))
                if progress:
                    progress(k + 1, n_chunks)
    os.replace(tmp, dst)
