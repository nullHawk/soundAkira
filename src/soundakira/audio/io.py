"""Audio file I/O and signal helpers. Arrays are float32; mono is shape (T,),
multi-channel is (C, T)."""

from __future__ import annotations

import os
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def audio_info(path: Path) -> tuple[int, int, int]:
    """(sample_rate, frames, channels)"""
    info = sf.info(str(path))
    return info.samplerate, info.frames, info.channels


def audio_duration(path: Path) -> float:
    sr, frames, _ = audio_info(path)
    return frames / sr


def read_audio(
    path: Path,
    start: float | None = None,
    end: float | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Read [start, end) seconds without loading the whole file."""
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        first = 0 if start is None else max(0, int(round(start * sr)))
        last = f.frames if end is None else min(f.frames, int(round(end * sr)))
        f.seek(min(first, f.frames))
        data = f.read(max(0, last - first), dtype="float32", always_2d=True).T
    if mono:
        data = data.mean(axis=0)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def write_audio(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Atomic write; format inferred from the extension (.wav / .flac)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(audio, -1.0, 1.0)
    if data.ndim == 2:
        data = data.T
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    sf.write(str(tmp), data, sr, subtype=subtype)
    os.replace(tmp, path)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else audio.mean(axis=0)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or audio.shape[-1] == 0:
        return audio.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    out = resample_poly(audio, sr_out // g, sr_in // g, axis=-1)
    return out.astype(np.float32, copy=False)


def fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad the last axis to exactly `length` samples."""
    n = audio.shape[-1]
    if n == length:
        return audio
    if n > length:
        return audio[..., :length]
    pad = [(0, 0)] * (audio.ndim - 1) + [(0, length - n)]
    return np.pad(audio, pad)


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    n = min(int(sr * fade_ms / 1000), audio.shape[-1] // 2)
    if n <= 0:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[..., :n] *= ramp
    out[..., -n:] *= ramp[::-1]
    return out


def normalize(audio: np.ndarray, mode: str, target_db: float) -> np.ndarray:
    """'peak' scales the peak to target dBFS, 'rms' scales RMS (peak-limited to -1 dBFS)."""
    if mode == "none" or audio.size == 0:
        return audio
    target = 10 ** (target_db / 20)
    if mode == "peak":
        ref = float(np.max(np.abs(audio)))
    elif mode == "rms":
        ref = float(np.sqrt(np.mean(audio**2)))
    else:
        raise ValueError(f"unknown normalize mode {mode!r}")
    if ref < 1e-6:
        return audio
    out = audio * (target / ref)
    peak = float(np.max(np.abs(out)))
    ceiling = 10 ** (-1 / 20)
    if peak > ceiling:
        out *= ceiling / peak
    return out.astype(np.float32, copy=False)
