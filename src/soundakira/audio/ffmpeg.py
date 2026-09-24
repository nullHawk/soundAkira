"""ffprobe/ffmpeg wrappers: audio-stream selection and decoding."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ChannelMode = Literal["stereo", "mono", "center"]


class FFmpegError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(f"`{name}` not found on PATH. Install ffmpeg (https://ffmpeg.org).")
    return path


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")
    return proc.stdout


def probe(path: Path) -> dict[str, Any]:
    out = _run([
        require_binary("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ])
    return json.loads(out)


@dataclass(frozen=True, slots=True)
class AudioStream:
    index: int  # position among the file's audio streams (ffmpeg `0:a:<index>`)
    codec: str | None
    channels: int
    channel_layout: str | None
    sample_rate: int | None
    language: str | None
    title: str | None
    is_default: bool
    is_commentary: bool
    is_audio_description: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_audio_streams(probe_data: dict[str, Any]) -> list[AudioStream]:
    streams = []
    audio = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"]
    for i, s in enumerate(audio):
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        disp = s.get("disposition") or {}
        sr = s.get("sample_rate")
        streams.append(AudioStream(
            index=i,
            codec=s.get("codec_name"),
            channels=int(s.get("channels") or 0),
            channel_layout=s.get("channel_layout"),
            sample_rate=int(sr) if sr else None,
            language=(tags.get("language") or "").lower() or None,
            title=tags.get("title") or tags.get("handler_name"),
            is_default=bool(disp.get("default")),
            is_commentary=bool(disp.get("comment")),
            is_audio_description=bool(disp.get("visual_impaired")),
        ))
    return streams


def select_audio_stream(
    streams: list[AudioStream],
    prefer_languages: list[str],
    skip_title_keywords: list[str],
    explicit_index: int | None = None,
) -> AudioStream:
    """Pick the main dialogue track: skip commentary / audio-description, then
    prefer a requested language, then the default track, then more channels."""
    if not streams:
        raise FFmpegError("source has no audio stream")
    if explicit_index is not None:
        for s in streams:
            if s.index == explicit_index:
                return s
        raise FFmpegError(f"audio stream {explicit_index} not found ({len(streams)} streams)")

    keywords = [k.lower() for k in skip_title_keywords]
    langs = [lang.lower() for lang in prefer_languages]

    def is_excluded(s: AudioStream) -> bool:
        title = (s.title or "").lower()
        return s.is_commentary or s.is_audio_description or any(k in title for k in keywords)

    candidates = [s for s in streams if not is_excluded(s)] or streams

    def rank(s: AudioStream) -> tuple:
        lang_rank = langs.index(s.language) if s.language in langs else len(langs)
        return (lang_rank, not s.is_default, -s.channels, s.index)

    return min(candidates, key=rank)


def _has_center(stream: AudioStream) -> bool:
    layout = (stream.channel_layout or "").lower()
    return stream.channels >= 3 and (layout.startswith(("5.1", "6.1", "7.1", "3.")) or not layout)


def build_extract_command(
    src: Path,
    dst: Path,
    stream: AudioStream,
    sample_rate: int,
    channels: ChannelMode,
) -> list[str]:
    cmd = [
        require_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-map", f"0:a:{stream.index}", "-vn", "-sn", "-dn",
    ]
    if channels == "center" and _has_center(stream):
        # Dialogue is mixed to the front-centre channel in surround mixes;
        # isolating it drops most music and effects before any model runs.
        cmd += ["-af", "pan=mono|c0=c2", "-ac", "1"]
    elif channels == "stereo":
        cmd += ["-ac", "2"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sample_rate), "-c:a", "flac", "-sample_fmt", "s16", str(dst)]
    return cmd


def extract_audio(
    src: Path, dst: Path, stream: AudioStream, sample_rate: int, channels: ChannelMode
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    _run(build_extract_command(src, tmp, stream, sample_rate, channels))
    os.replace(tmp, dst)


def convert_audio(src: Path, dst: Path, sample_rate: int, channels: int = 1) -> None:
    """Streamed resample/downmix of an audio file (constant memory)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    _run([
        require_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-ac", str(channels), "-ar", str(sample_rate),
        "-c:a", "flac", "-sample_fmt", "s16", str(tmp),
    ])
    os.replace(tmp, dst)
