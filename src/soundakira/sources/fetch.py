"""Download remote sources with yt-dlp (audio only: the video track is never needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soundakira.config import FetchConfig


def download(url: str, out_dir: Path, cfg: FetchConfig) -> tuple[Path, dict[str, Any]]:
    """Returns (media path, trimmed info dict). Retries and resumes via yt-dlp."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    opts: dict[str, Any] = {
        "format": cfg.format,
        "outtmpl": str(out_dir / "media.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "overwrites": False,
    }
    if cfg.cookies_file:
        opts["cookiefile"] = str(cfg.cookies_file)
    opts.update(cfg.yt_dlp_options)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloads = info.get("requested_downloads") or []
        path = Path(downloads[0]["filepath"]) if downloads else Path(ydl.prepare_filename(info))
    if not path.exists():
        raise FileNotFoundError(f"yt-dlp reported {path} but it does not exist")
    keep = ("id", "title", "uploader", "channel", "duration", "webpage_url", "extractor_key",
            "upload_date", "language", "license")
    return path, {k: info.get(k) for k in keep if info.get(k) is not None}
