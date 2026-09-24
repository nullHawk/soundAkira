"""Turn user inputs into a de-duplicated list of `Source`s.

Accepted inputs (any mix):
- a media file (video or audio; anything ffmpeg can decode)
- a directory (scanned recursively for media)
- a URL: YouTube videos and playlists/channels, or any site yt-dlp supports,
  including direct links to media files
- a manifest (.txt / .csv / .tsv / .jsonl) listing any of the above, with an
  optional language hint per line::

      https://youtu.be/abc123   en
      /data/movies/film.mkv     hi

Source IDs are deterministic, so re-running on the same inputs resumes work
instead of redoing it.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from soundakira.utils.hashing import file_fingerprint, slugify, stable_hash
from soundakira.utils.text import normalize_language

log = logging.getLogger(__name__)

MEDIA_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
        ".ts",
        ".flv",
        ".wmv",
        ".mpg",
        ".mpeg",
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
    }
)
MANIFEST_EXTENSIONS = frozenset({".txt", ".csv", ".tsv", ".jsonl", ".list"})

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    uri: str
    kind: Literal["local", "url"]
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Source:
        return cls(**d)


def is_url(text: str) -> bool:
    return urlparse(text).scheme in ("http", "https")


def youtube_video_id(url: str) -> str | None:
    u = urlparse(url)
    host = (u.hostname or "").removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        vid = u.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if u.path == "/watch":
            vid = parse_qs(u.query).get("v", [""])[0]
        elif u.path.startswith(("/shorts/", "/live/", "/embed/")):
            vid = u.path.split("/")[2]
        else:
            return None
    else:
        return None
    return vid if _YT_ID.match(vid) else None


def looks_like_playlist(url: str) -> bool:
    u = urlparse(url)
    host = (u.hostname or "").lower()
    if "youtube.com" not in host:
        return False
    if u.path == "/playlist":
        return True
    return u.path.startswith(("/@", "/channel/", "/c/", "/user/")) or (
        "list" in parse_qs(u.query) and youtube_video_id(url) is None
    )


def url_source(url: str, language: str | None) -> Source:
    vid = youtube_video_id(url)
    source_id = f"yt-{vid}" if vid else f"url-{stable_hash(url, 12)}"
    canonical = f"https://www.youtube.com/watch?v={vid}" if vid else url
    return Source(source_id, canonical, "url", language)


def local_source(path: Path, language: str | None) -> Source:
    path = path.resolve()
    slug = slugify(path.stem)
    fp = file_fingerprint(path)[:10]
    return Source(f"{slug}-{fp}" if slug else f"file-{fp}", str(path), "local", language)


def expand_playlist(url: str, options: dict[str, Any] | None = None) -> list[str]:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", **(options or {})}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    urls: list[str] = []

    def walk(entry: dict[str, Any]) -> None:
        if entry.get("_type") in ("playlist", "multi_video"):
            for e in entry.get("entries") or []:
                if e:
                    walk(e)
            return
        u = entry.get("webpage_url") or entry.get("url")
        if u and not is_url(u) and entry.get("ie_key", "").lower().startswith("youtube"):
            u = f"https://www.youtube.com/watch?v={u}"
        if u:
            urls.append(u)

    walk(info or {})
    return urls


def _read_manifest(path: Path) -> Iterable[tuple[str, str | None]]:
    suffix = path.suffix.lower()
    base = path.parent
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                yield _manifest_uri(row["uri"], base), row.get("language")
    elif suffix in (".csv", ".tsv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ",")
            for row in reader:
                yield _manifest_uri(row["uri"], base), row.get("language") or None
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2 and not is_url(parts[1]) and len(parts[1]) <= 8:
                yield _manifest_uri(parts[0], base), parts[1]
            else:
                yield _manifest_uri(line, base), None


def _manifest_uri(uri: str, base: Path) -> str:
    uri = uri.strip()
    if is_url(uri) or Path(uri).is_absolute():
        return uri
    return str((base / uri).resolve())


def resolve_inputs(
    inputs: Iterable[str],
    default_language: str | None = None,
    expand_playlists: bool = True,
    yt_dlp_options: dict[str, Any] | None = None,
) -> list[Source]:
    sources: dict[str, Source] = {}

    def add(src: Source) -> None:
        sources.setdefault(src.source_id, src)

    def handle(item: str, language: str | None) -> None:
        language = normalize_language(language) or normalize_language(default_language)
        if is_url(item):
            if expand_playlists and looks_like_playlist(item):
                entries = expand_playlist(item, yt_dlp_options)
                log.info("expanded playlist %s -> %d videos", item, len(entries))
                for u in entries:
                    add(url_source(u, language))
            else:
                add(url_source(item, language))
            return
        path = Path(item).expanduser()
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                if (
                    p.is_file()
                    and p.suffix.lower() in MEDIA_EXTENSIONS
                    and not p.name.startswith(".")
                ):
                    add(local_source(p, language))
        elif path.is_file() and path.suffix.lower() in MANIFEST_EXTENSIONS:
            for uri, lang in _read_manifest(path):
                handle(uri, lang or language)
        elif path.is_file():
            add(local_source(path, language))
        else:
            raise FileNotFoundError(f"input not found: {item}")

    for item in inputs:
        handle(item, None)
    return sorted(sources.values(), key=lambda s: s.source_id)
