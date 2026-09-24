from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_SAMPLE_BYTES = 4 * 1024 * 1024


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: Path) -> str:
    """Cheap content fingerprint for large media: size + head + tail.
    Hashing a whole 4 GB movie would dominate ingest time."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_SAMPLE_BYTES))
        if size > 2 * _SAMPLE_BYTES:
            f.seek(size - _SAMPLE_BYTES)
            h.update(f.read(_SAMPLE_BYTES))
    return h.hexdigest()


def slugify(text: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].strip("-")


def shard_of(key: str, num_shards: int) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % num_shards
