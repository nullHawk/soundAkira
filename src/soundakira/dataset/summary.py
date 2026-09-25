"""Dataset-level summaries (speakers.csv, dataset.json totals), shared by
`build` (local output) and `push` (the merged dataset on the Hub)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from soundakira.dataset.export import write_csv

SPEAKER_COLUMNS = [
    "speaker_id",
    "speaker_name",
    "split",
    "num_utterances",
    "total_duration_s",
    "num_sources",
    "num_references",
    "sources",
]


_FINGERPRINT = re.compile(r"-[0-9a-f]{10}$")
_EPISODE = re.compile(r"[-_ ](s\d+[-_ ]?e\d+|ep?\d+|episode[-_ ]?\d+|\d{1,4})$", re.IGNORECASE)


def series_of(source_id: str) -> str:
    """Group sources into series for reporting: 'naruto-ep01-3fa2c1d9e0' -> 'naruto',
    'smoking-s01e07-…' -> 'smoking'. YouTube/URL sources group as 'youtube'/'url'."""
    if source_id.startswith("yt-"):
        return "youtube"
    if source_id.startswith("url-"):
        return "url"
    base = _FINGERPRINT.sub("", source_id)
    for _ in range(2):  # e.g. 'show-s01-e02'
        base = _EPISODE.sub("", base)
    return base or source_id


def speaker_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    per: dict[int, dict[str, Any]] = {}
    for row in rows:
        sid = int(row["speaker_id"])
        agg = per.setdefault(
            sid,
            {
                "speaker_id": sid,
                "speaker_name": row.get("speaker_name"),
                "split": row["split"],
                "num_utterances": 0,
                "total_duration_s": 0.0,
                "sources": set(),
                "references": set(),
            },
        )
        agg["num_utterances"] += 1
        agg["total_duration_s"] += float(row["duration"])
        agg["sources"].add(row["source_id"])
        if row.get("ref_id"):
            agg["references"].add(row["ref_id"])
    return [
        {
            **agg,
            "total_duration_s": round(agg["total_duration_s"], 2),
            "num_sources": len(agg["sources"]),
            "num_references": len(agg["references"]),
            "sources": ";".join(sorted(agg["sources"])),
        }
        for agg in sorted(per.values(), key=lambda a: a["speaker_id"])
    ]


def write_speakers_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, SPEAKER_COLUMNS)


def totals(rows: list[dict[str, Any]], speakers: list[dict[str, Any]]) -> dict[str, Any]:
    hours = sum(float(r["duration"]) for r in rows) / 3600
    by_split: dict[str, dict[str, float]] = defaultdict(
        lambda: {"utterances": 0, "hours": 0.0, "speakers": 0}
    )
    for r in rows:
        by_split[r["split"]]["utterances"] += 1
        by_split[r["split"]]["hours"] += float(r["duration"]) / 3600
    for s in speakers:
        by_split[s["split"]]["speakers"] += 1
    by_series: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"hours": 0.0, "utterances": 0, "sources": set(), "speakers": set()}
    )
    for r in rows:
        g = by_series[series_of(r["source_id"])]
        g["hours"] += float(r["duration"]) / 3600
        g["utterances"] += 1
        g["sources"].add(r["source_id"])
        g["speakers"].add(int(r["speaker_id"]))
    return {
        "total_speakers": len(speakers),
        "total_utterances": len(rows),
        "total_hours": round(hours, 3),
        "series": {
            name: {
                "hours": round(g["hours"], 3),
                "utterances": g["utterances"],
                "sources": len(g["sources"]),
                "speakers": len(g["speakers"]),
            }
            for name, g in sorted(by_series.items(), key=lambda kv: -kv[1]["hours"])
        },
        "num_sources": len({r["source_id"] for r in rows}),
        "splits": {k: {**v, "hours": round(v["hours"], 3)} for k, v in by_split.items()},
        "languages": dict(Counter(r.get("language") for r in rows)),
    }
