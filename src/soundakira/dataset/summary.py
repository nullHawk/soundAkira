"""Dataset-level summaries (speakers.csv, dataset.json totals), shared by
`build` (local output) and `push` (the merged dataset on the Hub)."""

from __future__ import annotations

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
    return {
        "total_speakers": len(speakers),
        "total_utterances": len(rows),
        "total_hours": round(hours, 3),
        "num_sources": len({r["source_id"] for r in rows}),
        "splits": {k: {**v, "hours": round(v["hours"], 3)} for k, v in by_split.items()},
        "languages": dict(Counter(r.get("language") for r in rows)),
    }
