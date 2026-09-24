"""Interval arithmetic over time spans."""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence

from soundakira.types import Span, Turn


def merge_spans(spans: Iterable[Span], max_gap: float = 0.0) -> list[Span]:
    """Union of spans; spans separated by at most `max_gap` are joined."""
    ordered = sorted((s for s in spans if s.end > s.start), key=lambda s: s.start)
    merged: list[Span] = []
    for s in ordered:
        if merged and s.start - merged[-1].end <= max_gap:
            last = merged[-1]
            merged[-1] = Span(last.start, max(last.end, s.end))
        else:
            merged.append(s)
    return merged


class SpanIndex:
    """Fast coverage queries against a set of spans (merged on construction)."""

    def __init__(self, spans: Iterable[Span]):
        self.spans = merge_spans(spans)
        self._starts = [s.start for s in self.spans]

    def coverage(self, start: float, end: float) -> float:
        """Seconds of [start, end] covered by the indexed spans."""
        if end <= start or not self.spans:
            return 0.0
        i = max(0, bisect.bisect_right(self._starts, start) - 1)
        total = 0.0
        while i < len(self.spans) and self.spans[i].start < end:
            s = self.spans[i]
            total += max(0.0, min(end, s.end) - max(start, s.start))
            i += 1
        return total

    def ratio(self, start: float, end: float) -> float:
        if end <= start:
            return 0.0
        return self.coverage(start, end) / (end - start)

    def intersecting(self, start: float, end: float) -> list[Span]:
        if end <= start or not self.spans:
            return []
        i = max(0, bisect.bisect_right(self._starts, start) - 1)
        out = []
        while i < len(self.spans) and self.spans[i].start < end:
            if self.spans[i].end > start:
                out.append(self.spans[i])
            i += 1
        return out


def overlap_regions(turns: Sequence[Turn]) -> list[Span]:
    """Regions where two or more *different* speakers are active."""
    events: list[tuple[float, int, str]] = []
    for t in turns:
        if t.end > t.start:
            events.append((t.start, 1, t.speaker))
            events.append((t.end, -1, t.speaker))
    # Process ends before starts at the same instant so abutting turns don't overlap.
    events.sort(key=lambda e: (e[0], e[1]))
    active: dict[str, int] = {}
    regions: list[Span] = []
    open_at: float | None = None
    for time, delta, spk in events:
        active[spk] = active.get(spk, 0) + delta
        if active[spk] == 0:
            del active[spk]
        n = len(active)
        if n >= 2 and open_at is None:
            open_at = time
        elif n < 2 and open_at is not None:
            if time > open_at:
                regions.append(Span(open_at, time))
            open_at = None
    return merge_spans(regions)


def group_spans(spans: Sequence[Span], max_length: float, max_gap: float) -> list[Span]:
    """Pack consecutive speech spans into chunks of at most `max_length` seconds,
    breaking at gaps longer than `max_gap`. Spans longer than `max_length` are
    cut into equal pieces. Used to feed chunk-based ASR backends."""
    chunks: list[Span] = []
    cur: Span | None = None
    for s in merge_spans(spans):
        pieces = [s]
        if s.duration > max_length:
            n = int(s.duration // max_length) + 1
            step = s.duration / n
            pieces = [Span(s.start + i * step, s.start + (i + 1) * step) for i in range(n)]
        for p in pieces:
            if cur is not None and p.start - cur.end <= max_gap and p.end - cur.start <= max_length:
                cur = Span(cur.start, p.end)
            else:
                if cur is not None:
                    chunks.append(cur)
                cur = p
    if cur is not None:
        chunks.append(cur)
    return chunks
