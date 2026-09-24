"""Words + diarization + VAD  ->  single-speaker segments.

Pure functions: no models or I/O, so the logic that decides where every clip
starts and ends is fully unit-testable.

1. Sanitise ASR words: sort them, repair inverted times, and optionally
   shrink each word to the VAD speech inside it. Whisper often stretches the
   first word after a pause back into the silence.
2. Give each word the speaker whose turns overlap it most. Words in
   overlapping speech, or with no speaker nearby, become hard breaks, so no
   clip ever contains two voices.
3. Group consecutive words of the same speaker into runs, breaking at pauses
   longer than `max_pause`.
4. Split runs longer than `max_duration` at the best boundary: sentence end,
   then clause end, then long pause. Prefer pieces that reach the preferred
   (training-target) length.
5. Pad each piece into the surrounding silence without crossing into
   neighbouring words or other speakers' turns, and compute text and alignment
   metrics used later for filtering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from soundakira.types import Segment, Span, Transcript, Turn, Word
from soundakira.utils.spans import SpanIndex, overlap_regions
from soundakira.utils.text import (
    ends_clause,
    ends_sentence,
    join_words,
    repetition_ratio,
    spoken_char_count,
)

_EPS = 0.01


@dataclass(frozen=True)
class SegmentationParams:
    min_duration: float = 3.0
    max_duration: float = 30.0
    preferred_min_duration: float = 15.0
    max_pause: float = 1.5
    pad: float = 0.15
    boundary_margin: float = 0.05
    speaker_max_distance: float = 0.3
    overlap_word_fraction: float = 0.3
    refine_with_vad: bool = True


def sanitize_words(words: list[Word]) -> list[Word]:
    out = []
    for w in words:
        if not w.text.strip():
            continue
        start, end = float(w.start), float(w.end)
        if end < start:
            start, end = end, start
        out.append(replace(w, start=max(0.0, start), end=max(0.0, end)))
    out.sort(key=lambda w: (w.start, w.end))
    return out


def refine_words_with_vad(words: list[Word], speech: SpanIndex, min_duration: float = 0.02) -> list[Word]:
    """Shrink each word to the VAD speech it actually overlaps."""
    out = []
    for w in words:
        hits = speech.intersecting(w.start, w.end)
        if hits:
            start, end = max(w.start, hits[0].start), min(w.end, hits[-1].end)
            if end - start >= min_duration:
                w = replace(w, start=start, end=end)
        out.append(w)
    return out


def assign_speakers(words: list[Word], turns: list[Turn], max_distance: float) -> list[str | None]:
    """Speaker with the most overlap for each word; else the nearest turn
    within `max_distance`; else None."""
    if not turns:
        return [None] * len(words)
    ts = np.array([t.start for t in turns])
    te = np.array([t.end for t in turns])
    labels = [t.speaker for t in turns]
    result: list[str | None] = []
    for w in words:
        ws, we = w.start, max(w.end, w.start + _EPS)
        ov = np.minimum(te, we) - np.maximum(ts, ws)
        hit = np.nonzero(ov > 0)[0]
        if hit.size:
            per_speaker: dict[str, float] = defaultdict(float)
            for i in hit:
                per_speaker[labels[i]] += float(ov[i])
            result.append(max(per_speaker.items(), key=lambda kv: (kv[1], kv[0]))[0])
            continue
        dist = np.maximum(ts - we, ws - te)
        j = int(np.argmin(dist))
        result.append(labels[j] if dist[j] <= max_distance else None)
    return result


def overlapped_words(words: list[Word], overlaps: SpanIndex, fraction: float) -> list[bool]:
    flags = []
    for w in words:
        ws, we = w.start, max(w.end, w.start + _EPS)
        flags.append(overlaps.coverage(ws, we) / (we - ws) >= fraction)
    return flags


def build_runs(
    words: list[Word], speakers: list[str | None], overlapped: list[bool], max_pause: float
) -> list[tuple[str, list[int]]]:
    """Consecutive same-speaker word indices; events attach to the open run."""
    runs: list[tuple[str, list[int]]] = []
    cur: tuple[str, list[int]] | None = None
    for i, w in enumerate(words):
        gap_ok = cur is not None and w.start - words[cur[1][-1]].end <= max_pause
        if w.kind == "event":
            if cur is not None and gap_ok:
                cur[1].append(i)
            continue
        spk = speakers[i]
        if spk is None or overlapped[i]:
            if cur is not None:
                runs.append(cur)
            cur = None
            continue
        if cur is not None and cur[0] == spk and gap_ok:
            cur[1].append(i)
        else:
            if cur is not None:
                runs.append(cur)
            cur = (spk, [i])
    if cur is not None:
        runs.append(cur)
    return runs


def split_run(
    words: list[Word], idx: list[int], max_duration: float, preferred_min: float
) -> list[list[int]]:
    """Split a run into pieces no longer than `max_duration` at natural boundaries."""
    pieces: list[list[int]] = []
    start, n = 0, len(idx)
    while start < n:
        first = words[idx[start]]
        if words[idx[-1]].end - first.start <= max_duration:
            pieces.append(idx[start:])
            break
        best, best_score = start + 1, float("-inf")
        for k in range(start + 1, n):
            prev, nxt = words[idx[k - 1]], words[idx[k]]
            first_dur = prev.end - first.start
            if first_dur > max_duration:
                break
            gap = max(0.0, nxt.start - prev.end)
            score = (
                3.0 * ends_sentence(prev.text)
                + 1.0 * ends_clause(prev.text)
                + 2.0 * min(gap, 1.0)
                + 2.0 * (first_dur >= preferred_min)
                + first_dur / max_duration
            )
            if score > best_score:
                best, best_score = k, score
        pieces.append(idx[start:best])
        start = best
    return pieces


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    total = sum(w for _, w in values)
    return sum(v * w for v, w in values) / total if total > 0 else None


def build_segments(
    source_id: str,
    transcript: Transcript,
    turns: list[Turn],
    speech: list[Span],
    total_duration: float,
    params: SegmentationParams,
) -> list[Segment]:
    speech_index = SpanIndex(speech)
    overlap_index = SpanIndex(overlap_regions(turns))

    words = sanitize_words(transcript.words)
    if params.refine_with_vad and speech:
        words = refine_words_with_vad(words, speech_index)
    speakers = assign_speakers(words, turns, params.speaker_max_distance)
    overlapped = overlapped_words(words, overlap_index, params.overlap_word_fraction)

    turns_by_speaker: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        turns_by_speaker[t.speaker].append(t)

    segments: list[Segment] = []
    for speaker, run in build_runs(words, speakers, overlapped, params.max_pause):
        for piece in split_run(words, run, params.max_duration, params.preferred_min_duration):
            seg = _make_segment(
                source_id, speaker, piece, words, transcript, turns_by_speaker,
                speech_index, overlap_index, total_duration, params,
            )
            if seg is not None:
                segments.append(seg)
    return segments


def _make_segment(
    source_id: str,
    speaker: str,
    piece: list[int],
    words: list[Word],
    transcript: Transcript,
    turns_by_speaker: dict[str, list[Turn]],
    speech_index: SpanIndex,
    overlap_index: SpanIndex,
    total_duration: float,
    p: SegmentationParams,
) -> Segment | None:
    first_i, last_i = piece[0], piece[-1]
    first, last = words[first_i], words[last_i]
    if last.end - first.start < p.min_duration:
        return None

    # Pad into silence, never into a neighbouring word or another speaker's turn.
    lo = words[first_i - 1].end + p.boundary_margin if first_i > 0 else 0.0
    hi = words[last_i + 1].start - p.boundary_margin if last_i + 1 < len(words) else total_duration
    for other, other_turns in turns_by_speaker.items():
        if other == speaker:
            continue
        for t in other_turns:
            if t.end <= first.start and t.end > lo:
                lo = t.end + p.boundary_margin
            if t.start >= last.end and t.start < hi:
                hi = t.start - p.boundary_margin
    start = min(first.start, max(first.start - p.pad, lo, 0.0))
    end = max(last.end, min(last.end + p.pad, hi, total_duration))

    seg_words = [words[i] for i in piece]
    text = join_words(seg_words)
    if not text:
        return None
    has_events = any(w.kind == "event" for w in seg_words)
    duration = end - start
    lexical = [w for w in seg_words if w.kind == "word"]
    probs = [w.prob for w in lexical if w.prob is not None]
    gaps = [b.start - a.end for a, b in zip(lexical, lexical[1:])]

    asr_stats: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in transcript.segments:
        ov = min(end, s.end) - max(start, s.start)
        if ov <= 0:
            continue
        for key in ("avg_logprob", "no_speech_prob", "compression_ratio"):
            val = getattr(s, key)
            if val is not None:
                asr_stats[key].append((val, ov))

    metrics: dict[str, float] = {
        "duration": duration,
        "num_words": float(len(lexical)),
        "chars_per_sec": spoken_char_count(text) / duration,
        "words_per_sec": len(lexical) / duration,
        "repetition_ratio": repetition_ratio(text),
        "speech_ratio": speech_index.ratio(start, end) if speech_index.spans else 1.0,
        "overlap_ratio": overlap_index.ratio(start, end),
        "max_word_gap": max(gaps, default=0.0),
    }
    if probs:
        metrics["asr_confidence"] = float(np.mean(probs))
        metrics["min_word_prob"] = float(np.min(probs))
    for key, vals in asr_stats.items():
        mean = _weighted_mean(vals)
        if mean is not None:
            metrics[f"asr_{key}"] = mean

    return Segment(
        segment_id=f"{source_id}-{round(start * 1000):09d}",
        source_id=source_id,
        speaker=speaker,
        start=round(start, 3),
        end=round(end, 3),
        text=text,
        text_tagged=join_words(seg_words, include_events=True) if has_events else None,
        language=transcript.language,
        words=seg_words,
        metrics={k: round(float(v), 5) for k, v in metrics.items()},
    )
