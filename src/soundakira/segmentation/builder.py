"""Words + diarization + VAD  ->  single-speaker segments.

Pure functions: no models or I/O, so the logic that decides where every clip
starts and ends is fully unit-testable.

1. Sanitise ASR words: sort them, repair inverted times, and optionally
   shrink each word to the VAD speech inside it. Whisper often stretches the
   first word after a pause back into the silence.
2. Give each word the speaker whose turns overlap it most. Words in
   overlapping speech, or with no speaker nearby, become hard breaks, so no
   clip ever contains two voices. Each speaker change is then snapped to the
   most natural word boundary (a pause or sentence end) inside the
   diarization's transition zone, because its boundaries are often a few
   hundred ms off.
3. Group consecutive words of the same speaker into runs, breaking at pauses
   longer than `max_pause`.
4. Split runs longer than `max_duration` at the best boundary: sentence end,
   then clause end, then long pause. Prefer pieces that reach the preferred
   (training-target) length.
5. Trim leading/trailing sentence fragments (e.g. a clip starting at
   "...think the most important decision") back to sentence boundaries when
   one is close, so clips start and end like complete utterances.
6. Pad each piece into the surrounding silence without crossing into
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
_SNAP_MARGIN = 0.1


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
    snap_speaker_changes: float = 0.5
    trim_to_sentence: float = 4.0


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


def refine_words_with_vad(
    words: list[Word], speech: SpanIndex, min_duration: float = 0.02
) -> list[Word]:
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


def _boundary_score(words: list[Word], c: int) -> float:
    """How natural a cut between words[c-1] and words[c] is."""
    gap = max(0.0, words[c].start - words[c - 1].end)
    prev = words[c - 1].text
    return min(gap, 1.0) + 0.3 * ends_sentence(prev) + 0.1 * ends_clause(prev)


def _transition_zone(
    turns_by_speaker: dict[str, list[Turn]], a: str, b: str, t: float, tolerance: float
) -> tuple[float, float]:
    """Where diarization says speaker `a` hands over to `b` near time t: from
    the end of a's nearest turn to the start of b's, widened by `tolerance`."""
    a_end = min((tr.end for tr in turns_by_speaker.get(a, [])), key=lambda e: abs(e - t), default=t)
    b_start = min(
        (tr.start for tr in turns_by_speaker.get(b, [])), key=lambda s: abs(s - t), default=t
    )
    return min(a_end, b_start) - tolerance, max(a_end, b_start) + tolerance


def snap_speaker_changes(
    words: list[Word],
    speakers: list[str | None],
    turns: list[Turn],
    tolerance: float,
) -> list[str | None]:
    """Move each A->B speaker change to the most natural word boundary
    (largest pause, sentence end) inside the diarization's transition zone.

    Diarization gives the rough location of a speaker change; its exact
    boundary is often a few hundred ms off. Example: "...in these apps? |
    Multiple reasons." where diarization switched after "Multiple". Candidates
    are limited to the zone between A's turn end and B's turn start
    (+/- tolerance), so a second sentence end further away can't pull the
    change the wrong way.
    """
    spk = list(speakers)
    if tolerance <= 0 or not turns:
        return spk
    by_speaker: dict[str, list[Turn]] = defaultdict(list)
    for tr in turns:
        by_speaker[tr.speaker].append(tr)
    n, i = len(words), 1
    while i < n:
        a, b = spk[i - 1], spk[i]
        if a is None or b is None or a == b:
            i += 1
            continue
        z0, z1 = _transition_zone(by_speaker, a, b, words[i].start, tolerance)
        lo = i
        while lo - 1 >= 1 and spk[lo - 1] == a and words[lo - 1].start >= z0:
            lo -= 1
        hi = i
        while hi + 1 < n and spk[hi] == b and words[hi + 1].start <= z1:
            hi += 1
        best = max(range(lo, hi + 1), key=lambda c: (_boundary_score(words, c), -abs(c - i)))
        if _boundary_score(words, best) < _boundary_score(words, i) + _SNAP_MARGIN:
            best = i  # only move for a clearly better boundary
        for k in range(min(best, i), max(best, i)):
            spk[k] = a if best > i else b
        i = max(best, i) + 1
    return spk


def is_cased(words: list[Word]) -> bool:
    """Does this transcript use capitalisation at all? Uncased ASR output
    (all lowercase) and uncased scripts must not be judged by letter case."""
    return any(ch.isupper() for w in words for ch in w.text)


def starts_lowercase(text: str) -> bool:
    """True if the first letter is lowercase: a mid-sentence start in cased
    transcripts (always False for uncased scripts such as Devanagari)."""
    for ch in text:
        if ch.isalpha():
            return ch.islower()
    return False


def trim_to_sentences(
    piece: list[int],
    words: list[Word],
    max_trim: float,
    context_gap: float = 5.0,
    cased: bool = True,
) -> tuple[list[int], bool, bool]:
    """Drop a leading/trailing sentence fragment of at most `max_trim` seconds.

    A piece starts mid-sentence when the previous word (any speaker, within
    `context_gap` s) does not end a sentence. It ends mid-sentence when its last
    word doesn't end one and more speech follows. Returns
    (piece, clean_start, clean_end). Pieces with no nearby sentence boundary
    are kept as-is and flagged instead.
    """

    def neighbour(i: int, step: int) -> int | None:
        j = i + step
        while 0 <= j < len(words) and words[j].kind != "word":
            j += step
        return j if 0 <= j < len(words) else None

    prev = neighbour(piece[0], -1)
    lower = starts_lowercase if cased else (lambda _t: False)
    clean_start = not lower(words[piece[0]].text) and (
        prev is None
        or ends_sentence(words[prev].text)
        or words[piece[0]].start - words[prev].end >= context_gap
    )
    if not clean_start and max_trim > 0:
        t0 = words[piece[0]].start
        for k in range(1, len(piece)):
            if words[piece[k]].start - t0 > max_trim:
                break
            if ends_sentence(words[piece[k - 1]].text) and not lower(words[piece[k]].text):
                piece, clean_start = piece[k:], True
                break

    nxt = neighbour(piece[-1], 1)
    clean_end = (
        nxt is None
        or ends_sentence(words[piece[-1]].text)
        or words[nxt].start - words[piece[-1]].end >= context_gap
    )
    if not clean_end and max_trim > 0:
        t1 = words[piece[-1]].end
        for k in range(len(piece) - 1, 0, -1):
            if t1 - words[piece[k - 1]].end > max_trim:
                break
            if ends_sentence(words[piece[k - 1]].text):
                piece, clean_end = piece[:k], True
                break
    return piece, clean_start, clean_end


def other_speaker_breaks(
    words: list[Word],
    speakers: list[str | None],
    turns_by_speaker: dict[str, list[Turn]],
    min_other: float = 0.3,
) -> set[int]:
    """Word indices i where the pause before words[i] holds at least `min_other`
    seconds of another speaker's diarized speech that ASR didn't transcribe
    (e.g. "Huh?", a laugh). The utterance must break there."""
    breaks: set[int] = set()
    for i in range(1, len(words)):
        spk = speakers[i]
        g0, g1 = words[i - 1].end, words[i].start
        if spk is None or speakers[i - 1] != spk or g1 - g0 < min_other:
            continue
        covered = sum(
            max(0.0, min(g1, t.end) - max(g0, t.start))
            for other, turns in turns_by_speaker.items()
            if other != spk
            for t in turns
        )
        if covered >= min_other:
            breaks.add(i)
    return breaks


def build_runs(
    words: list[Word],
    speakers: list[str | None],
    overlapped: list[bool],
    max_pause: float,
    breaks: set[int] | None = None,
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
        if cur is not None and cur[0] == spk and gap_ok and not (breaks and i in breaks):
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


def _other_speaker_time(
    speaker: str,
    start: float,
    end: float,
    turns_by_speaker: dict[str, list[Turn]],
    p: SegmentationParams,
) -> float:
    """Seconds inside the clip that diarization assigns to other speakers,
    usually an untranscribed interjection ("Huh?") sitting in a pause. The
    edges are excluded because speaker-change snapping may legitimately reach
    into another speaker's (imprecise) turn there."""
    lo, hi = start + p.snap_speaker_changes, end - p.snap_speaker_changes
    if hi <= lo:
        return 0.0
    return sum(
        max(0.0, min(hi, t.end) - max(lo, t.start))
        for other, turns in turns_by_speaker.items()
        if other != speaker
        for t in turns
    )


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
    cased = is_cased(words)
    # Unpunctuated ASR output (common for Hindi fine-tunes): sentence boundaries are
    # unknown, so don't trim or flag clips as mid-sentence.
    punctuated = any(ends_sentence(w.text) for w in words if w.kind == "word")
    speakers = snap_speaker_changes(words, speakers, turns, params.snap_speaker_changes)
    overlapped = overlapped_words(words, overlap_index, params.overlap_word_fraction)

    turns_by_speaker: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        turns_by_speaker[t.speaker].append(t)

    segments: list[Segment] = []
    breaks = other_speaker_breaks(words, speakers, turns_by_speaker)
    for speaker, run in build_runs(words, speakers, overlapped, params.max_pause, breaks):
        for raw_piece in split_run(words, run, params.max_duration, params.preferred_min_duration):
            if punctuated:
                piece, clean_start, clean_end = trim_to_sentences(
                    raw_piece, words, params.trim_to_sentence, cased=cased
                )
            else:
                piece, clean_start, clean_end = raw_piece, True, True
            seg = _make_segment(
                source_id,
                speaker,
                piece,
                words,
                transcript,
                turns_by_speaker,
                speech_index,
                overlap_index,
                total_duration,
                params,
            )
            if seg is not None:
                if punctuated:
                    seg.metrics["sentence_start"] = float(clean_start)
                    seg.metrics["sentence_end"] = float(clean_end)
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
        "other_speaker_s": _other_speaker_time(speaker, start, end, turns_by_speaker, p),
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
