"""Reference (voice prompt) selection for zero-shot TTS.

Each target utterance gets a reference clip of the same speaker that:
- is never the target itself and never overlaps it in time (no leakage), and
- preferably comes from another source, so the model learns the voice and
  not the room, microphone or episode.

Short segments are reference candidates. So are word-aligned excerpts of
long segments, because a speaker who only has long turns still needs 4-12 s
prompts. Every prompt starts at a sentence start.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from soundakira.config import ReferenceConfig
from soundakira.dataset.filters import first_failure
from soundakira.segmentation.builder import is_cased, starts_lowercase
from soundakira.types import Segment, Word
from soundakira.utils.text import ends_sentence, join_words, spoken_char_count


@dataclass
class Reference:
    ref_id: str
    parent_id: str
    source_id: str
    speaker_id: int
    start: float
    end: float
    text: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def derive_excerpt(
    seg: Segment, min_d: float, max_d: float
) -> tuple[float, float, list[Word]] | None:
    """Word-aligned excerpt of `seg` lasting [min_d, max_d] seconds, ending at a
    sentence end where possible. If the segment starts mid-sentence
    (`sentence_start == 0`), the excerpt starts at its first sentence start
    instead, because a prompt that begins mid-phrase teaches the model odd
    onsets. Returns (start, end, words), or None if no excerpt fits."""
    words = seg.words
    cased = is_cased(words)
    first, start = 0, seg.start
    if seg.metrics.get("sentence_start", 1.0) < 1.0:
        for k in range(1, len(words)):
            if (
                words[k - 1].kind == "word"
                and ends_sentence(words[k - 1].text)
                and not (cased and starts_lowercase(words[k].text))
            ):
                first, start = k, max(words[k - 1].end + 0.03, words[k].start - 0.12)
                break
        else:
            return None
    best: tuple[int, float, bool] | None = None  # (index, dur, sentence_end)
    for k in range(first, len(words)):
        w = words[k]
        dur = w.end - start
        if dur > max_d:
            break
        if dur >= min_d and w.kind == "word":
            cand = (k, dur, ends_sentence(w.text))
            if best is None or (cand[2], cand[1]) >= (best[2], best[1]):
                best = cand
    if best is None:
        return None
    k = best[0]
    nxt = words[k + 1].start - 0.03 if k + 1 < len(words) else seg.end
    end = min(words[k].end + 0.12, max(words[k].end, nxt))
    return start, end, words[first : k + 1]


def build_reference_pool(
    segments: list[Segment], speaker_of: dict[str, int], cfg: ReferenceConfig
) -> tuple[dict[int, list[Reference]], dict[str, int]]:
    pool: dict[int, list[Reference]] = defaultdict(list)
    rejected: dict[str, int] = defaultdict(int)
    for seg in segments:
        spk = speaker_of.get(seg.segment_id)
        if spk is None:
            continue
        whole_ok = seg.metrics.get("sentence_start", 1.0) >= 1.0
        if cfg.min_duration <= seg.duration <= cfg.max_duration and whole_ok:
            ref = Reference(
                seg.segment_id,
                seg.segment_id,
                seg.source_id,
                spk,
                seg.start,
                seg.end,
                seg.text,
                dict(seg.metrics),
            )
        elif seg.duration >= cfg.min_duration and (
            cfg.derive_from_long_segments or seg.duration <= cfg.max_duration
        ):
            excerpt = derive_excerpt(seg, cfg.min_duration, cfg.max_duration)
            if excerpt is None:
                continue
            start, end, words = excerpt
            text = join_words(words)
            metrics = dict(seg.metrics)
            metrics["duration"] = end - start
            metrics["chars_per_sec"] = spoken_char_count(text) / (end - start)
            metrics["sentence_start"] = 1.0
            probs = [w.prob for w in words if w.prob is not None and w.kind == "word"]
            if probs:
                metrics["asr_confidence"] = float(np.mean(probs))
            ref = Reference(
                f"{seg.segment_id}-ref",
                seg.segment_id,
                seg.source_id,
                spk,
                round(start, 3),
                round(end, 3),
                text,
                metrics,
            )
        else:
            continue
        reason = first_failure(ref.metrics, cfg.filters)
        if reason:
            rejected[reason] += 1
            continue
        pool[spk].append(ref)

    def rank(r: Reference) -> tuple:
        return tuple(-r.metrics.get(f, float("-inf")) for f in cfg.rank_by) + (r.ref_id,)

    for refs in pool.values():
        refs.sort(key=rank)
    return pool, dict(rejected)


def _stable_index(key: str, n: int) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % n


def assign_reference(
    target: Segment, speaker_id: int, pool: dict[int, list[Reference]], cfg: ReferenceConfig
) -> Reference | None:
    """Best-ranked valid reference; ties among the top-k are spread
    deterministically so one clip doesn't prompt every utterance."""
    valid = [
        r
        for r in pool.get(speaker_id, [])
        if r.parent_id != target.segment_id
        and not (r.source_id == target.source_id and r.start < target.end and target.start < r.end)
    ]
    if not valid:
        return None
    if cfg.prefer_other_source:
        other = [r for r in valid if r.source_id != target.source_id]
        valid = other or valid
    top = valid[: max(1, cfg.top_k)]
    return top[_stable_index(target.segment_id, len(top))]
