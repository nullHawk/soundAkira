"""Reference (voice prompt) selection for zero-shot TTS.

Each target utterance gets a reference clip of the same speaker that:
- is never the target itself and never overlaps it in time (no leakage), and
- preferably comes from another source, so the model learns the voice and
  not the room, microphone or episode.

Short segments are reference candidates. So are prefixes cut at word
boundaries from long segments, because a speaker who only has long turns
still needs 4-12 s prompts.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from soundakira.config import ReferenceConfig
from soundakira.dataset.filters import first_failure
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


def derive_prefix(seg: Segment, min_d: float, max_d: float) -> tuple[float, list[Word]] | None:
    """Longest word-aligned prefix of `seg` within [min_d, max_d], preferring a
    sentence end. Returns (end_time, words)."""
    words = seg.words
    best: tuple[int, float, bool] | None = None  # (index, dur, sentence_end)
    for k, w in enumerate(words):
        dur = w.end - seg.start
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
    return end, words[: k + 1]


def build_reference_pool(
    segments: list[Segment], speaker_of: dict[str, int], cfg: ReferenceConfig
) -> tuple[dict[int, list[Reference]], dict[str, int]]:
    pool: dict[int, list[Reference]] = defaultdict(list)
    rejected: dict[str, int] = defaultdict(int)
    for seg in segments:
        spk = speaker_of.get(seg.segment_id)
        if spk is None:
            continue
        if cfg.min_duration <= seg.duration <= cfg.max_duration:
            ref = Reference(seg.segment_id, seg.segment_id, seg.source_id, spk,
                            seg.start, seg.end, seg.text, dict(seg.metrics))
        elif seg.duration > cfg.max_duration and cfg.derive_from_long_segments:
            prefix = derive_prefix(seg, cfg.min_duration, cfg.max_duration)
            if prefix is None:
                continue
            end, words = prefix
            text = join_words(words)
            metrics = dict(seg.metrics)
            metrics["duration"] = end - seg.start
            metrics["chars_per_sec"] = spoken_char_count(text) / (end - seg.start)
            probs = [w.prob for w in words if w.prob is not None and w.kind == "word"]
            if probs:
                metrics["asr_confidence"] = float(np.mean(probs))
            ref = Reference(f"{seg.segment_id}-ref", seg.segment_id, seg.source_id, spk,
                            seg.start, round(end, 3), text, metrics)
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
        r for r in pool.get(speaker_id, [])
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
