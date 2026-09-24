"""Core data types shared by every stage.

All times are in seconds, relative to the start of the extracted audio track.
Everything here is plain data (JSON round-trippable) so stages can persist
their results to the workspace and be resumed independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

WordKind = Literal["word", "event"]


@dataclass(frozen=True, slots=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True, slots=True)
class Turn:
    """A diarization turn: `speaker` is a label local to one source."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Word:
    """One timed token from ASR.

    `text` carries any leading whitespace needed to concatenate it with the
    previous word (Whisper convention), so ``"".join(w.text for w in words)``
    is correct for both spaced scripts (English, Hindi) and unspaced ones
    (Chinese, Japanese).

    `kind="event"` marks a non-verbal tag such as ``[laugh]`` emitted by a
    tag-aware ASR. Events are kept out of `Segment.text` and only appear in
    `Segment.text_tagged`.
    """

    text: str
    start: float
    end: float
    prob: float | None = None
    kind: WordKind = "word"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class TranscriptSegment:
    """A decoder-level segment, kept for its confidence statistics."""

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


@dataclass(slots=True)
class Transcript:
    language: str | None
    language_prob: float | None
    words: list[Word] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transcript:
        return cls(
            language=d.get("language"),
            language_prob=d.get("language_prob"),
            words=[Word(**w) for w in d.get("words", [])],
            segments=[TranscriptSegment(**s) for s in d.get("segments", [])],
            backend=d.get("backend", ""),
        )


@dataclass(slots=True)
class Segment:
    """A single-speaker stretch of speech: the unit that becomes a dataset row
    (if long enough) or a reference-prompt candidate."""

    segment_id: str
    source_id: str
    speaker: str
    start: float
    end: float
    text: str
    text_tagged: str | None
    language: str | None
    words: list[Word] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segment:
        d = dict(d)
        d["words"] = [Word(**w) for w in d.get("words", [])]
        return cls(**d)
