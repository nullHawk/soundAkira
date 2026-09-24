"""Deterministic stand-ins for model components, registered through the
public plugin API. They exercise the whole pipeline with no ML dependencies.

Synthetic "speech": each speaker is a pure tone at its own frequency; each
word is a 0.4 s burst with 0.1 s gaps. The fakes recover speakers and words
from those signals, so the data flow (and the timestamps) are real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from soundakira.components.base import Diarizer, SpeakerEmbedder, Transcriber
from soundakira.registry import register
from soundakira.types import Span, Transcript, TranscriptSegment, Turn, Word

SR = 16000
WORD, GAP = 0.4, 0.1
FREQS = {"alice": 220.0, "bob": 330.0, "carol": 440.0}


def dominant_freq(x: np.ndarray, sr: int) -> float:
    if len(x) < 64 or np.max(np.abs(x)) < 1e-3:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spec))])


def synth_source(
    path: Path, script: list[tuple[str, int]], sr: int = 44100, pause: float = 1.0
) -> list[tuple[str, float, float]]:
    """script = [(speaker, n_words), ...] -> writes audio, returns (speaker, start, end) turns."""
    chunks, turns, t = [np.zeros(int(pause * sr))], [], pause
    for speaker, n in script:
        f = FREQS[speaker]
        start = t
        for _ in range(n):
            tt = np.arange(int(WORD * sr)) / sr
            chunks.append((0.5 * np.sin(2 * np.pi * f * tt)).astype(np.float32))
            chunks.append(np.zeros(int(GAP * sr), np.float32))
            t += WORD + GAP
        turns.append((speaker, start, t - GAP))
        chunks.append(np.zeros(int(pause * sr), np.float32))
        t += pause
    sf.write(str(path), np.concatenate(chunks), sr)
    return turns


def _bursts(audio: np.ndarray, sr: int) -> list[Span]:
    hop = int(0.01 * sr)
    n = len(audio) // hop
    active = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)) > 0.05
    spans, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            spans.append(Span(start * hop / sr, i * hop / sr))
            start = None
    if start is not None:
        spans.append(Span(start * hop / sr, n * hop / sr))
    return spans


@register("diarizer", "fake")
class FakeDiarizer(Diarizer):
    def diarize(self, audio: np.ndarray, sr: int) -> list[Turn]:
        turns: list[Turn] = []
        for b in _bursts(audio, sr):
            f = dominant_freq(audio[int(b.start * sr) : int(b.end * sr)], sr)
            label = min(FREQS, key=lambda k: abs(FREQS[k] - f))
            if turns and turns[-1].speaker == label and b.start - turns[-1].end < 0.5:
                turns[-1] = Turn(turns[-1].start, b.end, label)
            else:
                turns.append(Turn(b.start, b.end, label))
        return [
            Turn(t.start, t.end, f"SPEAKER_{sorted(FREQS).index(t.speaker):02d}") for t in turns
        ]


@register("asr", "fake")
class FakeTranscriber(Transcriber):
    def transcribe(
        self, audio: np.ndarray, sr: int, speech: list[Span], language: str | None
    ) -> Transcript:
        words = []
        for i, b in enumerate(_bursts(audio, sr)):
            text = f" word{i}" + ("." if i % 6 == 5 else "")
            words.append(Word(text, b.start, b.end, 0.9))
        segs = (
            [TranscriptSegment(words[0].start, words[-1].end, "", -0.2, 0.01, 1.2)] if words else []
        )
        return Transcript(language or "en", 1.0, words, segs, backend="fake")


@register("embedder", "fake")
class FakeEmbedder(SpeakerEmbedder):
    centers = np.arange(100.0, 600.0, 10.0)

    def embed(self, clips: list[np.ndarray], sr: int) -> np.ndarray:
        out = []
        for clip in clips:
            f = dominant_freq(clip, sr)
            out.append(np.exp(-((self.centers - f) ** 2) / (2 * 15.0**2)))
        return np.stack(out).astype(np.float32)
