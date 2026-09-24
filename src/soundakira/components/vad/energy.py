"""Dependency-free energy VAD with an adaptive noise floor.

Much weaker than Silero on noisy audio, but it needs no model: useful on
already-separated vocals, for CPU-only smoke runs, and in tests.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import VAD
from soundakira.types import Span
from soundakira.utils.spans import merge_spans


class EnergyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_ms: float = 30.0
    margin_db: float = 12.0
    floor_percentile: float = 10.0
    absolute_floor_db: float = -55.0
    min_speech_ms: float = 250.0
    min_silence_ms: float = 300.0


class EnergyVAD(VAD):
    Params = EnergyParams
    params: EnergyParams

    def detect(self, audio: np.ndarray, sr: int) -> list[Span]:
        p = self.params
        hop = max(1, int(sr * p.frame_ms / 1000))
        n = len(audio) // hop
        if n == 0:
            return []
        frames = audio[: n * hop].reshape(n, hop)
        db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
        threshold = max(np.percentile(db, p.floor_percentile) + p.margin_db, p.absolute_floor_db)
        active = db > threshold
        spans, start = [], None
        for i, a in enumerate(active):
            if a and start is None:
                start = i
            elif not a and start is not None:
                spans.append(Span(start * hop / sr, i * hop / sr))
                start = None
        if start is not None:
            spans.append(Span(start * hop / sr, n * hop / sr))
        merged = merge_spans(spans, max_gap=p.min_silence_ms / 1000)
        return [s for s in merged if s.duration >= p.min_speech_ms / 1000]
