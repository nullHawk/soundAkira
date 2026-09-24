"""Silero VAD. Install: ``pip install 'soundakira[vad]'``."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import VAD
from soundakira.types import Span


class SileroParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    pad_ms: int = 30
    onnx: bool = False


class SileroVAD(VAD):
    Params = SileroParams
    params: SileroParams

    def load(self) -> None:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        self._torch = torch
        self._get_ts = get_speech_timestamps
        self._model = load_silero_vad(onnx=self.params.onnx)

    def detect(self, audio: np.ndarray, sr: int) -> list[Span]:
        p = self.params
        stamps = self._get_ts(
            self._torch.from_numpy(audio), self._model, sampling_rate=sr,
            threshold=p.threshold, min_speech_duration_ms=p.min_speech_ms,
            min_silence_duration_ms=p.min_silence_ms, speech_pad_ms=p.pad_ms,
        )
        # Sample indices, not return_seconds=True: that rounds to 0.1 s.
        return [Span(s["start"] / sr, s["end"] / sr) for s in stamps]
