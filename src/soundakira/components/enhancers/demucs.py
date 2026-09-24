"""Vocal isolation with Demucs (htdemucs_ft). Install: ``pip install 'soundakira[demucs]'``."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Enhancer
from soundakira.utils.device import resolve_device


class DemucsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "htdemucs_ft"
    stem: str = "vocals"
    shifts: int = 1
    overlap: float = 0.25


class DemucsEnhancer(Enhancer):
    Params = DemucsParams
    sample_rate = 44100
    params: DemucsParams

    def load(self) -> None:
        import torch
        from demucs.pretrained import get_model

        self._torch = torch
        self._device = resolve_device(self.ctx.device)
        self._model = get_model(self.params.model).to(self._device).eval()
        self.sample_rate = int(self._model.samplerate)
        if self.params.stem not in self._model.sources:
            raise ValueError(f"model has no stem {self.params.stem!r}: {self._model.sources}")

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from demucs.apply import apply_model

        torch = self._torch
        channels = self._model.audio_channels
        x = audio if audio.shape[0] == channels else np.repeat(audio.mean(0, keepdims=True), channels, 0)
        wav = torch.from_numpy(np.ascontiguousarray(x))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.inference_mode():
            sources = apply_model(
                self._model, ((wav - mean) / std)[None], device=self._device,
                shifts=self.params.shifts, split=True, overlap=self.params.overlap, progress=False,
            )[0]
        stem = sources[self._model.sources.index(self.params.stem)] * std + mean
        return stem.cpu().numpy().astype(np.float32)
