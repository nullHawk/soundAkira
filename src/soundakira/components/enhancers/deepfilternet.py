"""Residual noise suppression with DeepFilterNet3 (48 kHz). Use it as a
second chain step after vocal separation to clean up hum, hiss and ambience
that the separator counts as "voice". Set `atten_lim_db` (e.g. 12) to keep
it from over-suppressing and adding artifacts.

Install: ``pip install 'soundakira[denoise]'``.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Enhancer


class DeepFilterNetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None  # None = DeepFilterNet3 default
    atten_lim_db: float | None = None


class DeepFilterNetEnhancer(Enhancer):
    Params = DeepFilterNetParams
    sample_rate = 48000
    params: DeepFilterNetParams

    def load(self) -> None:
        import torch
        from df.enhance import enhance, init_df

        self._torch = torch
        self._enhance = enhance
        self._model, self._state, _ = init_df(model_base_dir=self.params.model)
        self.sample_rate = int(self._state.sr())

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        wav = self._torch.from_numpy(np.ascontiguousarray(audio))
        out = self._enhance(self._model, self._state, wav, atten_lim_db=self.params.atten_lim_db)
        return out.cpu().numpy().astype(np.float32)
