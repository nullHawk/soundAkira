"""DNSMOS P.835 (via torchmetrics): non-intrusive MOS for signal, background
and overall quality. Install: ``pip install 'soundakira[quality]'``."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Scorer


class DNSMOSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    personalized: bool = False


class DNSMOSScorer(Scorer):
    Params = DNSMOSParams
    sample_rate = 16000
    params: DNSMOSParams

    def load(self) -> None:
        import torch
        from torchmetrics.functional.audio.dnsmos import (
            deep_noise_suppression_mean_opinion_score as dnsmos,
        )

        self._torch = torch
        self._model = dnsmos

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        vals = self._model(self._torch.from_numpy(audio), sr, self.params.personalized)
        p808, sig, bak, ovr = (float(v) for v in vals.reshape(-1)[:4])
        return {"dnsmos_p808": p808, "dnsmos_sig": sig, "dnsmos_bak": bak, "dnsmos_ovr": ovr}
