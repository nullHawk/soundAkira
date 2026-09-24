"""Speaker embeddings with pyannote's WeSpeaker ResNet34 (the same family
the diarization pipeline uses internally)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import SpeakerEmbedder
from soundakira.components.diarization.pyannote import load_pretrained
from soundakira.utils.device import resolve_device


class PyannoteEmbedderParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"


class PyannoteEmbedder(SpeakerEmbedder):
    Params = PyannoteEmbedderParams
    params: PyannoteEmbedderParams

    def load(self) -> None:
        import torch
        from pyannote.audio import Inference, Model

        self._torch = torch
        model = load_pretrained(Model.from_pretrained, self.params.model, self.ctx.hf_token)
        self._model = Inference(model, window="whole")
        self._model.to(torch.device(resolve_device(self.ctx.device)))

    def embed(self, clips: list[np.ndarray], sr: int) -> np.ndarray:
        out = []
        for clip in clips:
            wav = self._torch.from_numpy(np.ascontiguousarray(clip))[None]
            out.append(np.asarray(self._model({"waveform": wav, "sample_rate": sr})).reshape(-1))
        return np.stack(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)
