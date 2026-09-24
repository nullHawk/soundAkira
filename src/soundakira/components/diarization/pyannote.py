"""pyannote.audio speaker diarization.

The default `pyannote/speaker-diarization-community-1` needs pyannote.audio
>= 4.0. `pyannote/speaker-diarization-3.1` works with 3.x and 4.x. Both are
gated: accept the model terms on Hugging Face and set HF_TOKEN.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Diarizer
from soundakira.types import Turn
from soundakira.utils.device import resolve_device


class PyannoteParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "pyannote/speaker-diarization-community-1"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


def load_pretrained(loader, name: str, token: str | None):
    """pyannote renamed `use_auth_token` to `token` in 4.0; support both."""
    try:
        return loader(name, token=token)
    except TypeError:
        return loader(name, use_auth_token=token)


class PyannoteDiarizer(Diarizer):
    Params = PyannoteParams
    params: PyannoteParams

    def load(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        self._torch = torch
        self._pipeline = load_pretrained(
            Pipeline.from_pretrained, self.params.model, self.ctx.hf_token
        )
        if self._pipeline is None:
            raise RuntimeError(
                f"could not load {self.params.model}: "
                "accept its terms on huggingface.co and set HF_TOKEN"
            )
        self._pipeline.to(torch.device(resolve_device(self.ctx.device)))

    def diarize(self, audio: np.ndarray, sr: int) -> list[Turn]:
        hints = {
            k: v
            for k, v in {
                "num_speakers": self.params.num_speakers,
                "min_speakers": self.params.min_speakers,
                "max_speakers": self.params.max_speakers,
            }.items()
            if v is not None
        }
        waveform = self._torch.from_numpy(np.ascontiguousarray(audio))[None]
        out = self._pipeline({"waveform": waveform, "sample_rate": sr}, **hints)
        # 4.x returns DiarizeOutput; 3.x an Annotation (or a tuple with embeddings).
        annotation = getattr(out, "speaker_diarization", None)
        if annotation is None:
            annotation = out[0] if isinstance(out, tuple) else out
        return [
            Turn(float(seg.start), float(seg.end), str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
