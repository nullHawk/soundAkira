"""Vocal isolation with python-audio-separator (UVR model zoo).

The default checkpoint is a BS-RoFormer vocals model, among the strongest
open vocal/instrumental separators. It strips music, effects and most
background noise, leaving all voices. Any audio-separator model filename
works, e.g. ``mel_band_roformer_kim_ft_unwa.ckpt``, or ``UVR-DeNoise.pth`` as a
second chain step.

Install: ``pip install 'soundakira[separation]'`` (use ``audio-separator[gpu]`` on CUDA).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from soundakira.audio.io import read_audio, write_audio
from soundakira.components.base import Enhancer


class AudioSeparatorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    stem: str = "Vocals"
    model_dir: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class AudioSeparatorEnhancer(Enhancer):
    Params = AudioSeparatorParams
    sample_rate = 44100
    params: AudioSeparatorParams

    def load(self) -> None:
        from audio_separator.separator import Separator

        self._tmp = Path(tempfile.mkdtemp(prefix="soundakira-sep-"))
        kwargs: dict[str, Any] = {
            "output_dir": str(self._tmp / "out"),
            "output_format": "WAV",
            "output_single_stem": self.params.stem,
            "log_level": logging.WARNING,
            **self.params.options,
        }
        if self.params.model_dir:
            kwargs["model_file_dir"] = self.params.model_dir
        self._model = Separator(**kwargs)
        self._model.load_model(model_filename=self.params.model)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        out_dir = self._tmp / "out"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        src = self._tmp / "chunk.wav"
        write_audio(src, audio, sr, subtype="FLOAT")
        produced = self._model.separate(str(src))
        stem = f"({self.params.stem})".lower()
        candidates = [out_dir / Path(p).name for p in produced] + list(out_dir.glob("*"))
        match = next((p for p in candidates if stem in p.name.lower() and p.exists()), None)
        if match is None:
            raise RuntimeError(f"audio-separator produced no {self.params.stem!r} stem: {produced}")
        vocals, vsr = read_audio(match, mono=False)
        if vsr != sr:
            from soundakira.audio.io import resample

            vocals = resample(vocals, vsr, sr)
        return vocals

    def unload(self) -> None:
        if getattr(self, "_tmp", None):
            shutil.rmtree(self._tmp, ignore_errors=True)
        super().unload()
