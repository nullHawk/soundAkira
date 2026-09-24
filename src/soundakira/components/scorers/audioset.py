"""Speech / singing / music probabilities from an AudioSet classifier (AST).

Vocal separation keeps singing, because singing is voice. Theme songs, musical
numbers and background vocals therefore reach the dataset as "speakers"
unless something detects them. This scorer runs an Audio Spectrogram
Transformer fine-tuned on AudioSet over each clip, in 10 s windows, and reports:

- `speech_prob`: mean "Speech" probability across windows
- `singing_prob`: max over windows of the strongest singing-type label
- `music_prob`: max over windows of "Music"

The default export filter drops clips with `singing_prob > 0.5`.

Install: ``pip install 'soundakira[quality]'`` (transformers).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Scorer
from soundakira.utils.device import resolve_device

SINGING_LABELS = (
    "Singing",
    "Song",
    "Choir",
    "Chant",
    "Vocal music",
    "A capella",
    "Rapping",
    "Humming",
    "Yodeling",
    "Male singing",
    "Female singing",
    "Child singing",
    "Synthetic singing",
)


class AudioSetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    window_seconds: float = 10.0


class AudioSetScorer(Scorer):
    Params = AudioSetParams
    sample_rate = 16000
    params: AudioSetParams

    def load(self) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._torch = torch
        self._device = torch.device(resolve_device(self.ctx.device))
        self._features = ASTFeatureExtractor.from_pretrained(self.params.model)
        self._model = ASTForAudioClassification.from_pretrained(self.params.model)
        self._model.to(self._device).eval()
        labels = self._model.config.id2label
        index = {name: int(i) for i, name in labels.items()}
        self._speech = [index["Speech"]]
        self._music = [index["Music"]]
        self._singing = [index[name] for name in SINGING_LABELS if name in index]

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        win = int(self.params.window_seconds * sr)
        windows = [audio[i : i + win] for i in range(0, max(1, len(audio)), win)]
        windows = [w for w in windows if len(w) >= sr] or [audio]
        inputs = self._features(windows, sampling_rate=sr, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self._model(inputs["input_values"].to(self._device)).logits
        probs = self._torch.sigmoid(logits).float().cpu().numpy()
        return {
            "speech_prob": float(probs[:, self._speech].max(axis=1).mean()),
            "singing_prob": float(probs[:, self._singing].max(axis=1).max()),
            "music_prob": float(probs[:, self._music].max(axis=1).max()),
        }
