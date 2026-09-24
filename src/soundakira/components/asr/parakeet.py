"""NVIDIA Parakeet-TDT via NeMo. `parakeet-tdt-0.6b-v3` (25 European
languages) and `-v2` (English) are near the top of the Open ASR Leaderboard,
fast, and give accurate word timestamps with punctuation and casing. They
report no word confidences, so `asr_confidence` filters are skipped for these
segments (`on_missing: keep`).

Install: ``pip install 'soundakira[parakeet]'``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.audio.io import write_audio
from soundakira.components.base import Transcriber
from soundakira.types import Span, Transcript, TranscriptSegment, Word
from soundakira.utils.device import resolve_device
from soundakira.utils.spans import group_spans
from soundakira.utils.text import with_leading_space


class ParakeetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "nvidia/parakeet-tdt-0.6b-v3"
    language: str = "en"  # reported language; the model itself does not do LID
    chunk_max_seconds: float = 60.0
    chunk_max_gap: float = 1.0
    batch_size: int = 8


class ParakeetTranscriber(Transcriber):
    Params = ParakeetParams
    params: ParakeetParams

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.params.model)
        device = resolve_device(self.ctx.device)
        if device.startswith("cuda"):
            self._model = self._model.to(torch.device(device))
        self._model.eval()

    def transcribe(
        self, audio: np.ndarray, sr: int, speech: list[Span], language: str | None
    ) -> Transcript:
        lang = language or self.params.language
        chunks = group_spans(speech, self.params.chunk_max_seconds, self.params.chunk_max_gap)
        tmp = Path(tempfile.mkdtemp(prefix="soundakira-asr-"))
        try:
            paths = []
            for i, c in enumerate(chunks):
                p = tmp / f"{i:06d}.wav"
                write_audio(p, audio[int(c.start * sr) : int(c.end * sr)], sr)
                paths.append(str(p))
            hyps = self._model.transcribe(paths, batch_size=self.params.batch_size, timestamps=True) if paths else []
            if isinstance(hyps, tuple):  # some NeMo versions return (best, all)
                hyps = hyps[0]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        words: list[Word] = []
        segments: list[TranscriptSegment] = []
        for chunk, hyp in zip(chunks, hyps):
            text = getattr(hyp, "text", str(hyp)).strip()
            if not text:
                continue
            segments.append(TranscriptSegment(chunk.start, chunk.end, text))
            for w in (getattr(hyp, "timestamp", None) or {}).get("word", []):
                words.append(Word(
                    with_leading_space(w["word"], lang),
                    chunk.start + float(w["start"]), chunk.start + float(w["end"]),
                ))
        return Transcript(lang, None, words, segments, backend=f"parakeet:{self.params.model}")
