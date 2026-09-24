"""Forced alignment with WhisperX's wav2vec2 CTC models: tighter word
boundaries than Whisper's attention-based timestamps, and word timings for
ASR backends that give text only. Install: ``pip install 'soundakira[align]'``.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Aligner
from soundakira.types import Transcript, Word
from soundakira.utils.device import resolve_device
from soundakira.utils.text import with_leading_space


class WhisperXParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None  # None = WhisperX default for the language
    fallback_language: str = "en"


class WhisperXAligner(Aligner):
    Params = WhisperXParams
    params: WhisperXParams

    def load(self) -> None:
        import whisperx

        self._whisperx = whisperx
        self._device = resolve_device(self.ctx.device)
        self._models: dict[str, tuple] = {}

    def _model_for(self, language: str):
        if language not in self._models:
            self._models[language] = self._whisperx.load_align_model(
                language_code=language, device=self._device, model_name=self.params.model
            )
        return self._models[language]

    def align(self, transcript: Transcript, audio: np.ndarray, sr: int) -> Transcript:
        lang = (transcript.language or self.params.fallback_language).split("-")[0]
        model, meta = self._model_for(lang)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in transcript.segments]
        result = self._whisperx.align(
            segs, model, meta, audio, self._device, return_char_alignments=False
        )
        raw = result.get("word_segments", [])
        words: list[Word] = []
        for i, w in enumerate(raw):
            start, end = w.get("start"), w.get("end")
            if start is None or end is None:
                # Tokens the aligner can't place (digits, symbols) get the gap
                # between their neighbours.
                prev_end = words[-1].end if words else 0.0
                nxt = next(
                    (r["start"] for r in raw[i + 1 :] if r.get("start") is not None), prev_end
                )
                start, end = prev_end, max(prev_end, nxt)
            words.append(
                Word(with_leading_space(w["word"], lang), float(start), float(end), w.get("score"))
            )
        return Transcript(
            transcript.language,
            transcript.language_prob,
            words,
            transcript.segments,
            backend=f"{transcript.backend}+whisperx",
        )

    def unload(self) -> None:
        self._models = {}
        super().unload()
