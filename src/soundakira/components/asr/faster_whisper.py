"""Whisper via faster-whisper (CTranslate2): multilingual, word timestamps,
per-segment confidence stats. Install: ``pip install 'soundakira[asr]'``.

The defaults lean towards dataset quality over speed. They include several
anti-hallucination settings (no conditioning on previous text, silence-based
hallucination skipping, compression-ratio and log-prob fallbacks), because
hallucinated text in a TTS dataset is worse than missing text.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from soundakira.components.base import Transcriber
from soundakira.types import Span, Transcript, TranscriptSegment, Word
from soundakira.utils.device import resolve_device, split_cuda_device


class FasterWhisperParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "large-v3"
    compute_type: str = "auto"
    beam_size: int = 5
    batch_size: int = 0  # >0 uses BatchedInferencePipeline (faster, slightly less accurate)
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    hallucination_silence_threshold: float | None = 2.0
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    initial_prompt: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class FasterWhisperTranscriber(Transcriber):
    Params = FasterWhisperParams
    params: FasterWhisperParams

    def load(self) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device, index = split_cuda_device(resolve_device(self.ctx.device))
        if device not in ("cuda", "cpu"):
            device = "cpu"  # CTranslate2 has no MPS backend
        compute = self.params.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            self.params.model, device=device, device_index=index, compute_type=compute,
            download_root=str(self.ctx.cache_dir) if self.ctx.cache_dir else None,
        )
        self._batched = (
            BatchedInferencePipeline(model=self._model) if self.params.batch_size > 0 else None
        )

    def _kwargs(self, language: str | None) -> dict[str, Any]:
        p = self.params
        kw: dict[str, Any] = {
            "language": language,
            "beam_size": p.beam_size,
            "word_timestamps": True,
            "vad_filter": p.vad_filter,
            "no_speech_threshold": p.no_speech_threshold,
            "compression_ratio_threshold": p.compression_ratio_threshold,
            "log_prob_threshold": p.log_prob_threshold,
            "initial_prompt": p.initial_prompt,
        }
        if self._batched is None:
            kw["condition_on_previous_text"] = p.condition_on_previous_text
            kw["hallucination_silence_threshold"] = p.hallucination_silence_threshold
        else:
            kw["batch_size"] = p.batch_size
        kw.update(p.options)
        return kw

    def transcribe(
        self, audio: np.ndarray, sr: int, speech: list[Span], language: str | None
    ) -> Transcript:
        if sr != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        engine = self._batched or self._model
        seg_iter, info = engine.transcribe(audio, **self._kwargs(language))
        words: list[Word] = []
        segments: list[TranscriptSegment] = []
        for s in seg_iter:
            segments.append(TranscriptSegment(
                start=float(s.start), end=float(s.end), text=s.text.strip(),
                avg_logprob=float(s.avg_logprob), no_speech_prob=float(s.no_speech_prob),
                compression_ratio=float(s.compression_ratio),
            ))
            for w in s.words or []:
                words.append(Word(w.word, float(w.start), float(w.end), float(w.probability)))
        return Transcript(
            language=info.language, language_prob=float(info.language_probability),
            words=words, segments=segments, backend=f"faster_whisper:{self.params.model}",
        )

    def detect_language(
        self, audio: np.ndarray, sr: int, speech: list[Span]
    ) -> tuple[str | None, float | None]:
        # Detect on up to 30 s of actual speech, not the opening music.
        pieces, total = [], 0
        for s in speech:
            piece = audio[int(s.start * sr) : int(s.end * sr)]
            pieces.append(piece)
            total += len(piece)
            if total >= 30 * sr:
                break
        sample = np.concatenate(pieces)[: 30 * sr] if pieces else audio[: 30 * sr]
        _, info = self._model.transcribe(sample, language=None, beam_size=1, vad_filter=False)
        return info.language, float(info.language_probability)
