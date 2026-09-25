"""Forced alignment with a character-level CTC model (wav2vec2 family).

Word timings come from aligning the *known* transcript against the CTC model's
frame-level character probabilities (Viterbi over the CTC trellis), not from the
ASR model's own attention. Fine-tuned Whisper models often have unreliable
attention-based timestamps, because their alignment heads were copied from the
base model. Example: a Hindi fine-tune produced single "words" 10-24 s long.

Works for any language with a character-level CTC checkpoint: Hindi
`theainerd/Wav2Vec2-large-xlsr-hindi` (default), English
`facebook/wav2vec2-base-960h`, etc. Each ASR segment is aligned within its own
time window (plus `pad`), so memory stays bounded on long audio.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Aligner
from soundakira.types import Transcript, Word
from soundakira.utils.device import resolve_device
from soundakira.utils.text import with_leading_space

log = logging.getLogger(__name__)


class CTCAlignerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "theainerd/Wav2Vec2-large-xlsr-hindi"
    pad: float = 0.3  # seconds of context around each ASR segment


def ctc_align(emission: np.ndarray, tokens: list[int], blank: int) -> list[int] | None:
    """Viterbi forced alignment. `emission` is (T, V) log-probs, `tokens` the
    target ids. Returns the frame at which each token is emitted, or None if
    the text doesn't fit the audio (more tokens than frames)."""
    t_len, j_len = emission.shape[0], len(tokens)
    if j_len == 0 or j_len > t_len:
        return None
    tok = np.asarray(tokens)
    trellis = np.full((t_len + 1, j_len + 1), -np.inf)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = np.cumsum(emission[:, blank])
    for t in range(t_len):
        stay = trellis[t, 1:] + emission[t, blank]
        change = trellis[t, :-1] + emission[t, tok]
        trellis[t + 1, 1:] = np.maximum(stay, change)
    frames = [0] * j_len
    j = j_len
    for t in range(t_len, 0, -1):
        if j == 0:
            break
        stayed = trellis[t - 1, j] + emission[t - 1, blank]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tok[j - 1]]
        if changed > stayed:
            frames[j - 1] = t - 1
            j -= 1
    return None if j > 0 else frames


class CTCAligner(Aligner):
    Params = CTCAlignerParams
    params: CTCAlignerParams

    def load(self) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

        self._torch = torch
        self._device = torch.device(resolve_device(self.ctx.device))
        self._fe = AutoFeatureExtractor.from_pretrained(self.params.model)
        self._model = AutoModelForCTC.from_pretrained(self.params.model).to(self._device).eval()
        tok = AutoTokenizer.from_pretrained(self.params.model)
        self._vocab = tok.get_vocab()
        self._blank = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._delim = self._vocab.get(tok.word_delimiter_token or "|")
        self.sample_rate = int(self._fe.sampling_rate)

    def _char_id(self, ch: str) -> int | None:
        for c in (ch, ch.lower(), ch.upper()):
            if c in self._vocab:
                return self._vocab[c]
        return None

    def _emission(self, clip: np.ndarray) -> np.ndarray:
        inputs = self._fe(clip, sampling_rate=self.sample_rate, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self._model(inputs.input_values.to(self._device)).logits[0]
        return self._torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def align(self, transcript: Transcript, audio: np.ndarray, sr: int) -> Transcript:
        lang = transcript.language
        total = len(audio) / sr
        out: list[Word] = []
        failed = 0
        segs = transcript.segments
        for k, seg in enumerate(segs):
            texts = seg.text.split()
            if not texts:
                continue
            # Token sequence: characters of each word, words joined by the delimiter.
            tokens: list[int] = []
            owner: list[int] = []  # word index per token (-1 for delimiters)
            for wi, text in enumerate(texts):
                ids = [i for i in (self._char_id(c) for c in text) if i is not None]
                if tokens and ids and self._delim is not None:
                    tokens.append(self._delim)
                    owner.append(-1)
                tokens += ids
                owner += [wi] * len(ids)
            # ASR segment boundaries can be off too. If the text doesn't fit, retry
            # with the window widened to the neighbouring segments' boundaries.
            prev_end = segs[k - 1].end if k > 0 else 0.0
            next_start = segs[k + 1].start if k + 1 < len(segs) else total
            windows = [
                (seg.start - self.params.pad, seg.end + self.params.pad),
                (
                    min(seg.start, prev_end) - self.params.pad,
                    max(seg.end, next_start) + self.params.pad,
                ),
            ]
            frames = None
            for w0, w1 in windows:
                s0, s1 = max(0.0, w0), min(total, w1)
                clip = audio[int(s0 * sr) : int(s1 * sr)]
                if len(clip) < sr // 10:
                    break
                emission = self._emission(clip)
                frames = ctc_align(emission, tokens, self._blank)
                if frames is not None:
                    frame_s = (s1 - s0) / emission.shape[0]
                    break
            if frames is None:
                failed += 1
                continue
            spans: dict[int, list[int]] = {}
            for f, o in zip(frames, owner):
                if o >= 0:
                    spans.setdefault(o, []).append(f)
            for wi, text in enumerate(texts):
                if wi not in spans:  # no alignable characters (digits, symbols)
                    continue
                fs = spans[wi]
                start = s0 + min(fs) * frame_s
                end = s0 + (max(fs) + 1) * frame_s
                char_ids = [i for i in (self._char_id(c) for c in text) if i is not None]
                probs = [np.exp(emission[f, i]) for f, i in zip(fs, char_ids)]
                out.append(
                    Word(
                        with_leading_space(text, lang),
                        round(start, 3),
                        round(end, 3),
                        float(np.mean(probs)) if probs else None,
                    )
                )
        if failed:
            log.warning(
                "CTC alignment failed for %d of %d segments (text longer than audio)",
                failed,
                len(transcript.segments),
            )
        out.sort(key=lambda w: w.start)
        return Transcript(
            transcript.language,
            transcript.language_prob,
            out,
            transcript.segments,
            backend=f"{transcript.backend}+ctc:{self.params.model}",
        )
