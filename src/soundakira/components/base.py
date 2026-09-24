"""Component interfaces.

A component wraps one model. It is constructed cheaply, loads weights in
`load()` (called once per stage, not per source) and releases them in
`unload()`, so a single GPU only ever holds one stage's models.

Implementations import their heavy dependencies inside `load()`, which keeps
`import soundakira` fast and lets users install only the backends they use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.types import Span, Transcript, Turn
from soundakira.utils.device import free_memory


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ComponentContext:
    device: str = "cpu"
    hf_token: str | None = None
    cache_dir: Path | None = None


class Component(ABC):
    kind: ClassVar[str]
    Params: ClassVar[type[BaseModel]] = NoParams
    #: Bump when a code change alters outputs, to invalidate cached stage results.
    version: ClassVar[str] = "1"

    def __init__(self, params: BaseModel, ctx: ComponentContext):
        self.params = params
        self.ctx = ctx
        self.name = type(self).__name__

    def load(self) -> None:  # noqa: B027 - optional hook
        """Load model weights. Called once before processing a batch of sources."""

    def unload(self) -> None:
        for attr in list(vars(self)):
            if attr.startswith("_model") or attr.startswith("_pipeline"):
                setattr(self, attr, None)
        free_memory()

    def identity(self) -> dict[str, Any]:
        """What determines this component's output; part of stage fingerprints."""
        return {
            "name": self.name,
            "version": self.version,
            "params": self.params.model_dump(mode="json"),
        }


class Enhancer(Component):
    """Audio -> cleaner audio (vocal separation, denoising, dereverb)."""

    kind = "enhancer"
    #: Rate the model expects; the stage resamples to/from it. None = any.
    sample_rate: int | None = None

    @abstractmethod
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """(C, T) float32 at `sr` -> (C', T) float32 at `sr`."""


class VAD(Component):
    kind = "vad"
    sample_rate: int = 16000

    @abstractmethod
    def detect(self, audio: np.ndarray, sr: int) -> list[Span]: ...


class Diarizer(Component):
    kind = "diarizer"
    sample_rate: int = 16000

    @abstractmethod
    def diarize(self, audio: np.ndarray, sr: int) -> list[Turn]: ...


class Transcriber(Component):
    kind = "asr"
    sample_rate: int = 16000
    #: False for text-only backends; the pipeline then requires an aligner.
    provides_word_timestamps: bool = True

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, sr: int, speech: list[Span], language: str | None
    ) -> Transcript: ...

    def detect_language(
        self, audio: np.ndarray, sr: int, speech: list[Span]
    ) -> tuple[str | None, float | None]:
        return None, None


class Aligner(Component):
    """Adds or refines word timestamps (forced alignment)."""

    kind = "aligner"
    sample_rate: int = 16000

    @abstractmethod
    def align(self, transcript: Transcript, audio: np.ndarray, sr: int) -> Transcript: ...


class SpeakerEmbedder(Component):
    kind = "embedder"
    sample_rate: int = 16000

    @abstractmethod
    def embed(self, clips: list[np.ndarray], sr: int) -> np.ndarray:
        """Mono clips -> (N, D) embeddings (need not be normalised)."""


class Scorer(Component):
    """Per-segment quality metrics. Keys become filterable metric fields."""

    kind = "scorer"
    sample_rate: int | None = None

    @abstractmethod
    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]: ...
