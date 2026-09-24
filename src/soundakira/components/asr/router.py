"""Language router: send each source to the best ASR for its language.

This is how new languages get added without touching the pipeline. English
can go to Whisper or Parakeet, Hindi or Hinglish to an Indic model, and so on::

    asr:
      name: router
      params:
        default: en
        detect_with: en          # route whose detect_language() is used when no hint
        routes:
          en: {name: faster_whisper, params: {model: large-v3}}
          hi: {name: faster_whisper, params: {model: large-v3}}
          hi-en: {name: my_hinglish_asr}   # any registered/plugin transcriber

The language comes from, in order: the source's hint (manifest or config),
then detection, then `default`. Only routes that are actually used get loaded.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict

from soundakira.components.base import Transcriber
from soundakira.config import ComponentSpec
from soundakira.types import Span, Transcript
from soundakira.utils.text import normalize_language

log = logging.getLogger(__name__)


class RouterParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routes: dict[str, ComponentSpec]
    default: str
    detect_with: str | None = None


class LanguageRouter(Transcriber):
    Params = RouterParams
    params: RouterParams

    def load(self) -> None:
        self._children: dict[str, Transcriber] = {}
        if self.params.default not in self.params.routes:
            raise ValueError(f"default route {self.params.default!r} not in routes")

    def _child(self, route: str) -> Transcriber:
        if route not in self._children:
            from soundakira import registry

            spec = self.params.routes[route]
            child = registry.create("asr", spec.name, spec.params, self.ctx)
            assert isinstance(child, Transcriber)
            child.load()
            self._children[route] = child
        return self._children[route]

    def _route_for(self, language: str | None) -> str | None:
        lang = normalize_language(language)
        if not lang:
            return None
        if lang in self.params.routes:
            return lang
        base = lang.split("-")[0]
        return base if base in self.params.routes else None

    def transcribe(
        self, audio: np.ndarray, sr: int, speech: list[Span], language: str | None
    ) -> Transcript:
        route = self._route_for(language)
        if route is None and self.params.detect_with:
            detected, prob = self._child(self.params.detect_with).detect_language(audio, sr, speech)
            route = self._route_for(detected)
            language = language or detected
            log.info("detected language %s (p=%.2f) -> route %s", detected, prob or 0, route)
        route = route or self.params.default
        transcript = self._child(route).transcribe(audio, sr, speech, normalize_language(language))
        transcript.backend = f"router[{route}]->{transcript.backend}"
        return transcript

    def unload(self) -> None:
        for child in getattr(self, "_children", {}).values():
            child.unload()
        self._children = {}
        super().unload()
