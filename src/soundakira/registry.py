"""Component registry: the extension point of the pipeline.

Every model-backed step (separation, VAD, diarization, ASR, alignment,
speaker embedding, quality scoring) is a *component* looked up by
`(kind, name)`. Three ways to add one:

1. Built-ins below: lazily imported, so no heavy deps load until used.
2. In-process: decorate a class with ``@register("asr", "my_asr")``.
3. From another package, with no changes here: an entry point in its pyproject::

       [project.entry-points."soundakira.plugins"]
       "asr.my_asr" = "my_pkg.asr:MyTranscriber"
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

from pydantic import ValidationError

from soundakira.components.base import Component, ComponentContext

log = logging.getLogger(__name__)

KINDS = ("enhancer", "vad", "diarizer", "asr", "aligner", "embedder", "scorer")

_BUILTINS: dict[str, dict[str, str]] = {
    "enhancer": {
        "roformer": "soundakira.components.enhancers.audio_separator:AudioSeparatorEnhancer",
        "audio_separator": "soundakira.components.enhancers.audio_separator:AudioSeparatorEnhancer",
        "demucs": "soundakira.components.enhancers.demucs:DemucsEnhancer",
        "deepfilternet": "soundakira.components.enhancers.deepfilternet:DeepFilterNetEnhancer",
    },
    "vad": {
        "silero": "soundakira.components.vad.silero:SileroVAD",
        "energy": "soundakira.components.vad.energy:EnergyVAD",
    },
    "diarizer": {
        "pyannote": "soundakira.components.diarization.pyannote:PyannoteDiarizer",
        "cluster": "soundakira.components.diarization.cluster:ClusterDiarizer",
    },
    "asr": {
        "faster_whisper": "soundakira.components.asr.faster_whisper:FasterWhisperTranscriber",
        "parakeet": "soundakira.components.asr.parakeet:ParakeetTranscriber",
        "router": "soundakira.components.asr.router:LanguageRouter",
    },
    "aligner": {
        "whisperx": "soundakira.components.aligners.whisperx:WhisperXAligner",
    },
    "embedder": {
        "pyannote": "soundakira.components.embedders.pyannote:PyannoteEmbedder",
    },
    "scorer": {
        "signal": "soundakira.components.scorers.signal:SignalScorer",
        "dnsmos": "soundakira.components.scorers.dnsmos:DNSMOSScorer",
    },
}

_registered: dict[str, dict[str, type[Component] | str]] = {
    k: dict(v) for k, v in _BUILTINS.items()
}
_entry_points_loaded = False

C = TypeVar("C", bound=type[Component])


class ComponentError(RuntimeError):
    pass


def register(kind: str, name: str) -> Callable[[C], C]:
    if kind not in KINDS:
        raise ValueError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def deco(cls: C) -> C:
        _registered[kind][name] = cls
        return cls

    return deco


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for ep in entry_points(group="soundakira.plugins"):
        kind, _, name = ep.name.partition(".")
        if kind in KINDS and name:
            _registered[kind].setdefault(name, ep.value)
        else:
            log.warning("ignoring plugin entry point %r (expected '<kind>.<name>')", ep.name)


def available(kind: str) -> list[str]:
    _load_entry_points()
    return sorted(_registered[kind])


def resolve(kind: str, name: str) -> type[Component]:
    _load_entry_points()
    try:
        target = _registered[kind][name]
    except KeyError:
        raise ComponentError(
            f"no {kind} named {name!r}; available: {', '.join(available(kind))}"
        ) from None
    if isinstance(target, str):
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ComponentError(
                f"{kind} {name!r} needs an optional dependency that is not installed ({e}). "
                "See the README 'Installation' section for the matching extra."
            ) from e
        target = getattr(module, attr)
        _registered[kind][name] = target
    return target


def create(kind: str, name: str, params: dict[str, Any], ctx: ComponentContext) -> Component:
    cls = resolve(kind, name)
    try:
        typed = cls.Params.model_validate(params or {})
    except ValidationError as e:
        raise ComponentError(f"invalid params for {kind} {name!r}:\n{e}") from e
    component = cls(typed, ctx)
    component.name = name
    return component
