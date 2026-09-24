"""Per-source pipeline stages.

Each stage reads upstream artifacts from the source's workspace, writes its
own artifacts, and declares what determines its output (`identity`). The
runner uses that to skip finished work and to redo everything downstream when
a setting or model changes. To add a stage, subclass `Stage` and insert it
into `STAGES`.
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from soundakira import registry
from soundakira.audio import ffmpeg
from soundakira.audio.chunking import process_file_in_chunks
from soundakira.audio.io import audio_duration, fit_length, read_audio, resample
from soundakira.components.base import (
    Aligner,
    Component,
    ComponentContext,
    Diarizer,
    Enhancer,
    Scorer,
    SpeakerEmbedder,
    Transcriber,
)
from soundakira.config import ComponentSpec, PipelineConfig
from soundakira.pipeline.workspace import SourceWorkspace
from soundakira.segmentation.builder import SegmentationParams, build_segments
from soundakira.types import Segment, Span, Transcript, Turn
from soundakira.utils.io import read_json, read_jsonl, write_json, write_jsonl
from soundakira.utils.text import container_language_tags, normalize_language

log = logging.getLogger(__name__)

ANALYSIS_SR = 16000


class Stage(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: PipelineConfig, ctx: ComponentContext):
        self.cfg = cfg
        self.ctx = ctx
        self.components: list[Component] = self.build_components()

    # -- declaration -----------------------------------------------------------
    def build_components(self) -> list[Component]:
        return []

    def params(self) -> dict[str, Any]:
        return {}

    @property
    def workers(self) -> int:
        return 1

    def identity(self) -> dict[str, Any]:
        return {"params": self.params(), "components": [c.identity() for c in self.components]}

    @abstractmethod
    def outputs(self, ws: SourceWorkspace) -> list[Path]: ...

    # -- execution -------------------------------------------------------------
    def setup(self) -> None:
        for c in self.components:
            try:
                c.load()
            except ImportError as e:
                raise registry.ComponentError(
                    f"{c.kind} {c.name!r} (stage {self.name}) is missing a dependency: {e}. "
                    "Run `soundakira doctor` for install hints."
                ) from e

    def teardown(self) -> None:
        for c in self.components:
            c.unload()

    @abstractmethod
    def run(self, ws: SourceWorkspace) -> dict[str, Any]: ...

    def _make(self, kind: str, spec: ComponentSpec) -> Component:
        return registry.create(kind, spec.name, spec.params, self.ctx)


def _load_analysis(ws: SourceWorkspace, sr: int) -> np.ndarray:
    audio, file_sr = read_audio(ws.analysis_audio)
    return resample(audio, file_sr, sr)


def _load_speech(ws: SourceWorkspace) -> list[Span]:
    return [Span(s, e) for s, e in read_json(ws.vad_json)["speech"]]


# ---------------------------------------------------------------------------
class FetchStage(Stage):
    name = "fetch"

    @property
    def workers(self) -> int:
        return self.cfg.fetch.workers

    def params(self) -> dict[str, Any]:
        return {"format": self.cfg.fetch.format}

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.media_json]

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        return ensure_media(ws, self.cfg, force=True)


def ensure_media(ws: SourceWorkspace, cfg: PipelineConfig, force: bool = False) -> dict[str, Any]:
    """Make sure the source media is on disk (re-downloading if it was pruned)."""
    if not force and ws.media_json.exists() and ws.media_path().exists():
        return read_json(ws.media_json)
    src = ws.source
    if src.kind == "local":
        path = Path(src.uri)
        if not path.exists():
            raise FileNotFoundError(f"local source missing: {path}")
        info: dict[str, Any] = {"path": str(path)}
    else:
        from soundakira.sources.fetch import download

        path, meta = download(src.uri, ws.download_dir, cfg.fetch)
        info = {"path": str(path), **meta}
    write_json(ws.media_json, info)
    return info


class ExtractStage(Stage):
    name = "extract"
    requires = ("fetch",)

    @property
    def workers(self) -> int:
        return self.cfg.extract.workers

    def params(self) -> dict[str, Any]:
        e = self.cfg.extract
        return e.model_dump(mode="json", exclude={"workers"})

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.extract_json] + ([ws.source_audio] if self.cfg.storage.keep_source_audio else [])

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        info = extract_source_audio(ws, self.cfg)
        if not self.cfg.storage.keep_download and ws.source.kind == "url":
            shutil.rmtree(ws.download_dir, ignore_errors=True)
        return {"duration": info["duration"], "stream": info["stream"]["index"]}


def extract_source_audio(ws: SourceWorkspace, cfg: PipelineConfig) -> dict[str, Any]:
    e = cfg.extract
    media = Path(ensure_media(ws, cfg)["path"])
    streams = ffmpeg.parse_audio_streams(ffmpeg.probe(media))
    prefer = container_language_tags(ws.source.language) + list(e.prefer_languages)
    stream = ffmpeg.select_audio_stream(streams, prefer, e.skip_title_keywords, e.stream_index)
    ffmpeg.extract_audio(media, ws.source_audio, stream, e.sample_rate, e.channels)
    duration = audio_duration(ws.source_audio)
    if duration < 1.0:
        raise ValueError(f"extracted audio is only {duration:.2f}s long")
    info = {"stream": stream.to_dict(), "num_audio_streams": len(streams), "duration": duration}
    write_json(ws.extract_json, info)
    return info


def ensure_source_audio(ws: SourceWorkspace, cfg: PipelineConfig) -> Path:
    if not ws.source_audio.exists():
        log.info("[%s] source audio was pruned; re-extracting", ws.source_id)
        extract_source_audio(ws, cfg)
    return ws.source_audio


class EnhanceStage(Stage):
    """Vocal separation / denoising chain -> clean.flac, plus 16 kHz analysis audio."""

    name = "enhance"
    requires = ("extract",)

    def build_components(self) -> list[Component]:
        return [self._make("enhancer", spec) for spec in self.cfg.enhance.chain]

    def params(self) -> dict[str, Any]:
        return self.cfg.enhance.model_dump(mode="json", exclude={"chain"})

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.clean_audio, ws.analysis_audio]

    def _process_chunk(self, x: np.ndarray, sr: int) -> np.ndarray:
        length = x.shape[-1]
        for comp in self.components:
            assert isinstance(comp, Enhancer)
            target = comp.sample_rate or sr
            y = np.atleast_2d(comp.process(resample(x, sr, target), target))
            x = fit_length(resample(y, target, sr), length)
        return x

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        src = ensure_source_audio(ws, self.cfg)
        cfg = self.cfg.enhance
        if self.components:
            process_file_in_chunks(
                src, ws.clean_audio, self._process_chunk, cfg.chunk_seconds, cfg.overlap_seconds
            )
        else:
            ffmpeg.convert_audio(src, ws.clean_audio, self.cfg.extract.sample_rate, channels=1)
        analysis_src = ws.clean_audio if cfg.analysis_source == "clean" else src
        ffmpeg.convert_audio(analysis_src, ws.analysis_audio, ANALYSIS_SR, channels=1)
        if not self.cfg.storage.keep_source_audio:
            src.unlink(missing_ok=True)
        return {"chain": [c.name for c in self.components]}


class VadStage(Stage):
    name = "vad"
    requires = ("enhance",)

    def build_components(self) -> list[Component]:
        return [self._make("vad", self.cfg.vad)]

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.vad_json]

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        vad = self.components[0]
        sr = vad.sample_rate  # type: ignore[attr-defined]
        spans = vad.detect(_load_analysis(ws, sr), sr)  # type: ignore[attr-defined]
        speech = sum(s.duration for s in spans)
        write_json(
            ws.vad_json,
            {
                "speech": [[round(s.start, 3), round(s.end, 3)] for s in spans],
                "speech_duration": speech,
            },
        )
        return {"speech_duration": round(speech, 1), "num_regions": len(spans)}


class DiarizeStage(Stage):
    name = "diarize"
    requires = ("enhance",)

    def build_components(self) -> list[Component]:
        d = self.cfg.diarization
        return [self._make("diarizer", ComponentSpec(name=d.name, params=d.params))]

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.diarization_json]

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        diarizer = self.components[0]
        assert isinstance(diarizer, Diarizer)
        turns = diarizer.diarize(_load_analysis(ws, diarizer.sample_rate), diarizer.sample_rate)
        speakers = sorted({t.speaker for t in turns})
        write_json(
            ws.diarization_json,
            {
                "num_speakers": len(speakers),
                "turns": [[round(t.start, 3), round(t.end, 3), t.speaker] for t in turns],
            },
        )
        return {"num_speakers": len(speakers), "num_turns": len(turns)}


class TranscribeStage(Stage):
    name = "transcribe"
    requires = ("enhance", "vad")

    def build_components(self) -> list[Component]:
        a = self.cfg.asr
        comps = [self._make("asr", ComponentSpec(name=a.name, params=a.params))]
        if a.aligner is not None:
            comps.append(self._make("aligner", a.aligner))
        return comps

    def params(self) -> dict[str, Any]:
        return {"language": self.cfg.asr.language}

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.transcript_json]

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        asr = self.components[0]
        assert isinstance(asr, Transcriber)
        audio = _load_analysis(ws, asr.sample_rate)
        language = normalize_language(ws.source.language or self.cfg.asr.language)
        transcript = asr.transcribe(audio, asr.sample_rate, _load_speech(ws), language)
        if len(self.components) > 1:
            aligner = self.components[1]
            assert isinstance(aligner, Aligner)
            transcript = aligner.align(transcript, audio, asr.sample_rate)
        if transcript.segments and not transcript.words:
            raise RuntimeError(
                f"ASR backend {asr.name!r} returned no word timestamps; configure asr.aligner"
            )
        write_json(ws.transcript_json, transcript.to_dict())
        return {"language": transcript.language, "num_words": len(transcript.words)}


class SegmentStage(Stage):
    name = "segment"
    requires = ("vad", "diarize", "transcribe")

    def params(self) -> dict[str, Any]:
        return self.cfg.segmentation.model_dump(mode="json")

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.segments_jsonl]

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        transcript = Transcript.from_dict(read_json(ws.transcript_json))
        turns = [Turn(s, e, spk) for s, e, spk in read_json(ws.diarization_json)["turns"]]
        segments = build_segments(
            ws.source_id,
            transcript,
            turns,
            _load_speech(ws),
            audio_duration(ws.clean_audio),
            SegmentationParams(**self.cfg.segmentation.model_dump()),
        )
        write_jsonl(ws.segments_jsonl, (s.to_dict() for s in segments))
        return {
            "num_segments": len(segments),
            "num_speakers": len({s.speaker for s in segments}),
            "total_duration": round(sum(s.duration for s in segments), 1),
        }


def load_segments(ws: SourceWorkspace) -> list[Segment]:
    return [Segment.from_dict(d) for d in read_jsonl(ws.segments_jsonl)]


def _segment_clip(ws: SourceWorkspace, seg: Segment, sr: int | None) -> tuple[np.ndarray, int]:
    audio, file_sr = read_audio(ws.clean_audio, seg.start, seg.end)
    if sr is not None and sr != file_sr:
        return resample(audio, file_sr, sr), sr
    return audio, file_sr


class ScoreStage(Stage):
    name = "score"
    requires = ("segment",)

    def build_components(self) -> list[Component]:
        return [self._make("scorer", spec) for spec in self.cfg.quality.scorers]

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.scores_json]

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        scores: dict[str, dict[str, float]] = {}
        for seg in load_segments(ws):
            merged: dict[str, float] = {}
            for scorer in self.components:
                assert isinstance(scorer, Scorer)
                audio, sr = _segment_clip(ws, seg, scorer.sample_rate)
                merged.update({k: round(float(v), 5) for k, v in scorer.score(audio, sr).items()})
            scores[seg.segment_id] = merged
        write_json(ws.scores_json, scores)
        return {"num_scored": len(scores)}


class EmbedStage(Stage):
    name = "embed"
    requires = ("segment",)
    batch_size = 16

    def build_components(self) -> list[Component]:
        return [self._make("embedder", self.cfg.speakers.embedder)]

    def outputs(self, ws: SourceWorkspace) -> list[Path]:
        return [ws.embeddings_npz]

    def run(self, ws: SourceWorkspace) -> dict[str, Any]:
        embedder = self.components[0]
        assert isinstance(embedder, SpeakerEmbedder)
        segments = load_segments(ws)
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i : i + self.batch_size]
            clips = [_segment_clip(ws, s, embedder.sample_rate)[0] for s in batch]
            vecs.append(embedder.embed(clips, embedder.sample_rate))
            ids.extend(s.segment_id for s in batch)
        emb = np.concatenate(vecs) if vecs else np.zeros((0, 0), np.float32)
        tmp = ws.embeddings_npz.with_name(".embeddings.tmp.npz")
        with open(tmp, "wb") as f:
            np.savez(f, ids=np.array(ids, dtype=str), embeddings=emb.astype(np.float32))
        os.replace(tmp, ws.embeddings_npz)
        return {"num_embeddings": len(ids), "dim": int(emb.shape[1]) if emb.size else 0}


def load_embeddings(ws: SourceWorkspace) -> dict[str, np.ndarray]:
    with np.load(ws.embeddings_npz) as data:
        return dict(zip(data["ids"].tolist(), data["embeddings"]))


STAGES: list[type[Stage]] = [
    FetchStage,
    ExtractStage,
    EnhanceStage,
    VadStage,
    DiarizeStage,
    TranscribeStage,
    SegmentStage,
    ScoreStage,
    EmbedStage,
]
STAGE_NAMES = [s.name for s in STAGES]
