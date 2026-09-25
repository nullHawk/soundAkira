"""Typed pipeline configuration.

The YAML file is validated against these models, so typos fail loudly at
startup instead of three hours into a run. The packaged
`resources/default.yaml` mirrors these defaults (a test keeps them in sync).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(_Strict):
    """Selects a registered component by name and passes it typed params."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class FilterRule(_Strict):
    """Keep a segment only if `min <= metrics[field] <= max`."""

    field: str
    min: float | None = None
    max: float | None = None
    on_missing: Literal["keep", "drop"] = "keep"


class FetchConfig(_Strict):
    format: str = "bestaudio/best"
    workers: int = 4
    cookies_file: Path | None = None
    expand_playlists: bool = True
    yt_dlp_options: dict[str, Any] = Field(default_factory=dict)


class ExtractConfig(_Strict):
    sample_rate: int = 44100
    channels: Literal["stereo", "mono", "center"] = "stereo"
    stream_index: int | None = None
    prefer_languages: list[str] = Field(default_factory=lambda: ["eng", "en"])
    skip_title_keywords: list[str] = Field(
        default_factory=lambda: ["commentary", "description", "descriptive"]
    )
    workers: int = 4


class EnhanceConfig(_Strict):
    chain: list[ComponentSpec] = Field(
        default_factory=lambda: [
            ComponentSpec(name="roformer", params={"model": "vocals_mel_band_roformer.ckpt"})
        ]
    )
    chunk_seconds: float = 300.0
    overlap_seconds: float = 2.0
    analysis_source: Literal["clean", "original"] = "clean"


class DiarizationConfig(_Strict):
    name: str = "pyannote"
    params: dict[str, Any] = Field(default_factory=dict)


class AsrConfig(_Strict):
    name: str = "faster_whisper"
    params: dict[str, Any] = Field(default_factory=dict)
    language: str | None = None
    aligner: ComponentSpec | None = None


class SegmentationConfig(_Strict):
    min_duration: float = 3.0
    max_duration: float = 30.0
    preferred_min_duration: float = 15.0
    max_pause: float = 1.5
    pad: float = 0.15
    boundary_margin: float = 0.05
    speaker_max_distance: float = 0.3
    overlap_word_fraction: float = 0.3
    refine_with_vad: bool = True
    snap_speaker_changes: float = 0.5
    trim_to_sentence: float = 4.0


class QualityConfig(_Strict):
    scorers: list[ComponentSpec] = Field(
        default_factory=lambda: [ComponentSpec(name="signal"), ComponentSpec(name="audioset")]
    )


class SpeakerConfig(_Strict):
    scope: Literal["global", "source"] = "global"
    embedder: ComponentSpec = Field(default_factory=lambda: ComponentSpec(name="pyannote"))
    cluster_threshold: float = 0.45
    linkage: Literal["average", "complete", "single"] = "average"
    purity_threshold: float = 0.45
    min_local_duration: float = 20.0
    registry_match_similarity: float = 0.6
    intruder_window: float = 1.5
    intruder_hop: float = 0.75
    intruder_margin: float = 0.1
    anchors_dir: Path | None = None
    anchor_threshold: float = 0.55


def _default_target_filters() -> list[FilterRule]:
    return [
        FilterRule(field="asr_confidence", min=0.55),
        FilterRule(field="chars_per_sec", min=4.0, max=25.0),
        FilterRule(field="repetition_ratio", max=0.4),
        FilterRule(field="speech_ratio", min=0.5),
        FilterRule(field="overlap_ratio", max=0.1),
        FilterRule(field="speaker_similarity", min=0.45),
        FilterRule(field="clip_ratio", max=0.001),
        FilterRule(field="speech_prob", min=0.15),
        FilterRule(field="intruder_s", max=0.0),
        FilterRule(field="other_speaker_s", max=0.3),
        FilterRule(field="singing_prob", max=0.5),
    ]


def _default_reference_filters() -> list[FilterRule]:
    return [
        FilterRule(field="asr_confidence", min=0.7),
        FilterRule(field="chars_per_sec", min=4.0, max=25.0),
        FilterRule(field="speech_ratio", min=0.6),
        FilterRule(field="overlap_ratio", max=0.02),
        FilterRule(field="speaker_similarity", min=0.6),
        FilterRule(field="clip_ratio", max=0.001),
        FilterRule(field="speech_prob", min=0.15),
        FilterRule(field="intruder_s", max=0.0),
        FilterRule(field="other_speaker_s", max=0.3),
        FilterRule(field="singing_prob", max=0.5),
    ]


class ReferenceConfig(_Strict):
    min_duration: float = 4.0
    max_duration: float = 12.0
    derive_from_long_segments: bool = True
    prefer_other_source: bool = True
    rank_by: list[str] = Field(default_factory=lambda: ["speaker_similarity", "asr_confidence"])
    top_k: int = 5
    require: bool = True
    filters: list[FilterRule] = Field(default_factory=_default_reference_filters)


class NormalizeConfig(_Strict):
    mode: Literal["none", "peak", "rms"] = "peak"
    target_db: float = -1.0


class SplitConfig(_Strict):
    test_speaker_fraction: float = 0.0
    seed: int = 0


class ExportConfig(_Strict):
    sample_rate: int = 24000
    format: Literal["wav", "flac"] = "wav"
    min_duration: float = 15.0
    max_duration: float = 30.0
    languages: list[str] | None = None
    fade_ms: float = 10.0
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    filters: list[FilterRule] = Field(default_factory=_default_target_filters)
    split: SplitConfig = Field(default_factory=SplitConfig)
    workers: int = 4
    write_jsonl: bool = True


class HubConfig(_Strict):
    """Hugging Face Hub dataset repo that `soundakira push` maintains."""

    repo_id: str | None = None  # falls back to $SOUNDAKIRA_HF_REPO
    # Token for the account that owns the dataset repo. Falls back to
    # $SOUNDAKIRA_HF_TOKEN, then to the model-download token (hf_token / $HF_TOKEN /
    # saved login). Keep it separate when publishing under another account.
    token: str | None = None
    private: bool = True
    branch: str = "main"
    sync_registry: bool = True  # `build` pulls speaker_registry.json from the Hub first
    # parquet: one shard per source with Audio columns (Hub viewer players,
    # `load_dataset` decoding). files: loose wav files + metadata.csv.
    audio_layout: Literal["parquet", "files"] = "parquet"
    commit_batch_size: int = 500

    def resolved_repo_id(self) -> str | None:
        return self.repo_id or os.environ.get("SOUNDAKIRA_HF_REPO") or None

    def resolved_token(self, fallback: str | None) -> str | None:
        return self.token or os.environ.get("SOUNDAKIRA_HF_TOKEN") or fallback


class StorageConfig(_Strict):
    keep_download: bool = True
    keep_source_audio: bool = True


class PipelineConfig(_Strict):
    work_dir: Path = Path("work")
    output_dir: Path = Path("dataset")
    device: str = "auto"
    cache_dir: Path | None = (
        None  # model caches (converted ASR models, ...); default ~/.cache/soundakira
    )
    hf_token: str | None = None
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    enhance: EnhanceConfig = Field(default_factory=EnhanceConfig)
    vad: ComponentSpec = Field(default_factory=lambda: ComponentSpec(name="silero"))
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    speakers: SpeakerConfig = Field(default_factory=SpeakerConfig)
    references: ReferenceConfig = Field(default_factory=ReferenceConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    hub: HubConfig = Field(default_factory=HubConfig)

    @model_validator(mode="after")
    def _check(self) -> PipelineConfig:
        seg, exp, ref = self.segmentation, self.export, self.references
        if exp.max_duration > seg.max_duration:
            raise ValueError("export.max_duration cannot exceed segmentation.max_duration")
        if ref.min_duration < seg.min_duration and not ref.derive_from_long_segments:
            raise ValueError("references.min_duration is below segmentation.min_duration")
        if exp.min_duration > exp.max_duration or ref.min_duration > ref.max_duration:
            raise ValueError("min_duration must be <= max_duration")
        return self

    def resolved_hf_token(self) -> str | None:
        return self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(obj: Any) -> Any:
    """Expand ${VAR} / ${VAR:-default} in string values."""
    if isinstance(obj, str):
        if not _ENV.search(obj):
            return obj
        # An unset variable with no default becomes null (e.g. hf_token: ${HF_TOKEN}).
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj) or None
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
        if not isinstance(d, dict):
            raise ValueError(f"cannot set {dotted!r}: {k!r} is not a mapping")
    d[keys[-1]] = value


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> PipelineConfig:
    """Load YAML (optional) then apply `key.path=value` overrides (values parsed as YAML)."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return PipelineConfig.model_validate(_expand_env(raw))


def default_config_text() -> str:
    return (Path(__file__).parent / "resources" / "default.yaml").read_text(encoding="utf-8")
