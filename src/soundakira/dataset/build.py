"""Dataset-level build: global speakers -> targets + references -> files.

Runs over every source whose per-source stages are complete, so it can be
re-run at any time (for example after processing more sources, or after
changing export filters) without redoing any model inference except anchor
embedding.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soundakira import __version__, registry
from soundakira.audio.io import read_audio, resample
from soundakira.components.base import ComponentContext, SpeakerEmbedder
from soundakira.config import PipelineConfig
from soundakira.dataset import summary
from soundakira.dataset.export import ClipJob, run_clip_jobs, write_csv
from soundakira.dataset.filters import first_failure
from soundakira.dataset.references import Reference, assign_reference, build_reference_pool
from soundakira.dataset.speakers import (
    Cluster,
    LocalSpeaker,
    SpeakerRegistry,
    apply_anchors,
    build_clusters,
    cluster_centroids,
    intruder_stats,
    l2norm,
    local_centroid,
)
from soundakira.pipeline.stages import load_embeddings, load_segments, load_window_embeddings
from soundakira.pipeline.workspace import SourceWorkspace, list_workspaces, now_iso
from soundakira.sources.resolve import MEDIA_EXTENSIONS
from soundakira.types import Segment
from soundakira.utils.device import resolve_device
from soundakira.utils.io import read_json, write_json, write_jsonl
from soundakira.utils.text import normalize_language

log = logging.getLogger(__name__)

REQUIRED_STAGES = ("segment", "score", "embed")

BASE_COLUMNS = [
    "utt_id",
    "audio_path",
    "text",
    "text_tagged",
    "speaker_id",
    "speaker_name",
    "language",
    "duration",
    "split",
    "source_id",
    "source_uri",
    "source_start",
    "source_end",
    "num_speakers_in_source",
    "ref_id",
    "ref_audio_path",
    "ref_text",
    "ref_duration",
    "ref_source_id",
]


@dataclass
class BuildReport:
    output_dir: str
    num_sources: int
    total_speakers: int
    num_utterances: int
    hours: float
    drop_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_test_speakers(ids: list[int], fraction: float, seed: int) -> set[int]:
    """Hold out whole speakers (unseen-voice evaluation for zero-shot TTS)."""
    if fraction <= 0 or len(ids) < 2:
        return set()
    n = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    ranked = sorted(ids, key=lambda i: hashlib.sha1(f"{seed}:{i}".encode()).hexdigest())
    return set(ranked[:n])


def _load_anchors(cfg: PipelineConfig, ctx: ComponentContext) -> dict[str, np.ndarray]:
    """anchors_dir/<speaker name>/*.wav -> one centroid per name."""
    root = cfg.speakers.anchors_dir
    assert root is not None
    spec = cfg.speakers.embedder
    embedder = registry.create("embedder", spec.name, spec.params, ctx)
    assert isinstance(embedder, SpeakerEmbedder)
    embedder.load()
    anchors: dict[str, np.ndarray] = {}
    try:
        for person in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            clips = []
            for f in sorted(person.iterdir()):
                if f.suffix.lower() in MEDIA_EXTENSIONS:
                    audio, sr = read_audio(f)
                    clips.append(resample(audio, sr, embedder.sample_rate))
            if clips:
                emb = embedder.embed(clips, embedder.sample_rate)
                anchors[person.name] = l2norm(l2norm(emb).mean(0))
    finally:
        embedder.unload()
    log.info("loaded %d speaker anchors", len(anchors))
    return anchors


def build_dataset(
    cfg: PipelineConfig, ctx: ComponentContext | None = None, clean: bool = False
) -> BuildReport:
    ctx = ctx or ComponentContext(
        device=resolve_device(cfg.device), hf_token=cfg.resolved_hf_token(), cache_dir=cfg.cache_dir
    )
    sp, exp, refcfg = cfg.speakers, cfg.export, cfg.references
    drops: Counter[str] = Counter()

    # -- load per-source results ----------------------------------------------
    workspaces = [
        ws for ws in list_workspaces(cfg.work_dir) if all(ws.is_done(s) for s in REQUIRED_STAGES)
    ]
    if not workspaces:
        raise RuntimeError("no fully processed sources in work_dir; run `soundakira process` first")
    ws_by_source: dict[str, SourceWorkspace] = {}
    num_speakers_in_source: dict[str, int] = {}
    segments: list[Segment] = []
    embeddings: dict[str, np.ndarray] = {}
    window_embeddings: dict[str, np.ndarray] = {}
    allowed = {normalize_language(x) for x in exp.languages} if exp.languages else None
    for ws in workspaces:
        ws_by_source[ws.source_id] = ws
        scores = read_json(ws.scores_json)
        embeddings.update(load_embeddings(ws))
        window_embeddings.update(load_window_embeddings(ws))
        num_speakers_in_source[ws.source_id] = read_json(ws.diarization_json)["num_speakers"]
        for seg in load_segments(ws):
            seg.metrics.update(scores.get(seg.segment_id, {}))
            lang = normalize_language(seg.language)
            if allowed and lang not in allowed and (lang or "").split("-")[0] not in allowed:
                drops["language_not_allowed"] += 1
                continue
            segments.append(seg)

    # -- local speakers ---------------------------------------------------------
    groups: dict[tuple[str, str], list[Segment]] = defaultdict(list)
    for seg in segments:
        if seg.segment_id in embeddings:
            groups[(seg.source_id, seg.speaker)].append(seg)
        else:
            drops["no_embedding"] += 1
    locals_: list[LocalSpeaker] = []
    local_key_of: dict[str, str] = {}
    source_centroids: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for (source_id, label), segs in sorted(groups.items()):
        emb = np.stack([embeddings[s.segment_id] for s in segs])
        durations = np.array([s.duration for s in segs])
        centroid, sims = local_centroid(emb, durations, sp.purity_threshold)
        source_centroids[source_id][label] = centroid
        for s, sim in zip(segs, sims):
            s.metrics["speaker_similarity"] = round(float(sim), 5)
        pure = float(durations[sims >= sp.purity_threshold].sum())
        if pure < sp.min_local_duration:
            drops["speaker_insufficient_audio"] += len(segs)
            continue
        spk = LocalSpeaker(source_id, label, centroid, pure)
        locals_.append(spk)
        for s in segs:
            local_key_of[s.segment_id] = spk.key

    # -- global speakers --------------------------------------------------------
    if locals_ and sp.scope == "global":
        labels = cluster_centroids(
            np.stack([ls.centroid for ls in locals_]), sp.cluster_threshold, sp.linkage
        )
    else:
        labels = np.arange(len(locals_))
    clusters: list[Cluster] = build_clusters(locals_, labels)
    if sp.anchors_dir:
        clusters = apply_anchors(clusters, _load_anchors(cfg, ctx), sp.anchor_threshold)
    registry_path = cfg.work_dir / "speakers" / "registry.json"
    if cfg.hub.resolved_repo_id() and cfg.hub.sync_registry:
        from soundakira.hub import pull_registry

        pull_registry(cfg, registry_path)
    reg = SpeakerRegistry.load(registry_path)
    ids = reg.assign(clusters, sp.registry_match_similarity, local_sources=set(ws_by_source))
    reg.save()
    member_to_id = {m: sid for c, sid in zip(clusters, ids) for m in c.members}
    centroid_of = {sid: c.centroid for c, sid in zip(clusters, ids)}
    speaker_of: dict[str, int] = {}
    for seg in segments:
        key = local_key_of.get(seg.segment_id)
        if key is None:
            continue
        sid = member_to_id[key]
        speaker_of[seg.segment_id] = sid
        e = l2norm(embeddings[seg.segment_id].astype(np.float64))
        seg.metrics["global_speaker_similarity"] = round(float(e @ centroid_of[sid]), 5)
    # Second-voice detection: windows closer to *another person* in the same
    # source. Other labels that clustering merged into the same global speaker
    # (diarization splitting one person) don't count as other people.
    for (source_id, label), segs in groups.items():
        own_id = member_to_id.get(f"{source_id}:{label}")
        others = [
            c
            for lab, c in source_centroids[source_id].items()
            if lab != label and (own_id is None or member_to_id.get(f"{source_id}:{lab}") != own_id)
        ]
        own = source_centroids[source_id][label]
        for s in segs:
            if s.segment_id in window_embeddings:
                secs, worst = intruder_stats(
                    window_embeddings[s.segment_id],
                    own,
                    others,
                    sp.intruder_margin,
                    sp.intruder_hop,
                )
                s.metrics["intruder_s"] = round(secs, 3)
                s.metrics["speaker_margin_min"] = round(worst, 5)
    eligible = [s for s in segments if s.segment_id in speaker_of]

    # -- targets + references ---------------------------------------------------
    pool, ref_rejects = build_reference_pool(eligible, speaker_of, refcfg)
    targets: list[tuple[Segment, Reference | None]] = []
    for seg in eligible:
        if seg.duration < exp.min_duration:
            drops["shorter_than_target (reference pool only)"] += 1
            continue
        if seg.duration > exp.max_duration:
            drops["longer_than_target"] += 1
            continue
        reason = first_failure(seg.metrics, exp.filters)
        if reason:
            drops[f"filter:{reason}"] += 1
            continue
        ref = assign_reference(seg, speaker_of[seg.segment_id], pool, refcfg)
        if ref is None and refcfg.require:
            drops["no_reference"] += 1
            continue
        targets.append((seg, ref))

    speakers_in_data = sorted({speaker_of[s.segment_id] for s, _ in targets})
    test_speakers = choose_test_speakers(
        speakers_in_data, exp.split.test_speaker_fraction, exp.split.seed
    )

    # -- export -----------------------------------------------------------------
    out = cfg.output_dir
    if clean:
        for sub in ("wavs", "refs"):
            shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    ext = exp.format

    def job(ws: SourceWorkspace, start: float, end: float, rel: str) -> ClipJob:
        return ClipJob(
            str(ws.clean_audio),
            start,
            end,
            str(out / rel),
            exp.sample_rate,
            exp.fade_ms,
            exp.normalize.mode,
            exp.normalize.target_db,
        )

    jobs: list[ClipJob] = []
    exported_refs: set[str] = set()
    rows: list[dict[str, Any]] = []
    metric_keys: set[str] = set()
    for seg, ref in sorted(targets, key=lambda t: (speaker_of[t[0].segment_id], t[0].segment_id)):
        sid = speaker_of[seg.segment_id]
        ws = ws_by_source[seg.source_id]
        rel = f"wavs/{sid:05d}/{seg.segment_id}.{ext}"
        jobs.append(job(ws, seg.start, seg.end, rel))
        ref_rel = None
        if ref is not None:
            ref_rel = f"refs/{sid:05d}/{ref.ref_id}.{ext}"
            if ref.ref_id not in exported_refs:
                exported_refs.add(ref.ref_id)
                jobs.append(job(ws_by_source[ref.source_id], ref.start, ref.end, ref_rel))
        metrics = {k: v for k, v in seg.metrics.items() if k != "duration"}
        metric_keys.update(metrics)
        rows.append(
            {
                "utt_id": seg.segment_id,
                "audio_path": rel,
                "text": seg.text,
                "text_tagged": seg.text_tagged,
                "speaker_id": sid,
                "speaker_name": reg.speakers[sid].get("name"),
                "language": seg.language,
                "duration": round(seg.duration, 3),
                "split": "test" if sid in test_speakers else "train",
                "source_id": seg.source_id,
                "source_uri": ws.source.uri,
                "source_start": seg.start,
                "source_end": seg.end,
                "num_speakers_in_source": num_speakers_in_source[seg.source_id],
                "ref_id": ref.ref_id if ref else None,
                "ref_audio_path": ref_rel,
                "ref_text": ref.text if ref else None,
                "ref_duration": round(ref.duration, 3) if ref else None,
                "ref_source_id": ref.source_id if ref else None,
                **metrics,
                "words": [
                    {
                        "text": w.text.strip(),
                        "start": round(w.start - seg.start, 3),
                        "end": round(w.end - seg.start, 3),
                        "kind": w.kind,
                    }
                    for w in seg.words
                ],
            }
        )

    log.info(
        "exporting %d clips (%d targets, %d references)", len(jobs), len(rows), len(exported_refs)
    )
    run_clip_jobs(jobs, exp.workers)

    columns = BASE_COLUMNS + sorted(metric_keys)
    write_csv(out / "metadata.csv", rows, columns)
    if exp.write_jsonl:
        write_jsonl(out / "metadata.jsonl", rows)

    speakers = summary.speaker_rows(rows)
    summary.write_speakers_csv(out / "speakers.csv", speakers)
    report = BuildReport(
        str(out),
        len(workspaces),
        len(speakers_in_data),
        len(rows),
        round(sum(r["duration"] for r in rows) / 3600, 3),
        dict(sorted(drops.items())),
    )
    write_json(
        out / "dataset.json",
        {
            "created_at": now_iso(),
            "soundakira_version": __version__,
            **summary.totals(rows, speakers),
            "source_ids": sorted(ws_by_source),
            "sample_rate": exp.sample_rate,
            "drop_reasons": report.drop_reasons,
            "reference_candidates_rejected": ref_rejects,
            "config": cfg.model_dump(mode="json", exclude={"hf_token"}),
        },
    )
    return report
