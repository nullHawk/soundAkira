"""Full pipeline on synthetic sources with fake models (needs ffmpeg)."""

from __future__ import annotations

import csv
import json
import shutil

import pytest
import soundfile as sf

import fakes
from soundakira.config import load_config
from soundakira.dataset.build import build_dataset
from soundakira.pipeline.runner import Runner
from soundakira.sources.resolve import resolve_inputs

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

OVERRIDES = [
    "device=cpu",
    "enhance.chain=[]",
    "vad.name=energy",
    "diarization.name=fake",
    "asr.name=fake",
    "speakers.embedder.name=fake",
    "quality.scorers=[{name: signal}]",
    "speakers.min_local_duration=4",
    "segmentation.min_duration=2",
    "segmentation.max_duration=12",
    "segmentation.preferred_min_duration=5",
    "references.min_duration=2",
    "references.max_duration=4",
    "export.min_duration=5",
    "export.max_duration=12",
    "export.workers=1",
    "fetch.workers=1",
    "extract.workers=1",
]


@pytest.fixture
def corpus(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    fakes.synth_source(
        media / "episode one.wav", [("alice", 16), ("bob", 14), ("alice", 6), ("bob", 6)]
    )
    fakes.synth_source(
        media / "episode two.wav", [("bob", 16), ("alice", 14), ("bob", 6), ("carol", 3)]
    )
    return media


def _config(tmp_path, extra=()):
    return load_config(
        None,
        [
            f"work_dir={tmp_path / 'work'}",
            f"output_dir={tmp_path / 'out'}",
            *OVERRIDES,
            *extra,
        ],
    )


def test_pipeline_end_to_end(tmp_path, corpus):
    cfg = _config(tmp_path)
    sources = resolve_inputs([str(corpus)])
    assert len(sources) == 2

    runner = Runner(cfg)
    summary = runner.run(runner.prepare(sources))
    assert summary.failed == 0, summary.counts
    assert summary.counts["embed"]["done"] == 2

    report = build_dataset(cfg)
    out = tmp_path / "out"
    rows = list(csv.DictReader(open(out / "metadata.csv", encoding="utf-8")))
    assert rows and report.num_utterances == len(rows)

    # Same voices across the two sources must share one global ID; carol has
    # too little speech (< min_local_duration) and is excluded.
    assert report.total_speakers == 2
    by_source: dict[str, set[str]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], set()).add(r["speaker_id"])
    assert len(by_source) == 2
    assert set.union(*by_source.values()) == set.intersection(*by_source.values())

    for r in rows:
        audio, sr = sf.read(out / r["audio_path"])
        assert sr == 24000
        assert 5.0 <= len(audio) / sr <= 12.5
        assert r["ref_audio_path"] and (out / r["ref_audio_path"]).exists()
        assert r["ref_id"] != r["utt_id"]
        # references come from the other source when one is available
        assert r["ref_source_id"] != r["source_id"]
        expected = "2" if r["source_id"].startswith("episode-one") else "3"
        assert r["num_speakers_in_source"] == expected

    speakers = list(csv.DictReader(open(out / "speakers.csv", encoding="utf-8")))
    assert len(speakers) == 2
    for spk in speakers:
        mine = [r for r in rows if r["speaker_id"] == spk["speaker_id"]]
        assert int(spk["num_utterances"]) == len(mine)
        assert int(spk["num_sources"]) == 2

    meta = json.loads((out / "dataset.json").read_text())
    assert meta["total_speakers"] == 2
    assert "hf_token" not in meta["config"]
    first_line = json.loads((out / "metadata.jsonl").read_text().splitlines()[0])
    assert first_line["words"][0]["start"] >= 0

    # Re-running is fully cached and speaker IDs are stable across builds.
    summary2 = runner.run(runner.prepare(sources))
    assert all(c.get("done", 0) == 0 for c in summary2.counts.values())
    build_dataset(cfg)
    rows2 = list(csv.DictReader(open(out / "metadata.csv", encoding="utf-8")))
    assert {(r["utt_id"], r["speaker_id"]) for r in rows} == {
        (r["utt_id"], r["speaker_id"]) for r in rows2
    }


def test_config_change_invalidates_downstream_only(tmp_path, corpus):
    cfg = _config(tmp_path)
    runner = Runner(cfg)
    ws = runner.prepare(resolve_inputs([str(corpus)]))
    runner.run(ws)

    cfg2 = _config(tmp_path, ["segmentation.max_pause=1.0"])
    summary = Runner(cfg2).run(ws)
    assert summary.counts["transcribe"]["cached"] == 2
    assert summary.counts["segment"]["done"] == 2
    assert summary.counts["embed"]["done"] == 2


def test_force_reruns_stage(tmp_path, corpus):
    cfg = _config(tmp_path)
    runner = Runner(cfg)
    ws = runner.prepare(resolve_inputs([str(corpus)]))
    runner.run(ws)
    summary = runner.run(ws, force=["vad"])
    assert summary.counts["vad"]["done"] == 2
    assert summary.counts["diarize"]["cached"] == 2
    assert summary.counts["transcribe"]["done"] == 2


def test_stage_that_cannot_load_fails_cleanly_and_independent_stages_continue(tmp_path, corpus):
    from soundakira.components.base import Diarizer
    from soundakira.registry import register

    @register("diarizer", "gated")
    class Gated(Diarizer):
        def load(self):
            import definitely_not_installed  # noqa: F401

        def diarize(self, audio, sr):
            return []

    cfg = _config(tmp_path, ["diarization.name=gated"])
    runner = Runner(cfg)
    workspaces = runner.prepare(resolve_inputs([str(corpus)]))
    summary = runner.run(workspaces)
    assert summary.counts["diarize"]["failed"] == 2
    assert summary.counts["transcribe"]["done"] == 2  # does not depend on diarize
    assert summary.counts["segment"]["blocked"] == 2  # does
    err = workspaces[0].stage_record("diarize")["error"]
    assert "missing a dependency" in err and "doctor" in err


def test_cluster_diarizer_recovers_speakers(tmp_path):
    from soundakira import registry
    from soundakira.audio.io import read_audio, resample
    from soundakira.components.base import ComponentContext

    truth = fakes.synth_source(
        tmp_path / "s.wav", [("alice", 10), ("bob", 8), ("alice", 6), ("carol", 1)]
    )
    audio, sr = read_audio(tmp_path / "s.wav")
    audio = resample(audio, sr, 16000)
    d = registry.create(
        "diarizer",
        "cluster",
        {"embedder": {"name": "fake"}, "vad": {"name": "energy"}, "min_cluster_duration": 2.0},
        ComponentContext(),
    )
    d.load()
    turns = d.diarize(audio, 16000)
    # alice/bob/alice; carol's single word is folded into a real speaker
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    for t, (_, start, end) in zip(turns, truth):
        assert abs(t.start - start) < 0.2 and abs(t.end - end) < 0.8
