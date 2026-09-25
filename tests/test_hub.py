"""Hub publishing: merge, incremental upload, stable speaker IDs across machines.

Uses an in-memory Hub and the fake-model pipeline, so no network is needed.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

import fakes
from soundakira import hub
from soundakira.config import load_config
from soundakira.dataset.build import build_dataset
from soundakira.pipeline.runner import Runner
from soundakira.sources.resolve import resolve_inputs
from test_e2e import OVERRIDES


class MemoryHub:
    """In-memory stand-in for a Hub dataset repo (with parent-commit checks)."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.sha: str | None = None
        self.commits = 0
        self.private: bool | None = None
        self.ensure_calls = 0

    def ensure_repo(self, private: bool) -> None:
        self.ensure_calls += 1
        if self.private is None:
            self.private = private

    def head(self) -> str | None:
        return self.sha

    def read(self, path: str, revision: str | None) -> bytes | None:
        assert revision == self.sha, "reads must be pinned to the current head"
        return self.files.get(path)

    def commit(self, adds, deletes, message, parent):
        if parent != self.sha:
            raise RuntimeError("concurrent modification")
        for p, v in adds.items():
            self.files[p] = v.read_bytes() if isinstance(v, Path) else v
        for p in deletes:
            del self.files[p]
        self.commits += 1
        self.sha = f"c{self.commits}"
        return self.sha

    def rows(self) -> list[dict]:
        return [json.loads(x) for x in self.files["metadata.jsonl"].decode().splitlines()]


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def remote(monkeypatch):
    mem = MemoryHub()
    monkeypatch.setattr(hub, "backend_for", lambda cfg: mem)
    monkeypatch.setenv("SOUNDAKIRA_HF_REPO", "me/test-dataset")
    return mem


def _machine(tmp_path: Path, name: str, script, extra=()):
    media = tmp_path / f"{name}_media"
    media.mkdir()
    fakes.synth_source(media / f"{name}.wav", script)
    cfg = load_config(
        None,
        [
            f"work_dir={tmp_path / name / 'work'}",
            f"output_dir={tmp_path / name / 'out'}",
            *OVERRIDES,
            *extra,
        ],
    )
    runner = Runner(cfg)
    ws = runner.prepare(resolve_inputs([str(media)]))
    assert runner.run(ws).failed == 0
    return cfg, ws


def test_two_machines_share_speaker_ids_and_merge(tmp_path, remote):
    # Machine A: episode with alice + bob. Push.
    cfg_a, _ = _machine(tmp_path, "epone", [("alice", 16), ("bob", 14), ("alice", 6), ("bob", 6)])
    build_dataset(cfg_a)
    rep_a = hub.push(cfg_a)
    assert remote.private is True and rep_a.uploaded > 0
    rows_a = remote.rows()
    ids_a = {int(r["speaker_id"]) for r in rows_a}
    assert len(ids_a) == 2

    # Machine B (fresh work dir, no local registry): another episode with the
    # same two voices plus a new one. build() pulls the Hub registry first.
    cfg_b, _ = _machine(
        tmp_path,
        "eptwo",
        [("bob", 16), ("alice", 14), ("carol", 16), ("bob", 6), ("alice", 6), ("carol", 6)],
    )
    build_dataset(cfg_b)
    before = dict(remote.files)
    rep_b = hub.push(cfg_b)
    rows = remote.rows()
    by_source: dict[str, set[int]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"].split("-")[0], set()).add(int(r["speaker_id"]))
    assert by_source["epone"] == ids_a  # machine A's rows kept, same IDs
    assert ids_a < by_source["eptwo"]  # alice/bob reused their IDs...
    assert len(by_source["eptwo"] - ids_a) == 1  # ...carol got exactly one new ID
    # Incremental: machine A's shard was not rewritten or re-uploaded.
    shard_a = hub.shard_path(rows_a[0]["source_id"])
    assert remote.files[shard_a] == before[shard_a]
    assert remote.ensure_calls == 1  # repo created once, never re-created
    assert (
        rep_b.total_utterances
        == len(rows)
        == len(rows_a) + len([r for r in rows if r["source_id"].startswith("eptwo")])
    )
    speakers = list(csv.DictReader(io.StringIO(remote.files["speakers.csv"].decode())))
    assert len(speakers) == 3
    manifest = json.loads(remote.files["manifest.json"])
    assert len(manifest["history"]) == 2
    assert set(manifest["files"]) == {hub.shard_path(r["source_id"]) for r in rows}
    assert json.loads(remote.files["dataset.json"])["total_speakers"] == 3
    assert not any(str(r["source_uri"]).startswith("/") for r in rows)  # no local paths
    assert str(tmp_path) not in remote.files["metadata.csv"].decode()
    readme = remote.files["README.md"].decode()
    assert "data/*.parquet" in readme and "soundakira" not in readme.lower()

    # Re-pushing the same build uploads nothing.
    assert hub.push(cfg_b).uploaded == 0


def test_rebuilding_a_source_rewrites_only_its_shard(tmp_path, remote):
    cfg, _ = _machine(tmp_path, "ep", [("alice", 16), ("bob", 14), ("alice", 6), ("bob", 6)])
    build_dataset(cfg)
    hub.push(cfg)
    n_before = len(remote.rows())
    shard = hub.shard_path(remote.rows()[0]["source_id"])
    old = remote.files[shard]
    # Stricter export: keeps the ~8.2 s clip, drops the ~7.2 s one.
    cfg2 = cfg.model_copy(update={"export": cfg.export.model_copy(update={"min_duration": 7.6})})
    build_dataset(cfg2, clean=True)
    rep = hub.push(cfg2)
    assert 0 < len(remote.rows()) < n_before
    assert rep.uploaded == 1 and remote.files[shard] != old


def test_files_layout_deletes_stale_audio(tmp_path, remote):
    cfg, _ = _machine(
        tmp_path,
        "ep",
        [("alice", 16), ("bob", 14), ("alice", 6), ("bob", 6)],
        ["hub.audio_layout=files"],
    )
    build_dataset(cfg)
    hub.push(cfg)
    n_before = len(remote.rows())
    cfg2 = cfg.model_copy(update={"export": cfg.export.model_copy(update={"min_duration": 7.6})})
    build_dataset(cfg2, clean=True)
    rep = hub.push(cfg2)
    rows = remote.rows()
    assert 0 < len(rows) < n_before and rep.deleted > 0
    assert set(json.loads(remote.files["manifest.json"])["files"]) == hub.referenced_files(rows)
    assert all(p in remote.files for p in hub.referenced_files(rows))


def test_parquet_shard_has_playable_audio_columns(tmp_path, remote):
    import pyarrow.parquet as pq
    import soundfile as sf

    cfg, _ = _machine(tmp_path, "ep", [("alice", 16), ("bob", 14), ("alice", 6), ("bob", 6)])
    build_dataset(cfg)
    hub.push(cfg)
    shard = hub.shard_path(remote.rows()[0]["source_id"])
    path = tmp_path / "shard.parquet"
    path.write_bytes(remote.files[shard])
    table = pq.read_table(path)
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    assert features["audio"]["_type"] == "Audio" and features["ref_audio"]["_type"] == "Audio"
    assert features["audio"]["sampling_rate"] == 24000
    first = table.column("audio")[0].as_py()
    audio, sr = sf.read(io.BytesIO(first["bytes"]))
    assert sr == 24000 and len(audio) / sr > 5
    assert table.column("ref_audio")[0].as_py()["bytes"]
    assert set(json.loads(table.column("metrics")[0].as_py())) >= {"asr_confidence", "intruder_s"}


def test_push_refuses_a_build_not_based_on_the_hub_registry(tmp_path, remote):
    cfg_a, _ = _machine(tmp_path, "epone", [("alice", 16), ("bob", 14), ("alice", 6), ("bob", 6)])
    build_dataset(cfg_a)
    hub.push(cfg_a)
    cfg_b, _ = _machine(
        tmp_path,
        "eptwo",
        [("carol", 16), ("bob", 14), ("carol", 6), ("bob", 6)],
        ["hub.sync_registry=false"],
    )
    build_dataset(cfg_b)  # IDs assigned from scratch -> would collide
    with pytest.raises(hub.HubError, match="speaker registry"):
        hub.push(cfg_b)


def test_plan_files_and_merge_are_pure():
    remote = [
        {
            "source_id": "a",
            "speaker_id": 0,
            "utt_id": "a1",
            "audio_path": "w/a1",
            "ref_audio_path": "r/x",
        },
        {
            "source_id": "b",
            "speaker_id": 1,
            "utt_id": "b1",
            "audio_path": "w/b1",
            "ref_audio_path": None,
        },
    ]
    local = [
        {
            "source_id": "a",
            "speaker_id": 0,
            "utt_id": "a2",
            "audio_path": "w/a2",
            "ref_audio_path": "r/x",
        }
    ]
    merged = hub.merge_rows(remote, local, {"a"})
    assert [r["utt_id"] for r in merged] == ["a2", "b1"]
    up, delete, manifest = hub.plan_files(
        merged, {"w/a2": "h2", "r/x": "hx"}, {"w/a1": "h1", "w/b1": "hb", "r/x": "hx"}
    )
    assert up == ["w/a2"] and delete == ["w/a1"]
    assert manifest == {"r/x": "hx", "w/a2": "h2", "w/b1": "hb"}
    with pytest.raises(hub.HubError, match="neither locally nor"):
        hub.plan_files([{"audio_path": "w/zz"}], {}, {})
