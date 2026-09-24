import numpy as np

from soundakira.config import FilterRule, ReferenceConfig
from soundakira.dataset.build import choose_test_speakers
from soundakira.dataset.filters import first_failure
from soundakira.dataset.references import (
    assign_reference,
    build_reference_pool,
    derive_excerpt,
)
from soundakira.dataset.speakers import (
    Cluster,
    SpeakerRegistry,
    apply_anchors,
    cluster_centroids,
    local_centroid,
)
from soundakira.types import Segment, Word


def unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def seg(sid, source, start, end, text="hello there.", metrics=None, words=None):
    return Segment(
        sid,
        source,
        "S0",
        start,
        end,
        text,
        None,
        "en",
        words or [],
        metrics or {"asr_confidence": 0.9, "speaker_similarity": 0.9},
    )


def test_filters():
    rules = [FilterRule(field="a", min=1), FilterRule(field="b", max=2, on_missing="drop")]
    assert first_failure({"a": 1, "b": 2}, rules) is None
    assert first_failure({"a": 0.5, "b": 1}, rules) == "a_below_1"
    assert first_failure({"a": 3}, rules) == "missing_b"
    assert first_failure({"b": 1}, rules) is None  # missing a -> keep


def test_local_centroid_downweights_outliers():
    emb = np.stack([unit(1, 0.1), unit(1, -0.1), unit(1, 0), unit(0, 1)])
    # The mislabelled segment is long (10 s) but still the minority.
    centroid, sims = local_centroid(emb, np.array([5, 5, 5, 10.0]), purity_threshold=0.5)
    assert centroid[0] > 0.99  # the long outlier did not drag the centroid
    assert sims[3] < 0.2


def test_clustering_merges_same_voice():
    c = np.stack([unit(1, 0, 0), unit(0.98, 0.05, 0), unit(0, 1, 0), unit(0, 0.97, 0.1)])
    labels = cluster_centroids(c, threshold=0.3)
    assert labels[0] == labels[1] != labels[2] == labels[3]


def test_registry_ids_are_stable_as_speakers_are_added(tmp_path):
    path = tmp_path / "registry.json"
    reg = SpeakerRegistry.load(path)
    first = reg.assign(
        [Cluster(unit(1, 0, 0), ["s1:A"], 10), Cluster(unit(0, 1, 0), ["s1:B"], 10)], 0.6
    )
    reg.speakers[first[1]]["name"] = "Narrator"  # a user edit
    reg.save()

    reg2 = SpeakerRegistry.load(path)
    # New build: a new speaker, and the old ones with extra members (in a different order).
    second = reg2.assign(
        [
            Cluster(unit(0, 0, 1), ["s2:C"], 10),
            Cluster(unit(0.05, 1, 0), ["s1:B", "s2:A"], 20),
            Cluster(unit(1, 0.02, 0), ["s1:A"], 10),
        ],
        0.6,
    )
    assert second[1] == first[1] and second[2] == first[0]
    assert second[0] not in first
    assert reg2.speakers[second[1]]["name"] == "Narrator"


def test_anchors_name_and_merge_clusters():
    clusters = [
        Cluster(unit(1, 0.1), ["a:0"], 5),
        Cluster(unit(1, -0.1), ["b:0"], 5),
        Cluster(unit(0, 1), ["c:0"], 5),
    ]
    out = apply_anchors(clusters, {"Priya": unit(1, 0)}, threshold=0.8)
    named = [c for c in out if c.name == "Priya"]
    assert len(out) == 2 and len(named) == 1
    assert named[0].members == ["a:0", "b:0"]


def test_derive_excerpt_prefers_sentence_end():
    words = [
        Word(f" w{i}" + ("." if i == 5 else ""), i * 1.0, i * 1.0 + 0.8, 0.9) for i in range(30)
    ]
    s = seg("x", "src", 0.0, 30.0, words=words)
    start, end, prefix = derive_excerpt(s, 4, 12)
    assert start == 0.0 and prefix[-1].text == " w5." and 5.8 <= end <= 5.93


def test_derive_excerpt_skips_leading_fragment():
    # Segment starts mid-sentence ("...on it.") -> the prompt starts at "Then".
    text = ["on", "it.", "Then", "we", "left", "the", "house", "early.", "And", "then"]
    words = [Word(" " + t, i * 1.0, i * 1.0 + 0.8, 0.9) for i, t in enumerate(text)]
    s = seg(
        "x",
        "src",
        0.0,
        10.0,
        words=words,
        metrics={"asr_confidence": 0.9, "speaker_similarity": 0.9, "sentence_start": 0.0},
    )
    start, _end, excerpt = derive_excerpt(s, 4, 8)
    assert excerpt[0].text == " Then" and excerpt[-1].text == " early."
    assert 1.8 < start < 2.0


def test_reference_assignment_avoids_leakage_and_prefers_other_source():
    cfg = ReferenceConfig(min_duration=2, max_duration=5, filters=[])
    target = seg("t", "src1", 10, 30)
    same_src_overlap = seg("o", "src1", 12, 15)
    same_src_clean = seg("c", "src1", 40, 44)
    other_src = seg("x", "src2", 0, 4)
    segs = [target, same_src_overlap, same_src_clean, other_src]
    pool, _ = build_reference_pool(segs, {s.segment_id: 7 for s in segs}, cfg)
    assert assign_reference(target, 7, pool, cfg).ref_id == "x"

    cfg_same = ReferenceConfig(
        min_duration=2, max_duration=5, filters=[], prefer_other_source=False, top_k=1
    )
    pool, _ = build_reference_pool(
        [target, same_src_overlap, same_src_clean], {"t": 7, "o": 7, "c": 7}, cfg_same
    )
    assert assign_reference(target, 7, pool, cfg_same).ref_id == "c"  # never the overlapping one
    assert assign_reference(target, 99, pool, cfg_same) is None


def test_test_split_is_deterministic_and_speaker_disjoint():
    ids = list(range(100))
    a = choose_test_speakers(ids, 0.1, seed=1)
    assert a == choose_test_speakers(ids, 0.1, seed=1)
    assert len(a) == 10
    assert choose_test_speakers([1], 0.5, 0) == set()
