from soundakira.segmentation.builder import (
    SegmentationParams,
    assign_speakers,
    build_runs,
    build_segments,
    refine_words_with_vad,
    sanitize_words,
    split_run,
)
from soundakira.types import Span, Transcript, Turn, Word
from soundakira.utils.spans import SpanIndex


def words_at(times, prefix="w", prob=0.9):
    return [Word(f" {prefix}{i}", s, e, prob) for i, (s, e) in enumerate(times)]


def test_sanitize_sorts_and_repairs():
    ws = sanitize_words([Word(" b", 2.0, 1.5), Word(" a", 0.0, 0.5), Word("  ", 1, 2)])
    assert [w.text for w in ws] == [" a", " b"]
    assert ws[1].start == 1.5 and ws[1].end == 2.0


def test_refine_with_vad_trims_leading_silence():
    # Whisper often starts the first word after a pause far too early.
    w = refine_words_with_vad([Word(" hi", 1.0, 3.0)], SpanIndex([Span(2.5, 3.2)]))
    assert (w[0].start, w[0].end) == (2.5, 3.0)


def test_assign_speakers_by_overlap_and_distance():
    turns = [Turn(0, 5, "A"), Turn(5, 10, "B"), Turn(4, 6, "A")]
    ws = [
        Word(" x", 1, 2),
        Word(" y", 5.5, 5.9),
        Word(" z", 7, 8),
        Word(" far", 20, 21),
        Word(" near", 10.1, 10.3),
    ]
    # "y" overlaps A (4-6) and B (5-10) equally; the tie is broken by label order.
    assert assign_speakers(ws, turns, max_distance=0.3) == ["A", "B", "B", None, "B"]


def test_runs_break_on_speaker_overlap_and_pause():
    ws = words_at([(0, 1), (1.1, 2), (2.1, 3), (6, 7), (7.1, 8)])
    speakers = ["A", "A", "B", "B", "B"]
    overlapped = [False, False, False, False, True]
    runs = build_runs(ws, speakers, overlapped, max_pause=1.5)
    assert runs == [("A", [0, 1]), ("B", [2]), ("B", [3])]


def test_events_attach_to_open_run():
    ws = [Word(" hi", 0, 1), Word(" [laugh]", 1.1, 1.5, kind="event"), Word(" there", 1.6, 2)]
    runs = build_runs(ws, ["A", None, "A"], [False] * 3, max_pause=1.0)
    assert runs == [("A", [0, 1, 2])]


def test_split_run_respects_max_and_prefers_sentence_end():
    # 40 words of 0.9 s + 0.1 s gap = 40 s; sentence end after word 19 (t=20 s).
    ws = [Word(f" w{i}" + ("." if i == 19 else ""), i, i + 0.9) for i in range(40)]
    pieces = split_run(ws, list(range(40)), max_duration=30, preferred_min=15)
    assert pieces[0][-1] == 19
    for p in pieces:
        assert ws[p[-1]].end - ws[p[0]].start <= 30
    assert sum(len(p) for p in pieces) == 40


def test_build_segments_padding_never_crosses_other_speech():
    a = words_at([(1.0 + i * 0.5, 1.4 + i * 0.5) for i in range(10)], "a")  # 1.0 - 5.9
    b = words_at([(6.0 + i * 0.5, 6.4 + i * 0.5) for i in range(10)], "b")  # 6.0 - 10.9
    turns = [Turn(0.9, 5.95, "A"), Turn(5.98, 11, "B")]
    t = Transcript("en", 1.0, a + b)
    segs = build_segments("src", t, turns, [], 12.0, SegmentationParams(min_duration=2, pad=0.3))
    assert [s.speaker for s in segs] == ["A", "B"]
    sa, sb = segs
    assert sa.end <= 6.0 - 0.05 + 1e-9  # stopped before B's first word
    assert sb.start >= 5.9  # started after A's last word
    assert sa.start == 0.7  # full pad into leading silence
    assert sa.text.startswith("a0 a1")
    assert sa.metrics["asr_confidence"] == 0.9
    assert sa.segment_id == "src-000000700"


def test_overlapping_speech_is_excluded():
    words = words_at([(i, i + 0.8) for i in range(10)])
    turns = [Turn(0, 10, "A"), Turn(4, 6, "B")]
    segs = build_segments(
        "s", Transcript("en", 1, words), turns, [], 10, SegmentationParams(min_duration=1, pad=0)
    )
    for s in segs:
        assert not (s.start < 6 and s.end > 4), "segment overlaps the two-speaker region"


def test_text_tagged_only_when_events_present():
    words = [
        Word(" hello", 0, 1, 0.9),
        Word(" [laugh]", 1.1, 1.6, kind="event"),
        Word(" world.", 1.7, 3.5, 0.9),
    ]
    segs = build_segments(
        "s",
        Transcript("en", 1, words),
        [Turn(0, 4, "A")],
        [],
        4,
        SegmentationParams(min_duration=1),
    )
    assert segs[0].text == "hello world."
    assert segs[0].text_tagged == "hello [laugh] world."


def test_speaker_change_snaps_to_sentence_boundary():
    from soundakira.segmentation.builder import snap_speaker_changes

    # "...apps? | Multiple reasons." but diarization switched after "Multiple".
    ws = [
        Word(" these", 29.9, 30.2),
        Word(" apps?", 30.3, 30.8),
        Word(" Multiple", 31.3, 31.9),
        Word(" reasons.", 32.0, 32.5),
        Word(" I'll", 33.0, 33.2),
    ]
    assert snap_speaker_changes(ws, ["A", "A", "A", "B", "B"], 1.0) == ["A", "A", "B", "B", "B"]
    # The other direction: diarization switched too early.
    assert snap_speaker_changes(ws, ["A", "B", "B", "B", "B"], 1.0) == ["A", "A", "B", "B", "B"]
    # No clearly better candidate: keep the original boundary.
    flat = [Word(f" w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(6)]
    assert snap_speaker_changes(flat, ["A", "A", "A", "B", "B", "B"], 1.0) == ["A"] * 3 + ["B"] * 3
    assert snap_speaker_changes(ws, ["A", "A", "A", "B", "B"], 0.0) == ["A", "A", "A", "B", "B"]
