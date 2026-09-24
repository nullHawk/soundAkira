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

    # Real case: "...apps? | Multiple reasons. I'll start..." with the
    # diarization handover between 31.58 (A ends) and 32.26 (B starts).
    ws = [
        Word(" these", 29.9, 30.2),
        Word(" apps?", 30.3, 30.9),
        Word(" Multiple", 31.3, 31.9),
        Word(" reasons.", 32.0, 32.5),
        Word(" I'll", 33.0, 33.2),
    ]
    turns = [Turn(20, 31.58, "A"), Turn(32.26, 40, "B")]
    # Diarization switched after "Multiple": move the change back to "apps?".
    assert snap_speaker_changes(ws, list("AAABB"), turns, 0.5) == list("AABBB")
    # "reasons." is also a sentence end, but outside the handover zone.
    assert snap_speaker_changes(ws, list("AABBB"), turns, 0.5) == list("AABBB")
    # Switched too early (at "apps?", before the zone): move it forward.
    early = [Turn(20, 30.9, "A"), Turn(31.0, 40, "B")]
    assert snap_speaker_changes(ws, list("ABBBB"), early, 0.5) == list("AABBB")
    # No clearly better candidate: keep the original boundary; 0 disables.
    flat = [Word(f" w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(6)]
    flat_turns = [Turn(0, 0.85, "A"), Turn(0.85, 2, "B")]
    assert snap_speaker_changes(flat, list("AAABBB"), flat_turns, 0.5) == list("AAABBB")
    assert snap_speaker_changes(ws, list("AAABB"), turns, 0.0) == list("AAABB")


def test_trim_to_sentences_drops_edge_fragments():
    from soundakira.segmentation.builder import trim_to_sentences

    text = ["And", "I", "think.", "The", "decision", "matters.", "So", "we"]
    ws = [Word(" " + t, i * 0.5, i * 0.5 + 0.4) for i, t in enumerate(text)]
    ws.append(Word(" next.", 5.0, 5.4))
    # Piece starts at "think." (the previous word "I" doesn't end a sentence)
    # and ends at "we" (more speech follows): both fragments are trimmed.
    piece, start_ok, end_ok = trim_to_sentences([2, 3, 4, 5, 6, 7], ws, 4.0)
    assert [ws[i].text.strip() for i in piece] == ["The", "decision", "matters."]
    assert start_ok and end_ok
    # Already clean: unchanged. max_trim=0: unchanged but flagged.
    assert trim_to_sentences([3, 4, 5], ws, 4.0) == ([3, 4, 5], True, True)
    assert trim_to_sentences([2, 3, 4, 5, 6, 7], ws, 0.0) == ([2, 3, 4, 5, 6, 7], False, False)


def test_segments_carry_sentence_flags():
    words = [
        Word(f" w{i}" + ("." if i in (4, 9) else ""), i * 0.5, i * 0.5 + 0.4, 0.9)
        for i in range(10)
    ]
    segs = build_segments(
        "s",
        Transcript("en", 1, words),
        [Turn(0, 6, "A")],
        [],
        6,
        SegmentationParams(min_duration=1),
    )
    assert segs[0].metrics["sentence_start"] == 1.0 and segs[0].metrics["sentence_end"] == 1.0


def test_lowercase_continuation_is_not_a_sentence_start():
    from soundakira.segmentation.builder import trim_to_sentences

    # ASR wrote "Hey... are you following me?": the "..." looks like a sentence
    # end, but the lowercase continuation shows it isn't one.
    text = ["Hey...", "are", "you", "following", "me?", "You", "want", "it."]
    ws = [Word(" " + t, i * 0.5, i * 0.5 + 0.4) for i, t in enumerate(text)]
    piece, clean_start, _ = trim_to_sentences(list(range(1, 8)), ws, 4.0)
    assert ws[piece[0]].text == " You" and clean_start
    # Uncased scripts are unaffected.
    hi = [
        Word(" नमस्ते।", 0, 0.4),
        Word(" आप", 0.5, 0.9),
        Word(" कैसे", 1.0, 1.4),
        Word(" हैं।", 1.5, 1.9),
    ]
    assert trim_to_sentences([1, 2, 3], hi, 4.0) == ([1, 2, 3], True, True)
