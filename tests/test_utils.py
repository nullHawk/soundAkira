import numpy as np
import pytest
import soundfile as sf

from soundakira.audio.chunking import chunk_boundaries, process_file_in_chunks
from soundakira.audio.io import normalize, read_audio, resample
from soundakira.types import Span, Turn
from soundakira.utils.spans import SpanIndex, group_spans, merge_spans, overlap_regions
from soundakira.utils.text import (
    container_language_tags,
    ends_sentence,
    normalize_language,
    repetition_ratio,
    spoken_char_count,
)


def test_merge_and_coverage():
    idx = SpanIndex([Span(0, 1), Span(0.5, 2), Span(3, 4)])
    assert idx.spans == [Span(0, 2), Span(3, 4)]
    assert idx.coverage(1, 3.5) == pytest.approx(1.5)
    assert idx.ratio(3, 4) == 1.0
    assert merge_spans([Span(0, 1), Span(1.2, 2)], max_gap=0.3) == [Span(0, 2)]


def test_overlap_regions_ignore_same_speaker_and_abutting_turns():
    turns = [Turn(0, 5, "A"), Turn(4, 8, "B"), Turn(2, 3, "A"), Turn(8, 9, "C")]
    assert overlap_regions(turns) == [Span(4, 5)]


def test_group_spans():
    chunks = group_spans([Span(0, 10), Span(10.5, 20), Span(30, 100)], max_length=25, max_gap=1)
    assert chunks[0] == Span(0, 20)
    assert all(c.duration <= 25 + 1e-9 for c in chunks)
    assert sum(c.duration for c in chunks) == pytest.approx(20 + 70)


def test_text_helpers_are_script_agnostic():
    assert spoken_char_count("नमस्ते, दोस्त!") == 11  # letters + viramas/matras, no punctuation
    assert ends_sentence("यह ठीक है।")
    assert ends_sentence('He said "stop."')
    assert repetition_ratio("thank you " * 10) > 0.5
    assert repetition_ratio("the quick brown fox jumps over the lazy dog") == 0
    assert normalize_language("Hindi") == "hi"
    assert "hin" in container_language_tags("hi")


def test_chunk_boundaries_never_leave_a_tiny_tail():
    b = chunk_boundaries(total=1050, chunk=500, overlap=40)
    assert b == [0, 500, 1050]
    assert chunk_boundaries(1200, 500, 40) == [0, 500, 1000, 1200]


@pytest.mark.parametrize("seconds", [3.0, 10.7, 25.3])
def test_chunked_processing_identity_is_lossless(tmp_path, seconds):
    sr = 8000
    rng = np.random.default_rng(0)
    x = (0.3 * rng.standard_normal(int(seconds * sr))).astype(np.float32)
    src, dst = tmp_path / "in.flac", tmp_path / "out.flac"
    sf.write(src, x, sr, subtype="PCM_16")
    process_file_in_chunks(src, dst, lambda a, _: a, chunk_seconds=4, overlap_seconds=0.5)
    y, _ = read_audio(dst)
    ref, _ = read_audio(src)
    assert len(y) == len(ref)
    assert np.max(np.abs(y - ref)) < 2e-4


def test_chunked_processing_crossfades_between_chunks(tmp_path):
    # A processor that adds a per-chunk DC offset: output must stay within the
    # offsets' range and change smoothly (no step) at chunk boundaries.
    sr = 1000
    src, dst = tmp_path / "in.flac", tmp_path / "out.flac"
    sf.write(src, np.zeros(10_000, np.float32), sr, subtype="PCM_16")
    calls = []

    def fn(a, _):
        calls.append(len(calls))
        return a + 0.1 * len(calls)

    process_file_in_chunks(src, dst, fn, chunk_seconds=3, overlap_seconds=0.5)
    y, _ = read_audio(dst)
    assert np.max(np.abs(np.diff(y))) < 0.01


def test_resample_and_normalize():
    x = np.sin(np.linspace(0, 100, 44100)).astype(np.float32)
    assert len(resample(x, 44100, 24000)) == 24000
    y = normalize(x * 0.1, "peak", -1.0)
    assert np.max(np.abs(y)) == pytest.approx(10 ** (-1 / 20), rel=1e-3)
