import numpy as np

from soundakira.components.aligners.ctc import ctc_align


def _emission(path: list[int], vocab: int, sharp: float = 8.0) -> np.ndarray:
    """Log-probs where each frame strongly prefers the given token."""
    e = np.full((len(path), vocab), -sharp)
    for t, tok in enumerate(path):
        e[t, tok] = 0.0
    return e - np.log(np.exp(e).sum(1, keepdims=True))


def test_ctc_align_recovers_token_frames():
    blank = 0
    # frames: _ a a _ b _ _ c c _   (tokens a=1, b=2, c=3)
    em = _emission([0, 1, 1, 0, 2, 0, 0, 3, 3, 0], vocab=4)
    frames = ctc_align(em, [1, 2, 3], blank)
    assert frames is not None
    assert frames[0] in (1, 2) and frames[1] == 4 and frames[2] in (7, 8)


def test_ctc_align_rejects_text_longer_than_audio():
    em = _emission([0, 1], vocab=3)
    assert ctc_align(em, [1, 2, 1], 0) is None
    assert ctc_align(em, [], 0) is None
