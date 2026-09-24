"""Model-free signal statistics: level, clipping and a crude SNR estimate."""

from __future__ import annotations

import numpy as np

from soundakira.components.base import Scorer


def _db(x: float) -> float:
    return float(10 * np.log10(max(x, 1e-12)))


class SignalScorer(Scorer):
    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        if audio.size == 0:
            return {}
        peak = float(np.max(np.abs(audio)))
        hop = max(1, int(sr * 0.02))
        n = len(audio) // hop
        snr = 0.0
        if n >= 10:
            energy = np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)
            # Loud frames vs quiet frames: speech-to-floor ratio. It is only a
            # relative ranking signal, not a calibrated SNR.
            snr = min(100.0, _db(float(np.percentile(energy, 90))) - _db(float(np.percentile(energy, 10))))
        return {
            "rms_dbfs": _db(float(np.mean(audio**2))),
            "peak_dbfs": 20 * float(np.log10(max(peak, 1e-6))),
            "clip_ratio": float(np.mean(np.abs(audio) >= 0.999)),
            "snr_est_db": snr,
        }
