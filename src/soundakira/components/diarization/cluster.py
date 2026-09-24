"""Diarization by clustering sliding-window speaker embeddings.

A fallback that needs no gated segmentation model. It uses any registered VAD
and speaker embedder:

1. VAD speech regions are covered with overlapping windows (1.5 s, 0.75 s hop).
2. Each window is embedded, and the windows are clustered agglomeratively by
   cosine distance (or into exactly `num_speakers` clusters when given).
3. Labels are mode-smoothed within each region, and tiny clusters (usually
   laughter, noise or crosstalk) are folded into the nearest real speaker.
4. Each window owns the time up to the midpoints with its neighbours;
   consecutive windows with the same label merge into turns.

Compared with pyannote it does not detect overlapped speech, and its speaker
boundaries are only accurate to about half a hop. Segmentation still cuts on
word boundaries, and per-segment `speaker_similarity` filters catch errors.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from soundakira import registry
from soundakira.audio.io import resample
from soundakira.components.base import VAD, Diarizer, SpeakerEmbedder
from soundakira.config import ComponentSpec
from soundakira.types import Span, Turn


class ClusterDiarizerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedder: ComponentSpec = Field(default_factory=lambda: ComponentSpec(name="pyannote"))
    vad: ComponentSpec = Field(default_factory=lambda: ComponentSpec(name="silero"))
    window: float = 1.5
    step: float = 0.75
    threshold: float = 0.7  # cosine distance cut (average linkage) when num_speakers is unknown
    num_speakers: int | None = None
    min_cluster_duration: float = 8.0
    smoothing: int = 3  # mode filter width in windows (odd)
    batch_size: int = 64


def window_spans(speech: list[Span], window: float, step: float) -> list[list[Span]]:
    """Windows per speech region; regions shorter than half a window are skipped."""
    out = []
    for s in speech:
        if s.duration < window / 2:
            continue
        wins, t = [], s.start
        while True:
            end = min(t + window, s.end)
            wins.append(Span(t, end))
            if end >= s.end:
                break
            t += step
        out.append(wins)
    return out


def mode_filter(labels: list[int], width: int) -> list[int]:
    if width <= 1 or len(labels) < width:
        return labels
    half = width // 2
    out = []
    for i in range(len(labels)):
        window = labels[max(0, i - half) : i + half + 1]
        out.append(Counter(window).most_common(1)[0][0])
    return out


def fold_small_clusters(
    emb: np.ndarray, labels: np.ndarray, durations: np.ndarray, min_duration: float
) -> np.ndarray:
    """Reassign windows of clusters with < min_duration total speech to the
    nearest remaining cluster centroid (if any cluster is big enough)."""
    totals = {k: float(durations[labels == k].sum()) for k in np.unique(labels)}
    big = [k for k, d in totals.items() if d >= min_duration]
    if not big or len(big) == len(totals):
        return labels
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    cents = np.stack([e[labels == k].mean(0) for k in big])
    cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
    out = labels.copy()
    small = ~np.isin(labels, big)
    out[small] = np.array(big)[np.argmax(e[small] @ cents.T, axis=1)]
    return out


def windows_to_turns(regions: list[list[Span]], labels: list[int]) -> list[Turn]:
    turns: list[Turn] = []
    i = 0
    for wins in regions:
        region_start, region_end = wins[0].start, wins[-1].end
        c = [(w.start + w.end) / 2 for w in wins]
        for j in range(len(wins)):
            start = region_start if j == 0 else (c[j - 1] + c[j]) / 2
            end = region_end if j == len(wins) - 1 else (c[j] + c[j + 1]) / 2
            label = f"SPEAKER_{labels[i]:02d}"
            if turns and turns[-1].speaker == label and abs(turns[-1].end - start) < 1e-6:
                turns[-1] = Turn(turns[-1].start, end, label)
            else:
                turns.append(Turn(start, end, label))
            i += 1
    return turns


class ClusterDiarizer(Diarizer):
    Params = ClusterDiarizerParams
    params: ClusterDiarizerParams

    def load(self) -> None:
        p = self.params
        self._embedder = registry.create("embedder", p.embedder.name, p.embedder.params, self.ctx)
        self._vad = registry.create("vad", p.vad.name, p.vad.params, self.ctx)
        self._embedder.load()
        self._vad.load()

    def unload(self) -> None:
        for c in (getattr(self, "_embedder", None), getattr(self, "_vad", None)):
            if c is not None:
                c.unload()
        super().unload()

    def diarize(self, audio: np.ndarray, sr: int) -> list[Turn]:
        p = self.params
        vad, embedder = self._vad, self._embedder
        assert isinstance(vad, VAD) and isinstance(embedder, SpeakerEmbedder)
        speech = vad.detect(resample(audio, sr, vad.sample_rate), vad.sample_rate)
        regions = window_spans(speech, p.window, p.step)
        flat = [w for wins in regions for w in wins]
        if not flat:
            return []
        esr = embedder.sample_rate
        audio_e = resample(audio, sr, esr)
        clips = [audio_e[int(w.start * esr) : int(w.end * esr)] for w in flat]
        emb = np.concatenate(
            [
                embedder.embed(clips[i : i + p.batch_size], esr)
                for i in range(0, len(clips), p.batch_size)
            ]
        )
        labels = self._cluster(emb)
        durations = np.full(len(flat), p.step)
        labels = fold_small_clusters(emb, labels, durations, p.min_cluster_duration)

        smoothed: list[int] = []
        i = 0
        for wins in regions:
            smoothed += mode_filter([int(x) for x in labels[i : i + len(wins)]], p.smoothing)
            i += len(wins)
        # Renumber by first appearance for readable, stable labels.
        order = {k: n for n, k in enumerate(dict.fromkeys(smoothed))}
        return windows_to_turns(regions, [order[k] for k in smoothed])

    def _cluster(self, emb: np.ndarray) -> np.ndarray:
        if len(emb) == 1:
            return np.zeros(1, dtype=int)
        from scipy.cluster.hierarchy import fcluster, linkage

        e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        z = linkage(e.astype(np.float64), method="average", metric="cosine")
        if self.params.num_speakers:
            raw = fcluster(z, t=self.params.num_speakers, criterion="maxclust")
        else:
            raw = fcluster(z, t=self.params.threshold, criterion="distance")
        return np.unique(raw, return_inverse=True)[1]
