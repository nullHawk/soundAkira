"""Global speaker identity across sources.

Diarization labels (SPEAKER_00, ...) only mean something inside one source.
To get dataset-wide speaker IDs:

1. Local centroids: per (source, local speaker), a duration-weighted mean of
   L2-normalised segment embeddings. Segments far from their own centroid are
   usually diarization mistakes: they are excluded from the centroid, and
   their similarity is kept as the `speaker_similarity` metric so filters can
   drop them.
2. Clustering: agglomerative clustering of local centroids by cosine distance
   merges the same person across sources, and also re-merges one speaker that
   diarization split in two within a source.
3. Anchors (optional): labelled clips of known people. Clusters that match an
   anchor are named after it, and clusters matching the same anchor are merged.
4. Registry: new clusters are matched to the previous build's speakers by
   voice (centroid similarity), preferring ones that share members, so IDs stay
   the same as sources are added. Shared labels alone never carry an ID over,
   because re-diarizing a source can permute its labels. Names edited in the
   registry file are kept.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from soundakira.utils.io import read_json, write_json


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@dataclass
class LocalSpeaker:
    source_id: str
    label: str
    centroid: np.ndarray
    duration: float  # seconds of speech consistent with the centroid

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.label}"


def local_centroid(
    embeddings: np.ndarray, durations: np.ndarray, purity_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (centroid, per-segment cosine similarity to it).

    Starts from the duration-weighted medoid rather than the mean. A mean gets
    pulled towards mislabelled segments, which then look "similar enough" and
    are never trimmed. It then re-estimates from segments within
    `purity_threshold`."""
    e = l2norm(embeddings.astype(np.float64))
    w = durations.astype(np.float64)
    centroid = e[int(np.argmax((e @ e.T) @ w))]
    for _ in range(3):
        keep = (e @ centroid) >= purity_threshold
        if not keep.any():
            break
        centroid = l2norm((e[keep] * w[keep, None]).sum(0))
    return centroid, e @ centroid


def cluster_centroids(
    centroids: np.ndarray, threshold: float, linkage: str = "average"
) -> np.ndarray:
    """Agglomerative clustering with a cosine-distance cut. Returns labels 0..k-1."""
    n = len(centroids)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    z = scipy_linkage(l2norm(centroids.astype(np.float64)), method=linkage, metric="cosine")
    labels = fcluster(z, t=threshold, criterion="distance")
    _, dense = np.unique(labels, return_inverse=True)
    return dense


@dataclass
class Cluster:
    centroid: np.ndarray
    members: list[str]  # "source_id:label"
    duration: float
    name: str | None = None
    anchor_similarity: float | None = None


def build_clusters(locals_: list[LocalSpeaker], labels: np.ndarray) -> list[Cluster]:
    groups: dict[int, list[LocalSpeaker]] = defaultdict(list)
    for spk, lab in zip(locals_, labels):
        groups[int(lab)].append(spk)
    clusters = []
    for _, members in sorted(groups.items()):
        w = np.array([m.duration for m in members])
        c = l2norm((np.stack([m.centroid for m in members]) * w[:, None]).sum(0))
        clusters.append(Cluster(c, sorted(m.key for m in members), float(w.sum())))
    return clusters


def apply_anchors(
    clusters: list[Cluster], anchors: dict[str, np.ndarray], threshold: float
) -> list[Cluster]:
    """Name clusters after the best-matching anchor; merge clusters that share one."""
    if not anchors:
        return clusters
    names = sorted(anchors)
    a = l2norm(np.stack([anchors[n] for n in names]))
    by_name: dict[str, list[tuple[Cluster, float]]] = defaultdict(list)
    out: list[Cluster] = []
    for c in clusters:
        sims = a @ c.centroid
        j = int(np.argmax(sims))
        if sims[j] >= threshold:
            by_name[names[j]].append((c, float(sims[j])))
        else:
            out.append(c)
    for name, matched in by_name.items():
        w = np.array([c.duration for c, _ in matched])
        centroid = l2norm((np.stack([c.centroid for c, _ in matched]) * w[:, None]).sum(0))
        members = sorted(m for c, _ in matched for m in c.members)
        out.append(Cluster(centroid, members, float(w.sum()), name, max(s for _, s in matched)))
    return out


@dataclass
class SpeakerRegistry:
    """Persistent global-ID table (work_dir/speakers/registry.json)."""

    path: Path
    next_id: int = 0
    speakers: dict[int, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> SpeakerRegistry:
        if not path.exists():
            return cls(path)
        data = read_json(path)
        return cls(path, data["next_id"], {int(k): v for k, v in data["speakers"].items()})

    def save(self) -> None:
        write_json(
            self.path,
            {
                "next_id": self.next_id,
                "speakers": {str(k): v for k, v in sorted(self.speakers.items())},
            },
        )

    def assign(self, clusters: list[Cluster], match_similarity: float) -> list[int]:
        """Stable IDs: greedy one-to-one matching of voices (centroid similarity
        >= match_similarity), preferring old speakers that share members."""
        old_ids = sorted(self.speakers)
        pairs: list[tuple[int, float, int, int]] = []
        if old_ids and clusters:
            old_c = l2norm(
                np.array([self.speakers[i]["centroid"] for i in old_ids], dtype=np.float64)
            )
            new_c = l2norm(np.stack([c.centroid for c in clusters]).astype(np.float64))
            sims = new_c @ old_c.T
            old_members = [set(self.speakers[i]["members"]) for i in old_ids]
            for ci, c in enumerate(clusters):
                mem = set(c.members)
                for oj, oid in enumerate(old_ids):
                    # The voice must match. Member labels alone are not enough:
                    # re-diarizing a source can permute its SPEAKER_xx labels.
                    if sims[ci, oj] >= match_similarity:
                        shared = len(mem & old_members[oj])
                        pairs.append((shared, float(sims[ci, oj]), ci, oid))
        pairs.sort(key=lambda p: (-p[0], -p[1], p[2], p[3]))

        ids: list[int | None] = [None] * len(clusters)
        taken: set[int] = set()
        for _, _, ci, oid in pairs:
            if ids[ci] is None and oid not in taken:
                ids[ci] = oid
                taken.add(oid)
        for ci in sorted(range(len(clusters)), key=lambda i: clusters[i].members[0]):
            if ids[ci] is None:
                ids[ci] = self.next_id
                self.next_id += 1

        for ci, c in enumerate(clusters):
            sid = ids[ci]
            assert sid is not None
            prev = self.speakers.get(sid, {})
            self.speakers[sid] = {
                "name": c.name or prev.get("name"),
                "centroid": [round(float(x), 6) for x in c.centroid],
                "members": c.members,
                "duration": round(c.duration, 2),
            }
        return [int(i) for i in ids]  # type: ignore[arg-type]
