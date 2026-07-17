"""
Templated fake reviews are a real, common pattern: someone (or something)
generates a batch of reviews from the same template with a couple of words
swapped ("Amazing product!!", "Amazing item!!", "Amazing gadget!!"). Plain
exact-match dedup won't catch these. This module embeds review text with
sentence-transformers and clusters on cosine distance so near-duplicates
(not just exact ones) get flagged as a group.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class DuplicateCluster:
    cluster_id: int
    review_indices: List[int]
    size: int
    representative_text: str


class TemplateDuplicateDetector:
    """
    embedder: object exposing `.encode(list[str]) -> np.ndarray`. In
    production that's a SentenceTransformer("all-MiniLM-L6-v2"); for tests
    a TF-IDF-based stand-in works fine since we only care about the
    clustering logic there, not embedding quality (see tests/test_dedup.py).
    """

    def __init__(self, embedder, eps: float = 0.15, min_samples: int = 2):
        self.embedder = embedder
        self.eps = eps                # max cosine distance within a cluster
        self.min_samples = min_samples  # need >=2 to call it a "duplicate cluster"

    def find_clusters(self, texts: List[str]) -> List[DuplicateCluster]:
        if len(texts) < self.min_samples:
            return []

        embeddings = self.embedder.encode(texts)
        embeddings = np.asarray(embeddings)

        distance_matrix = cosine_distances(embeddings)
        labels = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="precomputed"
        ).fit_predict(distance_matrix)

        clusters = []
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue  # noise / not part of any duplicate group
            indices = [i for i, l in enumerate(labels) if l == cluster_id]
            clusters.append(DuplicateCluster(
                cluster_id=int(cluster_id),
                review_indices=indices,
                size=len(indices),
                representative_text=texts[indices[0]],
            ))
        return clusters

    def flag_map(self, texts: List[str]) -> dict:
        """Convenience: index -> cluster_id for every text that landed in a duplicate cluster."""
        clusters = self.find_clusters(texts)
        flags = {}
        for c in clusters:
            for idx in c.review_indices:
                flags[idx] = c.cluster_id
        return flags


class SentenceTransformerEmbedder:
    """Thin wrapper so the rest of the code doesn't import sentence_transformers directly."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=False)
