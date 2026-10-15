"""
app/core/semantic_cache.py
Semantic vector caching engine based on cosine distance cutoffs.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SemanticCache:
    """
    In-memory semantic cache storing dense query embeddings and retrieved responses.
    Cache hit is triggered when cosine distance <= distance_threshold.
    """

    def __init__(self, dimension: int = 8, distance_threshold: float = 0.08, capacity: int = 500):
        self.dimension = dimension
        self.distance_threshold = distance_threshold
        self.capacity = capacity
        self.query_vectors: List[np.ndarray] = []
        self.cached_payloads: List[Dict[str, Any]] = []
        self.hits = 0
        self.misses = 0

    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot = float(np.dot(v1, v2))
        norm_product = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm_product == 0.0:
            return 1.0
        similarity = dot / norm_product
        return max(0.0, 1.0 - similarity)

    def lookup(self, query_vector: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float]:
        start = time.perf_counter()
        if not self.query_vectors:
            self.misses += 1
            return None, (time.perf_counter() - start) * 1000.0

        q_norm = query_vector / (np.linalg.norm(query_vector) or 1.0)
        min_dist = float("inf")
        best_idx = -1

        for idx, cached_vec in enumerate(self.query_vectors):
            dist = self._cosine_distance(q_norm, cached_vec)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        lookup_latency = (time.perf_counter() - start) * 1000.0

        if min_dist <= self.distance_threshold and best_idx != -1:
            self.hits += 1
            payload = self.cached_payloads[best_idx].copy()
            payload["cache_hit"] = True
            payload["semantic_distance"] = round(min_dist, 5)
            return payload, lookup_latency

        self.misses += 1
        return None, lookup_latency

    def insert(self, query_vector: np.ndarray, response_payload: Dict[str, Any]) -> None:
        if len(self.query_vectors) >= self.capacity:
            # Evict oldest entry (FIFO)
            self.query_vectors.pop(0)
            self.cached_payloads.pop(0)

        q_norm = query_vector / (np.linalg.norm(query_vector) or 1.0)
        self.query_vectors.append(q_norm)
        self.cached_payloads.append(response_payload)

    def get_telemetry(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "capacity": self.capacity,
            "cached_entries": len(self.query_vectors),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "distance_threshold": self.distance_threshold,
        }
