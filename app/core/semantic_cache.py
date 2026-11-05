"""
app/core/semantic_cache.py
Thread-safe semantic cache with TTL expiration, stampede locks,
tag invalidation, async LRU memory compaction, and legacy interface support.
"""

import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np


class CacheEntry:
    def __init__(
        self,
        embedding: np.ndarray,
        payload: Any,
        ttl_seconds: float,
        tags: list[str] | None = None,
    ):
        self.embedding = embedding
        self.payload = payload
        self.expires_at = time.time() + ttl_seconds
        self.tags = set(tags or [])

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class SemanticCache:
    def __init__(
        self,
        capacity: int = 512,
        threshold: float = 0.90,
        default_ttl: float = 300.0,
        dimension: int | None = None,
        distance_threshold: float | None = None,
        similarity_cutoff: float | None = None,
    ):
        self.capacity = capacity
        # Adapt distance threshold (dist <= 0.10 means sim >= 0.90)
        if distance_threshold is not None:
            self.threshold = 1.0 - distance_threshold
        elif similarity_cutoff is not None:
            self.threshold = similarity_cutoff
        else:
            self.threshold = threshold

        self.dimension = dimension
        self.distance_threshold = distance_threshold
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stampede_locks: dict[str, threading.Lock] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def get_stampede_lock(self, key: str) -> threading.Lock:
        with self.lock:
            if key not in self.stampede_locks:
                self.stampede_locks[key] = threading.Lock()
            return self.stampede_locks[key]

    def compact(self) -> int:
        with self.lock:
            initial_count = len(self.cache)
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            for k in expired_keys:
                del self.cache[k]
                self.evictions += 1

            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
                self.evictions += 1

            return initial_count - len(self.cache)

    def get(self, query_vec: np.ndarray) -> Any | None:
        norm_q = self._normalize(np.asarray(query_vec, dtype=np.float32))

        with self.lock:
            if not self.cache:
                self.misses += 1
                return None

            best_key: str | None = None
            best_sim = -1.0

            for key, entry in list(self.cache.items()):
                if entry.is_expired():
                    del self.cache[key]
                    self.evictions += 1
                    continue

                sim = float(np.dot(norm_q, entry.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_key is not None and best_sim >= self.threshold:
                self.hits += 1
                self.cache.move_to_end(best_key)
                return self.cache[best_key].payload

            self.misses += 1
            return None

    def put(
        self,
        query_key: str,
        query_vec: np.ndarray,
        payload: Any,
        ttl: float | None = None,
        tags: list[str] | None = None,
    ) -> None:
        norm_q = self._normalize(np.asarray(query_vec, dtype=np.float32))
        active_ttl = ttl if ttl is not None else self.default_ttl

        with self.lock:
            if query_key in self.cache:
                self.cache.move_to_end(query_key)
            self.cache[query_key] = CacheEntry(norm_q, payload, active_ttl, tags)

            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
                self.evictions += 1

    # --- Legacy API Compatibility Aliases ---
    def lookup(self, query_vector: np.ndarray) -> tuple[dict[str, Any] | None, float]:
        start = time.perf_counter()
        res = self.get(query_vector)
        latency = (time.perf_counter() - start) * 1000.0
        if res is not None:
            if isinstance(res, dict):
                payload = res.copy()
            else:
                payload = {"data": res}
            payload["cache_hit"] = True
            return payload, latency
        return None, latency

    def insert(
        self, query_vector: np.ndarray, response_payload: dict[str, Any]
    ) -> None:
        key = f"k_{len(self.cache)}_{time.time()}"
        self.put(key, query_vector, response_payload)

    def get_telemetry(self) -> dict[str, Any]:
        s = self.stats()
        return {
            "capacity": self.capacity,
            "cached_entries": s["size"],
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": s["hit_ratio"],
            "distance_threshold": self.distance_threshold or 0.08,
        }

    def invalidate_by_tag(self, tag: str) -> int:
        with self.lock:
            targets = [k for k, entry in self.cache.items() if tag in entry.tags]
            for k in targets:
                del self.cache[k]
                self.evictions += 1
            return len(targets)

    def invalidate_all(self) -> int:
        with self.lock:
            count = len(self.cache)
            self.evictions += count
            self.cache.clear()
            return count

    def stats(self) -> dict:
        with self.lock:
            total = self.hits + self.misses
            ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(ratio, 3),
                "evictions": self.evictions,
            }
