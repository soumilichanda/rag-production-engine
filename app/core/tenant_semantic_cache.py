"""
app/core/tenant_semantic_cache.py
Phase 35: Tenant-Scoped Semantic Caching with Namespace Isolation & Adaptive SLA TTL.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

TIER_CACHE_POLICIES: dict[str, dict[str, Any]] = {
    "free": {"ttl_seconds": 30.0, "max_entries": 16, "similarity_threshold": 0.96},
    "pro": {"ttl_seconds": 300.0, "max_entries": 128, "similarity_threshold": 0.94},
    "enterprise": {
        "ttl_seconds": 3600.0,
        "max_entries": 1024,
        "similarity_threshold": 0.92,
    },
}


@dataclass
class TenantCacheEntry:
    query: str
    embedding: np.ndarray
    payload: Any
    created_at: float
    expires_at: float
    tier: str


class TenantSemanticCache:
    """
    Namespace-isolated semantic query cache enforcing strict tenant boundaries
    and tier-adaptive TTL + capacity policies.
    """

    def __init__(self, policies: dict[str, dict[str, Any]] | None = None):
        self.policies = policies or TIER_CACHE_POLICIES
        self._namespaces: dict[str, OrderedDict[str, TenantCacheEntry]] = {}
        self._metrics: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def _resolve_policy(self, tier: str) -> tuple[str, dict[str, Any]]:
        normalized = (tier or "free").strip().lower()
        if normalized not in self.policies:
            normalized = "free"
        return normalized, self.policies[normalized]

    def _ensure_tenant(self, tenant_id: str) -> None:
        if tenant_id not in self._namespaces:
            self._namespaces[tenant_id] = OrderedDict()
            self._metrics[tenant_id] = {"hits": 0, "misses": 0, "evictions": 0}

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr
        return arr / norm

    def get(
        self,
        tenant_id: str,
        query_embedding: np.ndarray,
        tier: str = "free",
        now: float | None = None,
    ) -> dict[str, Any] | None:
        current_time = time.monotonic() if now is None else now
        resolved_tier, policy = self._resolve_policy(tier)
        threshold = float(policy["similarity_threshold"])
        q_norm = self._normalize(query_embedding)

        with self._lock:
            self._ensure_tenant(tenant_id)
            bucket = self._namespaces[tenant_id]

            # Purge expired entries in this tenant namespace first
            expired_keys = [
                k for k, entry in bucket.items() if current_time >= entry.expires_at
            ]
            for k in expired_keys:
                del bucket[k]
                self._metrics[tenant_id]["evictions"] += 1

            best_key: str | None = None
            best_score = -1.0

            for key, entry in bucket.items():
                if entry.embedding.shape != q_norm.shape:
                    continue
                score = float(np.dot(q_norm, entry.embedding))
                if score >= threshold and score > best_score:
                    best_score = score
                    best_key = key

            if best_key is None:
                self._metrics[tenant_id]["misses"] += 1
                return None

            entry = bucket[best_key]
            bucket.move_to_end(best_key)
            self._metrics[tenant_id]["hits"] += 1
            remaining_ttl = max(0.0, round(entry.expires_at - current_time, 3))
            return {
                "tenant_id": tenant_id,
                "tier": resolved_tier,
                "matched_query": entry.query,
                "similarity": round(best_score, 4),
                "ttl_remaining_s": remaining_ttl,
                "payload": entry.payload,
            }

    def put(
        self,
        tenant_id: str,
        query: str,
        query_embedding: np.ndarray,
        payload: Any,
        tier: str = "free",
        ttl_override: float | None = None,
        now: float | None = None,
    ) -> TenantCacheEntry:
        current_time = time.monotonic() if now is None else now
        resolved_tier, policy = self._resolve_policy(tier)
        ttl = (
            float(ttl_override)
            if ttl_override is not None
            else float(policy["ttl_seconds"])
        )
        max_entries = int(policy["max_entries"])
        q_norm = self._normalize(query_embedding)

        with self._lock:
            self._ensure_tenant(tenant_id)
            bucket = self._namespaces[tenant_id]

            if query in bucket:
                del bucket[query]

            while len(bucket) >= max_entries:
                bucket.popitem(last=False)
                self._metrics[tenant_id]["evictions"] += 1

            entry = TenantCacheEntry(
                query=query,
                embedding=q_norm,
                payload=payload,
                created_at=current_time,
                expires_at=current_time + ttl,
                tier=resolved_tier,
            )
            bucket[query] = entry
            return entry

    def invalidate_tenant(self, tenant_id: str) -> int:
        with self._lock:
            if tenant_id not in self._namespaces:
                return 0
            count = len(self._namespaces[tenant_id])
            self._namespaces[tenant_id].clear()
            self._metrics[tenant_id]["evictions"] += count
            return count

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            tenants_report: dict[str, Any] = {}
            total_hits = 0
            total_misses = 0

            for tid, bucket in self._namespaces.items():
                active = sum(1 for e in bucket.values() if now < e.expires_at)
                stats = self._metrics[tid]
                hits = stats["hits"]
                misses = stats["misses"]
                total = hits + misses
                hit_rate = round(hits / total, 4) if total > 0 else 0.0
                total_hits += hits
                total_misses += misses
                tenants_report[tid] = {
                    "active_entries": active,
                    "hits": hits,
                    "misses": misses,
                    "evictions": stats["evictions"],
                    "hit_rate": hit_rate,
                }

            overall_total = total_hits + total_misses
            return {
                "policies": self.policies,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "overall_hit_rate": round(total_hits / overall_total, 4)
                if overall_total > 0
                else 0.0,
                "tenants": tenants_report,
            }
