"""
app/resilience/rate_limiter.py
Hierarchical Token-Bucket Rate Limiter & Per-Tenant Quota Governance.
"""

import math
import threading
import time
from dataclasses import dataclass


@dataclass
class QuotaDecision:
    allowed: bool
    tenant_id: str
    limit: int
    remaining: int
    retry_after: float
    scope: str = "tenant"


class TokenBucket:
    """Thread-safe monotonic token bucket supporting fractional refill rates."""

    def __init__(self, capacity: int, refill_rate: float) -> None:
        if capacity <= 0 or refill_rate <= 0:
            raise ValueError("capacity and refill_rate must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_refill)
        if elapsed > 0.0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def try_acquire(self, cost: float = 1.0) -> tuple[bool, int, float]:
        with self._lock:
            now = time.monotonic()
            self._refill(now)

            if self.tokens >= cost:
                self.tokens -= cost
                return True, math.floor(self.tokens), 0.0

            deficit = cost - self.tokens
            retry_after = round(deficit / self.refill_rate, 3)
            return False, 0, retry_after

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            self._refill(time.monotonic())
            return {
                "capacity": int(self.capacity),
                "refill_rate_per_sec": self.refill_rate,
                "available_tokens": round(self.tokens, 2),
            }


class HierarchicalQuotaManager:
    """Enforces global system capacity alongside isolated per-tenant token buckets."""

    def __init__(
        self,
        default_tenant_capacity: int = 20,
        default_tenant_refill_rate: float = 5.0,
        global_capacity: int = 200,
        global_refill_rate: float = 50.0,
    ) -> None:
        self.default_tenant_capacity = default_tenant_capacity
        self.default_tenant_refill_rate = default_tenant_refill_rate
        self.global_bucket = TokenBucket(global_capacity, global_refill_rate)
        self._tenant_buckets: dict[str, TokenBucket] = {}
        self._tenant_overrides: dict[str, tuple[int, float]] = {}
        self._registry_lock = threading.Lock()

    def configure_tenant(self, tenant_id: str, capacity: int, refill_rate: float) -> None:
        with self._registry_lock:
            self._tenant_overrides[tenant_id] = (capacity, refill_rate)
            self._tenant_buckets[tenant_id] = TokenBucket(capacity, refill_rate)

    def _get_or_create_bucket(self, tenant_id: str) -> TokenBucket:
        with self._registry_lock:
            bucket = self._tenant_buckets.get(tenant_id)
            if bucket is None:
                cap, rate = self._tenant_overrides.get(
                    tenant_id,
                    (self.default_tenant_capacity, self.default_tenant_refill_rate),
                )
                bucket = TokenBucket(cap, rate)
                self._tenant_buckets[tenant_id] = bucket
            return bucket

    def evaluate(self, tenant_id: str | None, cost: float = 1.0) -> QuotaDecision:
        resolved_tenant = (tenant_id or "default").strip() or "default"
        bucket = self._get_or_create_bucket(resolved_tenant)

        tenant_ok, tenant_rem, tenant_retry = bucket.try_acquire(cost=cost)
        if not tenant_ok:
            return QuotaDecision(
                allowed=False,
                tenant_id=resolved_tenant,
                limit=int(bucket.capacity),
                remaining=0,
                retry_after=tenant_retry,
                scope="tenant",
            )

        global_ok, _, global_retry = self.global_bucket.try_acquire(cost=cost)
        if not global_ok:
            return QuotaDecision(
                allowed=False,
                tenant_id=resolved_tenant,
                limit=int(self.global_bucket.capacity),
                remaining=0,
                retry_after=global_retry,
                scope="global",
            )

        return QuotaDecision(
            allowed=True,
            tenant_id=resolved_tenant,
            limit=int(bucket.capacity),
            remaining=tenant_rem,
            retry_after=0.0,
            scope="tenant",
        )

    def stats(self) -> dict[str, object]:
        with self._registry_lock:
            tenants = {tid: b.snapshot() for tid, b in self._tenant_buckets.items()}
        return {
            "global_bucket": self.global_bucket.snapshot(),
            "active_tenants": len(tenants),
            "tenants": tenants,
        }
