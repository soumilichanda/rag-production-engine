"""
app/resilience/sla_tiers.py
Phase 34: Multi-Tenant SLA Tiers & Differentiated Quota Governance.
"""

import threading
from dataclasses import dataclass

from app.resilience.rate_limiter import TokenBucket


@dataclass(frozen=True)
class SLATierSpec:
    tier: str
    burst_capacity: int
    refill_rate_per_sec: float
    max_top_k: int


DEFAULT_SLA_TIERS: dict[str, SLATierSpec] = {
    "free": SLATierSpec(
        tier="free",
        burst_capacity=3,
        refill_rate_per_sec=1.0,
        max_top_k=3,
    ),
    "pro": SLATierSpec(
        tier="pro",
        burst_capacity=20,
        refill_rate_per_sec=10.0,
        max_top_k=10,
    ),
    "enterprise": SLATierSpec(
        tier="enterprise",
        burst_capacity=100,
        refill_rate_per_sec=50.0,
        max_top_k=50,
    ),
}


@dataclass
class SLADecision:
    allowed: bool
    tenant_id: str
    tier: str
    limit: int
    remaining: int
    retry_after: float
    max_top_k: int


class TieredQuotaManager:
    """Enforces tier-differentiated token budgets and retrieval depth ceilings."""

    def __init__(self, tiers: dict[str, SLATierSpec] | None = None) -> None:
        self.tiers: dict[str, SLATierSpec] = dict(tiers or DEFAULT_SLA_TIERS)
        self._buckets: dict[str, TokenBucket] = {}
        self._tenant_tiers: dict[str, str] = {}
        self._lock = threading.Lock()

    def resolve_tier(self, raw_tier: str | None) -> SLATierSpec:
        normalized = (raw_tier or "enterprise").strip().lower()
        return self.tiers.get(normalized, self.tiers["enterprise"])

    def assign_tenant_tier(self, tenant_id: str, tier: str) -> SLATierSpec:
        spec = self.resolve_tier(tier)
        with self._lock:
            self._tenant_tiers[tenant_id] = spec.tier
            self._buckets[tenant_id] = TokenBucket(
                capacity=spec.burst_capacity,
                refill_rate=spec.refill_rate_per_sec,
            )
        return spec

    def _get_tenant_bucket(
        self, tenant_id: str, requested_tier: str | None
    ) -> tuple[SLATierSpec, TokenBucket]:
        with self._lock:
            if requested_tier:
                spec = self.resolve_tier(requested_tier)
                if (
                    self._tenant_tiers.get(tenant_id) != spec.tier
                    or tenant_id not in self._buckets
                ):
                    self._tenant_tiers[tenant_id] = spec.tier
                    self._buckets[tenant_id] = TokenBucket(
                        capacity=spec.burst_capacity,
                        refill_rate=spec.refill_rate_per_sec,
                    )
            elif tenant_id in self._buckets:
                spec = self.tiers[self._tenant_tiers[tenant_id]]
            else:
                spec = self.tiers["enterprise"]
                self._tenant_tiers[tenant_id] = spec.tier
                self._buckets[tenant_id] = TokenBucket(
                    capacity=spec.burst_capacity,
                    refill_rate=spec.refill_rate_per_sec,
                )
            return spec, self._buckets[tenant_id]

    def evaluate(
        self,
        tenant_id: str | None,
        tier: str | None = None,
        cost: float = 1.0,
    ) -> SLADecision:
        resolved_id = (tenant_id or "default").strip() or "default"
        spec, bucket = self._get_tenant_bucket(resolved_id, tier)
        allowed, remaining, retry_after = bucket.try_acquire(cost=cost)
        return SLADecision(
            allowed=allowed,
            tenant_id=resolved_id,
            tier=spec.tier,
            limit=spec.burst_capacity,
            remaining=remaining,
            retry_after=retry_after,
            max_top_k=spec.max_top_k,
        )

    def max_top_k_for(self, tenant_id: str | None, tier: str | None = None) -> int:
        resolved_id = (tenant_id or "default").strip() or "default"
        spec, _ = self._get_tenant_bucket(resolved_id, tier)
        return spec.max_top_k

    def stats(self) -> dict[str, object]:
        with self._lock:
            tenants = {
                tid: {
                    "tier": self._tenant_tiers[tid],
                    "max_top_k": self.tiers[self._tenant_tiers[tid]].max_top_k,
                    **bucket.snapshot(),
                }
                for tid, bucket in self._buckets.items()
            }
        return {
            "tiers": {
                name: {
                    "burst_capacity": s.burst_capacity,
                    "refill_rate_per_sec": s.refill_rate_per_sec,
                    "max_top_k": s.max_top_k,
                }
                for name, s in self.tiers.items()
            },
            "active_tenants": len(tenants),
            "tenants": tenants,
        }
