"""
Phase 36 & Phase 37: Streaming Token Budget Accounting, Real-Time Cost Attribution,
Tenant Spend Caps, Budget Alerts & Automated Quota Enforcement.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    prompt_per_1k_usd: float
    completion_per_1k_usd: float


DEFAULT_PRICING = ModelPricing(prompt_per_1k_usd=0.0015, completion_per_1k_usd=0.0060)


@dataclass
class TenantBudgetPolicy:
    tenant_id: str
    spend_cap_usd: float
    alert_threshold_ratio: float = 0.80


@dataclass
class BudgetEvaluation:
    tenant_id: str
    allowed: bool
    spend_cap_usd: float | None
    spent_usd: float
    remaining_usd: float | None
    utilization_ratio: float | None
    alert_triggered: bool


@dataclass
class TenantUsageRecord:
    tenant_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stream_requests: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(
        self,
        policy: TenantBudgetPolicy | None = None,
        alert_triggered: bool = False,
    ) -> dict[str, Any]:
        remaining_usd: float | None = None
        spend_cap_usd: float | None = None
        utilization_ratio: float | None = None
        if policy is not None:
            spend_cap_usd = round(policy.spend_cap_usd, 6)
            remaining_usd = round(
                max(0.0, policy.spend_cap_usd - self.estimated_cost_usd), 6
            )
            utilization_ratio = (
                round(self.estimated_cost_usd / policy.spend_cap_usd, 4)
                if policy.spend_cap_usd > 0
                else 1.0
            )
        return {
            "tenant_id": self.tenant_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "stream_requests": self.stream_requests,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "spend_cap_usd": spend_cap_usd,
            "remaining_usd": remaining_usd,
            "utilization_ratio": utilization_ratio,
            "alert_triggered": alert_triggered,
        }


class TokenCostLedger:
    """Thread-safe per-tenant streaming token ledger with USD spend cap enforcement."""

    def __init__(self, pricing: ModelPricing | None = None) -> None:
        self._pricing = pricing or DEFAULT_PRICING
        self._lock = threading.Lock()
        self._records: dict[str, TenantUsageRecord] = {}
        self._budgets: dict[str, TenantBudgetPolicy] = {}
        self._alerts: list[dict[str, Any]] = []
        self._alerted_tenants: set[str] = set()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        # Deterministic subword-aware approximation (~1.3 tokens/word)
        return max(1, round(len(stripped.split()) * 1.3))

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int = 0) -> float:
        safe_prompt = max(0, prompt_tokens)
        safe_completion = max(0, completion_tokens)
        return (safe_prompt / 1000.0) * self._pricing.prompt_per_1k_usd + (
            safe_completion / 1000.0
        ) * self._pricing.completion_per_1k_usd

    def set_tenant_budget(
        self,
        tenant_id: str,
        spend_cap_usd: float,
        alert_threshold_ratio: float = 0.80,
    ) -> dict[str, Any]:
        if spend_cap_usd < 0:
            raise ValueError("spend_cap_usd must be non-negative")
        if not (0.0 < alert_threshold_ratio <= 1.0):
            raise ValueError("alert_threshold_ratio must be within (0.0, 1.0]")
        with self._lock:
            policy = TenantBudgetPolicy(
                tenant_id=tenant_id,
                spend_cap_usd=float(spend_cap_usd),
                alert_threshold_ratio=float(alert_threshold_ratio),
            )
            self._budgets[tenant_id] = policy
            self._alerted_tenants.discard(tenant_id)
            rec = self._records.get(tenant_id) or TenantUsageRecord(tenant_id=tenant_id)
            self._evaluate_alert_locked(tenant_id, rec, policy)
            return rec.to_dict(
                policy=policy, alert_triggered=tenant_id in self._alerted_tenants
            )

    def _evaluate_alert_locked(
        self,
        tenant_id: str,
        rec: TenantUsageRecord,
        policy: TenantBudgetPolicy,
    ) -> bool:
        if policy.spend_cap_usd <= 0:
            ratio = 1.0
        else:
            ratio = rec.estimated_cost_usd / policy.spend_cap_usd
        if ratio >= policy.alert_threshold_ratio:
            if tenant_id not in self._alerted_tenants:
                self._alerted_tenants.add(tenant_id)
                self._alerts.append(
                    {
                        "tenant_id": tenant_id,
                        "spent_usd": round(rec.estimated_cost_usd, 6),
                        "spend_cap_usd": round(policy.spend_cap_usd, 6),
                        "utilization_ratio": round(ratio, 4),
                        "threshold_ratio": policy.alert_threshold_ratio,
                        "exhausted": rec.estimated_cost_usd >= policy.spend_cap_usd,
                    }
                )
            return True
        return False

    def check_budget_available(
        self,
        tenant_id: str,
        projected_cost_usd: float = 0.0,
    ) -> BudgetEvaluation:
        with self._lock:
            policy = self._budgets.get(tenant_id)
            rec = self._records.get(tenant_id)
            spent = rec.estimated_cost_usd if rec is not None else 0.0
            if policy is None:
                return BudgetEvaluation(
                    tenant_id=tenant_id,
                    allowed=True,
                    spend_cap_usd=None,
                    spent_usd=round(spent, 6),
                    remaining_usd=None,
                    utilization_ratio=None,
                    alert_triggered=False,
                )
            remaining = max(0.0, policy.spend_cap_usd - spent)
            ratio = (
                round(spent / policy.spend_cap_usd, 4)
                if policy.spend_cap_usd > 0
                else 1.0
            )
            allowed = (
                spent + max(0.0, projected_cost_usd)
            ) <= policy.spend_cap_usd and (policy.spend_cap_usd > 0.0)
            return BudgetEvaluation(
                tenant_id=tenant_id,
                allowed=allowed,
                spend_cap_usd=round(policy.spend_cap_usd, 6),
                spent_usd=round(spent, 6),
                remaining_usd=round(remaining, 6),
                utilization_ratio=ratio,
                alert_triggered=tenant_id in self._alerted_tenants,
            )

    def record_stream_usage(
        self,
        tenant_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TenantUsageRecord:
        safe_prompt = max(0, prompt_tokens)
        safe_completion = max(0, completion_tokens)
        cost = self.estimate_cost(safe_prompt, safe_completion)
        with self._lock:
            rec = self._records.get(tenant_id)
            if rec is None:
                rec = TenantUsageRecord(tenant_id=tenant_id)
                self._records[tenant_id] = rec
            rec.prompt_tokens += safe_prompt
            rec.completion_tokens += safe_completion
            rec.stream_requests += 1
            rec.estimated_cost_usd += cost
            policy = self._budgets.get(tenant_id)
            if policy is not None:
                self._evaluate_alert_locked(tenant_id, rec, policy)
            return TenantUsageRecord(
                tenant_id=rec.tenant_id,
                prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                stream_requests=rec.stream_requests,
                estimated_cost_usd=rec.estimated_cost_usd,
            )

    def get_tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(tenant_id) or TenantUsageRecord(tenant_id=tenant_id)
            policy = self._budgets.get(tenant_id)
            return rec.to_dict(
                policy=policy, alert_triggered=tenant_id in self._alerted_tenants
            )

    def get_all_usage(self) -> dict[str, Any]:
        with self._lock:
            tenants = {
                tid: rec.to_dict(
                    policy=self._budgets.get(tid),
                    alert_triggered=tid in self._alerted_tenants,
                )
                for tid, rec in self._records.items()
            }
            for tid, policy in self._budgets.items():
                if tid not in tenants:
                    tenants[tid] = TenantUsageRecord(tenant_id=tid).to_dict(
                        policy=policy,
                        alert_triggered=tid in self._alerted_tenants,
                    )
            total_prompt = sum(r.prompt_tokens for r in self._records.values())
            total_completion = sum(r.completion_tokens for r in self._records.values())
            total_cost = round(
                sum(r.estimated_cost_usd for r in self._records.values()), 6
            )
            return {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "total_estimated_cost_usd": total_cost,
                "active_alerts": len(self._alerts),
                "alerts": list(self._alerts),
                "tenants": tenants,
            }

    def get_alerts(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._alerts)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._budgets.clear()
            self._alerts.clear()
            self._alerted_tenants.clear()
