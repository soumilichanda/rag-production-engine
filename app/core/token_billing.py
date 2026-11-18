"""
Phase 36: Streaming Token Budget Accounting & Real-Time Cost Attribution
Tracks per-tenant prompt/completion tokens and real-time USD cost rollups across SSE streams.
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
class TenantUsageRecord:
    tenant_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stream_requests: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "stream_requests": self.stream_requests,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


class TokenCostLedger:
    """Thread-safe per-tenant streaming token & USD cost attribution ledger."""

    def __init__(self, pricing: ModelPricing | None = None) -> None:
        self._pricing = pricing or DEFAULT_PRICING
        self._lock = threading.Lock()
        self._records: dict[str, TenantUsageRecord] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        # Deterministic subword-aware approximation (~1.3 tokens/word)
        return max(1, int(round(len(stripped.split()) * 1.3)))

    def record_stream_usage(
        self,
        tenant_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TenantUsageRecord:
        safe_prompt = max(0, prompt_tokens)
        safe_completion = max(0, completion_tokens)
        cost = (
            (safe_prompt / 1000.0) * self._pricing.prompt_per_1k_usd
            + (safe_completion / 1000.0) * self._pricing.completion_per_1k_usd
        )
        with self._lock:
            rec = self._records.get(tenant_id)
            if rec is None:
                rec = TenantUsageRecord(tenant_id=tenant_id)
                self._records[tenant_id] = rec
            rec.prompt_tokens += safe_prompt
            rec.completion_tokens += safe_completion
            rec.stream_requests += 1
            rec.estimated_cost_usd += cost
            return TenantUsageRecord(
                tenant_id=rec.tenant_id,
                prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                stream_requests=rec.stream_requests,
                estimated_cost_usd=rec.estimated_cost_usd,
            )

    def get_tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(tenant_id)
            if rec is None:
                return TenantUsageRecord(tenant_id=tenant_id).to_dict()
            return rec.to_dict()

    def get_all_usage(self) -> dict[str, Any]:
        with self._lock:
            tenants = {tid: rec.to_dict() for tid, rec in self._records.items()}
            total_prompt = sum(r.prompt_tokens for r in self._records.values())
            total_completion = sum(r.completion_tokens for r in self._records.values())
            total_cost = round(sum(r.estimated_cost_usd for r in self._records.values()), 6)
            return {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "total_estimated_cost_usd": total_cost,
                "tenants": tenants,
            }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
