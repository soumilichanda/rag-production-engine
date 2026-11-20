"""
Phase 38: Model Routing & Cost-Aware Tiered LLM Fallback
Selects optimal LLM profiles (economy, balanced, flagship) based on query complexity,
tenant SLA tier, and remaining TokenCostLedger budget headroom.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from app.core.token_billing import ModelPricing


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    pricing: ModelPricing
    max_context_tokens: int


MODEL_PROFILES: dict[str, ModelProfile] = {
    "economy": ModelProfile(
        name="economy",
        model_id="gpt-4o-mini-tier",
        pricing=ModelPricing(prompt_per_1k_usd=0.00015, completion_per_1k_usd=0.00060),
        max_context_tokens=4096,
    ),
    "balanced": ModelProfile(
        name="balanced",
        model_id="claude-sonnet-tier",
        pricing=ModelPricing(prompt_per_1k_usd=0.00150, completion_per_1k_usd=0.00600),
        max_context_tokens=16384,
    ),
    "flagship": ModelProfile(
        name="flagship",
        model_id="o3-reasoning-tier",
        pricing=ModelPricing(prompt_per_1k_usd=0.00500, completion_per_1k_usd=0.01500),
        max_context_tokens=32768,
    ),
}


@dataclass(frozen=True)
class RoutingDecision:
    tenant_id: str
    selected_profile: str
    model_id: str
    complexity: str
    downgraded: bool
    reason: str
    pricing: ModelPricing


class CostAwareModelRouter:
    """Routes queries across economy/balanced/flagship models using complexity & budget headroom."""

    _COMPLEX_KEYWORDS = (
        "compare",
        "synthesize",
        "architect",
        "tradeoff",
        "evaluate",
        "root cause",
        "multi-step",
        "analyze",
    )

    def __init__(
        self,
        flagship_min_headroom_usd: float = 0.0200,
        balanced_min_headroom_usd: float = 0.0020,
    ) -> None:
        self._flagship_min_headroom = flagship_min_headroom_usd
        self._balanced_min_headroom = balanced_min_headroom_usd
        self._lock = threading.Lock()
        self._route_counts: dict[str, int] = {
            "economy": 0,
            "balanced": 0,
            "flagship": 0,
        }
        self._downgrade_count: int = 0

    def classify_complexity(self, query: str, context_chunks_count: int = 0) -> str:
        lowered = query.lower().strip()
        word_count = len(lowered.split())
        has_complex_cue = any(kw in lowered for kw in self._COMPLEX_KEYWORDS)

        if word_count >= 18 or has_complex_cue or context_chunks_count >= 5:
            return "complex"
        if word_count >= 7 or context_chunks_count >= 3:
            return "moderate"
        return "simple"

    def route(
        self,
        tenant_id: str,
        query: str,
        tenant_tier: str = "pro",
        remaining_budget_usd: float | None = None,
        context_chunks_count: int = 0,
        preferred_profile: str | None = None,
    ) -> RoutingDecision:
        complexity = self.classify_complexity(query, context_chunks_count)
        normalized_tier = (tenant_tier or "pro").strip().lower()

        if preferred_profile and preferred_profile.strip().lower() in MODEL_PROFILES:
            desired = preferred_profile.strip().lower()
            reason = f"explicit_override:{desired}"
        elif normalized_tier == "free":
            desired = "economy"
            reason = "free_tier_policy"
        elif normalized_tier == "enterprise" and complexity == "complex":
            desired = "flagship"
            reason = "enterprise_complex_query"
        elif complexity in ("moderate", "complex"):
            desired = "balanced"
            reason = f"{normalized_tier}_{complexity}_query"
        else:
            desired = "economy"
            reason = "simple_query_optimization"

        selected = desired
        downgraded = False

        # Cost-aware fallback when remaining tenant budget headroom is constrained
        if remaining_budget_usd is not None:
            if (
                selected == "flagship"
                and remaining_budget_usd < self._flagship_min_headroom
            ):
                if remaining_budget_usd >= self._balanced_min_headroom:
                    selected = "balanced"
                    reason = "budget_headroom_downgrade:flagship->balanced"
                else:
                    selected = "economy"
                    reason = "budget_headroom_downgrade:flagship->economy"
                downgraded = True
            elif (
                selected == "balanced"
                and remaining_budget_usd < self._balanced_min_headroom
            ):
                selected = "economy"
                reason = "budget_headroom_downgrade:balanced->economy"
                downgraded = True

        profile = MODEL_PROFILES[selected]
        with self._lock:
            self._route_counts[selected] = self._route_counts.get(selected, 0) + 1
            if downgraded:
                self._downgrade_count += 1

        return RoutingDecision(
            tenant_id=tenant_id,
            selected_profile=profile.name,
            model_id=profile.model_id,
            complexity=complexity,
            downgraded=downgraded,
            reason=reason,
            pricing=profile.pricing,
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = sum(self._route_counts.values())
            return {
                "total_routed": total,
                "downgrade_count": self._downgrade_count,
                "by_profile": dict(self._route_counts),
                "profiles": {
                    name: {
                        "model_id": prof.model_id,
                        "prompt_per_1k_usd": prof.pricing.prompt_per_1k_usd,
                        "completion_per_1k_usd": prof.pricing.completion_per_1k_usd,
                        "max_context_tokens": prof.max_context_tokens,
                    }
                    for name, prof in MODEL_PROFILES.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._route_counts = {"economy": 0, "balanced": 0, "flagship": 0}
            self._downgrade_count = 0
