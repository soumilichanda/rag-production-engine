"""
app/retrieval/query_planner.py
Phase 42: Adaptive Query Decomposition & Sub-Question Execution Planner.
Splits complex multi-hop analytical queries into atomic sub-queries,
executes parallel retrieval across branches, and synthesizes merged context.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubQueryStep:
    step_id: int
    sub_query: str
    intent: str = "factual_lookup"


@dataclass
class ExecutionPlan:
    original_query: str
    strategy: str  # "direct" | "decomposed"
    complexity_score: float
    steps: list[SubQueryStep] = field(default_factory=list)


class AdaptiveQueryDecomposer:
    """
    Deterministically analyzes query structure and splits multi-hop or
    comparative analytical questions into bounded atomic sub-questions.
    """

    _COMPARISON_PATTERN = re.compile(
        r"\b(?:compare|versus|vs\.?|difference between|relationship between)\b",
        re.IGNORECASE,
    )
    _CONJUNCTION_SPLIT = re.compile(
        r"\s*(?:\band\b|\bwhile\b|\bwhereas\b|\balso\b|;|\?\s+)\s*",
        re.IGNORECASE,
    )

    def __init__(self, max_sub_queries: int = 4) -> None:
        self.max_sub_queries = max(1, max_sub_queries)

    def build_plan(self, query: str) -> ExecutionPlan:
        cleaned = " ".join(query.strip().split())
        if not cleaned:
            return ExecutionPlan(
                original_query=query,
                strategy="direct",
                complexity_score=0.0,
                steps=[SubQueryStep(step_id=1, sub_query="", intent="empty")],
            )

        is_comparative = bool(self._COMPARISON_PATTERN.search(cleaned))
        raw_parts = [
            p.strip(" ?.,;")
            for p in self._CONJUNCTION_SPLIT.split(cleaned)
            if p.strip(" ?.,;")
        ]

        sub_queries: list[str] = []
        if is_comparative and len(raw_parts) >= 2:
            for part in raw_parts[: self.max_sub_queries]:
                sub_queries.append(f"Core specifications and behavior of {part}")
            sub_queries.append(cleaned)
        elif len(raw_parts) >= 2 and all(len(p.split()) >= 2 for p in raw_parts):
            sub_queries = raw_parts[: self.max_sub_queries]
        else:
            sub_queries = [cleaned]

        # Deduplicate preserving order while capping to max_sub_queries
        deduped: list[str] = []
        seen = set()
        for sq in sub_queries:
            norm = sq.lower()
            if norm not in seen:
                seen.add(norm)
                deduped.append(sq)
        deduped = deduped[: self.max_sub_queries]

        strategy = "decomposed" if len(deduped) > 1 else "direct"
        complexity = min(
            1.0, round(0.25 * len(deduped) + (0.2 if is_comparative else 0.0), 3)
        )

        steps = [
            SubQueryStep(
                step_id=idx + 1,
                sub_query=sq,
                intent="comparative_branch"
                if is_comparative
                else ("sub_hop" if len(deduped) > 1 else "direct_lookup"),
            )
            for idx, sq in enumerate(deduped)
        ]

        return ExecutionPlan(
            original_query=cleaned,
            strategy=strategy,
            complexity_score=complexity,
            steps=steps,
        )


class SubQuestionExecutionPlanner:
    """
    Executes decomposed sub-questions concurrently via asyncio.gather,
    fuses multi-branch candidate chunks with coverage-weighted RRF, and
    records planner execution telemetry.
    """

    def __init__(
        self, decomposer: AdaptiveQueryDecomposer | None = None, rrf_k: int = 60
    ) -> None:
        self.decomposer = decomposer or AdaptiveQueryDecomposer()
        self.rrf_k = rrf_k
        self._lock = threading.RLock()
        self._total_plans = 0
        self._decomposed_plans = 0
        self._direct_plans = 0
        self._total_sub_queries = 0

    async def execute(
        self,
        query: str,
        retriever_fn: Callable[[str, int], Awaitable[list[dict[str, Any]]]],
        top_k: int = 5,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        plan = self.decomposer.build_plan(query)

        branch_results = await asyncio.gather(
            *(retriever_fn(step.sub_query, top_k) for step in plan.steps)
        )

        rrf_scores: dict[str, float] = defaultdict(float)
        coverage_counts: dict[str, int] = defaultdict(int)
        chunk_store: dict[str, dict[str, Any]] = {}

        for step, candidates in zip(plan.steps, branch_results):
            for rank, item in enumerate(candidates, start=1):
                cid = str(item["chunk_id"])
                chunk_store[cid] = dict(item)
                rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
                coverage_counts[cid] += 1

        # Boost chunks corroborated across multiple sub-questions
        final_scores: dict[str, float] = {}
        for cid, base_rrf in rrf_scores.items():
            coverage_bonus = 1.0 + 0.15 * (coverage_counts[cid] - 1)
            final_scores[cid] = round(base_rrf * coverage_bonus, 6)

        ranked_ids = sorted(
            final_scores.keys(), key=lambda c: final_scores[c], reverse=True
        )[:top_k]

        merged_chunks: list[dict[str, Any]] = []
        for cid in ranked_ids:
            entry = chunk_store[cid]
            entry["planner_score"] = final_scores[cid]
            entry["sub_query_hits"] = coverage_counts[cid]
            merged_chunks.append(entry)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        with self._lock:
            self._total_plans += 1
            if plan.strategy == "decomposed":
                self._decomposed_plans += 1
            else:
                self._direct_plans += 1
            self._total_sub_queries += len(plan.steps)

        summary_snippets = [
            c.get("text", "").strip() for c in merged_chunks[:3] if c.get("text")
        ]
        synthesized_context = " | ".join(summary_snippets)

        return {
            "original_query": plan.original_query,
            "strategy": plan.strategy,
            "complexity_score": plan.complexity_score,
            "sub_queries": [
                {"step_id": s.step_id, "sub_query": s.sub_query, "intent": s.intent}
                for s in plan.steps
            ],
            "merged_results": merged_chunks,
            "synthesized_context": synthesized_context,
            "latency_ms": latency_ms,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            avg_sub = (
                round(self._total_sub_queries / self._total_plans, 2)
                if self._total_plans > 0
                else 0.0
            )
            return {
                "total_plans": self._total_plans,
                "decomposed_plans": self._decomposed_plans,
                "direct_plans": self._direct_plans,
                "total_sub_queries": self._total_sub_queries,
                "avg_sub_queries_per_plan": avg_sub,
            }
