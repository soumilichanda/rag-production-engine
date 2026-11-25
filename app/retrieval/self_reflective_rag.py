"""
Phase 43: Self-Reflective Critique Loop & Iterative Retrieval Refinement (Self-RAG).
Evaluates retrieved context sufficiency against query facets and executes bounded
follow-up retrieval refinement passes when facet coverage falls below threshold.
"""

import inspect
import re
import threading
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "compare",
    "could",
    "describe",
    "do",
    "does",
    "explain",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "then",
    "these",
    "this",
    "to",
    "versus",
    "vs",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
}


@dataclass
class CritiquePassResult:
    iteration: int
    retrieval_query: str
    coverage_score: float
    covered_facets: list[str]
    missing_facets: list[str]
    chunks_added: int
    sufficient: bool


@dataclass
class SelfRAGExecutionResult:
    query: str
    final_coverage_score: float
    initial_coverage_score: float
    total_passes: int
    refinement_triggered: bool
    sufficient: bool
    facets: list[str]
    missing_facets: list[str]
    critique_history: list[dict[str, Any]]
    results: list[dict[str, Any]]


class SelfReflectiveCritiqueLoop:
    """Critiques context coverage across query facets and iteratively refines retrieval."""

    def __init__(
        self,
        coverage_threshold: float = 0.75,
        max_iterations: int = 3,
    ) -> None:
        self.coverage_threshold = coverage_threshold
        self.max_iterations = max(1, max_iterations)
        self._lock = threading.RLock()
        self._total_executions = 0
        self._refinement_triggers = 0
        self._total_passes = 0
        self._cumulative_coverage_gain = 0.0

    @staticmethod
    def extract_facets(query: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_-]{2,}", query.lower())
        facets: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok not in _STOPWORDS and tok not in seen:
                seen.add(tok)
                facets.append(tok)
        return facets

    def critique_context(
        self, facets: list[str], chunks: list[dict[str, Any]]
    ) -> tuple[float, list[str], list[str]]:
        if not facets:
            return 1.0, [], []

        corpus = " ".join(str(c.get("text", "")).lower() for c in chunks)
        corpus_tokens = set(re.findall(r"[a-zA-Z0-9_-]{2,}", corpus))

        covered: list[str] = []
        missing: list[str] = []
        for facet in facets:
            if facet in corpus_tokens or facet in corpus:
                covered.append(facet)
            else:
                missing.append(facet)

        coverage = round(len(covered) / len(facets), 4)
        return coverage, covered, missing

    @staticmethod
    def build_refinement_query(original_query: str, missing_facets: list[str]) -> str:
        focus = " ".join(missing_facets[:5])
        return f"{focus} details specifications ({original_query.strip()})".strip()

    async def execute(
        self,
        query: str,
        retrieve_fn: Callable[
            [str, int], list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]
        ],
        top_k: int = 4,
    ) -> SelfRAGExecutionResult:
        facets = self.extract_facets(query)
        merged_by_id: dict[str, dict[str, Any]] = {}
        critique_history: list[CritiquePassResult] = []

        current_query = query.strip()
        initial_coverage = 0.0
        final_coverage = 0.0
        covered: list[str] = []
        missing: list[str] = list(facets)

        for iteration in range(1, self.max_iterations + 1):
            raw_res = retrieve_fn(current_query, top_k)
            fetched = await raw_res if inspect.isawaitable(raw_res) else raw_res

            added = 0
            for idx, item in enumerate(fetched):
                cid = str(item.get("chunk_id") or f"pass{iteration}_chunk_{idx}")
                score = float(item.get("score", item.get("similarity_score", 0.0)))
                if cid not in merged_by_id:
                    merged_by_id[cid] = {
                        "chunk_id": cid,
                        "text": str(item.get("text", "")),
                        "score": round(score, 4),
                        "retrieved_pass": iteration,
                        "metadata": dict(item.get("metadata") or {}),
                    }
                    added += 1
                else:
                    existing_score = float(merged_by_id[cid]["score"])
                    merged_by_id[cid]["score"] = round(
                        max(existing_score, score) + 0.05, 4
                    )

            current_chunks = list(merged_by_id.values())
            coverage, covered, missing = self.critique_context(facets, current_chunks)
            if iteration == 1:
                initial_coverage = coverage
            final_coverage = coverage

            sufficient = coverage >= self.coverage_threshold or not missing
            critique_history.append(
                CritiquePassResult(
                    iteration=iteration,
                    retrieval_query=current_query,
                    coverage_score=coverage,
                    covered_facets=covered,
                    missing_facets=missing,
                    chunks_added=added,
                    sufficient=sufficient,
                )
            )

            if sufficient or iteration == self.max_iterations:
                break

            current_query = self.build_refinement_query(query, missing)

        ranked_chunks = sorted(
            merged_by_id.values(),
            key=lambda c: float(c["score"]),
            reverse=True,
        )[:top_k]

        total_passes = len(critique_history)
        refinement_triggered = total_passes > 1
        coverage_gain = max(0.0, final_coverage - initial_coverage)

        with self._lock:
            self._total_executions += 1
            if refinement_triggered:
                self._refinement_triggers += 1
            self._total_passes += total_passes
            self._cumulative_coverage_gain += coverage_gain

        return SelfRAGExecutionResult(
            query=query,
            final_coverage_score=final_coverage,
            initial_coverage_score=initial_coverage,
            total_passes=total_passes,
            refinement_triggered=refinement_triggered,
            sufficient=final_coverage >= self.coverage_threshold or not missing,
            facets=facets,
            missing_facets=missing,
            critique_history=[asdict(p) for p in critique_history],
            results=ranked_chunks,
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            avg_passes = (
                round(self._total_passes / self._total_executions, 3)
                if self._total_executions
                else 0.0
            )
            avg_gain = (
                round(self._cumulative_coverage_gain / self._total_executions, 4)
                if self._total_executions
                else 0.0
            )
            return {
                "total_executions": self._total_executions,
                "refinement_triggers": self._refinement_triggers,
                "average_passes": avg_passes,
                "average_coverage_gain": avg_gain,
                "coverage_threshold": self.coverage_threshold,
                "max_iterations": self.max_iterations,
            }
