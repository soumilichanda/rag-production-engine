"""
Phase 44: Contextual Compression & Semantic Deduplication Pipeline.
Prunes near-duplicate sentence spans across multi-pass Self-RAG chunks via pairwise
cosine similarity and packs highest-salience spans within a strict token budget.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass
class SentenceSpan:
    span_id: str
    chunk_id: str
    text: str
    token_count: int
    salience: float
    original_index: int
    metadata: dict[str, Any]


class SemanticContextCompressor:
    """
    Compresses multi-pass retrieved chunks by:
    1. Splitting chunks into atomic sentence spans with provenance.
    2. Deduplicating redundant spans via pairwise cosine similarity.
    3. Solving 0/1 knapsack over token budgets to maximize retained salience.
    """

    def __init__(self, dedup_threshold: float = 0.88, default_token_budget: int = 180):
        self.dedup_threshold = dedup_threshold
        self.default_token_budget = default_token_budget
        self._lock = RLock()
        self._total_requests = 0
        self._total_raw_tokens = 0
        self._total_compressed_tokens = 0
        self._total_deduped_spans = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @classmethod
    def _cosine_similarity(cls, text_a: str, text_b: str) -> float:
        tokens_a = cls._tokenize(text_a)
        tokens_b = cls._tokenize(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        counts_a = Counter(tokens_a)
        counts_b = Counter(tokens_b)
        shared = set(counts_a.keys()) & set(counts_b.keys())
        dot = sum(counts_a[t] * counts_b[t] for t in shared)
        norm_a = math.sqrt(sum(v * v for v in counts_a.values()))
        norm_b = math.sqrt(sum(v * v for v in counts_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _score_salience(
        self, query_tokens: set[str], span_text: str, base_score: float
    ) -> float:
        span_tokens = self._tokenize(span_text)
        if not span_tokens:
            return 0.0
        if not query_tokens:
            lexical = 0.25
        else:
            overlap = len(set(span_tokens) & query_tokens)
            lexical = overlap / len(query_tokens)
        return round(0.45 * max(0.0, float(base_score)) + 0.55 * lexical, 6)

    def compress(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        max_token_budget: int | None = None,
        dedup_threshold: float | None = None,
    ) -> dict[str, Any]:
        budget = (
            max_token_budget
            if max_token_budget is not None
            else self.default_token_budget
        )
        threshold = (
            dedup_threshold if dedup_threshold is not None else self.dedup_threshold
        )
        query_tokens = set(self._tokenize(query))

        raw_spans: list[SentenceSpan] = []
        raw_tokens = 0
        idx = 0

        for c_idx, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id", f"chunk_{c_idx}"))
            text = str(chunk.get("text", "")).strip()
            base_score = float(chunk.get("similarity_score", chunk.get("score", 0.5)))
            meta = dict(chunk.get("metadata", {}))
            if not text:
                continue

            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
            ]
            for s_idx, sent in enumerate(sentences):
                tok_len = max(1, len(self._tokenize(sent)))
                raw_tokens += tok_len
                salience = self._score_salience(query_tokens, sent, base_score)
                raw_spans.append(
                    SentenceSpan(
                        span_id=f"{chunk_id}:s{s_idx}",
                        chunk_id=chunk_id,
                        text=sent,
                        token_count=tok_len,
                        salience=salience,
                        original_index=idx,
                        metadata=meta,
                    )
                )
                idx += 1

        # Sort descending by salience so deduplication keeps highest-value span
        ranked_spans = sorted(raw_spans, key=lambda s: s.salience, reverse=True)
        unique_spans: list[SentenceSpan] = []
        deduped_count = 0

        for candidate in ranked_spans:
            is_dup = any(
                self._cosine_similarity(candidate.text, kept.text) >= threshold
                for kept in unique_spans
            )
            if is_dup:
                deduped_count += 1
            else:
                unique_spans.append(candidate)

        # 0/1 Knapsack DP maximizing salience within token budget
        n = len(unique_spans)
        cap = max(0, int(budget))
        dp = [[0.0] * (cap + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            w = unique_spans[i - 1].token_count
            v = unique_spans[i - 1].salience + 1e-4
            for b in range(cap + 1):
                dp[i][b] = dp[i - 1][b]
                if w <= b and dp[i - 1][b - w] + v > dp[i][b]:
                    dp[i][b] = dp[i - 1][b - w] + v

        selected: list[SentenceSpan] = []
        rem = cap
        for i in range(n, 0, -1):
            if dp[i][rem] != dp[i - 1][rem]:
                span = unique_spans[i - 1]
                selected.append(span)
                rem -= span.token_count

        # Restore original narrative order
        selected.sort(key=lambda s: s.original_index)
        compressed_tokens = sum(s.token_count for s in selected)
        compressed_text = " ".join(s.text for s in selected)
        compression_ratio = (
            round(1.0 - (compressed_tokens / raw_tokens), 4) if raw_tokens > 0 else 0.0
        )

        with self._lock:
            self._total_requests += 1
            self._total_raw_tokens += raw_tokens
            self._total_compressed_tokens += compressed_tokens
            self._total_deduped_spans += deduped_count

        return {
            "query": query,
            "compressed_text": compressed_text,
            "selected_spans": [
                {
                    "span_id": s.span_id,
                    "chunk_id": s.chunk_id,
                    "text": s.text,
                    "token_count": s.token_count,
                    "salience": round(s.salience, 4),
                    "metadata": s.metadata,
                }
                for s in selected
            ],
            "metrics": {
                "raw_span_count": len(raw_spans),
                "deduplicated_spans_removed": deduped_count,
                "selected_span_count": len(selected),
                "raw_tokens": raw_tokens,
                "compressed_tokens": compressed_tokens,
                "token_budget": cap,
                "compression_ratio": compression_ratio,
            },
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            avg_saving = (
                round(1.0 - (self._total_compressed_tokens / self._total_raw_tokens), 4)
                if self._total_raw_tokens > 0
                else 0.0
            )
            return {
                "total_requests": self._total_requests,
                "total_raw_tokens": self._total_raw_tokens,
                "total_compressed_tokens": self._total_compressed_tokens,
                "total_deduped_spans": self._total_deduped_spans,
                "average_token_reduction_ratio": avg_saving,
                "dedup_threshold": self.dedup_threshold,
                "default_token_budget": self.default_token_budget,
            }
