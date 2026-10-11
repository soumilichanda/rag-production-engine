"""
app/core/reranker.py
Semantic Cross-Encoder scoring and context reranking harness.
"""

from typing import List
import numpy as np
from app.schemas.payloads import RetrievedChunk


class ExactTokenOverlapReranker:
    """
    Lightweight deterministic reranker scoring exact token term-frequency
    and lexical overlap to break vector similarity ties and re-rank candidate chunks.
    """

    def __init__(self, token_weight: float = 0.35):
        self.token_weight = token_weight

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not candidates:
            return []

        query_tokens = set(query.lower().split())
        reranked = []

        for chunk in candidates:
            chunk_tokens = set(chunk.text.lower().split())
            if not query_tokens:
                overlap_ratio = 0.0
            else:
                overlap_ratio = len(query_tokens & chunk_tokens) / len(query_tokens)

            # Combined score: Base vector similarity + token lexical alignment
            composite_score = float((1.0 - self.token_weight) * chunk.similarity_score + self.token_weight * overlap_ratio)

            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=round(composite_score, 4),
                    metadata={**chunk.metadata, "lexical_overlap": round(overlap_ratio, 3)}
                )
            )

        # Sort descending by composite relevance score
        reranked.sort(key=lambda item: item.similarity_score, reverse=True)
        return reranked[:top_k]
