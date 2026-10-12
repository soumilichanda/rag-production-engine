"""
app/retrieval/hybrid_search.py
Reciprocal Rank Fusion (RRF) combining dense vector similarity with sparse token matches.
"""

from typing import List, Dict
from app.schemas.payloads import RetrievedChunk


class HybridRRFSearcher:
    """
    Blends dense vector search rankings and sparse keyword rankings
    using Reciprocal Rank Fusion (RRF) with constant k=60.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        dense_results: List[RetrievedChunk],
        sparse_query: str,
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        if not dense_results:
            return []

        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # 1. Dense reciprocal rank contribution
        for rank, chunk in enumerate(dense_results):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # 2. Sparse lexical rank contribution
        query_terms = set(sparse_query.lower().split())
        lexical_ranked = sorted(
            dense_results,
            key=lambda c: len(set(c.text.lower().split()) & query_terms),
            reverse=True
        )

        for rank, chunk in enumerate(lexical_ranked):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # 3. Sort by fused RRF score
        fused_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        fused_results: List[RetrievedChunk] = []
        for cid in fused_ids[:top_k]:
            original = chunk_map[cid]
            fused_score = round(scores[cid], 5)
            fused_results.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    similarity_score=fused_score,
                    metadata={**original.metadata, "rrf_fused": True}
                )
            )

        return fused_results
