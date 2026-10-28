"""
app/retrieval/fusion.py
Reciprocal Rank Fusion (RRF) aggregator for multi-query candidate chunk lists.
"""

from collections import defaultdict

from app.schemas.payloads import RetrievedChunk


class RRFAggregator:
    """
    Merges multiple ranked retrieval lists using Reciprocal Rank Fusion (RRF).
    RRF(d) = sum(1 / (k + rank))
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self, ranked_lists: list[list[RetrievedChunk]], top_k: int = 5
    ) -> list[RetrievedChunk]:
        if not ranked_lists:
            return []

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                chunk_id = chunk.chunk_id
                chunk_map[chunk_id] = chunk
                rrf_scores[chunk_id] += 1.0 / (self.k + rank)

        sorted_chunks = sorted(
            chunk_map.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        fused_results: list[RetrievedChunk] = []
        for cid in sorted_chunks[:top_k]:
            original = chunk_map[cid]
            fused_results.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    similarity_score=round(rrf_scores[cid], 5),
                    metadata={
                        **original.metadata,
                        "fusion_strategy": "rrf",
                        "rrf_k": self.k,
                    },
                )
            )

        return fused_results
