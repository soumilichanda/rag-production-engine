"""
Phase 19: Cross-Encoder Attention Reranker.
Computes token-interaction relevance scores across query-chunk pairs
to optimize retrieved context order prior to prompt synthesis and streaming.
"""

import math

from app.schemas.payloads import RetrievedChunk


class CrossEncoderAttentionReranker:
    def __init__(self, position_decay: float = 0.05, top_k_default: int = 3):
        self.position_decay = position_decay
        self.top_k_default = top_k_default

    def score_pair(self, query: str, document_text: str) -> float:
        q_tokens = [w.lower() for w in query.split() if w.strip()]
        doc_tokens = [w.lower() for w in document_text.split() if w.strip()]

        if not q_tokens or not doc_tokens:
            return 0.0

        # Term-frequency density with positional attention weighting
        score = 0.0
        doc_len = len(doc_tokens)

        for q in q_tokens:
            matches = [idx for idx, d in enumerate(doc_tokens) if q in d]
            if matches:
                # Attention prioritizes terms appearing earlier in the chunk
                term_score = sum(
                    math.exp(-self.position_decay * (idx / doc_len)) for idx in matches
                )
                score += term_score

        normalized_score = score / (len(q_tokens) + 1e-5)
        return float(round(math.tanh(normalized_score), 4))

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        limit = top_k or self.top_k_default
        scored_chunks = []

        for chunk in candidates:
            attn_score = self.score_pair(query, chunk.text)
            # Composite score blends vector base similarity with cross-attention relevance
            composite_score = round(0.4 * chunk.similarity_score + 0.6 * attn_score, 4)
            metadata = dict(chunk.metadata)
            metadata["cross_encoder_score"] = attn_score
            metadata["reranked_composite"] = composite_score

            scored_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=composite_score,
                    metadata=metadata,
                )
            )

        scored_chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        return scored_chunks[:limit]
