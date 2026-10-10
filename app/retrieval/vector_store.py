"""
Vector store retrieval engine using L2-normalized cosine similarity.
"""

import time
from typing import List, Tuple
import numpy as np
from app.schemas.payloads import DocumentChunk, RetrievedChunk


class InMemoryVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add_documents(self, documents: List[DocumentChunk], embeddings: np.ndarray) -> int:
        if len(documents) != len(embeddings):
            raise ValueError("Document count must match embedding matrix rows")

        # L2-normalize vectors to unit hypersphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_embeddings = embeddings / norms

        self.chunks.extend(documents)
        if self.embeddings.shape[0] == 0:
            self.embeddings = normalized_embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_embeddings]).astype(np.float32)

        return len(documents)

    def query(self, query_embedding: np.ndarray, top_k: int = 3, threshold: float = 0.0) -> Tuple[List[RetrievedChunk], float]:
        start_time = time.perf_counter()

        if self.embeddings.shape[0] == 0:
            latency = (time.perf_counter() - start_time) * 1000.0
            return [], latency

        # L2-normalize query vector
        norm = np.linalg.norm(query_embedding)
        norm = 1e-12 if norm == 0 else norm
        norm_query = (query_embedding / norm).astype(np.float32)

        # Dot product of unit vectors equals cosine similarity
        similarities = np.dot(self.embeddings, norm_query)

        # Extract top-k indices descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                chunk = self.chunks[idx]
                results.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        similarity_score=round(score, 4),
                        metadata=chunk.metadata,
                    )
                )

        latency = (time.perf_counter() - start_time) * 1000.0
        return results, latency
