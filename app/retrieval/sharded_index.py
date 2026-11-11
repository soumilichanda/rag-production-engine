"""
app/retrieval/sharded_index.py
Lock-striped partitioned vector index supporting high-concurrency ingestion,
consistent-hash ring routing, and scatter-gather retrieval.
"""

import threading

import numpy as np

from app.retrieval.consistent_hash import ConsistentHashRing
from app.schemas.payloads import DocumentChunk, RetrievedChunk


class ShardPartition:
    def __init__(self, shard_id: int, dimension: int):
        self.shard_id = shard_id
        self.dimension = dimension
        self.lock = threading.RLock()
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def insert(self, chunk: DocumentChunk, embedding: np.ndarray) -> None:
        with self.lock:
            norm = np.linalg.norm(embedding)
            norm_vec = embedding / norm if norm > 0 else embedding
            self.chunks.append(chunk)
            if self.embeddings.shape[0] == 0:
                self.embeddings = np.expand_dims(norm_vec, axis=0).astype(np.float32)
            else:
                self.embeddings = np.vstack(
                    [self.embeddings, norm_vec.astype(np.float32)]
                )

    def query(
        self, query_vec: np.ndarray, top_k: int, threshold: float
    ) -> list[RetrievedChunk]:
        with self.lock:
            if self.embeddings.shape[0] == 0:
                return []

            norm = np.linalg.norm(query_vec)
            q_norm = query_vec / norm if norm > 0 else query_vec
            scores = np.dot(self.embeddings, q_norm)

            candidates: list[RetrievedChunk] = []
            for idx, score in enumerate(scores):
                if score >= threshold:
                    chunk = self.chunks[idx]
                    candidates.append(
                        RetrievedChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            similarity_score=float(score),
                            metadata={**chunk.metadata, "shard_id": self.shard_id},
                        )
                    )
            return candidates


class ShardedVectorIndex:
    def __init__(
        self, dimension: int = 384, num_shards: int = 8, virtual_replicas: int = 64
    ):
        self.dimension = dimension
        self.num_shards = num_shards
        shard_ids = list(range(num_shards))
        self.hash_ring = ConsistentHashRing(
            shards=shard_ids, virtual_replicas=virtual_replicas
        )
        self.shards: dict[int, ShardPartition] = {
            i: ShardPartition(shard_id=i, dimension=dimension) for i in shard_ids
        }

    def _route(self, chunk_id: str) -> ShardPartition:
        shard_id = self.hash_ring.get_shard(chunk_id)
        return self.shards[shard_id]

    def add_document(self, chunk: DocumentChunk, embedding: np.ndarray) -> int:
        shard = self._route(chunk.chunk_id)
        shard.insert(chunk, embedding)
        return shard.shard_id

    def search(
        self, query_vec: np.ndarray, top_k: int = 5, threshold: float = 0.0
    ) -> list[RetrievedChunk]:
        all_results: list[RetrievedChunk] = []
        for shard in self.shards.values():
            all_results.extend(shard.query(query_vec, top_k=top_k, threshold=threshold))

        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:top_k]

    def total_documents(self) -> int:
        return sum(len(s.chunks) for s in self.shards.values())
