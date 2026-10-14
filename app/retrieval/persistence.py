"""
app/retrieval/persistence.py
Disk serialization and cold-start rehydration for normalized vector indexes.
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from app.schemas.payloads import DocumentChunk
from app.retrieval.vector_store import InMemoryVectorStore


class IndexPersistenceManager:
    """
    Serializes and hydrates InMemoryVectorStore state using compressed NumPy (.npz)
    for embedding vectors and newline-delimited JSON (.jsonl) for chunk metadata.
    """

    def __init__(self, storage_dir: str = "data/index"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.matrix_path = self.storage_dir / "embeddings.npz"
        self.payload_path = self.storage_dir / "chunks.jsonl"

    def save(self, store: InMemoryVectorStore) -> Tuple[int, int]:
        """Serializes current in-memory chunks and embeddings to disk."""
        # 1. Save normalized embeddings matrix
        np.savez_compressed(self.matrix_path, embeddings=store.embeddings)

        # 2. Save document payload list to JSONL
        with open(self.payload_path, "w", encoding="utf-8") as f:
            for chunk in store.chunks:
                f.write(chunk.model_dump_json() + "\n")

        return len(store.chunks), store.embeddings.shape[0]

    def load(self, store: InMemoryVectorStore) -> int:
        """Hydrates store from disk if persistence artifacts exist."""
        if not self.matrix_path.exists() or not self.payload_path.exists():
            return 0

        # 1. Load document payloads
        hydrated_chunks: List[DocumentChunk] = []
        with open(self.payload_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    hydrated_chunks.append(DocumentChunk.model_validate_json(line))

        # 2. Load embeddings matrix
        with np.load(self.matrix_path) as data:
            hydrated_embeddings = data["embeddings"]

        if len(hydrated_chunks) != hydrated_embeddings.shape[0]:
            raise ValueError("Corrupted index state: chunk payload count does not match matrix rows")

        store.chunks = hydrated_chunks
        store.embeddings = hydrated_embeddings.astype(np.float32)
        return len(store.chunks)
