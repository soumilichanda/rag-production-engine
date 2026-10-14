from pathlib import Path

import numpy as np

from app.ingestion.chunker import RecursiveTokenChunker
from app.retrieval.persistence import IndexPersistenceManager
from app.retrieval.vector_store import InMemoryVectorStore


def test_chunker_overlap_and_hash_invariants():
    chunker = RecursiveTokenChunker(chunk_size=50, chunk_overlap=15)
    text = (
        "Retrieval-Augmented Generation enhances LLM accuracy. "
        "Deterministic chunking prevents semantic fragmentation. "
        "Persistent indexing ensures sub-millisecond cold start reloads."
    )
    chunks = chunker.split_text(text, base_metadata={"source": "rag_spec.md"})

    assert len(chunks) > 1
    # Check deterministic SHA-256 chunk_id prefix and length
    for chunk in chunks:
        assert chunk.chunk_id.startswith("chk_")
        assert len(chunk.chunk_id) == 20  # "chk_" + 16 hex chars
        assert chunk.metadata["source"] == "rag_spec.md"


def test_chunker_empty_input_handling():
    chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\n  ") == []


def test_persistence_serialization_round_trip(tmp_path: Path):
    store = InMemoryVectorStore(dimension=4)
    chunker = RecursiveTokenChunker(chunk_size=80, chunk_overlap=20)
    raw_docs = chunker.split_text(
        "Production ML inference pipelines require deterministic batching and state persistence."
    )
    dummy_embeddings = np.array(
        [[1.0, 0.0, 0.0, 0.0] for _ in raw_docs], dtype=np.float32
    )

    store.add_documents(raw_docs, dummy_embeddings)

    # Serialize to isolated temp dir
    mgr = IndexPersistenceManager(storage_dir=str(tmp_path / "index_test"))
    saved_chunks, saved_embeddings = mgr.save(store)
    assert saved_chunks == len(raw_docs)
    assert saved_embeddings == len(raw_docs)

    # Hydrate into clean store
    new_store = InMemoryVectorStore(dimension=4)
    loaded_count = mgr.load(new_store)

    assert loaded_count == len(raw_docs)
    assert len(new_store.chunks) == len(raw_docs)
    assert new_store.embeddings.shape == store.embeddings.shape
    assert np.allclose(new_store.embeddings, store.embeddings)
    assert new_store.chunks[0].chunk_id == store.chunks[0].chunk_id
