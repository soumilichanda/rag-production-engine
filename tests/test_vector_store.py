import numpy as np

from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.payloads import DocumentChunk


def test_vector_store_cosine_ranking():
    dim = 4
    store = InMemoryVectorStore(dimension=dim)

    docs = [
        DocumentChunk(chunk_id="chunk_ml", text="FastAPI inference microservice deployment", metadata={"domain": "mlops"}),
        DocumentChunk(chunk_id="chunk_dsa", text="Topological sort Kahn algorithm implementation", metadata={"domain": "dsa"}),
        DocumentChunk(chunk_id="chunk_net", text="Low-latency WebSockets client server", metadata={"domain": "systems"}),
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    store.add_documents(docs, embeddings)

    # Query aligned with chunk_ml
    query_vec = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
    results, latency_ms = store.query(query_vec, top_k=2, threshold=0.1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_ml"
    assert results[0].similarity_score > 0.9
    assert latency_ms >= 0.0


def test_empty_store_query():
    store = InMemoryVectorStore(dimension=8)
    query_vec = np.ones(8, dtype=np.float32)
    results, latency = store.query(query_vec)
    assert results == []
    assert latency >= 0.0
