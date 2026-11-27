"""Async pytest suite for Phase 45: Adaptive Semantic Chunking & Structural Boundary Splitting."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.ingestion.semantic_chunker import AdaptiveSemanticChunker
from app.main import app


def test_structural_heading_boundary_split():
    doc = (
        "# Kubernetes Pod Scheduling\n"
        "The kube-scheduler watches for newly created pods with no assigned node. "
        "It filters nodes using resource requests and affinity rules.\n\n"
        "# PostgreSQL WAL Replication\n"
        "Write-ahead logging streams binary change records to standby replicas. "
        "Streaming replication ensures point-in-time recovery durability."
    )
    chunker = AdaptiveSemanticChunker(max_chunk_tokens=200, min_chunk_tokens=10, similarity_threshold=0.2)
    chunks = chunker.chunk_document("doc_infra_01", doc)

    assert len(chunks) >= 2
    assert "kube-scheduler" in chunks[0].text
    assert "Write-ahead logging" in chunks[1].text
    assert chunks[0].boundary_reason == "structural"


def test_semantic_topical_shift_detection():
    def topic_embedder(text: str) -> list[float]:
        if "neural" in text.lower() or "gradient" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    doc = (
        "Neural networks optimize weights via stochastic gradient descent. "
        "Gradient clipping prevents exploding updates during deep backpropagation. "
        "Sourdough fermentation relies on wild lactobacilli and yeast cultures. "
        "High hydration dough develops open crumb structure during steam baking."
    )
    chunker = AdaptiveSemanticChunker(
        max_chunk_tokens=300,
        min_chunk_tokens=20,
        window_size=0,
        similarity_threshold=0.5,
        embed_fn=topic_embedder,
    )
    chunks = chunker.chunk_document("doc_mixed_02", doc)

    assert len(chunks) == 2
    assert chunks[0].boundary_reason == "semantic_shift"
    assert "Neural networks" in chunks[0].text
    assert "Sourdough fermentation" in chunks[1].text


def test_max_token_bound_enforcement():
    sentences = [f"Sentence number {i} contains telemetry metrics for service mesh routing." for i in range(25)]
    doc = " ".join(sentences)
    chunker = AdaptiveSemanticChunker(max_chunk_tokens=45, min_chunk_tokens=10, similarity_threshold=0.0)
    chunks = chunker.chunk_document("doc_long_03", doc)

    assert len(chunks) > 2
    assert all(c.token_estimate <= 45 for c in chunks)
    assert any(c.boundary_reason == "max_tokens" for c in chunks)


@pytest.mark.asyncio
async def test_semantic_ingestion_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/ingest/semantic",
            json={
                "doc_id": "api_doc_45",
                "text": (
                    "# Vector Indexing\n"
                    "HNSW graphs build multi-layer navigable small world structures for ANN search.\n\n"
                    "# Distributed Consensus\n"
                    "Raft elects a cluster leader using randomized election timeouts and term counters."
                ),
                "max_chunk_tokens": 120,
                "min_chunk_tokens": 8,
                "similarity_threshold": 0.4,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "api_doc_45"
        assert data["chunk_count"] >= 2

        telem = await client.get("/v1/ingestion/semantic/telemetry")
        assert telem.status_code == 200
        stats = telem.json()
        assert stats["documents_processed"] >= 1
        assert stats["total_chunks_emitted"] >= 2
