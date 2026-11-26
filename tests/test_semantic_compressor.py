"""
Async pytest suite for Phase 44: Contextual Compression & Semantic Deduplication Pipeline.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.retrieval.semantic_compressor import SemanticContextCompressor


def test_semantic_compressor_deduplicates_redundant_spans():
    compressor = SemanticContextCompressor(
        dedup_threshold=0.80, default_token_budget=200
    )
    chunks = [
        {
            "chunk_id": "c1",
            "text": "HNSW vector indexing accelerates approximate nearest neighbor search. Kafka handles event streaming.",
            "similarity_score": 0.92,
            "metadata": {"source": "doc_a"},
        },
        {
            "chunk_id": "c2",
            "text": "HNSW vector indexing accelerates approximate nearest neighbor search! Redis caches hot keys.",
            "similarity_score": 0.85,
            "metadata": {"source": "doc_b"},
        },
    ]

    result = compressor.compress(
        query="HNSW vector search and Redis caching",
        chunks=chunks,
        max_token_budget=200,
    )

    assert result["metrics"]["deduplicated_spans_removed"] == 1
    assert result["metrics"]["selected_span_count"] == 3
    assert "HNSW vector indexing" in result["compressed_text"]
    assert "Redis caches hot keys." in result["compressed_text"]


def test_semantic_compressor_enforces_strict_token_budget():
    compressor = SemanticContextCompressor(
        dedup_threshold=0.90, default_token_budget=12
    )
    chunks = [
        {
            "chunk_id": "c1",
            "text": (
                "Background filler sentence with low relevance to the user question. "
                "Self-RAG verifies facet coverage thresholds."
            ),
            "similarity_score": 0.80,
        }
    ]

    result = compressor.compress(
        query="Self-RAG facet coverage",
        chunks=chunks,
        max_token_budget=8,
    )

    assert result["metrics"]["compressed_tokens"] <= 8
    assert "Self-RAG" in result["compressed_text"]
    assert "filler" not in result["compressed_text"]


@pytest.mark.asyncio
async def test_compression_gateway_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/context/compress",
            json={
                "query": "knapsack token budget pruning",
                "max_token_budget": 15,
                "chunks": [
                    {
                        "chunk_id": "chunk_10",
                        "text": "Knapsack optimization packs high salience spans within a strict token budget. Unrelated weather note.",
                        "similarity_score": 0.91,
                        "metadata": {"tier": "pro"},
                    },
                    {
                        "chunk_id": "chunk_11",
                        "text": "Knapsack optimization packs high salience spans within a strict token budget.",
                        "similarity_score": 0.84,
                        "metadata": {"tier": "pro"},
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metrics"]["deduplicated_spans_removed"] == 1
        assert data["metrics"]["compressed_tokens"] <= 15
        assert data["selected_spans"][0]["metadata"] == {"tier": "pro"}

        telem = await client.get("/v1/compression/telemetry")
        assert telem.status_code == 200
        body = telem.json()
        assert body["status"] == "operational"
        assert body["compression"]["total_requests"] >= 1
        assert body["compression"]["total_deduped_spans"] >= 1
