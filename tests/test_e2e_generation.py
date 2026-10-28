import pytest
from httpx import ASGITransport, AsyncClient

from app.core.synthesizer import GroundedContextSynthesizer
from app.main import app
from app.schemas.payloads import RetrievedChunk


def test_synthesizer_with_citations():
    synthesizer = GroundedContextSynthesizer()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_doc_1",
            text="Dense vector retrieval indexes semantic meaning.",
            similarity_score=0.92,
        ),
        RetrievedChunk(
            chunk_id="chunk_doc_2",
            text="BM25 matches keyword lexical occurrences.",
            similarity_score=0.88,
        ),
    ]
    res = synthesizer.synthesize(
        query="What is dense retrieval?", context=chunks, latency_ms=12.4
    )
    assert res.query == "What is dense retrieval?"
    assert res.citations == ["chunk_doc_1", "chunk_doc_2"]
    assert res.retrieved_count == 2
    assert "chunk_doc_1" in res.answer
    assert res.latency_ms == 12.4


def test_generation_empty_context_fallback():
    synthesizer = GroundedContextSynthesizer()
    res = synthesizer.synthesize(query="unmatched query", context=[])
    assert res.citations == []
    assert res.retrieved_count == 0
    assert "No relevant context" in res.answer


@pytest.mark.asyncio
async def test_generation_endpoint_with_seeded_document():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Seed valid document to satisfy guardrails & index invariants
        doc_payload = {
            "documents": [
                {
                    "chunk_id": "seed_chunk_1",
                    "text": "Production RAG platforms combine dense semantic indexing with context reranking.",
                    "metadata": {"source": "manual"},
                }
            ]
        }
        idx_res = await client.post("/v1/index", json=doc_payload)
        assert idx_res.status_code in [200, 201]

        # Query the generate endpoint
        gen_payload = {
            "query": "Production RAG platforms indexing",
            "top_k": 1,
            "score_threshold": 0.0,
        }
        gen_res = await client.post("/v1/generate", json=gen_payload)
        assert gen_res.status_code == 200
        data = gen_res.json()
        assert "citations" in data
        assert "answer" in data
        assert data["retrieved_count"] >= 1
