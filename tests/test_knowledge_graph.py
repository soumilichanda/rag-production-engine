"""Async pytest suite verifying Phase 41 Graph-RAG triple extraction & 2-hop traversal."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.retrieval.knowledge_graph import KnowledgeGraphIndex


def test_triple_extraction_and_two_hop_traversal():
    kg = KnowledgeGraphIndex(max_hops=2)
    kg.ingest_chunk(
        "c_hop1",
        "Gateway routes QueryPlanner and [QueryPlanner -> invokes -> VectorIndex].",
    )
    kg.ingest_chunk(
        "c_hop2",
        "VectorIndex uses HNSW and [HNSW -> accelerates -> CosineSearch].",
    )

    result = kg.traverse_2hop("How does Gateway reach HNSW?", top_k=3)
    assert "gateway" in result["seed_entities"]
    assert any("queryplanner" in p for p in result["paths"])
    assert any("hnsw" in p for p in result["paths"])
    chunk_ids = [c["chunk_id"] for c in result["chunks"]]
    assert "c_hop1" in chunk_ids
    assert "c_hop2" in chunk_ids


@pytest.mark.asyncio
async def test_graph_rag_gateway_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ingest_resp = await client.post(
            "/v1/graph/ingest",
            json={
                "documents": [
                    {
                        "chunk_id": "g_doc_1",
                        "text": "Retriever connects Reranker and [Reranker -> feeds -> Synthesizer].",
                    },
                    {
                        "chunk_id": "g_doc_2",
                        "text": "Synthesizer verifies GroundingAudit.",
                    },
                ]
            },
        )
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        assert ingest_data["triples_indexed"] >= 3

        trav_resp = await client.post(
            "/v1/graph/traverse",
            json={"query": "Retriever Synthesizer", "top_k": 3},
        )
        assert trav_resp.status_code == 200
        trav_data = trav_resp.json()
        assert "retriever" in trav_data["seed_entities"]
        assert len(trav_data["fused_results"]) >= 1

        tel_resp = await client.get("/v1/graph/telemetry")
        assert tel_resp.status_code == 200
        tel_data = tel_resp.json()
        assert tel_data["status"] == "operational"
        assert tel_data["graph"]["triples"] >= 3
