"""Async pytest suite for Phase 46: Hierarchical Parent-Document Retrieval."""

import httpx
import pytest

from app.main import app
from app.retrieval.hierarchical_retriever import HierarchicalParentRetriever

SAMPLE_ARCHITECTURE_DOC = """
# Distributed Vector Storage Layer
Our vector engine partitions dense embeddings across immutable segments backed by memory-mapped Float32 arrays.
Each segment tracks cosine similarity norms and supports zero-copy SIMD dot-product scans.
Write-ahead logging guarantees crash recovery across restarts without losing uncommitted batch mutations.

# Adaptive Canary Traffic Routing
Canary traffic routing splits live inference queries across stable and candidate retrieval pipelines using deterministic tenant hashing.
Shadow comparisons record latency percentiles and recall divergence before promoting candidate configurations.
"""


def test_hierarchical_parent_child_indexing_and_deduplication():
    retriever = HierarchicalParentRetriever(
        parent_max_tokens=300,
        child_max_tokens=40,
        child_overlap_tokens=8,
    )
    summary = retriever.ingest_document(
        doc_id="arch_spec_v1",
        text=SAMPLE_ARCHITECTURE_DOC,
        metadata={"tier": "core"},
    )
    assert summary["parent_sections"] == 2
    assert summary["child_chunks"] >= 3

    response = retriever.retrieve(
        query="memory-mapped Float32 arrays SIMD dot-product cosine similarity",
        top_k_parents=2,
        child_candidate_pool=6,
    )
    assert response["parents_returned"] >= 1
    top_parent = response["results"][0]
    assert top_parent["heading"] == "Distributed Vector Storage Layer"
    assert "Write-ahead logging guarantees crash recovery" in top_parent["parent_text"]
    assert response["context_expansion_ratio"] > 0.0
    assert (
        top_parent["parent_tokens"]
        >= top_parent["matched_children"][0]["token_estimate"]
    )
    # Verify unique parent IDs after small-to-big deduplication
    returned_ids = [item["parent_id"] for item in response["results"]]
    assert len(returned_ids) == len(set(returned_ids))


@pytest.mark.asyncio
async def test_hierarchical_endpoints_and_telemetry():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/retrieve/hierarchical",
            json={
                "query": "canary traffic routing deterministic tenant hashing",
                "top_k_parents": 2,
                "child_candidate_pool": 8,
                "documents": [
                    {
                        "doc_id": "gateway_guide",
                        "text": SAMPLE_ARCHITECTURE_DOC,
                        "metadata": {"phase": 46},
                    }
                ],
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["parents_returned"] >= 1
        assert body["results"][0]["heading"] == "Adaptive Canary Traffic Routing"

        tel = await client.get("/v1/retrieval/hierarchical/telemetry")
        assert tel.status_code == 200
        tel_body = tel.json()
        assert tel_body["status"] == "operational"
        assert tel_body["hierarchical_retrieval"]["documents_indexed"] >= 1
        assert tel_body["hierarchical_retrieval"]["retrieval_calls"] >= 1
