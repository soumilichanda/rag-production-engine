import pytest
from httpx import ASGITransport, AsyncClient

from app.core.citation_audit import ComplianceAuditLog, SemanticCitationGrounder
from app.main import app


def test_semantic_citation_grounder_provenance():
    grounder = SemanticCitationGrounder(attribution_threshold=0.35)
    chunks = [
        {
            "chunk_id": "doc_1",
            "text": "Reciprocal Rank Fusion combines multiple retrieval rankings cleanly.",
        },
        {
            "chunk_id": "doc_2",
            "text": "Write-Ahead Logging persists vector mutations with CRC32 frames.",
        },
    ]
    answer = (
        "Reciprocal Rank Fusion combines retrieval rankings cleanly. "
        "Quantum teleportation happens instantaneously across galaxies."
    )
    report = grounder.verify(answer, chunks)
    assert report.total_claims == 2
    assert report.grounded_claims == 1
    assert report.grounding_score == 0.5
    assert report.cited_chunk_ids == ["doc_1"]
    assert len(report.ungrounded_claims) == 1


def test_compliance_audit_log_tamper_detection():
    ledger = ComplianceAuditLog()
    r1 = ledger.record("tenant_alpha", "What is RRF?", ["doc_1"], 1.0, "passed")
    r2 = ledger.record("tenant_alpha", "Explain WAL.", ["doc_2"], 0.9, "passed")

    assert r2.prev_hash == r1.entry_hash
    assert ledger.verify_chain() is True

    # Simulate tampering with historical record
    ledger._entries[0].grounding_score = 0.01
    assert ledger.verify_chain() is False


@pytest.mark.asyncio
async def test_stream_grounding_headers_and_audit_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/index",
            json={
                "documents": [
                    {
                        "chunk_id": "chunk_phase40",
                        "text": "Semantic citation grounding verifies claim attribution against retrieved chunks.",
                        "metadata": {"phase": 40},
                    }
                ]
            },
        )

        res = await client.post(
            "/v1/pipeline/stream",
            headers={"X-Tenant-ID": "tenant_audit_test"},
            json={
                "query": "Semantic citation grounding verifies claim attribution",
                "top_k": 2,
            },
        )
        assert res.status_code == 200
        assert "X-Grounding-Score" in res.headers
        assert "X-Cited-Chunks" in res.headers
        audit_id = res.headers.get("X-Audit-ID")
        assert audit_id and audit_id.startswith("aud_")

        logs_res = await client.get(
            "/v1/audit/logs", params={"tenant_id": "tenant_audit_test"}
        )
        assert logs_res.status_code == 200
        payload = logs_res.json()
        assert payload["chain_valid"] is True
        assert any(entry["audit_id"] == audit_id for entry in payload["entries"])

        summary_res = await client.get("/v1/audit/summary")
        assert summary_res.status_code == 200
        assert summary_res.json()["audit"]["chain_valid"] is True
