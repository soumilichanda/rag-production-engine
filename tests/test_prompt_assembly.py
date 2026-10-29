import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.pipeline.prompt_assembler import PromptContextAssembler


@pytest.fixture
def sample_chunks():
    return [
        {
            "document_id": "doc_alpha",
            "chunk_id": 1,
            "score": 0.65,
            "text": "Low priority background information that may be pruned if budget runs out." * 5,
        },
        {
            "document_id": "doc_beta",
            "chunk_id": 2,
            "score": 0.95,
            "text": "High priority critical finding: distributed consensus achieved." * 5,
        },
        {
            "document_id": "doc_gamma",
            "chunk_id": 3,
            "score": 0.88,
            "text": "Medium priority supporting evidence on latency profiling." * 5,
        },
    ]


def test_assembler_ordering_and_budget(sample_chunks):
    assembler = PromptContextAssembler(max_context_tokens=100)
    res = assembler.assemble(query="What is the consensus?", retrieved_chunks=sample_chunks)

    assert res.retained_chunks_count >= 1
    assert res.provenance_sources[0]["document_id"] == "doc_beta"
    assert res.total_estimated_tokens > 0
    assert "Context:" in res.user_prompt


@pytest.mark.asyncio
async def test_gateway_assemble_route(sample_chunks):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "query": "Explain consensus protocol",
            "chunks": sample_chunks,
            "max_context_tokens": 800,
        }
        resp = await ac.post("/v1/pipeline/assemble", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["retained_chunks_count"] == 3
        assert len(data["provenance_sources"]) == 3
        assert data["provenance_sources"][0]["document_id"] == "doc_beta"
