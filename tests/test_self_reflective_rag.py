import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.retrieval.self_reflective_rag import SelfReflectiveCritiqueLoop


@pytest.mark.asyncio
async def test_self_rag_early_exit_when_coverage_sufficient():
    loop = SelfReflectiveCritiqueLoop(coverage_threshold=0.75, max_iterations=3)
    calls: list[str] = []

    def mock_retrieve(q: str, top_k: int) -> list[dict]:
        calls.append(q)
        return [
            {
                "chunk_id": "c_hnsw",
                "text": "HNSW vector index builds multi-layer navigable small world graphs for low latency search.",
                "score": 0.92,
            }
        ]

    res = await loop.execute(
        "HNSW vector index latency", retrieve_fn=mock_retrieve, top_k=3
    )
    assert res.sufficient is True
    assert res.total_passes == 1
    assert res.refinement_triggered is False
    assert res.final_coverage_score == 1.0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_self_rag_triggers_refinement_for_missing_facets():
    loop = SelfReflectiveCritiqueLoop(coverage_threshold=0.80, max_iterations=3)
    calls: list[str] = []

    async def mock_retrieve(q: str, top_k: int) -> list[dict]:
        calls.append(q)
        if len(calls) == 1:
            return [
                {
                    "chunk_id": "c_part1",
                    "text": "Redis semantic cache stores query embeddings in memory.",
                    "score": 0.85,
                }
            ]
        return [
            {
                "chunk_id": "c_part2",
                "text": "TTL eviction and stampede mutex locks prevent cache herd thundering.",
                "score": 0.88,
            }
        ]

    res = await loop.execute(
        "Redis semantic cache TTL eviction mutex",
        retrieve_fn=mock_retrieve,
        top_k=4,
    )
    assert len(calls) == 2
    assert res.refinement_triggered is True
    assert res.initial_coverage_score < 0.80
    assert res.final_coverage_score == 1.0
    assert res.sufficient is True
    assert {c["chunk_id"] for c in res.results} == {"c_part1", "c_part2"}


@pytest.mark.asyncio
async def test_self_rag_gateway_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        exec_resp = await client.post(
            "/v1/self-rag/execute",
            json={
                "query": "hybrid retrieval reciprocal rank fusion latency",
                "top_k": 3,
            },
        )
        assert exec_resp.status_code == 200
        body = exec_resp.json()
        assert body["total_passes"] >= 1
        assert "critique_history" in body

        tel_resp = await client.get("/v1/self-rag/telemetry")
        assert tel_resp.status_code == 200
        tel_data = tel_resp.json()
        assert tel_data["status"] == "operational"
        assert tel_data["self_rag"]["total_executions"] >= 1
