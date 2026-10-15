import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_indexing_and_retrieval_flow():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Ingest test documents
        ingest_payload = {
            "documents": [
                {
                    "chunk_id": "c1",
                    "text": "Continuous deployment workflows on Kubernetes cluster",
                    "metadata": {"env": "prod"},
                },
                {
                    "chunk_id": "c2",
                    "text": "Asynchronous microservice concurrency with FastAPI and Redis",
                    "metadata": {"env": "staging"},
                },
            ]
        }
        res_index = await client.post("/v1/index", json=ingest_payload)
        assert res_index.status_code == 201
        assert res_index.json()["ingested_count"] == 2

        # 2. Query matching context
        query_payload = {
            "query": "Kubernetes continuous deployment",
            "top_k": 2,
            "score_threshold": 0.0,
        }
        res_query = await client.post("/v1/retrieve", json=query_payload)
        assert res_query.status_code == 200
        data = res_query.json()
        assert data["retrieved_count"] > 0
        assert data["results"][0]["chunk_id"] == "c1"
