import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_compressed_retrieval_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Seed test chunk
        seed_payload = {
            "documents": [
                {
                    "chunk_id": "seed_compression_doc",
                    "text": "PostgreSQL index query optimizer latency was reduced by 40%. The devops team bought new chairs. Query optimizer tuning helped.",
                    "metadata": {"category": "database"}
                }
            ]
        }
        idx_res = await client.post("/v1/index", json=seed_payload)
        assert idx_res.status_code == 201

        # Query compressed retrieval endpoint
        response = await client.post(
            "/v1/retrieve/compressed",
            json={"query": "index latency query optimizer", "top_k": 1, "score_threshold": 0.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieved_count"] >= 1
        result = data["results"][0]
        assert result["metadata"]["compression_applied"] is True
        assert result["metadata"]["compressed_len"] <= result["metadata"]["original_len"]
