import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_compressed_retrieval_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Seed test chunk with long filler content
        seed_payload = {
            "documents": [
                {
                    "chunk_id": "unique_compression_eval_chunk_99",
                    "text": "PostgreSQL index query optimizer latency was reduced by 40%. Unrelated company meeting was held at noon. Additional filler words were spoken during lunch. The query optimizer tuning directly solved the issue.",
                    "metadata": {"category": "database"},
                }
            ]
        }
        idx_res = await client.post("/v1/index", json=seed_payload)
        assert idx_res.status_code == 201

        # Query compressed retrieval endpoint with schema-valid score_threshold (>= 0.0)
        response = await client.post(
            "/v1/retrieve/compressed",
            json={
                "query": "index latency query optimizer",
                "top_k": 5,
                "score_threshold": 0.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieved_count"] >= 1

        target = next(
            (
                r
                for r in data["results"]
                if r["chunk_id"] == "unique_compression_eval_chunk_99"
            ),
            data["results"][0],
        )
        assert target["metadata"]["compression_applied"] is True
        assert len(target["text"]) <= target["metadata"]["original_len"]
