import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app, store, dummy_text_to_embedding
from app.schemas.payloads import DocumentChunk
import numpy as np


@pytest.fixture(autouse=True)
def populate_index():
    docs = [
        DocumentChunk(
            chunk_id="chunk_db",
            text="PostgreSQL index latency was reduced by 40%. The devops team bought new chairs. Query optimizer tuning helped.",
            metadata={"category": "database"}
        )
    ]
    embeddings = np.array([dummy_text_to_embedding(docs[0].text)], dtype=np.float32)
    store.add_documents(docs, embeddings)


@pytest.mark.asyncio
async def test_compressed_retrieval_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/retrieve/compressed",
            json={"query": "index latency query optimizer", "top_k": 1, "score_threshold": 0.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieved_count"] == 1
        result = data["results"][0]
        assert "latency" in result["text"] or "optimizer" in result["text"]
        assert result["metadata"]["compression_applied"] is True
        assert result["metadata"]["compressed_len"] <= result["metadata"]["original_len"]
