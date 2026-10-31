"""
tests/test_streaming.py
Integration tests for Server-Sent Events (SSE) token streaming endpoint.
"""

import pytest
import json
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_stream_pipeline_sse_flow():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Index sample test data to establish context chunks (returns 201 Created)
        index_payload = {
            "documents": [
                {
                    "chunk_id": "stream_chunk_01",
                    "text": "Server-Sent Events enable unidirectional low-latency token streaming for generative AI."
                }
            ]
        }
        idx_res = await client.post("/v1/index", json=index_payload)
        assert idx_res.status_code == 201

        # 2. Invoke streaming endpoint
        stream_payload = {
            "query": "How do SSE streams work in generative AI?",
            "top_k": 2
        }
        res = await client.post("/v1/pipeline/stream", json=stream_payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

        # 3. Parse stream chunks and assert SSE boundaries
        raw_stream = res.text
        assert "data: " in raw_stream
        assert "data: [DONE]" in raw_stream

        # Validate token payloads and citations payload
        lines = [line.replace("data: ", "").strip() for line in raw_stream.splitlines() if line.startswith("data: ")]
        
        token_events = []
        citation_event = None
        has_done = False

        for item in lines:
            if item == "[DONE]":
                has_done = True
                continue
            parsed = json.loads(item)
            if parsed.get("event") == "token":
                token_events.append(parsed)
            elif parsed.get("event") == "citations":
                citation_event = parsed

        assert len(token_events) > 0
        assert citation_event is not None
        assert "citations" in citation_event.get("data", {})
        assert has_done is True
