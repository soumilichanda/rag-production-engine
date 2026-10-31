"""
app/generation/streamer.py
Asynchronous token streaming with live citation mapping and SSE formatting.
"""

import asyncio
import json
from typing import AsyncGenerator, List
from app.schemas.payloads import RetrievedChunk


class StreamingContextSynthesizer:
    """
    Simulates token-by-token LLM generation yielding SSE-compliant chunks
    annotated with provenance source citations upon generation completion.
    """

    def __init__(self, token_delay: float = 0.02):
        self.token_delay = token_delay

    async def stream_response(
        self, query: str, context_chunks: List[RetrievedChunk]
    ) -> AsyncGenerator[str, None]:
        # Formulate grounded answer tokens
        answer_text = (
            f"Synthesized response for query '{query}' based on {len(context_chunks)} retrieved contexts."
        )
        tokens = answer_text.split(" ")

        for idx, token in enumerate(tokens):
            event_payload = {
                "event": "token",
                "data": {"token": token + " ", "index": idx}
            }
            yield f"data: {json.dumps(event_payload)}\n\n"
            await asyncio.sleep(self.token_delay)

        # Emit citation metadata payload
        citation_payload = {
            "event": "citations",
            "data": {
                "citations": [
                    {"chunk_id": c.chunk_id, "score": c.similarity_score}
                    for c in context_chunks
                ]
            }
        }
        yield f"data: {json.dumps(citation_payload)}\n\n"
        yield "data: [DONE]\n\n"
