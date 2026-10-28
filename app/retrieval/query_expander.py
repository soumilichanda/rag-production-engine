"""
app/retrieval/query_expander.py
Generates diverse query perspectives and gathers sub-query vector candidate chunks.
"""

import asyncio
from typing import Any

from app.schemas.payloads import RetrievedChunk


class MultiQueryRetriever:
    """
    Expands an incoming query into diverse syntactically distinct variations
    and executes concurrent asynchronous vector store queries.
    """

    def __init__(self, vector_store: Any):
        self.vector_store = vector_store

    def generate_perspectives(self, query: str) -> list[str]:
        cleaned = query.strip()
        return [
            cleaned,
            f"technical overview and implementation details of {cleaned}",
            f"core concepts, definitions, and mechanics of {cleaned}",
        ]

    async def retrieve_perspectives(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> list[list[RetrievedChunk]]:
        perspectives = self.generate_perspectives(query)

        # Query all perspectives concurrently
        tasks = [
            asyncio.to_thread(self.vector_store.query, p, top_k, score_threshold)
            for p in perspectives
        ]
        return await asyncio.gather(*tasks)
