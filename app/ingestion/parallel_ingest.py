"""
app/ingestion/parallel_ingest.py
Concurrent batch ingestion pipeline streaming multi-document matrix inserts across sharded partitions.
"""

import asyncio

import numpy as np

from app.retrieval.sharded_index import ShardedVectorIndex
from app.schemas.payloads import DocumentChunk


class ParallelIngestPipeline:
    def __init__(self, index: ShardedVectorIndex, batch_size: int = 32):
        self.index = index
        self.batch_size = batch_size

    async def ingest_batch_async(
        self, items: list[tuple[DocumentChunk, np.ndarray]]
    ) -> int:
        """Asynchronously dispatches documents to striped shards without global mutex contention."""
        loop = asyncio.get_running_loop()

        def _worker(batch: list[tuple[DocumentChunk, np.ndarray]]) -> int:
            for doc, emb in batch:
                self.index.add_document(doc, emb)
            return len(batch)

        chunks = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        tasks = [loop.run_in_executor(None, _worker, batch) for batch in chunks]
        results = await asyncio.gather(*tasks)
        return sum(results)
