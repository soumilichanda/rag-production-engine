"""
app/retrieval/shard_merger.py
Background shard consolidation worker for ShardedVectorIndex partitions.
Merges fragmented low-volume shards to optimize scatter-gather search latency.
"""

import asyncio
from typing import Any

import numpy as np

from app.retrieval.sharded_index import ShardedVectorIndex, ShardPartition


class BackgroundShardMerger:
    def __init__(self, index: ShardedVectorIndex, min_threshold: int = 5):
        self.index = index
        self.min_threshold = min_threshold

    def find_mergeable_shards(self) -> list[int]:
        mergeable = []
        for shard_id, partition in self.index.shards.items():
            if len(partition.chunks) < self.min_threshold and len(partition.chunks) > 0:
                mergeable.append(shard_id)
        return mergeable

    async def merge_sparse_shards(
        self, source_id: int, target_id: int
    ) -> dict[str, Any]:
        if source_id == target_id:
            return {"status": "skipped", "reason": "identical shards"}

        source_shard: ShardPartition = self.index.shards.get(source_id)
        target_shard: ShardPartition = self.index.shards.get(target_id)

        if not source_shard or not target_shard:
            return {"status": "failed", "reason": "invalid shard IDs"}

        # Non-blocking lock acquisition across both partitions to maintain concurrency safety
        loop = asyncio.get_running_loop()

        def _merge_worker() -> int:
            with source_shard.lock, target_shard.lock:
                count = len(source_shard.chunks)
                if count == 0:
                    return 0

                # Transfer chunk metadata
                for c in source_shard.chunks:
                    c.metadata["merged_from"] = source_id
                    target_shard.chunks.append(c)

                # Transfer embeddings matrix
                if target_shard.embeddings.shape[0] == 0:
                    target_shard.embeddings = source_shard.embeddings.copy()
                else:
                    target_shard.embeddings = np.vstack(
                        [target_shard.embeddings, source_shard.embeddings]
                    )

                # Reset source partition
                source_shard.chunks.clear()
                source_shard.embeddings = np.empty(
                    (0, source_shard.dimension), dtype=np.float32
                )
                return count

        moved_count = await loop.run_in_executor(None, _merge_worker)
        return {
            "status": "success",
            "source_shard": source_id,
            "target_shard": target_id,
            "records_moved": moved_count,
            "target_total": len(target_shard.chunks),
        }
