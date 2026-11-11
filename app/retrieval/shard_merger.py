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

    def _get_shard(self, shard_id: int) -> ShardPartition | None:
        if isinstance(self.index.shards, dict):
            return self.index.shards.get(shard_id)
        elif isinstance(self.index.shards, list) and 0 <= shard_id < len(
            self.index.shards
        ):
            return self.index.shards[shard_id]
        return None

    def find_mergeable_shards(self) -> list[int]:
        mergeable: list[int] = []
        partitions = (
            self.index.shards.items()
            if isinstance(self.index.shards, dict)
            else enumerate(self.index.shards)
        )
        for shard_id, partition in partitions:
            if 0 < len(partition.chunks) < self.min_threshold:
                mergeable.append(shard_id)
        return mergeable

    async def merge_sparse_shards(
        self, source_id: int, target_id: int
    ) -> dict[str, Any]:
        if source_id == target_id:
            return {"status": "skipped", "reason": "identical shards"}

        source_shard = self._get_shard(source_id)
        target_shard = self._get_shard(target_id)

        if source_shard is None or target_shard is None:
            return {"status": "failed", "reason": "invalid shard IDs"}

        loop = asyncio.get_running_loop()

        def _merge_worker() -> int:
            with source_shard.lock, target_shard.lock:
                count = len(source_shard.chunks)
                if count == 0:
                    return 0

                for c in source_shard.chunks:
                    c.metadata["merged_from"] = source_id
                    target_shard.chunks.append(c)

                if target_shard.embeddings.shape[0] == 0:
                    target_shard.embeddings = source_shard.embeddings.copy()
                else:
                    target_shard.embeddings = np.vstack(
                        [target_shard.embeddings, source_shard.embeddings]
                    )

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
