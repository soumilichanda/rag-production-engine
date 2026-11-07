"""
app/retrieval/consistent_hash.py
Consistent hashing ring with configurable virtual node replication for balanced partition routing.
"""

import hashlib
import bisect
from typing import List, Dict, Optional


class ConsistentHashRing:
    """A consistent hash ring distributing partition assignments evenly via virtual nodes."""

    def __init__(self, shards: List[int], virtual_replicas: int = 100):
        self.virtual_replicas = virtual_replicas
        self.ring: List[int] = []
        self.ring_to_shard: Dict[int, int] = {}
        self.shards: List[int] = []

        for shard_id in shards:
            self.add_shard(shard_id)

    def _hash(self, key: str) -> int:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def add_shard(self, shard_id: int) -> None:
        if shard_id not in self.shards:
            self.shards.append(shard_id)
            for replica in range(self.virtual_replicas):
                virtual_key = f"shard-{shard_id}-rep-{replica}"
                node_hash = self._hash(virtual_key)
                bisect.insort(self.ring, node_hash)
                self.ring_to_shard[node_hash] = shard_id

    def remove_shard(self, shard_id: int) -> None:
        if shard_id in self.shards:
            self.shards.remove(shard_id)
            for replica in range(self.virtual_replicas):
                virtual_key = f"shard-{shard_id}-rep-{replica}"
                node_hash = self._hash(virtual_key)
                idx = bisect.bisect_left(self.ring, node_hash)
                if idx < len(self.ring) and self.ring[idx] == node_hash:
                    del self.ring[idx]
                self.ring_to_shard.pop(node_hash, None)

    def get_shard(self, chunk_id: str) -> int:
        if not self.ring:
            raise RuntimeError("Consistent hash ring is empty.")

        key_hash = self._hash(chunk_id)
        idx = bisect.bisect_right(self.ring, key_hash)

        # Wrap around the ring if the hash exceeds the largest virtual node
        if idx == len(self.ring):
            idx = 0

        return self.ring_to_shard[self.ring[idx]]
