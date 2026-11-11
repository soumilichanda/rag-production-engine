from collections import Counter

import numpy as np

from app.retrieval.consistent_hash import ConsistentHashRing
from app.retrieval.sharded_index import ShardedVectorIndex
from app.schemas.payloads import DocumentChunk


def test_consistent_hash_uniformity():
    ring = ConsistentHashRing(shards=[0, 1, 2, 3], virtual_replicas=100)
    assignments = Counter()

    for i in range(1000):
        shard = ring.get_shard(f"doc_uuid_{i}")
        assignments[shard] += 1

    # Every shard must receive between 15% and 35% of the total dataset
    for shard_id in range(4):
        assert 150 <= assignments[shard_id] <= 350


def test_consistent_hash_minimal_movement():
    ring = ConsistentHashRing(shards=[0, 1, 2], virtual_replicas=100)
    keys = [f"key_{i}" for i in range(300)]
    initial_mapping = {k: ring.get_shard(k) for k in keys}

    # Add a 4th shard
    ring.add_shard(3)
    rebalanced_mapping = {k: ring.get_shard(k) for k in keys}

    # Keys either stay in place or move to the new shard
    moved_to_existing = 0
    for k in keys:
        if initial_mapping[k] != rebalanced_mapping[k] and rebalanced_mapping[k] != 3:
            moved_to_existing += 1

    assert moved_to_existing == 0


def test_sharded_index_with_hash_ring():
    dim = 8
    index = ShardedVectorIndex(dimension=dim, num_shards=4, virtual_replicas=50)

    for i in range(20):
        vec = np.zeros(dim, dtype=np.float32)
        vec[i % dim] = 1.0
        chunk = DocumentChunk(chunk_id=f"ring_chunk_{i}", text=f"Chunk payload {i}")
        index.add_document(chunk, vec)

    assert index.total_documents() == 20
