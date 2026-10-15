import numpy as np

from app.core.semantic_cache import SemanticCache


def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(dimension=4, distance_threshold=0.10)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    hit, latency = cache.lookup(q1)
    assert hit is None
    assert latency >= 0.0

    cache.insert(q1, {"results": ["chunk_alpha"]})

    # Exact query hit
    hit, _ = cache.lookup(q1)
    assert hit is not None
    assert hit["cache_hit"] is True

    # Near query hit (distance < 0.10)
    q_near = np.array([0.98, 0.02, 0.0, 0.0], dtype=np.float32)
    hit_near, _ = cache.lookup(q_near)
    assert hit_near is not None
    assert hit_near["cache_hit"] is True

    # Far query miss (orthogonal)
    q_far = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    hit_far, _ = cache.lookup(q_far)
    assert hit_far is None


def test_semantic_cache_telemetry():
    cache = SemanticCache(dimension=2, distance_threshold=0.05)
    q = np.array([1.0, 0.0], dtype=np.float32)

    cache.lookup(q)  # Miss
    cache.insert(q, {"data": 1})
    cache.lookup(q)  # Hit

    telemetry = cache.get_telemetry()
    assert telemetry["hits"] == 1
    assert telemetry["misses"] == 1
    assert telemetry["hit_rate"] == 0.5
