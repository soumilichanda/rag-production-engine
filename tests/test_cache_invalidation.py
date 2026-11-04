import time
import numpy as np
import pytest
from app.core.semantic_cache import SemanticCache


def test_ttl_expiry():
    cache = SemanticCache(capacity=10, threshold=0.9, default_ttl=0.2)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cache.put("q_temp", vec, {"data": "ephemeral"}, ttl=0.1)

    assert cache.get(vec) is not None
    time.sleep(0.15)
    assert cache.get(vec) is None
    assert cache.stats()["misses"] >= 1


def test_tag_invalidation():
    cache = SemanticCache(capacity=10, threshold=0.9, default_ttl=60)
    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    cache.put("doc_1", vec1, {"result": "finance"}, tags=["domain:finance", "v1"])
    cache.put("doc_2", vec2, {"result": "healthcare"}, tags=["domain:health", "v1"])

    evicted = cache.invalidate_by_tag("domain:finance")
    assert evicted == 1
    assert cache.get(vec1) is None
    assert cache.get(vec2) is not None


def test_flush_all():
    cache = SemanticCache(capacity=10, threshold=0.9)
    cache.put("q1", np.array([1.0, 0.0]), {"val": 1})
    cache.put("q2", np.array([0.0, 1.0]), {"val": 2})

    cleared = cache.invalidate_all()
    assert cleared == 2
    assert cache.stats()["size"] == 0
