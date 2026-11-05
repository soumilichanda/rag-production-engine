import numpy as np
import pytest
from app.core.semantic_cache import SemanticCache


def test_cache_compaction_lru():
    cache = SemanticCache(capacity=3, threshold=0.9, default_ttl=60)
    for i in range(5):
        vec = np.array([float(i), 1.0], dtype=np.float32)
        cache.put(f"key_{i}", vec, {"idx": i})

    assert len(cache.cache) <= 3
    stats = cache.stats()
    assert stats["evictions"] >= 2


def test_explicit_compact_call():
    cache = SemanticCache(capacity=10, threshold=0.9, default_ttl=0.1)
    vec = np.array([1.0, 0.0], dtype=np.float32)
    cache.put("exp1", vec, {"d": 1})
    cache.put("exp2", vec, {"d": 2})

    import time
    time.sleep(0.15)
    evicted = cache.compact()
    assert evicted == 2
    assert cache.stats()["size"] == 0
