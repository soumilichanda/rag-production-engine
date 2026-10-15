"""
benchmarks/benchmark_cache.py
Empirical latency profiling: Semantic Cache Hits vs Vector Search Misses.
"""

import time

import numpy as np

from app.core.semantic_cache import SemanticCache


def run_cache_benchmarks(iterations: int = 1000):
    cache = SemanticCache(dimension=8, distance_threshold=0.08)
    base_vectors = [np.random.randn(8).astype(np.float32) for _ in range(20)]

    # Seed cache
    for vec in base_vectors[:5]:
        cache.insert(vec, {"mock_payload": "precomputed_chunks"})

    start = time.perf_counter()
    for i in range(iterations):
        # 60% probability of selecting an existing vector with minor perturbation
        if i % 10 < 6:
            idx = i % 5
            query_vec = base_vectors[idx] + np.random.normal(0, 0.01, 8).astype(
                np.float32
            )
        else:
            query_vec = np.random.randn(8).astype(np.float32)

        payload, _ = cache.lookup(query_vec)
        if payload is None:
            time.sleep(0.0002)  # Simulating vector retrieval latency
            cache.insert(query_vec, {"mock_payload": "dynamically_retrieved"})

    total_time = (time.perf_counter() - start) * 1000.0
    telemetry = cache.get_telemetry()

    print("=== Semantic Cache Latency Profiling Benchmark ===")
    print(f"Iterations Executed : {iterations}")
    print(f"Total Wall Time     : {total_time:.2f} ms")
    print(f"Average Latency/Req : {total_time / iterations:.3f} ms")
    print(f"Hit Rate            : {telemetry['hit_rate'] * 100:.1f}%")
    print(f"Hits / Misses       : {telemetry['hits']} / {telemetry['misses']}")


if __name__ == "__main__":
    run_cache_benchmarks()
