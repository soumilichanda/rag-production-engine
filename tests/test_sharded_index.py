import asyncio
import numpy as np
import pytest
from app.schemas.payloads import DocumentChunk
from app.retrieval.sharded_index import ShardedVectorIndex
from app.ingestion.parallel_ingest import ParallelIngestPipeline


def test_sharded_vector_index_scatter_gather():
    dim = 8
    index = ShardedVectorIndex(dimension=dim, num_shards=4)

    target_vec = np.zeros(dim, dtype=np.float32)
    target_vec[0] = 1.0

    noise_vec = np.zeros(dim, dtype=np.float32)
    noise_vec[1] = 1.0

    chunk_1 = DocumentChunk(chunk_id="chunk_primary", text="Primary high relevance match")
    chunk_2 = DocumentChunk(chunk_id="chunk_secondary", text="Secondary background chunk")

    index.add_document(chunk_1, target_vec)
    index.add_document(chunk_2, noise_vec)

    query = np.zeros(dim, dtype=np.float32)
    query[0] = 0.99
    query[1] = 0.01

    results = index.search(query, top_k=1, threshold=0.5)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_primary"
    assert "shard_id" in results[0].metadata


@pytest.mark.asyncio
async def test_concurrent_parallel_ingestion():
    dim = 16
    index = ShardedVectorIndex(dimension=dim, num_shards=8)
    pipeline = ParallelIngestPipeline(index=index, batch_size=10)

    dataset = []
    for i in range(100):
        c = DocumentChunk(chunk_id=f"doc_stream_{i}", text=f"Streaming ingestion benchmark line {i}")
        emb = np.random.randn(dim).astype(np.float32)
        dataset.append((c, emb))

    ingested = await pipeline.ingest_batch_async(dataset)
    assert ingested == 100
    assert index.total_documents() == 100
