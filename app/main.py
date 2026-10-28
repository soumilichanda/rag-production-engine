import time
from typing import List
from fastapi import FastAPI, HTTPException, status
from app.schemas.payloads import QueryRequest, QueryResponse, RetrievedChunk, IngestionRequest, IngestionResponse
from app.retrieval.vector_store import InMemoryVectorStore
from app.core.reranker import ExactTokenOverlapReranker
from app.retrieval.context_compressor import CrossEncoderContextCompressor
import numpy as np

app = FastAPI(
    title="Production RAG Engine",
    description="Vector retrieval microservice with context compression",
    version="1.3.0"
)

DIMENSION = 384
store = InMemoryVectorStore(dimension=DIMENSION)
reranker = ExactTokenOverlapReranker()
compressor = CrossEncoderContextCompressor(max_tokens_per_chunk=64)


def dummy_text_to_embedding(text: str, dim: int = DIMENSION) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "indexed_documents": len(getattr(store, "chunks", []))}


@app.post("/v1/retrieve/compressed", response_model=QueryResponse, tags=["Retrieval"])
async def retrieve_compressed_context(payload: QueryRequest):
    start = time.perf_counter()
    query_vec = dummy_text_to_embedding(payload.query)

    candidates, _ = store.query(query_vec, top_k=payload.top_k * 2, threshold=payload.score_threshold)
    ranked = reranker.rerank(payload.query, candidates, top_k=payload.top_k)

    compressed_results = []
    for chunk in ranked:
        compressed_text = compressor.compress_chunk(payload.query, chunk.text)
        compressed_results.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=compressed_text,
                similarity_score=chunk.similarity_score,
                metadata={
                    **chunk.metadata,
                    "original_len": len(chunk.text),
                    "compressed_len": len(compressed_text),
                    "compression_applied": True,
                }
            )
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    return QueryResponse(
        query=payload.query,
        results=compressed_results,
        retrieved_count=len(compressed_results),
        latency_ms=round(latency_ms, 2)
    )
