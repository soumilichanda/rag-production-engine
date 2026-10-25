from app.core.tracer import PipelineTracer
from app.api.diagnostics import router as diagnostics_router
"""
app/main.py
FastAPI gateway with semantic caching and telemetry profiling endpoints.
"""

import time

import numpy as np
from fastapi import FastAPI, HTTPException, status
from app.core.rate_limiter import RateLimitMiddleware

from app.core.guardrails import GroundednessGuardrail
from app.core.reranker import ExactTokenOverlapReranker
from app.core.semantic_cache import SemanticCache
from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.payloads import (
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(
    title="Production RAG Engine",
    description="Vector retrieval microservice with semantic caching and latency telemetry profiling",
    version="1.2.0",
)

app.add_middleware(RateLimitMiddleware, limit=30, window_seconds=60)

DIMENSION = 8
store = InMemoryVectorStore(dimension=DIMENSION)
reranker = ExactTokenOverlapReranker()
guardrail = GroundednessGuardrail(min_confidence=0.20)
semantic_cache = SemanticCache(dimension=DIMENSION, distance_threshold=0.08)


def dummy_text_to_embedding(text: str, dim: int = DIMENSION) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for i, char in enumerate(text.encode("utf-8")):
        vec[i % dim] += float(char)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "indexed_documents": len(store.chunks),
        "telemetry": semantic_cache.get_telemetry(),
    }


@app.get("/v1/telemetry", tags=["Telemetry"])
async def telemetry_stats():
    return semantic_cache.get_telemetry()


@app.post(
    "/v1/index",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def index_documents(payload: IngestionRequest):
    embeddings_list = [dummy_text_to_embedding(doc.text) for doc in payload.documents]
    embeddings_matrix = np.vstack(embeddings_list)
    count = store.add_documents(payload.documents, embeddings_matrix)
    return IngestionResponse(ingested_count=count, status="success")


@app.post("/v1/retrieve", response_model=QueryResponse, tags=["Retrieval"])
async def retrieve_context(payload: QueryRequest):
    start = time.perf_counter()
    query_vec = dummy_text_to_embedding(payload.query)

    # 1. Semantic Cache Interception
    cached_hit, _cache_lookup_ms = semantic_cache.lookup(query_vec)
    if cached_hit:
        return QueryResponse(
            query=payload.query,
            results=cached_hit["results"],
            retrieved_count=len(cached_hit["results"]),
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    # 2. Vector Candidate Search
    candidates, _ = store.query(
        query_vec, top_k=payload.top_k * 2, threshold=payload.score_threshold
    )

    # 3. Reranking
    ranked = reranker.rerank(payload.query, candidates, top_k=payload.top_k)

    # 4. Guardrail Evaluation
    is_grounded, _, reason = guardrail.evaluate_groundedness(payload.query, ranked)
    if not is_grounded:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Query unanswerable with grounded context",
                "reason": reason,
            },
        )

    total_latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
    response = QueryResponse(
        query=payload.query,
        results=ranked,
        retrieved_count=len(ranked),
        latency_ms=total_latency_ms,
    )

    # Populate cache
    semantic_cache.insert(query_vec, {"results": ranked})
    return response

app.include_router(diagnostics_router)

from app.core.synthesizer import GroundedContextSynthesizer, GenerationResponse

synthesizer = GroundedContextSynthesizer()


@app.post("/v1/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate_response(payload: QueryRequest):
    """Unified end-to-end endpoint: retrieve relevant context chunks and synthesize citation-grounded response."""
    retrieval_res = await retrieve_context(payload)
    return synthesizer.synthesize(
        query=payload.query,
        context=retrieval_res.results,
        latency_ms=retrieval_res.latency_ms
    )

from app.core.synthesizer import GroundedContextSynthesizer, GenerationResponse

synthesizer = GroundedContextSynthesizer()


@app.post("/v1/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate_response(payload: QueryRequest):
    """Unified end-to-end endpoint: retrieve relevant context chunks and synthesize citation-grounded response."""
    retrieval_res = await retrieve_context(payload)
    return synthesizer.synthesize(
        query=payload.query,
        context=retrieval_res.results,
        latency_ms=retrieval_res.latency_ms
    )
