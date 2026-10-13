"""
app/main.py
FastAPI gateway for chunk indexing, vector retrieval, and reranking.
"""

import time

import numpy as np
from fastapi import FastAPI, HTTPException, status

from app.core.guardrails import GroundednessGuardrail
from app.core.reranker import ExactTokenOverlapReranker
from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.payloads import (
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(
    title="Production RAG Engine",
    description="Vector retrieval microservice with semantic reranking and hallucination guardrails",
    version="1.0.0"
)

# Shared in-memory engine instances
DIMENSION = 8
store = InMemoryVectorStore(dimension=DIMENSION)
reranker = ExactTokenOverlapReranker()
guardrail = GroundednessGuardrail(min_confidence=0.20)


def dummy_text_to_embedding(text: str, dim: int = DIMENSION) -> np.ndarray:
    """Deterministic hash-based embedding generator for reproducible retrieval tests."""
    vec = np.zeros(dim, dtype=np.float32)
    for i, char in enumerate(text.encode("utf-8")):
        vec[i % dim] += float(char)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "indexed_documents": len(store.chunks)}


@app.post("/v1/index", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
async def index_documents(payload: IngestionRequest):
    embeddings_list = [dummy_text_to_embedding(doc.text) for doc in payload.documents]
    embeddings_matrix = np.vstack(embeddings_list)
    count = store.add_documents(payload.documents, embeddings_matrix)
    return IngestionResponse(ingested_count=count, status="success")


@app.post("/v1/retrieve", response_model=QueryResponse, tags=["Retrieval"])
async def retrieve_context(payload: QueryRequest):
    start = time.perf_counter()
    query_vec = dummy_text_to_embedding(payload.query)

    # 1. Bi-Encoder candidate retrieval
    candidates, _ = store.query(query_vec, top_k=payload.top_k * 2, threshold=payload.score_threshold)

    # 2. Semantic reranking
    ranked = reranker.rerank(payload.query, candidates, top_k=payload.top_k)

    # 3. Groundedness guardrail check
    is_grounded, _top_score, reason = guardrail.evaluate_groundedness(payload.query, ranked)
    if not is_grounded:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Query unanswerable with grounded context", "reason": reason}
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    return QueryResponse(
        query=payload.query,
        results=ranked,
        retrieved_count=len(ranked),
        latency_ms=round(latency_ms, 2)
    )
