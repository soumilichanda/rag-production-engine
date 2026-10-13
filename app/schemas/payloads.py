"""
Strict Pydantic V2 data contracts for document ingestion and retrieval queries.
"""

from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Deterministic unique identifier for the chunk")
    text: str = Field(..., min_length=5, description="Raw chunk text payload")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dictionary")


class IngestionRequest(BaseModel):
    documents: list[DocumentChunk] = Field(..., min_length=1, description="List of document chunks to index")


class IngestionResponse(BaseModel):
    ingested_count: int
    status: str = "success"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language search query")
    top_k: int = Field(default=3, ge=1, le=20, description="Maximum number of chunks to retrieve")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff")


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    similarity_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    retrieved_count: int
    latency_ms: float
