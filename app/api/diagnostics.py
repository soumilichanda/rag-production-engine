"""
app/api/diagnostics.py
System diagnostics, vector cache telemetry, and Kubernetes liveness probes.
"""

import time
from fastapi import APIRouter

router = APIRouter(prefix="/v1/diagnostics", tags=["diagnostics"])


@router.get("/health")
async def health_check():
    """Liveness probe returning process uptime status."""
    return {
        "status": "healthy",
        "service": "rag-production-engine",
        "timestamp": time.time(),
        "ready": True
    }


@router.get("/telemetry")
async def telemetry_probe():
    """Readiness probe reporting operational telemetry parameters."""
    return {
        "status": "operational",
        "cache_layer": "lru_semantic_bounded",
        "vector_indexing": "normalized_cosine",
        "timestamp": time.time()
    }
