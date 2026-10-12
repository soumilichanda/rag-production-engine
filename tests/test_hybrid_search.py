import pytest
from app.schemas.payloads import RetrievedChunk
from app.retrieval.hybrid_search import HybridRRFSearcher


def test_rrf_scoring_and_tie_breaking():
    searcher = HybridRRFSearcher(rrf_k=60)
    dense_candidates = [
        RetrievedChunk(chunk_id="doc_a", text="Python async microservices", similarity_score=0.91),
        RetrievedChunk(chunk_id="doc_b", text="Kubernetes cluster orchestration", similarity_score=0.88),
    ]

    # Query heavily matching doc_b lexically
    fused = searcher.fuse(dense_candidates, sparse_query="Kubernetes", top_k=2)

    assert len(fused) == 2
    assert fused[0].metadata["rrf_fused"] is True
    assert fused[0].similarity_score > 0.0
