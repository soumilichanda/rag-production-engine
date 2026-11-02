from app.retrieval.cross_encoder_reranker import CrossEncoderAttentionReranker
from app.schemas.payloads import RetrievedChunk


def test_cross_encoder_scoring_and_ranking():
    reranker = CrossEncoderAttentionReranker()
    query = "kubernetes cluster scheduling pods"

    candidates = [
        RetrievedChunk(
            chunk_id="c1",
            text="Unrelated text about python syntax and indentation.",
            similarity_score=0.90,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Kubernetes cluster scheduling handles pods deployment across nodes.",
            similarity_score=0.60,
        ),
    ]

    reranked = reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
    assert (
        reranked[0].metadata["cross_encoder_score"]
        > reranked[1].metadata["cross_encoder_score"]
    )
    assert "reranked_composite" in reranked[0].metadata


def test_cross_encoder_empty_inputs():
    reranker = CrossEncoderAttentionReranker()
    assert reranker.rerank("query", []) == []
    assert reranker.score_pair("", "") == 0.0
