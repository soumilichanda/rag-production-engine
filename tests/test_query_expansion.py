import pytest

from app.retrieval.fusion import RRFAggregator
from app.retrieval.query_expander import MultiQueryRetriever
from app.schemas.payloads import RetrievedChunk


class MockVectorStore:
    def __init__(self):
        self.corpus = [
            RetrievedChunk(
                chunk_id="c1",
                text="Kahn algorithm detects cycles in directed graphs.",
                similarity_score=0.92,
                metadata={"source": "dsa"},
            ),
            RetrievedChunk(
                chunk_id="c2",
                text="Breadth-first search discovers shortest transformation sequences.",
                similarity_score=0.88,
                metadata={"source": "dsa"},
            ),
            RetrievedChunk(
                chunk_id="c3",
                text="FastAPI routes mount asynchronous endpoints for web retrieval.",
                similarity_score=0.75,
                metadata={"source": "web"},
            ),
        ]

    def query(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> list[RetrievedChunk]:
        return self.corpus[:top_k]


@pytest.fixture
def mock_store():
    return MockVectorStore()


@pytest.mark.asyncio
async def test_multiquery_perspectives_generation(mock_store):
    retriever = MultiQueryRetriever(mock_store)
    perspectives = retriever.generate_perspectives("graph cycle detection")

    assert len(perspectives) == 3
    assert perspectives[0] == "graph cycle detection"
    assert any("technical overview" in p for p in perspectives)


@pytest.mark.asyncio
async def test_multiquery_retrieval_and_rrf_fusion(mock_store):
    retriever = MultiQueryRetriever(mock_store)
    aggregator = RRFAggregator(k=60)

    ranked_lists = await retriever.retrieve_perspectives("graphs", top_k=2)
    assert len(ranked_lists) == 3

    fused = aggregator.fuse(ranked_lists, top_k=2)
    assert len(fused) <= 2
    assert all("fusion_strategy" in item.metadata for item in fused)
    assert all(item.similarity_score > 0 for item in fused)
