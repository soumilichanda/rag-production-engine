"""
Async pytest suite for Phase 42: Adaptive Query Decomposition & Sub-Question Execution Planner.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, knowledge_graph_index
from app.retrieval.query_planner import (
    AdaptiveQueryDecomposer,
    SubQuestionExecutionPlanner,
)


def test_decomposer_splits_comparative_and_conjunctive_queries():
    decomposer = AdaptiveQueryDecomposer(max_sub_queries=4)

    atomic_plan = decomposer.build_plan("What is HNSW indexing?")
    assert atomic_plan.strategy == "direct"
    assert len(atomic_plan.steps) == 1

    comp_plan = decomposer.build_plan(
        "Compare HNSW vector search vs BM25 lexical retrieval"
    )
    assert comp_plan.strategy == "decomposed"
    assert len(comp_plan.steps) >= 2
    assert comp_plan.complexity_score > atomic_plan.complexity_score

    multi_plan = decomposer.build_plan(
        "How does WAL rotation work and how does snapshot compaction truncate logs?"
    )
    assert multi_plan.strategy == "decomposed"
    assert len(multi_plan.steps) == 2


@pytest.mark.asyncio
async def test_planner_parallel_execution_and_coverage_boost():
    planner = SubQuestionExecutionPlanner(
        decomposer=AdaptiveQueryDecomposer(max_sub_queries=4)
    )

    async def mock_retriever(sub_q: str, top_k: int):
        if "wal" in sub_q.lower():
            return [
                {
                    "chunk_id": "shared_arch",
                    "text": "WAL and snapshots coordinate durability.",
                },
                {
                    "chunk_id": "wal_only",
                    "text": "Segmented WAL rolls over at byte limits.",
                },
            ]
        return [
            {
                "chunk_id": "shared_arch",
                "text": "WAL and snapshots coordinate durability.",
            },
            {
                "chunk_id": "snap_only",
                "text": "Snapshot engine writes atomic checkpoints.",
            },
        ]

    res = await planner.execute(
        query="Explain WAL rotation and snapshot compaction",
        retriever_fn=mock_retriever,
        top_k=3,
    )
    assert res["strategy"] == "decomposed"
    assert len(res["sub_queries"]) == 2
    assert res["merged_results"][0]["chunk_id"] == "shared_arch"
    assert res["merged_results"][0]["sub_query_hits"] == 2
    assert "WAL and snapshots coordinate durability." in res["synthesized_context"]


def test_gateway_planner_execute_and_telemetry_endpoints():
    knowledge_graph_index.ingest_chunk(
        "kg_p42",
        "[QueryPlanner -> decomposes -> MultiHopQuery] and [SubQuestionExecutor -> merges -> HybridContext]",
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/planner/execute",
        json={
            "query": "How does QueryPlanner work and how does SubQuestionExecutor merge context?",
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "decomposed"
    assert len(body["sub_queries"]) >= 2

    telem = client.get("/v1/planner/telemetry")
    assert telem.status_code == 200
    assert telem.json()["status"] == "operational"
    assert telem.json()["planner"]["total_plans"] >= 1
