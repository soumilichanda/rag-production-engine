import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.evaluation.faithfulness import FaithfulnessEvaluator


def test_evaluator_direct_grounded():
    evaluator = FaithfulnessEvaluator(token_overlap_threshold=0.4)
    ctx = ["Vector search relies on cosine similarity in high-dimensional hyperspheres."]
    ans = "Cosine similarity is computed in vector search."
    res = evaluator.evaluate(ans, ctx)
    assert res["score"] >= 0.70
    assert not res["hallucinated"]


def test_evaluator_direct_hallucination():
    evaluator = FaithfulnessEvaluator(token_overlap_threshold=0.5)
    ctx = ["Apples and oranges are common orchard fruits."]
    ans = "Quantum computing uses superconductors to calculate matrix determinants."
    res = evaluator.evaluate(ans, ctx)
    assert res["score"] < 0.70
    assert res["hallucinated"]


@pytest.mark.asyncio
async def test_evaluate_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "answer": "FastAPI is a modern web framework with high async performance support.",
            "contexts": ["FastAPI is a modern web framework for Python with async performance support."],
            "threshold": 0.4
        }
        resp = await ac.post("/v1/pipeline/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "evaluation" in data
        assert data["evaluation"]["score"] >= 0.70
        assert not data["evaluation"]["hallucinated"]
