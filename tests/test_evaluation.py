from app.core.evaluator import RAGTriadEvaluator


def test_rag_triad_perfect_grounding():
    evaluator = RAGTriadEvaluator(relevance_threshold=0.5, faithfulness_threshold=0.5)
    query = "vector database indexing"
    contexts = [
        "vector database indexing accelerates nearest neighbor search",
        "indexing strategies optimize high dimensional vectors",
    ]
    answer = "vector database indexing accelerates nearest neighbor search"
    metrics = evaluator.evaluate_triad(query, contexts, answer)

    assert metrics["context_relevance"] > 0.5
    assert metrics["faithfulness"] == 1.0
    assert metrics["passed_triad"] is True


def test_rag_triad_hallucination_detection():
    evaluator = RAGTriadEvaluator(faithfulness_threshold=0.8)
    query = "system memory requirements"
    contexts = ["allocated 16 gigabytes ram for the container"]
    answer = "the system requires 64 gigabytes gpu vram for inference"
    metrics = evaluator.evaluate_triad(query, contexts, answer)

    assert metrics["faithfulness"] < 0.6
    assert metrics["passed_triad"] is False
