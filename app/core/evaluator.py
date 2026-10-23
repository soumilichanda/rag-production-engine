"""
RAG Triad Production Evaluator Module
Computes Context Relevance, Faithfulness, and Answer Relevance metrics.
"""

from typing import List, Dict, Any
import numpy as np


class RAGTriadEvaluator:
    def __init__(self, relevance_threshold: float = 0.70, faithfulness_threshold: float = 0.75):
        self.relevance_threshold = relevance_threshold
        self.faithfulness_threshold = faithfulness_threshold

    def evaluate_context_relevance(self, query_tokens: set[str], retrieved_contexts: List[str]) -> float:
        if not retrieved_contexts or not query_tokens:
            return 0.0

        scores = []
        for ctx in retrieved_contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(query_tokens.intersection(ctx_tokens))
            scores.append(overlap / max(1, len(query_tokens)))

        return float(np.mean(scores))

    def evaluate_faithfulness(self, synthesized_answer: str, retrieved_contexts: List[str]) -> float:
        if not synthesized_answer or not retrieved_contexts:
            return 0.0

        all_context_tokens = set(" ".join(retrieved_contexts).lower().split())
        answer_tokens = set(synthesized_answer.lower().split())

        supported_tokens = answer_tokens.intersection(all_context_tokens)
        return float(len(supported_tokens) / max(1, len(answer_tokens)))

    def evaluate_triad(
        self, query: str, retrieved_contexts: List[str], synthesized_answer: str
    ) -> Dict[str, Any]:
        q_tokens = set(query.lower().split())
        ctx_relevance = self.evaluate_context_relevance(q_tokens, retrieved_contexts)
        faithfulness = self.evaluate_faithfulness(synthesized_answer, retrieved_contexts)
        
        # Answer relevance proxy via bidirectional term recall
        ans_tokens = set(synthesized_answer.lower().split())
        ans_relevance = float(len(q_tokens.intersection(ans_tokens)) / max(1, len(q_tokens)))

        return {
            "context_relevance": round(ctx_relevance, 4),
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(ans_relevance, 4),
            "passed_triad": bool(
                ctx_relevance >= self.relevance_threshold and faithfulness >= self.faithfulness_threshold
            ),
        }
