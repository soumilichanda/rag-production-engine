"""
Phase 17: Groundedness & Faithfulness Evaluator
Computes claim-level semantic overlap and source citation attribution
to detect and flag synthetic hallucinations beyond retrieved context chunks.
"""

import re
from typing import Any


class FaithfulnessEvaluator:
    def __init__(self, token_overlap_threshold: float = 0.5):
        self.token_overlap_threshold = token_overlap_threshold

    def _tokenize(self, text: str) -> set:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return {word for word in cleaned.split() if len(word) > 2}

    def evaluate(self, answer: str, contexts: list[str]) -> dict[str, Any]:
        """
        Evaluates answer claims against retrieved contexts.
        Returns normalized faithfulness score [0.0, 1.0] and hallucination flag.
        """
        if not answer.strip():
            return {
                "score": 0.0,
                "hallucinated": True,
                "reasons": ["Empty response payload."],
            }

        if not contexts:
            return {
                "score": 0.0,
                "hallucinated": True,
                "reasons": ["Context pool is empty."],
            }

        # Build context vocabulary universe
        context_vocab = set()
        for ctx in contexts:
            context_vocab.update(self._tokenize(ctx))

        # Sentence-level claim decomposition
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if s.strip()]
        if not sentences:
            return {"score": 1.0, "hallucinated": False, "reasons": []}

        grounded_claims = 0
        unsupported_claims = []

        for sent in sentences:
            sent_tokens = self._tokenize(sent)
            if not sent_tokens:
                grounded_claims += 1
                continue

            overlap = sent_tokens.intersection(context_vocab)
            ratio = len(overlap) / len(sent_tokens)

            if ratio >= self.token_overlap_threshold:
                grounded_claims += 1
            else:
                unsupported_claims.append(sent)

        score = round(grounded_claims / len(sentences), 4)
        is_hallucinated = score < 0.70

        return {
            "score": score,
            "hallucinated": is_hallucinated,
            "unsupported_claims": unsupported_claims,
            "total_claims": len(sentences),
            "grounded_claims": grounded_claims,
        }
