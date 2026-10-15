"""
app/core/guardrails.py
Groundedness verification and context guardrails for hallucination prevention.
"""

from app.schemas.payloads import RetrievedChunk


class GroundednessGuardrail:
    """
    Validates that retrieved contexts contain sufficient signal to support answering,
    preventing hallucinations caused by out-of-domain queries or empty context.
    """

    def __init__(self, min_confidence: float = 0.25):
        self.min_confidence = min_confidence

    def evaluate_groundedness(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[bool, float, str]:
        if not chunks:
            return (
                False,
                0.0,
                "Zero context chunks retrieved: Query rejected to prevent hallucination.",
            )

        top_score = max(chunk.similarity_score for chunk in chunks)
        if top_score < self.min_confidence:
            return (
                False,
                top_score,
                f"Top similarity score ({top_score:.3f}) below groundedness threshold ({self.min_confidence:.3f}).",
            )

        return True, top_score, "Context adequately grounded for generation."
