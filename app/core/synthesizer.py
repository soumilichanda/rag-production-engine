"""
app/core/synthesizer.py
Context synthesis and citation extraction engine.
Combines retrieved context chunks into deterministic grounded responses.
"""

from pydantic import BaseModel, Field

from app.schemas.payloads import RetrievedChunk


class GenerationResponse(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(
        default_factory=list, description="IDs of cited context chunks"
    )
    retrieved_count: int = 0
    latency_ms: float = 0.0


class GroundedContextSynthesizer:
    """
    Synthesizes answers grounded exclusively in retrieved context passages.
    Extracts citation references and guarantees non-hallucinatory fallbacks.
    """

    def synthesize(
        self, query: str, context: list[RetrievedChunk], latency_ms: float = 0.0
    ) -> GenerationResponse:
        if not context:
            return GenerationResponse(
                query=query,
                answer="No relevant context could be retrieved to answer this query.",
                citations=[],
                retrieved_count=0,
                latency_ms=round(latency_ms, 2),
            )

        citations = [chunk.chunk_id for chunk in context]
        snippet_text = " ".join([chunk.text.strip() for chunk in context[:2]])
        cited_tags = ", ".join([f"[{cid}]" for cid in citations])
        answer = f"According to verified context {cited_tags}: {snippet_text}"

        return GenerationResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_count=len(context),
            latency_ms=round(latency_ms, 2),
        )
