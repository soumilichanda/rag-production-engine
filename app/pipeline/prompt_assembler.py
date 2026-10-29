"""
Phase 16: Dynamic Prompt Context Assembler
Assembles compressed, relevance-ordered context chunks into bounded prompt templates.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class AssembledPrompt(BaseModel):
    system_prompt: str
    user_prompt: str
    formatted_context: str
    retained_chunks_count: int
    total_estimated_tokens: int
    provenance_sources: List[Dict[str, Any]]


class PromptContextAssembler:
    def __init__(
        self,
        max_context_tokens: int = 1500,
        tokens_per_word_ratio: float = 1.33,
    ):
        self.max_context_tokens = max_context_tokens
        self.tokens_per_word_ratio = tokens_per_word_ratio

    def estimate_tokens(self, text: str) -> int:
        words = len(text.strip().split())
        return max(1, int(words * self.tokens_per_word_ratio))

    def assemble(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> AssembledPrompt:
        # Sort by relevance score descending
        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda c: c.get("score", 0.0),
            reverse=True,
        )

        default_system = (
            system_instruction
            or "You are an accurate, grounded assistant. Answer the user query using strictly the provided context chunks. Cite source IDs in brackets."
        )

        accumulated_context: List[str] = []
        provenance: List[Dict[str, Any]] = []
        current_tokens = 0

        for idx, chunk in enumerate(sorted_chunks):
            doc_id = chunk.get("document_id", f"doc_{idx}")
            chunk_id = chunk.get("chunk_id", idx)
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            header = f"[Doc: {doc_id} | Chunk: {chunk_id}]"
            block = f"{header}\n{text}\n"
            block_tokens = self.estimate_tokens(block)

            if current_tokens + block_tokens > self.max_context_tokens:
                break

            accumulated_context.append(block)
            provenance.append({
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "score": score,
            })
            current_tokens += block_tokens

        formatted_context = "\n---\n".join(accumulated_context)
        user_prompt = f"Context:\n{formatted_context}\n\nQuery: {query}\nAnswer:"
        total_tokens = current_tokens + self.estimate_tokens(default_system) + self.estimate_tokens(query)

        return AssembledPrompt(
            system_prompt=default_system,
            user_prompt=user_prompt,
            formatted_context=formatted_context,
            retained_chunks_count=len(accumulated_context),
            total_estimated_tokens=total_tokens,
            provenance_sources=provenance,
        )
