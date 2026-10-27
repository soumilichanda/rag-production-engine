"""
Phase 15: Cross-Encoder Context Compression & Extractive Reranking.
Extracts high-salience semantic sentences to optimize prompt token budgets.
"""

from typing import List, Dict, Any


class CrossEncoderContextCompressor:
    def __init__(self, max_tokens_per_chunk: int = 128):
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def compress_chunk(self, query: str, text: str) -> str:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return text

        query_terms = set(query.lower().split())

        def score_sentence(s: str) -> float:
            words = set(s.lower().split())
            overlap = len(words.intersection(query_terms))
            return overlap / (len(words) + 1e-5)

        scored = sorted(sentences, key=score_sentence, reverse=True)
        selected = []
        token_count = 0

        for sent in scored:
            sent_tokens = len(sent.split())
            if token_count + sent_tokens <= self.max_tokens_per_chunk:
                selected.append(sent)
                token_count += sent_tokens

        return ". ".join(selected) + ("." if selected else "")

    def compress_candidates(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        compressed = []
        for cand in candidates:
            item = dict(cand)
            original_text = item.get("text", "")
            item["compressed_text"] = self.compress_chunk(query, original_text)
            item["original_length"] = len(original_text)
            item["compressed_length"] = len(item["compressed_text"])
            compressed.append(item)
        return compressed
