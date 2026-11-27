"""
Phase 45: Adaptive Semantic Chunking & Structural Boundary Splitting.

Decomposes documents along structural anchors (headings, paragraphs) and detects
topical shift breakpoints using sliding-window sentence embedding cosine valleys.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticChunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    token_estimate: int
    boundary_reason: str  # "structural" | "semantic_shift" | "max_tokens" | "eof"
    coherence_score: float


@dataclass
class SemanticChunkingTelemetry:
    documents_processed: int = 0
    total_chunks_emitted: int = 0
    semantic_breakpoints: int = 0
    structural_splits: int = 0
    max_token_splits: int = 0
    cumulative_coherence: float = 0.0

    @property
    def avg_coherence(self) -> float:
        if self.total_chunks_emitted == 0:
            return 0.0
        return round(self.cumulative_coherence / self.total_chunks_emitted, 4)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "documents_processed": self.documents_processed,
            "total_chunks_emitted": self.total_chunks_emitted,
            "semantic_breakpoints": self.semantic_breakpoints,
            "structural_splits": self.structural_splits,
            "max_token_splits": self.max_token_splits,
            "avg_coherence": self.avg_coherence,
        }


def _default_hash_embedder(text: str, dim: int = 64) -> list[float]:
    """Deterministic L2-normalized token-feature embedding for zero-dependency execution."""
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class AdaptiveSemanticChunker:
    """Splits documents at structural boundaries and cosine-similarity topical valleys."""

    _HEADING_RE = re.compile(r"(?m)^(?=#{1,6}\s+.+)")
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        max_chunk_tokens: int = 180,
        min_chunk_tokens: int = 24,
        window_size: int = 1,
        similarity_threshold: float = 0.35,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        if min_chunk_tokens <= 0 or max_chunk_tokens < min_chunk_tokens:
            raise ValueError("Invalid token bounds: require 0 < min_chunk_tokens <= max_chunk_tokens")
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.window_size = max(0, window_size)
        self.similarity_threshold = similarity_threshold
        self.embed_fn = embed_fn or _default_hash_embedder
        self.telemetry = SemanticChunkingTelemetry()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        words = text.split()
        return max(1, math.ceil(len(words) * 1.3)) if words else 0

    def _split_structural_blocks(self, text: str) -> list[str]:
        sections = [s.strip() for s in self._HEADING_RE.split(text.strip()) if s.strip()]
        blocks: list[str] = []
        for sec in sections:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec) if p.strip()]
            blocks.extend(paragraphs)
        return blocks

    def _split_sentences(self, block: str) -> list[str]:
        sentences = [s.strip() for s in self._SENTENCE_RE.split(block.strip()) if s.strip()]
        return sentences if sentences else ([block.strip()] if block.strip() else [])

    def _build_window_embeddings(self, sentences: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        n = len(sentences)
        for i in range(n):
            lo = max(0, i - self.window_size)
            hi = min(n, i + self.window_size + 1)
            window_text = " ".join(sentences[lo:hi])
            embeddings.append(self.embed_fn(window_text))
        return embeddings

    def chunk_document(self, doc_id: str, text: str) -> list[SemanticChunk]:
        cleaned = text.strip()
        if not cleaned:
            return []

        blocks = self._split_structural_blocks(cleaned)
        chunks: list[SemanticChunk] = []
        current_sents: list[str] = []
        current_vecs: list[list[float]] = []
        current_tokens = 0

        def _flush(reason: str) -> None:
            nonlocal current_sents, current_vecs, current_tokens
            if not current_sents:
                return
            joined = " ".join(current_sents).strip()
            if len(current_vecs) <= 1:
                coherence = 1.0
            else:
                sims = [
                    _cosine_similarity(current_vecs[i], current_vecs[i + 1])
                    for i in range(len(current_vecs) - 1)
                ]
                coherence = round(max(0.0, sum(sims) / len(sims)), 4)

            idx = len(chunks)
            chunks.append(
                SemanticChunk(
                    chunk_id=f"{doc_id}::chunk_{idx:04d}",
                    doc_id=doc_id,
                    index=idx,
                    text=joined,
                    token_estimate=self.estimate_tokens(joined),
                    boundary_reason=reason,
                    coherence_score=coherence,
                )
            )
            if reason == "semantic_shift":
                self.telemetry.semantic_breakpoints += 1
            elif reason == "structural":
                self.telemetry.structural_splits += 1
            elif reason == "max_tokens":
                self.telemetry.max_token_splits += 1
            self.telemetry.cumulative_coherence += coherence

            current_sents = []
            current_vecs = []
            current_tokens = 0

        for b_idx, block in enumerate(blocks):
            is_heading = block.startswith("#")
            if is_heading and current_tokens >= self.min_chunk_tokens:
                _flush("structural")

            sentences = self._split_sentences(block)
            if not sentences:
                continue
            vecs = self._build_window_embeddings(sentences)

            for sent, vec in zip(sentences, vecs, strict=False):
                sent_tokens = self.estimate_tokens(sent)

                if current_sents and (current_tokens + sent_tokens > self.max_chunk_tokens):
                    _flush("max_tokens")

                if current_sents and current_tokens >= self.min_chunk_tokens:
                    sim = _cosine_similarity(current_vecs[-1], vec)
                    if sim < self.similarity_threshold:
                        _flush("semantic_shift")

                current_sents.append(sent)
                current_vecs.append(vec)
                current_tokens += sent_tokens

            if b_idx < len(blocks) - 1 and blocks[b_idx + 1].startswith("#") and current_sents:
                _flush("structural")

        if current_sents:
            _flush("eof")

        self.telemetry.documents_processed += 1
        self.telemetry.total_chunks_emitted += len(chunks)
        return chunks
