"""
app/ingestion/chunker.py
Deterministic recursive text chunker with sliding window token overlap
and SHA-256 chunk hash generation for drift detection.
"""

import hashlib
import re
from typing import Any, ClassVar

from app.schemas.payloads import DocumentChunk


class RecursiveTokenChunker:
    """
    Recursively splits text on natural linguistic boundaries (paragraphs, sentences, words)
    while enforcing bounded chunk character sizes and sliding overlap windows.
    """

    SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def compute_sha256(text: str) -> str:
        """Generates a deterministic 16-character hexadecimal chunk hash."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return list(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        result: list[str] = []

        for piece in splits:
            if not piece:
                continue
            if len(piece) > self.chunk_size and remaining_seps:
                result.extend(self._split_text(piece, remaining_seps))
            else:
                result.append(piece)

        return result

    def split_text(
        self, text: str, base_metadata: dict[str, Any] | None = None
    ) -> list[DocumentChunk]:
        """
        Splits raw text into bounded DocumentChunks with sliding overlap windows
        and deterministic hash identifiers.
        """
        if base_metadata is None:
            base_metadata = {}

        normalized = re.sub(r"\r\n|\r", "\n", text.strip())
        if not normalized:
            return []

        raw_pieces = self._split_text(normalized, self.SEPARATORS)

        chunks: list[DocumentChunk] = []
        current_chunk_tokens: list[str] = []
        current_length = 0

        for piece in raw_pieces:
            piece_len = len(piece) + 1  # Accounting for separator space
            if current_length + piece_len > self.chunk_size and current_chunk_tokens:
                chunk_str = " ".join(current_chunk_tokens).strip()
                if len(chunk_str) >= 5:
                    chunk_hash = f"chk_{self.compute_sha256(chunk_str)}"
                    meta = {
                        **base_metadata,
                        "char_len": len(chunk_str),
                        "hash": chunk_hash,
                    }
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_hash, text=chunk_str, metadata=meta
                        )
                    )

                overlap_tokens: list[str] = []
                overlap_length = 0
                for token in reversed(current_chunk_tokens):
                    if overlap_length + len(token) <= self.chunk_overlap:
                        overlap_tokens.insert(0, token)
                        overlap_length += len(token) + 1
                    else:
                        break
                current_chunk_tokens = overlap_tokens
                current_length = sum(len(t) + 1 for t in current_chunk_tokens)

            current_chunk_tokens.append(piece)
            current_length += piece_len

        if current_chunk_tokens:
            final_str = " ".join(current_chunk_tokens).strip()
            if len(final_str) >= 5:
                chunk_hash = f"chk_{self.compute_sha256(final_str)}"
                meta = {**base_metadata, "char_len": len(final_str), "hash": chunk_hash}
                chunks.append(
                    DocumentChunk(chunk_id=chunk_hash, text=final_str, metadata=meta)
                )

        return chunks
