"""
Phase 46: Hierarchical Parent-Document Retrieval & Small-to-Big Context Expansion.

Indexes documents across two structural tiers:
1. Parent Sections: Broad structural sections preserving full architectural/narrative context.
2. Child Chunks: Fine-grained, high-precision retrieval units linked via `parent_id`.

At query time, high-precision child hits expand upward ("small-to-big") to deduplicated
parent sections with sibling corroboration scoring.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class ParentSection:
    parent_id: str
    doc_id: str
    section_index: int
    heading: str
    text: str
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildChunk:
    child_id: str
    parent_id: str
    doc_id: str
    child_index: int
    text: str
    token_estimate: int
    embedding: tuple[float, ...]


def _hash_embed(text: str, dim: int = 64) -> tuple[float, ...]:
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return tuple(vec)
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return tuple(vec)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    return max(-1.0, min(1.0, dot))


class HierarchicalParentRetriever:
    """Two-tier parent/child index supporting small-to-big parent context expansion."""

    _HEADING_SPLIT_RE = re.compile(r"(?m)^(?=#{1,6}\s+.+)")

    def __init__(
        self,
        parent_max_tokens: int = 420,
        child_max_tokens: int = 96,
        child_overlap_tokens: int = 16,
    ) -> None:
        if child_max_tokens <= 0 or parent_max_tokens < child_max_tokens:
            raise ValueError("Require 0 < child_max_tokens <= parent_max_tokens")
        self.parent_max_tokens = parent_max_tokens
        self.child_max_tokens = child_max_tokens
        self.child_overlap_tokens = max(
            0, min(child_overlap_tokens, child_max_tokens // 2)
        )

        self._lock = RLock()
        self._parents: dict[str, ParentSection] = {}
        self._children: list[ChildChunk] = []
        self._documents_indexed = 0
        self._retrieval_calls = 0
        self._total_child_hits = 0
        self._total_parents_expanded = 0
        self._cumulative_expansion_ratio = 0.0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        words = text.split()
        return max(1, math.ceil(len(words) * 1.3)) if words else 0

    def _split_into_parent_blocks(self, text: str) -> list[tuple[str, str]]:
        raw_sections = [
            s.strip() for s in self._HEADING_SPLIT_RE.split(text.strip()) if s.strip()
        ]
        blocks: list[tuple[str, str]] = []

        for sec_idx, sec in enumerate(raw_sections):
            lines = sec.splitlines()
            first_line = lines[0].strip() if lines else ""
            heading = (
                first_line.lstrip("#").strip()
                if first_line.startswith("#")
                else f"Section {sec_idx + 1}"
            )

            words = sec.split()
            max_words = max(16, int(self.parent_max_tokens / 1.3))
            if len(words) <= max_words:
                blocks.append((heading, sec))
            else:
                for start in range(0, len(words), max_words):
                    part = " ".join(words[start : start + max_words]).strip()
                    if part:
                        blocks.append((heading, part))
        return blocks

    def _split_into_child_texts(self, parent_text: str) -> list[str]:
        words = parent_text.split()
        if not words:
            return []
        max_words = max(8, int(self.child_max_tokens / 1.3))
        overlap_words = int(self.child_overlap_tokens / 1.3)
        step = max(1, max_words - overlap_words)

        slices: list[str] = []
        for start in range(0, len(words), step):
            chunk_words = words[start : start + max_words]
            if not chunk_words:
                break
            slices.append(" ".join(chunk_words))
            if start + max_words >= len(words):
                break
        return slices

    def ingest_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return {"doc_id": doc_id, "parent_sections": 0, "child_chunks": 0}

        meta = dict(metadata or {})
        parent_blocks = self._split_into_parent_blocks(cleaned)
        new_parents: list[ParentSection] = []
        new_children: list[ChildChunk] = []

        for p_idx, (heading, p_text) in enumerate(parent_blocks):
            parent_id = f"{doc_id}::parent_{p_idx:03d}"
            parent = ParentSection(
                parent_id=parent_id,
                doc_id=doc_id,
                section_index=p_idx,
                heading=heading,
                text=p_text,
                token_estimate=self._estimate_tokens(p_text),
                metadata=meta,
            )
            new_parents.append(parent)

            child_texts = self._split_into_child_texts(p_text)
            for c_idx, c_text in enumerate(child_texts):
                child = ChildChunk(
                    child_id=f"{parent_id}::child_{c_idx:03d}",
                    parent_id=parent_id,
                    doc_id=doc_id,
                    child_index=c_idx,
                    text=c_text,
                    token_estimate=self._estimate_tokens(c_text),
                    embedding=_hash_embed(c_text),
                )
                new_children.append(child)

        with self._lock:
            for p in new_parents:
                self._parents[p.parent_id] = p
            self._children.extend(new_children)
            self._documents_indexed += 1

        return {
            "doc_id": doc_id,
            "parent_sections": len(new_parents),
            "child_chunks": len(new_children),
        }

    def retrieve(
        self,
        query: str,
        top_k_parents: int = 3,
        child_candidate_pool: int = 12,
    ) -> dict[str, Any]:
        q_clean = query.strip()
        if not q_clean:
            return {
                "query": query,
                "results": [],
                "child_hits_evaluated": 0,
                "parents_returned": 0,
                "context_expansion_ratio": 1.0,
            }

        q_vec = _hash_embed(q_clean)
        q_tokens = set(re.findall(r"[a-z0-9_]+", q_clean.lower()))

        with self._lock:
            children_snapshot = list(self._children)
            parents_snapshot = dict(self._parents)

        if not children_snapshot:
            return {
                "query": query,
                "results": [],
                "child_hits_evaluated": 0,
                "parents_returned": 0,
                "context_expansion_ratio": 1.0,
            }

        scored_children: list[tuple[float, ChildChunk]] = []
        for child in children_snapshot:
            cos_sim = _cosine(q_vec, child.embedding)
            c_tokens = set(re.findall(r"[a-z0-9_]+", child.text.lower()))
            lexical = (len(q_tokens & c_tokens) / len(q_tokens)) if q_tokens else 0.0
            blended = round(0.65 * max(0.0, cos_sim) + 0.35 * lexical, 6)
            scored_children.append((blended, child))

        scored_children.sort(key=lambda pair: pair[0], reverse=True)
        top_children = scored_children[: max(1, child_candidate_pool)]

        parent_buckets: dict[str, dict[str, Any]] = {}
        for score, child in top_children:
            parent = parents_snapshot.get(child.parent_id)
            if parent is None:
                continue
            bucket = parent_buckets.get(parent.parent_id)
            if bucket is None:
                parent_buckets[parent.parent_id] = {
                    "parent": parent,
                    "best_child_score": score,
                    "matched_children": [
                        {
                            "child_id": child.child_id,
                            "child_index": child.child_index,
                            "score": round(score, 4),
                            "snippet": child.text,
                            "token_estimate": child.token_estimate,
                        }
                    ],
                }
            else:
                bucket["best_child_score"] = max(bucket["best_child_score"], score)
                bucket["matched_children"].append(
                    {
                        "child_id": child.child_id,
                        "child_index": child.child_index,
                        "score": round(score, 4),
                        "snippet": child.text,
                        "token_estimate": child.token_estimate,
                    }
                )

        ranked_parents: list[dict[str, Any]] = []
        for pid, data in parent_buckets.items():
            parent_obj: ParentSection = data["parent"]
            hits = len(data["matched_children"])
            # Corroboration boost when multiple sibling child chunks match within the same parent section
            corroborated_score = round(
                data["best_child_score"] * (1.0 + 0.12 * (hits - 1)), 4
            )
            ranked_parents.append(
                {
                    "parent_id": pid,
                    "doc_id": parent_obj.doc_id,
                    "section_index": parent_obj.section_index,
                    "heading": parent_obj.heading,
                    "parent_text": parent_obj.text,
                    "parent_tokens": parent_obj.token_estimate,
                    "aggregated_score": corroborated_score,
                    "matched_child_count": hits,
                    "matched_children": data["matched_children"],
                    "metadata": parent_obj.metadata,
                }
            )

        ranked_parents.sort(key=lambda item: item["aggregated_score"], reverse=True)
        selected_parents = ranked_parents[: max(1, top_k_parents)]

        matched_child_tokens = sum(
            c["token_estimate"] for p in selected_parents for c in p["matched_children"]
        )
        expanded_parent_tokens = sum(p["parent_tokens"] for p in selected_parents)
        expansion_ratio = (
            round(expanded_parent_tokens / max(1, matched_child_tokens), 3)
            if selected_parents
            else 1.0
        )

        with self._lock:
            self._retrieval_calls += 1
            self._total_child_hits += len(top_children)
            self._total_parents_expanded += len(selected_parents)
            self._cumulative_expansion_ratio += expansion_ratio

        return {
            "query": query,
            "results": selected_parents,
            "child_hits_evaluated": len(top_children),
            "parents_returned": len(selected_parents),
            "context_expansion_ratio": expansion_ratio,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            avg_ratio = (
                round(self._cumulative_expansion_ratio / self._retrieval_calls, 3)
                if self._retrieval_calls > 0
                else 1.0
            )
            return {
                "documents_indexed": self._documents_indexed,
                "total_parent_sections": len(self._parents),
                "total_child_chunks": len(self._children),
                "retrieval_calls": self._retrieval_calls,
                "total_child_hits": self._total_child_hits,
                "total_parents_expanded": self._total_parents_expanded,
                "average_context_expansion_ratio": avg_ratio,
                "parent_max_tokens": self.parent_max_tokens,
                "child_max_tokens": self.child_max_tokens,
            }
