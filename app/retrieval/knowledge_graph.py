"""
Phase 41: Hybrid Graph-RAG Entity Extraction & Multi-Hop Knowledge Traversal.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from typing import Any


class KnowledgeGraphIndex:
    """Thread-safe in-memory knowledge graph for entity-relation extraction and 2-hop traversal."""

    _ARROW_PATTERN = re.compile(
        r"\[([A-Za-z0-9_.-]+)\s*->\s*([A-Za-z0-9_.-]+)\s*->\s*([A-Za-z0-9_.-]+)\]"
    )
    _SVO_PATTERN = re.compile(
        r"\b([A-Z][A-Za-z0-9_-]{1,30})\s+"
        r"(uses|connects|manages|routes|stores|verifies|implements|calls|depends_on|indexes)\s+"
        r"([A-Z][A-Za-z0-9_-]{1,30})\b"
    )

    def __init__(self, max_hops: int = 2, max_neighbors_per_hop: int = 12) -> None:
        self.max_hops = max_hops
        self.max_neighbors_per_hop = max_neighbors_per_hop
        self._lock = threading.RLock()
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        self._entity_chunks: dict[str, set[str]] = {}
        self._chunk_texts: dict[str, str] = {}
        self._triples: set[tuple[str, str, str]] = set()
        self._queries_served = 0

    @staticmethod
    def _norm(token: str) -> str:
        return token.strip().lower().replace(" ", "_")

    def extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """Extract deterministic (subject, relation, object) triples from raw or structured text."""
        found: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        for subj, rel, obj in self._ARROW_PATTERN.findall(text):
            triple = (self._norm(subj), self._norm(rel), self._norm(obj))
            if triple not in seen and triple[0] != triple[2]:
                seen.add(triple)
                found.append(triple)

        for subj, rel, obj in self._SVO_PATTERN.findall(text):
            triple = (self._norm(subj), self._norm(rel), self._norm(obj))
            if triple not in seen and triple[0] != triple[2]:
                seen.add(triple)
                found.append(triple)

        return found

    def ingest_chunk(self, chunk_id: str, text: str) -> list[tuple[str, str, str]]:
        """Extract triples from a chunk and index its entity-relation edges and provenance."""
        triples = self.extract_triples(text)
        with self._lock:
            self._chunk_texts[chunk_id] = text
            for subj, rel, obj in triples:
                self._triples.add((subj, rel, obj))
                self._adj.setdefault(subj, []).append((rel, obj, chunk_id))
                self._adj.setdefault(obj, []).append((f"inv_{rel}", subj, chunk_id))
                self._entity_chunks.setdefault(subj, set()).add(chunk_id)
                self._entity_chunks.setdefault(obj, set()).add(chunk_id)
        return triples

    def ingest_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Batch-ingest a list of chunk dicts containing chunk_id and text."""
        total_triples = 0
        for idx, ch in enumerate(chunks):
            cid = str(ch.get("chunk_id") or f"graph_chunk_{idx}")
            triples = self.ingest_chunk(cid, str(ch.get("text", "")))
            total_triples += len(triples)
        return total_triples

    def _match_seed_entities(self, query: str) -> list[str]:
        q_norm = re.sub(r"[^\w\s-]", " ", query.lower())
        tokens = {tok for tok in q_norm.split() if len(tok) > 1}
        with self._lock:
            seeds = [
                ent
                for ent in self._adj
                if ent in tokens or any(part in tokens for part in ent.split("_"))
            ]
        return sorted(seeds)

    def traverse_2hop(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """Perform bounded multi-hop BFS traversal from query entities and rank supporting chunks."""
        seeds = self._match_seed_entities(query)
        with self._lock:
            self._queries_served += 1
            if not seeds:
                return {
                    "seed_entities": [],
                    "paths": [],
                    "chunks": [],
                }

            visited_depth: dict[str, int] = {s: 0 for s in seeds}
            queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
            paths: list[str] = []
            chunk_scores: dict[str, float] = {}

            for seed in seeds:
                for cid in self._entity_chunks.get(seed, set()):
                    chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 1.0

            while queue:
                curr, depth = queue.popleft()
                if depth >= self.max_hops:
                    continue

                neighbors = self._adj.get(curr, [])[: self.max_neighbors_per_hop]
                next_depth = depth + 1
                hop_weight = round(1.0 / (1.0 + next_depth), 4)

                for rel, nxt, cid in neighbors:
                    paths.append(f"{curr} -[{rel}]-> {nxt}")
                    chunk_scores[cid] = round(chunk_scores.get(cid, 0.0) + hop_weight, 4)
                    if nxt not in visited_depth:
                        visited_depth[nxt] = next_depth
                        queue.append((nxt, next_depth))

            ranked_cids = sorted(chunk_scores.keys(), key=lambda c: chunk_scores[c], reverse=True)[
                :top_k
            ]
            ranked_chunks = [
                {
                    "chunk_id": cid,
                    "text": self._chunk_texts.get(cid, ""),
                    "graph_score": chunk_scores[cid],
                    "retrieval_source": "knowledge_graph_2hop",
                }
                for cid in ranked_cids
            ]

            dedup_paths = list(dict.fromkeys(paths))[:10]
            return {
                "seed_entities": seeds,
                "paths": dedup_paths,
                "chunks": ranked_chunks,
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entities": len(self._adj),
                "triples": len(self._triples),
                "indexed_chunks": len(self._chunk_texts),
                "max_hops": self.max_hops,
                "queries_served": self._queries_served,
            }
