"""
Phase 40: Semantic Citation Grounding, Attribution Verification & Immutable Audit Logging.
"""

import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ClaimAttribution:
    claim: str
    chunk_id: str | None
    attribution_score: float
    grounded: bool


@dataclass
class GroundingReport:
    grounding_score: float
    grounded_claims: int
    total_claims: int
    cited_chunk_ids: list[str]
    attributions: list[ClaimAttribution]
    ungrounded_claims: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounding_score": self.grounding_score,
            "grounded_claims": self.grounded_claims,
            "total_claims": self.total_claims,
            "cited_chunk_ids": self.cited_chunk_ids,
            "attributions": [asdict(a) for a in self.attributions],
            "ungrounded_claims": self.ungrounded_claims,
        }


class SemanticCitationGrounder:
    """Maps response claims to retrieved context chunks via lexical/semantic token attribution."""

    def __init__(self, attribution_threshold: float = 0.35):
        self.attribution_threshold = attribution_threshold

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return {tok for tok in cleaned.split() if len(tok) > 2}

    def verify(self, answer: str, chunks: list[dict[str, Any]]) -> GroundingReport:
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer.strip()) if s.strip()]
        if not sentences:
            return GroundingReport(
                grounding_score=1.0,
                grounded_claims=0,
                total_claims=0,
                cited_chunk_ids=[],
                attributions=[],
                ungrounded_claims=[],
            )

        chunk_tokens = []
        for idx, ch in enumerate(chunks):
            cid = str(ch.get("chunk_id") or f"chunk_{idx}")
            toks = self._tokenize(str(ch.get("text", "")))
            chunk_tokens.append((cid, toks))

        attributions: list[ClaimAttribution] = []
        ungrounded: list[str] = []
        cited_set: list[str] = []
        grounded_count = 0

        for sent in sentences:
            s_toks = self._tokenize(sent)
            if not s_toks:
                grounded_count += 1
                attributions.append(ClaimAttribution(sent, None, 1.0, True))
                continue

            best_cid: str | None = None
            best_score = 0.0

            for cid, c_toks in chunk_tokens:
                if not c_toks:
                    continue
                overlap = len(s_toks & c_toks) / len(s_toks)
                if overlap > best_score:
                    best_score = overlap
                    best_cid = cid

            score = round(best_score, 4)
            is_grounded = best_cid is not None and score >= self.attribution_threshold

            if is_grounded and best_cid is not None:
                grounded_count += 1
                if best_cid not in cited_set:
                    cited_set.append(best_cid)
            else:
                ungrounded.append(sent)

            attributions.append(
                ClaimAttribution(
                    claim=sent,
                    chunk_id=best_cid if is_grounded else None,
                    attribution_score=score,
                    grounded=is_grounded,
                )
            )

        overall = round(grounded_count / len(sentences), 4)
        return GroundingReport(
            grounding_score=overall,
            grounded_claims=grounded_count,
            total_claims=len(sentences),
            cited_chunk_ids=cited_set,
            attributions=attributions,
            ungrounded_claims=ungrounded,
        )


@dataclass
class AuditRecord:
    audit_id: str
    timestamp: float
    tenant_id: str
    query_sha256: str
    cited_chunk_ids: list[str]
    grounding_score: float
    safety_status: str
    prev_hash: str
    entry_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ComplianceAuditLog:
    """Thread-safe, tamper-evident SHA-256 hash-chained compliance audit ledger."""

    GENESIS_HASH = "0" * 64

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: list[AuditRecord] = []

    @staticmethod
    def _compute_hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def record(
        self,
        tenant_id: str,
        query: str,
        cited_chunk_ids: list[str],
        grounding_score: float,
        safety_status: str = "PASS",
        metadata: dict[str, Any] | None = None,
    ) -> AuditRecord:
        with self._lock:
            prev_hash = self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH
            audit_id = f"aud_{uuid.uuid4().hex[:12]}"
            ts = round(time.time(), 6)
            query_sha = hashlib.sha256(query.encode("utf-8")).hexdigest()
            meta = dict(metadata or {})

            hash_body = {
                "audit_id": audit_id,
                "timestamp": ts,
                "tenant_id": tenant_id,
                "query_sha256": query_sha,
                "cited_chunk_ids": list(cited_chunk_ids),
                "grounding_score": round(float(grounding_score), 4),
                "safety_status": safety_status,
                "prev_hash": prev_hash,
                "metadata": meta,
            }
            entry_hash = self._compute_hash(hash_body)

            rec = AuditRecord(
                audit_id=audit_id,
                timestamp=ts,
                tenant_id=tenant_id,
                query_sha256=query_sha,
                cited_chunk_ids=list(cited_chunk_ids),
                grounding_score=round(float(grounding_score), 4),
                safety_status=safety_status,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
                metadata=meta,
            )
            self._entries.append(rec)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries :]
            return rec

    def verify_chain(self) -> bool:
        with self._lock:
            expected_prev = self.GENESIS_HASH
            for idx, rec in enumerate(self._entries):
                if idx > 0 and rec.prev_hash != expected_prev:
                    return False
                hash_body = {
                    "audit_id": rec.audit_id,
                    "timestamp": rec.timestamp,
                    "tenant_id": rec.tenant_id,
                    "query_sha256": rec.query_sha256,
                    "cited_chunk_ids": rec.cited_chunk_ids,
                    "grounding_score": rec.grounding_score,
                    "safety_status": rec.safety_status,
                    "prev_hash": rec.prev_hash,
                    "metadata": rec.metadata,
                }
                if self._compute_hash(hash_body) != rec.entry_hash:
                    return False
                expected_prev = rec.entry_hash
            return True

    def list_entries(self, tenant_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = self._entries
            if tenant_id:
                items = [e for e in items if e.tenant_id == tenant_id]
            return [asdict(e) for e in items[-limit:]]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            avg_grounding = (
                round(sum(e.grounding_score for e in self._entries) / len(self._entries), 4)
                if self._entries
                else 1.0
            )
            return {
                "total_records": len(self._entries),
                "chain_valid": self.verify_chain(),
                "average_grounding_score": avg_grounding,
                "latest_hash": self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH,
            }
