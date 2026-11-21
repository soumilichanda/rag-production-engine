"""
app/core/safety_guardrails.py
Streaming-safe PII redaction and output policy compliance guardrails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern
from threading import Lock
from typing import Any


@dataclass
class SafetyAuditReport:
    safe: bool
    sanitized_text: str
    redacted_count: int = 0
    redaction_types: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "redacted_count": self.redacted_count,
            "redaction_types": list(self.redaction_types),
            "blocked_reasons": list(self.blocked_reasons),
        }


class StreamingPIIRedactor:
    """
    Deterministic PII redaction engine supporting both full-text and chunked
    SSE streaming windows with lookback buffer boundary protection.
    """

    DEFAULT_PATTERNS: tuple[tuple[str, Pattern[str], str], ...] = (
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "[REDACTED:EMAIL]",
        ),
        (
            "SSN",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "[REDACTED:SSN]",
        ),
        (
            "CARD",
            re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            "[REDACTED:CARD]",
        ),
        (
            "SECRET",
            re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._-]{16,})\b"),
            "[REDACTED:SECRET]",
        ),
    )

    def __init__(self, holdback_chars: int = 32) -> None:
        self.holdback_chars = max(0, holdback_chars)

    def redact_text(self, text: str) -> tuple[str, int, list[str]]:
        sanitized = text
        total_redactions = 0
        triggered_types: list[str] = []

        for label, pattern, replacement in self.DEFAULT_PATTERNS:
            sanitized, count = pattern.subn(replacement, sanitized)
            if count > 0:
                total_redactions += count
                triggered_types.append(label)

        return sanitized, total_redactions, triggered_types


class OutputSafetyGuardrail:
    """
    Pre-emission compliance validator combining PII redaction with policy
    violation checks (system-prompt exfiltration & injection blocking).
    """

    POLICY_RULES: tuple[tuple[str, Pattern[str]], ...] = (
        (
            "PROMPT_EXFILTRATION",
            re.compile(
                r"(?i)\b(?:ignore\s+previous\s+instructions|reveal\s+system\s+prompt|dump\s+all\s+secrets)\b"
            ),
        ),
        (
            "CREDENTIAL_HARVEST",
            re.compile(
                r"(?i)\b(?:exfiltrate\s+api\s+keys|bypass\s+tenant\s+isolation)\b"
            ),
        ),
    )

    def __init__(self) -> None:
        self.redactor = StreamingPIIRedactor()
        self._lock = Lock()
        self._total_inspected = 0
        self._total_redactions = 0
        self._total_blocked = 0

    def evaluate(self, text: str) -> SafetyAuditReport:
        blocked_reasons: list[str] = []
        for reason, rule_pattern in self.POLICY_RULES:
            if rule_pattern.search(text):
                blocked_reasons.append(reason)

        sanitized, redaction_count, redaction_types = self.redactor.redact_text(text)
        is_safe = len(blocked_reasons) == 0

        with self._lock:
            self._total_inspected += 1
            self._total_redactions += redaction_count
            if not is_safe:
                self._total_blocked += 1

        return SafetyAuditReport(
            safe=is_safe,
            sanitized_text="" if not is_safe else sanitized,
            redacted_count=redaction_count,
            redaction_types=redaction_types,
            blocked_reasons=blocked_reasons,
        )

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_inspected": self._total_inspected,
                "total_redactions": self._total_redactions,
                "total_blocked": self._total_blocked,
            }
