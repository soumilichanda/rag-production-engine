"""
app/resilience/chaos.py
Controlled Chaos Fault Injection Engine for resilience validation.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any


class ChaosInjectedError(RuntimeError):
    """Raised when a deterministic or probabilistic chaos fault is triggered."""


@dataclass
class ChaosConfig:
    enabled: bool = False
    error_rate: float = 0.0
    latency_ms: float = 0.0


class ChaosFaultInjector:
    """Injects bounded latency delays or transient faults to test circuit breakers & retries."""

    def __init__(
        self,
        enabled: bool = False,
        error_rate: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        self.config = ChaosConfig(
            enabled=enabled,
            error_rate=max(0.0, min(1.0, error_rate)),
            latency_ms=max(0.0, latency_ms),
        )
        self.injected_latencies = 0
        self.injected_errors = 0

    def configure(
        self, enabled: bool, error_rate: float = 0.0, latency_ms: float = 0.0
    ) -> None:
        self.config = ChaosConfig(
            enabled=enabled,
            error_rate=max(0.0, min(1.0, error_rate)),
            latency_ms=max(0.0, latency_ms),
        )

    async def maybe_inject(self, header_directive: str | None = None) -> str | None:
        directive = (header_directive or "").strip().lower()

        if directive == "latency":
            delay_ms = self.config.latency_ms if self.config.latency_ms > 0 else 25.0
            self.injected_latencies += 1
            await asyncio.sleep(delay_ms / 1000.0)
            return "latency"

        if directive == "error":
            self.injected_errors += 1
            raise ChaosInjectedError(
                "Deterministic chaos error injected via X-Chaos-Inject"
            )

        if not self.config.enabled:
            return None

        if self.config.latency_ms > 0.0:
            self.injected_latencies += 1
            await asyncio.sleep(self.config.latency_ms / 1000.0)

        if self.config.error_rate > 0.0 and random.random() < self.config.error_rate:
            self.injected_errors += 1
            raise ChaosInjectedError("Probabilistic chaos fault triggered")

        return "latency" if self.config.latency_ms > 0.0 else None

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "error_rate": self.config.error_rate,
            "latency_ms": self.config.latency_ms,
            "injected_latencies": self.injected_latencies,
            "injected_errors": self.injected_errors,
        }
