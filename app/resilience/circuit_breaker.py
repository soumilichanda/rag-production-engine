"""
app/resilience/circuit_breaker.py
Asynchronous three-state Circuit Breaker protecting upstream retrieval and inference dependencies.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(RuntimeError):
    """Raised when an operation is rejected because the circuit breaker is OPEN."""


class AsyncCircuitBreaker:
    """Thread/task-safe asynchronous circuit breaker with half-open recovery probing."""

    def __init__(
        self,
        name: str = "upstream-rag-service",
        failure_threshold: int = 3,
        recovery_timeout: float = 15.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_sec": self.recovery_timeout,
        }

    async def _before_call(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None
                if (now - self._opened_at) >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_in_flight = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit '{self.name}' is OPEN; rejecting call during cooldown."
                    )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_in_flight >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit '{self.name}' is HALF_OPEN; probe permit saturated."
                    )
                self._half_open_in_flight += 1

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None

    async def _on_failure(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        await self._before_call()
        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        await self._on_success()
        return result
