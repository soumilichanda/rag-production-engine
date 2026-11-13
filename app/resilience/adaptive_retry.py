"""
app/resilience/adaptive_retry.py
Adaptive exponential backoff with full jitter and speculative tail-latency request hedging.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class RetryExhaustedError(RuntimeError):
    """Raised when all adaptive retry attempts fail."""


class DecorrelatedJitterRetry:
    """Asynchronous retry executor using exponential backoff with full jitter."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_sec: float = 0.05,
        max_delay_sec: float = 2.0,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self.retryable_exceptions = retryable_exceptions
        self.total_retries = 0

    def compute_backoff(self, attempt: int) -> float:
        ceiling = min(self.max_delay_sec, self.base_delay_sec * (2**attempt))
        return random.uniform(0.0, ceiling)

    async def execute(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.retryable_exceptions as exc:
                last_exc = exc
                if attempt == self.max_attempts - 1:
                    break
                self.total_retries += 1
                delay = self.compute_backoff(attempt)
                await asyncio.sleep(delay)

        raise RetryExhaustedError(
            f"Operation failed after {self.max_attempts} attempts"
        ) from last_exc


class HedgedRequestExecutor:
    """Races a primary async call against a speculative backup call after hedge_delay_sec."""

    def __init__(self, hedge_delay_sec: float = 0.05) -> None:
        self.hedge_delay_sec = hedge_delay_sec
        self.hedges_triggered = 0
        self.hedges_won = 0

    async def execute(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        primary_task = asyncio.create_task(func(*args, **kwargs))

        done, _ = await asyncio.wait({primary_task}, timeout=self.hedge_delay_sec)
        if primary_task in done:
            return await primary_task

        self.hedges_triggered += 1
        backup_task = asyncio.create_task(func(*args, **kwargs))
        tasks = {primary_task, backup_task}

        try:
            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    if finished.exception() is None:
                        if finished is backup_task:
                            self.hedges_won += 1
                        for p in pending:
                            p.cancel()
                        return finished.result()
                    tasks.remove(finished)

            # Both failed; raise primary exception
            return await primary_task
        finally:
            for t in (primary_task, backup_task):
                if not t.done():
                    t.cancel()
