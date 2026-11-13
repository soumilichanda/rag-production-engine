"""
app/resilience/load_shedder.py
Concurrency admission controller and overload load shedder.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class LoadSheddingRejectedError(RuntimeError):
    """Raised when in-flight concurrency reaches saturation and excess load is shed."""


class ConcurrencyLoadShedder:
    """Bounds active concurrent executions and sheds excess traffic immediately."""

    def __init__(self, max_concurrent: int = 32) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self._active = 0
        self._admitted = 0
        self._shed = 0
        self._lock = asyncio.Lock()

    @property
    def active_requests(self) -> int:
        return self._active

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._active >= self.max_concurrent:
                self._shed += 1
                raise LoadSheddingRejectedError(
                    f"Concurrency saturation ({self._active}/{self.max_concurrent}); shedding load."
                )
            self._active += 1
            self._admitted += 1

        try:
            yield
        finally:
            async with self._lock:
                self._active = max(0, self._active - 1)

    def stats(self) -> dict[str, Any]:
        return {
            "max_concurrent": self.max_concurrent,
            "active_requests": self._active,
            "total_admitted": self._admitted,
            "total_shed": self._shed,
        }
