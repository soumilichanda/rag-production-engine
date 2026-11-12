"""
app/resilience/drain_manager.py
Zero-drop graceful shutdown coordinator and shard health probe aggregator.
"""

import asyncio
from typing import Any


class GracefulDrainManager:
    """Tracks in-flight requests and coordinates zero-drop drain during termination."""

    def __init__(self) -> None:
        self._in_flight = 0
        self._draining = False
        self._drained_event = asyncio.Event()
        self._drained_event.set()
        self._lock = asyncio.Lock()

    @property
    def is_draining(self) -> bool:
        return self._draining

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire_slot(self) -> bool:
        async with self._lock:
            if self._draining:
                return False
            self._in_flight += 1
            self._drained_event.clear()
            return True

    async def release_slot(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if self._in_flight == 0:
                self._drained_event.set()

    async def initiate_drain(self, timeout: float = 5.0) -> bool:
        async with self._lock:
            self._draining = True
            if self._in_flight == 0:
                self._drained_event.set()
                return True

        try:
            await asyncio.wait_for(self._drained_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def status(self) -> dict[str, Any]:
        return {
            "accepting_traffic": not self._draining,
            "draining": self._draining,
            "in_flight_requests": self._in_flight,
        }
