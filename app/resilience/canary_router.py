"""
app/resilience/canary_router.py
Deterministic weighted canary traffic splitter for progressive retrieval rollouts.
"""

import hashlib
import threading
from typing import Any


class CanaryTrafficRouter:
    """Routes requests deterministically between stable and canary serving tracks."""

    def __init__(self, canary_weight_pct: int = 10) -> None:
        if not 0 <= canary_weight_pct <= 100:
            raise ValueError("canary_weight_pct must be between 0 and 100")
        self.canary_weight_pct = canary_weight_pct
        self._stable_routed = 0
        self._canary_routed = 0
        self._lock = threading.Lock()

    def set_weight(self, canary_weight_pct: int) -> None:
        if not 0 <= canary_weight_pct <= 100:
            raise ValueError("canary_weight_pct must be between 0 and 100")
        with self._lock:
            self.canary_weight_pct = canary_weight_pct

    def select_track(self, routing_key: str, force_track: str | None = None) -> str:
        override = (force_track or "").strip().lower()
        if override in ("canary", "stable"):
            track = override
        else:
            digest = hashlib.sha256(routing_key.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % 100
            track = "canary" if bucket < self.canary_weight_pct else "stable"

        with self._lock:
            if track == "canary":
                self._canary_routed += 1
            else:
                self._stable_routed += 1
        return track

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._stable_routed + self._canary_routed
            observed_pct = (
                round((self._canary_routed / total) * 100.0, 2) if total > 0 else 0.0
            )
            return {
                "configured_canary_pct": self.canary_weight_pct,
                "observed_canary_pct": observed_pct,
                "stable_requests": self._stable_routed,
                "canary_requests": self._canary_routed,
            }
