"""
app/core/rate_limiter.py
Sliding Window Token Bucket Rate Limiter with RFC-compliant headers.
"""

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Tuple
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class SlidingWindowRateLimiter:
    """
    In-memory thread-safe sliding window rate limiter tracking request timestamps.
    """

    def __init__(self, limit: int = 60, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.client_windows: Dict[str, deque] = defaultdict(deque)
        self.lock = Lock()

    def acquire(self, client_id: str) -> Tuple[bool, int, int]:
        now = time.time()
        boundary = now - self.window_seconds

        with self.lock:
            queue = self.client_windows[client_id]
            while queue and queue[0] <= boundary:
                queue.popleft()

            if len(queue) < self.limit:
                queue.append(now)
                remaining = self.limit - len(queue)
                return True, remaining, 0

            oldest_timestamp = queue[0]
            retry_after = max(1, int(oldest_timestamp + self.window_seconds - now))
            return False, 0, retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = SlidingWindowRateLimiter(limit=limit, window_seconds=window_seconds)

    async def dispatch(self, request: Request, call_next):
        # Allow internal health checks to bypass rate limits
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown_client"
        allowed, remaining, retry_after = self.limiter.acquire(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Too many requests."},
                headers={
                    "X-RateLimit-Limit": str(self.limiter.limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(retry_after),
                },
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
