from app.core.rate_limiter import SlidingWindowRateLimiter


def test_rate_limiter_burst_and_exhaustion():
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=10)
    client = "192.168.1.100"

    ok, rem, _ = limiter.acquire(client)
    assert ok and rem == 2
    ok, rem, _ = limiter.acquire(client)
    assert ok and rem == 1
    ok, rem, _ = limiter.acquire(client)
    assert ok and rem == 0

    ok, rem, retry_after = limiter.acquire(client)
    assert not ok
    assert rem == 0
    assert retry_after > 0


def test_rate_limiter_client_isolation():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=10)
    ok_1, _, _ = limiter.acquire("client_a")
    ok_2, _, _ = limiter.acquire("client_b")
    assert ok_1 is True
    assert ok_2 is True
