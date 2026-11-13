import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.resilience.adaptive_retry import (
    DecorrelatedJitterRetry,
    HedgedRequestExecutor,
    RetryExhaustedError,
)
from app.resilience.load_shedder import (
    ConcurrencyLoadShedder,
    LoadSheddingRejectedError,
)


@pytest.mark.asyncio
async def test_decorrelated_jitter_retry_recovers_transient_faults():
    policy = DecorrelatedJitterRetry(
        max_attempts=3,
        base_delay_sec=0.01,
        max_delay_sec=0.05,
        retryable_exceptions=(ConnectionError,),
    )
    attempts = 0

    async def flaky_upstream():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("transient packet drop")
        return "recovered"

    result = await policy.execute(flaky_upstream)
    assert result == "recovered"
    assert attempts == 3
    assert policy.total_retries == 2


@pytest.mark.asyncio
async def test_decorrelated_jitter_retry_exhaustion():
    policy = DecorrelatedJitterRetry(max_attempts=2, base_delay_sec=0.005)

    async def always_fails():
        raise TimeoutError("upstream dead")

    with pytest.raises(RetryExhaustedError):
        await policy.execute(always_fails)


@pytest.mark.asyncio
async def test_hedged_request_executor_tail_latency_win():
    executor = HedgedRequestExecutor(hedge_delay_sec=0.02)
    call_count = 0

    async def straggler_endpoint():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(0.25)
            return "slow-primary"
        await asyncio.sleep(0.005)
        return "fast-hedge"

    result = await executor.execute(straggler_endpoint)
    assert result == "fast-hedge"
    assert executor.hedges_triggered == 1
    assert executor.hedges_won == 1


@pytest.mark.asyncio
async def test_concurrency_load_shedder_rejects_overflow():
    shedder = ConcurrencyLoadShedder(max_concurrent=2)
    release_event = asyncio.Event()

    async def hold_slot():
        async with shedder.acquire():
            await release_event.wait()

    t1 = asyncio.create_task(hold_slot())
    t2 = asyncio.create_task(hold_slot())
    await asyncio.sleep(0.02)

    assert shedder.active_requests == 2
    with pytest.raises(LoadSheddingRejectedError):
        async with shedder.acquire():
            pass

    release_event.set()
    await asyncio.gather(t1, t2)
    assert shedder.active_requests == 0
    assert shedder.stats()["total_shed"] == 1


@pytest.mark.asyncio
async def test_resilience_stats_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/resilience/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "load_shedder" in data
        assert "hedged_executor" in data
