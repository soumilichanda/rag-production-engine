import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.resilience.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from app.resilience.drain_manager import GracefulDrainManager


@pytest.mark.asyncio
async def test_circuit_breaker_trip_and_half_open_recovery():
    breaker = AsyncCircuitBreaker(
        name="test-breaker",
        failure_threshold=2,
        recovery_timeout=0.05,
        half_open_max_calls=1,
    )

    async def failing_op():
        raise ValueError("simulated upstream timeout")

    async def healthy_op():
        return "ok"

    for _ in range(2):
        with pytest.raises(ValueError):
            await breaker.call(failing_op)

    assert breaker.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(healthy_op)

    await asyncio.sleep(0.06)
    res = await breaker.call(healthy_op)
    assert res == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_graceful_drain_zero_drop():
    manager = GracefulDrainManager()
    assert await manager.acquire_slot() is True
    assert manager.in_flight == 1

    async def finish_inflight():
        await asyncio.sleep(0.03)
        await manager.release_slot()

    asyncio.create_task(finish_inflight())
    drained = await manager.initiate_drain(timeout=1.0)
    assert drained is True
    assert manager.in_flight == 0
    assert await manager.acquire_slot() is False


@pytest.mark.asyncio
async def test_deep_health_checkpoint_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/health/deep")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["circuit_breaker"]["state"] == "CLOSED"
        assert body["traffic_drain"]["accepting_traffic"] is True
