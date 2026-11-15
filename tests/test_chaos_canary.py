import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.resilience.canary_router import CanaryTrafficRouter
from app.resilience.chaos import ChaosFaultInjector, ChaosInjectedError


def test_canary_router_sticky_bucketing():
    router = CanaryTrafficRouter(canary_weight_pct=25)
    first_choice = router.select_track("enterprise-tenant-42")
    for _ in range(10):
        assert router.select_track("enterprise-tenant-42") == first_choice

    assert router.select_track("any-tenant", force_track="canary") == "canary"
    assert router.select_track("any-tenant", force_track="stable") == "stable"


@pytest.mark.asyncio
async def test_chaos_injector_directives():
    injector = ChaosFaultInjector(enabled=False, latency_ms=15.0)
    assert await injector.maybe_inject() is None

    applied = await injector.maybe_inject(header_directive="latency")
    assert applied == "latency"
    assert injector.stats()["injected_latencies"] == 1

    with pytest.raises(ChaosInjectedError):
        await injector.maybe_inject(header_directive="error")


@pytest.mark.asyncio
async def test_gateway_chaos_and_canary_headers():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/index",
            json={
                "documents": [
                    {
                        "chunk_id": "chaos_seed_1",
                        "text": "Progressive canary rollout and fault injection validation.",
                        "metadata": {"source": "test"},
                    }
                ]
            },
        )
        payload = {"query": "canary rollout", "top_k": 1, "score_threshold": 0.0}

        # 1. Verify forced canary routing + latency injection header
        res_ok = await client.post(
            "/v1/retrieve",
            json=payload,
            headers={
                "X-Tenant-ID": "canary-tester",
                "X-Canary-Force": "canary",
                "X-Chaos-Inject": "latency",
            },
        )
        assert res_ok.status_code == 200
        assert res_ok.headers["X-Serving-Track"] == "canary"
        assert res_ok.headers["X-Chaos-Applied"] == "latency"

        # 2. Verify deterministic HTTP 503 chaos error injection
        res_err = await client.post(
            "/v1/retrieve",
            json=payload,
            headers={
                "X-Tenant-ID": "canary-tester",
                "X-Canary-Force": "stable",
                "X-Chaos-Inject": "error",
            },
        )
        assert res_err.status_code == 503
        assert res_err.headers["X-Serving-Track"] == "stable"
        assert res_err.json()["chaos_injected"] is True

        # 3. Verify telemetry endpoint
        telemetry = await client.get("/v1/resilience/canary")
        assert telemetry.status_code == 200
        body = telemetry.json()
        assert body["chaos_injector"]["injected_errors"] >= 1
        assert body["canary_router"]["canary_requests"] >= 1
