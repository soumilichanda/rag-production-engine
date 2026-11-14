import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app, quota_manager
from app.resilience.rate_limiter import HierarchicalQuotaManager


def test_token_bucket_burst_and_monotonic_refill():
    mgr = HierarchicalQuotaManager(
        default_tenant_capacity=3, default_tenant_refill_rate=20.0
    )
    assert mgr.evaluate("tenant-alpha").allowed is True
    assert mgr.evaluate("tenant-alpha").allowed is True
    assert mgr.evaluate("tenant-alpha").allowed is True

    blocked = mgr.evaluate("tenant-alpha")
    assert blocked.allowed is False
    assert blocked.remaining == 0
    assert blocked.retry_after > 0.0

    time.sleep(0.06)
    recovered = mgr.evaluate("tenant-alpha")
    assert recovered.allowed is True


def test_multi_tenant_isolation():
    mgr = HierarchicalQuotaManager(
        default_tenant_capacity=2, default_tenant_refill_rate=1.0
    )
    assert mgr.evaluate("tenant-a").allowed is True
    assert mgr.evaluate("tenant-a").allowed is True
    assert mgr.evaluate("tenant-a").allowed is False

    tenant_b = mgr.evaluate("tenant-b")
    assert tenant_b.allowed is True
    assert tenant_b.remaining == 1


@pytest.mark.asyncio
async def test_gateway_http_429_and_quota_telemetry():
    quota_manager.configure_tenant("burst-tenant", capacity=2, refill_rate=0.5)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/index",
            json={
                "documents": [
                    {
                        "chunk_id": "quota_seed_1",
                        "text": "Vector retrieval latency and per-tenant token bucket governance.",
                        "metadata": {"source": "test"},
                    }
                ]
            },
        )
        payload = {
            "query": "vector retrieval latency",
            "top_k": 1,
            "score_threshold": 0.0,
        }
        headers = {"X-Tenant-ID": "burst-tenant"}

        r1 = await client.post("/v1/retrieve", json=payload, headers=headers)
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Limit"] == "2"
        assert r1.headers["X-RateLimit-Remaining"] == "1"

        r2 = await client.post("/v1/retrieve", json=payload, headers=headers)
        assert r2.status_code == 200
        assert r2.headers["X-RateLimit-Remaining"] == "0"

        r3 = await client.post("/v1/retrieve", json=payload, headers=headers)
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
        assert r3.headers["X-RateLimit-Remaining"] == "0"
        assert r3.json()["tenant_id"] == "burst-tenant"

        stats_res = await client.get("/v1/resilience/quotas")
        assert stats_res.status_code == 200
        assert "burst-tenant" in stats_res.json()["tenants"]
