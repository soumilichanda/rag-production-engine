import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.resilience.sla_tiers import TieredQuotaManager


def test_tiered_quota_manager_profiles():
    mgr = TieredQuotaManager()

    free_dec = mgr.evaluate("t-free", tier="free")
    assert free_dec.allowed is True
    assert free_dec.tier == "free"
    assert free_dec.limit == 3
    assert free_dec.max_top_k == 3

    ent_dec = mgr.evaluate("t-ent", tier="enterprise")
    assert ent_dec.allowed is True
    assert ent_dec.tier == "enterprise"
    assert ent_dec.limit == 100
    assert ent_dec.max_top_k == 50


@pytest.mark.asyncio
async def test_gateway_sla_tier_isolation_and_topk_caps():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/index",
            json={
                "documents": [
                    {
                        "chunk_id": "sla_doc_1",
                        "text": "Multi-tenant SLA tier enforcement isolates free and enterprise traffic.",
                        "metadata": {"source": "sla_test"},
                    }
                ]
            },
        )

        # 1. Free-tier tenant requesting top_k=10 is blocked with HTTP 403
        r_cap = await client.post(
            "/v1/retrieve",
            json={"query": "multi-tenant SLA", "top_k": 10, "score_threshold": 0.0},
            headers={"X-Tenant-ID": "noisy-free-tenant", "X-Tenant-Tier": "free"},
        )
        assert r_cap.status_code == 403
        assert r_cap.headers["X-Tenant-Tier"] == "free"
        assert r_cap.headers["X-SLA-Max-TopK"] == "3"

        # 2. Remaining 2 free-tier burst tokens succeed at top_k=2, 4th triggers HTTP 429
        r_ok1 = await client.post(
            "/v1/retrieve",
            json={"query": "multi-tenant SLA", "top_k": 2, "score_threshold": 0.0},
            headers={"X-Tenant-ID": "noisy-free-tenant", "X-Tenant-Tier": "free"},
        )
        assert r_ok1.status_code == 200
        assert r_ok1.headers["X-Tenant-Tier"] == "free"

        r_ok2 = await client.post(
            "/v1/retrieve",
            json={"query": "multi-tenant SLA", "top_k": 2, "score_threshold": 0.0},
            headers={"X-Tenant-ID": "noisy-free-tenant", "X-Tenant-Tier": "free"},
        )
        assert r_ok2.status_code == 200

        r_throttled = await client.post(
            "/v1/retrieve",
            json={"query": "multi-tenant SLA", "top_k": 2, "score_threshold": 0.0},
            headers={"X-Tenant-ID": "noisy-free-tenant", "X-Tenant-Tier": "free"},
        )
        assert r_throttled.status_code == 429
        assert "Retry-After" in r_throttled.headers

        # 3. Enterprise tenant operates unaffected above free/pro caps (top_k=15 <= schema max 20)
        r_ent = await client.post(
            "/v1/retrieve",
            json={"query": "multi-tenant SLA", "top_k": 15, "score_threshold": 0.0},
            headers={
                "X-Tenant-ID": "vip-enterprise-tenant",
                "X-Tenant-Tier": "enterprise",
            },
        )
        assert r_ent.status_code == 200
        assert r_ent.headers["X-Tenant-Tier"] == "enterprise"
        assert r_ent.headers["X-SLA-Max-TopK"] == "50"

        # 4. Verify live SLA telemetry endpoint
        sla_resp = await client.get("/v1/resilience/sla")
        assert sla_resp.status_code == 200
        tenants = sla_resp.json()["sla"]["tenants"]
        assert tenants["noisy-free-tenant"]["tier"] == "free"
        assert tenants["vip-enterprise-tenant"]["tier"] == "enterprise"
