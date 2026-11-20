import pytest
from httpx import ASGITransport, AsyncClient

from app.core.model_router import CostAwareModelRouter
from app.main import app, model_router, token_cost_ledger


def test_router_complexity_and_budget_downgrade():
    router = CostAwareModelRouter(
        flagship_min_headroom_usd=0.02,
        balanced_min_headroom_usd=0.002,
    )

    # 1. Enterprise + complex query + ample budget -> flagship
    d1 = router.route(
        tenant_id="ent-1",
        query="Compare and synthesize distributed consensus tradeoffs across multi-region clusters",
        tenant_tier="enterprise",
        remaining_budget_usd=0.50,
    )
    assert d1.selected_profile == "flagship"
    assert d1.downgraded is False

    # 2. Enterprise + complex query + tight budget ($0.005) -> downgraded to balanced
    d2 = router.route(
        tenant_id="ent-1",
        query="Compare and synthesize distributed consensus tradeoffs across multi-region clusters",
        tenant_tier="enterprise",
        remaining_budget_usd=0.005,
    )
    assert d2.selected_profile == "balanced"
    assert d2.downgraded is True

    # 3. Any tier with critically low budget ($0.0005) -> downgraded to economy
    d3 = router.route(
        tenant_id="ent-1",
        query="Compare and synthesize distributed consensus tradeoffs across multi-region clusters",
        tenant_tier="enterprise",
        remaining_budget_usd=0.0005,
    )
    assert d3.selected_profile == "economy"
    assert d3.downgraded is True


@pytest.mark.asyncio
async def test_gateway_stream_model_routing_headers_and_telemetry():
    token_cost_ledger.reset()
    model_router.reset()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Enterprise complex query with healthy budget routes to flagship
        await client.post(
            "/v1/billing/budgets/tenant-ent",
            json={"spend_cap_usd": 1.00, "alert_threshold_ratio": 0.80},
        )
        r1 = await client.post(
            "/v1/pipeline/stream",
            json={
                "query": "Analyze and synthesize root cause architectural tradeoffs across vector stores",
                "top_k": 2,
            },
            headers={"X-Tenant-ID": "tenant-ent", "X-Tenant-Tier": "enterprise"},
        )
        assert r1.status_code == 200
        assert r1.headers["X-Model-Tier"] == "flagship"
        assert r1.headers["X-Model-Selected"] == "o3-reasoning-tier"
        assert r1.headers["X-Model-Downgraded"] == "false"

        # 2. Constrained budget ($0.0008) automatically triggers cost-aware downgrade to economy
        await client.post(
            "/v1/billing/budgets/tenant-low-headroom",
            json={"spend_cap_usd": 0.0008, "alert_threshold_ratio": 0.80},
        )
        r2 = await client.post(
            "/v1/pipeline/stream",
            json={
                "query": "Analyze and synthesize root cause architectural tradeoffs across vector stores",
                "top_k": 2,
            },
            headers={
                "X-Tenant-ID": "tenant-low-headroom",
                "X-Tenant-Tier": "enterprise",
            },
        )
        assert r2.status_code == 200
        assert r2.headers["X-Model-Tier"] == "economy"
        assert r2.headers["X-Model-Selected"] == "gpt-4o-mini-tier"
        assert r2.headers["X-Model-Downgraded"] == "true"

        # 3. Verify routing telemetry endpoint tracks total routed & downgrade count
        tel = await client.get("/v1/routing/telemetry")
        assert tel.status_code == 200
        stats = tel.json()["routing"]
        assert stats["total_routed"] == 2
        assert stats["downgrade_count"] == 1
