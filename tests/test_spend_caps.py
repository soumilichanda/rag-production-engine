import pytest
from httpx import ASGITransport, AsyncClient

from app.core.token_billing import ModelPricing, TokenCostLedger
from app.main import app, token_cost_ledger


def test_ledger_spend_cap_and_threshold_alerts():
    ledger = TokenCostLedger(
        pricing=ModelPricing(prompt_per_1k_usd=0.01, completion_per_1k_usd=0.02)
    )
    ledger.set_tenant_budget(
        "tenant-capped", spend_cap_usd=0.05, alert_threshold_ratio=0.80
    )

    # Spend $0.02 (40% utilization) -> allowed, no alert yet
    ledger.record_stream_usage(
        "tenant-capped", prompt_tokens=1000, completion_tokens=500
    )
    eval1 = ledger.check_budget_available("tenant-capped", projected_cost_usd=0.01)
    assert eval1.allowed is True
    assert eval1.remaining_usd == pytest.approx(0.03, rel=1e-5)
    assert eval1.alert_triggered is False

    # Spend another $0.025 (total $0.045 = 90% utilization >= 80% threshold) -> triggers alert
    ledger.record_stream_usage(
        "tenant-capped", prompt_tokens=1500, completion_tokens=500
    )
    eval2 = ledger.check_budget_available("tenant-capped", projected_cost_usd=0.002)
    assert eval2.allowed is True
    assert eval2.alert_triggered is True
    alerts = ledger.get_alerts()
    assert len(alerts) == 1
    assert alerts[0]["tenant_id"] == "tenant-capped"

    # Projecting $0.01 more exceeds $0.05 cap -> rejected
    eval3 = ledger.check_budget_available("tenant-capped", projected_cost_usd=0.01)
    assert eval3.allowed is False


@pytest.mark.asyncio
async def test_gateway_http_402_spend_cap_enforcement():
    token_cost_ledger.reset()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Configure a tight $0.000100 USD spend cap for tenant-budget-test
        cap_res = await client.post(
            "/v1/billing/budgets/tenant-budget-test",
            json={"spend_cap_usd": 0.000100, "alert_threshold_ratio": 0.50},
        )
        assert cap_res.status_code == 200
        assert cap_res.json()["spend_cap_usd"] == pytest.approx(0.0001, rel=1e-5)

        # 2. First streaming request succeeds and emits X-Billing-Remaining-USD header
        r1 = await client.post(
            "/v1/pipeline/stream",
            json={"query": "budget check", "top_k": 1},
            headers={
                "X-Tenant-ID": "tenant-budget-test",
                "X-Tenant-Tier": "enterprise",
            },
        )
        assert r1.status_code == 200
        assert "X-Billing-Remaining-USD" in r1.headers
        assert r1.headers["X-Billing-Spend-Cap-USD"] == "0.000100"

        # 3. Exhaust remaining budget cap & verify subsequent stream returns HTTP 402 Payment Required
        token_cost_ledger.record_stream_usage(
            "tenant-budget-test", prompt_tokens=500, completion_tokens=500
        )
        r2 = await client.post(
            "/v1/pipeline/stream",
            json={"query": "blocked over budget", "top_k": 1},
            headers={
                "X-Tenant-ID": "tenant-budget-test",
                "X-Tenant-Tier": "enterprise",
            },
        )
        assert r2.status_code == 402
        assert r2.headers["X-Billing-Remaining-USD"] == "0.000000"
        body = r2.json()
        assert body["detail"] == "Tenant USD spend cap exhausted"
        assert body["tenant_id"] == "tenant-budget-test"

        # 4. Verify alert telemetry endpoint recorded threshold exhaustion
        alerts_res = await client.get("/v1/billing/alerts")
        assert alerts_res.status_code == 200
        assert alerts_res.json()["count"] >= 1
