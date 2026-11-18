import pytest
from httpx import ASGITransport, AsyncClient

from app.core.token_billing import ModelPricing, TokenCostLedger
from app.main import app, token_cost_ledger


def test_token_cost_ledger_unit_attribution():
    ledger = TokenCostLedger(
        pricing=ModelPricing(prompt_per_1k_usd=0.002, completion_per_1k_usd=0.008)
    )
    ledger.record_stream_usage("tenant-alpha", prompt_tokens=500, completion_tokens=250)
    ledger.record_stream_usage("tenant-alpha", prompt_tokens=500, completion_tokens=250)
    ledger.record_stream_usage("tenant-beta", prompt_tokens=1000, completion_tokens=0)

    alpha = ledger.get_tenant_usage("tenant-alpha")
    assert alpha["prompt_tokens"] == 1000
    assert alpha["completion_tokens"] == 500
    assert alpha["total_tokens"] == 1500
    assert alpha["stream_requests"] == 2
    assert alpha["estimated_cost_usd"] == pytest.approx(0.006, rel=1e-5)

    rollup = ledger.get_all_usage()
    assert rollup["total_prompt_tokens"] == 2000
    assert rollup["total_completion_tokens"] == 500
    assert "tenant-alpha" in rollup["tenants"]
    assert "tenant-beta" in rollup["tenants"]


@pytest.mark.asyncio
async def test_streaming_pipeline_attributes_tenant_tokens():
    token_cost_ledger.reset()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/index",
            json={
                "documents": [
                    {
                        "chunk_id": "billing_doc_1",
                        "text": "Distributed RAG streaming token budgets attribute prompt and completion costs per tenant.",
                    }
                ]
            },
            headers={"X-Tenant-ID": "acme-corp", "X-Tenant-Tier": "enterprise"},
        )

        res = await client.post(
            "/v1/pipeline/stream",
            json={"query": "How are streaming token budgets attributed?", "top_k": 1},
            headers={"X-Tenant-ID": "acme-corp", "X-Tenant-Tier": "enterprise"},
        )
        assert res.status_code == 200
        assert "data: [DONE]" in res.text

        usage_res = await client.get("/v1/billing/usage/acme-corp")
        assert usage_res.status_code == 200
        usage = usage_res.json()
        assert usage["tenant_id"] == "acme-corp"
        assert usage["stream_requests"] == 1
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["estimated_cost_usd"] > 0.0
