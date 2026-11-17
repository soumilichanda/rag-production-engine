import httpx
import numpy as np
import pytest

from app.core.tenant_semantic_cache import TenantSemanticCache
from app.main import app


def test_tenant_namespace_isolation():
    cache = TenantSemanticCache()
    emb = np.array([0.6, 0.8, 0.0], dtype=np.float32)

    cache.put(
        "tenant-alpha",
        "confidential roadmap",
        emb,
        {"answer": "alpha-secret"},
        tier="enterprise",
        now=100.0,
    )

    # Tenant Beta querying with identical embedding must receive a cache miss (zero cross-tenant leakage)
    assert cache.get("tenant-beta", emb, tier="enterprise", now=101.0) is None

    # Tenant Alpha receives a verified hit
    hit = cache.get("tenant-alpha", emb, tier="enterprise", now=101.0)
    assert hit is not None
    assert hit["payload"]["answer"] == "alpha-secret"
    assert hit["tenant_id"] == "tenant-alpha"


def test_tier_adaptive_ttl_and_capacity():
    cache = TenantSemanticCache()
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Free tier TTL = 30s; Enterprise tier TTL = 3600s
    cache.put("free-user", "q1", emb, {"v": 1}, tier="free", now=1000.0)
    cache.put("ent-user", "q1", emb, {"v": 2}, tier="enterprise", now=1000.0)

    # At t = 1045s (+45s): free-user entry expired, ent-user entry still warm
    assert cache.get("free-user", emb, tier="free", now=1045.0) is None
    ent_hit = cache.get("ent-user", emb, tier="enterprise", now=1045.0)
    assert ent_hit is not None
    assert ent_hit["payload"]["v"] == 2
    assert ent_hit["ttl_remaining_s"] == 3555.0


@pytest.mark.asyncio
async def test_cache_telemetry_and_purge_endpoints():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/cache/telemetry")
        assert resp.status_code == 200
        body = resp.json()
        assert "policies" in body
        assert "free" in body["policies"]
        assert "enterprise" in body["policies"]

        del_resp = await client.delete("/v1/cache/tenant/nonexistent-tenant")
        assert del_resp.status_code == 200
        assert del_resp.json()["evicted_entries"] == 0
