import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_diagnostics_health_probe():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/diagnostics/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["ready"] is True


@pytest.mark.asyncio
async def test_diagnostics_telemetry_probe():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/diagnostics/telemetry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "cache_layer" in data
