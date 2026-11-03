import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.telemetry.metrics import PrometheusMetricsExporter, metrics_exporter


@pytest.mark.asyncio
async def test_metrics_exporter_recording():
    exporter = PrometheusMetricsExporter()
    exporter.record_request("/v1/retrieve", 200, 18.5)
    exporter.record_request("/v1/retrieve", 200, 42.1)
    exporter.record_request("/v1/retrieve", 500, 110.0)

    scraped = exporter.scrape_metrics()
    assert "rag_http_requests_total" in scraped
    assert 'endpoint="/v1/retrieve",status="200"' in scraped
    assert 'endpoint="/v1/retrieve",status="500"' in scraped
    assert "rag_request_duration_ms_bucket" in scraped
    assert 'le="+Inf"' in scraped


@pytest.mark.asyncio
async def test_gateway_metrics_route():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        metrics_exporter.record_request("/health", 200, 1.2)
        res = await client.get("/metrics")
        assert res.status_code == 200
        assert "rag_http_requests_total" in res.text
