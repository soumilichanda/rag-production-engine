import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.telemetry.tracer import DistributedPipelineTracer


def test_tracer_parent_child_lineage():
    tracer = DistributedPipelineTracer(service_name="test-rag-engine")

    with tracer.start_as_current_span("root.orchestrator") as root:
        root.set_attribute("pipeline.stage", "retrieval")
        with tracer.start_as_current_span("child.cross_encoder") as child:
            child.set_attribute("candidate_count", 5)

    batch = tracer.export_telemetry_batch()
    assert batch["total_spans"] == 2
    root_span = batch["spans"][0]
    child_span = batch["spans"][1]

    assert root_span["name"] == "root.orchestrator"
    assert child_span["name"] == "child.cross_encoder"
    assert child_span["parent_id"] == root_span["span_id"]
    assert child_span["status"] == "OK"
    assert batch["traceparent"].startswith("00-")


def test_tracer_exception_propagation():
    tracer = DistributedPipelineTracer()

    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("failing.stream"):
            raise RuntimeError("stream disconnection during chunk synthesis")

    batch = tracer.export_telemetry_batch()
    assert batch["total_spans"] == 1
    err_span = batch["spans"][0]
    assert err_span["status"] == "ERROR"
    assert err_span["attributes"]["error.type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_gateway_telemetry_trace_route():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/telemetry/traces/latest")
        assert res.status_code == 200
        data = res.json()
        assert "traceparent" in data
        assert data["total_spans"] == 2
