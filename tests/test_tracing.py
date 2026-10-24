import pytest
import time
from app.core.tracer import PipelineTracer


def test_tracer_span_hierarchy():
    tracer = PipelineTracer(service_name="rag-production-test")

    with tracer.start_as_current_span("root.pipeline") as root:
        root.set_attribute("query", "distributed tracing")
        with tracer.start_as_current_span("child.retrieval") as retrieval:
            retrieval.set_attribute("chunks_found", 4)
            time.sleep(0.01)

    summary = tracer.export_trace_summary()
    assert summary["span_count"] == 2
    assert summary["spans"][0]["name"] == "root.pipeline"
    assert summary["spans"][1]["name"] == "child.retrieval"
    assert summary["spans"][1]["parent_id"] == summary["spans"][0]["span_id"]
    assert summary["spans"][0]["status"] == "OK"
    assert summary["spans"][1]["status"] == "OK"
    assert "00-" in summary["traceparent"]


def test_tracer_error_capture():
    tracer = PipelineTracer()
    with pytest.raises(ValueError):
        with tracer.start_as_current_span("failing.operation") as span:
            raise ValueError("simulated retrieval timeout")

    summary = tracer.export_trace_summary()
    assert summary["span_count"] == 1
    assert summary["spans"][0]["status"] == "ERROR"
    assert summary["spans"][0]["attributes"]["error.type"] == "ValueError"
