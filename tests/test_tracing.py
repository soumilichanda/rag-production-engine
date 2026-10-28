import pytest

from app.core.tracer import PipelineTracer


def test_tracer_span_hierarchy():
    tracer = PipelineTracer()
    with tracer.start_as_current_span("parent_span"), tracer.start_as_current_span("child_span"):
        pass
    assert len(tracer.spans) == 2
    assert tracer.spans[0].name == "parent_span"
    assert tracer.spans[1].name == "child_span"


def test_tracer_error_capture():
    tracer = PipelineTracer()
    with pytest.raises(ValueError), tracer.start_as_current_span("failing.operation"):
        raise ValueError("simulated retrieval timeout")

    assert len(tracer.spans) == 1
    assert tracer.spans[0].status == "ERROR"
    assert "simulated retrieval timeout" in tracer.spans[0].attributes.get("error.message", "")
