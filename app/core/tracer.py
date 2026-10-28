"""
Production OpenTelemetry Pipeline Tracer & Distributed Context Engine.
Tracks hierarchical spans across retrieval, reranking, and synthesis.
"""

import time
import uuid
from contextlib import contextmanager
from typing import Any


class Span:
    def __init__(self, name: str, parent_id: str | None = None):
        self.name: str = name
        self.span_id: str = uuid.uuid4().hex[:16]
        self.parent_id: str | None = parent_id
        self.start_time: float = time.perf_counter()
        self.end_time: float | None = None
        self.duration_ms: float = 0.0
        self.attributes: dict[str, Any] = {}
        self.status: str = "UNSET"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def finish(self, status: str = "OK") -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": self.attributes,
        }


class PipelineTracer:
    def __init__(self, service_name: str = "rag-production-engine"):
        self.service_name: str = service_name
        self.trace_id: str = uuid.uuid4().hex
        self.spans: list[Span] = []
        self._current_span: Span | None = None

    @contextmanager
    def start_as_current_span(
        self, name: str, attributes: dict[str, Any] | None = None
    ):
        parent_id = self._current_span.span_id if self._current_span else None
        span = Span(name=name, parent_id=parent_id)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)

        previous_span = self._current_span
        self._current_span = span
        self.spans.append(span)

        try:
            yield span
            span.finish(status="OK")
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("error.message", str(exc))
            span.finish(status="ERROR")
            raise
        finally:
            self._current_span = previous_span

    def get_w3c_traceparent(self) -> str:
        sampled = "01"
        root_span_id = self.spans[0].span_id if self.spans else uuid.uuid4().hex[:16]
        return f"00-{self.trace_id}-{root_span_id}-{sampled}"

    def export_trace_summary(self) -> dict[str, Any]:
        total_duration = sum(s.duration_ms for s in self.spans if s.parent_id is None)
        return {
            "service_name": self.service_name,
            "trace_id": self.trace_id,
            "traceparent": self.get_w3c_traceparent(),
            "total_root_duration_ms": round(total_duration, 3),
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }
