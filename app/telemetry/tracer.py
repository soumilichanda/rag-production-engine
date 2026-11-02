"""
app/telemetry/tracer.py
Phase 20: OpenTelemetry Distributed Tracer & W3C Span Context Propagation.
Provides hierarchical span creation, error status capture, and W3C traceparent formatting.
"""

import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class TelemetrySpan:
    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.name: str = name
        self.trace_id: str = trace_id
        self.span_id: str = uuid.uuid4().hex[:16]
        self.parent_id: Optional[str] = parent_id
        self.start_time: float = time.perf_counter()
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.attributes: Dict[str, Any] = {}
        self.status: str = "OK"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def finish(self, status: str = "OK") -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": self.attributes,
        }


class DistributedPipelineTracer:
    def __init__(self, service_name: str = "rag-production-engine"):
        self.service_name: str = service_name
        self.trace_id: str = uuid.uuid4().hex
        self.spans: List[TelemetrySpan] = []
        self._current_span: Optional[TelemetrySpan] = None

    @contextmanager
    def start_as_current_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        parent_id = self._current_span.span_id if self._current_span else None
        span = TelemetrySpan(name=name, trace_id=self.trace_id, parent_id=parent_id)
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

    def export_telemetry_batch(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "trace_id": self.trace_id,
            "traceparent": self.get_w3c_traceparent(),
            "total_spans": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }
