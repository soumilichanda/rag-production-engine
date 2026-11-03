"""
app/telemetry/metrics.py
Production Prometheus exposition collector with monotonic latency histograms.
"""

import threading
import time
from typing import Dict, List


class PrometheusMetricsExporter:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PrometheusMetricsExporter, cls).__new__(cls)
                cls._instance._init_metrics()
            return cls._instance

    def _init_metrics(self):
        self._request_counter: Dict[str, int] = {}
        self._error_counter: Dict[str, int] = {}
        # Latency buckets in milliseconds
        self._histogram_buckets: List[float] = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]
        self._histogram_observations: Dict[str, List[float]] = {}
        self._metric_lock = threading.Lock()

    def record_request(self, endpoint: str, status_code: int, duration_ms: float):
        with self._metric_lock:
            key = f'{endpoint}:{status_code}'
            self._request_counter[key] = self._request_counter.get(key, 0) + 1

            if status_code >= 400:
                self._error_counter[endpoint] = self._error_counter.get(endpoint, 0) + 1

            if endpoint not in self._histogram_observations:
                self._histogram_observations[endpoint] = []
            self._histogram_observations[endpoint].append(duration_ms)

    def scrape_metrics(self) -> str:
        lines: List[str] = [
            "# HELP rag_http_requests_total Total HTTP requests processed by endpoint and code.",
            "# TYPE rag_http_requests_total counter",
        ]

        with self._metric_lock:
            for key, count in sorted(self._request_counter.items()):
                endpoint, code = key.rsplit(":", 1)
                lines.append(f'rag_http_requests_total{{endpoint="{endpoint}",status="{code}"}} {count}')

            lines.append("# HELP rag_request_duration_ms HTTP request latency summary in milliseconds.")
            lines.append("# TYPE rag_request_duration_ms histogram")

            for endpoint, observations in sorted(self._histogram_observations.items()):
                total_count = len(observations)
                total_sum = sum(observations)
                for bucket in self._histogram_buckets:
                    cumulative = sum(1 for obs in observations if obs <= bucket)
                    lines.append(
                        f'rag_request_duration_ms_bucket{{endpoint="{endpoint}",le="{bucket}"}} {cumulative}'
                    )
                lines.append(f'rag_request_duration_ms_bucket{{endpoint="{endpoint}",le="+Inf"}} {total_count}')
                lines.append(f'rag_request_duration_ms_sum{{endpoint="{endpoint}"}} {round(total_sum, 3)}')
                lines.append(f'rag_request_duration_ms_count{{endpoint="{endpoint}"}} {total_count}')

        return "\n".join(lines) + "\n"


metrics_exporter = PrometheusMetricsExporter()
