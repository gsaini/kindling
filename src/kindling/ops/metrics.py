"""Prometheus metrics, in the default registry so other modules can add their own and have them exported.

Defined once per process: prometheus_client refuses to register a name twice, and tests build many apps.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# LLM requests are slow, so the buckets reach well past the usual web defaults.
_LATENCY_BUCKETS = (0.005, 0.025, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)

HTTP_REQUESTS = Counter(
    "kindling_http_requests_total", "HTTP requests by method, route template and status", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "kindling_http_request_duration_seconds",
    "Time from request to the end of the response body (whole stream for streaming replies)",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
)

GENERATIONS_IN_FLIGHT = Gauge("kindling_generations_in_flight", "Model streams holding a slot in the gate")
GENERATIONS_QUEUED = Gauge("kindling_generations_queued", "Model streams waiting for a slot")
GENERATIONS_REJECTED = Counter(
    "kindling_generations_rejected_total", "Model streams turned away by the gate", ["reason"]
)  # reason: queue_full / timeout
QUEUE_WAIT = Histogram(
    "kindling_generation_queue_wait_seconds", "Time spent waiting for a slot", buckets=_LATENCY_BUCKETS
)

GENERATIONS = Counter("kindling_generations_total", "Model streams that finished", ["model"])
GENERATION_SECONDS = Histogram(
    "kindling_generation_duration_seconds", "Time a model stream held its slot", ["model"], buckets=_LATENCY_BUCKETS
)
PROMPT_TOKENS = Counter("kindling_prompt_tokens_total", "Prompt tokens read, from the model's done event", ["model"])
OUTPUT_TOKENS = Counter("kindling_output_tokens_total", "Tokens generated, from the model's done event", ["model"])

_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}


def observe_request(method: str, route: str, status: int, seconds: float) -> None:
    method = method if method in _METHODS else "OTHER"  # keep label values bounded
    HTTP_REQUESTS.labels(method, route, str(status)).inc()
    HTTP_LATENCY.labels(method, route).observe(seconds)
