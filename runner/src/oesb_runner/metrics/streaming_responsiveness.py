"""Streaming Responsiveness (docs/specs/metrics.md: `streaming_responsiveness`).

The spec defines this as "a composite of update frequency and stability
against latency; defined per profile" and leaves the exact formula open.
This is OESB's default, documented here since nothing else pins it down:
rewards higher update rate and higher stability, penalizes higher first-
partial latency. Profiles may define a different formula in a future
schema revision; until then every streaming profile gets this one.
"""
from __future__ import annotations

METRIC_ID = "streaming_responsiveness"
UNIT = "index"


def compute(
    update_frequency_hz: float,
    partial_stability: float,
    first_partial_latency_p50_ms: float,
) -> float:
    if first_partial_latency_p50_ms <= 0:
        raise ValueError("first_partial_latency_p50_ms must be > 0")
    first_partial_latency_p50_s = first_partial_latency_p50_ms / 1000.0
    return max(0.0, (update_frequency_hz * partial_stability) / first_partial_latency_p50_s)
