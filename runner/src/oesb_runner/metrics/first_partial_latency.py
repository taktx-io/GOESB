"""First Partial Latency (docs/specs/metrics.md: `first_partial_latency`)."""
from __future__ import annotations

from typing import Sequence

from ..streaming import StreamTrace

METRIC_ID = "first_partial_latency"
UNIT = "ms"


def compute(traces: Sequence[StreamTrace]) -> list[float]:
    """Per-utterance ms from first speech audio to the first non-empty partial
    hypothesis. Returns one value per utterance, not a corpus aggregate —
    unlike WER/RTF, latency's spec-mandated aggregation is p50/p95 across the
    pack (docs/specs/metrics.md "Reporting"), computed by the caller.
    """
    latencies: list[float] = []
    for trace in traces:
        first = next((u for u in trace.updates if u.text), None)
        if first is None:
            raise ValueError(f"utterance {trace.utterance_id!r} produced no partial hypothesis")
        latencies.append(first.emit_time_s * 1000.0)
    return latencies
