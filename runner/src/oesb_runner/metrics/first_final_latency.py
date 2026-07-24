"""First Final Latency (docs/specs/metrics.md: `first_final_latency`)."""
from __future__ import annotations

from collections.abc import Sequence

from ..streaming import StreamTrace

METRIC_ID = "first_final_latency"
UNIT = "ms"


def compute(traces: Sequence[StreamTrace]) -> list[float]:
    """Per-utterance ms from first speech audio to the first finalized
    (non-revisable) token/segment. For short, single-segment utterances this
    often coincides with the last update (nothing is finalized until the
    audio itself ends) — a real property of short clips, not a bug.
    """
    latencies: list[float] = []
    for trace in traces:
        first = next((u for u in trace.updates if u.committed_word_count > 0), None)
        if first is None:
            raise ValueError(f"utterance {trace.utterance_id!r} never finalized any words")
        latencies.append(first.emit_time_s * 1000.0)
    return latencies
