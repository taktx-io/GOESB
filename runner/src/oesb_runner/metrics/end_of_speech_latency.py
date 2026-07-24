"""End-of-Speech Latency (docs/specs/metrics.md: `end_of_speech_latency`)."""
from __future__ import annotations

from collections.abc import Sequence

from ..streaming import StreamTrace

METRIC_ID = "end_of_speech_latency"
UNIT = "ms"


def compute(traces: Sequence[StreamTrace]) -> list[float]:
    """Per-utterance ms from the true end of speech (last chunk's audio
    boundary) to the final transcript being emitted."""
    latencies: list[float] = []
    for trace in traces:
        if not trace.updates:
            raise ValueError(f"utterance {trace.utterance_id!r} produced no updates")
        last = trace.updates[-1]
        latencies.append((last.emit_time_s - trace.audio_duration_s) * 1000.0)
    return latencies
