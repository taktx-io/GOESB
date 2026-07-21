"""Real-Time Factor (docs/specs/metrics.md: `real_time_factor`)."""
from __future__ import annotations

METRIC_ID = "real_time_factor"
UNIT = "ratio"


def compute(processing_time_s: float, audio_duration_s: float) -> float:
    """`processing_time / audio_duration`. Pass corpus totals for a pack-level RTF."""
    if audio_duration_s <= 0:
        raise ValueError("audio_duration_s must be > 0")
    return processing_time_s / audio_duration_s
