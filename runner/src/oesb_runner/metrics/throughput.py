"""Throughput (docs/specs/metrics.md: `throughput`) — seconds of audio
processed per wall-clock second. Higher is better; ~= 1/RTF for a
sequential batch run, but computed independently since a concurrent run's
wall-clock time isn't the sum of its per-utterance processing times."""
from __future__ import annotations

METRIC_ID = "throughput"
UNIT = "audio-s/s"


def compute(total_audio_s: float, wall_s: float) -> float:
    if wall_s <= 0:
        raise ValueError("wall_s must be > 0")
    return total_audio_s / wall_s
