"""Update Frequency (docs/specs/metrics.md: `update_frequency`)."""
from __future__ import annotations

from collections.abc import Sequence

from ..streaming import StreamTrace

METRIC_ID = "update_frequency"
UNIT = "Hz"


def compute(traces: Sequence[StreamTrace]) -> float:
    """Corpus-level rate of text-changing partial-hypothesis updates: total
    updates whose text differs from the previous update, over total audio
    duration — aggregated across the whole pack, like WER, not averaged
    per-utterance."""
    total_updates = 0
    total_duration_s = 0.0
    for trace in traces:
        total_duration_s += trace.audio_duration_s
        previous_text = None
        for update in trace.updates:
            if update.text != previous_text:
                total_updates += 1
            previous_text = update.text
    if total_duration_s <= 0:
        raise ValueError("total audio duration must be > 0")
    return total_updates / total_duration_s
