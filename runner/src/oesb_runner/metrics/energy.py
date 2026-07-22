"""Energy (docs/specs/metrics.md: `energy_wh`)."""
from __future__ import annotations

METRIC_ID = "energy_wh"
UNIT = "Wh"

_UJ_PER_WH = 1_000_000.0 * 3600.0  # microjoules -> joules -> watt-hours


def compute(delta_uj: float) -> float:
    """Wh consumed, from a microjoule delta between two
    `energy.read_rapl_uj()` readings taken before and after the timed work
    (or an explicit external-meter reading, already in the same unit
    conversion path). A negative delta (a RAPL counter wraparound mid-run)
    is rejected rather than silently reported as negative energy — a real
    result would need a wraparound-aware re-read strategy, out of scope here
    (see energy.py's own docstring)."""
    if delta_uj < 0:
        raise ValueError(
            "RAPL energy counter went backwards (likely wraparound); "
            "cannot compute a reliable delta for this run"
        )
    return delta_uj / _UJ_PER_WH
