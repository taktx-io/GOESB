"""Temperature (docs/specs/metrics.md: `temperature_c`) — peak package/SoC
temperature during the run, a throttling indicator."""
from __future__ import annotations

METRIC_ID = "temperature_c"
UNIT = "°C"


def reduce_peak_temp_c(samples: list[float]) -> float:
    """Peak temperature across the run's hwmon samples — same reducer shape
    as `cpu_ram.reduce_peak_ram_mb`."""
    if not samples:
        raise ValueError("at least one sample required")
    return max(samples)
