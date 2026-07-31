"""GPU utilisation (docs/specs/metrics.md: `gpu_pct`) — mean GPU utilisation
during the run, where an NVIDIA GPU and `nvidia-smi` are actually present.

Sampled via `nvidia-smi`, same subprocess idiom `environment.py`'s
`_capture_gpu` already uses for the one-shot model/driver/VRAM probe — this
is the repeated, in-run counterpart. Each call spawns a process (unlike
`cpu_ram`'s in-process `psutil` sampling or `energy.py`'s hwmon file reads),
so the caller samples this less often than the 200ms CPU/RAM/temperature
tick to keep subprocess overhead from skewing the very timing it measures.
"""
from __future__ import annotations

from ..environment import _run

METRIC_ID = "gpu_pct"
UNIT = "%"


def sample_gpu_pct() -> float | None:
    """One utilisation reading, or `None` on any failure (no NVIDIA GPU, no
    `nvidia-smi`, a transient probe error) — same "absent is itself
    information, never a fabricated zero" convention as RAPL/hwmon."""
    out = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    if out is None:
        return None
    try:
        return float(out.strip())
    except ValueError:
        return None


def reduce_mean_gpu_pct(samples: list[float]) -> float:
    """Mean GPU utilisation across the run's samples."""
    if not samples:
        raise ValueError("at least one sample required")
    return sum(samples) / len(samples)
