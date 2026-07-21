"""CPU/RAM sampling (docs/specs/metrics.md: `cpu_pct`, `ram_mb`).

`cpu_pct` is mean CPU across the run; `ram_mb` is peak resident memory of the
benchmark process tree — the two use different reducers, so sampling and
reduction are kept separate.
"""
from __future__ import annotations

from typing import TypedDict

import psutil

CPU_METRIC_ID = "cpu_pct"
CPU_UNIT = "%"
RAM_METRIC_ID = "ram_mb"
RAM_UNIT = "MB"


class Sample(TypedDict):
    cpu_pct: float
    rss_mb: float


def sample_process_tree(proc: psutil.Process) -> Sample:
    """One CPU%/RSS sample across `proc` and all its live children."""
    procs = [proc, *proc.children(recursive=True)]
    cpu = 0.0
    rss = 0
    for p in procs:
        try:
            cpu += p.cpu_percent(interval=None)
            rss += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"cpu_pct": cpu, "rss_mb": rss / (1024 * 1024)}


def reduce_cpu_pct(samples: list[Sample]) -> float:
    """Mean CPU% across the run."""
    if not samples:
        raise ValueError("at least one sample required")
    return sum(s["cpu_pct"] for s in samples) / len(samples)


def reduce_peak_ram_mb(samples: list[Sample]) -> float:
    """Peak resident memory across the run."""
    if not samples:
        raise ValueError("at least one sample required")
    return max(s["rss_mb"] for s in samples)
