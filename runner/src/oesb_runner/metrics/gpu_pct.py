"""GPU utilisation (docs/specs/metrics.md: `gpu_pct`) -- mean GPU utilisation
during the run, where an NVIDIA GPU is present and its driver's NVML library
is loadable.

Reads NVML directly via `nvidia-ml-py` (the official binding NVIDIA
publishes, imports as `pynvml`) instead of shelling out to the `nvidia-smi`
CLI: `nvidia-smi`'s own utilization.gpu column is itself just a thin wrapper
over this same `nvmlDeviceGetUtilizationRates()` call, so this reads
identical data with the process-spawn-and-CSV-parse peeled away -- in-process
like `cpu_ram.py`'s psutil sampling, not the per-sample subprocess overhead
`_sample_during`'s own docstring (cli.py) explains this metric used to need
a coarser sampling cadence to work around.

CUDA's own runtime/driver API has no equivalent query -- it manages kernels,
memory, and streams, not system-level utilization. NVML is the correct,
separate management-layer API for this, the same one `nvidia-smi` itself
calls under the hood. Still NVIDIA-only either way: no vendor-neutral GPU
utilization API exists, so an Apple Silicon/AMD equivalent would need its
own, unrelated probe.

`nvidia-ml-py` is an optional dependency (the `cuda` extra) -- absent
entirely on a machine with no NVIDIA GPU to observe (every sample just
returns `None`), same "absent is itself information, never a fabricated
zero" convention as RAPL/hwmon.
"""
from __future__ import annotations

try:
    import pynvml
except ImportError:
    pynvml = None

METRIC_ID = "gpu_pct"
UNIT = "%"

_handle = None
_unavailable = pynvml is None


def _device_handle():
    """NVML init + a GPU 0 handle, cached at module scope for the life of
    the process -- nvmlInit() is expensive enough (loads the driver
    library, opens a device handle) that redoing it every sample would
    reintroduce the exact per-call overhead this module exists to remove.
    Never explicitly torn down via nvmlShutdown(): the process exits when
    the run does, the same "don't bother tearing down" treatment
    `cpu_ram.py`'s own un-closed `psutil.Process()` already gets.

    Only ever selects GPU 0 (`nvmlDeviceGetHandleByIndex(0)`) -- a
    pre-existing single-GPU assumption carried over unchanged from the
    nvidia-smi implementation this replaces (its CSV output has one row
    per GPU; the old code only ever parsed a single float out of it)."""
    global _handle, _unavailable
    if _unavailable:
        return None
    if _handle is not None:
        return _handle
    try:
        pynvml.nvmlInit()
        _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        _unavailable = True
        return None
    return _handle


def sample_gpu_pct() -> float | None:
    """One utilisation reading, or `None` on any failure (no NVIDIA GPU, no
    NVML library, a transient probe error) -- same "absent is itself
    information, never a fabricated zero" convention as RAPL/hwmon."""
    handle = _device_handle()
    if handle is None:
        return None
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except pynvml.NVMLError:
        return None


def reduce_mean_gpu_pct(samples: list[float]) -> float:
    """Mean GPU utilisation across the run's samples."""
    if not samples:
        raise ValueError("at least one sample required")
    return sum(samples) / len(samples)
