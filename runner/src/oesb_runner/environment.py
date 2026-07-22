"""Reproducibility environment capture.

Captures a best-effort fingerprint of the machine the benchmark runs on, per
docs/specs/environment-capture.md. Fields that cannot be probed on the current
platform are set to `null` and explained in `unavailable` — never silently
omitted, since a missing field is itself information (FR-5.1/5.2).
"""
from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil

from . import energy

SCHEMA_VERSION = "0.2"


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _capture_cpu(unavailable: dict[str, str]) -> dict[str, Any]:
    model: str | None = None
    system = platform.system()
    if system == "Darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            model = None
    elif system == "Windows":
        model = platform.processor() or None

    if model is None:
        unavailable["cpu.model"] = f"no CPU model probe implemented for {system!r}"

    freq = psutil.cpu_freq()
    return {
        "model": model,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "base_clock_mhz": round(freq.min, 1) if freq and freq.min else None,
        "max_clock_mhz": round(freq.max, 1) if freq and freq.max else None,
    }


def _capture_gpu(unavailable: dict[str, str]) -> dict[str, Any] | None:
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    if smi:
        name, mem, driver = (p.strip() for p in smi.split(",", 2))
        return {"model": name, "vram": mem, "driver": driver}
    unavailable["gpu"] = "no NVIDIA GPU/nvidia-smi found on this machine"
    return None


def _capture_power(unavailable: dict[str, str]) -> dict[str, Any] | None:
    try:
        battery = psutil.sensors_battery()
    except Exception:  # pragma: no cover - platform-dependent
        battery = None
    if battery is None:
        unavailable["power"] = "no battery present or no power-source probe on this platform"
        return None
    return {
        "source": "AC" if battery.power_plugged else "battery",
        "battery_percent": battery.percent,
    }


def _capture_cooling(unavailable: dict[str, str]) -> dict[str, Any] | None:
    """Best-effort cooling/thermal-sensor presence signal (M2) — hwmon is
    Linux-only, so this is `null` with a reason on macOS/Windows, same
    convention as every other unprobed field here."""
    if not energy.hwmon_available():
        unavailable["cooling"] = "no hwmon thermal sensors found on this platform"
        return None
    return {"active_sensor_present": True}


def capture_environment() -> dict[str, Any]:
    """Return the best-effort environment fingerprint for this machine."""
    unavailable: dict[str, str] = {}

    fingerprint: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "cpu": _capture_cpu(unavailable),
        "ram": {"total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1)},
        "gpu": _capture_gpu(unavailable),
        "npu": None,
        "storage": None,
        "firmware": None,
        "cooling": _capture_cooling(unavailable),
        "power": _capture_power(unavailable),
    }

    unavailable.setdefault("npu", "no NPU probe implemented")
    unavailable.setdefault("storage", "storage type/model probe not implemented")
    unavailable.setdefault("firmware", "no BIOS/firmware version probe implemented")
    # "cooling" is set by _capture_cooling itself when unavailable — no
    # setdefault needed (and one would wrongly overwrite a real reading with
    # a stale "not implemented" reason once it is implemented).

    fingerprint["unavailable"] = dict(sorted(unavailable.items()))
    return fingerprint
