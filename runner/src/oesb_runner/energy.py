"""Energy + thermal probes (docs/specs/metrics.md: `energy_wh`,
`temperature_c`; roadmap M2).

Both probes are Linux-only by construction — RAPL (`intel-rapl` sysfs) and
hwmon are Linux kernel interfaces with no macOS/Windows equivalent exposed to
userspace without elevated privileges (macOS `powermetrics` needs sudo;
Windows has no public per-process energy counter). `None` on any other
platform, or when the sysfs tree simply isn't present (e.g. a non-Intel/AMD
RAPL-less Linux box, a Linux VM/container with no `/sys/class/powercap`) —
never a silent zero, matching the environment fingerprint's own
never-omit-just-explain convention (docs/specs/environment-capture.md).

Both accept an injectable `root` so they're unit-testable against a synthetic
sysfs fixture tree without needing real Linux hardware.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_RAPL_ROOT = Path("/sys/class/powercap")
DEFAULT_HWMON_ROOT = Path("/sys/class/hwmon")


def read_rapl_uj(root: Path = DEFAULT_RAPL_ROOT) -> float | None:
    """Sum of `energy_uj` across every `intel-rapl:*` domain, in microjoules.

    A single point-in-time counter reading, not a delta — callers read this
    once before and once after the timed work and subtract, same shape as
    the existing CPU/RAM sampler (`cpu_ram.sample_process_tree` +
    reducers). RAPL counters wrap around at a platform-specific max value;
    detecting/handling wraparound is out of scope for a run short enough
    that wraparound within it is not a realistic concern.
    """
    if not root.exists():
        return None
    total_uj = 0.0
    found = False
    for domain_dir in sorted(root.glob("intel-rapl:*")):
        energy_file = domain_dir / "energy_uj"
        if not energy_file.exists():
            continue
        try:
            total_uj += float(energy_file.read_text().strip())
            found = True
        except (OSError, ValueError):
            continue
    return total_uj if found else None


def sample_hwmon_temp_c(root: Path = DEFAULT_HWMON_ROOT) -> float | None:
    """Max temperature (°C) across every `hwmon*/temp*_input` sensor.

    hwmon reports millidegrees Celsius; peak (not mean) is what matters for
    a throttling indicator, same reasoning as `cpu_ram.reduce_peak_ram_mb`.
    """
    if not root.exists():
        return None
    peak_c: float | None = None
    for hwmon_dir in sorted(root.glob("hwmon*")):
        for temp_file in sorted(hwmon_dir.glob("temp*_input")):
            try:
                millidegrees = float(temp_file.read_text().strip())
            except (OSError, ValueError):
                continue
            celsius = millidegrees / 1000.0
            if peak_c is None or celsius > peak_c:
                peak_c = celsius
    return peak_c


def hwmon_available(root: Path = DEFAULT_HWMON_ROOT) -> bool:
    """Whether any hwmon sensor is readable at all — used for the
    environment fingerprint's best-effort `cooling` field, independent of
    any single run's temperature sampling."""
    return sample_hwmon_temp_c(root) is not None
