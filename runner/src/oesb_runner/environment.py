"""Reproducibility environment capture (scaffold).

Captures a best-effort fingerprint of the machine the benchmark runs on.
The full spec lives in docs/specs/environment-capture.md.
"""
from __future__ import annotations

import platform
from typing import Any


def capture_environment() -> dict[str, Any]:
    """Return a partial environment fingerprint. Extended in later iterations."""
    return {
        "schema_version": "0.1",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        # TODO(M1): CPU/GPU/NPU, RAM, drivers, firmware, thermals, power.
    }
