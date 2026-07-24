#!/usr/bin/env python3
"""Precompute adapter source hashes for a frozen (PyInstaller) build.

sha256_module_source() normally hashes a running adapter's own .py file
directly — meaningless once frozen, since bundled modules import straight
out of the archive with no live source file on disk. Run this BEFORE
invoking PyInstaller: it writes a manifest of the same hashes, computed from
the real source while it still exists, which hashing.py's frozen-build
fallback reads back at runtime (see hashing.py's _frozen_module_hash).

Usage:
    python scripts/generate_frozen_adapter_hashes.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner" / "src"))

from oesb_runner.hashing import sha256_module_source  # noqa: E402

# The adapter modules sha256_module_source is actually called on (see
# oesb_runner.adapters.__init__'s registry) — one entry per runtime.
ADAPTER_MODULES = [
    "oesb_runner.adapters.faster_whisper",
    "oesb_runner.adapters.vosk",
    "oesb_runner.adapters.whisper_cpp",
]


def main() -> int:
    manifest = {}
    for name in ADAPTER_MODULES:
        module = importlib.import_module(name)
        manifest[name] = sha256_module_source(module)

    out_path = ROOT / "runner" / "src" / "oesb_runner" / "_frozen_adapter_hashes.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path}:")
    for name, digest in manifest.items():
        print(f"  {name}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
