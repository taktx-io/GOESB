#!/usr/bin/env python3
"""One-time migration (ADR-0009): declare override-eligible parameters on
every already-committed faster-whisper/whisper-cpp profile (batch and the
one streaming profile).

Not idempotent-by-design like generate_bulk_assets.py — this rewrites
existing files in place, once. Re-running is safe (skips any profile that
already has an `overridable` block), but there's no ongoing reason to run it
again: new profiles created via generate_bulk_assets.py get `overridable`
from the start (see its own write_profile()/overridable_block_for()).

Scope: runtime.name in {faster-whisper, whisper-cpp}, benchmark_type in
{batch, streaming}. vosk gets no overridable parameters (its adapter applies
none — "no silent knobs", ADR-0009 §2/§6).

Per-engine sets (see generate_bulk_assets.overridable_block_for() and the
adapter registry in runner/src/oesb_runner/adapters/__init__.py for what
each adapter genuinely applies, not just accepts for call-shape parity):
faster-whisper gets beam_size/vad/quantization/threads (+ chunk_ms on the
streaming profile); whisper-cpp gets threads only — its adapter accepts
beam_size/vad/quantization purely for parity and never applies them.

Usage:
    python scripts/add_overridable_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_bulk_assets import overridable_block_for  # noqa: E402


def _changelog_notes(overridable: dict) -> str:
    return f"Declare {'/'.join(sorted(overridable))} override-eligible (ADR-0009)."


def _bump_minor(version: str) -> str:
    major, minor, _patch = version.split(".")
    return f"{major}.{int(minor) + 1}.0"


def migrate_profile(path: Path) -> bool:
    doc = yaml.safe_load(path.read_text())
    if "overridable" in doc:
        print(f"skip (already has overridable): {path}", file=sys.stderr)
        return False
    if doc.get("benchmark_type") not in ("batch", "streaming"):
        return False
    engine = doc.get("runtime", {}).get("name")
    overridable = overridable_block_for(engine, doc["benchmark_type"])
    if not overridable:
        return False

    new_version = _bump_minor(doc["version"])
    doc["version"] = new_version
    changelog = doc.pop("changelog", [])
    changelog.append({"version": new_version, "notes": _changelog_notes(overridable)})
    # Re-insert overridable before changelog (matches generate_bulk_assets.py's
    # write_profile() key order for newly-created profiles) rather than just
    # appending, which would put it after changelog since changelog already
    # exists in the loaded dict.
    doc["overridable"] = overridable
    doc["changelog"] = changelog

    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"migrated {path} -> {new_version}", file=sys.stderr)
    return True


def main() -> None:
    migrated = 0
    for path in sorted((ROOT / "profiles").glob("*/profile.yaml")):
        if migrate_profile(path):
            migrated += 1
    print(f"Migrated {migrated} profile(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
