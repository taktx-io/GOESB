#!/usr/bin/env python3
"""Validate all profiles and pack manifests against the JSON Schemas.

Run in CI and locally. Exits non-zero on the first invalid asset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
    from jsonschema import Draft202012Validator
except ImportError:
    print("Install deps: pip install jsonschema pyyaml", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent.parent


def load_schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text())


def validate_dir(subdir: str, filename: str, schema: dict) -> list[str]:
    errors: list[str] = []
    validator = Draft202012Validator(schema)
    for path in (ROOT / subdir).glob(f"*/{filename}"):
        data = yaml.safe_load(path.read_text())
        for err in validator.iter_errors(data):
            errors.append(f"{path}: {err.message}")
    return errors


def validate_file(path: Path, schema: dict) -> list[str]:
    errors: list[str] = []
    validator = Draft202012Validator(schema)
    data = json.loads(path.read_text())
    for err in validator.iter_errors(data):
        errors.append(f"{path}: {err.message}")
    return errors


def main() -> int:
    errors: list[str] = []
    errors += validate_dir("profiles", "profile.yaml",
                           load_schema("benchmark-profile.schema.json"))
    errors += validate_dir("packs", "pack.yaml",
                           load_schema("benchmark-pack.schema.json"))
    errors += validate_file(
        ROOT / "schemas" / "examples" / "benchmark-result.example.json",
        load_schema("benchmark-result.schema.json"),
    )
    if errors:
        print("INVALID assets:")
        for e in errors:
            print("  -", e)
        return 1
    print("All profiles and packs validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
