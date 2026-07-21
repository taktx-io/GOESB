"""JSON Schema validation for profiles, packs, and results.

Locates schemas/ by walking up from this file to the OESB monorepo root.
Standalone (non-monorepo) packaging of schemas is an M2 concern — not needed
for M1's "runs on one machine locally" exit criterion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def _find_repo_schemas_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "schemas"
        if (candidate / "benchmark-profile.schema.json").exists():
            return candidate
    raise RuntimeError(
        "could not locate schemas/ directory; oesb-runner's schema validation "
        "currently requires running from within an OESB monorepo checkout "
        "(standalone schema packaging is tracked for M2)"
    )


def load_schema(filename: str) -> dict[str, Any]:
    return json.loads((_find_repo_schemas_dir() / filename).read_text())


def validate_against(data: dict[str, Any], schema_filename: str) -> list[str]:
    validator = Draft202012Validator(load_schema(schema_filename))
    return [err.message for err in validator.iter_errors(data)]
