"""JSON Schema validation for profiles, packs, and results.

Schemas ship as package data inside `oesb_runner` itself (`importlib.resources`)
— works identically whether the package is installed normally, via `pip -e`,
or frozen into a standalone binary (PyInstaller), unlike the old approach of
walking up from this file to a sibling `schemas/` directory in a full GOESB
monorepo checkout, which only a real checkout (or an editable install of one)
could ever satisfy.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator


def load_schema(filename: str) -> dict[str, Any]:
    return json.loads(resources.files("oesb_runner").joinpath("schemas", filename).read_text())


def validate_against(data: dict[str, Any], schema_filename: str) -> list[str]:
    validator = Draft202012Validator(load_schema(schema_filename))
    return [err.message for err in validator.iter_errors(data)]
