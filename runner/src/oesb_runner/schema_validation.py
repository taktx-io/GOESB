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
from packaging.version import Version


def load_schema(filename: str) -> dict[str, Any]:
    return json.loads(resources.files("oesb_runner").joinpath("schemas", filename).read_text())


def validate_against(data: dict[str, Any], schema_filename: str) -> list[str]:
    validator = Draft202012Validator(load_schema(schema_filename))
    return [err.message for err in validator.iter_errors(data)]


def unrecognized_pack_source_type(pack_data: dict[str, Any]) -> str | None:
    """If `pack_data.audio.source.type` is set to a value this runner's own
    bundled schema doesn't know about, return it — signals a pack fetched
    from a newer platform than this runner understands (e.g. a new
    `mozilla_data_collective`-style provider), as opposed to any other kind
    of schema failure. `None` if the type is absent, known, or the pack has
    no `audio.source` at all."""
    source_type = pack_data.get("audio", {}).get("source", {}).get("type")
    if source_type is None:
        return None
    schema = load_schema("benchmark-pack.schema.json")
    known = schema["properties"]["audio"]["properties"]["source"]["properties"]["type"]["enum"]
    return source_type if source_type not in known else None


def unmet_min_runner_version(pack_data: dict[str, Any], installed_version: str) -> str | None:
    """If `pack_data.min_runner_version` is set and `installed_version` is
    older than it, return the required version — an explicit floor a pack
    can declare (e.g. it relies on a manifest.jsonl field an older
    `load_pack` doesn't know to check), independent of whether the pack.yaml
    itself happens to validate against this runner's schema. `None` if the
    field is absent or the installed version already satisfies it."""
    required = pack_data.get("min_runner_version")
    if required is None:
        return None
    return required if Version(installed_version) < Version(required) else None
