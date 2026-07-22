"""In-memory profiles/packs asset registry (M3, docs/03-roadmap.md).

Profiles and packs are static, git-reviewed data — every `profile.yaml` in
the monorepo *is* official by definition (see docs/04-glossary.md: a
Benchmark Profile is "official, versioned"; there is no separate "official"
flag anywhere in the schema). So rather than a DB table, this loads +
schema-validates every profile/pack once at startup; `ingest.py` checks a
submitted result's profile/pack id+version+sha256 against exactly this set
as the "official profile / open pack" gate (FR-7.3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from oesb_runner.hashing import canonical_asset_sha256
from oesb_runner.schema_validation import _find_repo_schemas_dir, validate_against

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Assets:
    profiles: dict[str, dict[str, Any]]  # profile_id -> profile document
    packs: dict[str, dict[str, Any]]  # pack_id -> pack document
    # Assets present on disk but skipped (bad schema/hash) — e.g. a
    # documented placeholder pack awaiting real audio (consent-gated
    # datasets). Logged as a warning, not a startup crash: one incomplete
    # asset shouldn't take the whole service down, but it also shouldn't be
    # silently served as if valid.
    skipped: list[str] = field(default_factory=list)


def _repo_root() -> Path:
    return _find_repo_schemas_dir().parent


def _load_profiles(root: Path, skipped: list[str]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("profiles/*/profile.yaml")):
        data = yaml.safe_load(path.read_text())
        errors = validate_against(data, "benchmark-profile.schema.json")
        if errors:
            logger.warning("skipping %s: failed schema validation: %s", path, errors)
            skipped.append(str(path))
            continue
        loaded[data["id"]] = data
    return loaded


def _load_packs(root: Path, skipped: list[str]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("packs/*/pack.yaml")):
        data = yaml.safe_load(path.read_text())
        errors = validate_against(data, "benchmark-pack.schema.json")
        if errors:
            logger.warning("skipping %s: failed schema validation: %s", path, errors)
            skipped.append(str(path))
            continue
        declared_sha256 = data["sha256"]
        actual_sha256 = canonical_asset_sha256(data)  # excludes "sha256" by default
        if actual_sha256 != declared_sha256:
            logger.warning(
                "skipping %s: content hash mismatch (declared %s, computed %s) "
                "— likely a documented placeholder pending real data",
                path, declared_sha256, actual_sha256,
            )
            skipped.append(str(path))
            continue
        loaded[data["id"]] = data
    return loaded


def load_assets(root: Path | None = None) -> Assets:
    root = root if root is not None else _repo_root()
    skipped: list[str] = []
    profiles = _load_profiles(root, skipped)
    packs = _load_packs(root, skipped)
    return Assets(profiles=profiles, packs=packs, skipped=skipped)


_cached: Assets | None = None


def get_assets() -> Assets:
    """FastAPI dependency: profiles/packs are static within a process
    lifetime (they only change via a new deployment), so load once and
    reuse — not a per-request filesystem walk."""
    global _cached
    if _cached is None:
        _cached = load_assets()
    return _cached
