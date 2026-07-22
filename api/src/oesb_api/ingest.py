"""Result ingestion + trust gate (M3, docs/03-roadmap.md; ADR-0004).

The actual re-verification the roadmap promises: schema, then hash +
signature (reusing the runner's own primitives — a submitted result is
re-checked exactly the way the runner checked itself before writing the
file, never trusted because it merely looks well-formed), then
official-profile / open-pack membership (assets.py). Only after every check
passes does a result get stored.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from oesb_runner.hashing import canonical_asset_sha256
from oesb_runner.schema_validation import validate_against
from oesb_runner.signing import verify_result_document
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .assets import Assets
from .models import Result


class IngestRejected(HTTPException):
    """Every rejection path raises this — routes don't need their own
    try/except, FastAPI already turns any HTTPException into the right
    response."""


def verify_and_ingest(body: dict[str, Any], db: Session, assets: Assets) -> Result:
    schema_errors = validate_against(body, "benchmark-result.schema.json")
    if schema_errors:
        raise IngestRejected(
            status_code=422, detail={"reason": "schema_invalid", "errors": schema_errors}
        )

    if not verify_result_document(body):
        raise IngestRejected(status_code=400, detail={"reason": "hash_or_signature_invalid"})

    profile_ref = body["profile"]
    profile = assets.profiles.get(profile_ref["id"])
    if profile is None or profile["version"] != profile_ref["version"]:
        raise IngestRejected(status_code=403, detail={"reason": "not_an_official_profile"})
    # Same computation cli.py performed when it produced this result
    # (canonical_asset_sha256(profile, exclude=()) — profiles have no
    # self-declared hash field, unlike packs) — re-derived from the actual
    # committed profile.yaml, not trusted from the submission.
    if canonical_asset_sha256(profile, exclude=()) != profile_ref["sha256"]:
        raise IngestRejected(status_code=403, detail={"reason": "profile_hash_mismatch"})

    pack_ref = body["pack"]
    pack = assets.packs.get(pack_ref["id"])
    if pack is None or pack["sha256"] != pack_ref["sha256"]:
        raise IngestRejected(status_code=403, detail={"reason": "not_a_known_pack"})
    if pack["visibility"] != "open" or pack_ref["visibility"] != "open":
        raise IngestRejected(status_code=403, detail={"reason": "pack_not_open"})

    values = {
        "id": body["payload_sha256"],
        "document": body,
        "profile_id": profile_ref["id"],
        "profile_version": profile_ref["version"],
        "pack_id": pack_ref["id"],
        "runtime_name": body["runtime"]["name"],
        "model_name": body["model"]["name"],
        "benchmark_type": profile["benchmark_type"],
        "language": profile.get("language"),
        "timestamp": body["timestamp"],
    }
    # payload_sha256 is content-addressed: resubmitting an identical result
    # is a no-op, not a duplicate-key error — natural idempotency, no
    # separate dedup logic needed.
    stmt = pg_insert(Result).values(**values).on_conflict_do_nothing(index_elements=["id"])
    db.execute(stmt)
    db.commit()
    return db.get(Result, values["id"])
