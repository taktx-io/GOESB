from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..assets import Assets, get_assets
from ..db import get_db
from ..ingest import verify_and_ingest
from ..models import Result
from ..schemas import SubmitBenchmarkResponse

router = APIRouter(prefix="/benchmark", tags=["benchmarks"])


@router.post("", status_code=201, response_model=SubmitBenchmarkResponse)
def submit_benchmark(
    body: dict[str, Any],
    db: Session = Depends(get_db),
    assets: Assets = Depends(get_assets),
) -> SubmitBenchmarkResponse:
    """Submit a signed benchmark result for verification & ranking.

    Re-verifies schema, hash, and signature (ADR-0004) and checks
    official-profile/open-pack membership (FR-7.3) before storing anything —
    see ingest.py. Rejections raise HTTPException with a machine-readable
    `reason` (schema_invalid / hash_or_signature_invalid /
    not_an_official_profile / profile_hash_mismatch / not_a_known_pack /
    pack_not_open), never a silent partial acceptance.
    """
    result = verify_and_ingest(body, db, assets)
    return SubmitBenchmarkResponse(id=result.id, accepted=True)


@router.get("/{benchmark_id}")
def get_benchmark(benchmark_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = db.get(Result, benchmark_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"reason": "not_found"})
    return result.document
