from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Result
from ..schemas import LeaderboardEntry, LeaderboardResponse

router = APIRouter(prefix="/leaderboards", tags=["leaderboards"])


@router.get("", response_model=LeaderboardResponse)
def list_leaderboards(
    db: Session = Depends(get_db),
    benchmark_type: str | None = Query(None),
    profile: str | None = Query(None),
    language: str | None = Query(None),
    runtime: str | None = Query(None),
    model: str | None = Query(None),
    hardware: str | None = Query(None),
    max_price_eur: float | None = Query(None),
) -> LeaderboardResponse:
    """Unfiltered leaderboard: every accepted result. Filter params are
    accepted for API-shape stability but not yet applied (M4 adds real
    filtering, per docs/03-roadmap.md)."""
    rows = db.execute(select(Result).order_by(Result.ingested_at.desc())).scalars().all()
    return LeaderboardResponse(
        filters={"benchmark_type": benchmark_type, "language": language},
        results=[
            LeaderboardEntry(
                id=r.id, profile_id=r.profile_id, pack_id=r.pack_id,
                runtime_name=r.runtime_name, model_name=r.model_name,
                timestamp=r.timestamp, metrics=r.document.get("metrics", {}),
            )
            for r in rows
        ],
    )
