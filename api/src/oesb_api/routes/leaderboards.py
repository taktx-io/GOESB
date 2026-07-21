from fastapi import APIRouter, Query

router = APIRouter(prefix="/leaderboards", tags=["leaderboards"])


@router.get("")
def list_leaderboards(
    benchmark_type: str | None = Query(None),
    profile: str | None = Query(None),
    language: str | None = Query(None),
    runtime: str | None = Query(None),
    model: str | None = Query(None),
    hardware: str | None = Query(None),
    max_price_eur: float | None = Query(None),
):
    """Filterable leaderboard query (stub)."""
    return {"filters": {"benchmark_type": benchmark_type, "language": language},
            "results": []}
