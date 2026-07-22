from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Result
from ..schemas import HardwareListResponse, HardwareRecord

router = APIRouter(prefix="/hardware", tags=["hardware"])


@router.get("", response_model=HardwareListResponse)
def list_hardware(db: Session = Depends(get_db)) -> HardwareListResponse:
    """Hardware records, derived from ingested results' environment
    fingerprints — distinct (CPU model, OS, architecture) combos and how
    many results exist for each. Not a separately curated table (M3 thin
    slice); dedicated hardware records are a later milestone concern."""
    cpu_model = Result.document["environment"]["cpu"]["model"].astext
    os_system = Result.document["environment"]["os"]["system"].astext
    os_machine = Result.document["environment"]["os"]["machine"].astext

    stmt = (
        select(cpu_model, os_system, os_machine, func.count().label("result_count"))
        .group_by(cpu_model, os_system, os_machine)
        .order_by(func.count().desc())
    )
    rows = db.execute(stmt).all()
    return HardwareListResponse(hardware=[
        HardwareRecord(cpu_model=r[0], os_system=r[1], os_machine=r[2], result_count=r[3])
        for r in rows
    ])
