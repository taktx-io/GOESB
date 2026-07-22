from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..assets import Assets, get_assets
from ..schemas import PackListResponse, PackSummary

router = APIRouter(prefix="/packs", tags=["packs"])


@router.get("", response_model=PackListResponse)
def list_packs(assets: Assets = Depends(get_assets)) -> PackListResponse:
    """List benchmark packs with id, version, sha256."""
    return PackListResponse(packs=[
        PackSummary(id=p["id"], version=p["version"], sha256=p["sha256"], visibility=p["visibility"])
        for p in assets.packs.values()
    ])


@router.get("/{pack_id}")
def get_pack(pack_id: str, assets: Assets = Depends(get_assets)) -> dict[str, Any]:
    """Full pack document, as validated against benchmark-pack.schema.json."""
    pack = assets.packs.get(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail={"reason": "not_found"})
    return pack
