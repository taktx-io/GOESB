from fastapi import APIRouter

router = APIRouter(prefix="/packs", tags=["packs"])


@router.get("")
def list_packs():
    """List benchmark packs with id, version, sha256 (stub)."""
    return {"packs": []}


@router.get("/{pack_id}")
def get_pack(pack_id: str):
    return {"id": pack_id}
