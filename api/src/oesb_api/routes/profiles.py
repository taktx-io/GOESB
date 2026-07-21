from fastapi import APIRouter

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("")
def list_profiles():
    """List official benchmark profiles (stub)."""
    return {"profiles": []}


@router.get("/{profile_id}")
def get_profile(profile_id: str):
    """Get a profile by id, including version and changelog (stub)."""
    return {"id": profile_id}
