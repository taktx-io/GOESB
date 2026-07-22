from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..assets import Assets, get_assets
from ..schemas import ProfileListResponse, ProfileSummary

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=ProfileListResponse)
def list_profiles(assets: Assets = Depends(get_assets)) -> ProfileListResponse:
    """List official benchmark profiles."""
    return ProfileListResponse(profiles=[
        ProfileSummary(
            id=p["id"], version=p["version"], title=p.get("title"),
            benchmark_type=p["benchmark_type"], language=p.get("language"),
        )
        for p in assets.profiles.values()
    ])


@router.get("/{profile_id}")
def get_profile(profile_id: str, assets: Assets = Depends(get_assets)) -> dict[str, Any]:
    """Get a profile by id, including version and changelog — the full
    document as validated against benchmark-profile.schema.json."""
    profile = assets.profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail={"reason": "not_found"})
    return profile
