"""Pydantic response models (M3, FR-12.3: published, versioned OpenAPI).

Only list/summary endpoints get real typed models here — a profile/pack/
result's true shape is already defined once, as the source of truth, by
schemas/*.json (docs/02-architecture.md §2.5). Duplicating those as strict
Pydantic models would just be a second definition to drift out of sync;
detail endpoints (`GET /profiles/{id}`, `/packs/{id}`, `/benchmark/{id}`)
return the schema-validated document as-is instead.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ProfileSummary(BaseModel):
    id: str
    version: str
    title: str | None = None
    benchmark_type: str
    language: str | None = None


class ProfileListResponse(BaseModel):
    profiles: list[ProfileSummary]


class PackSummary(BaseModel):
    id: str
    version: str
    sha256: str
    visibility: str


class PackListResponse(BaseModel):
    packs: list[PackSummary]


class HardwareRecord(BaseModel):
    cpu_model: str | None
    os_system: str | None
    os_machine: str | None
    result_count: int


class HardwareListResponse(BaseModel):
    hardware: list[HardwareRecord]


class LeaderboardEntry(BaseModel):
    id: str
    profile_id: str
    pack_id: str
    runtime_name: str
    model_name: str
    timestamp: str
    metrics: dict[str, Any]


class LeaderboardResponse(BaseModel):
    filters: dict[str, Any]
    results: list[LeaderboardEntry]


class SubmitBenchmarkResponse(BaseModel):
    id: str
    accepted: bool
