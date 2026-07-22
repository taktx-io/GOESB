"""SQLAlchemy models (M3, docs/03-roadmap.md).

One table for this milestone: `results`. The full, already-verified result
document is stored as-is in `document` (content-addressed by its own
`payload_sha256`, which is also this row's primary key — see ingest.py) so
nothing is lost or reshaped; the remaining columns are just an index over
the handful of fields the M3 endpoints actually filter/derive from
(`GET /leaderboards`, `GET /hardware`) — not a full relational decomposition
of the result schema.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Result(Base):
    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # = payload_sha256
    document: Mapped[dict] = mapped_column(JSONB, nullable=False)

    profile_id: Mapped[str] = mapped_column(String, index=True)
    profile_version: Mapped[str] = mapped_column(String)
    pack_id: Mapped[str] = mapped_column(String, index=True)
    runtime_name: Mapped[str] = mapped_column(String, index=True)
    model_name: Mapped[str] = mapped_column(String, index=True)
    benchmark_type: Mapped[str] = mapped_column(String, index=True)
    language: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    timestamp: Mapped[str] = mapped_column(String)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
