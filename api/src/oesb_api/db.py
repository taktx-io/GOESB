"""Database engine/session (M3, docs/03-roadmap.md).

Real Postgres only — no SQLite fallback, since the `results` table relies on
`JSONB` (see models.py), and hiding Postgres-specific behavior behind a
different dialect in tests would defeat the point of testing against it.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://oesb:oesb@localhost:5432/oesb"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


engine = create_engine(get_database_url())
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
