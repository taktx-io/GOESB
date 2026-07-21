"""OESB API application (scaffold).

A thin, documented FastAPI app exposing the public read endpoints and the
submission endpoint. Business logic is intentionally stubbed; see the roadmap.
"""
from __future__ import annotations

from fastapi import FastAPI

from . import __version__
from .routes import benchmarks, leaderboards, packs, profiles

app = FastAPI(
    title="Open Edge Speech Benchmark API",
    version=__version__,
    description="Objective, reproducible benchmarks for local (edge) speech AI.",
)

app.include_router(leaderboards.router)
app.include_router(profiles.router)
app.include_router(packs.router)
app.include_router(benchmarks.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
