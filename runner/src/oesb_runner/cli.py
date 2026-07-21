"""OESB runner command-line interface (scaffold).

This is an intentional stub. See docs/02-architecture.md and the roadmap for
the planned implementation across iterations M1-M3.
"""
from __future__ import annotations

import json

import typer

from . import __version__
from .environment import capture_environment

app = typer.Typer(help="Open Edge Speech Benchmark runner")


@app.command()
def version() -> None:
    """Print the runner version."""
    typer.echo(f"oesb-runner {__version__}")


@app.command()
def env() -> None:
    """Capture and print the reproducibility environment fingerprint."""
    typer.echo(json.dumps(capture_environment(), indent=2))


@app.command()
def validate(path: str) -> None:
    """Validate a profile or pack against its JSON Schema (stub)."""
    typer.echo(f"[stub] would validate: {path}")


@app.command()
def run(profile: str, pack: str) -> None:
    """Run a benchmark for a profile + pack (stub)."""
    typer.echo(f"[stub] would run profile={profile} pack={pack}")


if __name__ == "__main__":
    app()
