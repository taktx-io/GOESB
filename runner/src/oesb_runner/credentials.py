"""Local credential store for gated-pack sources (ADR-0010).

Sibling to `signing.DEFAULT_KEY_DIR` under `~/.goesb/` — same root, same
chmod 0600 discipline — so there's one obvious place on disk a
security-conscious user finds, audits, or deletes all of goesb's local
secrets. A credential here authenticates directly against a gated source's
own API (e.g. Mozilla Data Collective) from the user's own machine; it is
never sent to goesb's own servers and never captured in the environment
fingerprint (see `environment.capture_environment`, which doesn't touch
`os.environ` at all).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CREDENTIALS_PATH = Path.home() / ".goesb" / "credentials.json"


def load_credential(env_var: str, *, path: Path = DEFAULT_CREDENTIALS_PATH) -> str | None:
    """Process environment first — an explicitly-exported var always wins,
    so this never shadows a user's own shell config — then the local store.
    A missing or corrupt store degrades to None rather than raising, same
    philosophy as `cli._load_profile_for_wizard`: a local-state problem
    must never crash the wizard."""
    value = os.environ.get(env_var)
    if value:
        return value
    if not path.exists():
        return None
    try:
        store = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(store, dict):
        return None
    return store.get(env_var)


def save_credential(env_var: str, value: str, *, path: Path = DEFAULT_CREDENTIALS_PATH) -> None:
    """Read-modify-write the on-disk JSON store, then chmod 0600 — mirrors
    `signing.load_or_create_keypair`'s exact permission discipline."""
    store: dict[str, str] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                store = existing
        except (OSError, json.JSONDecodeError):
            store = {}

    store[env_var] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
