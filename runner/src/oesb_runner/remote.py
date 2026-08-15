"""Fetch-on-demand for official profiles/packs, with a local disk cache — so
`goesb run` works against an already-published profile/pack with zero local
GOESB checkout.

Unlike model weights (content-addressed, genuinely immutable once fetched)
or audio (same), a profile/pack's own YAML is small, server-mutable
metadata — this repo's own history has renamed pack ids, added a gated
pack's credential requirement, and bumped versions, all to already-fetched
packs. An earlier version of this module cached purely on file existence,
never revalidating: whatever a user's very first fetch happened to see was
served forever after, however out of date the server-side document later
became — confirmed live: a pack that gained a required API key after a
user's first (uncredentialed) fetch kept silently serving the old,
credential-less copy, so the wizard's own credential preflight never even
saw the requirement to prompt for. Network is now always tried first; the
cache is a fallback for when the network genuinely isn't reachable, not a
substitute for asking again.

Packs are the partial case: GOESB never hosts audio (privacy-first), so
fetching a pack only ever gets you its metadata (pack.yaml) + transcript
index (manifest.jsonl) — the actual audio still needs its own fetch step,
per the pack's own `audio.source.fetch_instructions`, same as it always has.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

# Override for pointing every subcommand -- including the interactive wizard,
# which has no --api-url flag of its own -- at a non-production API (e.g.
# test.goesb.com) without editing code. Read once at import time, which is
# correct here: `export GOESB_API_URL=... && goesb ...` always sets it in the
# shell before the `goesb` process (and this import) ever starts.
DEFAULT_API_URL = os.environ.get("GOESB_API_URL", "https://www.goesb.com/api")

# manifest.jsonl isn't part of a pack's own document, so it isn't served by
# the platform API — the public GOESB repo is the source for it regardless
# of which platform API a pack's pack.yaml came from.
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/taktx-io/GOESB/main"

CACHE_ROOT = Path.home() / ".goesb" / "cache"


def _fetch_json(url: str, timeout: int = 15) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310 - caller-controlled --api-url
        return json.loads(resp.read())


def _fetch_text(url: str, timeout: int = 15) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310 - fixed public GitHub URL
        return resp.read().decode("utf-8")


def fetch_profile(profile_id: str, api_url: str) -> dict[str, Any]:
    """Fetch an official profile from the platform API and cache it locally
    — a profile is pure configuration, so this alone is everything `run`
    needs for it, no separate audio/manifest step. Network first, always:
    the cache is only a fallback for when the network request itself fails
    (see module docstring for why re-serving a possibly-stale cache by
    default was a real bug, not a feature)."""
    cache_path = CACHE_ROOT / "profiles" / f"{profile_id}.json"
    try:
        data = _fetch_json(f"{api_url.rstrip('/')}/profiles/{profile_id}")
    except (urllib.error.URLError, urllib.error.HTTPError):
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_pack(pack_id: str, api_url: str) -> Path:
    """Fetch an official pack's pack.yaml + manifest.jsonl into the local
    cache and return that directory, shaped exactly like a local
    --packs-dir/<pack_id> would be. Never fetches audio — the caller still
    needs to populate <returned dir>/audio per pack.yaml's own
    audio.source.fetch_instructions before the pack is actually runnable.
    Network first, always — see fetch_profile's docstring; both fetches
    (API + GitHub-hosted manifest) must succeed to count as fresh, so a
    half-updated cache is never written."""
    cache_dir = CACHE_ROOT / "packs" / pack_id
    pack_yaml_path = cache_dir / "pack.yaml"
    manifest_path = cache_dir / "manifest.jsonl"
    try:
        pack_data = _fetch_json(f"{api_url.rstrip('/')}/packs/{pack_id}")
        manifest_text = _fetch_text(f"{GITHUB_RAW_BASE}/packs/{pack_id}/manifest.jsonl")
    except (urllib.error.URLError, urllib.error.HTTPError):
        if pack_yaml_path.exists() and manifest_path.exists():
            return cache_dir
        raise

    cache_dir.mkdir(parents=True, exist_ok=True)
    # newline="" + explicit utf-8, not bare write_text: a pack's declared
    # manifest_sha256 is over the *bytes* GitHub serves (LF-terminated,
    # UTF-8). Python text mode defaults to newline=None, which translates
    # every "\n" to "\r\n" on Windows, and to the locale encoding, which is
    # cp1252 there — so on Windows a bare write_text stored a manifest whose
    # bytes could never hash to the declared value, failing every remotely
    # fetched pack with "manifest.jsonl hash mismatch", and outright
    # crashing on the non-ASCII references in the nl/fr/de/es/pt packs.
    pack_yaml_path.write_text(
        yaml.safe_dump(pack_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="",
    )
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="")

    return cache_dir
