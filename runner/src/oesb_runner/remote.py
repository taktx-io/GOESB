"""Fetch-on-demand for official profiles/packs, with a local disk cache — so
`goesb run` works against an already-published profile/pack with zero local
GOESB checkout. Mirrors how model weights already work in every adapter:
fetch once, cache, fully offline after.

Packs are the partial case: GOESB never hosts audio (privacy-first), so
fetching a pack only ever gets you its metadata (pack.yaml) + transcript
index (manifest.jsonl) — the actual audio still needs its own fetch step,
per the pack's own `audio.source.fetch_instructions`, same as it always has.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml

DEFAULT_API_URL = "https://www.goesb.com/api"

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
    needs for it, no separate audio/manifest step."""
    cache_path = CACHE_ROOT / "profiles" / f"{profile_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    data = _fetch_json(f"{api_url.rstrip('/')}/profiles/{profile_id}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


def fetch_pack(pack_id: str, api_url: str) -> Path:
    """Fetch an official pack's pack.yaml + manifest.jsonl into the local
    cache and return that directory, shaped exactly like a local
    --packs-dir/<pack_id> would be. Never fetches audio — the caller still
    needs to populate <returned dir>/audio per pack.yaml's own
    audio.source.fetch_instructions before the pack is actually runnable."""
    cache_dir = CACHE_ROOT / "packs" / pack_id
    pack_yaml_path = cache_dir / "pack.yaml"
    manifest_path = cache_dir / "manifest.jsonl"
    if pack_yaml_path.exists() and manifest_path.exists():
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    pack_data = _fetch_json(f"{api_url.rstrip('/')}/packs/{pack_id}")
    pack_yaml_path.write_text(yaml.safe_dump(pack_data, sort_keys=False, allow_unicode=True))

    manifest_text = _fetch_text(f"{GITHUB_RAW_BASE}/packs/{pack_id}/manifest.jsonl")
    manifest_path.write_text(manifest_text)

    return cache_dir
