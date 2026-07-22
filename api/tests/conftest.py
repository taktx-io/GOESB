"""Shared fixtures: a real local Postgres (docker-compose / CI service),
migrated once per test session, truncated between tests for isolation. No
SQLite stand-in — see db.py's own docstring for why."""
import base64
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from oesb_runner.hashing import canonical_asset_sha256
from oesb_runner.schema_validation import validate_against
from oesb_runner.signing import (
    generate_ephemeral_keypair,
    public_key_bytes_for,
    sign_with_key,
    verify_result_document,
)
from sqlalchemy import text

from oesb_api.assets import _repo_root
from oesb_api.db import engine
from oesb_api.main import app

API_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _migrate_db():
    subprocess.run(["alembic", "upgrade", "head"], cwd=str(API_ROOT), check=True)


@pytest.fixture(autouse=True)
def _clean_tables():
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE results, runner_tokens"))
    yield


@pytest.fixture
def client():
    return TestClient(app)


def issue_test_token(client: TestClient) -> tuple[str, Any]:
    """Real call-home flow (ADR-0005): generate an ephemeral keypair, ask
    the (test) API to vouch for its public key. Returns (token_id,
    private_key) — the caller signs with private_key, same as `oesb submit`
    does for real."""
    private_key = generate_ephemeral_keypair()
    public_key_b64 = base64.b64encode(public_key_bytes_for(private_key)).decode("ascii")
    r = client.post("/runner-tokens", json={"public_key": public_key_b64})
    assert r.status_code == 201, r.text
    return r.json()["token_id"], private_key


def make_signed_result(
    client: TestClient,
    profile_id: str = "whisper-medium-en-batch",
    pack_id: str = "example-librispeech-en-batch",
) -> dict[str, Any]:
    """A real, genuinely signed result document — built from the actual
    committed profile/pack (so hashes match what ingest re-derives), signed
    via a real call-home token from the (test) API (ADR-0005), just without
    running an actual ASR model. Exercises the real trust-gate code path
    end-to-end; it isn't a mock.
    """
    root = _repo_root()
    profile = yaml.safe_load((root / "profiles" / profile_id / "profile.yaml").read_text())
    pack = yaml.safe_load((root / "packs" / pack_id / "pack.yaml").read_text())

    result = {
        "schema_version": "0.1",
        "profile": {
            "id": profile["id"],
            "version": profile["version"],
            "sha256": canonical_asset_sha256(profile, exclude=()),
        },
        "pack": {
            "id": pack["id"],
            "version": pack["version"],
            "sha256": pack["sha256"],
            "visibility": pack["visibility"],
        },
        "runtime": {"name": profile["runtime"]["name"], "version": "test", "sha256": "0" * 64},
        "model": {"name": profile["model"]["name"], "quantization": "int8", "sha256": "0" * 64},
        "config_sha256": "0" * 64,
        "environment": {"schema_version": "0.2", "os": {"system": "Linux"}},
        "metrics": {"wer": {"value": 0.1, "unit": "ratio"}},
        "repeats": 1,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runner": {"version": "test"},
    }
    payload_sha256 = canonical_asset_sha256(result, exclude=())
    result["payload_sha256"] = payload_sha256

    token_id, private_key = issue_test_token(client)
    result["signature"] = sign_with_key(payload_sha256, private_key, token_id)

    assert validate_against(result, "benchmark-result.schema.json") == []
    assert verify_result_document(
        result, public_key_bytes=public_key_bytes_for(private_key)
    ) is True
    return result
