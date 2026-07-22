"""ADR-0005: call-home ephemeral signing tokens."""
import base64

from oesb_runner.signing import generate_ephemeral_keypair, public_key_bytes_for

from oesb_api.tokens import RATE_LIMIT_MAX_PER_IP


def _public_key_b64() -> str:
    return base64.b64encode(public_key_bytes_for(generate_ephemeral_keypair())).decode("ascii")


def test_issue_token_returns_id_and_expiry(client):
    r = client.post("/runner-tokens", json={"public_key": _public_key_b64()})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["token_id"]
    assert body["expires_at"]


def test_invalid_public_key_rejected(client):
    r = client.post("/runner-tokens", json={"public_key": "not-valid-base64!!"})
    assert r.status_code == 422

    # valid base64, wrong length for an ed25519 key
    r2 = client.post("/runner-tokens", json={"public_key": base64.b64encode(b"too-short").decode()})
    assert r2.status_code == 422


def test_rate_limit_per_ip(client):
    for _ in range(RATE_LIMIT_MAX_PER_IP):
        r = client.post("/runner-tokens", json={"public_key": _public_key_b64()})
        assert r.status_code == 201

    r = client.post("/runner-tokens", json={"public_key": _public_key_b64()})
    assert r.status_code == 429
    assert r.json()["detail"]["reason"] == "rate_limited"
