"""M3 exit criterion (docs/03-roadmap.md): a locally produced result is
submitted, verified, and retrievable via the API; tampered results are
rejected. Also ADR-0005: call-home signing tokens are single-use and
expiring, not just "any signature verifies"."""
import copy
from datetime import datetime, timedelta, timezone

from conftest import issue_test_token, make_signed_result
from oesb_runner.hashing import canonical_asset_sha256
from oesb_runner.signing import sign_with_key

from oesb_api.db import SessionLocal
from oesb_api.models import RunnerToken


def test_submit_and_retrieve_real_signed_result(client):
    result = make_signed_result(client)
    r = client.post("/benchmark", json=result)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["id"] == result["payload_sha256"]

    r2 = client.get(f"/benchmark/{body['id']}")
    assert r2.status_code == 200
    assert r2.json() == result


def test_resubmitting_identical_result_is_idempotent(client):
    result = make_signed_result(client)
    r1 = client.post("/benchmark", json=result)
    r2 = client.post("/benchmark", json=result)
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"]


def test_reusing_a_token_for_a_different_result_is_rejected(client):
    token_id, private_key = issue_test_token(client)

    first = make_signed_result(client)
    first["signature"] = sign_with_key(first["payload_sha256"], private_key, token_id)
    r1 = client.post("/benchmark", json=first)
    assert r1.status_code == 201

    # a genuinely different payload (different repeats -> different hash),
    # signed with the SAME (now-used) token
    second = make_signed_result(client)
    second["repeats"] = 2
    second["payload_sha256"] = canonical_asset_sha256(second, exclude=("payload_sha256", "signature"))
    second["signature"] = sign_with_key(second["payload_sha256"], private_key, token_id)

    r2 = client.post("/benchmark", json=second)
    assert r2.status_code == 400
    assert r2.json()["detail"]["reason"] == "signing_token_already_used"


def test_expired_token_is_rejected(client):
    result = make_signed_result(client)
    token_id = result["signature"]["key_id"]

    with SessionLocal() as db:
        token = db.get(RunnerToken, token_id)
        token.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()

    r = client.post("/benchmark", json=result)
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "signing_token_expired"


def test_unknown_signing_token_is_rejected(client):
    result = make_signed_result(client)
    result["signature"]["key_id"] = "no-such-token"

    r = client.post("/benchmark", json=result)
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "unknown_signing_token"


def test_tampered_metric_is_rejected_and_not_stored(client):
    result = make_signed_result(client)
    tampered = copy.deepcopy(result)
    tampered["metrics"]["wer"]["value"] = 0.0  # mutate after signing

    r = client.post("/benchmark", json=tampered)
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "hash_or_signature_invalid"

    # never stored under the (now-mismatched) claimed id
    r2 = client.get(f"/benchmark/{tampered['payload_sha256']}")
    assert r2.status_code == 404


def test_tampered_signature_is_rejected(client):
    result = make_signed_result(client)
    tampered = copy.deepcopy(result)
    tampered["signature"]["sig"] = "AAAA"

    r = client.post("/benchmark", json=tampered)
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "hash_or_signature_invalid"


def test_schema_invalid_submission_is_rejected(client):
    r = client.post("/benchmark", json={"not": "a valid result"})
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "schema_invalid"


def test_unknown_profile_is_rejected(client):
    result = make_signed_result(client)
    result["profile"]["id"] = "no-such-profile"
    # re-sign (fresh token) over the mutated content so this fails on
    # profile lookup, not on the signature check that would otherwise fire first
    del result["payload_sha256"]
    del result["signature"]
    result["payload_sha256"] = canonical_asset_sha256(result, exclude=())
    token_id, private_key = issue_test_token(client)
    result["signature"] = sign_with_key(result["payload_sha256"], private_key, token_id)

    r = client.post("/benchmark", json=result)
    assert r.status_code == 403
    assert r.json()["detail"]["reason"] == "not_an_official_profile"


def test_non_open_pack_is_rejected(client):
    result = make_signed_result(client)
    result["pack"]["visibility"] = "private"
    del result["payload_sha256"]
    del result["signature"]
    result["payload_sha256"] = canonical_asset_sha256(result, exclude=())
    token_id, private_key = issue_test_token(client)
    result["signature"] = sign_with_key(result["payload_sha256"], private_key, token_id)

    r = client.post("/benchmark", json=result)
    assert r.status_code == 403
    assert r.json()["detail"]["reason"] == "pack_not_open"


def test_get_unknown_benchmark_404s(client):
    r = client.get("/benchmark/does-not-exist")
    assert r.status_code == 404
