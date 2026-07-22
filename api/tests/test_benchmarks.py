"""M3 exit criterion (docs/03-roadmap.md): a locally produced result is
submitted, verified, and retrievable via the API; tampered results are
rejected."""
import copy

from conftest import make_signed_result


def test_submit_and_retrieve_real_signed_result(client):
    result = make_signed_result()
    r = client.post("/benchmark", json=result)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["id"] == result["payload_sha256"]

    r2 = client.get(f"/benchmark/{body['id']}")
    assert r2.status_code == 200
    assert r2.json() == result


def test_resubmitting_identical_result_is_idempotent(client):
    result = make_signed_result()
    r1 = client.post("/benchmark", json=result)
    r2 = client.post("/benchmark", json=result)
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"]


def test_tampered_metric_is_rejected_and_not_stored(client):
    result = make_signed_result()
    tampered = copy.deepcopy(result)
    tampered["metrics"]["wer"]["value"] = 0.0  # mutate after signing

    r = client.post("/benchmark", json=tampered)
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "hash_or_signature_invalid"

    # never stored under the (now-mismatched) claimed id
    r2 = client.get(f"/benchmark/{tampered['payload_sha256']}")
    assert r2.status_code == 404


def test_tampered_signature_is_rejected(client):
    result = make_signed_result()
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
    result = make_signed_result()
    result["profile"]["id"] = "no-such-profile"
    # re-sign over the mutated content so this fails on profile lookup,
    # not on the signature check that would otherwise fire first
    from oesb_runner.hashing import canonical_asset_sha256
    from oesb_runner.signing import sign_payload_sha256

    del result["payload_sha256"]
    del result["signature"]
    payload_sha256 = canonical_asset_sha256(result, exclude=())
    result["payload_sha256"] = payload_sha256
    result["signature"] = sign_payload_sha256(payload_sha256)

    r = client.post("/benchmark", json=result)
    assert r.status_code == 403
    assert r.json()["detail"]["reason"] == "not_an_official_profile"


def test_non_open_pack_is_rejected(client):
    result = make_signed_result()
    result["pack"]["visibility"] = "private"
    from oesb_runner.hashing import canonical_asset_sha256
    from oesb_runner.signing import sign_payload_sha256

    del result["payload_sha256"]
    del result["signature"]
    payload_sha256 = canonical_asset_sha256(result, exclude=())
    result["payload_sha256"] = payload_sha256
    result["signature"] = sign_payload_sha256(payload_sha256)

    r = client.post("/benchmark", json=result)
    assert r.status_code == 403
    assert r.json()["detail"]["reason"] == "pack_not_open"


def test_get_unknown_benchmark_404s(client):
    r = client.get("/benchmark/does-not-exist")
    assert r.status_code == 404
