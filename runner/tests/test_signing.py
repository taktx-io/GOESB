import hashlib

from oesb_runner import signing


def _payload_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_sign_and_verify_roundtrip(tmp_path):
    key_dir = tmp_path / "keys"
    payload = _payload_sha256("some result document")
    signature = signing.sign_payload_sha256(payload, key_dir=key_dir)

    assert signature["algo"] == "ed25519"
    assert signing.verify_signature(payload, signature, key_dir=key_dir) is True


def test_verify_rejects_tampered_payload(tmp_path):
    key_dir = tmp_path / "keys"
    payload = _payload_sha256("original")
    signature = signing.sign_payload_sha256(payload, key_dir=key_dir)

    tampered_payload = _payload_sha256("tampered")
    assert signing.verify_signature(tampered_payload, signature, key_dir=key_dir) is False


def test_verify_rejects_tampered_signature(tmp_path):
    key_dir = tmp_path / "keys"
    payload = _payload_sha256("original")
    signature = signing.sign_payload_sha256(payload, key_dir=key_dir)
    signature["sig"] = signature["sig"][:-4] + "abcd"

    assert signing.verify_signature(payload, signature, key_dir=key_dir) is False


def test_verify_without_known_key_fails_closed(tmp_path):
    payload = _payload_sha256("original")
    signature = {"algo": "ed25519", "key_id": "unknown", "sig": "not-a-real-signature"}
    assert signing.verify_signature(payload, signature, key_dir=tmp_path / "no-keys-here") is False


def test_load_or_create_keypair_is_persistent(tmp_path):
    key_dir = tmp_path / "keys"
    key1 = signing.load_or_create_keypair(key_dir)
    key2 = signing.load_or_create_keypair(key_dir)
    assert key1.private_bytes_raw() == key2.private_bytes_raw()


def _make_signed_document(key_dir) -> dict:
    from oesb_runner.hashing import canonical_asset_sha256

    doc = {"metrics": {"wer": {"value": 0.1}}, "profile": {"id": "x"}}
    payload = canonical_asset_sha256(doc, exclude=("payload_sha256", "signature"))
    doc["payload_sha256"] = payload
    doc["signature"] = signing.sign_payload_sha256(payload, key_dir=key_dir)
    return doc


def test_verify_result_document_roundtrip(tmp_path):
    key_dir = tmp_path / "keys"
    doc = _make_signed_document(key_dir)
    assert signing.verify_result_document(doc, key_dir=key_dir) is True


def test_verify_result_document_detects_tampering_after_signing(tmp_path):
    key_dir = tmp_path / "keys"
    doc = _make_signed_document(key_dir)
    doc["metrics"]["wer"]["value"] = 0.0  # tamper after signing
    assert signing.verify_result_document(doc, key_dir=key_dir) is False


def test_verify_result_document_rejects_malformed_input(tmp_path):
    key_dir = tmp_path / "keys"
    assert signing.verify_result_document({"nonsense": True}, key_dir=key_dir) is False
