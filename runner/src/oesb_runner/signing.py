"""Result-document signing (FR-8.3, FR-9.3, ADR-0004).

The signing key is **runner-embedded, not per-user identity** (see ADR-0004):
it attests "a genuine, unmodified official runner produced this," not "this
person submitted it." For M1 this is a local keypair generated on first use;
full signing-key distribution/governance for released runner builds is a
later cross-cutting track, not a blocker here.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .hashing import canonical_asset_sha256

ALGO = "ed25519"
DEFAULT_KEY_DIR = Path.home() / ".oesb" / "keys"
DEFAULT_KEY_ID = "oesb-runner-dev-key-1"
# These two fields didn't exist yet when payload_sha256 was first computed
# (they're added to the document afterward), so re-verification must exclude
# them before recomputing the hash it was actually signed over.
_RESULT_HASH_EXCLUDED_FIELDS = ("payload_sha256", "signature")


def _key_paths(key_dir: Path) -> tuple[Path, Path]:
    return key_dir / "runner_ed25519.key", key_dir / "runner_ed25519.pub"


def load_or_create_keypair(key_dir: Path = DEFAULT_KEY_DIR) -> Ed25519PrivateKey:
    """Load the runner's signing key, generating one on first use."""
    key_dir.mkdir(parents=True, exist_ok=True)
    priv_path, pub_path = _key_paths(key_dir)
    if priv_path.exists():
        return Ed25519PrivateKey.from_private_bytes(priv_path.read_bytes())

    private_key = Ed25519PrivateKey.generate()
    priv_path.write_bytes(private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    priv_path.chmod(0o600)
    pub_path.write_bytes(private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ))
    return private_key


def sign_payload_sha256(
    payload_sha256_hex: str,
    *,
    key_dir: Path = DEFAULT_KEY_DIR,
    key_id: str = DEFAULT_KEY_ID,
) -> dict[str, str]:
    """Sign a result document's payload_sha256; returns the `signature` block."""
    private_key = load_or_create_keypair(key_dir)
    signature = private_key.sign(bytes.fromhex(payload_sha256_hex))
    return {
        "algo": ALGO,
        "key_id": key_id,
        "sig": base64.b64encode(signature).decode("ascii"),
    }


def verify_signature(
    payload_sha256_hex: str,
    signature: dict[str, str],
    *,
    key_dir: Path = DEFAULT_KEY_DIR,
) -> bool:
    """Re-verify a result's signature against payload_sha256 (the ingest gate,
    ADR-0004). Returns False (never raises) on any mismatch or missing key."""
    if signature.get("algo") != ALGO:
        return False
    _, pub_path = _key_paths(key_dir)
    if not pub_path.exists():
        return False
    public_key = Ed25519PublicKey.from_public_bytes(pub_path.read_bytes())
    try:
        public_key.verify(
            base64.b64decode(signature["sig"]),
            bytes.fromhex(payload_sha256_hex),
        )
        return True
    except InvalidSignature:
        return False


def verify_result_document(result: dict[str, Any], *, key_dir: Path = DEFAULT_KEY_DIR) -> bool:
    """The actual ingest-time trust gate (ADR-0004): recompute payload_sha256
    from the document's own content and check both that it matches the
    document's declared `payload_sha256` and that the signature verifies
    against it. Never raises; a malformed document simply fails to verify."""
    try:
        recomputed = canonical_asset_sha256(result, exclude=_RESULT_HASH_EXCLUDED_FIELDS)
        if recomputed != result.get("payload_sha256"):
            return False
        return verify_signature(recomputed, result["signature"], key_dir=key_dir)
    except (KeyError, TypeError):
        return False
