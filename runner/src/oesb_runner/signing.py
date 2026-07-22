"""Result-document signing (FR-8.3, FR-9.3, ADR-0004, ADR-0005).

A signature proves the document hasn't been altered since it was signed
(tamper-evidence) and, combined with ADR-0005's call-home tokens, which
server-issued credential authorized the signing — **not** "a genuine
official runner produced this measurement." ADR-0004's original framing
overclaimed that; ADR-0005 corrects it and explains why no signature scheme
over an open-source, pip-installed client can prove more than this.

Two signing paths:
- `sign_payload_sha256` — the persistent **local** identity
  (`load_or_create_keypair`, generated on first use, stored under
  `key_dir`). Fine for local/private results that are never submitted
  publicly; producing a result never requires network access.
- `sign_with_key` — sign with an explicit, caller-supplied key. Used for
  ADR-0005's ephemeral call-home tokens: the runner generates a fresh
  in-memory-only keypair per submission (`generate_ephemeral_keypair`),
  asks the API to vouch for its public key, and signs with that instead —
  the private key never touches disk or leaves the machine that made it.

Verification (`verify_signature`/`verify_result_document`) takes an optional
`public_key_bytes` override so callers with their own key source (the API,
resolving a call-home token from its database) don't need this module to
know anything about where that key came from; omitting it falls back to the
local `key_dir` file, unchanged from the original M1 behavior.
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
    """Load the runner's persistent local signing key, generating one on
    first use. This identity is for locally-signed, non-submitted results —
    ADR-0005's call-home tokens are the path for public submission."""
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


def generate_ephemeral_keypair() -> Ed25519PrivateKey:
    """A fresh, in-memory-only keypair for one-off call-home signing
    (ADR-0005) — never written to disk, unlike `load_or_create_keypair`'s
    persistent local identity. The caller sends only the public half to the
    API when requesting a token; the private key stays local."""
    return Ed25519PrivateKey.generate()


def public_key_bytes_for(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def sign_with_key(payload_sha256_hex: str, private_key: Ed25519PrivateKey, key_id: str) -> dict[str, str]:
    """Sign with an explicit, caller-supplied key rather than the persistent
    local identity — e.g. an ADR-0005 ephemeral call-home key, where
    `key_id` is the server-issued token id."""
    signature = private_key.sign(bytes.fromhex(payload_sha256_hex))
    return {
        "algo": ALGO,
        "key_id": key_id,
        "sig": base64.b64encode(signature).decode("ascii"),
    }


def sign_payload_sha256(
    payload_sha256_hex: str,
    *,
    key_dir: Path = DEFAULT_KEY_DIR,
    key_id: str = DEFAULT_KEY_ID,
) -> dict[str, str]:
    """Sign a result document's payload_sha256 with the persistent local
    identity; returns the `signature` block."""
    private_key = load_or_create_keypair(key_dir)
    return sign_with_key(payload_sha256_hex, private_key, key_id)


def verify_signature(
    payload_sha256_hex: str,
    signature: dict[str, str],
    *,
    public_key_bytes: bytes | None = None,
    key_dir: Path = DEFAULT_KEY_DIR,
) -> bool:
    """Re-verify a result's signature against payload_sha256. Returns False
    (never raises) on any mismatch or missing key.

    `public_key_bytes`, when given, is used as-is (e.g. the API resolving an
    ADR-0005 call-home token from its database) — the caller owns deciding
    whether that key is trusted; this function only checks the cryptography.
    Omitted, this falls back to the local `key_dir` file (the original M1
    behavior, still used for the CLI's own post-sign self-verification).
    """
    if signature.get("algo") != ALGO:
        return False
    if public_key_bytes is None:
        _, pub_path = _key_paths(key_dir)
        if not pub_path.exists():
            return False
        public_key_bytes = pub_path.read_bytes()
    public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        public_key.verify(
            base64.b64decode(signature["sig"]),
            bytes.fromhex(payload_sha256_hex),
        )
        return True
    except InvalidSignature:
        return False


def verify_result_document(
    result: dict[str, Any],
    *,
    public_key_bytes: bytes | None = None,
    key_dir: Path = DEFAULT_KEY_DIR,
) -> bool:
    """The tamper-evidence check (ADR-0004/ADR-0005): recompute payload_sha256
    from the document's own content and check both that it matches the
    document's declared `payload_sha256` and that the signature verifies
    against it. Never raises; a malformed document simply fails to verify.
    See `verify_signature` for what `public_key_bytes` means."""
    try:
        recomputed = canonical_asset_sha256(result, exclude=_RESULT_HASH_EXCLUDED_FIELDS)
        if recomputed != result.get("payload_sha256"):
            return False
        return verify_signature(
            recomputed, result["signature"], public_key_bytes=public_key_bytes, key_dir=key_dir
        )
    except (KeyError, TypeError):
        return False
