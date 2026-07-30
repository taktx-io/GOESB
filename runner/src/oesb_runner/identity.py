"""Local pseudonymous identity for crediting `goesb submit` results
(~/.goesb/identity.json) -- sibling to credentials.py/signing.py under the
same ~/.goesb/ root, same chmod 0600 discipline.

The secret passphrase a user enters is used ONCE, in memory, to derive a
`discriminator` and is never written to disk or sent over the network --
only the (callsign, discriminator) pair persists and is submitted. This
lets two different people who pick the same callsign (e.g. "anon") show
up as visibly distinct entries on the public leaderboard, without goesb
ever needing real accounts, emails, or a server-side identity system.

discriminator = PBKDF2-HMAC-SHA256(secret, salt=callsign, 200_000 iters)[:4].hex()
Using the callsign itself as the KDF salt (rather than a random salt,
which would need to be stored) means the same secret produces an
UNRELATED discriminator under a different callsign -- so a leaked/guessed
secret for one callsign doesn't let an attacker correlate or impersonate
across callsigns. PBKDF2 (not plain SHA-256) because the discriminator is
public -- submitted with every result -- so a slow KDF raises the cost of
brute-forcing a weak secret against a known callsign to claim someone
else's identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IDENTITY_PATH = Path.home() / ".goesb" / "identity.json"
_PBKDF2_ITERATIONS = 200_000
_DISCRIMINATOR_BYTES = 4


@dataclass(frozen=True)
class Identity:
    callsign: str
    discriminator: str


def compute_discriminator(callsign: str, secret: str) -> str:
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        callsign.encode("utf-8"),
        _PBKDF2_ITERATIONS,
        dklen=_DISCRIMINATOR_BYTES,
    )
    return derived.hex()


def load_identity(*, path: Path = DEFAULT_IDENTITY_PATH) -> Identity | None:
    """A missing or corrupt store degrades to None rather than raising --
    same philosophy as credentials.load_credential: local-state problems
    must never crash a submit."""
    if not path.exists():
        return None
    try:
        store = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(store, dict):
        return None
    callsign, discriminator = store.get("callsign"), store.get("discriminator")
    if not callsign or not discriminator:
        return None
    return Identity(callsign, discriminator)


def save_identity(identity: Identity, *, path: Path = DEFAULT_IDENTITY_PATH) -> None:
    """Overwrites whatever was saved -- there is only ever one local
    identity, not a store of many (contrast credentials.py, which is
    keyed by env_var since a user can have several gated-source creds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    store = {"callsign": identity.callsign, "discriminator": identity.discriminator}
    path.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def clear_identity(*, path: Path = DEFAULT_IDENTITY_PATH) -> None:
    path.unlink(missing_ok=True)
