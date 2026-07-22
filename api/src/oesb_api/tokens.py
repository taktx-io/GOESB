"""Ephemeral signing-token issuance (ADR-0005).

The runner generates a keypair locally and asks us to vouch for its public
key for exactly one submission, rather than shipping a static secret in the
pip-distributed package — see the ADR for why embedding a secret in an
open-source Python package can't work. The private key never leaves the
runner's machine; we only ever see the public half.
"""
from __future__ import annotations

import base64
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import RunnerToken

TOKEN_TTL = timedelta(hours=24)
RATE_LIMIT_WINDOW = timedelta(hours=1)
RATE_LIMIT_MAX_PER_IP = 20
_ED25519_PUBLIC_KEY_LEN = 32


def issue_token(public_key_b64: str, requester_ip: str | None, db: Session) -> RunnerToken:
    if requester_ip is not None:
        window_start = datetime.now(timezone.utc) - RATE_LIMIT_WINDOW
        recent_count = db.execute(
            select(func.count()).select_from(RunnerToken)
            .where(RunnerToken.requester_ip == requester_ip, RunnerToken.issued_at >= window_start)
        ).scalar_one()
        if recent_count >= RATE_LIMIT_MAX_PER_IP:
            raise HTTPException(status_code=429, detail={"reason": "rate_limited"})

    try:
        public_key = base64.b64decode(public_key_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail={"reason": "invalid_public_key"}) from exc
    if len(public_key) != _ED25519_PUBLIC_KEY_LEN:
        raise HTTPException(status_code=422, detail={"reason": "invalid_public_key"})

    now = datetime.now(timezone.utc)
    token = RunnerToken(
        id=secrets.token_urlsafe(24),
        public_key=public_key,
        issued_at=now,
        expires_at=now + TOKEN_TTL,
        requester_ip=requester_ip,
    )
    db.add(token)
    db.commit()
    db.refresh(token)
    return token
