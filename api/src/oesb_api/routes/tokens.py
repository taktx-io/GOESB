from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from ..db import get_db
from ..schemas import TokenRequest, TokenResponse
from ..tokens import issue_token

router = APIRouter(prefix="/runner-tokens", tags=["tokens"])


@router.post("", response_model=TokenResponse, status_code=201)
def request_token(
    body: TokenRequest, request: Request, db: Session = Depends(get_db)
) -> TokenResponse:
    """Issue a short-lived, single-use signing token bound to a
    caller-supplied public key (ADR-0005). The caller's private key never
    leaves their machine — only the public half is sent here."""
    client_ip = request.client.host if request.client else None
    token = issue_token(body.public_key, client_ip, db)
    return TokenResponse(token_id=token.id, expires_at=token.expires_at.isoformat())
