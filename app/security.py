from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
from fastapi import Header, HTTPException, Request, status


def require_internal_service(
    request: Request,
    x_service_key: str | None = Header(default=None, alias="X-Service-Key"),
) -> str:
    expected = request.app.state.container.settings.internal_service_key
    if not x_service_key or not secrets.compare_digest(x_service_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service credential")
    return "internal-service"


def subject_from_profile_payload(payload: Any) -> str:
    try:
        nip_baru = payload["data"]["nip_baru"]
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ABSENWEB returned an invalid profile response",
        ) from exc
    if not isinstance(nip_baru, str):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ABSENWEB profile does not contain a valid NIP",
        )
    normalized_nip = re.sub(r"\D", "", nip_baru)
    if len(normalized_nip) != 18:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ABSENWEB profile does not contain a valid 18-digit NIP",
        )
    return f"nip-{normalized_nip}"


async def require_mobile_subject(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login token is required")

    settings = request.app.state.container.settings
    profile_url = (
        f"{settings.absenweb_base_url.rstrip('/')}"
        f"/{settings.absenweb_profile_path.lstrip('/')}"
    )
    try:
        async with httpx.AsyncClient(timeout=settings.mobile_auth_timeout_seconds) as client:
            response = await client.get(
                profile_url,
                headers={"Accept": "application/json", "Authorization": authorization},
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ABSENWEB authentication service is unavailable",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login token is invalid")
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ABSENWEB rejected the profile lookup",
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ABSENWEB returned an invalid profile response",
        ) from exc
    return subject_from_profile_payload(payload)


def issue_upload_token(
    *,
    secret: str,
    issuer: str,
    purpose: str,
    session_id: str,
    subject_id: str,
    ttl_seconds: int,
) -> str:
    now = datetime.now(UTC)
    payload = {
        "iss": issuer,
        "sub": subject_id,
        "aud": "face-upload",
        "purpose": purpose,
        "session_id": session_id,
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(seconds=ttl_seconds),
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_upload_token(request: Request, authorization: str | None) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Upload token is required")
    token = authorization.removeprefix("Bearer ").strip()
    settings = request.app.state.container.settings
    try:
        return jwt.decode(
            token,
            settings.upload_token_secret,
            algorithms=["HS256"],
            audience="face-upload",
            issuer=settings.token_issuer,
            options={"require": ["exp", "iat", "nbf", "sub", "purpose", "session_id"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Upload token is invalid"
        ) from exc


def require_upload_scope(
    request: Request,
    authorization: str | None,
    *,
    purpose: str,
    session_id: str,
) -> dict[str, Any]:
    claims = decode_upload_token(request, authorization)
    if claims.get("purpose") != purpose or claims.get("session_id") != session_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Upload token scope is invalid")
    return claims
