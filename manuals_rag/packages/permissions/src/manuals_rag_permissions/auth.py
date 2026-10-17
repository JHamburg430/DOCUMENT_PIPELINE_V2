from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException

from manuals_rag_common.config import settings


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    tenant_id: str


TOKEN_ROLE_MAP = {
    settings.local_admin_token: "admin",
    settings.local_operator_token: "operator",
    settings.local_end_user_token: "end_user",
    settings.local_auditor_token: "auditor",
}


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _jwt_hs256_principal(token: str) -> Principal:
    """Validate a gateway-issued HS256 JWT without accepting unsigned tokens."""
    parts = token.split(".")
    if len(parts) != 3 or not settings.auth_jwt_secret:
        raise HTTPException(status_code=401, detail="Invalid token.")
    encoded_header, encoded_payload, encoded_signature = parts
    try:
        header = json.loads(_b64url_decode(encoded_header))
        claims = json.loads(_b64url_decode(encoded_payload))
        signature = _b64url_decode(encoded_signature)
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Invalid token.") from None
    if header.get("alg") != "HS256" or header.get("typ", "JWT") != "JWT":
        raise HTTPException(status_code=401, detail="Invalid token.")
    expected = hmac.new(
        settings.auth_jwt_secret.encode("utf-8"),
        f"{encoded_header}.{encoded_payload}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid token.")
    now = time.time()
    skew = settings.auth_jwt_clock_skew_seconds
    if not isinstance(claims.get("exp"), (int, float)) or now > float(claims["exp"]) + skew:
        raise HTTPException(status_code=401, detail="Token expired.")
    if isinstance(claims.get("nbf"), (int, float)) and now + skew < float(claims["nbf"]):
        raise HTTPException(status_code=401, detail="Token not active.")
    if settings.auth_jwt_issuer and claims.get("iss") != settings.auth_jwt_issuer:
        raise HTTPException(status_code=401, detail="Invalid token issuer.")
    if settings.auth_jwt_audience:
        audiences = claims.get("aud")
        audiences = audiences if isinstance(audiences, list) else [audiences]
        if settings.auth_jwt_audience not in audiences:
            raise HTTPException(status_code=401, detail="Invalid token audience.")
    subject = str(claims.get("sub") or "")
    tenant_id = str(claims.get("tenant_id") or claims.get("tid") or "")
    role_claim = claims.get("role") or claims.get("roles")
    role = role_claim[0] if isinstance(role_claim, list) and role_claim else role_claim
    role = str(role or "")
    if not subject or not tenant_id or role not in {"admin", "operator", "end_user", "auditor"}:
        raise HTTPException(status_code=401, detail="Invalid token claims.")
    return Principal(subject=subject, role=role, tenant_id=tenant_id)


def require_role(*roles: str):
    def dependency(authorization: str | None = Header(default=None)) -> Principal:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if settings.auth_mode == "jwt_hs256":
            principal = _jwt_hs256_principal(token)
            role = principal.role
        elif settings.auth_mode == "local":
            role = TOKEN_ROLE_MAP.get(token)
            if not role:
                raise HTTPException(status_code=401, detail="Invalid token.")
            principal = Principal(subject=role, role=role, tenant_id=settings.default_tenant_id)
        else:
            raise HTTPException(status_code=501, detail="Configured auth mode is not implemented.")
        if roles and role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role.")
        return principal

    return dependency
