"""Microsoft Entra access-token validation for node enrollment.

Validation is fail-closed and intentionally depends on PyJWT with its crypto
extra in the gateway environment. The node never needs this module: it only
uses the public-client device-code flow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EntraValidatorConfig:
    tenant_id: str
    audience: str
    issuer: str
    jwks_url: str
    client_id: str = ""
    required_scope: str = ""
    allowed_roles: tuple[str, ...] = ()
    allowed_groups: tuple[str, ...] = ()


class EntraTokenValidator:
    def __init__(self, config: EntraValidatorConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EntraTokenValidator":
        env = os.environ if env is None else env
        tenant_id = str(env.get("INFERENCE_ENTRA_TENANT_ID") or "").strip()
        audience = str(env.get("INFERENCE_ENTRA_AUDIENCE") or "").strip()
        if not tenant_id or not audience:
            raise RuntimeError("INFERENCE_ENTRA_TENANT_ID and INFERENCE_ENTRA_AUDIENCE are required")
        issuer = str(env.get("INFERENCE_ENTRA_ISSUER") or f"https://login.microsoftonline.com/{tenant_id}/v2.0").strip().rstrip("/")
        jwks_url = str(env.get("INFERENCE_ENTRA_JWKS_URL") or f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys").strip()
        client_id = str(env.get("INFERENCE_ENTRA_CLIENT_ID") or "").strip()
        required_scope = str(env.get("INFERENCE_ENTRA_SCOPE") or "").strip()
        allowed_roles = tuple(item.strip() for item in str(env.get("INFERENCE_ENTRA_ALLOWED_ROLES") or "").split(",") if item.strip())
        allowed_groups = tuple(item.strip() for item in str(env.get("INFERENCE_ENTRA_ALLOWED_GROUPS") or "").split(",") if item.strip())
        return cls(EntraValidatorConfig(tenant_id, audience, issuer, jwks_url, client_id, required_scope, allowed_roles, allowed_groups))

    def validate(self, access_token: str) -> dict[str, Any]:
        token = str(access_token or "").strip()
        if not token:
            raise PermissionError("missing Entra access token")
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as error:
            raise RuntimeError("PyJWT with crypto support is required for Entra validation") from error
        try:
            signing_key = PyJWKClient(self.config.jwks_url).get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"require": ["exp", "iat", "iss", "aud", "tid", "sub"]},
            )
        except Exception as error:
            raise PermissionError("invalid Entra access token") from error
        if str(claims.get("tid") or "") != self.config.tenant_id:
            raise PermissionError("Entra token tenant mismatch")
        if self.config.client_id:
            authorized_party = str(claims.get("azp") or claims.get("appid") or "")
            if authorized_party != self.config.client_id:
                raise PermissionError("Entra token client mismatch")
        if self.config.required_scope:
            scopes = {item.strip() for item in str(claims.get("scp") or "").split() if item.strip()}
            required_scope = self.config.required_scope.rstrip("/").rsplit("/", 1)[-1]
            if not scopes.intersection({self.config.required_scope, required_scope}):
                raise PermissionError("Entra token lacks the required delegated enrollment scope")
        roles = {str(item) for item in (claims.get("roles") or [])}
        groups = {str(item) for item in (claims.get("groups") or [])}
        if self.config.allowed_roles and not roles.intersection(self.config.allowed_roles):
            raise PermissionError("Entra token lacks an approved role")
        if self.config.allowed_groups and not groups.intersection(self.config.allowed_groups):
            raise PermissionError("Entra token lacks an approved group")
        return {
            "subject": str(claims["sub"]),
            "tenant_id": str(claims["tid"]),
            "issuer": str(claims["iss"]),
            "roles": sorted(roles.intersection(self.config.allowed_roles)) if self.config.allowed_roles else sorted(roles),
            "groups": sorted(groups.intersection(self.config.allowed_groups)) if self.config.allowed_groups else sorted(groups),
        }
