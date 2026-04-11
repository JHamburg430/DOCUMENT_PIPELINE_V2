from __future__ import annotations

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


def require_role(*roles: str):
    def dependency(authorization: str | None = Header(default=None)) -> Principal:
        if settings.auth_mode != "local":
            raise HTTPException(status_code=501, detail="Only local auth bootstrap is implemented.")
        token = (authorization or "").removeprefix("Bearer ").strip()
        role = TOKEN_ROLE_MAP.get(token)
        if not role:
            raise HTTPException(status_code=401, detail="Invalid token.")
        if roles and role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role.")
        return Principal(subject=role, role=role, tenant_id=settings.default_tenant_id)

    return dependency
