"""Dependency-free Microsoft Entra device-code client for the desktop node."""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch_enrollment_config(*, gateway_url: str) -> dict:
    request = Request(f"{gateway_url.rstrip('/')}/v1/node/enrollment-config", method="GET")
    with urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode())
    required = ("tenant_id", "client_id", "scope")
    if not all(str(result.get(key) or "").strip() for key in required):
        raise RuntimeError("gateway returned incomplete enrollment configuration")
    return result


def _post_form(url: str, values: dict[str, str], *, timeout: float = 10.0) -> dict:
    request = Request(
        url,
        data=urlencode(values).encode(),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def request_device_code(*, tenant_id: str, client_id: str, scope: str) -> dict:
    tenant = str(tenant_id).strip()
    if not tenant or not client_id.strip() or not scope.strip():
        raise ValueError("Entra tenant ID, client ID, and API scope are required")
    return _post_form(
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode",
        {"client_id": client_id.strip(), "scope": scope.strip()},
    )


def poll_device_token(*, tenant_id: str, client_id: str, device_code: str, interval: int = 5, expires_in: int = 900) -> dict:
    deadline = time.time() + max(60, int(expires_in))
    wait_seconds = max(2, int(interval))
    while time.time() < deadline:
        result = _post_form(
            f"https://login.microsoftonline.com/{str(tenant_id).strip()}/oauth2/v2.0/token",
            {"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "client_id": client_id.strip(), "device_code": device_code},
        )
        if result.get("access_token"):
            return result
        error = str(result.get("error") or "")
        if error == "authorization_pending":
            time.sleep(wait_seconds)
            continue
        if error == "slow_down":
            wait_seconds += 5
            time.sleep(wait_seconds)
            continue
        raise RuntimeError(str(result.get("error_description") or error or "Entra device login failed"))
    raise TimeoutError("Entra device login expired")


def enroll_node_with_entra(*, gateway_url: str, access_token: str, node_id: str, endpoint: str, agent_url: str, tailnet_name: str) -> dict:
    body = json.dumps({
        "node_id": node_id,
        "endpoint": endpoint,
        "agent_url": agent_url,
        "tailnet_name": tailnet_name,
    }).encode()
    request = Request(
        f"{gateway_url.rstrip('/')}/v1/nodes/enroll/entra",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"},
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode())
