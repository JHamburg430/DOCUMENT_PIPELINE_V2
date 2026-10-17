"""Dependency-free Microsoft Entra device-code client for the desktop node."""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_enrollment_config(*, gateway_url: str) -> dict:
    request = Request(f"{gateway_url.rstrip('/')}/v1/node/enrollment-config", method="GET")
    with urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode())
    required = ("tenant_id", "client_id", "scope")
    if not all(str(result.get(key) or "").strip() for key in required):
        raise RuntimeError("gateway returned incomplete enrollment configuration")
    return result


def _decode_json_object(raw: bytes) -> dict:
    # OAuth error bodies are small JSON objects. Bound the response so a bad
    # endpoint cannot make the desktop client retain or process an unbounded
    # authentication response.
    if len(raw) > 64 * 1024:
        raise RuntimeError("Entra authentication response was too large")
    try:
        result = json.loads(raw.decode("utf-8", errors="replace"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Entra authentication returned an invalid response") from error
    if not isinstance(result, dict):
        raise RuntimeError("Entra authentication returned an invalid response")
    return result


def _post_form(url: str, values: dict[str, str], *, timeout: float = 10.0) -> dict:
    request = Request(
        url,
        data=urlencode(values).encode(),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return _decode_json_object(response.read())
    except HTTPError as error:
        # Microsoft Entra uses HTTP 400 for normal device-code states such as
        # authorization_pending and slow_down. Return only the parsed OAuth
        # fields so the polling state machine can handle them without logging
        # or surfacing raw authentication responses.
        try:
            result = _decode_json_object(error.read(64 * 1024))
        except Exception as parse_error:
            raise RuntimeError(f"Entra authentication request failed (HTTP {error.code})") from parse_error
        if str(result.get("error") or ""):
            return result
        raise RuntimeError(f"Entra authentication request failed (HTTP {error.code})")
    except URLError as error:
        raise RuntimeError("Entra authentication service is unavailable") from error


def request_device_code(*, tenant_id: str, client_id: str, scope: str) -> dict:
    tenant = str(tenant_id).strip()
    if not tenant or not client_id.strip() or not scope.strip():
        raise ValueError("Entra tenant ID, client ID, and API scope are required")
    result = _post_form(
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode",
        {"client_id": client_id.strip(), "scope": scope.strip()},
    )
    if not result.get("device_code"):
        raise RuntimeError("Entra device sign-in could not be started")
    return result


def poll_device_token(*, tenant_id: str, client_id: str, device_code: str, interval: int = 5, expires_in: int = 900) -> dict:
    deadline = time.monotonic() + max(1, int(expires_in))
    wait_seconds = max(1, int(interval))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Microsoft sign-in expired before approval")
        result = _post_form(
            f"https://login.microsoftonline.com/{str(tenant_id).strip()}/oauth2/v2.0/token",
            {"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "client_id": client_id.strip(), "device_code": device_code},
        )
        if result.get("access_token"):
            return result
        error = str(result.get("error") or "")
        if error == "authorization_pending":
            time.sleep(min(wait_seconds, max(0.0, remaining)))
            continue
        if error == "slow_down":
            wait_seconds += 5
            time.sleep(min(wait_seconds, max(0.0, remaining)))
            continue
        if error in {"authorization_declined", "access_denied"}:
            raise PermissionError("Microsoft sign-in was denied")
        if error in {"expired_token", "bad_verification_code", "invalid_grant"}:
            raise TimeoutError("Microsoft sign-in expired or was cancelled")
        raise RuntimeError("Microsoft sign-in could not be completed")


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
