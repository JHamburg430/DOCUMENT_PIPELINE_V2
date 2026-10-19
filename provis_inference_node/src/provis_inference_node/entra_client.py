"""Dependency-free Microsoft Entra device-code client for the desktop node."""

from __future__ import annotations

import json
import hashlib
import hmac
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .registration_proof import registration_proof_bytes


class _HttpResult(dict):
    """OAuth response fields with protocol status kept out of the payload."""

    def __init__(self, values: dict, http_status: object = None) -> None:
        super().__init__(values)
        self.http_status = http_status


class EntraHttpError(RuntimeError):
    """Safe request failure that retains only the numeric HTTP status."""

    def __init__(self, http_status: int) -> None:
        self.http_status = http_status
        super().__init__("Entra authentication request failed")


class EntraNetworkError(ConnectionError):
    """Safe connectivity failure without the underlying URL or message."""

    def __init__(self) -> None:
        super().__init__("Entra authentication service is unavailable")


def _raise_with_http_status(error: Exception, response: dict, *, category: str | None = None) -> None:
    status = getattr(response, "http_status", None)
    if status is not None:
        error.http_status = status
    if category is not None:
        error.diagnostic_category = category
    raise error


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
            return _HttpResult(_decode_json_object(response.read()), getattr(response, "status", None))
    except HTTPError as error:
        # Microsoft Entra uses HTTP 400 for normal device-code states such as
        # authorization_pending and slow_down. Return only the parsed OAuth
        # fields so the polling state machine can handle them without logging
        # or surfacing raw authentication responses.
        try:
            result = _decode_json_object(error.read(64 * 1024))
        except Exception:
            raise EntraHttpError(error.code) from None
        if str(result.get("error") or ""):
            return _HttpResult(result, error.code)
        raise EntraHttpError(error.code) from None
    except URLError:
        raise EntraNetworkError() from None


def request_device_code(*, tenant_id: str, client_id: str, scope: str) -> dict:
    tenant = str(tenant_id).strip()
    if not tenant or not client_id.strip() or not scope.strip():
        raise ValueError("Entra tenant ID, client ID, and API scope are required")
    result = _post_form(
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode",
        {"client_id": client_id.strip(), "scope": scope.strip()},
    )
    if not result.get("device_code"):
        _raise_with_http_status(
            RuntimeError("Entra device sign-in could not be started"),
            result,
            category="invalid_response",
        )
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
            _raise_with_http_status(PermissionError("Microsoft sign-in was denied"), result)
        if error in {"expired_token", "bad_verification_code", "invalid_grant"}:
            _raise_with_http_status(TimeoutError("Microsoft sign-in expired or was cancelled"), result)
        _raise_with_http_status(RuntimeError("Microsoft sign-in could not be completed"), result)


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


def verify_node_registration_with_entra(*, gateway_url: str, access_token: str, node_id: str, shared_secret: str) -> bool:
    """Ask the gateway to verify this signed-in owner against a saved registration."""
    if not node_id.strip() or not shared_secret:
        return False
    challenge_request = Request(
        f"{gateway_url.rstrip('/')}/v1/nodes/registration-challenge/entra",
        data=json.dumps({"node_id": node_id}).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"},
    )
    try:
        with urlopen(challenge_request, timeout=15) as response:
            issued = _decode_json_object(response.read(16385))
    except HTTPError as error:
        if error.code in {401, 403, 404}:
            denied = PermissionError("saved node registration could not be verified")
            denied.http_status = error.code
            denied.diagnostic_category = "permission_denied"
            raise denied from None
        raise EntraHttpError(error.code) from None
    except URLError:
        raise EntraNetworkError() from None
    challenge = issued.get("challenge")
    if issued.get("status") != "challenge" or not isinstance(challenge, str) or len(challenge) != 43:
        raise PermissionError("saved node registration could not be verified")
    signature = hmac.new(shared_secret.encode(), registration_proof_bytes(node_id, challenge, access_token), hashlib.sha256).hexdigest()
    body = json.dumps({"node_id": node_id, "challenge": challenge, "signature": signature}).encode()
    request = Request(
        f"{gateway_url.rstrip('/')}/v1/nodes/verify-registration/entra",
        data=json.dumps({"node_id": node_id, "challenge": challenge, "signature": signature}).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = _decode_json_object(response.read())
    except HTTPError as error:
        if error.code in {401, 403, 404}:
            return False
        raise EntraHttpError(error.code) from None
    except URLError:
        raise EntraNetworkError() from None
    return result.get("status") == "accepted"


def reuse_or_enroll_node_with_entra(*, config: dict, gateway_url: str, access_token: str, node_id: str, endpoint: str, agent_url: str, tailnet_name: str) -> dict:
    """Reuse verified saved credentials, enrolling only when no credentials exist."""
    saved_secret = str(config.get("shared_secret") or "")
    if saved_secret:
        saved_node_id = str(config.get("node_id") or "")
        if not verify_node_registration_with_entra(
            gateway_url=gateway_url,
            access_token=access_token,
            node_id=saved_node_id,
            shared_secret=saved_secret,
        ):
            raise PermissionError("saved node registration could not be verified")
        return config
    result = enroll_node_with_entra(
        gateway_url=gateway_url,
        access_token=access_token,
        node_id=node_id,
        endpoint=endpoint,
        agent_url=agent_url,
        tailnet_name=tailnet_name,
    )
    return {**config, **result}
