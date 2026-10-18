"""Sanitized connection diagnostics for the managed desktop node."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError


CONNECT_PHASES = frozenset({
    "gateway_discovery", "device_code_request", "device_code_poll",
    "entra_registration_check", "entra_enrollment", "local_service_start",
    "connect_complete",
})
ERROR_CATEGORIES = frozenset({
    "authorization_denied", "authorization_expired", "http_error",
    "invalid_response", "local_service_error", "network_error",
    "permission_denied", "timeout", "unknown_error",
})


def _status(error: BaseException) -> int | None:
    for name in ("http_status", "status_code", "code"):
        try:
            value = int(getattr(error, name, None))
        except (TypeError, ValueError):
            continue
        if 100 <= value <= 599:
            return value
    return None


def _category(error: BaseException, phase: str) -> str:
    declared = getattr(error, "diagnostic_category", None)
    if declared in ERROR_CATEGORIES:
        return declared
    if phase == "device_code_poll" and isinstance(error, PermissionError):
        return "authorization_denied"
    if phase == "device_code_poll" and isinstance(error, TimeoutError):
        return "authorization_expired"
    if _status(error) is not None or isinstance(error, HTTPError):
        return "http_error"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (URLError, ConnectionError)):
        return "network_error"
    if isinstance(error, OSError):
        return "local_service_error" if phase == "local_service_start" else "network_error"
    if isinstance(error, (ValueError, KeyError, TypeError, json.JSONDecodeError)):
        return "invalid_response"
    return "unknown_error"


def append_connect_diagnostic(path: str | Path, phase: str, *, error: BaseException | None = None) -> None:
    """Append only an allowlisted phase, HTTP status, and safe category."""
    safe_phase = phase if phase in CONNECT_PHASES else "unknown"
    record: dict[str, object] = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "phase": safe_phase,
    }
    if error is not None:
        status = _status(error)
        if status is not None:
            record["http_status"] = status
        record["error_category"] = _category(error, safe_phase)
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        if os.name != "nt":
            log_path.chmod(0o600)
    except OSError:
        # Diagnostics must not change whether the connection succeeds.
        return
