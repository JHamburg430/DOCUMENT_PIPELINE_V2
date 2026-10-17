"""Bounded node-control commands; no shell or Docker control is exposed."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControlPolicy:
    allowed_model_prefixes: tuple[str, ...] = ("qwen", "nomic", "bge", "mxbai")
    max_model_size_bytes: int = 80_000_000_000
    max_clock_skew_seconds: int = 60


def sign_command(command: dict[str, Any], secret: bytes) -> str:
    body = json.dumps(command, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def verify_command(command: dict[str, Any], signature: str, secret: bytes, *, now: float | None = None, policy: ControlPolicy | None = None) -> bool:
    policy = policy or ControlPolicy()
    now = time.time() if now is None else now
    if abs(now - float(command.get("issued_at", 0))) > policy.max_clock_skew_seconds:
        return False
    if command.get("action") not in {"pull_model", "evict_model"}:
        return False
    model = str(command.get("model") or "")
    if not any(model.startswith(prefix) for prefix in policy.allowed_model_prefixes):
        return False
    if command.get("action") == "pull_model":
        size_bytes = int(command.get("size_bytes") or 0)
        if size_bytes <= 0 or size_bytes > policy.max_model_size_bytes:
            return False
    expected = sign_command(command, secret)
    return hmac.compare_digest(expected, signature)
