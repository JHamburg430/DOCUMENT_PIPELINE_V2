"""One-time node enrollment without exposing Tailscale auth keys."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


NODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
TAILNET_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def validate_node_endpoint(
    value: str,
    *,
    allowed_host_suffixes: tuple[str, ...] = (".ts.net", ".tailnet"),
    allow_local_endpoints: bool = True,
) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("node endpoint must use http/https and include a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("node endpoint must not contain credentials, query, or fragment")
    hostname = parsed.hostname.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None:
        local_allowed = allow_local_endpoints and (address.is_loopback or address.is_link_local)
        if not (local_allowed or address.is_private or address in TAILNET_CGNAT):
            raise ValueError("node endpoint IP must be private or on the Tailscale CGNAT range")
        return
    suffixes = tuple(str(item).lower() for item in allowed_host_suffixes if str(item).strip())
    if allow_local_endpoints and hostname in {"localhost", "localhost.localdomain"}:
        return
    if not any(hostname.endswith(suffix) for suffix in suffixes):
        raise ValueError("node endpoint hostname is not an approved tailnet hostname")


@dataclass(frozen=True)
class EnrolledNode:
    node_id: str
    secret: str
    endpoint: str
    agent_url: str
    tailnet_name: str
    enrolled_at_epoch: float
    enrollment_method: str = "pairing"
    identity_subject: str = ""
    identity_tenant: str = ""
    allowed_applications: tuple[str, ...] = ()


class EnrollmentManager:
    """Issue short-lived one-time pairing tokens and persist node credentials."""

    def __init__(self, state_path: str | Path | None = None, *, clock=time.time, allowed_host_suffixes: tuple[str, ...] = (".ts.net", ".tailnet"), allow_local_endpoints: bool = True) -> None:
        self.state_path = Path(state_path) if state_path else None
        self._clock = clock
        self._allowed_host_suffixes = allowed_host_suffixes
        self._allow_local_endpoints = allow_local_endpoints
        self._lock = threading.RLock()
        self._tokens: dict[str, dict[str, Any]] = {}
        self._nodes: dict[str, EnrolledNode] = {}
        self._inference_nonces: dict[str, float] = {}
        self._load()

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text())
        self._tokens = dict(payload.get("tokens") or {})
        self._inference_nonces = dict(payload.get("inference_nonces") or {})
        self._nodes = {
            node_id: EnrolledNode(**record)
            for node_id, record in (payload.get("nodes") or {}).items()
        }

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "tokens": self._tokens,
            "inference_nonces": self._inference_nonces,
            "nodes": {node_id: asdict(node) for node_id, node in self._nodes.items()},
        }, sort_keys=True))
        temporary.chmod(0o600)
        temporary.replace(self.state_path)
        self.state_path.chmod(0o600)

    def create_pairing_token(self, *, ttl_seconds: int = 600) -> dict[str, Any]:
        ttl_seconds = max(60, min(int(ttl_seconds), 900))
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._tokens[self._digest(token)] = {
                "expires_at_epoch": self._clock() + ttl_seconds,
                "used": False,
            }
            self._persist()
        return {"token": token, "expires_at_epoch": self._clock() + ttl_seconds}

    def enroll(
        self,
        *,
        pairing_token: str,
        node_id: str,
        endpoint: str,
        agent_url: str,
        tailnet_name: str,
        enrollment_method: str = "pairing",
        identity_subject: str = "",
        identity_tenant: str = "",
        allowed_applications: tuple[str, ...] = (),
    ) -> EnrolledNode:
        if not NODE_ID_PATTERN.fullmatch(node_id):
            raise ValueError("invalid node_id")
        validate_node_endpoint(
            endpoint,
            allowed_host_suffixes=self._allowed_host_suffixes,
            allow_local_endpoints=self._allow_local_endpoints,
        )
        validate_node_endpoint(
            agent_url,
            allowed_host_suffixes=self._allowed_host_suffixes,
            allow_local_endpoints=self._allow_local_endpoints,
        )
        digest = self._digest(pairing_token)
        with self._lock:
            record = self._tokens.get(digest)
            if not record or record.get("used") or float(record.get("expires_at_epoch", 0)) <= self._clock():
                raise PermissionError("invalid or expired pairing token")
            if node_id in self._nodes:
                raise ValueError("node_id is already enrolled")
            node = EnrolledNode(
                node_id=node_id,
                secret=secrets.token_urlsafe(32),
                endpoint=endpoint.rstrip("/"),
                agent_url=agent_url.rstrip("/"),
                tailnet_name=tailnet_name,
                enrolled_at_epoch=self._clock(),
                enrollment_method=enrollment_method,
                identity_subject=identity_subject,
                identity_tenant=identity_tenant,
                allowed_applications=allowed_applications,
            )
            record["used"] = True
            self._nodes[node_id] = node
            self._persist()
            return node

    def nodes(self) -> tuple[EnrolledNode, ...]:
        with self._lock:
            return tuple(self._nodes.values())

    def claim_inference_request(self, node_id: str, request_id: str, issued_at: float) -> None:
        """Persist accepted nonces so a service restart cannot permit replay."""
        with self._lock:
            now = self._clock()
            self._inference_nonces = {key: stamp for key, stamp in self._inference_nonces.items() if stamp >= now - 60}
            key = json.dumps([node_id, request_id], separators=(",", ":"))
            if key in self._inference_nonces:
                raise PermissionError("replayed node inference request")
            if len(self._inference_nonces) >= 10000:
                raise PermissionError("node inference nonce capacity exceeded")
            self._inference_nonces[key] = issued_at
            self._persist()

    def revoke(self, node_id: str) -> bool:
        with self._lock:
            removed = self._nodes.pop(node_id, None) is not None
            if removed:
                self._persist()
            return removed
