"""Central scheduler and Ollama-compatible inference gateway.

The gateway is intentionally a small, dependency-free prototype. It can run
behind Tailscale and is not enabled by existing applications until its rollout
gates are satisfied.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import logging
import os
import threading
import time
import urllib.error
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.request import Request, urlopen

from .models import InferenceRequest, ModelSpec, NodeSnapshot
from .scheduler import InferenceScheduler, SchedulerPolicy
from .control import sign_command
from .enrollment import EnrollmentManager
from .entra import EntraTokenValidator
from .streaming import NDJSONResponse, relay_ndjson, watch_disconnect


class AuthenticationError(PermissionError):
    """The caller did not present valid operator/application credentials."""


class AuthorizationError(PermissionError):
    """The caller is authenticated but is not allowed for the operation."""


class InferencePhaseError(RuntimeError):
    def __init__(self, phase: str, model: str, code: str, *, node_id: str | None = None, upstream_status: int | None = None, status: int = 503):
        super().__init__(code)
        self.phase, self.model, self.code = phase, model, code
        self.node_id, self.upstream_status, self.status = node_id, upstream_status, status

    def public(self):
        return {"detail": "model preparation failed" if self.phase in {"model_pull", "model_warm"} else "inference unavailable", "phase": self.phase, "model": self.model, "code": self.code, "node_id": self.node_id, "upstream_status": self.upstream_status}


def _log_phase_failure(error: InferencePhaseError):
    logging.getLogger(__name__).warning("inference_failure %s", json.dumps(error.public(), sort_keys=True))


def _stream_failure(request, routing):
    error = InferencePhaseError("inference_failed", request.model, "upstream_stream_failed", node_id=routing.get("node_id"))
    _log_phase_failure(error)
    return {"error": error.public(), "done": True}


def _admission_failure(request, decision):
    phase = "model_not_installed" if "model_not_installed_and_pull_not_allowed" in decision.reason else "admission_rejected"
    return InferencePhaseError(phase, request.model, phase)


def heartbeat_signature(payload: bytes, secret: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def node_request_signature(payload: dict[str, Any], secret: bytes) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def validate_gateway_runtime_security(
    *,
    gateway_token: str,
    bind: str,
    allow_public_bind: bool = False,
) -> None:
    """Fail closed on weak client authentication or accidental public bind."""
    if len(gateway_token.strip()) < 32:
        raise RuntimeError("INFERENCE_GATEWAY_TOKEN must contain at least 32 characters")
    if bind.strip() in {"", "0.0.0.0", "::"} and not allow_public_bind:
        raise RuntimeError("gateway wildcard bind requires explicit public-bind approval")


class InferenceGateway:
    """Registry, admission, and forwarding boundary for Ollama requests."""

    def __init__(self, scheduler: InferenceScheduler | None = None, *, node_secrets: dict[str, bytes] | None = None, node_agent_urls: dict[str, str] | None = None, gateway_token: str | None = None, post: Callable[..., dict[str, Any]] | None = None, enrollment: EnrollmentManager | None = None, require_node_agents: bool = False, model_catalog: dict[str, ModelSpec] | None = None, require_model_catalog: bool = False, application_tokens: dict[str, str] | None = None, require_application_identity: bool = False, entra_validator: EntraTokenValidator | None = None, require_entra_enrollment: bool = False, entra_client_id: str = "", entra_scope: str = "", node_default_tailnet: str = "", node_default_applications: tuple[str, ...] = (), node_default_port: int = 8765, application_worker_allowlist: dict[str, tuple[str, ...]] | None = None, allow_node_agent_endpoint_refresh: bool = False) -> None:
        self.scheduler = scheduler or InferenceScheduler()
        self.node_secrets = node_secrets or {}
        self.node_agent_urls = node_agent_urls or {}
        self.gateway_token = gateway_token
        self.require_node_agents = require_node_agents
        self.model_catalog = dict(model_catalog or {})
        self.require_model_catalog = require_model_catalog
        self.application_tokens = dict(application_tokens or {})
        self.require_application_identity = require_application_identity
        self.entra_validator = entra_validator
        self.require_entra_enrollment = require_entra_enrollment
        self.entra_client_id = entra_client_id.strip()
        self.entra_scope = entra_scope.strip()
        self.node_default_tailnet = node_default_tailnet.strip()
        self.node_default_applications = tuple(item for item in node_default_applications if item)
        self.node_default_port = int(node_default_port)
        self.allow_node_agent_endpoint_refresh = allow_node_agent_endpoint_refresh is True
        self.application_worker_allowlist = dict(application_worker_allowlist or {})
        if any(not isinstance(app, str) or not app or not isinstance(nodes, (list, tuple)) or not nodes or any(not isinstance(node, str) or not node for node in nodes) for app, nodes in self.application_worker_allowlist.items()):
            raise ValueError("invalid application worker allowlist")
        self._heartbeat_lock = threading.Lock()
        self.agent_link = None
        self._last_heartbeat_observed: dict[str, float] = {}
        self._post = post or self._post_json
        self.enrollment = enrollment
        if self.enrollment is not None:
            for node in self.enrollment.nodes():
                self.node_secrets.setdefault(node.node_id, node.secret.encode())
                self.node_agent_urls.setdefault(node.node_id, node.agent_url)

    def create_pairing_token(self, *, ttl_seconds: int = 600) -> dict[str, Any]:
        if self.enrollment is None:
            raise RuntimeError("node enrollment is not configured")
        return self.enrollment.create_pairing_token(ttl_seconds=ttl_seconds)

    def enroll_node(self, *, pairing_token: str, node_id: str, endpoint: str, agent_url: str, tailnet_name: str) -> dict[str, Any]:
        if self.require_entra_enrollment:
            raise PermissionError("Microsoft Entra enrollment is required")
        if self.enrollment is None:
            raise RuntimeError("node enrollment is not configured")
        node = self.enrollment.enroll(
            pairing_token=pairing_token,
            node_id=node_id,
            endpoint=endpoint,
            agent_url=agent_url,
            tailnet_name=tailnet_name,
            allowed_applications=self.node_default_applications,
        )
        self.node_secrets[node.node_id] = node.secret.encode()
        self.node_agent_urls[node.node_id] = node.agent_url
        return {
            "node_id": node.node_id,
            "shared_secret": node.secret,
            "endpoint": node.endpoint,
            "agent_url": node.agent_url,
            "tailnet_name": node.tailnet_name,
        }

    def enroll_node_with_entra(self, *, access_token: str, node_id: str, endpoint: str, agent_url: str, tailnet_name: str) -> dict[str, Any]:
        """Enroll a node after validating a Microsoft Entra API access token."""
        if self.enrollment is None:
            raise RuntimeError("node enrollment is not configured")
        if self.entra_validator is None:
            raise RuntimeError("Entra enrollment is not configured")
        identity = self.entra_validator.validate(access_token)
        # Entra enrollment still consumes a single-use internal token so the
        # existing EnrollmentManager remains the only node-secret generator.
        pairing = self.enrollment.create_pairing_token(ttl_seconds=60)
        node = self.enrollment.enroll(
            pairing_token=pairing["token"],
            node_id=node_id,
            endpoint=endpoint,
            agent_url=agent_url,
            tailnet_name=tailnet_name,
            enrollment_method="entra",
            identity_subject=identity["subject"],
            identity_tenant=identity["tenant_id"],
            allowed_applications=self.node_default_applications,
        )
        self.node_secrets[node.node_id] = node.secret.encode()
        self.node_agent_urls[node.node_id] = node.agent_url
        return {
            "node_id": node.node_id,
            "shared_secret": node.secret,
            "endpoint": node.endpoint,
            "agent_url": node.agent_url,
            "tailnet_name": node.tailnet_name,
            "enrollment_method": node.enrollment_method,
        }

    def enrollment_config(self) -> dict[str, Any]:
        """Return non-secret bootstrap settings needed by the desktop client."""
        if self.entra_validator is None or not self.entra_client_id or not self.entra_scope:
            raise RuntimeError("Entra enrollment bootstrap is not configured")
        return {
            "tenant_id": self.entra_validator.config.tenant_id,
            "client_id": self.entra_client_id,
            "scope": self.entra_scope,
            "tailnet_name": self.node_default_tailnet,
            "allowed_applications": list(self.node_default_applications),
            "port": self.node_default_port,
        }

    def revoke_node(self, node_id: str) -> dict[str, Any]:
        if self.enrollment is None:
            raise RuntimeError("node enrollment is not configured")
        removed = self.enrollment.revoke(node_id)
        self.node_secrets.pop(node_id, None)
        self.node_agent_urls.pop(node_id, None)
        self.scheduler.remove_node(node_id)
        return {"node_id": node_id, "revoked": removed}

    def disconnect_node(self, *, node_id: str, raw_body: bytes, signature: str) -> dict[str, Any]:
        """Remove a node from routing while preserving its enrollment record."""
        secret = self.node_secrets.get(node_id)
        if secret is None or not hmac.compare_digest(heartbeat_signature(raw_body, secret), signature):
            raise PermissionError("invalid node disconnect signature")
        try:
            payload = json.loads(raw_body.decode())
            issued_at = float(payload.get("issued_at") or 0)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid node disconnect payload") from error
        if abs(time.time() - issued_at) > 60:
            raise PermissionError("stale node disconnect request")
        self.scheduler.remove_node(node_id)
        self._last_heartbeat_observed.pop(node_id, None)
        return {"node_id": node_id, "disconnected": True, "revoked": False}

    def authorize_operator(self, token: str | None) -> None:
        if self.gateway_token is not None and not hmac.compare_digest(self.gateway_token, token or ""):
            raise AuthenticationError("gateway operator authentication required")

    def authorize_inference(self, token: str | None, *, application: str | None = None, application_token: str | None = None) -> None:
        self.authorize_operator(token)
        if self.require_application_identity:
            expected = self.application_tokens.get(str(application or ""))
            if expected is None or not hmac.compare_digest(expected, application_token or ""):
                raise AuthenticationError("valid application identity required for inference")

    def authorize_application(self, *, application: str, application_token: str | None) -> None:
        """Application tokens grant inference only, never operator administration."""
        expected = self.application_tokens.get(application)
        if not expected or not hmac.compare_digest(expected, application_token or ""):
            raise AuthenticationError("valid application identity required for inference")

    def authorize_client(self, token: str | None, *, application: str | None = None, application_token: str | None = None) -> None:
        """Backward-compatible name for inference authorization."""
        self.authorize_inference(token, application=application, application_token=application_token)

    def register_heartbeat(self, snapshot: NodeSnapshot, raw_body: bytes, signature: str) -> None:
        secret = self.node_secrets.get(snapshot.node_id)
        if secret is None or not hmac.compare_digest(heartbeat_signature(raw_body, secret), signature):
            raise PermissionError("invalid node heartbeat signature")
        observed_at = float(snapshot.observed_at_epoch or 0)
        now = time.time()
        if observed_at <= 0 or observed_at > now + 60:
            raise PermissionError("invalid node heartbeat timestamp")
        if now - observed_at > self.scheduler.policy.heartbeat_ttl_seconds:
            raise PermissionError("stale node heartbeat")
        with self._heartbeat_lock:
            previous = self._last_heartbeat_observed.get(snapshot.node_id)
            if previous is not None and observed_at <= previous:
                raise PermissionError("replayed node heartbeat")
            self._last_heartbeat_observed[snapshot.node_id] = observed_at
        self.scheduler.upsert_node(snapshot)

    def refresh_node_agent_endpoint(self, raw_body: bytes, signature: str) -> dict[str, str]:
        if not self.allow_node_agent_endpoint_refresh or self.enrollment is None:
            raise PermissionError("node listener endpoint refresh is disabled")
        try:
            payload = json.loads(raw_body)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid node listener refresh request") from error
        if not isinstance(payload, dict):
            raise ValueError("invalid node listener refresh request")
        node_id, agent_url = payload.get("node_id"), payload.get("agent_url")
        issued_at, nonce = payload.get("issued_at"), payload.get("nonce")
        secret = self.node_secrets.get(node_id) if isinstance(node_id, str) else None
        if secret is None or not hmac.compare_digest(heartbeat_signature(raw_body, secret), signature):
            raise AuthenticationError("invalid node listener authentication")
        now = time.time()
        if (isinstance(issued_at, bool) or not isinstance(issued_at, (int, float))
                or not math.isfinite(issued_at) or abs(now - issued_at) > 60
                or not isinstance(nonce, str) or not 16 <= len(nonce) <= 128
                or not isinstance(agent_url, str) or len(agent_url) > 512):
            raise AuthenticationError("fresh node listener proof required")
        node = self.enrollment.refresh_agent_url(node_id, agent_url, issued_at=issued_at, nonce=nonce)
        self.node_agent_urls[node_id] = node.agent_url
        return {"status": "accepted", "node_id": node_id}

    def admit(self, request: InferenceRequest):
        return self.scheduler.choose(self._authoritative_request(request))

    def authenticate_node_inference(self, raw: bytes, signature: str) -> dict[str, Any]:
        """Authenticate the exact body using persistent enrollment, never app tokens."""
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("invalid node inference payload")
            node_id = body.get("node_id")
            if not isinstance(node_id, str):
                raise ValueError("node_id is required")
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid node inference payload") from error
        record = next((node for node in self.enrollment.nodes() if node.node_id == node_id), None) if self.enrollment else None
        if record is None or not hmac.compare_digest(heartbeat_signature(raw, record.secret.encode()), signature):
            raise AuthenticationError("invalid node authentication")
        issued_at = body.get("issued_at")
        if isinstance(issued_at, bool) or not isinstance(issued_at, (int, float)) or not math.isfinite(issued_at) or abs(time.time() - issued_at) > 60:
            raise AuthenticationError("stale or invalid node inference timestamp")
        request = body.get("request")
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200:
            raise ValueError("invalid request_id")
        snapshot = next((node for node in self.scheduler.snapshots() if node.node_id == node_id), None)
        now = time.time()
        if snapshot is None or not snapshot.healthy or not snapshot.inference_enabled or not 0 <= now - snapshot.observed_at_epoch <= self.scheduler.policy.heartbeat_ttl_seconds:
            raise AuthorizationError("ingress node is not eligible")
        # Legacy enrollment records use only the server's configured defaults.
        # A heartbeat can narrow this approval; it cannot broaden it.
        approved = record.allowed_applications or self.node_default_applications
        application = request.get("application")
        if not isinstance(application, str) or application not in approved or application not in snapshot.allowed_applications:
            raise AuthorizationError("application is not approved for ingress node")
        model = request.get("model")
        if not isinstance(model, str) or model not in self.model_catalog:
            raise InferencePhaseError("model_not_in_catalog", "", "model_not_in_catalog", status=400)
        try:
            self.enrollment.claim_inference_request(node_id, request_id, issued_at)
        except PermissionError as error:
            raise AuthenticationError("replayed or unavailable node inference nonce") from error
        return body

    def infer(self, request: InferenceRequest, *, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation not in {"chat", "generate", "embeddings"}:
            raise ValueError("unsupported inference operation")
        request = self._authoritative_request(request)
        payload = dict(payload)
        payload_model = payload.get("model")
        if payload_model is not None and str(payload_model) != request.model:
            raise RuntimeError("payload_model_mismatch")
        payload["model"] = request.model
        decision = self.scheduler.choose_and_reserve(request)
        if not decision.admitted or not decision.endpoint:
            raise _admission_failure(request, decision)
        try:
            self._prepare_node(request, decision)
            ollama_operation = {"embeddings": "embed"}.get(operation, operation)
            timeout = max(30.0, (request.max_latency_ms or 180000) / 1000.0)
            node_agent_url = self.node_agent_urls.get(decision.node_id or "")
            if self.require_node_agents and not node_agent_url:
                raise RuntimeError("node_agent_required_for_inference")
            if node_agent_url:
                secret = self.node_secrets.get(decision.node_id or "")
                if secret is None:
                    raise RuntimeError("node agent is configured without a node secret")
                envelope = {
                    "operation": operation,
                    "payload": payload,
                    "request_id": request.request_id,
                    "issued_at": time.time(),
                }
                response = self._post(
                    f"{node_agent_url.rstrip('/')}/v1/node/inference",
                    {**envelope, "signature": node_request_signature(envelope, secret)},
                    timeout=timeout,
                )
            else:
                response = self._post(f"{decision.endpoint}/api/{ollama_operation}", payload, timeout=timeout)
            return {"routing": decision.to_dict(), "response": response}
        except InferencePhaseError:
            raise
        except Exception as error:
            raise InferencePhaseError("inference_failed", request.model, "worker_inference_failed", node_id=decision.node_id, upstream_status=getattr(error, "code", None)) from None
        finally:
            self.scheduler.release(decision.node_id, request.request_id)

    @contextmanager
    def stream_infer(self, request: InferenceRequest, *, operation: str, payload: dict[str, Any]):
        if operation not in {"chat", "generate"}:
            raise ValueError("streaming is only supported for chat and generate")
        request = self._authoritative_request(request)
        if payload.get("model", request.model) != request.model:
            raise ValueError("payload model does not match request")
        decision = self.scheduler.choose_and_reserve(request)
        if not decision.admitted or not decision.endpoint:
            raise _admission_failure(request, decision)
        stream = None
        deadline = time.monotonic() + max(0.001, min(900, (request.max_latency_ms or 180000) / 1000))
        try:
            self._prepare_node(request, decision)
            url = self.node_agent_urls.get(decision.node_id or "")
            secret = self.node_secrets.get(decision.node_id or "")
            if not url or secret is None:
                raise RuntimeError("signed node streaming is required")
            envelope = {"operation": operation, "payload": {**payload, "model": request.model, "stream": True}, "request_id": request.request_id, "issued_at": time.time()}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("inference preparation timed out")
            stream = NDJSONResponse(f"{url.rstrip('/')}/v1/node/inference", {**envelope, "signature": node_request_signature(envelope, secret)}, timeout=remaining)
            yield decision.to_dict(), stream
        except InferencePhaseError:
            raise
        except Exception as error:
            raise InferencePhaseError("inference_failed", request.model, "worker_stream_failed", node_id=decision.node_id, upstream_status=getattr(error, "code", None)) from None
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                self.scheduler.release(decision.node_id, request.request_id)

    def _authoritative_request(self, request: InferenceRequest) -> InferenceRequest:
        """Use central catalog truth instead of caller-supplied model resources."""
        from dataclasses import replace
        # Worker restrictions are server-owned; callers cannot widen or bypass them.
        request = replace(request, allowed_node_ids=tuple(self.application_worker_allowlist.get(request.application, ())))
        if not self.model_catalog:
            if self.require_model_catalog:
                raise RuntimeError("model_catalog_required")
            return request
        spec = self.model_catalog.get(request.model)
        if spec is None:
            raise InferencePhaseError("model_not_in_catalog", "", "model_not_in_catalog", status=400)
        missing = set(request.required_capabilities) - set(spec.capabilities)
        if missing:
            raise InferencePhaseError("admission_rejected", request.model, "model_capability_unavailable")
        from dataclasses import replace

        return replace(request, model_size_bytes=spec.size_bytes, min_vram_mb=spec.min_vram_mb)

    def _prepare_node(self, request: InferenceRequest, decision) -> None:
        """Perform only explicitly authorized preparation before inference."""
        if not decision.requires_model_pull and not decision.eviction_candidates and not decision.requires_model_warm:
            return
        if decision.eviction_candidates and not request.allow_eviction:
            raise RuntimeError("scheduler returned unauthorized eviction candidates")
        if decision.requires_model_pull and not request.allow_model_pull:
            raise RuntimeError("scheduler returned unauthorized model pull")
        if decision.node_id not in self.node_agent_urls or decision.node_id not in self.node_secrets:
            raise RuntimeError("node preparation is not configured for the selected node")
        for model in decision.eviction_candidates:
            result = self.issue_node_command(
                node_id=decision.node_id,
                action="evict_model",
                model=model,
                timeout_seconds=120,
            )
            if (result.get("result") or {}).get("status") != "completed":
                raise RuntimeError(f"model eviction failed for {model}")
        if decision.requires_model_pull:
            result = self._preparation_command(request, decision, "pull_model", "model_pull")
            if (result.get("result") or {}).get("status") != "completed":
                raise InferencePhaseError("model_pull", request.model, "model_pull_failed", node_id=decision.node_id)
        if decision.requires_model_warm:
            result = self._preparation_command(request, decision, "warm_model", "model_warm")
            if (result.get("result") or {}).get("status") != "completed" or (result.get("result") or {}).get("loaded") is not True:
                raise InferencePhaseError("model_warm", request.model, "model_readiness_unconfirmed", node_id=decision.node_id)

    def _preparation_command(self, request, decision, action, phase):
        try:
            return self.issue_node_command(
                node_id=decision.node_id,
                action=action,
                model=request.model,
                size_bytes=request.model_size_bytes,
                timeout_seconds=min(900, max(1, int((request.max_latency_ms or 180000) / 1000))),
                context_tokens=request.context_tokens,
            )
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read(8192))
                upstream_status = payload.get("upstream_status")
                if type(upstream_status) is not int or not 100 <= upstream_status <= 599:
                    upstream_status = None
            except Exception:
                upstream_status = None
            code = f"ollama_http_{upstream_status}" if upstream_status else f"node_http_{error.code}"
            raise InferencePhaseError(phase, request.model, code, node_id=decision.node_id, upstream_status=upstream_status or error.code) from None
        except (OSError, urllib.error.URLError):
            raise InferencePhaseError("selected_node_unreachable", request.model, "selected_node_unreachable", node_id=decision.node_id) from None

    def issue_node_command(self, *, node_id: str, action: str, model: str, size_bytes: int = 0, timeout_seconds: int = 900, context_tokens: int = 0) -> dict[str, Any]:
        if action in {"pull_model", "warm_model"} and self.model_catalog:
            spec = self.model_catalog.get(model)
            if spec is None:
                raise RuntimeError("model_not_in_catalog")
            size_bytes = spec.size_bytes
        elif action in {"pull_model", "warm_model"} and self.require_model_catalog:
            raise RuntimeError("model_catalog_required")
        secret = self.node_secrets.get(node_id)
        endpoint = self.node_agent_urls.get(node_id)
        if secret is None or endpoint is None:
            raise RuntimeError("node control is not configured for this node")
        command = {"action": action, "model": model, "size_bytes": int(size_bytes), "issued_at": time.time(), "timeout_seconds": int(timeout_seconds)}
        if action == "warm_model":
            command["context_tokens"] = int(context_tokens)
        signature = sign_command(command, secret)
        result = self._post(f"{endpoint.rstrip('/')}/v1/node/command", {"command": command, "signature": signature}, timeout=max(30, timeout_seconds + 15))
        return {"node_id": node_id, "command": {"action": action, "model": model}, "result": result}

    def evict_idle_models(self, *, max_items: int = 10, maintenance_approved: bool = False) -> dict[str, Any]:
        if maintenance_approved is not True:
            raise PermissionError("explicit maintenance approval is required")
        candidates = self.scheduler.plan_idle_evictions(max_items=max_items)
        results = []
        for candidate in candidates:
            try:
                result = self.issue_node_command(
                    node_id=candidate.node_id,
                    action="evict_model",
                    model=candidate.model,
                    timeout_seconds=120,
                )
                results.append({**candidate.to_dict(), "status": "completed", "result": result})
            except Exception as error:
                results.append({**candidate.to_dict(), "status": "failed", "error": error.__class__.__name__})
        return {"candidates": [candidate.to_dict() for candidate in candidates], "results": results}

    @staticmethod
    def _post_http(url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        request = Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())

    def _post_json(self, url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        return self._post_http(url, body, timeout=timeout)


def _snapshot_from_payload(payload: dict[str, Any]) -> NodeSnapshot:
    from .models import GpuSnapshot, LoadedModel

    return NodeSnapshot(
        node_id=str(payload["node_id"]),
        endpoint=str(payload["endpoint"]),
        tailnet_name=str(payload.get("tailnet_name") or ""),
        gpus=tuple(GpuSnapshot(**item) for item in payload.get("gpus") or []),
        loaded_models=tuple(LoadedModel(**item) for item in payload.get("loaded_models") or []),
        active_requests=int(payload.get("active_requests") or 0),
        queued_requests=int(payload.get("queued_requests") or 0),
        allowed_applications=tuple(payload.get("allowed_applications") or []),
        capabilities=tuple(payload.get("capabilities") or []),
        observed_at_epoch=float(payload.get("observed_at_epoch") or 0),
        healthy=bool(payload.get("healthy", True)),
        request_accounting_available=bool(payload.get("request_accounting_available", False)),
        inference_enabled=bool(payload.get("inference_enabled", True)),
        node_app_version=str(payload.get("node_app_version") or "")[:40],
        node_app_platform=str(payload.get("node_app_platform") or "")[:40],
        remote_updates_enabled=payload.get("remote_updates_enabled") is True,
        installed_models=tuple(str(model) for model in payload.get("installed_models") or []),
        installed_model_capabilities={str(model): tuple(capabilities) for model, capabilities in (payload.get("installed_model_capabilities") or {}).items()},
    )


def _request_from_payload(payload: dict[str, Any]) -> InferenceRequest:
    fields = {key: payload[key] for key in ("request_id", "application", "workload", "model", "model_size_bytes", "min_vram_mb")}
    fields.update({key: payload[key] for key in ("context_tokens", "context_vram_mb_per_1k", "priority", "max_latency_ms", "allow_model_pull", "allow_eviction") if key in payload})
    fields["required_capabilities"] = tuple(payload.get("required_capabilities") or ["chat"])
    return InferenceRequest(**fields)


def _broker_request_from_payload(metadata: dict[str, Any], operation: str) -> InferenceRequest:
    fields = {key: metadata[key] for key in ("request_id", "application", "model")}
    fields.update(workload=f"node_{operation}", model_size_bytes=0, min_vram_mb=0)
    for key, default, lower, upper in (("context_tokens", 0, 0, 1_000_000), ("priority", 50, 0, 100), ("max_latency_ms", 180000, 1, 900000)):
        value = metadata.get(key, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("invalid inference limits")
        fields[key] = value
    for key in ("allow_model_pull", "allow_eviction"):
        value = metadata.get(key, False)
        if not isinstance(value, bool):
            raise ValueError("invalid model preparation option")
        fields[key] = value
    capabilities = metadata.get("required_capabilities", [])
    if not isinstance(capabilities, list) or len(capabilities) > 32 or any(not isinstance(item, str) for item in capabilities):
        raise ValueError("invalid required capabilities")
    fields["required_capabilities"] = tuple(sorted(set(capabilities) | {"embeddings" if operation == "embeddings" else "chat"}))
    return InferenceRequest(**fields)


def _model_catalog_from_payload(payload: dict[str, Any]) -> dict[str, ModelSpec]:
    catalog: dict[str, ModelSpec] = {}
    for model, item in payload.items():
        if not isinstance(item, dict):
            raise ValueError(f"model catalog entry {model!r} must be an object")
        catalog[str(model)] = ModelSpec(
            size_bytes=int(item.get("size_bytes") or 0),
            min_vram_mb=int(item.get("min_vram_mb") or 0),
            capabilities=tuple(str(value) for value in (item.get("capabilities") or [])),
        )
    return catalog


def create_gateway_handler(gateway: InferenceGateway):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 2_000_000:
                raise ValueError("request body too large")
            return self.rfile.read(length)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "nodes": len(gateway.scheduler.snapshots())})
            elif self.path == '/v1/agent-link/availability':
                self._send(200, {'enabled': gateway.agent_link is not None})
            elif self.path == "/v1/node/enrollment-config":
                try:
                    self._send(200, gateway.enrollment_config())
                except RuntimeError as error:
                    self._send(503, {"detail": str(error)})
            elif self.path == "/v1/nodes":
                gateway.authorize_operator(self.headers.get("X-Gateway-Token"))
                self._send(200, {"nodes": [node.to_dict() for node in gateway.scheduler.snapshots()]})
            elif self.path.startswith('/v1/node-releases/'):
                try:
                    updates = gateway.agent_link.updates if gateway.agent_link else None
                    if updates is None: raise PermissionError('Updates unavailable')
                    _, artifact = updates.catalog.get(self.path.removeprefix('/v1/node-releases/'))
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(artifact.stat().st_size))
                    self.end_headers()
                    with artifact.open('rb') as stream:
                        while chunk := stream.read(65536): self.wfile.write(chunk)
                except (OSError, ValueError, PermissionError):
                    self._send(404, {'detail': 'Release unavailable'})
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                raw = self._body()
                payload = json.loads(raw.decode())
                if self.path in ('/v1/nodes/updates/poll', '/v1/nodes/updates/report'):
                    updates = gateway.agent_link.updates if gateway.agent_link else None
                    if updates is None:
                        self._send(503, {'detail': 'Remote updates are not enabled'})
                        return
                    if len(raw) > 16384: raise ValueError('Bounded update request required')
                    self._send(200, updates.node(self.path.rsplit('/', 1)[1], raw, self.headers.get('X-Node-Signature', '')))
                    return
                if self.path.startswith('/v1/agent-link/'):

                    if gateway.agent_link is None:
                        self._send(503, {'detail': 'Owner agent link is not enabled'})
                        return
                    if len(raw) > 65536:
                        raise ValueError('Owner request too large')
                    scheme, _, bearer = self.headers.get('Authorization', '').partition(' ')
                    if scheme.lower() != 'bearer' or not bearer:
                        raise AuthenticationError('Owner sign-in required')
                    result = gateway.agent_link.dispatch(self.path.removeprefix('/v1/agent-link/'), bearer, payload)
                    self._send(200, result)
                    return
                if self.path in {"/v1/nodes/inference", "/v1/applications/inference"}:
                    if self.path == "/v1/nodes/inference":
                        payload = gateway.authenticate_node_inference(raw, self.headers.get("X-Node-Signature") or "")
                    else:
                        metadata = payload.get("request")
                        if not isinstance(metadata, dict) or not isinstance(metadata.get("application"), str):
                            raise ValueError("application identity required")
                        gateway.authorize_application(application=metadata["application"], application_token=self.headers.get("X-Inference-Application-Token"))
                    metadata = payload["request"]
                    operation = payload.get("operation")
                    model_payload = payload.get("payload")
                    if operation not in {"chat", "generate", "embeddings"} or not isinstance(model_payload, dict):
                        raise ValueError("invalid inference operation or payload")
                    if "stream" in model_payload and not isinstance(model_payload["stream"], bool):
                        raise ValueError("stream must be a boolean")
                    if model_payload.get("model", metadata["model"]) != metadata["model"]:
                        raise ValueError("payload model does not match request")
                    request = _broker_request_from_payload(metadata, operation)
                    if model_payload.get("stream") is True:
                        with gateway.stream_infer(request, operation=operation, payload=model_payload) as (routing, stream):
                            with watch_disconnect(self.connection, stream.cancel):
                                relay_ndjson(self, stream, wrap=lambda event: {"routing": routing, "response": event}, on_error=lambda error: _stream_failure(request, routing))
                    else:
                        self._send(200, gateway.infer(request, operation=operation, payload=model_payload))
                    return
                if self.path == "/v1/nodes/heartbeat":
                    snapshot = _snapshot_from_payload(payload)
                    gateway.register_heartbeat(snapshot, raw, self.headers.get("X-Node-Signature") or "")
                    self._send(200, {"status": "accepted", "node_id": snapshot.node_id})
                    return
                if self.path == "/v1/nodes/agent-endpoint":
                    if len(raw) > 4096:
                        raise ValueError("bounded node listener refresh required")
                    result = gateway.refresh_node_agent_endpoint(raw, self.headers.get("X-Node-Signature") or "")
                    self._send(200, result)
                    return
                if self.path == "/v1/nodes/disconnect":
                    result = gateway.disconnect_node(
                        node_id=str(payload["node_id"]),
                        raw_body=raw,
                        signature=self.headers.get("X-Node-Signature") or "",
                    )
                    self._send(200, result)
                    return
                if self.path == "/v1/nodes/enroll":
                    result = gateway.enroll_node(
                        pairing_token=str(payload["pairing_token"]),
                        node_id=str(payload["node_id"]),
                        endpoint=str(payload["endpoint"]),
                        agent_url=str(payload["agent_url"]),
                        tailnet_name=str(payload.get("tailnet_name") or ""),
                    )
                    self._send(200, result)
                    return
                if self.path == "/v1/nodes/enroll/entra":
                    authorization = str(self.headers.get("Authorization") or "")
                    scheme, _, access_token = authorization.partition(" ")
                    if scheme.lower() != "bearer" or not access_token.strip():
                        raise PermissionError("Bearer Entra access token is required")
                    result = gateway.enroll_node_with_entra(
                        access_token=access_token.strip(),
                        node_id=str(payload["node_id"]),
                        endpoint=str(payload["endpoint"]),
                        agent_url=str(payload["agent_url"]),
                        tailnet_name=str(payload.get("tailnet_name") or ""),
                    )
                    self._send(200, result)
                    return
                if self.path == "/v1/nodes/pairing-token":
                    gateway.authorize_operator(self.headers.get("X-Gateway-Token"))
                    result = gateway.create_pairing_token(ttl_seconds=int(payload.get("ttl_seconds") or 600))
                    self._send(200, result)
                    return
                if self.path == "/v1/nodes/revoke":
                    gateway.authorize_operator(self.headers.get("X-Gateway-Token"))
                    result = gateway.revoke_node(str(payload["node_id"]))
                    self._send(200, result)
                    return
                if self.path == "/v1/inference":
                    request = _request_from_payload(payload["request"])
                    gateway.authorize_client(
                        self.headers.get("X-Gateway-Token"),
                        application=request.application,
                        application_token=self.headers.get("X-Inference-Application-Token"),
                    )
                    result = gateway.infer(request, operation=str(payload.get("operation") or "chat"), payload=dict(payload.get("payload") or {}))
                    self._send(200, result)
                    return
                if self.path == "/v1/node-command":
                    gateway.authorize_operator(self.headers.get("X-Gateway-Token"))
                    result = gateway.issue_node_command(node_id=str(payload["node_id"]), action=str(payload["action"]), model=str(payload["model"]), size_bytes=int(payload.get("size_bytes") or 0), timeout_seconds=int(payload.get("timeout_seconds") or 900))
                    self._send(200, result)
                    return
                if self.path == "/v1/maintenance/evict-idle":
                    gateway.authorize_operator(self.headers.get("X-Gateway-Token"))
                    result = gateway.evict_idle_models(
                        max_items=int(payload.get("max_items") or 10),
                        maintenance_approved=payload.get("maintenance_approved") is True,
                    )
                    self._send(200, result)
                    return
                self._send(404, {"detail": "not found"})
            except InferencePhaseError as error:
                _log_phase_failure(error)
                self._send(error.status, error.public())
            except AuthenticationError as error:
                self._send(401, {"detail": str(error)})
            except AuthorizationError as error:
                self._send(403, {"detail": str(error)})
            except PermissionError as error:
                self._send(403, {"detail": str(error)})
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self._send(400, {"detail": "invalid node inference request" if self.path in {"/v1/nodes/inference", "/v1/applications/inference"} else str(error)})
            except (RuntimeError, urllib.error.URLError) as error:
                self._send(503, {"detail": "node inference unavailable" if self.path in {"/v1/nodes/inference", "/v1/applications/inference"} else str(error)})
            except (OSError, TypeError) as error:
                self._send((400 if isinstance(error, TypeError) else 503), {"detail": "invalid request or upstream unavailable"})
            except Exception as error:
                self._send(500, {"detail": f"gateway failure: {error.__class__.__name__}"})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def run_gateway() -> None:
    from .preflight import validate_gateway_environment

    preflight = validate_gateway_environment(os.environ)
    if not preflight.ok:
        raise RuntimeError("gateway preflight failed: " + "; ".join(preflight.errors))
    secret_map = {}
    raw_secrets = os.getenv("INFERENCE_NODE_SECRETS_JSON", "{}")
    for node_id, secret in json.loads(raw_secrets).items():
        secret_map[str(node_id)] = str(secret).encode()
    agent_urls = {str(node_id): str(url) for node_id, url in json.loads(os.getenv("INFERENCE_NODE_AGENT_URLS_JSON", "{}")).items()}
    enrollment_path = os.getenv("INFERENCE_GATEWAY_STATE_PATH", "").strip() or None
    if enrollment_path is None:
        raise RuntimeError("INFERENCE_GATEWAY_STATE_PATH is required")
    bind = os.getenv("INFERENCE_GATEWAY_BIND", "127.0.0.1")
    gateway_token = os.environ["INFERENCE_GATEWAY_TOKEN"]
    model_catalog = _model_catalog_from_payload(json.loads(os.getenv("INFERENCE_MODEL_CATALOG_JSON", "{}")))
    if not model_catalog:
        raise RuntimeError("INFERENCE_MODEL_CATALOG_JSON must contain at least one model")
    application_tokens = {
        str(application): str(token)
        for application, token in json.loads(os.getenv("INFERENCE_APPLICATION_TOKENS_JSON", "{}")).items()
    }
    require_application_identity = os.getenv("INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY", "false").lower() == "true"
    if not require_application_identity:
        raise RuntimeError("INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY must be true")
    require_node_agents = os.getenv("INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS", "false").lower() == "true"
    if not require_node_agents:
        raise RuntimeError("INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS must be true")
    require_application_allowlist = os.getenv("INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST", "false").lower() == "true"
    if not require_application_allowlist:
        raise RuntimeError("INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST must be true")
    validate_gateway_runtime_security(
        gateway_token=gateway_token,
        bind=bind,
        allow_public_bind=os.getenv("INFERENCE_GATEWAY_ALLOW_PUBLIC_BIND", "false").lower() == "true",
    )
    require_entra_enrollment = os.getenv("INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT", "false").lower() == "true"
    entra_configured = bool(os.getenv("INFERENCE_ENTRA_TENANT_ID", "").strip() or os.getenv("INFERENCE_ENTRA_AUDIENCE", "").strip())
    entra_validator = EntraTokenValidator.from_env() if (require_entra_enrollment or entra_configured) else None
    entra_client_id = os.getenv("INFERENCE_ENTRA_CLIENT_ID", "").strip()
    entra_scope = os.getenv("INFERENCE_ENTRA_SCOPE", "").strip()
    if not entra_scope and os.getenv("INFERENCE_ENTRA_AUDIENCE", "").strip():
        entra_scope = os.getenv("INFERENCE_ENTRA_AUDIENCE", "").strip().rstrip("/") + "/.default"
    node_default_applications = tuple(
        item.strip()
        for item in os.getenv("INFERENCE_NODE_DEFAULT_APPLICATIONS", "manuals_rag,provis_employee_agent,openclaw").split(",")
        if item.strip()
    )
    gateway = InferenceGateway(
        scheduler=InferenceScheduler(
            policy=SchedulerPolicy(
                require_application_allowlist=require_application_allowlist,
                idle_model_ttl_seconds=max(0.0, float(os.getenv("INFERENCE_IDLE_MODEL_TTL_SECONDS", "900"))),
            ),
        ),
        node_secrets=secret_map,
        node_agent_urls=agent_urls,
        gateway_token=gateway_token,
        model_catalog=model_catalog,
        require_model_catalog=True,
        application_tokens=application_tokens,
        require_application_identity=require_application_identity,
        entra_validator=entra_validator,
        require_entra_enrollment=require_entra_enrollment,
        entra_client_id=entra_client_id,
        entra_scope=entra_scope,
        node_default_tailnet=os.getenv("INFERENCE_NODE_DEFAULT_TAILNET", "").strip(),
        node_default_applications=node_default_applications,
        node_default_port=int(os.getenv("INFERENCE_NODE_DEFAULT_PORT", "8765")),
        application_worker_allowlist=json.loads(os.getenv("INFERENCE_APPLICATION_WORKER_ALLOWLIST_JSON", "{}")),
        allow_node_agent_endpoint_refresh=os.getenv("INFERENCE_NODE_ALLOW_ENDPOINT_REFRESH", "false").lower() == "true",
        enrollment=EnrollmentManager(
            enrollment_path,
            allowed_host_suffixes=tuple(
                item.strip()
                for item in os.getenv("INFERENCE_NODE_ALLOWED_HOST_SUFFIXES", ".ts.net,.tailnet").split(",")
                if item.strip()
            ),
            allow_local_endpoints=os.getenv("INFERENCE_ENROLLMENT_ALLOW_LOCAL_ENDPOINTS", "false").lower() == "true",
        ),
        require_node_agents=require_node_agents,
    )
    owner_object_id = os.getenv('INFERENCE_AGENT_LINK_OWNER_OBJECT_ID', '').strip()
    if owner_object_id:
        from pathlib import Path
        from .agent_link import AgentLink
        if entra_validator is None:
            raise RuntimeError('Entra authentication required for owner agent link')
        gateway.agent_link = AgentLink(entra_validator, gateway.enrollment, owner_object_id=owner_object_id,
                                      path=Path(enrollment_path).parent / 'agent-link.sqlite3')
    release_dir = os.getenv('INFERENCE_NODE_UPDATE_RELEASE_DIR', '').strip()
    release_keys = os.getenv('INFERENCE_NODE_UPDATE_PUBLIC_KEYS_JSON', '').strip()
    if release_dir or release_keys:
        from .node_release import ReleaseCatalog
        from .node_update_queue import NodeUpdateQueue
        if not gateway.agent_link or not release_dir or not release_keys:
            raise RuntimeError('Owner identity, release directory and publisher keys required for updates')
        catalog = ReleaseCatalog(release_dir, json.loads(release_keys))
        catalog.list()  # refuse startup with unverified releases
        gateway.agent_link.updates = NodeUpdateQueue(catalog, gateway.enrollment, gateway.scheduler,
                                                   gateway.agent_link.db, gateway.agent_link.lock)
    server = ThreadingHTTPServer((bind, int(os.getenv("INFERENCE_GATEWAY_PORT", "8780"))), create_gateway_handler(gateway))
    server.serve_forever()
