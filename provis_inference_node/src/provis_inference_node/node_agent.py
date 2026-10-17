"""Small Ollama node-agent transport using only the Python standard library."""

from __future__ import annotations

import json
import hmac
import ipaddress
import os
import secrets
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .control import ControlPolicy, verify_command
from .gateway import heartbeat_signature, node_request_signature
from .models import GpuSnapshot, LoadedModel, NodeSnapshot


TAILSCALE_IPV4_RANGE = ipaddress.ip_network("100.64.0.0/10")


def _json_request(url: str, *, method: str = "GET", body: dict | None = None, timeout: float = 3.0, headers: dict[str, str] | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = Request(url, data=data, method=method, headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _safe_ollama_failure(error: BaseException, phase: str) -> str:
    """Return a phase/status diagnostic without exposing Ollama response bodies."""
    if isinstance(error, HTTPError):
        return f"{phase}_failed:ollama_http_{error.code}"
    return f"{phase}_failed:{error.__class__.__name__}"


def _model_names(payload: dict) -> tuple[str, ...]:
    names: list[str] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def enroll_node(*, gateway_url: str, pairing_token: str, node_id: str, endpoint: str, agent_url: str, tailnet_name: str) -> dict:
    return _json_request(
        f"{gateway_url.rstrip('/')}/v1/nodes/enroll",
        method="POST",
        body={
            "pairing_token": pairing_token,
            "node_id": node_id,
            "endpoint": endpoint,
            "agent_url": agent_url,
            "tailnet_name": tailnet_name,
        },
        timeout=10,
    )


def _load_node_config(path: str) -> dict:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _save_node_config(path: str, config: dict) -> None:
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, sort_keys=True))
    config_path.chmod(0o600)


def _csv_values(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _local_broker_networks(value: object = None) -> tuple[ipaddress._BaseNetwork, ...]:
    raw = value if value is not None else os.getenv(
        "INFERENCE_NODE_LOCAL_BROKER_CIDRS",
        "127.0.0.0/8,172.16.0.0/12,::1/128",
    )
    networks = []
    for item in _csv_values(raw):
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            continue
    return tuple(networks)


def _local_broker_allowed(source: str, networks: tuple[ipaddress._BaseNetwork, ...]) -> bool:
    try:
        address = ipaddress.ip_address(source)
    except ValueError:
        return False
    return any(address in network for network in networks)


def _detect_tailscale_ipv4() -> str | None:
    """Return a local Tailscale IPv4 address without accepting public output."""
    try:
        completed = subprocess.run(
            ["tailscale", "ip", "-4"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for value in completed.stdout.split():
        try:
            address = ipaddress.ip_address(value.strip())
        except ValueError:
            continue
        if address.version == 4 and address in TAILSCALE_IPV4_RANGE:
            return str(address)
    return None


def _gpu_snapshots() -> tuple[GpuSnapshot, ...]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    snapshots = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            snapshots.append(GpuSnapshot(fields[0], fields[1], int(fields[2]), int(fields[3]), float(fields[4]), float(fields[5])))
        except ValueError:
            continue
    return tuple(snapshots)


class OllamaNode:
    def __init__(self, *, node_id: str, endpoint: str, tailnet_name: str, secret: bytes, policy: ControlPolicy | None = None, proxy_only: bool = False, allowed_applications: tuple[str, ...] = ()) -> None:
        self.node_id = node_id
        self.endpoint = endpoint.rstrip("/")
        self.tailnet_name = tailnet_name
        self.secret = secret
        self.policy = policy or ControlPolicy()
        self.proxy_only = proxy_only
        self.allowed_applications = tuple(item for item in allowed_applications if item)
        self._request_lock = threading.Lock()
        self._active_by_model: dict[str, int] = {}
        self._last_used_by_model: dict[str, float] = {}
        self._seen_inference_requests: dict[str, float] = {}
        self._inference_enabled = True
        self._activity: deque[dict[str, object]] = deque(maxlen=200)
        self._probe_health: dict[str, tuple[str, str]] = {}

    def _record_probe(self, operation: str, status: str, error: str = "") -> None:
        state = (status, error)
        with self._request_lock:
            if self._probe_health.get(operation) == state:
                return
            self._probe_health[operation] = state
        self._record_activity(operation, "", status, error=error)

    @property
    def inference_enabled(self) -> bool:
        with self._request_lock:
            return self._inference_enabled

    def set_inference_enabled(self, enabled: bool) -> None:
        with self._request_lock:
            self._inference_enabled = bool(enabled)
        self._record_activity("control", "", "enabled" if enabled else "disabled")

    def activity(self) -> list[dict[str, object]]:
        with self._request_lock:
            return list(self._activity)

    def _record_activity(self, operation: str, model: str, status: str, *, started_at: float | None = None, error: str = "") -> None:
        event: dict[str, object] = {
            "timestamp": time.time(),
            "operation": operation,
            "model": model,
            "status": status,
        }
        if started_at is not None:
            event["duration_ms"] = round(max(0.0, time.time() - started_at) * 1000, 1)
        if error:
            event["error"] = error
        with self._request_lock:
            self._activity.append(event)

    def snapshot(self) -> NodeSnapshot:
        loaded: list[LoadedModel] = []
        installed_models: tuple[str, ...] = ()
        ollama_healthy = True
        try:
            payload = _json_request(f"{self.endpoint}/api/ps")
            for item in payload.get("models") or []:
                model = str(item.get("name") or item.get("model") or "")
                with self._request_lock:
                    active_requests = self._active_by_model.get(model, 0)
                    last_used = self._last_used_by_model.get(model, 0.0)
                loaded.append(LoadedModel(model, int(item.get("size_vram") or item.get("size") or 0), tuple(str(item.get("gpu_id")) for _ in [0] if item.get("gpu_id")), last_used, active_requests))
        except Exception as error:
            ollama_healthy = False
            self._record_probe("ollama", "failed", error.__class__.__name__)
        if ollama_healthy:
            self._record_probe("ollama", "healthy")
        try:
            installed_models = self.installed_models(timeout=5.0)
            self._record_probe("ollama_model_inventory", "available")
        except Exception as error:
            self._record_probe("ollama_model_inventory", "failed", error.__class__.__name__)
        gpus = _gpu_snapshots()
        self._record_probe("gpu", "detected" if gpus else "unavailable", "" if gpus else "gpu_probe_unavailable")
        with self._request_lock:
            active_requests = sum(self._active_by_model.values())
        return NodeSnapshot(
            self.node_id,
            self.endpoint,
            self.tailnet_name,
            gpus,
            tuple(loaded),
            active_requests=active_requests,
            allowed_applications=self.allowed_applications,
            observed_at_epoch=time.time(),
            request_accounting_available=self.proxy_only,
            healthy=ollama_healthy,
            inference_enabled=self.inference_enabled,
            installed_models=installed_models,
        )

    def installed_models(self, *, timeout: float = 5.0) -> tuple[str, ...]:
        """Return the local Ollama inventory without triggering a model pull."""
        return _model_names(_json_request(f"{self.endpoint}/api/tags", timeout=timeout))

    def _warm_model(self, model: str, *, timeout: float) -> None:
        """Load a model and verify that Ollama reports it as ready."""
        _json_request(
            f"{self.endpoint}/api/generate",
            method="POST",
            body={
                "model": model,
                "prompt": " ",
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        )
        loaded = set(_model_names(_json_request(f"{self.endpoint}/api/ps", timeout=min(10.0, timeout))))
        if model not in loaded:
            raise RuntimeError("model_not_ready")

    def _prepare_model(self, model: str, *, timeout: float) -> dict:
        started_at = time.time()
        self._record_activity("model_prepare", model, "started")
        source = "installed"
        phase = "model_inventory"
        try:
            installed = set(self.installed_models(timeout=min(10.0, timeout)))
            if model not in installed:
                source = "pulled"
                phase = "model_pull"
                self._record_activity("model_pull", model, "started")
                _json_request(
                    f"{self.endpoint}/api/pull",
                    method="POST",
                    body={"model": model, "stream": False},
                    timeout=timeout,
                )
                self._record_activity("model_pull", model, "completed")
            else:
                self._record_activity("model_pull", model, "skipped", error="already_installed")
            phase = "model_warm"
            self._record_activity("model_warm", model, "started")
            self._warm_model(model, timeout=timeout)
            self._record_activity("model_warm", model, "completed")
            self._record_activity("model_prepare", model, "completed", started_at=started_at)
            return {"status": "completed", "phase": "ready", "source": source}
        except Exception as error:
            diagnostic = _safe_ollama_failure(error, phase)
            self._record_activity("model_prepare", model, "failed", started_at=started_at, error=diagnostic)
            raise RuntimeError(diagnostic) from None

    def _prepare_inference(self, envelope: dict, signature: str, *, allow_stream: bool = False):
        unsigned = {
            "operation": envelope.get("operation"),
            "payload": envelope.get("payload"),
            "request_id": envelope.get("request_id"),
            "issued_at": envelope.get("issued_at"),
        }
        if not hmac.compare_digest(node_request_signature(unsigned, self.secret), signature):
            raise PermissionError("invalid node inference signature")
        request_id = str(unsigned.get("request_id") or "")
        issued_at = float(unsigned.get("issued_at") or 0)
        if not request_id or abs(time.time() - issued_at) > self.policy.max_clock_skew_seconds:
            raise PermissionError("stale or missing node inference timestamp")
        with self._request_lock:
            cutoff = time.time() - self.policy.max_clock_skew_seconds
            self._seen_inference_requests = {
                key: value for key, value in self._seen_inference_requests.items() if value >= cutoff
            }
            if request_id in self._seen_inference_requests:
                raise PermissionError("duplicate node inference request")
            self._seen_inference_requests[request_id] = issued_at
        operation = str(unsigned.get("operation") or "")
        if operation not in {"chat", "generate", "embeddings"}:
            raise PermissionError("unsupported node inference operation")
        payload = unsigned.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("node inference payload must be an object")
        model = str(payload.get("model") or "")
        if not model:
            raise ValueError("node inference model is required")
        if payload.get("stream") is True and not allow_stream:
            raise ValueError("streaming requires the streaming node path")
        if not self.inference_enabled:
            self._record_activity(operation, model, "rejected", error="inference_disabled")
            raise PermissionError("inference is disabled on this node")
        return unsigned, operation, payload, model

    def infer(self, envelope: dict, signature: str) -> dict:
        _, operation, payload, model = self._prepare_inference(envelope, signature)
        ollama_operation = {"embeddings": "embed"}.get(operation, operation)
        started_at = time.time()
        self._record_activity(operation, model, "started")
        with self._request_lock:
            self._active_by_model[model] = self._active_by_model.get(model, 0) + 1
            self._last_used_by_model[model] = time.time()
        try:
            response = _json_request(
                f"{self.endpoint}/api/{ollama_operation}",
                method="POST",
                body=payload,
                timeout=900.0,
            )
            self._record_activity(operation, model, "completed", started_at=started_at)
            return response
        except Exception as error:
            self._record_activity(operation, model, "failed", started_at=started_at, error=error.__class__.__name__)
            raise
        finally:
            with self._request_lock:
                remaining = self._active_by_model.get(model, 1) - 1
                if remaining > 0:
                    self._active_by_model[model] = remaining
                else:
                    self._active_by_model.pop(model, None)

    def _finish_stream(self, operation: str, model: str, started_at: float, *, error: str = "") -> None:
        self._record_activity(operation, model, "failed" if error else "completed", started_at=started_at, error=error)
        with self._request_lock:
            remaining = self._active_by_model.get(model, 1) - 1
            if remaining > 0:
                self._active_by_model[model] = remaining
            else:
                self._active_by_model.pop(model, None)

    def infer_stream(self, envelope: dict, signature: str):
        """Open a real Ollama NDJSON response for the signed node protocol."""
        _, operation, payload, model = self._prepare_inference(envelope, signature, allow_stream=True)
        if operation not in {"chat", "generate"} or payload.get("stream") is not True:
            raise ValueError("streaming is only supported for chat and generate")
        started_at = time.time()
        self._record_activity(operation, model, "started")
        with self._request_lock:
            self._active_by_model[model] = self._active_by_model.get(model, 0) + 1
            self._last_used_by_model[model] = time.time()
        try:
            ollama_operation = {"embeddings": "embed"}.get(operation, operation)
            request = Request(
                f"{self.endpoint}/api/{ollama_operation}",
                data=json.dumps(payload).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            response = urlopen(request, timeout=900.0)
            if response.headers.get_content_type() != "application/x-ndjson":
                response.close()
                raise RuntimeError("Ollama did not return an NDJSON stream")
            return response, operation, model, started_at
        except Exception as error:
            self._finish_stream(operation, model, started_at, error=error.__class__.__name__)
            raise

    def broker_infer(self, envelope: dict, gateway_url: str) -> dict:
        """Forward a local application request through this node's gateway identity."""
        if not gateway_url.strip():
            raise RuntimeError("node gateway is not configured")
        request = envelope.get("request")
        payload = envelope.get("payload")
        operation = str(envelope.get("operation") or "")
        if not isinstance(request, dict) or not isinstance(payload, dict):
            raise ValueError("broker inference requires request and payload objects")
        application = str(request.get("application") or "")
        if application not in self.allowed_applications:
            raise PermissionError("application is not allowed on this node")
        if operation not in {"chat", "generate", "embeddings"}:
            raise ValueError("unsupported inference operation")
        body = {
            "node_id": self.node_id,
            "issued_at": time.time(),
            "request": request,
            "operation": operation,
            "payload": payload,
        }
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        request_obj = Request(
            f"{gateway_url.rstrip('/')}/v1/nodes/inference",
            data=raw,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Node-Signature": heartbeat_signature(raw, self.secret),
            },
        )
        with urlopen(request_obj, timeout=900.0) as response:
            return json.loads(response.read().decode())

    def broker_stream(self, envelope: dict, gateway_url: str):
        """Forward a broker request and return the unbuffered NDJSON response."""
        if not gateway_url.strip():
            raise RuntimeError("node gateway is not configured")
        request = envelope.get("request")
        payload = envelope.get("payload")
        operation = str(envelope.get("operation") or "")
        if not isinstance(request, dict) or not isinstance(payload, dict):
            raise ValueError("broker inference requires request and payload objects")
        application = str(request.get("application") or "")
        if application not in self.allowed_applications:
            raise PermissionError("application is not allowed on this node")
        if operation not in {"chat", "generate"} or payload.get("stream") is not True:
            raise ValueError("streaming is only supported for chat and generate")
        body = {
            "node_id": self.node_id,
            "issued_at": time.time(),
            "request": request,
            "operation": operation,
            "payload": payload,
        }
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        request_obj = Request(
            f"{gateway_url.rstrip('/')}/v1/nodes/inference",
            data=raw,
            method="POST",
            headers={"Content-Type": "application/json", "X-Node-Signature": heartbeat_signature(raw, self.secret)},
        )
        response = urlopen(request_obj, timeout=900.0)
        if response.headers.get_content_type() != "application/x-ndjson":
            response.close()
            raise RuntimeError("gateway did not return an NDJSON stream")
        return response

    def openai_stream(self, payload: dict, *, operation: str, application: str, gateway_url: str):
        """Translate a local OpenAI stream into a broker stream."""
        model = str(payload.get("model") or "")
        if not model:
            raise ValueError("model is required")
        options = {
            key: payload[key]
            for key in ("temperature", "top_p", "seed", "num_ctx", "num_batch")
            if payload.get(key) is not None
        }
        if payload.get("max_tokens") is not None:
            options["num_predict"] = payload["max_tokens"]
        if payload.get("options") and isinstance(payload["options"], dict):
            options.update(payload["options"])
        if operation == "chat":
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            upstream = {"model": model, "messages": messages, "stream": True}
            if payload.get("think") is not None:
                upstream["think"] = payload["think"]
        elif operation == "generate":
            prompt = payload.get("prompt")
            if not isinstance(prompt, (str, list)):
                raise ValueError("prompt is required")
            if isinstance(prompt, list):
                prompt = "\n".join(str(item) for item in prompt)
            upstream = {"model": model, "prompt": prompt, "stream": True}
            if payload.get("think") is not None:
                upstream["think"] = payload["think"]
        else:
            raise ValueError("streaming is only supported for chat and generate")
        if options:
            upstream["options"] = options
        request = {
            "request_id": f"openai-{secrets.token_hex(12)}",
            "application": application,
            "workload": f"openai_{operation}",
            "model": model,
            "context_tokens": int(options.get("num_ctx") or 0),
            "required_capabilities": ["chat"],
            "max_latency_ms": int(payload.get("timeout_ms") or 900000),
            "allow_model_pull": payload.get("allow_model_pull") is True,
            "allow_eviction": payload.get("allow_eviction") is True,
        }
        response = self.broker_stream({"request": request, "operation": operation, "payload": upstream}, gateway_url)
        return response, model, operation

    def openai_infer(self, payload: dict, *, operation: str, application: str, gateway_url: str) -> dict:
        """Translate a local OpenAI-compatible request to the broker protocol."""
        model = str(payload.get("model") or "")
        if not model:
            raise ValueError("model is required")
        if payload.get("stream") is True:
            raise ValueError("streaming requests must use the streaming node handler")
        options = {
            key: payload[key]
            for key in ("temperature", "top_p", "seed", "num_ctx", "num_batch")
            if payload.get(key) is not None
        }
        if payload.get("max_tokens") is not None:
            options["num_predict"] = payload["max_tokens"]
        if payload.get("options") and isinstance(payload["options"], dict):
            options.update(payload["options"])
        if operation == "chat":
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            upstream = {"model": model, "messages": messages, "stream": False}
            if payload.get("think") is not None:
                upstream["think"] = payload["think"]
            if options:
                upstream["options"] = options
        elif operation == "generate":
            prompt = payload.get("prompt")
            if not isinstance(prompt, (str, list)):
                raise ValueError("prompt is required")
            if isinstance(prompt, list):
                prompt = "\n".join(str(item) for item in prompt)
            upstream = {"model": model, "prompt": prompt, "stream": False}
            if payload.get("think") is not None:
                upstream["think"] = payload["think"]
            if options:
                upstream["options"] = options
        elif operation == "embeddings":
            inputs = payload.get("input")
            if isinstance(inputs, str):
                inputs = [inputs]
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("input must be a non-empty string or list")
            upstream = {"model": model, "input": inputs}
        else:
            raise ValueError("unsupported OpenAI operation")
        request = {
            "request_id": f"openai-{secrets.token_hex(12)}",
            "application": application,
            "workload": f"openai_{operation}",
            "model": model,
            "context_tokens": int(options.get("num_ctx") or 0),
            "required_capabilities": ["embeddings"] if operation == "embeddings" else ["chat"],
            "max_latency_ms": int(payload.get("timeout_ms") or 900000),
            "allow_model_pull": payload.get("allow_model_pull") is True,
            "allow_eviction": payload.get("allow_eviction") is True,
        }
        result = self.broker_infer(
            {"request": request, "operation": operation, "payload": upstream},
            gateway_url,
        )
        response = dict(result.get("response") or {})
        routing = result.get("routing") or {}
        if operation == "embeddings":
            vectors = response.get("embeddings") or []
            return {
                "object": "list",
                "data": [{"object": "embedding", "embedding": vector, "index": index} for index, vector in enumerate(vectors)],
                "model": model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "_inference_routing": routing,
            }
        if operation == "generate":
            text = str(response.get("response") or "")
            return {
                "id": f"cmpl-{secrets.token_hex(12)}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "_inference_routing": routing,
            }
        message = dict(response.get("message") or {})
        content = str(message.get("content") or response.get("response") or "")
        return {
            "id": f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": str(message.get("role") or "assistant"), "content": content},
                "finish_reason": "stop",
            }],
            "_inference_routing": routing,
        }

    def disconnect(self, gateway_url: str) -> dict:
        payload = {"node_id": self.node_id, "issued_at": time.time()}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        request = Request(
            f"{gateway_url.rstrip('/')}/v1/nodes/disconnect",
            data=raw,
            method="POST",
            headers={"Content-Type": "application/json", "X-Node-Signature": heartbeat_signature(raw, self.secret)},
        )
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode())
        self._record_activity("control", "", "disconnected")
        return result

    def command(self, command: dict, signature: str) -> dict:
        if not verify_command(command, signature, self.secret, policy=self.policy):
            raise PermissionError("invalid or unauthorized node command")
        action = command["action"]
        model = command["model"]
        if action == "pull_model":
            result = self._prepare_model(model, timeout=float(command.get("timeout_seconds") or 900))
            return {"action": action, "model": model, **result}
        if action == "evict_model":
            with self._request_lock:
                if self._active_by_model.get(model, 0) > 0:
                    raise PermissionError("cannot evict a model with active requests")
            _json_request(f"{self.endpoint}/api/generate", method="POST", body={"model": model, "prompt": "", "keep_alive": 0, "stream": False}, timeout=30)
            return {"action": action, "model": model, "status": "completed"}
        raise PermissionError("unsupported node command")

    def publish_heartbeat(self, gateway_url: str) -> dict:
        started_at = time.time()
        payload = json.dumps(asdict(self.snapshot()), sort_keys=True, separators=(",", ":")).encode()
        request = Request(
            f"{gateway_url.rstrip('/')}/v1/nodes/heartbeat",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "X-Node-Signature": heartbeat_signature(payload, self.secret)},
        )
        try:
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode())
        except Exception as error:
            category = f"HTTP_{error.code}" if isinstance(error, HTTPError) else error.__class__.__name__
            self._record_activity("heartbeat", "", "failed", started_at=started_at, error=category)
            raise
        self._record_activity("heartbeat", "", "accepted", started_at=started_at)
        return result


def create_handler(
    agent: OllamaNode,
    *,
    local_control_token: str = "",
    gateway_url: str = "",
    local_broker_only: bool = False,
    local_broker_networks: tuple[ipaddress._BaseNetwork, ...] = (),
    local_api_token: str = "",
    require_local_api_token: bool = False,
    default_application: str = "manuals_rag",
):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_openai_stream(self, response, model: str, operation: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for line in response:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("invalid gateway stream event")
                    if event.get("error"):
                        payload = {"error": {"message": "inference stream failed", "type": "gateway_error"}}
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                        self.wfile.flush()
                        return
                    data = event.get("response", {})
                    if not isinstance(data, dict):
                        raise ValueError("invalid gateway stream response")
                    done = data.get("done") is True
                    if operation == "chat":
                        message = data.get("message") or {}
                        delta = {} if done else {key: str(message[key]) for key in ("role", "content") if message.get(key) is not None}
                        chunk = {
                            "id": f"chatcmpl-{secrets.token_hex(8)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if done else None}],
                        }
                    else:
                        chunk = {
                            "id": f"cmpl-{secrets.token_hex(8)}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{"index": 0, "text": "" if done else str(data.get("response") or ""), "finish_reason": "stop" if done else None}],
                        }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    if done:
                        break
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def do_GET(self) -> None:  # noqa: N802
            if local_broker_only:
                if self.path in {"/v1/models", "/models"}:
                    try:
                        if not _local_broker_allowed(self.client_address[0], local_broker_networks):
                            raise PermissionError("local broker access is restricted")
                        models = _csv_values(os.getenv("INFERENCE_NODE_ADVERTISED_MODELS", ""))
                        self._send(200, {
                            "object": "list",
                            "data": [{"id": model, "object": "model", "owned_by": "provis-inference-node"} for model in models],
                        })
                    except PermissionError as error:
                        self._send(403, {"detail": str(error)})
                    return
                self._send(404, {"detail": "not found"})
                return
            if self.path == "/health":
                self._send(200, {"status": "ok", "node_id": agent.node_id})
            elif self.path == "/v1/node/snapshot":
                self._send(200, asdict(agent.snapshot()))
            elif self.path == "/v1/node/status":
                self._send(200, {
                    "node_id": agent.node_id,
                    "inference_enabled": agent.inference_enabled,
                    "activity_count": len(agent.activity()),
                })
            elif self.path == "/v1/node/activity":
                self._send(200, {"events": agent.activity()})
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if local_broker_only and self.path in {"/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/v1/embeddings", "/embeddings"}:
                try:
                    if not _local_broker_allowed(self.client_address[0], local_broker_networks):
                        raise PermissionError("local broker access is restricted")
                    if require_local_api_token:
                        authorization = str(self.headers.get("Authorization") or "")
                        scheme, _, supplied = authorization.partition(" ")
                        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, local_api_token):
                            raise PermissionError("local inference API authentication required")
                    length = int(self.headers.get("Content-Length") or "0")
                    if length > 2_000_000:
                        raise ValueError("inference body too large")
                    payload = json.loads(self.rfile.read(length).decode())
                    application = str(
                        self.headers.get("X-ProVIS-Application")
                        or self.headers.get("X-Inference-Application")
                        or default_application
                    ).strip()
                    if self.path.endswith("chat/completions"):
                        operation = "chat"
                    elif self.path.endswith("/completions"):
                        operation = "generate"
                    else:
                        operation = "embeddings"
                    if payload.get("stream") is True:
                        response, model, stream_operation = agent.openai_stream(
                            payload, operation=operation, application=application, gateway_url=gateway_url,
                        )
                        try:
                            self._send_openai_stream(response, model, stream_operation)
                        finally:
                            response.close()
                        return
                    self._send(200, agent.openai_infer(payload, operation=operation, application=application, gateway_url=gateway_url))
                except PermissionError as error:
                    self._send(403, {"detail": str(error)})
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    self._send(400, {"detail": str(error)})
                except HTTPError as error:
                    self._send(error.code if 400 <= error.code < 600 else 502, {"detail": f"gateway broker rejected request ({error.code})"})
                except Exception as error:
                    self._send(502, {"detail": f"gateway broker failed: {error.__class__.__name__}"})
                return
            if self.path == "/v1/node/broker/inference":
                try:
                    if not _local_broker_allowed(self.client_address[0], local_broker_networks):
                        raise PermissionError("local broker access is restricted")
                    length = int(self.headers.get("Content-Length") or "0")
                    if length > 2_000_000:
                        raise ValueError("inference body too large")
                    payload = json.loads(self.rfile.read(length).decode())
                    self._send(200, agent.broker_infer(payload, gateway_url))
                except PermissionError as error:
                    self._send(403, {"detail": str(error)})
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    self._send(400, {"detail": str(error)})
                except Exception as error:
                    self._send(502, {"detail": f"gateway broker failed: {error.__class__.__name__}"})
                return
            if local_broker_only:
                self._send(404, {"detail": "not found"})
                return
            if self.path == "/v1/node/inference":
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                    if length > 2_000_000:
                        raise ValueError("inference body too large")
                    payload = json.loads(self.rfile.read(length).decode())
                    signature = str(payload.pop("signature"))
                    if isinstance(payload.get("payload"), dict) and payload["payload"].get("stream") is True:
                        response, operation, model, started_at = agent.infer_stream(payload, signature)
                        try:
                            self.send_response(200)
                            self.send_header("Content-Type", "application/x-ndjson")
                            self.send_header("Connection", "close")
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            self.close_connection = True
                            for line in response:
                                self.wfile.write(line)
                                self.wfile.flush()
                            agent._finish_stream(operation, model, started_at)
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            agent._finish_stream(operation, model, started_at, error="client_disconnect")
                        except Exception as error:
                            agent._finish_stream(operation, model, started_at, error=error.__class__.__name__)
                            try:
                                self.wfile.write(b'{"error":{"code":"upstream_stream_failed"},"done":true}\n')
                                self.wfile.flush()
                            except OSError:
                                pass
                        finally:
                            response.close()
                        return
                    self._send(200, agent.infer(payload, signature))
                except PermissionError as error:
                    self._send(403, {"detail": str(error)})
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    self._send(400, {"detail": str(error)})
                except Exception as error:
                    self._send(502, {"detail": f"ollama inference failed: {error.__class__.__name__}"})
                return
            if self.path == "/v1/node/settings":
                try:
                    source = self.client_address[0]
                    supplied = self.headers.get("X-Node-Local-Token") or ""
                    if source not in {"127.0.0.1", "::1"} and (not local_control_token or not hmac.compare_digest(local_control_token, supplied)):
                        raise PermissionError("local node control is restricted")
                    length = int(self.headers.get("Content-Length") or "0")
                    if length > 8192:
                        raise ValueError("settings body too large")
                    payload = json.loads(self.rfile.read(length).decode())
                    action = str(payload.get("action") or "")
                    if action not in {"enable_inference", "disable_inference"}:
                        raise ValueError("unsupported node setting")
                    agent.set_inference_enabled(action == "enable_inference")
                    self._send(200, {"node_id": agent.node_id, "inference_enabled": agent.inference_enabled})
                except PermissionError as error:
                    self._send(403, {"detail": str(error)})
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    self._send(400, {"detail": str(error)})
                return
            if self.path != "/v1/node/command":
                self._send(404, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length > 8192:
                    raise ValueError("command body too large")
                payload = json.loads(self.rfile.read(length).decode())
                self._send(200, agent.command(payload["command"], str(payload["signature"])))
            except PermissionError as error:
                self._send(403, {"detail": str(error)})
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self._send(400, {"detail": str(error)})
            except Exception as error:
                self._send(502, {"detail": f"ollama command failed: {error.__class__.__name__}"})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def create_node_server(config_path: str | None = None) -> tuple[ThreadingHTTPServer, OllamaNode, str]:
    configured_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", "").strip()
    if configured_path:
        config_path = configured_path
    elif config_path:
        config_path = config_path
    else:
        config_home = os.getenv("XDG_CONFIG_HOME", "").strip()
        provis_path = (Path(config_home).expanduser() if config_home else Path.home() / ".config") / "ProVIS" / "inference-node.json"
        config_path = str(provis_path)
    config = _load_node_config(config_path)
    node_id = os.getenv("INFERENCE_NODE_ID") or str(config.get("node_id") or f"node-{secrets.token_hex(6)}")
    endpoint = os.getenv("OLLAMA_NODE_URL") or str(config.get("endpoint") or "http://127.0.0.1:11434")
    tailnet_name = os.getenv("TAILNET_NAME") or str(config.get("tailnet_name") or "local")
    port = int(os.getenv("INFERENCE_NODE_PORT") or config.get("port") or "8765")
    tailscale_ipv4 = _detect_tailscale_ipv4()
    agent_url = os.getenv("INFERENCE_NODE_ADVERTISE_URL") or str(config.get("agent_url") or "")
    if not agent_url and tailscale_ipv4:
        agent_url = f"http://{tailscale_ipv4}:{port}"
    shared_secret = os.getenv("INFERENCE_NODE_SHARED_SECRET") or str(config.get("shared_secret") or "")
    if not shared_secret:
        gateway_url = os.getenv("INFERENCE_GATEWAY_URL", "").strip()
        pairing_token = os.getenv("INFERENCE_GATEWAY_ENROLLMENT_TOKEN", "").strip()
        if not gateway_url or not pairing_token or not agent_url:
            raise RuntimeError("node enrollment requires INFERENCE_GATEWAY_URL, INFERENCE_GATEWAY_ENROLLMENT_TOKEN, and a Tailscale-advertised node URL")
        config = enroll_node(
            gateway_url=gateway_url,
            pairing_token=pairing_token,
            node_id=node_id,
            endpoint=endpoint,
            agent_url=agent_url,
            tailnet_name=tailnet_name,
        )
        config["gateway_url"] = gateway_url
        _save_node_config(config_path, config)
        shared_secret = str(config["shared_secret"])
    gateway_url = os.getenv("INFERENCE_GATEWAY_URL", "").strip() or str(config.get("gateway_url") or "")
    configured_applications_env = os.getenv("INFERENCE_NODE_ALLOWED_APPLICATIONS", "").strip()
    configured_applications = config.get("allowed_applications")
    if configured_applications_env:
        allowed_applications = tuple(item.strip() for item in configured_applications_env.split(",") if item.strip())
    elif isinstance(configured_applications, (list, tuple)):
        allowed_applications = tuple(str(item).strip() for item in configured_applications if str(item).strip())
    else:
        allowed_applications = tuple(
            item.strip()
            for item in (os.getenv("INFERENCE_NODE_ALLOWED_APPLICATIONS") or str(configured_applications or "")).split(",")
            if item.strip()
        )
    configured_prefixes = _csv_values(
        os.getenv("INFERENCE_NODE_ALLOWED_MODEL_PREFIXES")
        or config.get("allowed_model_prefixes")
        or ControlPolicy().allowed_model_prefixes
    )
    try:
        max_model_size_bytes = int(os.getenv("INFERENCE_NODE_MAX_MODEL_SIZE_BYTES") or config.get("max_model_size_bytes") or ControlPolicy().max_model_size_bytes)
        max_clock_skew_seconds = int(os.getenv("INFERENCE_NODE_MAX_CLOCK_SKEW_SECONDS") or config.get("max_clock_skew_seconds") or ControlPolicy().max_clock_skew_seconds)
    except (TypeError, ValueError) as error:
        raise RuntimeError("node control policy limits must be integers") from error
    agent = OllamaNode(
        node_id=node_id,
        endpoint=endpoint,
        tailnet_name=tailnet_name,
        secret=shared_secret.encode(),
        policy=ControlPolicy(
            allowed_model_prefixes=configured_prefixes,
            max_model_size_bytes=max_model_size_bytes,
            max_clock_skew_seconds=max_clock_skew_seconds,
        ),
        proxy_only=os.getenv("INFERENCE_NODE_PROXY_ONLY", "false").lower() == "true",
        allowed_applications=allowed_applications,
    )
    bind = os.getenv("INFERENCE_NODE_BIND") or str(config.get("bind") or tailscale_ipv4 or "127.0.0.1")
    if os.getenv("INFERENCE_NODE_REQUIRE_PRODUCTION", "false").lower() == "true":
        from .preflight import validate_node_environment

        startup_env = dict(os.environ)
        startup_env.update({
            "INFERENCE_NODE_ID": node_id,
            "INFERENCE_GATEWAY_URL": gateway_url,
            "INFERENCE_NODE_ADVERTISE_URL": agent_url,
            "INFERENCE_NODE_BIND": bind,
            "INFERENCE_NODE_PORT": str(port),
            "OLLAMA_NODE_URL": endpoint,
            "INFERENCE_NODE_ALLOWED_APPLICATIONS": ",".join(allowed_applications),
        })
        result = validate_node_environment(startup_env)
        if not result.ok:
            raise RuntimeError("node preflight failed: " + "; ".join(result.errors))
    local_control_token = os.getenv("INFERENCE_NODE_LOCAL_CONTROL_TOKEN") or str(config.get("local_control_token") or secrets.token_urlsafe(32))
    config["local_control_token"] = local_control_token
    _save_node_config(config_path, config)
    stop_event = threading.Event()
    if gateway_url:
        interval = max(2.0, float(os.getenv("INFERENCE_HEARTBEAT_INTERVAL_SECONDS", "5")))

        def heartbeat_loop() -> None:
            while not stop_event.is_set():
                try:
                    agent.publish_heartbeat(gateway_url)
                except Exception:
                    pass
                stop_event.wait(interval)

        threading.Thread(target=heartbeat_loop, daemon=True, name="inference-heartbeat").start()
    broker_port = int(os.getenv("INFERENCE_NODE_LOCAL_BROKER_PORT", "8767"))
    broker_bind = os.getenv("INFERENCE_NODE_LOCAL_BROKER_BIND", "0.0.0.0")
    broker_networks = _local_broker_networks()
    local_api_token = os.getenv("INFERENCE_NODE_LOCAL_API_TOKEN") or str(config.get("local_api_token") or secrets.token_urlsafe(32))
    require_local_api_token = os.getenv("INFERENCE_NODE_REQUIRE_LOCAL_API_TOKEN", "false").lower() == "true"
    default_application = os.getenv("INFERENCE_NODE_DEFAULT_APPLICATION") or str(config.get("default_application") or (allowed_applications[0] if allowed_applications else "manuals_rag"))
    config["local_broker_port"] = broker_port
    config["local_api_token"] = local_api_token
    config["default_application"] = default_application
    _save_node_config(config_path, config)
    server = ThreadingHTTPServer((bind, port), create_handler(agent, local_control_token=local_control_token))
    local_broker_server = None
    if broker_port > 0 and broker_port != port:
        local_broker_server = ThreadingHTTPServer(
            (broker_bind, broker_port),
            create_handler(
                agent,
                gateway_url=gateway_url,
                local_broker_only=True,
                local_broker_networks=broker_networks,
                local_api_token=local_api_token,
                require_local_api_token=require_local_api_token,
                default_application=default_application,
            ),
        )
    server.local_broker_server = local_broker_server
    server.inference_stop_event = stop_event
    server.inference_gateway_url = gateway_url
    return server, agent, gateway_url


def run_node_agent() -> None:
    server, _agent, _gateway_url = create_node_server()
    local_broker_server = getattr(server, "local_broker_server", None)
    if local_broker_server is not None:
        threading.Thread(target=local_broker_server.serve_forever, daemon=True, name="inference-local-broker").start()
    try:
        server.serve_forever()
    finally:
        server.inference_stop_event.set()
        if local_broker_server is not None:
            local_broker_server.shutdown()
            local_broker_server.server_close()
        server.server_close()
