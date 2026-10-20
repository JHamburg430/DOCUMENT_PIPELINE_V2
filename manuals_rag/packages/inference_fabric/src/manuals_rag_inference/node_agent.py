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
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .control import ControlPolicy, verify_command
from .gateway import heartbeat_signature, node_request_signature
from .models import GpuSnapshot, LoadedModel, NodeSnapshot
from .onnx_backend import OnnxBackend
from .streaming import NDJSONResponse, relay_ndjson, watch_disconnect
from provis_inference_node.openai_broker import OpenAIBrokerMixin, create_openai_broker_handler, create_openai_broker_server


TAILSCALE_IPV4_RANGE = ipaddress.ip_network("100.64.0.0/10")
NODE_LISTENER_FALLBACK_PORTS = 8


def bind_node_listener(bind: str, port: int, handler, *, allow_fallback: bool = False,
                       excluded_ports=(), server_factory=None):
    """Bind only the configured port and a small contiguous fallback window."""
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("node listener port must be between 1 and 65535")
    if allow_fallback and port > 65528:
        raise ValueError("fallback requires an eight-port window below 65536")
    excluded_ports = set(excluded_ports)
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535
           for value in excluded_ports):
        raise ValueError("excluded listener ports must be valid TCP ports")
    factory = server_factory or ThreadingHTTPServer
    last_error = None
    attempts = NODE_LISTENER_FALLBACK_PORTS if allow_fallback else 1
    for candidate in range(port, min(65535, port + attempts - 1) + 1):
        if candidate in excluded_ports:
            continue
        try:
            return factory((bind, candidate), handler)
        except OSError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise OSError("no node listener ports available")


def node_agent_url_for_port(value: str, port: int) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/") or not 1 <= port <= 65535):
        raise ValueError("valid enrolled node listener URL required")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    return urlunsplit((parsed.scheme, f"{host}:{port}", "", "", ""))


def listener_endpoint_needs_refresh(agent_url: str, listener_port: int) -> bool:
    """Whether the gateway-advertised port differs from the bound listener."""
    try:
        parsed = urlsplit(agent_url)
        advertised_port = parsed.port
    except ValueError as error:
        raise ValueError("valid advertised node listener URL required") from error
    if advertised_port is None:
        advertised_port = 443 if parsed.scheme == "https" else 80
    return advertised_port != listener_port


def publish_agent_endpoint(gateway_url: str, node_id: str, secret: bytes, agent_url: str) -> dict:
    gateway = urlsplit(gateway_url)
    if (gateway.scheme != "https" or not gateway.hostname or gateway.username or gateway.password
            or gateway.query or gateway.fragment):
        raise ValueError("listener refresh requires the configured HTTPS gateway")
    payload = {"node_id": node_id, "agent_url": agent_url, "issued_at": time.time(), "nonce": secrets.token_urlsafe(24)}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    request = Request(f"{gateway_url.rstrip('/')}/v1/nodes/agent-endpoint", data=raw, method="POST",
                      headers={"Content-Type": "application/json", "X-Node-Signature": heartbeat_signature(raw, secret)})
    with urlopen(request, timeout=5) as response:
        body = response.read(4097)
    if len(body) > 4096:
        raise ValueError("bounded node listener response required")
    result = json.loads(body.decode())
    if result.get("status") != "accepted" or result.get("node_id") != node_id:
        raise PermissionError("gateway rejected node listener refresh")
    return result


class ModelPreparationError(RuntimeError):
    def __init__(self, category: str, upstream_status: int | None = None):
        super().__init__(category)
        self.category = category
        self.upstream_status = upstream_status


def _model_error_category(message: str) -> str:
    text = message.lower()
    for phrase, category in (
        ("newer version", "ollama_upgrade_required"),
        ("no space left", "insufficient_disk_space"),
        ("certificate", "registry_tls_error"),
        ("not found", "registry_model_not_found"),
        ("timed out", "registry_timeout"),
        ("deadline exceeded", "registry_timeout"),
        ("connection refused", "registry_connection_failed"),
        ("manifest", "registry_manifest_failed"),
    ):
        if phrase in text:
            return category
    return "ollama_model_pull_failed"


def _installed_model_names(payload: dict) -> tuple[str, ...]:
    return tuple(str(item.get("name") or item.get("model")) for item in payload.get("models") or [] if item.get("name") or item.get("model"))


def _json_request(url: str, *, method: str = "GET", body: dict | None = None, timeout: float = 3.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


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
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    snapshots = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) not in {6, 7}:
            continue
        try:
            if len(fields) == 7:
                index, uuid, name, total, used, utilization, temperature = fields
                snapshots.append(GpuSnapshot(uuid, name, int(total), int(used), float(utilization), float(temperature), gpu_uuid=uuid, gpu_index=int(index), backend="cuda", memory_unknown=False))
            else:
                snapshots.append(GpuSnapshot(fields[0], fields[1], int(fields[2]), int(fields[3]), float(fields[4]), float(fields[5])))
        except ValueError:
            continue
    return tuple(snapshots)


class OllamaNode(OpenAIBrokerMixin):
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
        self._updating = False
        self._active_commands = 0
        self.app_metadata = {}
        self.heartbeat_accepted = threading.Event()
        self._activity: deque[dict[str, object]] = deque(maxlen=200)
        self._probe_health: dict[str, tuple[str, str]] = {}
        self._inventory_lock = threading.Lock()
        self._model_metadata: dict[str, tuple[str, tuple[str, ...]]] = {}
        self.onnx_backend = OnnxBackend.from_env()

    def _inventory(self) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        payload = _json_request(f"{self.endpoint}/api/tags")
        installed = _installed_model_names(payload)
        deadline = time.monotonic() + 2
        with self._inventory_lock:
            for item in payload.get("models") or []:
                model = str(item.get("name") or item.get("model") or "")
                digest = str(item.get("digest") or "")
                if not model or (model in self._model_metadata and self._model_metadata[model][0] == digest):
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    metadata = _json_request(f"{self.endpoint}/api/show", method="POST", body={"model": model}, timeout=remaining)
                    capabilities = tuple("embeddings" if value == "embedding" else value for value in metadata.get("capabilities", []) if value in {"chat", "completion", "embedding", "embeddings", "vision", "tools", "thinking", "insert"})
                    if "completion" in capabilities and "chat" not in capabilities:
                        capabilities += ("chat",)
                    self._model_metadata[model] = (digest, capabilities)
                except Exception:
                    pass
            onnx_models = self.onnx_backend.model_names()
            installed = tuple(sorted(set(installed) | set(onnx_models)))
            capabilities = {model: self._model_metadata.get(model, ("", ()))[1] for model in installed if model not in onnx_models}
            capabilities.update({model: self.onnx_backend.capabilities(model) for model in onnx_models})
            return installed, capabilities

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
            if enabled and self._updating: raise PermissionError("App update is draining active work")
            self._inference_enabled = bool(enabled)
        self._record_activity("control", "", "enabled" if enabled else "disabled")

    def begin_update(self):
        with self._request_lock:
            self._updating = True
            self._inference_enabled = False

    def cancel_update(self, previous):
        with self._request_lock:
            self._updating = False
            self._inference_enabled = previous

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
        ollama_healthy = True
        try:
            payload = _json_request(f"{self.endpoint}/api/ps")
            for item in payload.get("models") or []:
                model = str(item.get("name") or item.get("model") or "")
                with self._request_lock:
                    active_requests = self._active_by_model.get(model, 0)
                    last_used = self._last_used_by_model.get(model, 0.0)
                gpu_ids = item.get("gpu_ids") or ((item.get("gpu_id"),) if item.get("gpu_id") else ())
                vram_by_gpu = item.get("vram_by_gpu_mb") or {}
                loaded.append(LoadedModel(
                    model,
                    int(item.get("size_vram") or item.get("size") or 0),
                    tuple(str(value) for value in gpu_ids),
                    last_used,
                    active_requests,
                    {str(key): int(value) for key, value in vram_by_gpu.items()},
                    bool(item.get("placement_known") or vram_by_gpu),
                ))
        except Exception as error:
            ollama_healthy = False
            self._record_probe("ollama", "failed", error.__class__.__name__)
        if ollama_healthy:
            self._record_probe("ollama", "healthy")
        try:
            installed_models, model_capabilities = self._inventory()
            self._record_probe("model_inventory", "available")
        except Exception as error:
            installed_models = ()
            model_capabilities = {}
            self._record_probe("model_inventory", "unavailable", error.__class__.__name__)
        gpus = list(_gpu_snapshots())
        for device in self.onnx_backend.gpu_devices():
            gpus.append(GpuSnapshot(device, device.removeprefix("openvino:"), 0, 0, 0.0, None, gpu_uuid=device, gpu_index=None, backend="openvino", memory_unknown=True))
        gpus = tuple(gpus)
        self._record_probe("gpu", "detected" if gpus else "unavailable", "" if gpus else "gpu_probe_unavailable")
        if self.onnx_backend.enabled:
            self._record_probe("onnx_runtime", "available")
        elif self.onnx_backend.manifest:
            self._record_probe("onnx_runtime", "unavailable", "onnx_runtime_unavailable")
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
            **self.app_metadata,
            installed_models=installed_models,
            installed_model_capabilities=model_capabilities,
            capabilities=("chat", "embeddings", "onnx") if self.onnx_backend.enabled else ("chat", "embeddings"),
            onnx_providers=self.onnx_backend.providers,
            gpu_placement_mode=os.getenv("INFERENCE_NODE_GPU_PLACEMENT_MODE", "multi_gpu"),
        )

    def _authorize_inference(self, envelope: dict, signature: str) -> tuple[str, dict, str]:
        unsigned = {
            "operation": envelope.get("operation"),
            "payload": envelope.get("payload"),
            "request_id": envelope.get("request_id"),
            "issued_at": envelope.get("issued_at"),
        }
        # Keep verification compatible with older gateways that did not sign
        # a placement field. New gateways always include it, including None.
        if "placement" in envelope:
            unsigned["placement"] = envelope.get("placement")
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
        if operation not in {"chat", "generate", "embeddings", "onnx"}:
            raise PermissionError("unsupported node inference operation")
        payload = unsigned.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("node inference payload must be an object")
        placement = unsigned.get("placement")
        if placement is not None:
            if not isinstance(placement, dict):
                raise ValueError("node inference placement must be an object")
            mode = str(placement.get("mode") or "")
            backend = str(placement.get("backend") or "")
            gpu_ids = placement.get("gpu_ids")
            if mode not in {"single_gpu", "multi_gpu"} or backend not in {"ollama", "onnx"}:
                raise ValueError("invalid node inference placement")
            if not isinstance(gpu_ids, list) or not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
                raise ValueError("node inference placement must identify unique GPUs")
            if operation == "onnx" and backend != "onnx":
                raise ValueError("ONNX inference requires an ONNX placement")
            if operation != "onnx" and backend != "ollama":
                raise ValueError("Ollama inference requires an Ollama placement")
        model = str(payload.get("model") or "")
        if not model:
            raise ValueError("node inference model is required")
        if not self.inference_enabled:
            self._record_activity(operation, model, "rejected", error="inference_disabled")
            raise PermissionError("inference is disabled on this node")
        return operation, payload, model

    def infer(self, envelope: dict, signature: str) -> dict:
        operation, payload, model = self._authorize_inference(envelope, signature)
        ollama_operation = {"embeddings": "embed"}.get(operation, operation)
        started_at = time.time()
        self._record_activity(operation, model, "started")
        with self._request_lock:
            if self._updating: raise PermissionError("App update is draining active work")
            self._active_by_model[model] = self._active_by_model.get(model, 0) + 1
            self._last_used_by_model[model] = time.time()
        try:
            if operation == "onnx":
                response = self.onnx_backend.infer(model, payload)
            else:
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

    @contextmanager
    def stream_infer(self, envelope: dict, signature: str):
        operation, payload, model = self._authorize_inference(envelope, signature)
        if operation not in {"chat", "generate"}:
            raise ValueError("embeddings cannot stream")
        started_at = time.time()
        stream = None
        self._record_activity(operation, model, "started")
        with self._request_lock:
            if self._updating: raise PermissionError("App update is draining active work")
            self._active_by_model[model] = self._active_by_model.get(model, 0) + 1
            self._last_used_by_model[model] = time.time()
        try:
            stream = NDJSONResponse(f"{self.endpoint}/api/{operation}", {**payload, "stream": True}, timeout=900)
            yield stream
            self._record_activity(operation, model, "completed" if stream.done else "cancelled", started_at=started_at)
        except Exception as error:
            self._record_activity(operation, model, "failed", started_at=started_at, error=type(error).__name__)
            raise
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                with self._request_lock:
                    remaining = self._active_by_model.get(model, 1) - 1
                    if remaining > 0:
                        self._active_by_model[model] = remaining
                    else:
                        self._active_by_model.pop(model, None)

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
        with self._request_lock:
            if self._updating: raise PermissionError('App update is draining active work')
            self._active_commands += 1
        try: return self._command(command, signature)
        finally:
            with self._request_lock: self._active_commands -= 1

    def _command(self, command: dict, signature: str) -> dict:
        if not verify_command(command, signature, self.secret, policy=self.policy):
            raise PermissionError("invalid or unauthorized node command")
        action = command["action"]
        model = command["model"]
        if action == "pull_model":
            started_at = time.time()
            self._record_activity("pull_model", model, "started")
            try:
                try:
                    self._record_activity("model_inventory_check", model, "started")
                    installed = _installed_model_names(_json_request(f"{self.endpoint}/api/tags"))
                except Exception:
                    installed = ()
                if model in installed:
                    self._record_activity("model_already_installed", model, "completed", started_at=started_at)
                    return {"action": action, "model": model, "status": "completed", "already_installed": True}
                self._record_activity("model_pull_started", model, "started")
                result = _json_request(f"{self.endpoint}/api/pull", method="POST", body={"model": model, "stream": False}, timeout=min(900, max(1, float(command.get("timeout_seconds") or 900))))
                if result.get("error"):
                    raise ModelPreparationError(_model_error_category(str(result["error"])))
                if result.get("status") != "success":
                    raise ModelPreparationError("ollama_pull_completion_unconfirmed")
                self._record_activity("pull_model", model, "completed", started_at=started_at)
                return {"action": action, "model": model, "status": "completed"}
            except HTTPError as error:
                try:
                    message = str(json.loads(error.read(8192)).get("error", ""))
                except Exception:
                    message = ""
                category = _model_error_category(message)
                self._record_activity("model_pull_failed", model, "failed", started_at=started_at, error=f"HTTP_{error.code}:{category}")
                raise ModelPreparationError(category, error.code) from None
            except Exception as error:
                self._record_activity("model_pull_failed", model, "failed", started_at=started_at, error=error.category if isinstance(error, ModelPreparationError) else type(error).__name__)
                raise
        if action == "warm_model":
            started_at = time.time()
            deadline = time.monotonic() + min(900, max(1, float(command.get("timeout_seconds") or 120)))
            self._record_activity("model_warm_started", model, "started")
            try:
                self._record_activity("model_inventory_check", model, "started")
                installed = _installed_model_names(_json_request(f"{self.endpoint}/api/tags", timeout=min(3, max(.001, deadline - time.monotonic()))))
                if model not in installed:
                    raise ModelPreparationError("model_not_installed")
                context_tokens = int(command.get("context_tokens") or 4096)
                if not 512 <= context_tokens <= 1_000_000:
                    raise ModelPreparationError("invalid_warm_context")
                reply = _json_request(f"{self.endpoint}/api/generate", method="POST", body={"model": model, "prompt": "", "stream": False, "keep_alive": 120, "options": {"num_predict": 0, "num_ctx": context_tokens}}, timeout=max(.001, deadline - time.monotonic()))
                if reply.get("error"):
                    raise ModelPreparationError(_model_error_category(str(reply["error"])))
                while time.monotonic() < deadline:
                    loaded = _installed_model_names(_json_request(f"{self.endpoint}/api/ps", timeout=min(3, max(.001, deadline - time.monotonic()))))
                    if model in loaded:
                        self._record_activity("model_warm_completed", model, "completed", started_at=started_at)
                        return {"action": action, "model": model, "status": "completed", "loaded": True}
                    time.sleep(min(.2, max(0, deadline - time.monotonic())))
                raise ModelPreparationError("model_readiness_unconfirmed")
            except HTTPError as error:
                self._record_activity("model_warm_failed", model, "failed", started_at=started_at, error=f"HTTP_{error.code}")
                raise ModelPreparationError(f"ollama_http_{error.code}", error.code) from None
            except Exception as error:
                self._record_activity("model_warm_failed", model, "failed", started_at=started_at, error=error.category if isinstance(error, ModelPreparationError) else type(error).__name__)
                raise
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
        self.heartbeat_accepted.set()
        self._record_activity("heartbeat", "", "accepted", started_at=started_at)
        return result


def create_handler(
    agent: OllamaNode,
    *,
    local_control_token: str = "",
    gateway_url: str = "",
    local_broker_only: bool = False,
    local_api_token: str = "",
    require_local_api_token: bool = False,
    default_application: str = "manuals_rag",
    advertised_models: tuple[str, ...] = (),
):
    if local_broker_only:
        return create_openai_broker_handler(
            agent, gateway_url=gateway_url,
            local_api_token=local_api_token, require_local_api_token=require_local_api_token,
            default_application=default_application, advertised_models=advertised_models,
        )
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
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
            if self.path == "/v1/node/inference":
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                    if length > 2_000_000:
                        raise ValueError("inference body too large")
                    payload = json.loads(self.rfile.read(length).decode())
                    signature = str(payload.pop("signature"))
                    if isinstance(payload.get("payload"), dict) and payload["payload"].get("stream") is True:
                        with agent.stream_infer(payload, signature) as stream:
                            with watch_disconnect(self.connection, stream.cancel):
                                relay_ndjson(self, stream)
                    else:
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
            except ModelPreparationError as error:
                self._send(502, {"detail": "model preparation failed", "code": error.category, "upstream_status": error.upstream_status})
            except PermissionError as error:
                self._send(403, {"detail": str(error)})
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self._send(400, {"detail": str(error)})
            except Exception as error:
                self._send(502, {"detail": f"ollama command failed: {error.__class__.__name__}"})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def create_node_server(config_path: str | None = None, *, app_metadata=None, startup_enabled=True) -> tuple[ThreadingHTTPServer, OllamaNode, str]:
    configured_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", "").strip()
    if configured_path:
        config_path = configured_path
    elif config_path:
        config_path = config_path
    else:
        provis_path = Path.home() / ".config/ProVIS/inference-node.json"
        legacy_path = Path.home() / ".config/manuals-rag/inference-node.json"
        config_path = str(legacy_path if legacy_path.exists() and not provis_path.exists() else provis_path)
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
    agent.app_metadata = dict(app_metadata or {})
    agent.set_inference_enabled(startup_enabled)
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
    # The bounded fallback is automatic for current app builds. Operators can
    # disable it explicitly for a host with INFERENCE_NODE_ALLOW_PORT_FALLBACK=false.
    allow_port_fallback = os.getenv("INFERENCE_NODE_ALLOW_PORT_FALLBACK", "true").strip().lower() == "true"
    broker_port = int(os.getenv("INFERENCE_NODE_LOCAL_BROKER_PORT", str(config.get("local_broker_port") or "8767")))
    if broker_port and not 1 <= broker_port <= 65535:
        raise RuntimeError("local broker port must be a valid TCP port")
    server = bind_node_listener(bind, port, create_handler(agent, local_control_token=local_control_token),
                                allow_fallback=allow_port_fallback,
                                excluded_ports={broker_port} if broker_port else ())
    # The persisted advertised URL may point at a previous fallback port even
    # though this restart successfully reclaimed the configured primary port.
    # Compare the listener against the advertised endpoint, not just the
    # configured bind port, so the gateway is refreshed in both directions.
    try:
        endpoint_changed = listener_endpoint_needs_refresh(agent_url, server.server_port)
    except ValueError as error:
        server.server_close()
        raise RuntimeError("invalid advertised node listener URL") from error
    if endpoint_changed:
        if not gateway_url:
            server.server_close()
            raise RuntimeError("gateway is required to publish a fallback listener endpoint")
        refreshed_url = node_agent_url_for_port(agent_url, server.server_port)
        try:
            publish_agent_endpoint(gateway_url, node_id, shared_secret.encode(), refreshed_url)
        except Exception:
            server.server_close()
            raise
        config["agent_url"] = refreshed_url
        _save_node_config(config_path, config)
    broker_server = None
    if broker_port:
        if broker_port == server.server_port:
            server.server_close()
            raise RuntimeError("local broker port must be a valid port distinct from the node control listener")
        broker_bind = os.getenv("INFERENCE_NODE_LOCAL_BROKER_BIND", "127.0.0.1").strip()
        try:
            default_application = str(
                os.getenv("INFERENCE_NODE_DEFAULT_APPLICATION")
                or config.get("default_application")
                or (allowed_applications[0] if allowed_applications else "manuals_rag")
            ).strip()
            local_api_token = os.getenv("INFERENCE_NODE_LOCAL_API_TOKEN", "")
            require_local_api_token = os.getenv("INFERENCE_NODE_REQUIRE_LOCAL_API_TOKEN", "false").lower() == "true"
            broker_server = create_openai_broker_server(
                agent, gateway_url=gateway_url, port=broker_port, bind=broker_bind,
                local_api_token=local_api_token, require_local_api_token=require_local_api_token,
                default_application=default_application,
                advertised_models=tuple(item.strip() for item in os.getenv("INFERENCE_NODE_ADVERTISED_MODELS", "").split(",") if item.strip()),
            )
            config["local_broker_port"] = broker_port
            config["default_application"] = default_application
            _save_node_config(config_path, config)
        except Exception:
            server.server_close()
            stop_event.set()
            raise
    server.local_broker_server = broker_server
    server.local_broker_thread = None
    server.inference_stop_event = stop_event
    server.inference_gateway_url = gateway_url
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
    return server, agent, gateway_url


def run_node_agent() -> None:
    server, _agent, _gateway_url = create_node_server()
    broker_server = getattr(server, "local_broker_server", None)
    broker_thread = None
    try:
        if broker_server is not None:
            broker_thread = threading.Thread(target=broker_server.serve_forever, daemon=True, name="provis-inference-local-broker")
            broker_thread.start()
            server.local_broker_thread = broker_thread
        server.serve_forever()
    finally:
        server.inference_stop_event.set()
        if broker_server is not None:
            if broker_thread is not None and broker_thread.is_alive():
                broker_server.shutdown()
            broker_server.server_close()
            if broker_thread is not None:
                broker_thread.join(timeout=5)
        server.server_close()
