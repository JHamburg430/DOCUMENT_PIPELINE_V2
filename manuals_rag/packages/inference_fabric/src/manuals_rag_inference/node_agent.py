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
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .control import ControlPolicy, verify_command
from .gateway import heartbeat_signature, node_request_signature
from .models import GpuSnapshot, LoadedModel, NodeSnapshot


TAILSCALE_IPV4_RANGE = ipaddress.ip_network("100.64.0.0/10")


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
        )

    def infer(self, envelope: dict, signature: str) -> dict:
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
        if not self.inference_enabled:
            self._record_activity(operation, model, "rejected", error="inference_disabled")
            raise PermissionError("inference is disabled on this node")
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
            _json_request(f"{self.endpoint}/api/pull", method="POST", body={"model": model, "stream": False}, timeout=float(command.get("timeout_seconds") or 900))
            return {"action": action, "model": model, "status": "completed"}
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


def create_handler(agent: OllamaNode, *, local_control_token: str = ""):
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
    server = ThreadingHTTPServer((bind, port), create_handler(agent, local_control_token=local_control_token))
    server.inference_stop_event = stop_event
    server.inference_gateway_url = gateway_url
    return server, agent, gateway_url


def run_node_agent() -> None:
    server, _agent, _gateway_url = create_node_server()
    try:
        server.serve_forever()
    finally:
        server.inference_stop_event.set()
        server.server_close()
