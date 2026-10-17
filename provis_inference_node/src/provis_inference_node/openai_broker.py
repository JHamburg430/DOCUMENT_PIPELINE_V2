"""Shared, loopback-only OpenAI compatibility layer for PROVIS node apps.

The node owns the enrolled identity and signs requests to the central inference
gateway. Local applications never receive gateway or node credentials.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .gateway import heartbeat_signature


SUPPORTED_GET_PATHS = frozenset({"/v1/models", "/models"})
SUPPORTED_POST_PATHS = frozenset({
    "/v1/chat/completions", "/chat/completions",
    "/v1/completions", "/completions",
    "/v1/embeddings", "/embeddings",
})


def local_broker_networks() -> tuple[ipaddress._BaseNetwork, ...]:
    """Only accept local loopback clients; no Docker/LAN CIDRs by default."""
    return (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))


def local_broker_peer_allowed(source: str, networks: tuple[ipaddress._BaseNetwork, ...] | None = None) -> bool:
    try:
        address = ipaddress.ip_address(source)
    except ValueError:
        return False
    return any(address in network for network in (networks or local_broker_networks()))


class OpenAIBrokerMixin:
    """Canonical OpenAI request translation and enrolled gateway dispatch."""

    def broker_infer(self, envelope: dict, gateway_url: str) -> dict:
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
        if isinstance(payload.get("options"), dict):
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
        response = self.broker_stream(
            {"request": request, "operation": operation, "payload": upstream}, gateway_url,
        )
        return response, model, operation

    def openai_infer(self, payload: dict, *, operation: str, application: str, gateway_url: str) -> dict:
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
        if isinstance(payload.get("options"), dict):
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
            {"request": request, "operation": operation, "payload": upstream}, gateway_url,
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


def create_openai_broker_handler(
    agent,
    *,
    gateway_url: str,
    local_api_token: str = "",
    require_local_api_token: bool = False,
    default_application: str = "manuals_rag",
    advertised_models: tuple[str, ...] = (),
):
    """Build the isolated local API handler; only the node holds gateway auth."""
    networks = local_broker_networks()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorize_local(self) -> None:
            if not local_broker_peer_allowed(self.client_address[0], networks):
                raise PermissionError("local broker access is restricted")
            supplied_application = str(
                self.headers.get("X-ProVIS-Application")
                or self.headers.get("X-Inference-Application")
                or default_application
            ).strip()
            if supplied_application != default_application:
                raise PermissionError("application identity is fixed by node configuration")
            if default_application not in agent.allowed_applications:
                raise PermissionError("application is not allowed on this node")
            if require_local_api_token:
                authorization = str(self.headers.get("Authorization") or "")
                scheme, _, supplied = authorization.partition(" ")
                if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, local_api_token):
                    raise PermissionError("local inference API authentication required")

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
            except Exception:
                # Headers have already been sent. Keep upstream errors private.
                try:
                    payload = {"error": {"message": "inference stream failed", "type": "gateway_error"}}
                    self.wfile.write(f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path not in SUPPORTED_GET_PATHS:
                self._send(404, {"detail": "not found"})
                return
            try:
                self._authorize_local()
                self._send(200, {
                    "object": "list",
                    "data": [{"id": model, "object": "model", "owned_by": "provis-inference-node"} for model in advertised_models],
                })
            except PermissionError as error:
                self._send(403, {"detail": str(error)})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in SUPPORTED_POST_PATHS:
                self._send(404, {"detail": "not found"})
                return
            try:
                self._authorize_local()
                length = int(self.headers.get("Content-Length") or "0")
                if length <= 0 or length > 2_000_000:
                    raise ValueError("inference body size is invalid")
                payload = json.loads(self.rfile.read(length).decode())
                if not isinstance(payload, dict):
                    raise ValueError("inference request must be a JSON object")
                if self.path.endswith("chat/completions"):
                    operation = "chat"
                elif self.path.endswith("/completions"):
                    operation = "generate"
                else:
                    operation = "embeddings"
                if operation == "embeddings" and payload.get("stream") is True:
                    raise ValueError("embeddings do not support streaming")
                if payload.get("stream") is True:
                    response, model, stream_operation = agent.openai_stream(
                        payload, operation=operation, application=default_application, gateway_url=gateway_url,
                    )
                    try:
                        self._send_openai_stream(response, model, stream_operation)
                    finally:
                        response.close()
                    return
                self._send(200, agent.openai_infer(
                    payload, operation=operation, application=default_application, gateway_url=gateway_url,
                ))
            except PermissionError as error:
                self._send(403, {"detail": str(error)})
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self._send(400, {"detail": str(error)})
            except HTTPError as error:
                self._send(error.code if 400 <= error.code < 600 else 502, {"detail": f"gateway broker rejected request ({error.code})"})
            except Exception as error:
                self._send(502, {"detail": f"gateway broker failed: {error.__class__.__name__}"})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def create_openai_broker_server(
    agent,
    *,
    gateway_url: str,
    port: int,
    bind: str = "127.0.0.1",
    local_api_token: str = "",
    require_local_api_token: bool = False,
    default_application: str = "manuals_rag",
    advertised_models: tuple[str, ...] = (),
) -> ThreadingHTTPServer:
    """Create a broker server with an enforced loopback-only bind."""
    if not gateway_url.strip():
        raise ValueError("local broker requires the enrolled central gateway URL")
    try:
        address = ipaddress.ip_address(bind)
    except ValueError as error:
        raise ValueError("local broker bind must be a loopback address") from error
    if not address.is_loopback:
        raise ValueError("local broker bind must be a loopback address")
    if not 0 <= int(port) <= 65535:
        raise ValueError("local broker port must be between 0 and 65535")
    if default_application not in agent.allowed_applications:
        raise ValueError("local broker application must be in the enrolled node allowlist")
    if require_local_api_token and not local_api_token:
        raise ValueError("local broker API token is required by configuration")
    server = ThreadingHTTPServer(
        (str(address), int(port)),
        create_openai_broker_handler(
            agent,
            gateway_url=gateway_url,
            local_api_token=local_api_token,
            require_local_api_token=require_local_api_token,
            default_application=default_application,
            advertised_models=advertised_models,
        ),
    )
    server.daemon_threads = True
    return server
