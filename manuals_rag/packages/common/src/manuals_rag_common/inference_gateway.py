"""Opt-in compatibility client for the central inference gateway."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx


def enabled() -> bool:
    return bool(os.getenv("INFERENCE_NODE_BROKER_URL", "").strip() or os.getenv("INFERENCE_GATEWAY_URL", "").strip())


def node_broker_enabled() -> bool:
    return bool(os.getenv("INFERENCE_NODE_BROKER_URL", "").strip())


def fallback_enabled() -> bool:
    """Return whether local Ollama may be used when the gateway is unavailable.

    This is explicit rather than implicit: production callers can keep the
    central policy boundary fail-closed, while local RAG deployments can opt
    into availability fallback.
    """
    return os.getenv("INFERENCE_GATEWAY_FALLBACK_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def should_fallback(error: BaseException) -> bool:
    """Allow fallback only for gateway availability failures, never auth/config errors."""
    if not fallback_enabled():
        return False
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {502, 503, 504}
    return False


def _catalog(model: str) -> dict[str, Any]:
    raw = os.getenv("INFERENCE_MODEL_CATALOG_JSON", "")
    if not raw:
        raise RuntimeError("INFERENCE_MODEL_CATALOG_JSON is required when the inference gateway is enabled")
    try:
        catalog = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("INFERENCE_MODEL_CATALOG_JSON is invalid") from error
    item = catalog.get(model)
    if not isinstance(item, dict):
        raise RuntimeError(f"model {model!r} is missing from INFERENCE_MODEL_CATALOG_JSON")
    return item


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("INFERENCE_GATEWAY_TOKEN")
    if token:
        headers["X-Gateway-Token"] = token
    application_token = os.getenv("INFERENCE_APPLICATION_TOKEN")
    if application_token:
        headers["X-Inference-Application-Token"] = application_token
    return headers


def _node_broker_call(
    *,
    model: str,
    operation: str,
    payload: dict[str, Any],
    purpose: str | None,
    timeout: float,
    context_tokens: int = 0,
    context_vram_mb_per_1k: float = 0.0,
    required_capabilities: list[str] | None = None,
    priority: int = 50,
    allow_model_pull: bool = False,
    allow_eviction: bool = False,
) -> dict[str, Any]:
    """Submit through a locally connected node; the node authenticates to the gateway."""
    request = {
        "request_id": str(uuid.uuid4()),
        "application": os.getenv("INFERENCE_APPLICATION", "manuals_rag"),
        "workload": os.getenv("INFERENCE_WORKLOAD", purpose or "interactive"),
        "model": model,
        "context_tokens": int(context_tokens or 0),
        "context_vram_mb_per_1k": float(context_vram_mb_per_1k or 0.0),
        "required_capabilities": list(required_capabilities or ["chat"]),
        "priority": int(priority),
        "max_latency_ms": int(timeout * 1000),
        "allow_model_pull": bool(allow_model_pull),
        "allow_eviction": bool(allow_eviction),
    }
    base_url = os.environ["INFERENCE_NODE_BROKER_URL"].rstrip("/")
    with httpx.Client(base_url=base_url, timeout=max(timeout, 30.0)) as client:
        response = client.post(
            "/v1/node/broker/inference",
            json={"request": request, "operation": operation, "payload": payload},
        )
        response.raise_for_status()
        result = response.json()
    body = dict(result.get("response") or {})
    body["_inference_routing"] = result.get("routing") or {}
    body["_inference_transport"] = "local_node_broker"
    return body


def chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    purpose: str | None,
    json_schema: dict[str, Any] | None = None,
    think: bool | None = None,
    timeout: float = 90.0,
    keep_alive: str | int | None = "10m",
    temperature: float | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
    num_ctx: int | None = None,
    num_batch: int | None = None,
) -> dict[str, Any]:
    if node_broker_enabled():
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": keep_alive,
                "think": think,
            }
            if json_schema is not None:
                payload["format"] = json_schema
            options = {key: value for key, value in {"temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": num_ctx, "num_batch": num_batch}.items() if value is not None}
            if options:
                payload["options"] = options
            return _node_broker_call(
                model=model,
                operation="chat",
                payload=payload,
                purpose=purpose,
                timeout=timeout,
                context_tokens=int(num_ctx or 0),
            )
        except Exception as error:
            if not should_fallback(error) or not os.getenv("INFERENCE_GATEWAY_URL", "").strip():
                raise
    item = _catalog(model)
    request = {
        "request_id": str(uuid.uuid4()),
        "application": os.getenv("INFERENCE_APPLICATION", "manuals_rag"),
        "workload": os.getenv("INFERENCE_WORKLOAD", purpose or "interactive"),
        "model": model,
        "model_size_bytes": int(item["size_bytes"]),
        "min_vram_mb": int(item["min_vram_mb"]),
        "context_tokens": int(num_ctx or 0),
        "context_vram_mb_per_1k": float(item.get("context_vram_mb_per_1k", 0.0)),
        "required_capabilities": list(item.get("capabilities") or ["chat"]),
        "priority": int(os.getenv("INFERENCE_PRIORITY", "50")),
        "max_latency_ms": int(timeout * 1000),
        "allow_model_pull": os.getenv("INFERENCE_ALLOW_MODEL_PULL", "false").lower() == "true",
        "allow_eviction": os.getenv("INFERENCE_ALLOW_MODEL_EVICTION", "false").lower() == "true",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "think": think,
    }
    if json_schema is not None:
        payload["format"] = json_schema
    options = {key: value for key, value in {"temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": num_ctx, "num_batch": num_batch}.items() if value is not None}
    if options:
        payload["options"] = options
    headers = _headers()
    with httpx.Client(base_url=os.environ["INFERENCE_GATEWAY_URL"].rstrip("/"), timeout=max(timeout, 30.0)) as client:
        response = client.post("/v1/inference", json={"request": request, "operation": "chat", "payload": payload}, headers=headers)
        response.raise_for_status()
        result = response.json()
    body = dict(result.get("response") or {})
    body["_inference_routing"] = result.get("routing") or {}
    return body


def generate(
    *,
    model: str,
    prompt: str,
    purpose: str | None,
    think: bool | None = None,
    options: dict[str, Any] | None = None,
    timeout: float = 90.0,
    keep_alive: str | int | None = "10m",
) -> dict[str, Any]:
    """Request Ollama ``/api/generate`` through the central scheduler.

    The gateway returns a complete Ollama-shaped response. Streaming callers
    are intentionally buffered at this compatibility boundary until the
    gateway has a true streaming protocol; this keeps routing and admission
    centralized without silently bypassing the scheduler.
    """
    options = dict(options or {})
    if node_broker_enabled():
        try:
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive,
            }
            if think is not None:
                payload["think"] = think
            if options:
                payload["options"] = options
            return _node_broker_call(
                model=model,
                operation="generate",
                payload=payload,
                purpose=purpose,
                timeout=timeout,
                context_tokens=int(options.get("num_ctx") or 0),
            )
        except Exception as error:
            if not should_fallback(error) or not os.getenv("INFERENCE_GATEWAY_URL", "").strip():
                raise
    item = _catalog(model)
    request = {
        "request_id": str(uuid.uuid4()),
        "application": os.getenv("INFERENCE_APPLICATION", "manuals_rag"),
        "workload": os.getenv("INFERENCE_WORKLOAD", purpose or "interactive"),
        "model": model,
        "model_size_bytes": int(item["size_bytes"]),
        "min_vram_mb": int(item["min_vram_mb"]),
        "context_tokens": int(options.get("num_ctx") or 0),
        "context_vram_mb_per_1k": float(item.get("context_vram_mb_per_1k", 0.0)),
        "required_capabilities": list(item.get("capabilities") or ["chat"]),
        "priority": int(os.getenv("INFERENCE_PRIORITY", "50")),
        "max_latency_ms": int(timeout * 1000),
        "allow_model_pull": os.getenv("INFERENCE_ALLOW_MODEL_PULL", "false").lower() == "true",
        "allow_eviction": os.getenv("INFERENCE_ALLOW_MODEL_EVICTION", "false").lower() == "true",
    }
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if think is not None:
        payload["think"] = think
    if options:
        payload["options"] = options
    headers = _headers()
    with httpx.Client(base_url=os.environ["INFERENCE_GATEWAY_URL"].rstrip("/"), timeout=max(timeout, 30.0)) as client:
        response = client.post(
            "/v1/inference",
            json={"request": request, "operation": "generate", "payload": payload},
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()
    body = dict(result.get("response") or {})
    body["_inference_routing"] = result.get("routing") or {}
    return body


def embeddings(
    *,
    model: str,
    inputs: list[str],
    purpose: str | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Request embeddings through the central gateway using the Ollama shape."""
    if node_broker_enabled():
        try:
            return _node_broker_call(
                model=model,
                operation="embeddings",
                payload={"model": model, "input": inputs},
                purpose=purpose or "embedding",
                timeout=timeout,
                required_capabilities=["embeddings"],
            )
        except Exception as error:
            if not should_fallback(error) or not os.getenv("INFERENCE_GATEWAY_URL", "").strip():
                raise
    item = _catalog(model)
    request = {
        "request_id": str(uuid.uuid4()),
        "application": os.getenv("INFERENCE_APPLICATION", "manuals_rag"),
        "workload": os.getenv("INFERENCE_WORKLOAD", purpose or "embedding"),
        "model": model,
        "model_size_bytes": int(item["size_bytes"]),
        "min_vram_mb": int(item["min_vram_mb"]),
        "required_capabilities": list(item.get("capabilities") or ["embeddings"]),
        "priority": int(os.getenv("INFERENCE_PRIORITY", "50")),
        "max_latency_ms": int(timeout * 1000),
        "allow_model_pull": os.getenv("INFERENCE_ALLOW_MODEL_PULL", "false").lower() == "true",
        "allow_eviction": os.getenv("INFERENCE_ALLOW_MODEL_EVICTION", "false").lower() == "true",
    }
    headers = _headers()
    with httpx.Client(base_url=os.environ["INFERENCE_GATEWAY_URL"].rstrip("/"), timeout=max(timeout, 30.0)) as client:
        response = client.post(
            "/v1/inference",
            json={"request": request, "operation": "embeddings", "payload": {"model": model, "input": inputs}},
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()
    body = dict(result.get("response") or {})
    body["_inference_routing"] = result.get("routing") or {}
    return body
