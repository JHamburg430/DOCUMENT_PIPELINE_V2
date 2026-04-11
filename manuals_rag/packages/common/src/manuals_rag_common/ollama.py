from __future__ import annotations

import json
import logging
import re
from collections import deque
from datetime import datetime, UTC
from threading import Lock
from typing import Any

import httpx

from manuals_rag_common.config import settings

logger = logging.getLogger(__name__)
DEFAULT_KEEP_ALIVE = "10m"
DEFAULT_LOAD_TIMEOUT = 180.0
RECENT_CALL_LIMIT = 200
_recent_ollama_calls: deque[dict[str, Any]] = deque(maxlen=RECENT_CALL_LIMIT)
_recent_ollama_calls_lock = Lock()


def _record_call(event: dict[str, Any]) -> None:
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        **event,
    }
    with _recent_ollama_calls_lock:
        _recent_ollama_calls.append(payload)


def recent_ollama_calls(*, limit: int = 50) -> list[dict[str, Any]]:
    bounded = max(1, min(limit, RECENT_CALL_LIMIT))
    with _recent_ollama_calls_lock:
        return list(_recent_ollama_calls)[-bounded:]


def model_family(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith("qwen"):
        return "qwen"
    if normalized.startswith("gpt-oss"):
        return "gpt_oss"
    return "other"


def supports_thinking_control(model: str) -> bool:
    return model_family(model) == "qwen"


def _thinking_directive(model: str, think: bool | None) -> str:
    if think is None or not supports_thinking_control(model):
        return ""
    return "/think" if think else "/no_think"


def _inject_thinking_directive(model: str, messages: list[dict[str, str]], think: bool | None) -> list[dict[str, str]]:
    directive = _thinking_directive(model, think)
    if not directive:
        return messages
    updated = [dict(message) for message in messages]
    if not updated:
        return [{"role": "system", "content": directive}]
    target_index = 0 if updated[0].get("role") == "system" else len(updated) - 1
    content = str(updated[target_index].get("content") or "").strip()
    updated[target_index]["content"] = f"{content}\n\n{directive}".strip()
    return updated


def _chat_options(model: str, *, json_mode: bool, num_predict: int | None = None) -> dict[str, Any]:
    family = model_family(model)
    if family == "qwen":
        if json_mode:
            options = {"temperature": 0.1, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
        else:
            options = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
        if num_predict is not None:
            options["num_predict"] = num_predict
        return options
    if json_mode:
        options = {"temperature": 0.0}
    else:
        options = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    return options


def build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any] | None = None,
    think: bool | None = None,
    keep_alive: str | None = DEFAULT_KEEP_ALIVE,
    stream: bool = False,
    num_predict: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _inject_thinking_directive(model, messages, think),
        "stream": stream,
        "options": _chat_options(model, json_mode=json_schema is not None, num_predict=num_predict),
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if json_schema is not None:
        payload["format"] = json_schema
    if supports_thinking_control(model) and think is not None:
        payload["think"] = think
    return payload


def extract_chat_content(payload: dict[str, Any]) -> str:
    message = payload.get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        return content
    # Defensive cleanup in case a thinking-capable model leaks inline reasoning markup.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    return content.strip()


def _available_models(client: httpx.Client) -> set[str]:
    response = client.get("/api/tags")
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    return {str(item.get("name") or "") for item in body.get("models", []) if item.get("name")}


def _loaded_models(client: httpx.Client) -> set[str]:
    response = client.get("/api/ps")
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    return {str(item.get("name") or "") for item in body.get("models", []) if item.get("name")}


def ensure_model_loaded(
    *,
    client: httpx.Client,
    model: str,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    force_reload: bool = False,
    purpose: str | None = None,
) -> None:
    available = _available_models(client)
    if model not in available:
        _record_call({"kind": "ensure_model_loaded", "model": model, "purpose": purpose, "status": "missing"})
        raise ValueError(f"Requested Ollama model is not installed: {model}")
    if not force_reload and model in _loaded_models(client):
        _record_call({"kind": "ensure_model_loaded", "model": model, "purpose": purpose, "status": "already_loaded"})
        return
    logger.info("Warming Ollama model before request: %s", model)
    _record_call({"kind": "ensure_model_loaded", "model": model, "purpose": purpose, "status": "warming", "force_reload": force_reload})
    response = client.post(
        "/api/generate",
        json={
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0.0},
        },
    )
    response.raise_for_status()
    _record_call({"kind": "ensure_model_loaded", "model": model, "purpose": purpose, "status": "warmed", "force_reload": force_reload})


def _post_chat(
    *,
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any] | None = None,
    think: bool | None = None,
    keep_alive: str | None = DEFAULT_KEEP_ALIVE,
    purpose: str | None = None,
    num_predict: int | None = None,
) -> dict[str, Any]:
    loaded_before = sorted(_loaded_models(client))
    request_payload = build_chat_payload(
        model=model,
        messages=messages,
        json_schema=json_schema,
        think=think,
        keep_alive=keep_alive,
        num_predict=num_predict,
    )
    _record_call(
        {
            "kind": "chat_request",
            "model": model,
            "purpose": purpose,
            "think": think,
            "json_mode": json_schema is not None,
            "loaded_models_before": loaded_before,
            "num_predict": num_predict,
        }
    )
    response = client.post("/api/chat", json=request_payload)
    response.raise_for_status()
    body = response.json()
    _record_call(
        {
            "kind": "chat_response",
            "model": model,
            "purpose": purpose,
            "status": "ok",
            "response_model": body.get("model"),
            "loaded_models_after": sorted(_loaded_models(client)),
        }
    )
    return body


def _post_chat_stream(
    *,
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any] | None = None,
    think: bool | None = None,
    keep_alive: str | None = DEFAULT_KEEP_ALIVE,
    purpose: str | None = None,
    on_token: Any | None = None,
    num_predict: int | None = None,
) -> dict[str, Any]:
    loaded_before = sorted(_loaded_models(client))
    request_payload = build_chat_payload(
        model=model,
        messages=messages,
        json_schema=json_schema,
        think=think,
        keep_alive=keep_alive,
        stream=True,
        num_predict=num_predict,
    )
    _record_call(
        {
            "kind": "chat_request",
            "model": model,
            "purpose": purpose,
            "think": think,
            "json_mode": json_schema is not None,
            "stream": True,
            "loaded_models_before": loaded_before,
            "num_predict": num_predict,
        }
    )
    chunks: list[str] = []
    final_body: dict[str, Any] = {}
    with client.stream("POST", "/api/chat", json=request_payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            token = str((event.get("message") or {}).get("content") or "")
            if token:
                chunks.append(token)
                if on_token is not None:
                    on_token(token)
            if event.get("done"):
                final_body = event
    body = {
        **final_body,
        "message": {
            **dict(final_body.get("message") or {}),
            "content": "".join(chunks),
        },
    }
    _record_call(
        {
            "kind": "chat_response",
            "model": model,
            "purpose": purpose,
            "status": "ok",
            "response_model": body.get("model"),
            "stream": True,
            "loaded_models_after": sorted(_loaded_models(client)),
        }
    )
    return body


def chat_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any],
    think: bool | None = None,
    timeout: float = 90.0,
    load_timeout: float = DEFAULT_LOAD_TIMEOUT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    purpose: str | None = None,
    num_predict: int | None = None,
) -> tuple[dict[str, Any], str]:
    with httpx.Client(base_url=settings.ollama_url, timeout=max(timeout, load_timeout)) as client:
        ensure_model_loaded(client=client, model=model, keep_alive=keep_alive, purpose=purpose)
        try:
            body = _post_chat(
                client=client,
                model=model,
                messages=messages,
                json_schema=json_schema,
                think=think,
                keep_alive=keep_alive,
                purpose=purpose,
                num_predict=num_predict,
            )
        except Exception as exc:
            logger.warning("Ollama chat_json failed for model=%s; reloading and retrying once: %s", model, exc)
            _record_call({"kind": "chat_error", "model": model, "purpose": purpose, "error": str(exc)})
            ensure_model_loaded(client=client, model=model, keep_alive=keep_alive, force_reload=True, purpose=purpose)
            body = _post_chat(
                client=client,
                model=model,
                messages=messages,
                json_schema=json_schema,
                think=think,
                keep_alive=keep_alive,
                purpose=purpose,
                num_predict=num_predict,
            )
    content = extract_chat_content(body)
    return json.loads(content or "{}"), content


def chat_json_stream(
    *,
    model: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any],
    on_token: Any | None = None,
    think: bool | None = None,
    timeout: float = 90.0,
    load_timeout: float = DEFAULT_LOAD_TIMEOUT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    purpose: str | None = None,
    num_predict: int | None = None,
) -> tuple[dict[str, Any], str]:
    with httpx.Client(base_url=settings.ollama_url, timeout=max(timeout, load_timeout)) as client:
        ensure_model_loaded(client=client, model=model, keep_alive=keep_alive, purpose=purpose)
        body = _post_chat_stream(
            client=client,
            model=model,
            messages=messages,
            json_schema=json_schema,
            think=think,
            keep_alive=keep_alive,
            purpose=purpose,
            on_token=on_token,
            num_predict=num_predict,
        )
    content = extract_chat_content(body)
    return json.loads(content or "{}"), content


def chat_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    think: bool | None = None,
    timeout: float = 90.0,
    load_timeout: float = DEFAULT_LOAD_TIMEOUT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    purpose: str | None = None,
    num_predict: int | None = None,
) -> tuple[str, dict[str, Any]]:
    with httpx.Client(base_url=settings.ollama_url, timeout=max(timeout, load_timeout)) as client:
        ensure_model_loaded(client=client, model=model, keep_alive=keep_alive, purpose=purpose)
        try:
            body = _post_chat(
                client=client,
                model=model,
                messages=messages,
                think=think,
                keep_alive=keep_alive,
                purpose=purpose,
                num_predict=num_predict,
            )
        except Exception as exc:
            logger.warning("Ollama chat_text failed for model=%s; reloading and retrying once: %s", model, exc)
            _record_call({"kind": "chat_error", "model": model, "purpose": purpose, "error": str(exc)})
            ensure_model_loaded(client=client, model=model, keep_alive=keep_alive, force_reload=True, purpose=purpose)
            body = _post_chat(
                client=client,
                model=model,
                messages=messages,
                think=think,
                keep_alive=keep_alive,
                purpose=purpose,
                num_predict=num_predict,
            )
    return extract_chat_content(body), body
