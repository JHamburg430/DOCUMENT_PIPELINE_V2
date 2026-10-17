import json

import httpx
import pytest

from manuals_rag_common import inference_gateway
from manuals_rag_retrieval import embeddings as retrieval_embeddings
from manuals_rag_common import ollama


def test_gateway_client_builds_application_aware_request(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self): return None
        def json(self): return {"routing": {"node_id": "gpu-a"}, "response": {"message": {"content": "{}"}}}

    class Client:
        def __init__(self, **kwargs): captured["client"] = kwargs
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def post(self, path, **kwargs): captured["request"] = (path, kwargs); return Response()

    monkeypatch.setenv("INFERENCE_GATEWAY_URL", "http://gateway")
    monkeypatch.setenv("INFERENCE_GATEWAY_TOKEN", "token")
    monkeypatch.setenv("INFERENCE_APPLICATION_TOKEN", "application-secret")
    monkeypatch.setenv("INFERENCE_APPLICATION", "provis_employee_agent")
    monkeypatch.setenv("INFERENCE_MODEL_CATALOG_JSON", json.dumps({"qwen3.5:9b": {"size_bytes": 10, "min_vram_mb": 8000}}))
    monkeypatch.setattr(inference_gateway.httpx, "Client", Client)
    result = inference_gateway.chat(model="qwen3.5:9b", messages=[], purpose="answer")
    assert result["_inference_routing"]["node_id"] == "gpu-a"
    assert captured["request"][1]["json"]["request"]["application"] == "provis_employee_agent"
    assert captured["request"][1]["headers"]["X-Gateway-Token"] == "token"
    assert captured["request"][1]["headers"]["X-Inference-Application-Token"] == "application-secret"


def test_node_broker_requires_no_gateway_credentials(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self): return None
        def json(self): return {"routing": {"node_id": "gpu-a"}, "response": {"message": {"content": "ok"}}}

    class Client:
        def __init__(self, **kwargs): captured["client"] = kwargs
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def post(self, path, **kwargs): captured["request"] = (path, kwargs); return Response()

    monkeypatch.delenv("INFERENCE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("INFERENCE_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("INFERENCE_APPLICATION_TOKEN", raising=False)
    monkeypatch.delenv("INFERENCE_MODEL_CATALOG_JSON", raising=False)
    monkeypatch.setenv("INFERENCE_NODE_BROKER_URL", "http://host.docker.internal:8767")
    monkeypatch.setenv("INFERENCE_APPLICATION", "manuals_rag")
    monkeypatch.setattr(inference_gateway.httpx, "Client", Client)

    result = inference_gateway.chat(model="qwen3.5:9b", messages=[], purpose="answer")

    assert result["_inference_transport"] == "local_node_broker"
    path, kwargs = captured["request"]
    assert path == "/v1/node/broker/inference"
    assert kwargs["json"]["request"]["application"] == "manuals_rag"
    assert "headers" not in kwargs


def test_gateway_client_builds_embedding_request(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self): return None
        def json(self): return {"routing": {"node_id": "gpu-b"}, "response": {"embeddings": [[0.1, 0.2]]}}

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def post(self, path, **kwargs): captured["request"] = (path, kwargs); return Response()

    monkeypatch.setenv("INFERENCE_GATEWAY_URL", "http://gateway")
    monkeypatch.setenv("INFERENCE_MODEL_CATALOG_JSON", json.dumps({"qwen3-embedding:0.6b": {"size_bytes": 10, "min_vram_mb": 1000}}))
    monkeypatch.setattr(inference_gateway.httpx, "Client", Client)
    result = inference_gateway.embeddings(model="qwen3-embedding:0.6b", inputs=["hello"], purpose="retrieval")
    assert result["embeddings"] == [[0.1, 0.2]]
    assert captured["request"][1]["json"]["operation"] == "embeddings"
    assert captured["request"][1]["json"]["request"]["required_capabilities"] == ["embeddings"]


def test_gateway_client_builds_generate_request(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self): return None
        def json(self): return {"routing": {"node_id": "gpu-c"}, "response": {"response": "generated"}}

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def post(self, path, **kwargs): captured["request"] = (path, kwargs); return Response()

    monkeypatch.setenv("INFERENCE_GATEWAY_URL", "http://gateway")
    monkeypatch.setenv("INFERENCE_GATEWAY_TOKEN", "token")
    monkeypatch.setenv("INFERENCE_MODEL_CATALOG_JSON", json.dumps({
        "qwen3.5:27b": {
            "size_bytes": 27,
            "min_vram_mb": 16000,
            "context_vram_mb_per_1k": 120,
            "capabilities": ["chat"],
        },
    }))
    monkeypatch.setattr(inference_gateway.httpx, "Client", Client)
    result = inference_gateway.generate(
        model="qwen3.5:27b",
        prompt="make a question",
        purpose="question_generation",
        options={"num_ctx": 32768, "num_predict": 700},
        timeout=180,
    )
    assert result["response"] == "generated"
    path, kwargs = captured["request"]
    assert path == "/v1/inference"
    assert kwargs["json"]["operation"] == "generate"
    assert kwargs["json"]["payload"]["stream"] is False
    assert kwargs["json"]["request"]["context_tokens"] == 32768


def test_retrieval_embedding_call_uses_gateway_when_enabled(monkeypatch):
    monkeypatch.setattr(retrieval_embeddings.inference_gateway, "enabled", lambda: True)
    monkeypatch.setattr(
        retrieval_embeddings.inference_gateway,
        "embeddings",
        lambda **kwargs: {"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
    )
    assert retrieval_embeddings.embed_dense(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_gateway_fallback_is_limited_to_availability_failures(monkeypatch):
    monkeypatch.setenv("INFERENCE_GATEWAY_FALLBACK_ENABLED", "true")
    assert inference_gateway.should_fallback(httpx.ConnectError("offline"))
    assert inference_gateway.should_fallback(httpx.ReadTimeout("timeout"))
    assert inference_gateway.should_fallback(httpx.HTTPStatusError(
        "unavailable",
        request=httpx.Request("POST", "http://gateway/v1/inference"),
        response=httpx.Response(503),
    ))
    assert not inference_gateway.should_fallback(httpx.HTTPStatusError(
        "unauthorized",
        request=httpx.Request("POST", "http://gateway/v1/inference"),
        response=httpx.Response(401),
    ))
    assert not inference_gateway.should_fallback(RuntimeError("missing model catalog"))


def test_chat_text_uses_local_ollama_when_gateway_is_unavailable(monkeypatch):
    monkeypatch.setattr(ollama.inference_gateway, "enabled", lambda: True)
    monkeypatch.setenv("INFERENCE_GATEWAY_FALLBACK_ENABLED", "true")
    monkeypatch.setattr(ollama.inference_gateway, "chat", lambda **_kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    monkeypatch.setattr(ollama, "ensure_model_loaded", lambda **_kwargs: None)
    monkeypatch.setattr(ollama, "_post_chat", lambda **_kwargs: {"message": {"content": "local answer"}})

    class Client:
        def __init__(self, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return None

    monkeypatch.setattr(ollama.httpx, "Client", Client)
    content, body = ollama.chat_text(model="qwen3.5:9b", messages=[], purpose="answer")
    assert content == "local answer"
    assert body["message"]["content"] == "local answer"


def test_gateway_auth_failure_does_not_fall_back_to_local(monkeypatch):
    monkeypatch.setattr(ollama.inference_gateway, "enabled", lambda: True)
    monkeypatch.setenv("INFERENCE_GATEWAY_FALLBACK_ENABLED", "true")
    response = httpx.Response(401, request=httpx.Request("POST", "http://gateway/v1/inference"))
    error = httpx.HTTPStatusError("unauthorized", request=response.request, response=response)
    monkeypatch.setattr(ollama.inference_gateway, "chat", lambda **_kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(httpx.HTTPStatusError):
        ollama.chat_text(model="qwen3.5:9b", messages=[], purpose="answer")


def test_embeddings_use_local_ollama_when_gateway_is_unavailable(monkeypatch):
    monkeypatch.setattr(retrieval_embeddings.inference_gateway, "enabled", lambda: True)
    monkeypatch.setenv("INFERENCE_GATEWAY_FALLBACK_ENABLED", "true")
    monkeypatch.setattr(
        retrieval_embeddings.inference_gateway,
        "embeddings",
        lambda **_kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )

    class Response:
        status_code = 200
        def raise_for_status(self): return None
        def json(self): return {"embeddings": [[0.9, 0.8]]}

    class Client:
        def __init__(self, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def post(self, *_args, **_kwargs): return Response()

    monkeypatch.setattr(retrieval_embeddings.httpx, "Client", Client)
    assert retrieval_embeddings.embed_dense(["fallback text"]) == [[0.9, 0.8]]
