import json

from manuals_rag_common import inference_gateway
from manuals_rag_retrieval import embeddings as retrieval_embeddings


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
