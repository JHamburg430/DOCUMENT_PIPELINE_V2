import io
import json
import time
from urllib.error import HTTPError

import pytest

from manuals_rag_inference import node_agent
from manuals_rag_inference.control import sign_command
from manuals_rag_inference.gateway import _snapshot_from_payload
from manuals_rag_inference.models import GpuSnapshot, InferenceRequest, NodeSnapshot
from manuals_rag_inference.scheduler import InferenceScheduler


def node():
    return node_agent.OllamaNode(node_id="worker", endpoint="http://localhost:11434", tailnet_name="test", secret=b"test-only-secret")


def pull(agent):
    command = {"action": "pull_model", "model": "qwen3.5:9b", "size_bytes": 6603314868, "issued_at": time.time(), "timeout_seconds": 120}
    return agent.command(command, sign_command(command, agent.secret))


def test_installed_model_is_reused_without_contacting_registry(monkeypatch):
    calls = []
    def request(url, **kwargs):
        calls.append(url)
        assert url.endswith("/api/tags")
        return {"models": [{"name": "qwen3.5:9b"}]}
    monkeypatch.setattr(node_agent, "_json_request", request)
    agent = node()
    assert pull(agent)["already_installed"] is True
    assert len(calls) == 1
    assert agent.activity()[-1]["status"] == "already_installed"


def test_missing_model_is_pulled_automatically(monkeypatch):
    calls = []
    def request(url, **kwargs):
        calls.append((url, kwargs))
        return {"models": []} if url.endswith("/api/tags") else {"status": "success"}
    monkeypatch.setattr(node_agent, "_json_request", request)
    agent = node()
    assert pull(agent)["status"] == "completed"
    assert calls[-1][0].endswith("/api/pull")
    assert calls[-1][1]["body"] == {"model": "qwen3.5:9b", "stream": False}
    assert agent.activity()[-1]["status"] == "completed"


def test_pull_error_is_categorized_without_raw_credentials(monkeypatch):
    def request(url, **kwargs):
        if url.endswith("/api/tags"):
            return {"models": []}
        raise HTTPError(url, 500, "private detail", {}, io.BytesIO(b'{"error":"pull model manifest: certificate failure secret-token"}'))
    monkeypatch.setattr(node_agent, "_json_request", request)
    agent = node()
    with pytest.raises(node_agent.ModelPreparationError) as failure:
        pull(agent)
    assert failure.value.category == "registry_tls_error"
    assert failure.value.upstream_status == 500
    assert agent.activity()[-1]["error"] == "HTTP_500:registry_tls_error"
    assert "secret-token" not in json.dumps(agent.activity())


@pytest.mark.parametrize("reply", [{"error": "model manifest not found"}, {"status": "pulling manifest"}])
def test_incomplete_or_error_response_is_not_reported_as_success(monkeypatch, reply):
    monkeypatch.setattr(node_agent, "_json_request", lambda url, **kwargs: {"models": []} if url.endswith("/api/tags") else reply)
    with pytest.raises(node_agent.ModelPreparationError):
        pull(node())


@pytest.mark.parametrize("installed,allowed,admitted,requires_pull", [(True, False, True, False), (False, True, True, True), (False, False, False, False)])
def test_admission_distinguishes_disk_inventory_from_loaded_models(installed, allowed, admitted, requires_pull):
    scheduler = InferenceScheduler()
    snapshot = NodeSnapshot("worker", "http://localhost:11434", "test", (GpuSnapshot("0", "test", 24000, 0),),
        installed_models=("qwen3.5:9b",) if installed else (), observed_at_epoch=time.time())
    scheduler.upsert_node(_snapshot_from_payload(snapshot.to_dict()))
    request = InferenceRequest("test", "manuals_rag", "chat", "qwen3.5:9b", 6603314868, 8192, allow_model_pull=allowed)
    decision = scheduler.choose(request)
    assert decision.admitted is admitted
    if admitted:
        assert decision.requires_model_pull is requires_pull
