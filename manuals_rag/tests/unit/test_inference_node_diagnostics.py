import io
import json
from urllib.error import HTTPError

import pytest

from manuals_rag_inference import node_agent


def test_heartbeat_failure_and_recovery_are_visible_without_response_secrets(monkeypatch):
    agent = node_agent.OllamaNode(node_id="test", endpoint="http://localhost:11434", tailnet_name="test", secret=b"private-test-secret")
    monkeypatch.setattr(node_agent, "_json_request", lambda *args, **kwargs: {"models": []})
    monkeypatch.setattr(node_agent, "_gpu_snapshots", lambda: ())

    def denied(*args, **kwargs):
        raise HTTPError("https://test/heartbeat", 403, "sensitive rejection detail", {}, None)

    monkeypatch.setattr(node_agent, "urlopen", denied)
    with pytest.raises(HTTPError):
        agent.publish_heartbeat("https://gateway.test")
    assert agent.activity()[-1]["error"] == "HTTP_403"
    assert agent.activity()[-1]["status"] == "failed"
    monkeypatch.setattr(node_agent, "urlopen", lambda *args, **kwargs: io.BytesIO(b'{"accepted": true}'))
    assert agent.publish_heartbeat("https://gateway.test") == {"accepted": True}
    assert agent.activity()[-1]["status"] == "accepted"
    assert len([event for event in agent.activity() if event["operation"] == "gpu"]) == 1
    assert "sensitive rejection detail" not in json.dumps(agent.activity())
    assert "private-test-secret" not in json.dumps(agent.activity())
