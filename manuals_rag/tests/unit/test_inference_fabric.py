import time

import json
import ipaddress
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from manuals_rag_inference.control import ControlPolicy, sign_command, verify_command
from manuals_rag_inference.gateway import InferenceGateway, heartbeat_signature, node_request_signature, run_gateway, validate_gateway_runtime_security
from manuals_rag_inference.enrollment import EnrollmentManager, validate_node_endpoint
from manuals_rag_inference.models import GpuSnapshot, InferenceRequest, LoadedModel, ModelSpec, NodeSnapshot
from manuals_rag_inference.scheduler import InferenceScheduler, SchedulerPolicy
from manuals_rag_inference.node_agent import OllamaNode, _detect_tailscale_ipv4, create_handler
import manuals_rag_inference.node_agent as node_agent_module


def node(node_id, *, model=None, observed=None, allowed=()):
    return NodeSnapshot(
        node_id=node_id,
        endpoint=f"http://{node_id}.tailnet:11434",
        tailnet_name="tailnet",
        gpus=(GpuSnapshot("0", "RTX", 24000, 4000),),
        loaded_models=((LoadedModel(model, 10_000_000_000, ("0",), time.time(), 0),) if model else ()),
        allowed_applications=allowed,
        observed_at_epoch=time.time() if observed is None else observed,
    )


def request(**overrides):
    values = dict(request_id="r1", application="provis_employee_agent", workload="interactive", model="qwen3.5:9b", model_size_bytes=10_000_000_000, min_vram_mb=8000, allow_model_pull=True)
    values.update(overrides)
    return InferenceRequest(**values)


def test_loaded_model_wins_over_free_but_cold_node():
    scheduler = InferenceScheduler()
    scheduler.upsert_node(node("cold"))
    scheduler.upsert_node(node("warm", model="qwen3.5:9b"))
    decision = scheduler.choose(request())
    assert decision.admitted and decision.node_id == "warm"
    assert decision.requires_model_pull is False


def test_stale_nodes_and_disallowed_pulls_are_rejected():
    scheduler = InferenceScheduler(policy=SchedulerPolicy(heartbeat_ttl_seconds=5))
    scheduler.upsert_node(node("stale", observed=time.time() - 30))
    decision = scheduler.choose(request(allow_model_pull=False))
    assert not decision.admitted
    assert "node_heartbeat_stale" in decision.reason


def test_scheduler_rejects_unknown_model_resource_requirements():
    scheduler = InferenceScheduler()
    scheduler.upsert_node(node("resource-check"))
    decision = scheduler.choose(request(model_size_bytes=0))
    assert not decision.admitted
    assert "invalid_model_resource_requirements" in decision.reason


def test_production_allowlist_mode_rejects_unscoped_nodes():
    scheduler = InferenceScheduler(policy=SchedulerPolicy(require_application_allowlist=True))
    scheduler.upsert_node(node("unscoped"))
    decision = scheduler.choose(request())
    assert not decision.admitted
    assert "application_allowlist_missing" in decision.reason


def test_production_allowlist_mode_routes_only_to_authorized_application():
    scheduler = InferenceScheduler(policy=SchedulerPolicy(require_application_allowlist=True))
    scheduler.upsert_node(node("rag-only", allowed=("manuals_rag",)))
    assert scheduler.choose(request(application="manuals_rag")).admitted
    assert not scheduler.choose(request(request_id="other", application="openclaw")).admitted


def test_scheduler_fails_over_from_stale_preferred_node():
    scheduler = InferenceScheduler(policy=SchedulerPolicy(heartbeat_ttl_seconds=5))
    scheduler.upsert_node(node("stale-warm", model="qwen3.5:9b", observed=time.time() - 30))
    scheduler.upsert_node(node("healthy-cold"))
    decision = scheduler.choose(request())
    assert decision.admitted
    assert decision.node_id == "healthy-cold"
    assert decision.requires_model_pull is True


def test_scheduler_reservations_bound_concurrent_admission_and_release():
    scheduler = InferenceScheduler(policy=SchedulerPolicy(max_queue_depth=1))
    scheduler.upsert_node(node("only-node", model="qwen3.5:9b"))
    first = scheduler.choose_and_reserve(request(request_id="r1", allow_model_pull=False))
    second = scheduler.choose_and_reserve(request(request_id="r2", allow_model_pull=False))
    assert first.admitted
    assert not second.admitted
    assert "queue_full" in second.reason
    assert scheduler.release(first.node_id, first.request_id) is True
    third = scheduler.choose_and_reserve(request(request_id="r3", allow_model_pull=False))
    assert third.admitted


def test_control_command_is_signed_bounded_and_expiring():
    secret = b"node-secret"
    command = {"action": "pull_model", "model": "qwen3.5:9b", "size_bytes": 10_000_000_000, "issued_at": 1000}
    signature = sign_command(command, secret)
    assert verify_command(command, signature, secret, now=1000)
    assert not verify_command({**command, "model": "../../etc/passwd"}, signature, secret, now=1000)
    assert not verify_command(command, signature, secret, now=2000)
    assert not verify_command({**command, "size_bytes": 100_000_000_000}, sign_command({**command, "size_bytes": 100_000_000_000}, secret), secret, now=1000, policy=ControlPolicy(max_model_size_bytes=1))
    zero_size_pull = {**command, "size_bytes": 0}
    assert not verify_command(zero_size_pull, sign_command(zero_size_pull, secret), secret, now=1000)


def test_control_model_prefixes_are_configurable_defense_in_depth():
    secret = b"node-secret"
    command = {"action": "pull_model", "model": "company-reasoner:9b", "size_bytes": 10_000_000_000, "issued_at": 1000}
    signature = sign_command(command, secret)
    assert not verify_command(command, signature, secret, now=1000)
    assert verify_command(
        command,
        signature,
        secret,
        now=1000,
        policy=ControlPolicy(allowed_model_prefixes=("company-",)),
    )


def test_node_runner_accepts_only_tailscale_ipv4_from_auto_detection(monkeypatch):
    class Completed:
        stdout = "100.117.6.6\n"

    monkeypatch.setattr("manuals_rag_inference.node_agent.subprocess.run", lambda *args, **kwargs: Completed())
    assert _detect_tailscale_ipv4() == "100.117.6.6"

    class PublicCompleted:
        stdout = "203.0.113.10\n"

    monkeypatch.setattr("manuals_rag_inference.node_agent.subprocess.run", lambda *args, **kwargs: PublicCompleted())
    assert _detect_tailscale_ipv4() is None


def test_node_agent_rejects_unsigned_commands_before_ollama_call():
    agent = OllamaNode(node_id="n1", endpoint="http://127.0.0.1:9", tailnet_name="t", secret=b"secret")
    try:
        agent.command({"action": "pull_model", "model": "qwen3.5:9b", "size_bytes": 1, "issued_at": time.time()}, "bad")
    except PermissionError:
        pass
    else:
        raise AssertionError("unsigned command was accepted")


def test_node_agent_heartbeat_carries_application_allowlist():
    agent = OllamaNode(
        node_id="allowlisted",
        endpoint="http://127.0.0.1:9",
        tailnet_name="tailnet",
        secret=b"secret",
        allowed_applications=("manuals_rag", "provis_employee_agent"),
    )
    assert agent.snapshot().allowed_applications == ("manuals_rag", "provis_employee_agent")


def test_node_agent_marks_ollama_unhealthy_when_snapshot_fails(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("ollama unavailable")

    monkeypatch.setattr("manuals_rag_inference.node_agent._json_request", unavailable)
    agent = OllamaNode(node_id="offline", endpoint="http://127.0.0.1:9", tailnet_name="tailnet", secret=b"secret")
    assert agent.snapshot().healthy is False


def test_node_command_reuses_installed_model_and_warms_without_pull(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b"}]}
        if url.endswith("/api/generate"):
            return {"model": "qwen3.5:9b", "done": True}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen3.5:9b"}]}
        raise AssertionError(url)

    monkeypatch.setattr(node_agent_module, "_json_request", fake_request)
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret")
    command = {"action": "pull_model", "model": "qwen3.5:9b", "size_bytes": 10, "issued_at": time.time()}
    result = agent.command(command, sign_command(command, b"secret"))
    assert result == {"action": "pull_model", "model": "qwen3.5:9b", "status": "completed", "phase": "ready", "source": "installed"}
    assert not any(url.endswith("/api/pull") for url, _ in calls)
    assert any(event["operation"] == "model_warm" and event["status"] == "completed" for event in agent.activity())


def test_node_command_pulls_missing_model_then_warms(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/api/tags"):
            return {"models": []}
        if url.endswith("/api/pull"):
            return {"status": "success"}
        if url.endswith("/api/generate"):
            return {"model": "qwen3.5:9b", "done": True}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen3.5:9b"}]}
        raise AssertionError(url)

    monkeypatch.setattr(node_agent_module, "_json_request", fake_request)
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret")
    command = {"action": "pull_model", "model": "qwen3.5:9b", "size_bytes": 10, "issued_at": time.time()}
    result = agent.command(command, sign_command(command, b"secret"))
    assert result["source"] == "pulled"
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == ["tags", "pull", "generate", "ps"]


def test_node_command_reports_sanitized_pull_phase_failure(monkeypatch):
    def fake_request(url, **kwargs):
        if url.endswith("/api/tags"):
            return {"models": []}
        raise HTTPError(url, 500, "upstream secret detail", {}, None)

    monkeypatch.setattr(node_agent_module, "_json_request", fake_request)
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret")
    command = {"action": "pull_model", "model": "qwen3.5:9b", "size_bytes": 10, "issued_at": time.time()}
    with pytest.raises(RuntimeError, match="model_pull_failed:ollama_http_500"):
        agent.command(command, sign_command(command, b"secret"))
    assert all("upstream secret detail" not in str(event) for event in agent.activity())


def test_node_snapshot_reports_installed_inventory(monkeypatch):
    def fake_request(url, **kwargs):
        if url.endswith("/api/ps"):
            return {"models": []}
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b"}, {"model": "nomic-embed-text:latest"}]}
        raise AssertionError(url)

    monkeypatch.setattr(node_agent_module, "_json_request", fake_request)
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret")
    assert agent.snapshot().installed_models == ("qwen3.5:9b", "nomic-embed-text:latest")


def test_openai_adapter_preserves_top_level_ollama_think_flag(monkeypatch):
    captured = {}
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret", allowed_applications=("manuals_rag",))

    def broker(envelope, gateway_url):
        captured.update(envelope)
        return {"response": {"message": {"role": "assistant", "content": "READY"}}}

    monkeypatch.setattr(agent, "broker_infer", broker)
    result = agent.openai_infer(
        {"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "Reply READY"}], "think": False},
        operation="chat",
        application="manuals_rag",
        gateway_url="https://gateway",
    )
    assert captured["payload"]["think"] is False
    assert result["choices"][0]["message"]["content"] == "READY"


def test_gateway_requires_signed_heartbeat_and_forwards_to_selected_node():
    calls = []
    gateway = InferenceGateway(node_secrets={"warm": b"heartbeat-secret"}, post=lambda url, body, timeout: calls.append((url, body)) or {"message": "ok"})
    snapshot = node("warm", model="qwen3.5:9b")
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, b"heartbeat-secret"))
    result = gateway.infer(request(), operation="chat", payload={"model": "qwen3.5:9b", "messages": []})
    assert result["routing"]["node_id"] == "warm"
    assert calls[0][0].endswith("/api/chat")
    try:
        gateway.register_heartbeat(snapshot, raw, "bad")
    except PermissionError:
        pass
    else:
        raise AssertionError("unsigned heartbeat was accepted")


def test_gateway_rejects_replayed_heartbeat_payload():
    gateway = InferenceGateway(node_secrets={"warm": b"heartbeat-secret"})
    snapshot = node("warm", model="qwen3.5:9b")
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    signature = heartbeat_signature(raw, b"heartbeat-secret")
    gateway.register_heartbeat(snapshot, raw, signature)
    with pytest.raises(PermissionError, match="replayed"):
        gateway.register_heartbeat(snapshot, raw, signature)


def test_gateway_rejects_stale_heartbeat_at_ingress():
    gateway = InferenceGateway(node_secrets={"stale": b"heartbeat-secret"})
    snapshot = node("stale", model="qwen3.5:9b", observed=time.time() - 120)
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(PermissionError, match="stale"):
        gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, b"heartbeat-secret"))


def test_gateway_uses_authoritative_model_catalog_resources():
    gateway = InferenceGateway(
        model_catalog={"qwen3.5:9b": ModelSpec(20_000_000_000, 16_000, ("chat",))},
        require_model_catalog=True,
    )
    authoritative = gateway._authoritative_request(request(model_size_bytes=1, min_vram_mb=1))
    assert authoritative.model_size_bytes == 20_000_000_000
    assert authoritative.min_vram_mb == 16_000


def test_gateway_rejects_unknown_or_incompatible_catalog_models():
    gateway = InferenceGateway(
        model_catalog={"embed-model": ModelSpec(1_000_000_000, 2_000, ("embeddings",))},
        require_model_catalog=True,
    )
    with pytest.raises(RuntimeError, match="model_not_in_catalog"):
        gateway._authoritative_request(request(model="unknown-model"))
    with pytest.raises(RuntimeError, match="missing_capabilities"):
        gateway._authoritative_request(request(model="embed-model", required_capabilities=("chat",)))


def test_gateway_rejects_payload_model_mismatch_before_forwarding():
    calls = []
    gateway = InferenceGateway(
        node_secrets={"warm": b"heartbeat-secret"},
        post=lambda url, body, timeout: calls.append((url, body)) or {"ok": True},
        model_catalog={"qwen3.5:9b": ModelSpec(10_000_000_000, 8000, ("chat",))},
        require_model_catalog=True,
    )
    snapshot = node("warm", model="qwen3.5:9b")
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, b"heartbeat-secret"))
    with pytest.raises(RuntimeError, match="payload_model_mismatch"):
        gateway.infer(request(), operation="chat", payload={"model": "uncataloged:latest", "messages": []})
    assert calls == []


def test_gateway_catalog_normalizes_direct_model_pull_size():
    calls = []
    gateway = InferenceGateway(
        node_secrets={"warm": b"heartbeat-secret"},
        node_agent_urls={"warm": "http://warm.tailnet:8765"},
        post=lambda url, body, timeout: calls.append((url, body)) or {"status": "completed"},
        model_catalog={"qwen3.5:9b": ModelSpec(20_000_000_000, 16_000, ("chat",))},
        require_model_catalog=True,
    )
    gateway.issue_node_command(node_id="warm", action="pull_model", model="qwen3.5:9b", size_bytes=1)
    assert calls[0][1]["command"]["size_bytes"] == 20_000_000_000
    with pytest.raises(RuntimeError, match="model_not_in_catalog"):
        gateway.issue_node_command(node_id="warm", action="pull_model", model="unknown:latest", size_bytes=1)


def test_gateway_fails_over_from_stale_node_agent_to_healthy_node():
    calls = []

    def post(url, body, timeout):
        calls.append(url)
        return {"message": "healthy-node-response"}

    gateway = InferenceGateway(
        node_secrets={"stale": b"stale-secret", "healthy": b"healthy-secret"},
        node_agent_urls={
            "stale": "http://stale.tailnet:8765",
            "healthy": "http://healthy.tailnet:8765",
        },
        post=post,
    )
    stale = node("stale", model="qwen3.5:9b", observed=time.time() - 120)
    healthy = node("healthy", model="qwen3.5:9b")
    # A stale snapshot can exist in the scheduler after a node stops sending
    # heartbeats, but the gateway ingress rejects that same stale payload.
    gateway.scheduler.upsert_node(stale)
    raw = json.dumps(healthy.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(healthy, raw, heartbeat_signature(raw, b"healthy-secret"))
    result = gateway.infer(
        request(request_id="failover-1", allow_model_pull=False),
        operation="chat",
        payload={"model": "qwen3.5:9b", "messages": []},
    )
    assert result["routing"]["node_id"] == "healthy"
    assert calls == ["http://healthy.tailnet:8765/v1/node/inference"]


def test_gateway_runtime_security_rejects_weak_token_and_wildcard_bind():
    with pytest.raises(RuntimeError, match="32 characters"):
        validate_gateway_runtime_security(gateway_token="short", bind="127.0.0.1")
    with pytest.raises(RuntimeError, match="wildcard bind"):
        validate_gateway_runtime_security(gateway_token="x" * 32, bind="0.0.0.0")
    validate_gateway_runtime_security(gateway_token="x" * 32, bind="100.64.0.10")


def test_gateway_entrypoint_runs_preflight_before_binding(monkeypatch):
    monkeypatch.setenv("INFERENCE_GATEWAY_TOKEN", "short")
    monkeypatch.delenv("INFERENCE_GATEWAY_STATE_PATH", raising=False)
    with pytest.raises(RuntimeError, match="gateway preflight failed"):
        run_gateway()


def test_gateway_can_require_per_application_identity():
    gateway = InferenceGateway(
        gateway_token="g" * 32,
        application_tokens={"manuals_rag": "a" * 32},
        require_application_identity=True,
    )
    gateway.authorize_client("g" * 32, application="manuals_rag", application_token="a" * 32)
    with pytest.raises(PermissionError, match="application identity"):
        gateway.authorize_client("g" * 32, application="openclaw", application_token="a" * 32)
    with pytest.raises(PermissionError, match="application identity"):
        gateway.authorize_client("g" * 32, application="manuals_rag", application_token="wrong")


def test_gateway_can_require_node_agent_forwarding():
    gateway = InferenceGateway(
        gateway_token="x" * 32,
        node_secrets={"direct-only": b"heartbeat-secret"},
        require_node_agents=True,
    )
    snapshot = node("direct-only", model="qwen3.5:9b")
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, b"heartbeat-secret"))
    # The node secret is present for heartbeat verification, but no node-agent
    # URL is registered; production mode must reject direct Ollama forwarding.
    with pytest.raises(RuntimeError, match="node_agent_required"):
        gateway.infer(request(), operation="chat", payload={"model": "qwen3.5:9b", "messages": []})


def test_gateway_prepares_cold_model_with_signed_pull_before_forwarding():
    calls = []

    def post(url, body, timeout):
        calls.append((url, body))
        if url.endswith("/v1/node/command"):
            return {"status": "completed"}
        return {"message": "ok"}

    gateway = InferenceGateway(
        node_secrets={"warm": b"heartbeat-secret"},
        node_agent_urls={"warm": "http://warm.tailnet:8765"},
        post=post,
    )
    snapshot = node("warm")
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, b"heartbeat-secret"))
    result = gateway.infer(request(), operation="chat", payload={"model": "qwen3.5:9b", "messages": []})
    assert result["response"]["message"] == "ok"
    assert calls[0][0].endswith("/v1/node/command")
    assert calls[0][1]["command"]["action"] == "pull_model"
    assert calls[1][0].endswith("/v1/node/inference")


def test_gateway_rejects_cold_model_when_node_control_is_not_configured():
    gateway = InferenceGateway(node_secrets={"warm": b"heartbeat-secret"})
    snapshot = node("warm")
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, b"heartbeat-secret"))
    try:
        gateway.infer(request(), operation="chat", payload={"model": "qwen3.5:9b", "messages": []})
    except RuntimeError as error:
        assert "preparation is not configured" in str(error)
    else:
        raise AssertionError("cold model was forwarded without a configured node agent")


def test_scheduler_rejects_eviction_without_request_accounting():
    constrained = NodeSnapshot(
        node_id="constrained",
        endpoint="http://constrained.tailnet:11434",
        tailnet_name="tailnet",
        gpus=(GpuSnapshot("0", "RTX", 24000, 23000),),
        loaded_models=(LoadedModel("other:9b", 10_000_000_000, ("0",), time.time(), 0),),
        observed_at_epoch=time.time(),
    )
    scheduler = InferenceScheduler()
    scheduler.upsert_node(constrained)
    decision = scheduler.choose(request(allow_eviction=True))
    assert not decision.admitted
    assert "eviction_request_accounting_unavailable" in decision.reason


def test_scheduler_accounts_for_context_vram():
    constrained = NodeSnapshot(
        node_id="context-limited",
        endpoint="http://context-limited.tailnet:11434",
        tailnet_name="tailnet",
        gpus=(GpuSnapshot("0", "RTX", 10000, 1000),),
        loaded_models=(LoadedModel("qwen3.5:9b", 10_000_000_000, ("0",), time.time(), 0),),
        observed_at_epoch=time.time(),
    )
    scheduler = InferenceScheduler()
    scheduler.upsert_node(constrained)
    decision = scheduler.choose(request(context_tokens=4000, context_vram_mb_per_1k=500))
    assert not decision.admitted
    assert "insufficient_vram_without_eviction" in decision.reason


def test_scheduler_plans_only_fresh_idle_proxy_models():
    now = 10_000.0
    scheduler = InferenceScheduler(
        policy=SchedulerPolicy(heartbeat_ttl_seconds=15, idle_model_ttl_seconds=900),
        clock=lambda: now,
    )
    scheduler.upsert_node(NodeSnapshot(
        node_id="proxy",
        endpoint="http://proxy.tailnet:11434",
        tailnet_name="tailnet",
        gpus=(GpuSnapshot("0", "RTX", 24000, 1000),),
        loaded_models=(
            LoadedModel("old-model", 1, ("0",), now - 1800, 0),
            LoadedModel("recent-model", 1, ("0",), now - 100, 0),
            LoadedModel("active-model", 1, ("0",), now - 1800, 1),
            LoadedModel("unknown-age", 1, ("0",), 0, 0),
        ),
        observed_at_epoch=now,
        request_accounting_available=True,
    ))
    scheduler.upsert_node(NodeSnapshot(
        node_id="telemetry-only",
        endpoint="http://telemetry.tailnet:11434",
        tailnet_name="tailnet",
        gpus=(GpuSnapshot("0", "RTX", 24000, 1000),),
        loaded_models=(LoadedModel("should-stay", 1, ("0",), now - 1800, 0),),
        observed_at_epoch=now,
        request_accounting_available=False,
    ))
    candidates = scheduler.plan_idle_evictions(now=now)
    assert [(item.node_id, item.model) for item in candidates] == [("proxy", "old-model")]


def test_gateway_idle_eviction_requires_explicit_maintenance_approval():
    gateway = InferenceGateway(
        node_secrets={"proxy": b"proxy-secret"},
        node_agent_urls={"proxy": "http://proxy.tailnet:8765"},
        post=lambda url, body, timeout: {"status": "completed"},
    )
    with pytest.raises(PermissionError, match="approval"):
        gateway.evict_idle_models(maintenance_approved=False)
    with pytest.raises(PermissionError, match="approval"):
        gateway.evict_idle_models(maintenance_approved="true")


def test_node_proxy_verifies_signature_and_reports_proxy_accounting(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"embeddings": [[0.1]]}

    monkeypatch.setattr("manuals_rag_inference.node_agent._json_request", fake_request)
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret", proxy_only=True)
    envelope = {
        "operation": "embeddings",
        "payload": {"model": "qwen3-embedding:0.6b", "input": ["hello"]},
        "request_id": "embed-request-1",
        "issued_at": time.time(),
    }
    result = agent.infer(envelope, node_request_signature(envelope, b"secret"))
    assert result["embeddings"] == [[0.1]]
    assert calls[0][0].endswith("/api/embed")
    assert agent.snapshot().request_accounting_available is True


def test_node_broker_signs_gateway_request_and_enforces_application_allowlist(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"routing":{"node_id":"gpu-a"},"response":{"message":{"content":"ok"}}}'

    def fake_open(request_obj, timeout):
        captured["body"] = request_obj.data
        captured["signature"] = request_obj.headers["X-node-signature"]
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("manuals_rag_inference.node_agent.urlopen", fake_open)
    agent = OllamaNode(
        node_id="broker",
        endpoint="http://ollama",
        tailnet_name="tailnet",
        secret=b"secret",
        allowed_applications=("manuals_rag",),
    )
    envelope = {
        "request": {"request_id": "broker-r1", "application": "manuals_rag", "model": "qwen3.5:9b"},
        "operation": "chat",
        "payload": {"model": "qwen3.5:9b", "messages": []},
    }
    result = agent.broker_infer(envelope, "http://gateway")
    assert result["routing"]["node_id"] == "gpu-a"
    assert captured["signature"] == heartbeat_signature(captured["body"], b"secret")
    assert json.loads(captured["body"])["node_id"] == "broker"

    blocked = {**envelope, "request": {**envelope["request"], "application": "openclaw"}}
    with pytest.raises(PermissionError, match="not allowed"):
        agent.broker_infer(blocked, "http://gateway")


def test_node_openai_compatibility_translates_chat_and_embeddings(monkeypatch):
    agent = OllamaNode(
        node_id="openai-node",
        endpoint="http://ollama",
        tailnet_name="tailnet",
        secret=b"secret",
        allowed_applications=("manuals_rag",),
    )
    captured = []

    def fake_broker(envelope, gateway_url):
        captured.append((envelope, gateway_url))
        if envelope["operation"] == "embeddings":
            return {"routing": {"node_id": "gpu-a"}, "response": {"embeddings": [[0.1, 0.2]]}}
        return {"routing": {"node_id": "gpu-a"}, "response": {"message": {"role": "assistant", "content": "answer"}}}

    monkeypatch.setattr(agent, "broker_infer", fake_broker)
    chat = agent.openai_infer(
        {"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hello"}]},
        operation="chat", application="manuals_rag", gateway_url="https://gateway",
    )
    embeddings = agent.openai_infer(
        {"model": "qwen3-embedding:0.6b", "input": "hello"},
        operation="embeddings", application="manuals_rag", gateway_url="https://gateway",
    )
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"]["content"] == "answer"
    assert chat["_inference_routing"]["node_id"] == "gpu-a"
    assert embeddings["data"][0]["embedding"] == [0.1, 0.2]
    assert captured[0][0]["payload"]["messages"][0]["content"] == "hello"
    assert captured[1][0]["operation"] == "embeddings"


def test_node_openai_compatibility_builds_unbuffered_stream_request(monkeypatch):
    agent = OllamaNode(
        node_id="openai-node",
        endpoint="http://ollama",
        tailnet_name="tailnet",
        secret=b"secret",
        allowed_applications=("manuals_rag",),
    )
    captured = []

    class FakeStream:
        def __iter__(self):
            return iter([b'{"response":{"message":{"content":"hello"},"done":false}}\n', b'{"response":{"done":true}}\n'])

        def close(self):
            pass

    def fake_broker_stream(envelope, gateway_url):
        captured.append((envelope, gateway_url))
        return FakeStream()

    monkeypatch.setattr(agent, "broker_stream", fake_broker_stream)
    response, model, operation = agent.openai_stream(
        {"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hello"}], "stream": True},
        operation="chat", application="manuals_rag", gateway_url="https://gateway",
    )
    assert model == "qwen3.5:9b"
    assert operation == "chat"
    assert list(response)[0].startswith(b'{"response"')
    assert captured[0][0]["payload"]["stream"] is True


def test_local_openai_handler_emits_sse_for_streaming_response(monkeypatch):
    agent = OllamaNode(
        node_id="openai-node",
        endpoint="http://ollama",
        tailnet_name="tailnet",
        secret=b"secret",
        allowed_applications=("manuals_rag",),
    )

    class FakeStream:
        def __iter__(self):
            return iter([
                b'{"response":{"message":{"role":"assistant","content":"hello"},"done":false}}\n',
                b'{"response":{"done":true}}\n',
            ])

        def close(self):
            pass

    monkeypatch.setattr(agent, "openai_stream", lambda *args, **kwargs: (FakeStream(), "test-model", "chat"))
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        create_handler(
            agent,
            gateway_url="https://gateway",
            local_broker_only=True,
            local_broker_networks=(ipaddress.ip_network("127.0.0.0/8"),),
            default_application="manuals_rag",
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request_obj = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps({"model": "test-model", "messages": [{"role": "user", "content": "hello"}], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request_obj, timeout=3) as response:
            body = response.read().decode()
            assert response.headers.get_content_type() == "text/event-stream"
        assert '"content": "hello"' in body
        assert '"finish_reason": "stop"' in body
        assert "data: [DONE]" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_openai_endpoint_forwards_without_gateway_credentials(monkeypatch):
    agent = OllamaNode(
        node_id="openai-node",
        endpoint="http://ollama",
        tailnet_name="tailnet",
        secret=b"secret",
        allowed_applications=("manuals_rag",),
    )
    monkeypatch.setattr(
        agent,
        "openai_infer",
        lambda payload, *, operation, application, gateway_url: {
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
    )
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        create_handler(
            agent,
            gateway_url="https://gateway",
            local_broker_only=True,
            local_broker_networks=(ipaddress.ip_network("127.0.0.0/8"),),
            default_application="manuals_rag",
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hello"}]}).encode()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode())
        assert result["choices"][0]["message"]["content"] == "ok"
    finally:
        server.shutdown()
        server.server_close()


def test_gateway_accepts_node_authenticated_inference_only_for_allowed_application(monkeypatch):
    secret = b"node-secret"
    gateway = InferenceGateway(
        node_secrets={"broker": secret},
        post=lambda url, body, timeout: {"message": {"content": "ok"}},
    )
    snapshot = node("broker", allowed=("manuals_rag",))
    raw_heartbeat = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw_heartbeat, heartbeat_signature(raw_heartbeat, secret))
    raw = json.dumps({"node_id": "broker", "application": "manuals_rag"}, sort_keys=True, separators=(",", ":")).encode()
    gateway.authorize_node_inference("broker", raw, heartbeat_signature(raw, secret), application="manuals_rag")
    with pytest.raises(PermissionError, match="not allowed"):
        gateway.authorize_node_inference("broker", raw, heartbeat_signature(raw, secret), application="openclaw")


def test_node_proxy_rejects_stale_and_replayed_inference_envelopes(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_inference.node_agent._json_request",
        lambda *args, **kwargs: {"message": {"content": "ok"}},
    )
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret")
    stale = {
        "operation": "chat",
        "payload": {"model": "qwen3.5:9b", "messages": []},
        "request_id": "stale-1",
        "issued_at": time.time() - 120,
    }
    with pytest.raises(PermissionError, match="stale"):
        agent.infer(stale, node_request_signature(stale, b"secret"))
    current = {**stale, "request_id": "current-1", "issued_at": time.time()}
    agent.infer(current, node_request_signature(current, b"secret"))
    with pytest.raises(PermissionError, match="duplicate"):
        agent.infer(current, node_request_signature(current, b"secret"))


def test_node_agent_refuses_eviction_while_model_is_active():
    agent = OllamaNode(node_id="n1", endpoint="http://ollama", tailnet_name="t", secret=b"secret", proxy_only=True)
    agent._active_by_model["qwen3.5:9b"] = 1
    command = {"action": "evict_model", "model": "qwen3.5:9b", "size_bytes": 0, "issued_at": time.time()}
    try:
        agent.command(command, sign_command(command, b"secret"))
    except PermissionError as error:
        assert "active requests" in str(error)
    else:
        raise AssertionError("active model eviction was accepted")


def test_node_enrollment_is_one_time_and_persistent(tmp_path):
    now = [1000.0]
    state_path = tmp_path / "gateway-state.json"
    manager = EnrollmentManager(state_path, clock=lambda: now[0])
    pairing = manager.create_pairing_token(ttl_seconds=600)
    node = manager.enroll(
        pairing_token=pairing["token"],
        node_id="gpu-a",
        endpoint="http://127.0.0.1:11434",
        agent_url="http://gpu-a.tailnet:8765",
        tailnet_name="tailnet",
    )
    assert len(node.secret) >= 32
    assert state_path.stat().st_mode & 0o077 == 0
    reloaded = EnrollmentManager(state_path, clock=lambda: now[0])
    assert reloaded.nodes()[0].node_id == "gpu-a"
    try:
        reloaded.enroll(
            pairing_token=pairing["token"],
            node_id="gpu-b",
            endpoint="http://127.0.0.1:11434",
            agent_url="http://gpu-b.tailnet:8765",
            tailnet_name="tailnet",
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("pairing token was reusable")


def test_node_endpoint_boundary_rejects_public_hosts_and_accepts_tailnet_addresses():
    validate_node_endpoint("http://100.64.10.20:8765", allow_local_endpoints=False)
    validate_node_endpoint("http://gpu-a.tailnet:8765", allow_local_endpoints=False)
    with pytest.raises(ValueError, match="tailnet"):
        validate_node_endpoint("https://example.com:8765", allow_local_endpoints=False)
    with pytest.raises(ValueError, match="private"):
        validate_node_endpoint("http://8.8.8.8:8765", allow_local_endpoints=False)


def test_gateway_restores_enrolled_node_credentials_after_restart(tmp_path):
    state_path = tmp_path / "gateway-state.json"
    manager = EnrollmentManager(state_path)
    pairing = manager.create_pairing_token()
    enrolled = manager.enroll(
        pairing_token=pairing["token"],
        node_id="gpu-a",
        endpoint="http://127.0.0.1:11434",
        agent_url="http://gpu-a.tailnet:8765",
        tailnet_name="tailnet",
    )
    restarted = InferenceGateway(enrollment=EnrollmentManager(state_path))
    assert restarted.node_secrets["gpu-a"] == enrolled.secret.encode()
    assert restarted.node_agent_urls["gpu-a"] == "http://gpu-a.tailnet:8765"


def test_gateway_maps_embedding_operation_to_ollama_embed_endpoint():
    calls = []
    gateway = InferenceGateway(
        node_secrets={"warm": b"heartbeat-secret"},
        post=lambda url, body, timeout: calls.append((url, body)) or {"embeddings": [[0.1]]},
    )
    snapshot = node("warm", model="qwen3-embedding:0.6b")
    raw = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, b"heartbeat-secret"))
    result = gateway.infer(
        request(model="qwen3-embedding:0.6b", min_vram_mb=1000, required_capabilities=("embeddings",)),
        operation="embeddings",
        payload={"model": "qwen3-embedding:0.6b", "input": ["hello"]},
    )
    assert result["response"]["embeddings"] == [[0.1]]
    assert calls[0][0].endswith("/api/embed")


def test_node_heartbeat_payload_is_signed(monkeypatch):
    agent = OllamaNode(node_id="n1", endpoint="http://127.0.0.1:9", tailnet_name="t", secret=b"secret")
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"status":"accepted"}'

    def fake_open(request, timeout):
        captured["body"] = request.data
        captured["signature"] = request.headers["X-node-signature"]
        return Response()

    monkeypatch.setattr("manuals_rag_inference.node_agent.urlopen", fake_open)
    assert agent.publish_heartbeat("http://gateway") == {"status": "accepted"}
    assert captured["signature"] == heartbeat_signature(captured["body"], b"secret")


def test_gateway_issues_only_configured_signed_node_command():
    calls = []
    gateway = InferenceGateway(node_secrets={"n1": b"secret"}, node_agent_urls={"n1": "http://n1.tailnet:8765"}, post=lambda url, body, timeout: calls.append((url, body)) or {"status": "completed"})
    result = gateway.issue_node_command(node_id="n1", action="pull_model", model="qwen3.5:9b", size_bytes=10)
    assert result["result"]["status"] == "completed"
    assert calls[0][0].endswith("/v1/node/command")
    assert calls[0][1]["command"]["model"] == "qwen3.5:9b"


def test_scheduler_rejects_a_node_when_inference_is_disabled():
    disabled = NodeSnapshot(**{**node("disabled").to_dict(), "inference_enabled": False})
    scheduler = InferenceScheduler()
    scheduler.upsert_node(disabled)
    decision = scheduler.choose(request())
    assert not decision.admitted
    assert "inference_disabled" in decision.reason


def test_node_activity_and_disable_gate_are_recorded(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_inference.node_agent._json_request",
        lambda *args, **kwargs: {"model": "qwen3.5:9b", "message": {"content": "ok"}},
    )
    agent = OllamaNode(node_id="activity", endpoint="http://ollama", tailnet_name="tailnet", secret=b"secret")
    envelope = {
        "operation": "chat",
        "payload": {"model": "qwen3.5:9b", "messages": []},
        "request_id": "activity-r1",
        "issued_at": time.time(),
    }
    assert agent.infer(envelope, node_request_signature(envelope, b"secret"))["message"]["content"] == "ok"
    agent.set_inference_enabled(False)
    disabled_envelope = {**envelope, "request_id": "activity-r2", "issued_at": time.time()}
    with pytest.raises(PermissionError, match="disabled"):
        agent.infer(disabled_envelope, node_request_signature(disabled_envelope, b"secret"))
    statuses = [event["status"] for event in agent.activity()]
    assert "completed" in statuses
    assert "disabled" in statuses
    assert agent.snapshot().inference_enabled is False


def test_node_disconnect_removes_routing_but_preserves_enrollment():
    secret = b"disconnect-secret"
    gateway = InferenceGateway(node_secrets={"n1": secret})
    snapshot = node("n1", model="qwen3.5:9b")
    raw_heartbeat = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    gateway.register_heartbeat(snapshot, raw_heartbeat, heartbeat_signature(raw_heartbeat, secret))
    payload = {"node_id": "n1", "issued_at": time.time()}
    raw_disconnect = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    result = gateway.disconnect_node(node_id="n1", raw_body=raw_disconnect, signature=heartbeat_signature(raw_disconnect, secret))
    assert result == {"node_id": "n1", "disconnected": True, "revoked": False}
    assert gateway.scheduler.snapshots() == ()
    assert gateway.node_secrets["n1"] == secret
