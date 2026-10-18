import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from manuals_rag_inference.enrollment import EnrollmentManager
from manuals_rag_inference.gateway import InferenceGateway, create_gateway_handler, heartbeat_signature
from manuals_rag_inference.node_agent import bind_node_listener, node_agent_url_for_port, listener_endpoint_needs_refresh


def enrolled_gateway(tmp_path, *, enabled=True):
    manager = EnrollmentManager(tmp_path / "gateway-state.json",
                                allowed_host_suffixes=(".ts.net",),
                                allow_local_endpoints=False)
    token = manager.create_pairing_token()["token"]
    manager.enroll(pairing_token=token, node_id="msi-node",
                   endpoint="http://ollama.msi-node.ts.net:11434",
                   agent_url="https://msi-node.ts.net:8765",
                   tailnet_name="msi-node")
    gateway = InferenceGateway(enrollment=manager,
                               allow_node_agent_endpoint_refresh=enabled)
    return manager, gateway, manager.nodes()[0]


def endpoint_payload(node, url, *, nonce="one-time-endpoint-proof"):
    return json.dumps({"node_id": node.node_id, "agent_url": url,
                       "issued_at": time.time(), "nonce": nonce},
                      sort_keys=True, separators=(",", ":")).encode()


def test_listener_refresh_is_off_by_default(tmp_path):
    _manager, gateway, node = enrolled_gateway(tmp_path, enabled=False)
    raw = endpoint_payload(node, "https://msi-node.ts.net:8766")
    signature = heartbeat_signature(raw, node.secret.encode())
    with pytest.raises(PermissionError, match="disabled"):
        gateway.refresh_node_agent_endpoint(raw, signature)


def test_authenticated_same_host_port_refresh_persists_and_rejects_replay(tmp_path):
    _manager, gateway, node = enrolled_gateway(tmp_path)
    raw = endpoint_payload(node, "https://msi-node.ts.net:8769")
    signature = heartbeat_signature(raw, node.secret.encode())
    assert gateway.refresh_node_agent_endpoint(raw, signature) == {
        "status": "accepted", "node_id": "msi-node"}
    assert gateway.node_agent_urls[node.node_id] == "https://msi-node.ts.net:8769"

    restored = EnrollmentManager(tmp_path / "gateway-state.json",
                                 allowed_host_suffixes=(".ts.net",),
                                 allow_local_endpoints=False)
    assert restored.nodes()[0].agent_url == "https://msi-node.ts.net:8769"
    restarted_gateway = InferenceGateway(enrollment=restored,
                                        allow_node_agent_endpoint_refresh=True)
    with pytest.raises(PermissionError, match="replayed"):
        restarted_gateway.refresh_node_agent_endpoint(raw, signature)


def test_enrolled_endpoint_refresh_wins_over_static_bootstrap_mapping_after_restart(tmp_path):
    manager, gateway, node = enrolled_gateway(tmp_path)
    raw = endpoint_payload(node, "https://msi-node.ts.net:8769")
    gateway.refresh_node_agent_endpoint(raw, heartbeat_signature(raw, node.secret.encode()))

    restored = EnrollmentManager(tmp_path / "gateway-state.json",
                                 allowed_host_suffixes=(".ts.net",),
                                 allow_local_endpoints=False)
    restarted = InferenceGateway(
        enrollment=restored,
        node_agent_urls={node.node_id: "https://msi-node.ts.net:8765"},
        allow_node_agent_endpoint_refresh=True,
    )
    assert restarted.node_agent_urls[node.node_id] == "https://msi-node.ts.net:8769"


def test_fallback_to_primary_restart_refreshes_gateway_endpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from manuals_rag_inference import node_agent

    config = {
        "node_id": "msi-node", "endpoint": "http://127.0.0.1:11434",
        "tailnet_name": "msi-node", "port": 8765,
        "agent_url": "https://msi-node.ts.net:8766",
        "gateway_url": "https://gateway.ts.net", "shared_secret": "fixture-secret",
    }
    saved = []
    class Server:
        def __init__(self, port): self.server_port = port
        def server_close(self): pass

    class Agent:
        def __init__(self, **_kwargs):
            self.app_metadata = {}
            self.allowed_applications = ("manuals_rag",)
        def set_inference_enabled(self, _enabled): pass
        def publish_heartbeat(self, _gateway): pass

    class NoStartThread:
        def __init__(self, **_kwargs): pass
        def start(self): pass

    servers = iter((Server(8766), Server(8765), Server(8765)))
    fallback_options = []
    excluded = []
    refreshed = []
    monkeypatch.setattr(node_agent, "_load_node_config", lambda _path: config)
    monkeypatch.setattr(node_agent, "_save_node_config", lambda _path, value: saved.append(dict(value)))
    monkeypatch.setattr(node_agent, "_detect_tailscale_ipv4", lambda: None)
    monkeypatch.setattr(node_agent, "OllamaNode", Agent)
    monkeypatch.setattr(node_agent, "create_handler", lambda *args, **kwargs: object())
    monkeypatch.setattr(node_agent, "create_openai_broker_server", lambda *args, **kwargs: Server(kwargs["port"]))
    def capture_fallback(*args, **kwargs):
        fallback_options.append(kwargs["allow_fallback"])
        excluded.append(kwargs["excluded_ports"])
        return next(servers)
    monkeypatch.setattr(node_agent, "bind_node_listener", capture_fallback)
    monkeypatch.setattr(node_agent, "publish_agent_endpoint", lambda gateway, node, secret, url: refreshed.append(url))
    monkeypatch.setattr(node_agent.threading, "Thread", NoStartThread)
    monkeypatch.setenv("INFERENCE_NODE_BIND", "127.0.0.1")
    monkeypatch.setenv("INFERENCE_NODE_PORT", "8765")
    monkeypatch.delenv("INFERENCE_NODE_ADVERTISE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_NODE_ID", raising=False)
    monkeypatch.delenv("INFERENCE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("INFERENCE_NODE_ALLOW_PORT_FALLBACK", raising=False)

    first, _, _ = node_agent.create_node_server(str(tmp_path / "node.json"))
    assert fallback_options == [True]
    assert excluded == [{8767}]
    assert refreshed == []
    assert config["agent_url"] == "https://msi-node.ts.net:8766"
    second, _, _ = node_agent.create_node_server(str(tmp_path / "node.json"))
    assert fallback_options == [True, True]
    assert refreshed == ["https://msi-node.ts.net:8765"]
    assert config["agent_url"] == "https://msi-node.ts.net:8765"
    monkeypatch.setenv("INFERENCE_NODE_ALLOW_PORT_FALLBACK", "false")
    third, _, _ = node_agent.create_node_server(str(tmp_path / "node.json"))
    assert fallback_options == [True, True, False]
    first.server_close(); second.server_close(); third.server_close()


def test_automatic_listener_fallback_skips_local_broker_port():
    attempts = []

    class Server:
        def __init__(self, address, _handler):
            self.server_port = address[1]

    def bind(address, handler):
        attempts.append(address[1])
        return Server(address, handler)

    server = bind_node_listener("100.64.0.8", 8767, object(), allow_fallback=True,
                                excluded_ports={8767}, server_factory=bind)
    assert attempts == [8768]
    assert server.server_port == 8768


def test_authenticated_endpoint_refresh_http_route(tmp_path):
    _manager, gateway, node = enrolled_gateway(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_gateway_handler(gateway))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        raw = endpoint_payload(node, "https://msi-node.ts.net:8766")
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/nodes/agent-endpoint",
                          data=raw, method="POST", headers={
                              "Content-Type": "application/json",
                              "X-Node-Signature": heartbeat_signature(raw, node.secret.encode()),
                          })
        with urlopen(request, timeout=3) as response:
            assert json.load(response) == {"status": "accepted", "node_id": node.node_id}
        assert gateway.node_agent_urls[node.node_id] == "https://msi-node.ts.net:8766"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("url", [
    "https://another-node.ts.net:8766",
    "https://msi-node.ts.net:8773",
    "https://user@msi-node.ts.net:8766",
    "https://msi-node.ts.net:8766/path",
    "https://msi-node.ts.net:443",
])
def test_listener_refresh_rejects_host_substitution_and_unbounded_url(tmp_path, url):
    _manager, gateway, node = enrolled_gateway(tmp_path)
    raw = endpoint_payload(node, url, nonce="distinct-endpoint-proof")
    signature = heartbeat_signature(raw, node.secret.encode())
    before = gateway.node_agent_urls[node.node_id]
    with pytest.raises((PermissionError, ValueError)):
        gateway.refresh_node_agent_endpoint(raw, signature)
    assert gateway.node_agent_urls[node.node_id] == before


def test_listener_refresh_rejects_invalid_signature(tmp_path):
    _manager, gateway, node = enrolled_gateway(tmp_path)
    raw = endpoint_payload(node, "https://msi-node.ts.net:8766")
    with pytest.raises(PermissionError, match="authentication"):
        gateway.refresh_node_agent_endpoint(raw, "invalid")
    assert gateway.node_agent_urls[node.node_id] == node.agent_url


def test_listener_fallback_is_bounded_and_does_not_stop_existing_processes():
    attempts = []

    def exhausted(address, _handler):
        attempts.append(address[1])
        raise OSError("address already in use")

    with pytest.raises(OSError, match="address already in use"):
        bind_node_listener("100.64.0.8", 8765, object(), allow_fallback=True,
                           server_factory=exhausted)
    assert attempts == list(range(8765, 8773))


def test_listener_fallback_off_tries_only_configured_port():
    attempts = []

    def occupied(address, _handler):
        attempts.append(address[1])
        raise OSError("address already in use")

    with pytest.raises(OSError):
        bind_node_listener("100.64.0.8", 8765, object(), server_factory=occupied)
    assert attempts == [8765]


def test_listener_fallback_binds_first_available_port():
    attempts = []

    class Server:
        def __init__(self, address, _handler):
            self.server_port = address[1]

    def occupied_twice_then_bind(address, handler):
        attempts.append(address[1])
        if len(attempts) < 3:
            raise OSError("address already in use")
        return Server(address, handler)

    server = bind_node_listener("100.64.0.8", 8765, object(), allow_fallback=True,
                                server_factory=occupied_twice_then_bind)
    assert attempts == [8765, 8766, 8767]
    assert server.server_port == 8767


def test_listener_fallback_rejects_port_without_full_bounded_window():
    with pytest.raises(ValueError, match="eight-port window"):
        bind_node_listener("100.64.0.8", 65529, object(), allow_fallback=True)


def test_advertised_url_uses_actual_listener_port_only():
    assert node_agent_url_for_port("https://msi-node.ts.net:8765", 8767) == "https://msi-node.ts.net:8767"
    with pytest.raises(ValueError):
        node_agent_url_for_port("https://other.ts.net:8765/path", 8767)


def test_listener_endpoint_comparison_handles_return_to_primary_port():
    assert listener_endpoint_needs_refresh("https://msi-node.ts.net:8766", 8765)
    assert not listener_endpoint_needs_refresh("https://msi-node.ts.net:8765", 8765)
