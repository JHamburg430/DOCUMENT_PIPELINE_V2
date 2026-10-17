import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from manuals_rag_inference.enrollment import EnrollmentManager
from manuals_rag_inference.gateway import InferenceGateway, create_gateway_handler
from manuals_rag_inference.models import GpuSnapshot, ModelSpec
from manuals_rag_inference.scheduler import InferenceScheduler, SchedulerPolicy
from manuals_rag_inference.node_agent import OllamaNode, create_handler
import manuals_rag_inference.node_agent as node_agent_module


class FakeOllama:
    def __init__(self):
        self.models = set()
        self.calls = []

    def handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or "0")).decode())

            def do_GET(self):  # noqa: N802
                if self.path == "/api/ps":
                    self._send({"models": [{"name": model, "size_vram": 4_000_000_000} for model in owner.models]})
                else:
                    self.send_error(404)

            def do_POST(self):  # noqa: N802
                payload = self._body()
                owner.calls.append((self.path, payload))
                if self.path == "/api/pull":
                    owner.models.add(payload["model"])
                    self._send({"status": "success"})
                elif self.path == "/api/chat":
                    self._send({"model": payload["model"], "message": {"role": "assistant", "content": "grounded response"}})
                elif self.path == "/api/generate":
                    self._send({"model": payload["model"], "response": "generated response"})
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                return

        return Handler


def _start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_enrollment_heartbeat_pull_and_inference_http_path(monkeypatch):
    fake_ollama = FakeOllama()
    ollama_server = ThreadingHTTPServer(("127.0.0.1", 0), fake_ollama.handler())
    ollama_thread = _start(ollama_server)

    placeholder = OllamaNode(
        node_id="gpu-a",
        endpoint=f"http://127.0.0.1:{ollama_server.server_port}",
        tailnet_name="test-tailnet",
        secret=b"placeholder",
        proxy_only=True,
    )
    node_server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(placeholder))
    node_thread = _start(node_server)

    enrollment = EnrollmentManager()
    gateway = InferenceGateway(gateway_token="gateway-token", enrollment=enrollment)
    pairing = gateway.create_pairing_token()
    enrolled = gateway.enroll_node(
        pairing_token=pairing["token"],
        node_id="gpu-a",
        endpoint=placeholder.endpoint,
        agent_url=f"http://127.0.0.1:{node_server.server_port}",
        tailnet_name="test-tailnet",
    )
    placeholder.secret = enrolled["shared_secret"].encode()
    monkeypatch.setattr(node_agent_module, "_gpu_snapshots", lambda: (GpuSnapshot("0", "test-gpu", 24000, 2000),))

    gateway_server = ThreadingHTTPServer(("127.0.0.1", 0), create_gateway_handler(gateway))
    gateway_thread = _start(gateway_server)
    try:
        placeholder.publish_heartbeat(f"http://127.0.0.1:{gateway_server.server_port}")
        body = {
            "request": {
                "request_id": "http-r1",
                "application": "manuals_rag",
                "workload": "test",
                "model": "qwen3.5:9b",
                "model_size_bytes": 10_000_000_000,
                "min_vram_mb": 8000,
                "required_capabilities": ["chat"],
                "allow_model_pull": True,
            },
            "operation": "chat",
            "payload": {"model": "qwen3.5:9b", "messages": [], "stream": False},
        }
        request = Request(
            f"http://127.0.0.1:{gateway_server.server_port}/v1/inference",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Gateway-Token": "gateway-token"},
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode())
        assert result["routing"]["node_id"] == "gpu-a"
        assert result["response"]["message"]["content"] == "grounded response"
        # The scheduler must observe the pull on the next signed heartbeat
        # before a subsequent request can rely on the model being warm.
        placeholder.publish_heartbeat(f"http://127.0.0.1:{gateway_server.server_port}")
        generate_body = {
            "request": {
                "request_id": "http-r2",
                "application": "manuals_rag",
                "workload": "question_generation",
                "model": "qwen3.5:9b",
                "model_size_bytes": 10_000_000_000,
                "min_vram_mb": 8000,
                "required_capabilities": ["chat"],
                "allow_model_pull": False,
            },
            "operation": "generate",
            "payload": {"model": "qwen3.5:9b", "prompt": "make a question", "stream": False},
        }
        generate_request = Request(
            f"http://127.0.0.1:{gateway_server.server_port}/v1/inference",
            data=json.dumps(generate_body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Gateway-Token": "gateway-token"},
        )
        with urlopen(generate_request, timeout=5) as response:
            generated = json.loads(response.read().decode())
        assert generated["routing"]["node_id"] == "gpu-a"
        assert generated["response"]["response"] == "generated response"
        assert [path for path, _payload in fake_ollama.calls] == ["/api/pull", "/api/chat", "/api/generate"]
    finally:
        gateway_server.shutdown()
        node_server.shutdown()
        ollama_server.shutdown()
        gateway_server.server_close()
        node_server.server_close()
        ollama_server.server_close()
        gateway_thread.join(timeout=2)
        node_thread.join(timeout=2)
        ollama_thread.join(timeout=2)


def test_two_node_http_failover_rejects_stale_node(monkeypatch):
    fake_a = FakeOllama()
    fake_b = FakeOllama()
    fake_a.models.add("qwen3.5:9b")
    fake_b.models.add("qwen3.5:9b")
    ollama_a = ThreadingHTTPServer(("127.0.0.1", 0), fake_a.handler())
    ollama_b = ThreadingHTTPServer(("127.0.0.1", 0), fake_b.handler())
    ollama_a_thread = _start(ollama_a)
    ollama_b_thread = _start(ollama_b)
    node_a = OllamaNode(
        node_id="gpu-a",
        endpoint=f"http://127.0.0.1:{ollama_a.server_port}",
        tailnet_name="test-tailnet",
        secret=b"placeholder-a",
        proxy_only=True,
    )
    node_b = OllamaNode(
        node_id="gpu-b",
        endpoint=f"http://127.0.0.1:{ollama_b.server_port}",
        tailnet_name="test-tailnet",
        secret=b"placeholder-b",
        proxy_only=True,
    )
    node_server_a = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(node_a))
    node_server_b = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(node_b))
    node_thread_a = _start(node_server_a)
    node_thread_b = _start(node_server_b)
    monkeypatch.setattr(node_agent_module, "_gpu_snapshots", lambda: (GpuSnapshot("0", "test-gpu", 24000, 2000),))
    enrollment = EnrollmentManager()
    gateway = InferenceGateway(
        scheduler=InferenceScheduler(policy=SchedulerPolicy(heartbeat_ttl_seconds=0.5)),
        gateway_token="gateway-token",
        enrollment=enrollment,
        require_node_agents=True,
        model_catalog={"qwen3.5:9b": ModelSpec(10_000_000_000, 8000, ("chat",))},
    )
    gateway_server = ThreadingHTTPServer(("127.0.0.1", 0), create_gateway_handler(gateway))
    gateway_thread = _start(gateway_server)
    try:
        for node, node_server in ((node_a, node_server_a), (node_b, node_server_b)):
            if node.node_id == "gpu-b":
                time.sleep(0.7)
            pairing = gateway.create_pairing_token()
            enrolled = gateway.enroll_node(
                pairing_token=pairing["token"],
                node_id=node.node_id,
                endpoint=node.endpoint,
                agent_url=f"http://127.0.0.1:{node_server.server_port}",
                tailnet_name="test-tailnet",
            )
            node.secret = enrolled["shared_secret"].encode()
            node.publish_heartbeat(f"http://127.0.0.1:{gateway_server.server_port}")

        body = {
            "request": {
                "request_id": "failover-http-1",
                "application": "manuals_rag",
                "workload": "interactive",
                "model": "qwen3.5:9b",
                "model_size_bytes": 10_000_000_000,
                "min_vram_mb": 8000,
                "required_capabilities": ["chat"],
                "allow_model_pull": False,
            },
            "operation": "chat",
            "payload": {"model": "qwen3.5:9b", "messages": [], "stream": False},
        }
        request = Request(
            f"http://127.0.0.1:{gateway_server.server_port}/v1/inference",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Gateway-Token": "gateway-token"},
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode())
        assert result["routing"]["node_id"] == "gpu-b"
        assert [path for path, _payload in fake_a.calls] == []
        assert [path for path, _payload in fake_b.calls] == ["/api/chat"]
    finally:
        gateway_server.shutdown()
        node_server_a.shutdown()
        node_server_b.shutdown()
        ollama_a.shutdown()
        ollama_b.shutdown()
        gateway_server.server_close()
        node_server_a.server_close()
        node_server_b.server_close()
        ollama_a.server_close()
        ollama_b.server_close()
        gateway_thread.join(timeout=2)
        node_thread_a.join(timeout=2)
        node_thread_b.join(timeout=2)
        ollama_a_thread.join(timeout=2)
        ollama_b_thread.join(timeout=2)
