import json
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from manuals_rag_inference.enrollment import EnrollmentManager
from manuals_rag_inference.gateway import InferenceGateway, create_gateway_handler, heartbeat_signature
from manuals_rag_inference.models import GpuSnapshot, LoadedModel, ModelSpec, NodeSnapshot
from manuals_rag_inference.scheduler import InferenceScheduler, SchedulerPolicy
from manuals_rag_inference.node_agent import OllamaNode, create_handler
from manuals_rag_inference.endpoint_ui import copy_endpoint_to_clipboard, local_broker_endpoints


@contextmanager
def serving(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def broker(tmp_path):
    enrollment = EnrollmentManager(tmp_path / "state.json")
    scheduler = InferenceScheduler(policy=SchedulerPolicy(require_application_allowlist=True))
    gateway = InferenceGateway(scheduler, enrollment=enrollment, require_node_agents=True,
        require_model_catalog=True, model_catalog={"test-model": ModelSpec(1000, 100, ("chat", "embeddings"))},
        gateway_token="operator-test-token", require_application_identity=True,
        application_tokens={"manuals_rag": "application-test-token"}, node_default_applications=("manuals_rag",))
    for node_id in ("ingress", "worker"):
        gateway.enroll_node(pairing_token=enrollment.create_pairing_token()["token"], node_id=node_id,
            endpoint="http://127.0.0.1:11434", agent_url="http://127.0.0.1:8765", tailnet_name="test")
        snapshot = NodeSnapshot(node_id, "http://127.0.0.1:11434", "test",
            (GpuSnapshot("0", "test-gpu", 24000, 0),) if node_id == "worker" else (),
            (LoadedModel("test-model", 1000),) if node_id == "worker" else (),
            allowed_applications=("manuals_rag",), observed_at_epoch=time.time())
        raw = json.dumps(snapshot.to_dict()).encode()
        gateway.register_heartbeat(snapshot, raw, heartbeat_signature(raw, gateway.node_secrets[node_id]))
    gateway._post = lambda *args, **kwargs: {"message": {"role": "assistant", "content": "fixture"}}
    return gateway


def envelope(**overrides):
    body = {"node_id": "ingress", "issued_at": time.time(),
        "request": {"request_id": str(uuid.uuid4()), "application": "manuals_rag", "model": "test-model"},
        "operation": "chat", "payload": {"model": "test-model", "messages": [], "stream": False}}
    body.update(overrides)
    return body


def call(url, gateway, body, *, signature=None):
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    signature = signature if signature is not None else heartbeat_signature(raw, gateway.node_secrets["ingress"])
    return urlopen(Request(url + "/v1/nodes/inference", data=raw,
        headers={"X-Node-Signature": signature, "Content-Type": "application/json"}), timeout=4)


def disconnect_client(response):
    # Explicitly close the TCP connection, rather than only its buffered reader.
    response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    response.close()


def test_endpoint_display_uses_local_broker_port_and_copy_allowlist():
    endpoints = local_broker_endpoints("8769")
    assert endpoints == (
        "http://127.0.0.1:8769/v1",
        "http://host.docker.internal:8769/v1",
    )
    assert local_broker_endpoints("invalid") == local_broker_endpoints(8767)
    clipboard = []
    actions = []
    assert copy_endpoint_to_clipboard(
        endpoints[1], endpoints,
        clipboard_clear=lambda: actions.append("clear"),
        clipboard_append=clipboard.append,
        update=lambda: actions.append("update"),
    )
    assert clipboard == [endpoints[1]]
    assert actions == ["clear", "update"]
    assert not copy_endpoint_to_clipboard(
        "https://gateway.example.test/?token=secret", endpoints,
        clipboard_clear=lambda: pytest.fail("unknown URL must not clear clipboard"),
        clipboard_append=lambda _: pytest.fail("unknown URL must not be copied"),
        update=lambda: pytest.fail("unknown URL must not update clipboard"),
    )


def test_signed_updater_app_explains_shared_compute_endpoint():
    source = Path(__file__).parents[2] / "scripts" / "run_inference_node_client.py"
    text = source.read_text(encoding="utf-8")
    assert "PROVIS shared compute — OpenAI-compatible endpoint" in text
    assert 'text="Copy URL"' in text
    assert "employee-built workflows and apps can use it too" in text
    assert "This is a local PC endpoint" in text
    assert "local_broker_endpoints(broker_port)" in text


def test_node_auth_without_gateway_tokens_selects_different_worker(broker):
    with serving(create_gateway_handler(broker)) as url:
        with call(url, broker, envelope()) as response:
            result = json.load(response)
        assert result["routing"]["node_id"] == "worker"
        assert result["response"]["message"]["content"] == "fixture"
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(url + "/v1/inference", data=b'{}'))
        assert rejected.value.code in (400, 401)
    assert not broker.scheduler._reservations


@pytest.mark.parametrize("case,status", [("wrong_node", 401), ("signature", 401), ("stale", 401),
    ("nan", 401), ("replay", 401), ("unhealthy", 403), ("disabled", 403), ("old_heartbeat", 403),
    ("application", 403), ("broadened_application", 403), ("unknown_model", 400), ("missing_model", 400),
    ("bad_limits", 400), ("stream_embeddings", 400)])
def test_ingress_rejections(broker, case, status):
    body = envelope()
    signature = None
    if case == "wrong_node": body["node_id"] = "unknown"
    if case == "signature": signature = "wrong"
    if case == "stale": body["issued_at"] -= 100
    if case == "nan": body["issued_at"] = float("nan")
    if case in {"unhealthy", "disabled", "old_heartbeat", "broadened_application"}:
        node = next(node for node in broker.scheduler.snapshots() if node.node_id == "ingress")
        values = {"unhealthy": {"healthy": False}, "disabled": {"inference_enabled": False},
            "old_heartbeat": {"observed_at_epoch": time.time() - 100},
            "broadened_application": {"allowed_applications": ("manuals_rag", "unauthorized")}}[case]
        broker.scheduler.upsert_node(replace(node, **values))
    if case in {"application", "broadened_application"}: body["request"]["application"] = "unauthorized"
    if case == "unknown_model": body["request"]["model"] = "unknown"
    if case == "missing_model": del body["request"]["model"]
    if case == "bad_limits": body["request"]["max_latency_ms"] = "unbounded"
    if case == "stream_embeddings":
        body["operation"] = "embeddings"
        body["payload"]["stream"] = True
    with serving(create_gateway_handler(broker)) as url:
        if case == "replay":
            call(url, broker, body).close()
            # Replay remains rejected after the persistent manager reloads.
            broker.enrollment = EnrollmentManager(broker.enrollment.state_path)
        with pytest.raises(HTTPError) as rejected:
            call(url, broker, body, signature=signature)
        assert rejected.value.code == status
    assert not broker.scheduler._reservations


def test_catalog_overrides_caller_resource_claims(broker):
    body = envelope()
    body["request"].update(model_size_bytes=1, min_vram_mb=1, required_capabilities=["vision"])
    with serving(create_gateway_handler(broker)) as url:
        with pytest.raises(HTTPError) as rejected:
            call(url, broker, body)
        assert rejected.value.code == 503
    assert not broker.scheduler._reservations


def test_direct_application_authentication_is_unchanged(broker):
    body = envelope()
    body["request"].update(workload="test", model_size_bytes=1, min_vram_mb=1)
    with serving(create_gateway_handler(broker)) as url:
        for headers in ({}, {"X-Gateway-Token": "operator-test-token"}):
            with pytest.raises(HTTPError) as rejected:
                urlopen(Request(url + "/v1/inference", data=json.dumps(body).encode(), headers=headers))
            assert rejected.value.code == 401
        headers = {"X-Gateway-Token": "operator-test-token", "X-Inference-Application-Token": "application-test-token"}
        with urlopen(Request(url + "/v1/inference", data=json.dumps(body).encode(), headers=headers)) as response:
            assert json.load(response)["routing"]["node_id"] == "worker"
    assert not broker.scheduler._reservations


@pytest.mark.parametrize("missing", [False, True])
def test_gateway_prepares_before_inference_without_unneeded_pull(broker, missing):
    snapshot = next(node for node in broker.scheduler.snapshots() if node.node_id == "worker")
    broker.scheduler.upsert_node(replace(snapshot, loaded_models=(), installed_models=() if missing else ("test-model",)))
    operations = []
    def post(url, body, **kwargs):
        if url.endswith("/command"):
            action = body["command"]["action"]
            operations.append(action)
            return {"status": "completed", "loaded": action == "warm_model"}
        operations.append("inference")
        return {"message": {"content": "fixture"}}
    broker._post = post
    body = envelope()
    body["request"]["allow_model_pull"] = missing
    with serving(create_gateway_handler(broker)) as url:
        with call(url, broker, body) as response:
            assert response.status == 200
    assert operations == (["pull_model", "warm_model", "inference"] if missing else ["warm_model", "inference"])
    assert not broker.scheduler._reservations


@pytest.mark.parametrize("failure_phase", ["model_pull", "model_warm"])
def test_gateway_preparation_error_phase_is_safe_and_logged(broker, caplog, failure_phase):
    import io
    snapshot = next(node for node in broker.scheduler.snapshots() if node.node_id == "worker")
    broker.scheduler.upsert_node(replace(snapshot, loaded_models=(), installed_models=("test-model",) if failure_phase == "model_warm" else ()))
    def post(url, body, **kwargs):
        raise HTTPError(url, 502, "private raw detail", {}, io.BytesIO(b'{"upstream_status":500,"code":"private secret prompt"}'))
    broker._post = post
    body = envelope()
    body["request"]["allow_model_pull"] = True
    with serving(create_gateway_handler(broker)) as url:
        with pytest.raises(HTTPError) as rejected:
            call(url, broker, body)
        result = json.load(rejected.value)
    assert result["phase"] == failure_phase
    assert result["code"] == "ollama_http_500"
    assert result["model"] == "test-model"
    assert failure_phase in caplog.text
    assert "private" not in caplog.text
    assert "secret" not in json.dumps(result)
    assert not broker.scheduler._reservations


@pytest.mark.parametrize("mode", ["empty", "error", "buffered"])
def test_pre_token_failures_are_controlled_503(broker, mode):
    class Worker(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if mode == "buffered" else "application/x-ndjson")
            self.end_headers()
            if mode == "error": self.wfile.write(b'{"error":"private upstream detail"}\n')
            if mode == "buffered": self.wfile.write(b'{"message":{"content":"buffered"},"done":true}')
        def log_message(self, *args): pass
    with serving(Worker) as worker_url, serving(create_gateway_handler(broker)) as gateway_url:
        broker.node_agent_urls["worker"] = worker_url
        body = envelope()
        body["payload"]["stream"] = True
        with pytest.raises(HTTPError) as rejected:
            call(gateway_url, broker, body)
        assert rejected.value.code == 503
        assert "private upstream detail" not in rejected.value.read().decode()
        assert not broker.scheduler._reservations


@pytest.mark.parametrize("mode", ["complete", "disconnect", "upstream_error", "timeout"])
def test_stream_incremental_and_reservation_cleanup(broker, mode):
    continue_stream = threading.Event()
    forwarded = []

    class Worker(BaseHTTPRequestHandler):
        def do_POST(self):
            forwarded.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"message":{"content":"first"},"done":false}\n')
            self.wfile.flush()
            continue_stream.wait(10)
            try:
                if mode == "upstream_error": return
                self.wfile.write(b'{"message":{"content":"second"},"done":false}\n')
                self.wfile.write(b'{"done":true}\n')
                self.wfile.flush()
            except OSError:
                pass
        def log_message(self, *args): pass

    try:
        with serving(Worker) as worker_url, serving(create_gateway_handler(broker)) as gateway_url:
            broker.node_agent_urls["worker"] = worker_url
            body = envelope()
            body["payload"]["stream"] = True
            if mode == "timeout": body["request"]["max_latency_ms"] = 250
            response = call(gateway_url, broker, body)
            first = json.loads(response.readline())
            assert first["response"]["message"]["content"] == "first"
            assert first["routing"]["node_id"] == "worker"
            assert broker.scheduler._reservations
            if mode == "disconnect":
                disconnect_client(response)
            else:
                if mode != "timeout": continue_stream.set()
                chunks = [json.loads(line) for line in response]
                if mode == "complete": assert chunks[-1]["response"]["done"] is True
                else: assert chunks[-1]["error"]["code"] == "upstream_stream_failed"
                response.close()
            deadline = time.monotonic() + 2
            while broker.scheduler._reservations and time.monotonic() < deadline:
                time.sleep(.02)
            assert not broker.scheduler._reservations
            assert len(forwarded) == 1  # Never retry after the first token.
    finally:
        continue_stream.set()


@pytest.mark.parametrize("disconnect", [False, True])
def test_real_node_streams_ollama_incrementally_and_cleans_accounting(broker, disconnect):
    finish = threading.Event()
    class Ollama(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/api/chat" and payload["stream"] is True
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"message":{"content":"first"},"done":false}\n')
            self.wfile.flush()
            finish.wait(2)
            try:
                self.wfile.write(b'{"message":{"content":"second"},"done":false}\n{"done":true}\n')
                self.wfile.flush()
            except OSError:
                pass
        def log_message(self, *args): pass
    try:
        with serving(Ollama) as ollama_url:
            agent = OllamaNode(node_id="worker", endpoint=ollama_url, tailnet_name="test", secret=broker.node_secrets["worker"])
            with serving(create_handler(agent)) as node_url, serving(create_gateway_handler(broker)) as gateway_url:
                broker.node_agent_urls["worker"] = node_url
                body = envelope()
                body["payload"]["stream"] = True
                response = call(gateway_url, broker, body)
                assert json.loads(response.readline())["response"]["message"]["content"] == "first"
                assert agent._active_by_model == {"test-model": 1}
                if disconnect:
                    disconnect_client(response)
                else:
                    finish.set()
                    assert [json.loads(line) for line in response][-1]["response"]["done"] is True
                    response.close()
                deadline = time.monotonic() + 2
                while (broker.scheduler._reservations or agent._active_by_model) and time.monotonic() < deadline:
                    time.sleep(.02)
                assert not agent._active_by_model
                assert not broker.scheduler._reservations
    finally:
        finish.set()
