from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from json import loads
from pathlib import Path
import re
from threading import Thread
from time import monotonic
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from apps.ui import server as ui_server


UI_DIR = Path(__file__).resolve().parents[2] / "apps" / "ui"


def _serve(handler_cls):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


class UpstreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_HEAD(self):
        self.send_response(204)
        self.send_header("X-Upstream", "head-ok")
        self.end_headers()


class StreamingUpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length") or "0")
        if content_length:
            self.rfile.read(content_length)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write(b'{"event":"eval_queued","run_id":"run-1"}\n')
        self.wfile.flush()
        sleep(0.4)
        self.wfile.write(b'{"event":"eval_started","run_id":"run-1"}\n')
        self.wfile.flush()


class UiHandler(ui_server.ManualsRagUiHandler):
    pass


def test_static_assets_are_served_with_webview_safe_cache_headers():
    httpd = _serve(UiHandler)
    try:
        with urlopen(f"http://127.0.0.1:{httpd.server_port}/app.js", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
            assert response.headers["Pragma"] == "no-cache"
            assert response.headers["Expires"] == "0"
    finally:
        httpd.shutdown()


def test_index_cache_buster_matches_app_asset_version():
    app_js = (UI_DIR / "app.js").read_text()
    index_html = (UI_DIR / "index.html").read_text()

    asset_version = re.search(r'ASSET_VERSION = "([^"]+)"', app_js).group(1)
    assert f"/app.js?v={asset_version}" in index_html
    assert f"/styles.css?v={asset_version}" in index_html


def test_progress_steps_have_clickable_detail_disclosures():
    app_js = (UI_DIR / "app.js").read_text()
    styles_css = (UI_DIR / "styles.css").read_text()

    assert "data-progress-step" in app_js
    assert "setupProgressInteractions()" in app_js
    assert "renderProgressStepDetails" in app_js
    assert "renderProgressPayload" in app_js
    assert "payloadObject" in app_js
    assert "No step payload was received" in app_js
    assert "progressState.details" in app_js
    assert "formatProgressPayload" in app_js
    assert ".progress-details" in styles_css
    assert ".progress-sample" in styles_css
    assert ".progress-payload" in styles_css


def test_api_proxy_keeps_manuals_rag_same_origin(monkeypatch):
    upstream = _serve(UpstreamHandler)
    monkeypatch.setattr(ui_server, "API_BASE", f"http://127.0.0.1:{upstream.server_port}")
    httpd = _serve(UiHandler)
    try:
        with urlopen(f"http://127.0.0.1:{httpd.server_port}/api/debug/documents?limit=1", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
            assert response.read() == b'{"ok":true}'
    finally:
        httpd.shutdown()
        upstream.shutdown()


def test_api_proxy_supports_head_requests(monkeypatch):
    upstream = _serve(UpstreamHandler)
    monkeypatch.setattr(ui_server, "API_BASE", f"http://127.0.0.1:{upstream.server_port}")
    httpd = _serve(UiHandler)
    try:
        request = Request(f"http://127.0.0.1:{httpd.server_port}/api/health", method="HEAD")
        with urlopen(request, timeout=5) as response:
            assert response.status == 204
            assert response.headers["X-Upstream"] == "head-ok"
            assert response.read() == b""
    finally:
        httpd.shutdown()
        upstream.shutdown()


def test_api_proxy_reports_upstream_failures_as_json_502(monkeypatch):
    monkeypatch.setattr(ui_server, "API_BASE", "http://127.0.0.1:9")
    httpd = _serve(UiHandler)
    try:
        try:
            urlopen(f"http://127.0.0.1:{httpd.server_port}/api/debug/documents", timeout=5)
        except HTTPError as error:
            assert error.code == 502
            assert error.headers["Content-Type"] == "application/json"
            payload = loads(error.read())
            assert payload["detail"].startswith("Manuals RAG API proxy failed:")
        else:
            raise AssertionError("Expected proxy failure")
    finally:
        httpd.shutdown()


def test_api_proxy_flushes_ndjson_stream_lines(monkeypatch):
    upstream = _serve(StreamingUpstreamHandler)
    monkeypatch.setattr(ui_server, "API_BASE", f"http://127.0.0.1:{upstream.server_port}")
    httpd = _serve(UiHandler)
    try:
        request = Request(
            f"http://127.0.0.1:{httpd.server_port}/api/eval/end-to-end-stream?sample_limit=5",
            data=b'{"corpus_ids":["corpus"]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = monotonic()
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/x-ndjson"
            first_line = response.readline()
            elapsed = monotonic() - started
            second_line = response.readline()

        assert elapsed < 0.3
        assert loads(first_line)["event"] == "eval_queued"
        assert loads(second_line)["event"] == "eval_started"
    finally:
        httpd.shutdown()
        upstream.shutdown()
