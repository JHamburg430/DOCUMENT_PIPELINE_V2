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


def test_eval_matrix_view_is_available():
    app_js = (UI_DIR / "app.js").read_text()
    index_html = (UI_DIR / "index.html").read_text()
    styles_css = (UI_DIR / "styles.css").read_text()

    assert 'data-tab="matrix"' in index_html
    assert 'id="matrix-run-all-bank"' in index_html
    assert 'id="matrix-run-column"' in index_html
    assert 'id="matrix-use-model-judge"' in index_html
    assert 'id="matrix-summary"' in index_html
    assert 'id="matrix-table"' in index_html
    assert "MATRIX_STAGES" in app_js
    assert "setupMatrixControls()" in app_js
    assert "startMatrixJob" in app_js
    assert "loadQuestionMatrix" in app_js
    assert "renderMatrixSummary" in app_js
    assert "renderQuestionMatrix(payload)" in app_js
    assert ".matrix-actions" in styles_css
    assert ".matrix-summary" in styles_css
    assert ".matrix-cell.blank" in styles_css


def test_question_matrix_loads_active_bank_and_latest_results(monkeypatch, tmp_path):
    reports = tmp_path / "test_reports"
    reports.mkdir()
    dataset_path = reports / "dataset.jsonl"
    old_dataset_path = reports / "old_dataset.jsonl"
    result_path = reports / "retrieval_eval_results_20260827_120000.jsonl"
    manifest_path = reports / "retrieval_accuracy_question_bank_manifest.json"
    case = {
        "case_id": "case-1",
        "query": "How should the trigger signal error be corrected?",
        "source_filename": "manual.pdf",
    }
    old_dataset_path.write_text('{"case_id":"old","query":"old"}\n', encoding="utf-8")
    dataset_path.write_text(f"{ui_server.json.dumps(case)}\n", encoding="utf-8")
    result_path.write_text(
        ui_server.json.dumps(
            {
                "case": case,
                "evaluation": {
                    "passed": True,
                    "rank": 1,
                    "metadata_document_selection": {"attempted": True, "passed": True, "rank": 1},
                },
                "answer_evaluation": {
                    "passed": False,
                    "failure_reasons": ["expected_terms_missing"],
                    "expected_document_used": True,
                    "missing_document_ids": [],
                    "citation_fidelity": {"checked": True, "passed": True, "checked_quote_count": 0},
                    "term_check": {"passed": False},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        ui_server.json.dumps(
            {
                "question_bank": {
                    "total_questions": 1,
                    "single_step_questions": 1,
                    "multi_step_questions": 0,
                    "datasets": [
                        {"path": "test_reports/old_dataset.jsonl", "status": "exploratory", "total_questions": 1},
                        {
                            "path": "test_reports/dataset.jsonl",
                            "status": "exploratory",
                            "total_questions": 1,
                            "supersedes": "test_reports/old_dataset.jsonl",
                        },
                        {"path": "test_reports/diagnostic.jsonl", "status": "diagnostic_only_not_promoted", "total_questions": 1},
                    ],
                    "run_exclusions": [],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)

    payload = ui_server._build_question_matrix()

    assert payload["official_total_questions"] == 1
    assert payload["loaded_questions"] == 1
    assert payload["rows"][0]["dataset"] == "test_reports/dataset.jsonl"
    assert payload["rows"][0]["cells"]["metadata"]["status"] == "pass"
    assert payload["rows"][0]["cells"]["retrieval"]["status"] == "pass"
    assert payload["rows"][0]["cells"]["citations"]["status"] == "pass"
    assert payload["rows"][0]["cells"]["terms"]["status"] == "fail"
    assert payload["rows"][0]["cells"]["answer"]["status"] == "blank"


def test_question_matrix_job_runs_active_bank_with_llm_answer_judge(monkeypatch, tmp_path):
    reports = tmp_path / "test_reports"
    reports.mkdir()
    dataset_path = reports / "dataset.jsonl"
    manifest_path = reports / "retrieval_accuracy_question_bank_manifest.json"
    dataset_path.write_text('{"case_id":"case-1","query":"q"}\n', encoding="utf-8")
    manifest_path.write_text(
        ui_server.json.dumps(
            {
                "question_bank": {
                    "datasets": [{"path": "test_reports/dataset.jsonl", "status": "promoted", "total_questions": 1}],
                    "run_exclusions": [],
                }
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class ImmediateThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            self.target(*self.args)

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Completed()

    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    monkeypatch.setattr(ui_server, "Thread", ImmediateThread)
    monkeypatch.setattr(ui_server.subprocess, "run", fake_run)
    ui_server.MATRIX_JOBS.clear()

    job = ui_server._start_question_matrix_job({"mode": "column", "column": "answer", "use_model_judge": True})

    assert job["status"] == "completed"
    assert job["response_mode"] == "answer_with_citations"
    assert job["use_model_judge"] is True
    assert len(calls) == 1
    cmd = calls[0][0]
    assert "--response-mode" in cmd
    assert cmd[cmd.index("--response-mode") + 1] == "answer_with_citations"
    assert "--use-llm-answer-judge" in cmd


def test_question_matrix_retrieval_column_uses_retrieval_only(monkeypatch, tmp_path):
    reports = tmp_path / "test_reports"
    reports.mkdir()
    dataset_path = reports / "dataset.jsonl"
    manifest_path = reports / "retrieval_accuracy_question_bank_manifest.json"
    dataset_path.write_text('{"case_id":"case-1","query":"q"}\n', encoding="utf-8")
    manifest_path.write_text(
        ui_server.json.dumps(
            {
                "question_bank": {
                    "datasets": [{"path": "test_reports/dataset.jsonl", "status": "promoted", "total_questions": 1}],
                    "run_exclusions": [],
                }
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class ImmediateThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            self.target(*self.args)

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    monkeypatch.setattr(ui_server, "Thread", ImmediateThread)
    monkeypatch.setattr(ui_server.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or Completed())
    ui_server.MATRIX_JOBS.clear()

    job = ui_server._start_question_matrix_job({"mode": "column", "column": "retrieval", "use_model_judge": True})

    assert job["status"] == "completed"
    assert job["response_mode"] == "retrieval_only"
    assert job["use_model_judge"] is False
    cmd = calls[0][0]
    assert cmd[cmd.index("--response-mode") + 1] == "retrieval_only"
    assert "--use-llm-answer-judge" not in cmd


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
