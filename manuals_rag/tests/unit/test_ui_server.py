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

import pytest

from apps.ui import server as ui_server
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase


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
    assert 'id="matrix-stop"' in index_html
    assert 'id="matrix-summary"' in index_html
    assert 'id="matrix-table"' in index_html
    assert "MATRIX_STAGES" in app_js
    assert "setupMatrixControls()" in app_js
    assert "startMatrixJob" in app_js
    assert "stopMatrixJob" in app_js
    assert "current-run-cell" in app_js
    assert "loadQuestionMatrix" in app_js
    assert "renderMatrixSummary" in app_js
    assert "renderQuestionMatrix(payload)" in app_js
    assert ".matrix-actions" in styles_css
    assert ".current-run-cell" in styles_css
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
    assert payload["rows"][0]["question_type"]["label"] == "Single"
    assert payload["rows"][0]["cells"]["metadata"]["status"] == "pass"
    assert payload["rows"][0]["cells"]["retrieval"]["status"] == "pass"
    assert payload["rows"][0]["cells"]["citations"]["status"] == "pass"
    assert payload["rows"][0]["cells"]["terms"]["status"] == "fail"
    assert payload["rows"][0]["cells"]["answer"]["status"] == "blank"


def test_question_matrix_includes_active_job_for_page_refresh(monkeypatch, tmp_path):
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
    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    ui_server.MATRIX_JOBS.clear()
    ui_server.MATRIX_JOBS["matrix-running"] = {
        "id": "matrix-running",
        "status": "running",
        "mode": "all_bank",
        "dataset_count": 1,
        "completed_datasets": 0,
        "live_cells": {},
    }

    try:
        payload = ui_server._build_question_matrix()
    finally:
        ui_server.MATRIX_JOBS.clear()

    assert payload["active_job"]["id"] == "matrix-running"
    assert payload["active_job"]["status"] == "running"


def test_question_matrix_recovers_active_job_from_state_file(monkeypatch, tmp_path):
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
    (reports / ".question_matrix_jobs.json").write_text(
        ui_server.json.dumps(
            {
                "jobs": {
                    "matrix-recovered": {
                        "id": "matrix-recovered",
                        "status": "running",
                        "mode": "column",
                        "column": "retrieval",
                        "dataset_count": 1,
                        "completed_datasets": 0,
                        "pid": 1234,
                        "live_cells": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    monkeypatch.setattr(ui_server, "_pid_is_running", lambda pid: pid == 1234)
    ui_server.MATRIX_JOBS.clear()
    ui_server.MATRIX_PROCESSES.clear()
    monkeypatch.setattr(ui_server, "MATRIX_JOBS_LOADED", False)

    try:
        payload = ui_server._build_question_matrix()
    finally:
        ui_server.MATRIX_JOBS.clear()
        ui_server.MATRIX_PROCESSES.clear()
        ui_server.MATRIX_JOBS_LOADED = True

    assert payload["active_job"]["id"] == "matrix-recovered"
    assert payload["active_job"]["status"] == "running"
    assert payload["active_job"]["recovered"] is True


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

    def fake_run_answer_dataset(job_id, dataset_rel, dataset_path_arg, case_numbers, dataset_index):
        calls.append((job_id, dataset_rel, dataset_path_arg, case_numbers, dataset_index))
        ui_server._update_question_matrix_job(
            job_id,
            current_dataset=dataset_rel,
            current_case_id="case-1",
            current_question_number=1,
            current_stage_key="answer",
            completed_datasets=dataset_index,
            returncode=0,
        )

    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    monkeypatch.setattr(ui_server, "Thread", ImmediateThread)
    monkeypatch.setattr(ui_server, "_run_answer_matrix_dataset", fake_run_answer_dataset)
    ui_server.MATRIX_JOBS.clear()

    job = ui_server._start_question_matrix_job({"mode": "column", "column": "answer", "use_model_judge": True})

    assert job["status"] == "completed"
    assert job["response_mode"] == "answer_with_citations"
    assert job["use_model_judge"] is True
    assert job["current_stage_key"] == "answer"
    assert len(calls) == 1
    assert calls[0][1] == "test_reports/dataset.jsonl"
    assert calls[0][3] == {"case-1": 1}


def test_question_matrix_model_judge_is_not_default_on():
    index_html = (ui_server.MANUALS_ROOT / "apps" / "ui" / "index.html").read_text(encoding="utf-8")

    assert 'id="matrix-use-model-judge" type="checkbox"' in index_html
    assert 'id="matrix-use-model-judge" type="checkbox" checked' not in index_html


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

    class FakeProcess:
        stdout = ['  "case_id": "case-1",\n']

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    monkeypatch.setattr(ui_server, "Thread", ImmediateThread)
    monkeypatch.setattr(ui_server.subprocess, "Popen", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or FakeProcess())
    ui_server.MATRIX_JOBS.clear()

    job = ui_server._start_question_matrix_job({"mode": "column", "column": "retrieval", "use_model_judge": True})

    assert job["status"] == "completed"
    assert job["response_mode"] == "retrieval_only"
    assert job["use_model_judge"] is False
    assert job["current_stage_key"] == "retrieval"
    cmd = calls[0][0]
    assert cmd[cmd.index("--response-mode") + 1] == "retrieval_only"
    assert "--use-llm-answer-judge" not in cmd


def test_question_matrix_rejects_second_active_job(monkeypatch, tmp_path):
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
    monkeypatch.setattr(ui_server, "MANUALS_ROOT", tmp_path)
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    ui_server.MATRIX_JOBS.clear()
    ui_server.MATRIX_JOBS["matrix-active"] = {"id": "matrix-active", "status": "running"}

    with pytest.raises(ValueError, match="already running"):
        ui_server._start_question_matrix_job({"mode": "column", "column": "retrieval"})

    ui_server.MATRIX_JOBS.clear()


def test_question_matrix_stop_marks_job_and_terminates_process(monkeypatch, tmp_path):
    class FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = FakeProcess()
    reports = tmp_path / "test_reports"
    reports.mkdir()
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    ui_server.MATRIX_JOBS.clear()
    ui_server.MATRIX_PROCESSES.clear()
    ui_server.MATRIX_JOBS["matrix-stop-me"] = {"id": "matrix-stop-me", "status": "running"}
    ui_server.MATRIX_PROCESSES["matrix-stop-me"] = process

    job = ui_server._stop_question_matrix_job("matrix-stop-me")

    assert job["status"] == "stopping"
    assert job["cancel_requested"] is True
    assert process.terminated is True
    ui_server.MATRIX_JOBS.clear()
    ui_server.MATRIX_PROCESSES.clear()


def test_question_matrix_stop_recovered_job_by_pid(monkeypatch, tmp_path):
    reports = tmp_path / "test_reports"
    reports.mkdir()
    (reports / ".question_matrix_jobs.json").write_text(
        ui_server.json.dumps(
            {
                "jobs": {
                    "matrix-stop-pid": {
                        "id": "matrix-stop-pid",
                        "status": "running",
                        "pid": 4321,
                        "live_cells": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    killed = []

    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    monkeypatch.setattr(ui_server, "_pid_is_running", lambda pid: pid == 4321)
    monkeypatch.setattr(ui_server.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    ui_server.MATRIX_JOBS.clear()
    ui_server.MATRIX_PROCESSES.clear()
    monkeypatch.setattr(ui_server, "MATRIX_JOBS_LOADED", False)

    try:
        job = ui_server._stop_question_matrix_job("matrix-stop-pid")
    finally:
        ui_server.MATRIX_JOBS.clear()
        ui_server.MATRIX_PROCESSES.clear()
        ui_server.MATRIX_JOBS_LOADED = True

    assert job["status"] == "stopping"
    assert job["cancel_requested"] is True
    assert killed == [(4321, ui_server.signal.SIGTERM)]


def test_question_matrix_blanks_answer_steps_after_retrieval_failure():
    cells = ui_server._build_row_cells(
        {
            "case": {
                "source_document_id": "doc-1",
                "source_chunk_id": "chunk-1",
            },
            "query_debug_result": {
                "completed_steps": [
                    "classify_query",
                    "build_filters",
                    "run_dense_search",
                    "assemble_context",
                    "judge_answer_inputs",
                    "summarize_answer_inputs",
                    "generate_answer",
                ]
            },
            "evaluation": {
                "passed": False,
                "failure_category": "expected_evidence_missing",
                "metadata_document_selection": {"attempted": True, "passed": True, "rank": 1},
            },
            "answer_evaluation": {"passed": True, "expected_document_used": True},
        }
    )

    assert cells["query_classify"]["status"] == "pass"
    assert cells["retrieval"]["status"] == "fail"
    assert cells["relevance"]["status"] == "blank"
    assert cells["summaries"]["status"] == "blank"
    assert cells["generation"]["status"] == "blank"
    assert cells["answer"]["status"] == "blank"


def test_question_matrix_labels_stage_target_retention():
    cells = ui_server._build_row_cells(
        {
            "case": {
                "source_document_id": "doc-expected",
                "source_chunk_id": "chunk-expected",
            },
            "query_debug_result": {
                "completed_steps": [
                    "classify_query",
                    "build_filters",
                    "run_dense_search",
                    "run_sparse_search",
                    "fuse_results",
                    "assemble_context",
                ],
                "stages": [
                    {"name": "run_dense_search", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-other"}]},
                    {"name": "run_sparse_search", "samples": [{"source_document_id": "doc-other", "chunk_id": "chunk-other"}]},
                    {"name": "fuse_results", "samples": [{"source_document_id": "doc-other", "chunk_id": "chunk-other"}]},
                    {"name": "assemble_context", "samples": [{"source_document_id": "doc-other", "chunk_id": "chunk-other"}]},
                ],
            },
            "evaluation": {
                "passed": False,
                "failure_category": "ranking_or_context_loss",
                "metadata_document_selection": {"attempted": False},
            },
        }
    )

    assert cells["query_classify"]["label"] == "DONE"
    assert cells["filters"]["label"] == "SET"
    assert cells["dense"]["label"] == "YES"
    assert cells["sparse"]["label"] == "DROPPED"
    assert cells["fuse"]["label"] == "NO"
    assert cells["retrieval"]["label"] == "FAIL"


def test_question_matrix_retrieval_tracks_final_document_retention_not_evidence_score():
    cells = ui_server._build_row_cells(
        {
            "case": {
                "source_document_id": "doc-expected",
                "source_chunk_id": "chunk-expected",
            },
            "query_debug_result": {
                "completed_steps": [
                    "classify_query",
                    "build_filters",
                    "run_sparse_search",
                    "fuse_results",
                    "rerank_results",
                    "assemble_context",
                ],
                "stages": [
                    {"name": "run_sparse_search", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-other"}]},
                    {"name": "fuse_results", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-other"}]},
                    {"name": "rerank_results", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-other"}]},
                    {"name": "assemble_context", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-other"}]},
                ],
            },
            "evaluation": {
                "passed": False,
                "failure_category": "ranking_or_context_loss",
                "candidate_recall": True,
                "metadata_document_selection": {"attempted": False},
            },
        }
    )

    assert cells["assemble"]["label"] == "YES"
    assert cells["retrieval"]["status"] == "pass"
    assert cells["retrieval"]["label"] == "PASS"


def test_question_matrix_blocks_answer_steps_when_context_retention_fails():
    cells = ui_server._build_row_cells(
        {
            "case": {
                "source_document_id": "doc-expected",
                "source_chunk_id": "chunk-expected",
            },
            "query_debug_result": {
                "completed_steps": [
                    "classify_query",
                    "build_filters",
                    "run_sparse_search",
                    "fuse_results",
                    "rerank_results",
                    "assemble_context",
                    "judge_answer_inputs",
                    "summarize_answer_inputs",
                    "generate_answer",
                ],
                "stages": [
                    {"name": "run_sparse_search", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-expected"}]},
                    {"name": "fuse_results", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-expected"}]},
                    {"name": "rerank_results", "samples": [{"source_document_id": "doc-expected", "chunk_id": "chunk-expected"}]},
                    {"name": "assemble_context", "samples": [{"source_document_id": "wrong-doc", "chunk_id": "wrong-chunk"}]},
                ],
            },
            "evaluation": {
                "passed": True,
                "candidate_recall": True,
                "metadata_document_selection": {"attempted": False},
            },
            "answer_evaluation": {
                "passed": True,
                "expected_document_used": True,
                "citation_fidelity": {"checked": True, "passed": True, "checked_quote_count": 1},
                "term_check": {"passed": True},
            },
        }
    )

    assert cells["retrieval"]["status"] == "fail"
    assert cells["retrieval"]["label"] == "FAIL"
    for key in ("relevance", "summaries", "generation", "answer_docs", "citations", "terms", "answer"):
        assert cells[key]["status"] == "blank"
        assert "blocked" in cells[key]["detail"]


def test_question_type_identifies_multi_step_and_multi_document_cases():
    multi_step = ui_server._question_type(
        {
            "retrieval_task": "multi_step_retrieval",
            "source_document_id": "doc-1",
            "expected_evidence": [
                {"chunk_id": "error", "expected_terms": ["error"]},
                {"chunk_id": "action", "expected_terms": ["action"]},
            ],
        }
    )
    multi_doc = ui_server._question_type(
        {
            "retrieval_task": "multi_step_retrieval",
            "generation_method": "cross_document_same_field_evidence",
            "source_document_id": "doc-1",
            "expected_evidence": [
                {"chunk_id": "left", "source_document_id": "doc-1", "expected_terms": ["24"]},
                {"chunk_id": "right", "source_document_id": "doc-2", "expected_terms": ["24"]},
            ],
        }
    )

    assert multi_step["label"] == "Multi-step"
    assert multi_step["multi_step"] is True
    assert multi_step["multi_document"] is False
    assert multi_step["expected_evidence_count"] == 2
    assert multi_doc["label"] == "Multi-doc"
    assert multi_doc["multi_document"] is True
    assert multi_doc["expected_document_count"] == 2


def test_query_debug_stream_stops_after_failed_retrieval(monkeypatch, tmp_path):
    case = RetrievalEvalCase(
        case_id="case-1",
        query="What voltage does MODEL-1 use?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="spec_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["24", "vdc"],
        expected_snippet="Power supply voltage: 24 VDC",
        generation_method="unit_test",
        source_metadata={"product_model": "MODEL-1"},
    )
    events = [
        {"event": "step_started", "step": "classify_query"},
        {"event": "step_completed", "step": "classify_query", "completed_steps": ["classify_query"], "step_timings_ms": {"classify_query": 1}, "payload": {}},
        {
            "event": "step_completed",
            "step": "assemble_context",
            "completed_steps": ["classify_query", "assemble_context"],
            "step_timings_ms": {"classify_query": 1, "assemble_context": 2},
            "payload": {
                "samples": [
                    {
                        "chunk_id": "wrong-chunk",
                        "source_document_id": "wrong-doc",
                        "content": "Power supply voltage: 12 VDC",
                    }
                ]
            },
        },
        {"event": "step_started", "step": "generate_answer"},
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            for event in events:
                yield (ui_server.json.dumps(event) + "\n").encode("utf-8")

    monkeypatch.setattr(ui_server, "urlopen", lambda request, timeout: FakeResponse())
    reports = tmp_path / "test_reports"
    reports.mkdir()
    monkeypatch.setattr(ui_server, "TEST_REPORTS_DIR", reports)
    ui_server.MATRIX_JOBS.clear()
    ui_server.MATRIX_JOBS["matrix-early"] = {"id": "matrix-early", "status": "running", "live_cells": {}}

    debug_result, evaluation = ui_server._run_query_debug_stream(
        "matrix-early",
        "case-1",
        1,
        case.query,
        eval_case=case,
    )

    assert evaluation["passed"] is False
    assert debug_result["early_stopped"] is True
    assert debug_result["answer"] == {}
    assert "generate_answer" not in debug_result["completed_steps"]
    assert ui_server.MATRIX_JOBS["matrix-early"]["live_cells"]["case-1"]["assemble"]["status"] == "fail"
    assert ui_server.MATRIX_JOBS["matrix-early"]["live_cells"]["case-1"]["assemble"]["label"] == "NO"
    ui_server.MATRIX_JOBS.clear()


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
