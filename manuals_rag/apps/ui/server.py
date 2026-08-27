from __future__ import annotations

import os
import json
import re
import subprocess
import sys
import time
import uuid
from json import dumps
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import timeout as SocketTimeout
from threading import Lock
from threading import Thread
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import parse_qs, urlparse

import psycopg
from psycopg.rows import dict_row


API_BASE = os.getenv("MANUALS_RAG_API_BASE", "http://api:8600").rstrip("/")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://manuals:manuals@postgres:5432/manuals_rag")
STATIC_DIR = Path(__file__).resolve().parent
MANUALS_ROOT = STATIC_DIR.parents[1]
TEST_REPORTS_DIR = MANUALS_ROOT / "test_reports"
DEFAULT_CORPUS_ID = os.getenv("MANUALS_RAG_DEFAULT_CORPUS", "manuals_vendor_keyence")
MATRIX_JOB_TIMEOUT_SECONDS = int(os.getenv("MATRIX_JOB_TIMEOUT_SECONDS", "7200"))

MATRIX_JOBS: dict[str, dict] = {}
MATRIX_PROCESSES: dict[str, subprocess.Popen] = {}
MATRIX_JOBS_LOCK = Lock()
ANSWER_STAGE_KEYS = {"relevance", "summaries", "generation", "answer_docs", "citations", "terms", "answer"}
DEBUG_STEP_TO_MATRIX_KEY = {
    "classify_query": "query_classify",
    "build_filters": "filters",
    "run_dense_search": "dense",
    "run_sparse_search": "sparse",
    "run_special_search": "special",
    "fuse_results": "fuse",
    "rerank_results": "rerank",
    "assemble_context": "assemble",
    "judge_answer_inputs": "relevance",
    "summarize_answer_inputs": "summaries",
    "generate_answer": "generation",
}


class MatrixJobCancelled(RuntimeError):
    pass


class ManualsRagUiHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/local/question-matrix/jobs/"):
            self._local_question_matrix_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path == "/local/question-matrix":
            self._local_question_matrix()
            return
        if parsed.path.startswith("/local/run-events"):
            self._local_run_events()
            return
        if self.path.startswith("/api/"):
            self._proxy()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self.path.startswith("/api/"):
            self._proxy()
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/local/question-matrix/run":
            self._start_local_question_matrix_run()
            return
        if parsed.path.startswith("/local/question-matrix/jobs/") and parsed.path.endswith("/stop"):
            self._stop_local_question_matrix_job(parsed.path.split("/")[-2])
            return
        if self.path.startswith("/api/"):
            self._proxy()
            return
        self.send_error(404, "Not found")

    def _proxy(self) -> None:
        upstream_path = self.path.removeprefix("/api")
        target = f"{API_BASE}{upstream_path}"
        body = None
        if self.command in {"POST", "PUT", "PATCH"}:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else None

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        request = Request(target, data=body, headers=headers, method=self.command)
        try:
            with urlopen(request, timeout=900) as response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in {"transfer-encoding", "connection", "content-encoding"}:
                        self.send_header(key, value)
                self.end_headers()
                if self.command == "HEAD":
                    return
                if self._is_streaming_response(response):
                    for line in response:
                        if not self._write(line):
                            break
                    return
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    if not self._write(chunk):
                        break
        except HTTPError as error:
            self.send_response(error.code)
            for key, value in error.headers.items():
                if key.lower() not in {"transfer-encoding", "connection", "content-encoding"}:
                    self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self._write(error.read())
        except (ConnectionError, SocketTimeout, URLError, TimeoutError, OSError) as error:
            if self.command == "HEAD":
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                return
            payload = dumps({
                "detail": f"Manuals RAG API proxy failed: {error.__class__.__name__}: {error}"
            }).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self._write(payload)
            except OSError:
                pass

    def _local_run_events(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        run_id = (query.get("run_id") or [""])[0]
        if not run_id:
            self.send_error(400, "run_id is required")
            return
        try:
            after = max(0, int((query.get("after") or ["0"])[0]))
            limit = max(1, min(int((query.get("limit") or ["1000"])[0]), 2000))
        except ValueError:
            self.send_error(400, "after and limit must be integers")
            return
        try:
            with psycopg.connect(POSTGRES_DSN, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        select event_index, event_json, created_at
                        from app_run_events
                        where run_id = %s
                          and event_index > %s
                          and coalesce(event_json #>> '{query_event,event}', '') <> 'llm_token'
                        order by event_index asc
                        limit %s
                        """,
                        (run_id, after, limit),
                    )
                    rows = cur.fetchall()
            payload = json.dumps(rows, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Local run event lookup failed: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _local_question_matrix(self) -> None:
        try:
            payload = json.dumps(_build_question_matrix(), default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Question matrix lookup failed: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _local_question_matrix_job(self, job_id: str) -> None:
        with MATRIX_JOBS_LOCK:
            job = dict(MATRIX_JOBS.get(job_id) or {})
        if not job:
            self.send_error(404, "Job not found")
            return
        payload = json.dumps(job, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _start_local_question_matrix_run(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else b"{}"
            request_payload = json.loads(body.decode("utf-8") or "{}")
            job = _start_question_matrix_job(request_payload)
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Question matrix run failed to start: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _stop_local_question_matrix_job(self, job_id: str) -> None:
        try:
            job = _stop_question_matrix_job(job_id)
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except KeyError:
            self.send_error(404, "Job not found")
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _is_streaming_response(self, response) -> bool:
        content_type = (response.headers.get("Content-Type") or "").lower()
        return "application/x-ndjson" in content_type or self.path.startswith("/api/eval/end-to-end-stream")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _is_active_dataset(dataset: dict) -> bool:
    status = str(dataset.get("status") or "").lower()
    return "diagnostic" not in status and "superseded" not in status


def _active_question_datasets(question_bank: dict) -> list[dict]:
    datasets = [dict(dataset) for dataset in question_bank.get("datasets") or []]
    active_candidates = [dataset for dataset in datasets if _is_active_dataset(dataset)]
    superseded_paths = {
        str(dataset.get("supersedes"))
        for dataset in active_candidates
        if dataset.get("supersedes")
    }
    return [
        dataset
        for dataset in active_candidates
        if str(dataset.get("path")) not in superseded_paths
    ]


def _case_key(case: dict) -> str:
    return str(case.get("case_id") or case.get("id") or case.get("source_chunk_id") or case.get("query") or "")


def _result_run_id(path: Path) -> str:
    name = path.name
    return name.removeprefix("retrieval_eval_results_").removesuffix(".jsonl")


def _load_result_index(excluded_run_ids: set[str]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    result_paths = sorted(TEST_REPORTS_DIR.glob("retrieval_eval_results_*.jsonl"), key=lambda path: path.stat().st_mtime)
    for path in result_paths:
        run_id = _result_run_id(path)
        if run_id in excluded_run_ids:
            continue
        try:
            rows = _read_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            case = row.get("case") or {}
            key = _case_key(case)
            if not key:
                continue
            query_debug_result = row.get("query_debug_result") or {}
            indexed[key] = {
                "run_id": run_id,
                "path": str(path.relative_to(MANUALS_ROOT)),
                "evaluation": row.get("retrieval_evaluation") or row.get("evaluation") or {},
                "answer_evaluation": row.get("answer_evaluation") or {},
                "query_debug_result": {
                    "completed_steps": query_debug_result.get("completed_steps") or [],
                    "step_timings_ms": query_debug_result.get("step_timings_ms") or {},
                    "stages": [
                        {
                            "name": stage.get("name"),
                            "label": stage.get("label"),
                            "sample_count": len(stage.get("samples") or []),
                        }
                        for stage in query_debug_result.get("stages") or []
                    ],
                },
            }
    return indexed


def _matrix_cell(status: str, detail: str = "") -> dict[str, str]:
    return {"status": status, "detail": detail}


def _build_row_cells(item: dict | None) -> dict[str, dict[str, str]]:
    cells = {
        key: _matrix_cell("blank", "not scored yet")
        for key in [
            "query_classify",
            "filters",
            "dense",
            "sparse",
            "special",
            "fuse",
            "rerank",
            "assemble",
            "metadata",
            "retrieval",
            "relevance",
            "summaries",
            "generation",
            "answer_docs",
            "citations",
            "terms",
            "answer",
        ]
    }
    if not item:
        return cells

    debug_result = item.get("query_debug_result") or {}
    completed_steps = set(debug_result.get("completed_steps") or [])
    step_to_key = {
        "classify_query": "query_classify",
        "build_filters": "filters",
        "run_dense_search": "dense",
        "run_sparse_search": "sparse",
        "run_special_search": "special",
        "fuse_results": "fuse",
        "rerank_results": "rerank",
        "assemble_context": "assemble",
        "judge_answer_inputs": "relevance",
        "summarize_answer_inputs": "summaries",
        "generate_answer": "generation",
    }
    for step, key in step_to_key.items():
        if step in completed_steps:
            cells[key] = _matrix_cell("pass", "debug step completed")

    retrieval = item.get("retrieval_evaluation") or item.get("evaluation") or {}
    metadata = retrieval.get("metadata_document_selection") or {}
    blocked = False
    if metadata.get("attempted"):
        cells["metadata"] = _matrix_cell(
            "pass" if metadata.get("passed") else "fail",
            f"rank {metadata.get('rank')}" if metadata.get("passed") else metadata.get("failure_category") or "expected document not selected",
        )
        blocked = not metadata.get("passed")
    else:
        cells["metadata"] = _matrix_cell("blank", "not attempted")

    if blocked:
        cells["retrieval"] = _matrix_cell("blank", "blocked by document selection")
        return cells

    cells["retrieval"] = _matrix_cell(
        "pass" if retrieval.get("passed") else "fail",
        f"rank {retrieval.get('rank')}" if retrieval.get("passed") else retrieval.get("failure_category") or "expected evidence missing",
    )
    if not retrieval.get("passed"):
        return cells

    answer_eval = item.get("answer_evaluation") or {}
    if not answer_eval:
        return cells

    missing_docs = answer_eval.get("missing_document_ids") or []
    cells["answer_docs"] = _matrix_cell(
        "fail" if answer_eval.get("expected_document_used") is False or missing_docs else "pass",
        f"missing {len(missing_docs)} expected document(s)" if missing_docs else "expected document used",
    )
    if cells["answer_docs"]["status"] == "fail":
        return cells

    citation = answer_eval.get("citation_fidelity") or {}
    if citation.get("checked"):
        cells["citations"] = _matrix_cell(
            "pass" if citation.get("passed") else "fail",
            f"{citation.get('checked_quote_count') or 0} quote(s) checked" if citation.get("passed") else "unsupported or missing citation evidence",
        )
        if not citation.get("passed"):
            return cells

    term_check = answer_eval.get("term_check") or {}
    if term_check:
        llm_info = term_check.get("llm_required_information") or answer_eval.get("llm_required_information") or {}
        term_detail = (
            f"model judge: {llm_info.get('reason') or 'required information judged'}"
            if term_check.get("llm_judged")
            else ("required terms present" if term_check.get("passed") else "expected terms/actions missing")
        )
        cells["terms"] = _matrix_cell(
            "pass" if term_check.get("passed") else "fail",
            term_detail,
        )
        if not term_check.get("passed"):
            return cells

    cells["answer"] = _matrix_cell(
        "pass" if answer_eval.get("passed") else "fail",
        ", ".join(answer_eval.get("failure_reasons") or []) or "overall answer score",
    )
    return cells


def _build_question_matrix() -> dict:
    manifest_path = TEST_REPORTS_DIR / "retrieval_accuracy_question_bank_manifest.json"
    manifest = _read_json(manifest_path)
    question_bank = manifest.get("question_bank") or {}
    active_datasets = _active_question_datasets(question_bank)
    excluded_run_ids = {
        str(exclusion.get("run_id"))
        for exclusion in (question_bank.get("run_exclusions") or manifest.get("run_exclusions") or [])
        if exclusion.get("run_id")
    }
    result_index = _load_result_index(excluded_run_ids)
    rows: list[dict] = []
    for dataset in active_datasets:
        dataset_path = MANUALS_ROOT / str(dataset.get("path") or "")
        if not dataset_path.exists():
            continue
        for row_index, case in enumerate(_read_jsonl(dataset_path), start=1):
            key = _case_key(case)
            result = result_index.get(key)
            item = result if result else None
            rows.append(
                {
                    "key": key,
                    "dataset": str(dataset_path.relative_to(MANUALS_ROOT)),
                    "dataset_status": dataset.get("status"),
                    "question_number": row_index,
                    "case": case,
                    "latest_result": {
                        "run_id": result.get("run_id"),
                        "path": result.get("path"),
                        "evaluation": result.get("evaluation"),
                        "answer_evaluation": result.get("answer_evaluation"),
                        "query_debug_result": result.get("query_debug_result"),
                    } if result else None,
                    "cells": _build_row_cells(item),
                }
            )
    return {
        "manifest_path": str(manifest_path.relative_to(MANUALS_ROOT)),
        "official_total_questions": question_bank.get("total_questions"),
        "official_single_step_questions": question_bank.get("single_step_questions"),
        "official_multi_step_questions": question_bank.get("multi_step_questions"),
        "loaded_questions": len(rows),
        "datasets": active_datasets,
        "rows": rows,
    }


def _question_matrix_job_snapshot(job_id: str) -> dict:
    with MATRIX_JOBS_LOCK:
        return dict(MATRIX_JOBS[job_id])


def _update_question_matrix_job(job_id: str, **updates: object) -> None:
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        job.update(updates)
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _active_question_matrix_job_id() -> str | None:
    for job_id, job in MATRIX_JOBS.items():
        if job.get("status") in {"queued", "running", "stopping"}:
            return job_id
    return None


def _question_matrix_job_cancel_requested(job_id: str) -> bool:
    with MATRIX_JOBS_LOCK:
        return bool((MATRIX_JOBS.get(job_id) or {}).get("cancel_requested"))


def _raise_if_question_matrix_job_cancelled(job_id: str) -> None:
    if _question_matrix_job_cancel_requested(job_id):
        raise MatrixJobCancelled("matrix job stopped")


def _stop_question_matrix_job(job_id: str) -> dict:
    with MATRIX_JOBS_LOCK:
        if job_id not in MATRIX_JOBS:
            raise KeyError(job_id)
        job = MATRIX_JOBS[job_id]
        if job.get("status") not in {"queued", "running", "stopping"}:
            raise ValueError(f"Job {job_id} is not running.")
        job["status"] = "stopping"
        job["cancel_requested"] = True
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        process = MATRIX_PROCESSES.get(job_id)
    if process and process.poll() is None:
        process.terminate()
    return _question_matrix_job_snapshot(job_id)


def _start_question_matrix_job(payload: dict) -> dict:
    mode = str(payload.get("mode") or "all_bank")
    column = str(payload.get("column") or "retrieval")
    use_model_judge = bool(payload.get("use_model_judge"))
    if mode not in {"all_bank", "column"}:
        raise ValueError("mode must be all_bank or column")
    valid_columns = {
        "query_classify",
        "filters",
        "dense",
        "sparse",
        "special",
        "fuse",
        "rerank",
        "assemble",
        "metadata",
        "retrieval",
        "relevance",
        "summaries",
        "generation",
        "answer_docs",
        "citations",
        "terms",
        "answer",
    }
    if mode == "column" and column not in valid_columns:
        raise ValueError(f"Unsupported matrix column: {column}")

    manifest = _read_json(TEST_REPORTS_DIR / "retrieval_accuracy_question_bank_manifest.json")
    datasets = _active_question_datasets(manifest.get("question_bank") or {})
    if not datasets:
        raise ValueError("No active question-bank datasets were found.")

    job_id = f"matrix-{uuid.uuid4().hex[:12]}"
    response_mode = "answer_with_citations" if mode == "all_bank" or column in ANSWER_STAGE_KEYS else "retrieval_only"
    job = {
        "id": job_id,
        "status": "queued",
        "mode": mode,
        "column": column if mode == "column" else "all",
        "response_mode": response_mode,
        "use_model_judge": use_model_judge and response_mode == "answer_with_citations",
        "started_at": None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_at": None,
        "dataset_count": len(datasets),
        "current_dataset": None,
        "current_case_id": None,
        "current_question_number": None,
        "current_stage_key": "answer" if response_mode == "answer_with_citations" else "retrieval",
        "completed_datasets": 0,
        "returncode": None,
        "error": None,
        "cancel_requested": False,
        "commands": [],
        "outputs": [],
    }
    with MATRIX_JOBS_LOCK:
        active_job_id = _active_question_matrix_job_id()
        if active_job_id:
            raise ValueError(f"Matrix job {active_job_id} is already running.")
        MATRIX_JOBS[job_id] = job
    thread = Thread(target=_run_question_matrix_job, args=(job_id, datasets), daemon=True)
    thread.start()
    return _question_matrix_job_snapshot(job_id)


def _run_question_matrix_job(job_id: str, datasets: list[dict]) -> None:
    job = _question_matrix_job_snapshot(job_id)
    _update_question_matrix_job(job_id, status="running", started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        for index, dataset in enumerate(datasets, start=1):
            _raise_if_question_matrix_job_cancelled(job_id)
            dataset_rel = str(dataset.get("path") or "")
            dataset_path = MANUALS_ROOT / dataset_rel
            if not dataset_path.exists():
                raise FileNotFoundError(f"Question-bank dataset not found: {dataset_rel}")
            case_numbers = {
                _case_key(case): case_index
                for case_index, case in enumerate(_read_jsonl(dataset_path), start=1)
            }
            if job["response_mode"] == "answer_with_citations":
                _run_answer_matrix_dataset(job_id, dataset_rel, dataset_path, case_numbers, index)
                continue
            cmd = [
                sys.executable,
                str(MANUALS_ROOT / "scripts" / "benchmark" / "run_large_retrieval_eval.py"),
                "--existing-corpus-id",
                DEFAULT_CORPUS_ID,
                "--dataset-path",
                str(dataset_path),
                "--max-queries",
                str(max(1, int(dataset.get("total_questions") or 100000))),
                "--search-mode",
                "http",
                "--response-mode",
                str(job["response_mode"]),
                "--per-query-timeout-seconds",
                "180",
                "--warmup-queries",
                "0",
            ]
            if job.get("use_model_judge"):
                cmd.append("--use-llm-answer-judge")
            command_record = {"dataset": dataset_rel, "command": cmd}
            _update_question_matrix_job(
                job_id,
                current_dataset=dataset_rel,
                current_case_id=None,
                current_question_number=None,
                current_stage_key=str(job["column"]) if job["mode"] == "column" else str(job["current_stage_key"]),
                commands=[*(_question_matrix_job_snapshot(job_id).get("commands") or []), command_record],
            )
            process = subprocess.Popen(
                cmd,
                cwd=MANUALS_ROOT,
                env={
                    **os.environ,
                    "API_BASE": API_BASE,
                    "LOCAL_ADMIN_TOKEN": os.getenv("LOCAL_ADMIN_TOKEN", "admin-token"),
                    "LOCAL_END_USER_TOKEN": os.getenv("LOCAL_END_USER_TOKEN", "user-token"),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with MATRIX_JOBS_LOCK:
                MATRIX_PROCESSES[job_id] = process
            output_lines: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                _raise_if_question_matrix_job_cancelled(job_id)
                output_lines.append(line)
                if len(output_lines) > 400:
                    output_lines = output_lines[-400:]
                match = re.search(r'"case_id"\s*:\s*"([^"]+)"', line)
                if match:
                    current_case_id = match.group(1)
                    _update_question_matrix_job(
                        job_id,
                        current_case_id=current_case_id,
                        current_question_number=case_numbers.get(current_case_id),
                    )
            returncode = process.wait(timeout=MATRIX_JOB_TIMEOUT_SECONDS)
            with MATRIX_JOBS_LOCK:
                MATRIX_PROCESSES.pop(job_id, None)
            _raise_if_question_matrix_job_cancelled(job_id)
            stdout_tail = "".join(output_lines)[-8000:]
            output = {
                "dataset": dataset_rel,
                "returncode": returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": "",
            }
            _update_question_matrix_job(
                job_id,
                outputs=[*(_question_matrix_job_snapshot(job_id).get("outputs") or []), output],
                completed_datasets=index,
                returncode=returncode,
            )
            if returncode != 0:
                raise RuntimeError(f"Benchmark failed for {dataset_rel} with exit code {returncode}")
        _update_question_matrix_job(
            job_id,
            status="completed",
            current_dataset=None,
            current_case_id=None,
            current_question_number=None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except MatrixJobCancelled as error:
        _update_question_matrix_job(
            job_id,
            status="cancelled",
            error=str(error),
            current_dataset=None,
            current_case_id=None,
            current_question_number=None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except Exception as error:
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
        _update_question_matrix_job(
            job_id,
            status="failed",
            error=f"{error.__class__.__name__}: {error}",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


def _run_answer_matrix_dataset(
    job_id: str,
    dataset_rel: str,
    dataset_path: Path,
    case_numbers: dict[str, int],
    dataset_index: int,
) -> None:
    from manuals_rag_evals.retrieval_eval import RetrievalEvalCase
    from manuals_rag_evals.retrieval_eval import score_answer_response
    from manuals_rag_evals.retrieval_eval import score_search_results

    job = _question_matrix_job_snapshot(job_id)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = TEST_REPORTS_DIR / f"retrieval_eval_results_{timestamp}_{uuid.uuid4().hex[:6]}.jsonl"
    manifest_path = TEST_REPORTS_DIR / f"retrieval_eval_manifest_{timestamp}_{uuid.uuid4().hex[:6]}.json"
    cases = _read_jsonl(dataset_path)
    results: list[dict] = []
    _update_question_matrix_job(
        job_id,
        current_dataset=dataset_rel,
        current_case_id=None,
        current_question_number=1 if cases else None,
        current_stage_key="query_classify",
    )
    with results_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            _raise_if_question_matrix_job_cancelled(job_id)
            case_id = _case_key(case)
            _update_question_matrix_job(
                job_id,
                current_case_id=case_id,
                current_question_number=case_numbers.get(case_id),
                current_stage_key="query_classify",
            )
            debug_result = _run_query_debug_stream(job_id, case_id, case_numbers.get(case_id), case["query"])
            _raise_if_question_matrix_job_cancelled(job_id)
            top_results = _debug_top_results(debug_result)
            eval_case = RetrievalEvalCase(**case)
            evaluation = score_search_results(eval_case, top_results)
            answer = dict(debug_result.get("answer") or {})
            for stage_key in ("answer_docs", "citations", "terms", "answer"):
                _update_question_matrix_job(job_id, current_stage_key=stage_key)
            answer_evaluation = score_answer_response(
                eval_case,
                answer,
                evaluation,
                top_results,
                use_llm_required_info_judge=bool(job.get("use_model_judge")),
            )
            record = {
                "case": case,
                "evaluation": evaluation,
                "top_results": top_results[:5],
                "answer": answer,
                "answer_evaluation": answer_evaluation,
                "query_debug_result": debug_result,
            }
            results.append(record)
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_id": DEFAULT_CORPUS_ID,
                "input_dataset_path": str(dataset_path),
                "search_mode": "http_debug_stream",
                "response_mode": "answer_with_citations",
                "use_llm_answer_judge": bool(job.get("use_model_judge")),
                "results_path": str(results_path),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _update_question_matrix_job(
        job_id,
        outputs=[
            *(_question_matrix_job_snapshot(job_id).get("outputs") or []),
            {"dataset": dataset_rel, "returncode": 0, "stdout_tail": json.dumps({"results": len(results), "results_path": str(results_path)}), "stderr_tail": ""},
        ],
        completed_datasets=dataset_index,
        returncode=0,
    )


def _run_query_debug_stream(job_id: str, case_id: str, question_number: int | None, query: str) -> dict:
    request_payload = {
        "query": query,
        "corpus_ids": [DEFAULT_CORPUS_ID],
        "filters": {},
        "response_mode": "answer_with_citations",
    }
    request = Request(
        f"{API_BASE}/debug/query-stream?sample_limit=100",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.getenv('LOCAL_ADMIN_TOKEN', 'admin-token')}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=900) as response:
        for raw_line in response:
            _raise_if_question_matrix_job_cancelled(job_id)
            if not raw_line.strip():
                continue
            event = json.loads(raw_line.decode("utf-8"))
            step = event.get("step")
            if event.get("event") == "step_started" and step in DEBUG_STEP_TO_MATRIX_KEY:
                _update_question_matrix_job(
                    job_id,
                    current_case_id=case_id,
                    current_question_number=question_number,
                    current_stage_key=DEBUG_STEP_TO_MATRIX_KEY[step],
                )
            if event.get("event") == "run_completed":
                return dict(event.get("result") or {})
            if event.get("event") == "run_failed":
                raise RuntimeError(str(event.get("error") or "debug query run failed"))
    raise RuntimeError("debug query stream ended without run_completed")


def _debug_top_results(debug_result: dict) -> list[dict]:
    for stage in debug_result.get("stages") or []:
        if stage.get("name") == "retrieval_results":
            return list(stage.get("samples") or [])
    return []


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8601), ManualsRagUiHandler)
    print(f"Serving Manuals RAG UI on http://0.0.0.0:8601 with API proxy {API_BASE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
