from __future__ import annotations

import os
import json
import re
import signal
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
MATRIX_JOBS_LOADED = False
ANSWER_STAGE_KEYS = {"relevance", "summaries", "generation", "answer_docs", "citations", "terms", "answer"}
MATRIX_STAGE_KEYS = [
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
RESULT_STAGE_STEPS = {
    "run_dense_search",
    "run_sparse_search",
    "run_special_search",
    "fuse_results",
    "rerank_results",
    "assemble_context",
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
        if parsed.path == "/local/question-matrix/clear":
            self._clear_local_question_matrix_results()
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
        _load_question_matrix_jobs_if_needed()
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

    def _clear_local_question_matrix_results(self) -> None:
        try:
            result = _clear_question_matrix_results()
            payload = json.dumps(result, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(409)
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
                            "samples": [
                                _compact_stage_sample(sample)
                                for sample in (stage.get("samples") or [])[:100]
                                if isinstance(sample, dict)
                            ],
                        }
                        for stage in query_debug_result.get("stages") or []
                    ],
                },
            }
    return indexed


def _matrix_cell(status: str, detail: str = "", label: str | None = None) -> dict[str, str]:
    cell = {"status": status, "detail": detail}
    if label is not None:
        cell["label"] = label
    return cell


def _compact_stage_sample(sample: dict) -> dict:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return {
        "chunk_id": sample.get("chunk_id") or sample.get("id"),
        "source_document_id": sample.get("source_document_id") or metadata.get("source_document_id"),
        "document_version_id": sample.get("document_version_id") or metadata.get("document_version_id"),
        "title": sample.get("title"),
        "chunk_type": sample.get("chunk_type") or metadata.get("chunk_type"),
        "retrieval_stage": sample.get("retrieval_stage") or metadata.get("retrieval_stage"),
        "score": sample.get("score"),
    }


def _expected_targets(case: dict) -> dict[str, set[str]]:
    expected_evidence = case.get("expected_evidence") if isinstance(case.get("expected_evidence"), list) else []
    document_ids = {str(case.get("source_document_id") or "")}
    chunk_ids = {str(case.get("source_chunk_id") or "")}
    for item in expected_evidence:
        if not isinstance(item, dict):
            continue
        document_ids.add(str(item.get("source_document_id") or case.get("source_document_id") or ""))
        chunk_ids.add(str(item.get("chunk_id") or ""))
    document_ids.discard("")
    chunk_ids.discard("")
    return {"document_ids": document_ids, "chunk_ids": chunk_ids}


def _stage_samples_by_step(debug_result: dict) -> dict[str, list[dict]]:
    return {
        str(stage.get("name") or ""): list(stage.get("samples") or [])
        for stage in debug_result.get("stages") or []
        if isinstance(stage, dict)
    }


def _stage_target_counts(samples: list[dict], targets: dict[str, set[str]]) -> tuple[int, int]:
    matched_docs: set[str] = set()
    matched_chunks: set[str] = set()
    document_ids = targets["document_ids"]
    chunk_ids = targets["chunk_ids"]
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        document_id = str(sample.get("source_document_id") or metadata.get("source_document_id") or "")
        chunk_id = str(sample.get("chunk_id") or sample.get("id") or "")
        if document_id in document_ids:
            matched_docs.add(document_id)
        if chunk_id in chunk_ids:
            matched_chunks.add(chunk_id)
    return len(matched_docs), len(matched_chunks)


def _result_stage_cell(
    key: str,
    samples_by_step: dict[str, list[dict]],
    targets: dict[str, set[str]],
    *,
    previously_found: bool,
) -> tuple[dict[str, str], bool]:
    step_name = next((step for step, mapped_key in DEBUG_STEP_TO_MATRIX_KEY.items() if mapped_key == key), "")
    samples = samples_by_step.get(step_name) or []
    expected_doc_count = max(1, len(targets["document_ids"]))
    matched_doc_count, matched_chunk_count = _stage_target_counts(samples, targets)
    found = bool(matched_doc_count or matched_chunk_count)
    detail = (
        f"{matched_doc_count}/{expected_doc_count} expected document(s), "
        f"{matched_chunk_count}/{len(targets['chunk_ids']) or 1} expected chunk(s) in stage sample window"
    )
    if matched_doc_count >= expected_doc_count:
        return _matrix_cell("pass", detail, "YES"), True
    if found:
        return _matrix_cell("fail", detail, "PARTIAL"), True
    if previously_found:
        return _matrix_cell("fail", f"Expected target was present earlier, then absent here; {detail}", "DROPPED"), False
    return _matrix_cell("fail", detail, "NO"), False


def _retrieval_retention_cell(cells: dict[str, dict[str, str]], retrieval: dict) -> dict[str, str]:
    context_cell = cells.get("assemble") or {}
    context_label = context_cell.get("label") or ""
    context_detail = context_cell.get("detail") or "final context was not recorded"
    if context_label == "YES":
        return _matrix_cell("pass", f"Expected document retained in final context; {context_detail}", "PASS")
    if context_label in {"NO", "PARTIAL", "DROPPED"}:
        return _matrix_cell("fail", f"Expected document missing from final context; {context_detail}", "FAIL")
    if retrieval:
        if retrieval.get("passed") and "candidate_recall" not in retrieval:
            return _matrix_cell("pass", f"rank {retrieval.get('rank')}" if retrieval.get("rank") else "retrieval passed", "PASS")
        if retrieval.get("candidate_recall"):
            return _matrix_cell("pass", "Expected document was present in the final scored retrieval window.", "PASS")
        return _matrix_cell("fail", retrieval.get("failure_category") or "expected document missing from retrieval", "FAIL")
    return _matrix_cell("blank", "not scored yet")


def _block_answer_cells(cells: dict[str, dict[str, str]], detail: str = "blocked by failed retrieval/context") -> dict[str, dict[str, str]]:
    for key in ANSWER_STAGE_KEYS:
        cells[key] = _matrix_cell("blank", detail)
    return cells


def _question_type(case: dict) -> dict[str, object]:
    expected_evidence = case.get("expected_evidence") if isinstance(case.get("expected_evidence"), list) else []
    expected_document_ids = {
        str(item.get("source_document_id") or case.get("source_document_id") or "")
        for item in expected_evidence
        if isinstance(item, dict)
    }
    expected_document_ids.discard("")
    multi_document = (
        len(expected_document_ids) > 1
        or str(case.get("generation_method") or "") == "cross_document_same_field_evidence"
    )
    multi_step = (
        str(case.get("retrieval_task") or "") == "multi_step_retrieval"
        or len(expected_evidence) > 1
        or multi_document
    )
    return {
        "label": "Multi-doc" if multi_document else ("Multi-step" if multi_step else "Single"),
        "multi_step": multi_step,
        "multi_document": multi_document,
        "retrieval_task": case.get("retrieval_task") or "single_step_retrieval",
        "generation_method": case.get("generation_method"),
        "expected_evidence_count": len(expected_evidence) or 1,
        "expected_document_count": len(expected_document_ids) or (1 if case.get("source_document_id") else 0),
    }


def _build_row_cells(item: dict | None, case: dict | None = None) -> dict[str, dict[str, str]]:
    cells = {
        key: _matrix_cell("blank", "not scored yet")
        for key in MATRIX_STAGE_KEYS
    }
    if not item:
        return cells

    case = case or item.get("case") or {}
    debug_result = item.get("query_debug_result") or {}
    completed_steps = set(debug_result.get("completed_steps") or [])
    retrieval_step_to_key = {
        "classify_query": "query_classify",
        "build_filters": "filters",
        "run_dense_search": "dense",
        "run_sparse_search": "sparse",
        "run_special_search": "special",
        "fuse_results": "fuse",
        "rerank_results": "rerank",
        "assemble_context": "assemble",
    }
    answer_step_to_key = {
        "judge_answer_inputs": "relevance",
        "summarize_answer_inputs": "summaries",
        "generate_answer": "generation",
    }
    samples_by_step = _stage_samples_by_step(debug_result)
    targets = _expected_targets(case)
    previously_found = False
    for step, key in retrieval_step_to_key.items():
        if step in completed_steps:
            if step == "classify_query":
                cells[key] = _matrix_cell("pass", "Query classification completed; this stage does not return documents.", "DONE")
            elif step == "build_filters":
                cells[key] = _matrix_cell("pass", "Filters were built; document loss is measured by the following search stages.", "SET")
            elif step in RESULT_STAGE_STEPS:
                if step not in samples_by_step:
                    cells[key] = _matrix_cell("blank", "debug step completed, but stage samples were not recorded")
                else:
                    cells[key], found = _result_stage_cell(key, samples_by_step, targets, previously_found=previously_found)
                    previously_found = found or previously_found and cells[key].get("label") != "DROPPED"
            else:
                cells[key] = _matrix_cell("pass", "debug step completed", "DONE")

    retrieval = item.get("retrieval_evaluation") or item.get("evaluation") or {}
    metadata = retrieval.get("metadata_document_selection") or {}
    if metadata.get("attempted"):
        cells["metadata"] = _matrix_cell(
            "pass" if metadata.get("passed") else "fail",
            f"rank {metadata.get('rank')}" if metadata.get("passed") else metadata.get("failure_category") or "expected document not selected",
        )
    else:
        cells["metadata"] = _matrix_cell("blank", "not attempted")

    cells["retrieval"] = _retrieval_retention_cell(cells, retrieval)
    if cells["retrieval"]["status"] != "pass":
        return _block_answer_cells(cells)

    for step, key in answer_step_to_key.items():
        if step in completed_steps:
            cells[key] = _matrix_cell("pass", "debug step completed")

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
                    "question_type": _question_type(case),
                    "latest_result": {
                        "run_id": result.get("run_id"),
                        "path": result.get("path"),
                        "evaluation": result.get("evaluation"),
                        "answer_evaluation": result.get("answer_evaluation"),
                        "query_debug_result": result.get("query_debug_result"),
                    } if result else None,
                    "cells": _build_row_cells(item, case),
                }
            )
    active_job = None
    active_job_id = _active_question_matrix_job_id()
    if active_job_id:
        try:
            active_job = _question_matrix_job_snapshot(active_job_id)
        except KeyError:
            active_job = None
    return {
        "manifest_path": str(manifest_path.relative_to(MANUALS_ROOT)),
        "official_total_questions": question_bank.get("total_questions"),
        "official_single_step_questions": question_bank.get("single_step_questions"),
        "official_multi_step_questions": question_bank.get("multi_step_questions"),
        "loaded_questions": len(rows),
        "datasets": active_datasets,
        "active_job": active_job,
        "rows": rows,
    }


def _question_matrix_job_snapshot(job_id: str) -> dict:
    _load_question_matrix_jobs_if_needed()
    with MATRIX_JOBS_LOCK:
        return dict(MATRIX_JOBS[job_id])


def _update_question_matrix_job(job_id: str, **updates: object) -> None:
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        job.update(updates)
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _persist_question_matrix_jobs_locked()


def _update_question_matrix_live_cell(job_id: str, case_id: str, key: str, status: str, detail: str = "", label: str | None = None) -> None:
    if key not in MATRIX_STAGE_KEYS:
        return
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        live_cells = dict(job.get("live_cells") or {})
        row_cells = dict(live_cells.get(case_id) or {})
        row_cells[key] = _matrix_cell(status, detail, label)
        live_cells[case_id] = row_cells
        job["live_cells"] = live_cells
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _persist_question_matrix_jobs_locked()


def _replace_question_matrix_live_cells(job_id: str, case_id: str, cells: dict[str, dict[str, str]]) -> None:
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        live_cells = dict(job.get("live_cells") or {})
        live_cells[case_id] = {key: dict(value) for key, value in cells.items()}
        job["live_cells"] = live_cells
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _persist_question_matrix_jobs_locked()


def _active_question_matrix_job_id() -> str | None:
    _load_question_matrix_jobs_if_needed()
    for job_id, job in MATRIX_JOBS.items():
        if job.get("status") in {"queued", "running", "stopping"}:
            return job_id
    return None


def _question_matrix_jobs_state_path() -> Path:
    return TEST_REPORTS_DIR / ".question_matrix_jobs.json"


def _persist_question_matrix_jobs_locked() -> None:
    state_path = _question_matrix_jobs_state_path()
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"jobs": MATRIX_JOBS}, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass


def _load_question_matrix_jobs_if_needed() -> None:
    global MATRIX_JOBS_LOADED
    if MATRIX_JOBS_LOADED:
        return
    with MATRIX_JOBS_LOCK:
        if MATRIX_JOBS_LOADED:
            return
        state_path = _question_matrix_jobs_state_path()
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                jobs = payload.get("jobs") if isinstance(payload, dict) else {}
                if isinstance(jobs, dict):
                    MATRIX_JOBS.update({str(job_id): dict(job) for job_id, job in jobs.items() if isinstance(job, dict)})
                    _refresh_recovered_question_matrix_jobs_locked()
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        MATRIX_JOBS_LOADED = True


def _refresh_recovered_question_matrix_jobs_locked() -> None:
    changed = False
    for job in MATRIX_JOBS.values():
        if job.get("status") not in {"queued", "running", "stopping"}:
            continue
        pid = _job_pid(job)
        if pid and _pid_is_running(pid):
            job["recovered"] = True
            changed = True
            continue
        job["status"] = "failed"
        job["error"] = "UI server restarted before this matrix job finished; no active worker process was found."
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        job["updated_at"] = job["completed_at"]
        changed = True
    if changed:
        _persist_question_matrix_jobs_locked()


def _job_pid(job: dict) -> int | None:
    try:
        pid = int(job.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _question_matrix_job_cancel_requested(job_id: str) -> bool:
    _load_question_matrix_jobs_if_needed()
    with MATRIX_JOBS_LOCK:
        return bool((MATRIX_JOBS.get(job_id) or {}).get("cancel_requested"))


def _raise_if_question_matrix_job_cancelled(job_id: str) -> None:
    if _question_matrix_job_cancel_requested(job_id):
        raise MatrixJobCancelled("matrix job stopped")


def _stop_question_matrix_job(job_id: str) -> dict:
    _load_question_matrix_jobs_if_needed()
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
        pid = _job_pid(job)
        _persist_question_matrix_jobs_locked()
    if process and process.poll() is None:
        process.terminate()
    elif pid and _pid_is_running(pid):
        os.kill(pid, signal.SIGTERM)
    return _question_matrix_job_snapshot(job_id)


def _clear_question_matrix_results() -> dict:
    _load_question_matrix_jobs_if_needed()
    active_job_id = _active_question_matrix_job_id()
    if active_job_id:
        raise ValueError(f"Stop matrix job {active_job_id} before clearing results.")

    patterns = {
        "results": "retrieval_eval_results_*.jsonl",
        "manifests": "retrieval_eval_manifest_*.json",
        "summaries": "retrieval_eval_summary_*.json",
    }
    deleted: dict[str, int] = {key: 0 for key in patterns}
    for key, pattern in patterns.items():
        for path in TEST_REPORTS_DIR.glob(pattern):
            if not path.is_file():
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            deleted[key] += 1

    state_path = _question_matrix_jobs_state_path()
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass

    with MATRIX_JOBS_LOCK:
        MATRIX_JOBS.clear()
        MATRIX_PROCESSES.clear()
        _persist_question_matrix_jobs_locked()
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass
    return {"deleted": deleted, "total_deleted": sum(deleted.values())}


def _start_question_matrix_job(payload: dict) -> dict:
    _load_question_matrix_jobs_if_needed()
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
        "live_cells": {},
        "commands": [],
        "outputs": [],
    }
    with MATRIX_JOBS_LOCK:
        active_job_id = next(
            (existing_id for existing_id, existing_job in MATRIX_JOBS.items() if existing_job.get("status") in {"queued", "running", "stopping"}),
            None,
        )
        if active_job_id:
            raise ValueError(f"Matrix job {active_job_id} is already running.")
        MATRIX_JOBS[job_id] = job
        _persist_question_matrix_jobs_locked()
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
                MATRIX_JOBS[job_id]["pid"] = getattr(process, "pid", None)
                MATRIX_JOBS[job_id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _persist_question_matrix_jobs_locked()
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
                if job_id in MATRIX_JOBS:
                    MATRIX_JOBS[job_id]["pid"] = None
                    _persist_question_matrix_jobs_locked()
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
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
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
            eval_case = RetrievalEvalCase(**case)
            _update_question_matrix_job(
                job_id,
                current_case_id=case_id,
                current_question_number=case_numbers.get(case_id),
                current_stage_key="query_classify",
            )
            debug_result, early_evaluation = _run_query_debug_stream(
                job_id,
                case_id,
                case_numbers.get(case_id),
                case["query"],
                eval_case=eval_case,
            )
            _raise_if_question_matrix_job_cancelled(job_id)
            top_results = _debug_top_results(debug_result)
            evaluation = early_evaluation or score_search_results(eval_case, top_results)
            answer: dict = {}
            answer_evaluation: dict = {}
            if evaluation.get("passed"):
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
            _replace_question_matrix_live_cells(job_id, case_id, _build_row_cells(record))
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


def _run_query_debug_stream(job_id: str, case_id: str, question_number: int | None, query: str, *, eval_case: object) -> tuple[dict, dict | None]:
    from manuals_rag_evals.retrieval_eval import score_search_results

    eval_case_dict = eval_case.to_dict() if hasattr(eval_case, "to_dict") else dict(getattr(eval_case, "__dict__", {}))
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
    completed_steps: list[str] = []
    step_timings_ms: dict = {}
    stages: list[dict] = []
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
            if event.get("event") == "step_completed" and step in DEBUG_STEP_TO_MATRIX_KEY:
                completed_steps = list(event.get("completed_steps") or completed_steps)
                step_timings_ms = dict(event.get("step_timings_ms") or step_timings_ms)
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                stages.append(
                    {
                        "name": step,
                        "label": event.get("label"),
                        "duration_ms": event.get("duration_ms"),
                        "samples": payload.get("samples") or payload.get("summaries") or [],
                    }
                )
                live_cells = _build_row_cells(
                    {
                        "case": eval_case_dict,
                        "query_debug_result": {
                            "completed_steps": completed_steps,
                            "step_timings_ms": step_timings_ms,
                            "stages": stages,
                        },
                    }
                )
                live_cell = live_cells.get(DEBUG_STEP_TO_MATRIX_KEY[step], _matrix_cell("pass", "debug step completed", "DONE"))
                _update_question_matrix_live_cell(
                    job_id,
                    case_id,
                    DEBUG_STEP_TO_MATRIX_KEY[step],
                    live_cell.get("status", "pass"),
                    live_cell.get("detail", "debug step completed"),
                    live_cell.get("label"),
                )
                if step == "assemble_context":
                    top_results = _top_results_from_assemble_payload(payload)
                    evaluation = score_search_results(eval_case, top_results)
                    _update_question_matrix_job(job_id, current_stage_key="retrieval")
                    partial_debug_result = {
                        "query": query,
                        "corpus_ids": [DEFAULT_CORPUS_ID],
                        "completed_steps": completed_steps,
                        "step_timings_ms": step_timings_ms,
                        "stages": stages,
                        "answer": {},
                    }
                    retrieval_cell = _build_row_cells(
                        {
                            "case": eval_case_dict,
                            "query_debug_result": partial_debug_result,
                            "evaluation": evaluation,
                        }
                    )["retrieval"]
                    retrieval_passed = retrieval_cell.get("status") == "pass"
                    partial_debug_result["early_stopped"] = not retrieval_passed
                    partial_debug_result["early_stop_reason"] = (
                        None
                        if retrieval_passed
                        else evaluation.get("failure_category") or retrieval_cell.get("detail") or "retrieval_failed"
                    )
                    if not retrieval_passed:
                        evaluation = dict(evaluation)
                        evaluation["passed"] = False
                        evaluation["failure_category"] = evaluation.get("failure_category") or "retrieval_context_missing"
                        evaluation.setdefault("failure_reasons", [])
                        if isinstance(evaluation["failure_reasons"], list):
                            evaluation["failure_reasons"].append(retrieval_cell.get("detail") or "Expected document missing from final context.")
                        return partial_debug_result, evaluation
            if event.get("event") == "run_completed":
                result = dict(event.get("result") or {})
                if completed_steps and "completed_steps" not in result:
                    result["completed_steps"] = completed_steps
                if step_timings_ms and "step_timings_ms" not in result:
                    result["step_timings_ms"] = step_timings_ms
                if stages and "stages" not in result:
                    result["stages"] = stages
                return result, None
            if event.get("event") == "run_failed":
                raise RuntimeError(str(event.get("error") or "debug query run failed"))
    raise RuntimeError("debug query stream ended without run_completed")


def _debug_top_results(debug_result: dict) -> list[dict]:
    for stage in debug_result.get("stages") or []:
        if stage.get("name") == "retrieval_results":
            return list(stage.get("samples") or [])
        if stage.get("name") == "assemble_context":
            return list(stage.get("samples") or [])
    return []


def _top_results_from_assemble_payload(payload: dict) -> list[dict]:
    return list(payload.get("samples") or [])


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8601), ManualsRagUiHandler)
    print(f"Serving Manuals RAG UI on http://0.0.0.0:8601 with API proxy {API_BASE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
