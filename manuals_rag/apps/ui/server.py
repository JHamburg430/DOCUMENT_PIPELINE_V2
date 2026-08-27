from __future__ import annotations

import os
import json
from json import dumps
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import timeout as SocketTimeout
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


class ManualsRagUiHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/local/question-matrix"):
            self._local_question_matrix()
            return
        if self.path.startswith("/local/run-events"):
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
        cells["terms"] = _matrix_cell(
            "pass" if term_check.get("passed") else "fail",
            "required terms present" if term_check.get("passed") else "expected terms/actions missing",
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
    datasets = [dict(dataset) for dataset in question_bank.get("datasets") or []]
    active_candidates = [dataset for dataset in datasets if _is_active_dataset(dataset)]
    superseded_paths = {
        str(dataset.get("supersedes"))
        for dataset in active_candidates
        if dataset.get("supersedes")
    }
    active_datasets = [
        dataset
        for dataset in active_candidates
        if str(dataset.get("path")) not in superseded_paths
    ]
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


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8601), ManualsRagUiHandler)
    print(f"Serving Manuals RAG UI on http://0.0.0.0:8601 with API proxy {API_BASE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
