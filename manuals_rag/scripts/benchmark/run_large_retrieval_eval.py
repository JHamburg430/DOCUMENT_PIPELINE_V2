from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[3]
MANUALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "evals" / "src"))

from manuals_rag_evals.retrieval_eval import (
    build_eval_cases_from_chunks,
    build_multi_step_eval_cases_from_chunks,
    chunk_is_queryworthy,
    score_answer_response,
    score_search_results,
)
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase


API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8600")
ADMIN_TOKEN = os.getenv("LOCAL_ADMIN_TOKEN", "admin-token")
USER_TOKEN = os.getenv("LOCAL_END_USER_TOKEN", "user-token")
DEFAULT_DOCS_DIR = REPO_ROOT / "Technical_Documents" / "Keyence"
OUTPUT_DIR = MANUALS_ROOT / "test_reports"


class QueryTimeoutError(TimeoutError):
    pass


@contextmanager
def query_timeout(seconds: int | None):
    if not seconds or seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise QueryTimeoutError(f"Search exceeded per-query timeout of {seconds} seconds.")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def timeout_evaluation(case: dict[str, Any], *, elapsed_seconds: float, timeout_seconds: int) -> dict[str, Any]:
    return {
        "passed": False,
        "rank": None,
        "matched_terms": [],
        "missing_terms": list(case.get("expected_terms") or []),
        "candidate_recall": False,
        "failure_category": "eval_timeout",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timeout_seconds": timeout_seconds,
    }


def is_query_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, QueryTimeoutError):
        return True
    return "Search exceeded per-query timeout" in str(exc)


def enforce_completed_query_timeout(*, start_time: float, timeout_seconds: int) -> float:
    elapsed_seconds = time.time() - start_time
    if timeout_seconds > 0 and elapsed_seconds > timeout_seconds:
        raise QueryTimeoutError(f"Search exceeded per-query timeout of {timeout_seconds} seconds.")
    return elapsed_seconds


def _query_postgres_rows(sql: str) -> list[dict[str, Any]]:
    if shutil.which("docker"):
        copy_sql = f"COPY ({sql}) TO STDOUT WITH CSV HEADER"
        cmd = ["docker", "exec", "-i", "compose-postgres-1", "psql", "-U", "manuals", "-d", "manuals_rag", "-c", copy_sql]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        reader = csv.DictReader(completed.stdout.splitlines())
        return [dict(row) for row in reader]

    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(os.getenv("POSTGRES_DSN", "postgresql://manuals:manuals@postgres:5432/manuals_rag"), row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            return [dict(row) for row in cursor.fetchall()]


def _json_request(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            body = None
            req_headers = {"Accept": "application/json", **(headers or {})}
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                req_headers["Content-Type"] = "application/json"
            req = request.Request(url, data=body, headers=req_headers, method=method)
            with request.urlopen(req, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, ConnectionResetError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def _multipart_upload(url: str, *, file_path: Path, form_fields: dict[str, str], headers: dict[str, str]) -> Any:
    boundary = f"----manualsrag{uuid.uuid4().hex}"
    body = bytearray()
    for key, value in form_fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/pdf"
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="files"; filename="{file_path.name}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    req = request.Request(
        url,
        data=bytes(body),
        headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with request.urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def select_large_documents(directory: Path, *, max_docs: int, max_bytes: int) -> list[Path]:
    files = sorted(
        [path for path in directory.glob("*.pdf") if path.stat().st_size <= max_bytes],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    preferred = [path for path in files if any(token in path.name.lower() for token in ["_um_", "manual", "_c_", "datasheet"])]
    selected: list[Path] = []
    for path in preferred + files:
        if path in selected:
            continue
        selected.append(path)
        if len(selected) >= max_docs:
            break
    return selected


def create_corpus(corpus_id: str) -> None:
    _json_request(
        f"{API_BASE}/corpora",
        method="POST",
        payload={"id": corpus_id, "name": corpus_id, "permissions": {"roles": ["admin", "operator", "end_user", "auditor"]}},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )


def requeue_embed_job(*, run_id: str, document_id: str, version_id: str) -> None:
    payload = json.dumps({"run_id": run_id, "document_id": document_id, "version_id": version_id}, separators=(",", ":"))
    subprocess.run(
        ["docker", "exec", "compose-redis-1", "redis-cli", "LPUSH", "embed_jobs", payload],
        check=True,
        capture_output=True,
        text=True,
    )


def upload_and_ingest(file_path: Path, *, corpus_id: str) -> dict[str, Any]:
    uploaded = _multipart_upload(
        f"{API_BASE}/documents/upload",
        file_path=file_path,
        form_fields={"corpus_id": corpus_id},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    document = uploaded["uploaded"][0]
    ingest = _json_request(
        f"{API_BASE}/documents/{document['document_id']}/ingest",
        method="POST",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    run_id = ingest["run_id"]
    deadline = time.time() + 3600
    parsed_polls = 0
    requeued = False
    while time.time() < deadline:
        run = _json_request(
            f"{API_BASE}/ingestion-runs/{run_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        if run["status"] == "parsed":
            parsed_polls += 1
            if parsed_polls >= 3 and not requeued:
                requeue_embed_job(run_id=run_id, document_id=document["document_id"], version_id=document["version_id"])
                requeued = True
            time.sleep(2)
            continue
        if run["status"] == "completed":
            doc = _json_request(
                f"{API_BASE}/documents/{document['document_id']}",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )
            return {
                "document_id": document["document_id"],
                "version_id": document["version_id"],
                "filename": file_path.name,
                "run_id": run_id,
                "document": doc,
            }
        if run["status"] == "failed":
            raise RuntimeError(f"Ingestion failed for {file_path.name}: {run.get('failure_reason')}")
        time.sleep(2)
    raise TimeoutError(f"Ingestion timed out for {file_path.name}")


def fetch_chunk_rows(document_ids: list[str]) -> list[dict[str, Any]]:
    if not document_ids:
        return []
    id_list = ",".join(f"'{value}'" for value in document_ids)
    sql = f"""
        SELECT
            rc.id,
            rc.source_document_id,
            rc.document_version_id,
            rc.chunk_type,
            rc.chunk_level,
            rc.title,
            rc.section_path_text,
            rc.page_from,
            rc.page_to,
            rc.content,
            rc.metadata_json::text AS metadata_json,
            sd.source_filename,
            COALESCE(sd.product_model, rc.metadata_json->>'product_model', '') AS product_model
        FROM retrieval_chunks rc
        JOIN source_documents sd ON sd.id = rc.source_document_id
        WHERE rc.source_document_id::text IN ({id_list})
          AND rc.chunk_level = 1
          AND length(rc.content) >= 40
          AND rc.chunk_type IN ('table_record','datasheet_record','spec_record','procedure_record','warning_record','atomic_text')
        ORDER BY
          CASE rc.chunk_type
            WHEN 'datasheet_record' THEN 1
            WHEN 'spec_record' THEN 2
            WHEN 'procedure_record' THEN 3
            WHEN 'warning_record' THEN 4
            WHEN 'table_record' THEN 5
            ELSE 6
          END,
          length(rc.content) DESC
    """
    rows: list[dict[str, Any]] = []
    for row in _query_postgres_rows(sql):
        row["metadata_json"] = json.loads(row.get("metadata_json") or "{}")
        row["page_from"] = int(row["page_from"])
        row["page_to"] = int(row["page_to"])
        rows.append(row)
    return rows


def fetch_documents_for_corpus(corpus_id: str) -> list[dict[str, Any]]:
    sql = f"""
        SELECT id, current_version_id, source_filename, title, ingest_status
        FROM source_documents
        WHERE corpus_id = '{corpus_id}'
        ORDER BY created_at
    """
    return _query_postgres_rows(sql)


def run_search(query: str, *, corpus_id: str, response_mode: str = "retrieval_only") -> dict[str, Any] | list[dict[str, Any]]:
    payload = {
        "query": query,
        "corpus_ids": [corpus_id],
        "filters": {},
        "response_mode": response_mode,
    }
    return _json_request(
        f"{API_BASE}/search",
        method="POST",
        payload=payload,
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )


def run_query_answer(query: str, *, corpus_id: str) -> dict[str, Any]:
    payload = {
        "query": query,
        "corpus_ids": [corpus_id],
        "filters": {},
        "response_mode": "answer_with_citations",
    }
    return _json_request(
        f"{API_BASE}/query",
        method="POST",
        payload=payload,
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )


def run_search_direct(query: str, *, corpus_id: str) -> list[dict[str, Any]]:
    from manuals_rag_retrieval.retriever import build_filters, retrieve

    filters = build_filters(query, {})
    return [item.model_dump() for item in retrieve(query, [corpus_id], filters)]


def generate_answer_payload(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    from manuals_rag_answering.generator import generate_answer_with_trace
    from manuals_rag_schemas.documents import SearchResult

    answer, trace = generate_answer_with_trace(query, [SearchResult(**item) for item in results])
    payload = answer.model_dump()
    final_answer_trace = trace.get("final_answer", {})
    payload["_eval_trace"] = {
        "answer_source": final_answer_trace.get("answer_source"),
        "used_fallback": bool(final_answer_trace.get("used_fallback")),
        "fallback_reason": final_answer_trace.get("fallback_reason"),
        "summary_count": trace.get("summarization", {}).get("summary_count"),
    }
    return payload


def normalize_api_answer_payload(answer: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(answer)
    trace = normalized.get("_eval_trace")
    if not isinstance(trace, dict):
        trace = {}
    normalized["_eval_trace"] = {
        **trace,
        "answer_transport": "http_api",
        "answer_source": trace.get("answer_source") or "api",
        "used_fallback": bool(trace.get("used_fallback", False)),
    }
    return normalized


def _answer_trace(answer: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(answer, dict):
        return {}
    trace = answer.get("_eval_trace")
    return trace if isinstance(trace, dict) else {}


def run_case_search(query: str, *, corpus_id: str, search_mode: str, response_mode: str = "retrieval_only") -> dict[str, Any]:
    if search_mode == "direct":
        results = run_search_direct(query, corpus_id=corpus_id)
        payload: dict[str, Any] = {"top_results": results}
        if response_mode == "answer_with_citations":
            payload["answer"] = generate_answer_payload(query, results)
        return payload
    payload = run_search(query, corpus_id=corpus_id, response_mode=response_mode)
    if isinstance(payload, list):
        results = payload
    else:
        results = list(payload.get("top_results") or payload.get("results") or [])
    normalized = {"top_results": results}
    if response_mode == "answer_with_citations":
        if isinstance(payload, dict) and isinstance(payload.get("answer"), dict):
            normalized["answer"] = normalize_api_answer_payload(payload["answer"])
        else:
            normalized["answer"] = normalize_api_answer_payload(run_query_answer(query, corpus_id=corpus_id))
    return normalized


def _top_results_from_search_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    return list(payload.get("top_results", []))


def run_warmup_searches(
    cases: list[dict[str, Any]],
    *,
    corpus_id: str,
    search_mode: str,
    warmup_queries: int,
    warmup_timeout_seconds: int,
) -> list[dict[str, Any]]:
    warmups: list[dict[str, Any]] = []
    if warmup_queries <= 0:
        return warmups
    for case in cases[:warmup_queries]:
        start_time = time.time()
        try:
            with query_timeout(warmup_timeout_seconds):
                search_payload = run_case_search(case["query"], corpus_id=corpus_id, search_mode=search_mode)
            warmups.append(
                {
                    "case_id": case["case_id"],
                    "status": "completed",
                    "elapsed_seconds": round(time.time() - start_time, 3),
                    "result_count": len(_top_results_from_search_payload(search_payload)),
                    "answer_generated": isinstance(search_payload, dict) and bool(search_payload.get("answer")),
                }
            )
        except Exception as exc:
            if not is_query_timeout_exception(exc):
                raise
            warmups.append(
                {
                    "case_id": case["case_id"],
                    "status": "eval_timeout",
                    "elapsed_seconds": round(time.time() - start_time, 3),
                    "timeout_seconds": warmup_timeout_seconds,
                    "result_count": 0,
                }
            )
    return warmups


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {path} line {line_number}.") from exc
    return records


def load_eval_cases_from_dataset(path: Path, *, max_cases: int) -> list[dict[str, Any]]:
    cases, _ = load_eval_cases_and_rejections_from_dataset(path, max_cases=max_cases, drop_invalid_cases=False)
    return cases


def _source_chunk_from_saved_case(case: RetrievalEvalCase) -> dict[str, Any]:
    metadata = dict(case.source_metadata or {})
    return {
        "id": case.source_chunk_id,
        "source_document_id": case.source_document_id,
        "document_version_id": case.document_version_id,
        "chunk_type": case.chunk_type,
        "title": case.source_title,
        "source_filename": case.source_filename,
        "section_path_text": case.section_path,
        "page_from": case.page_from,
        "page_to": case.page_to,
        "content": case.expected_snippet,
        "metadata_json": metadata,
        "product_model": metadata.get("product_model", ""),
    }


def saved_case_quality_rejection_reason(case: RetrievalEvalCase) -> str | None:
    if case.retrieval_task != "single_step_retrieval":
        return None
    chunk = _source_chunk_from_saved_case(case)
    anchors = list(case.anchor_terms or case.expected_terms)
    if not chunk_is_queryworthy(chunk, anchors):
        return "not_queryworthy_source_chunk"
    return None


def load_eval_cases_and_rejections_from_dataset(
    path: Path,
    *,
    max_cases: int,
    drop_invalid_cases: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in read_jsonl(path):
        case = record.get("case") if isinstance(record.get("case"), dict) else record
        if not isinstance(case, dict):
            raise ValueError(f"Dataset record in {path} does not contain an eval case object.")
        eval_case = RetrievalEvalCase(**case)
        if drop_invalid_cases:
            rejection_reason = saved_case_quality_rejection_reason(eval_case)
            if rejection_reason:
                rejected.append(
                    {
                        "case_id": eval_case.case_id,
                        "query": eval_case.query,
                        "source_chunk_id": eval_case.source_chunk_id,
                        "reason": rejection_reason,
                    }
                )
                continue
        cases.append(eval_case.to_dict())
        if len(cases) >= max_cases:
            break
    if not cases:
        raise ValueError(f"No eval cases found in {path}.")
    return cases, rejected


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result["evaluation"]["passed"])
    ranks = [result["evaluation"]["rank"] for result in results if result["evaluation"]["rank"] is not None]
    chunk_type_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    retrieval_task_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    doc_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    failure_categories = Counter()
    benchmark_quality = Counter()
    candidate_recall_hits = 0
    metadata_selection_attempts = 0
    metadata_selection_hits = 0
    metadata_selection_rank_1_hits = 0
    answer_eval_count = 0
    answer_passed = 0
    answer_failure_reasons = Counter()
    answer_elapsed_seconds: list[float] = []
    answer_fallback_count = 0
    answer_sources = Counter()
    answer_fallback_reasons = Counter()
    answer_summary_counts: list[int] = []
    for result in results:
        chunk_type = result["case"]["chunk_type"]
        chunk_type_stats[chunk_type]["total"] += 1
        chunk_type_stats[chunk_type]["passed"] += int(result["evaluation"]["passed"])
        retrieval_task = result["case"].get("retrieval_task", "single_step_retrieval")
        retrieval_task_stats[retrieval_task]["total"] += 1
        retrieval_task_stats[retrieval_task]["passed"] += int(result["evaluation"]["passed"])
        filename = result["case"]["source_filename"]
        doc_stats[filename]["total"] += 1
        doc_stats[filename]["passed"] += int(result["evaluation"]["passed"])
        benchmark_quality[result["case"].get("benchmark_quality", "unknown")] += 1
        if result["evaluation"].get("failure_category"):
            failure_categories[result["evaluation"]["failure_category"]] += 1
        if result["evaluation"].get("candidate_recall"):
            candidate_recall_hits += 1
        metadata_selection = result["evaluation"].get("metadata_document_selection", {})
        if metadata_selection.get("attempted"):
            metadata_selection_attempts += 1
            metadata_selection_hits += int(bool(metadata_selection.get("passed")))
            metadata_selection_rank_1_hits += int(metadata_selection.get("rank") == 1)
        answer_evaluation = result.get("answer_evaluation")
        if isinstance(answer_evaluation, dict):
            answer_eval_count += 1
            answer_passed += int(bool(answer_evaluation.get("passed")))
            answer_failure_reasons.update(answer_evaluation.get("failure_reasons", []))
            elapsed_seconds = answer_evaluation.get("elapsed_seconds")
            if isinstance(elapsed_seconds, int | float):
                answer_elapsed_seconds.append(float(elapsed_seconds))
        trace = _answer_trace(result.get("answer"))
        if trace:
            if trace.get("used_fallback"):
                answer_fallback_count += 1
            if trace.get("answer_source"):
                answer_sources[str(trace["answer_source"])] += 1
            if trace.get("fallback_reason"):
                answer_fallback_reasons[str(trace["fallback_reason"])] += 1
            summary_count = trace.get("summary_count")
            if isinstance(summary_count, int):
                answer_summary_counts.append(summary_count)
    sorted_answer_elapsed = sorted(answer_elapsed_seconds)
    answer_latency = {
        "min_seconds": round(sorted_answer_elapsed[0], 3),
        "max_seconds": round(sorted_answer_elapsed[-1], 3),
        "mean_seconds": round(sum(sorted_answer_elapsed) / len(sorted_answer_elapsed), 3),
    } if sorted_answer_elapsed else None
    if sorted_answer_elapsed:
        p95_index = max(0, min(len(sorted_answer_elapsed) - 1, int(len(sorted_answer_elapsed) * 0.95 + 0.999) - 1))
        answer_latency["p95_seconds"] = round(sorted_answer_elapsed[p95_index], 3)
    return {
        "total_queries": total,
        "passed_queries": passed,
        "failed_queries": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "pass_at_1": round(sum(1 for result in results if result["evaluation"]["rank"] == 1) / total, 4) if total else 0.0,
        "pass_at_3": round(sum(1 for result in results if (result["evaluation"]["rank"] or 999) <= 3) / total, 4) if total else 0.0,
        "pass_at_5": round(sum(1 for result in results if (result["evaluation"]["rank"] or 999) <= 5) / total, 4) if total else 0.0,
        "mean_passing_rank": round(sum(ranks) / len(ranks), 3) if ranks else None,
        "by_chunk_type": dict(chunk_type_stats),
        "by_retrieval_task": dict(retrieval_task_stats),
        "by_document": dict(doc_stats),
        "failure_categories": dict(failure_categories),
        "benchmark_quality": dict(benchmark_quality),
        "benchmark_validity_rate": round(benchmark_quality.get("validated", 0) / total, 4) if total else 0.0,
        "candidate_recall_rate": round(candidate_recall_hits / total, 4) if total else 0.0,
        "metadata_document_selection_attempts": metadata_selection_attempts,
        "metadata_document_selection_recall_rate": round(metadata_selection_hits / metadata_selection_attempts, 4)
        if metadata_selection_attempts
        else 0.0,
        "metadata_document_selection_rank_1_rate": round(metadata_selection_rank_1_hits / metadata_selection_attempts, 4)
        if metadata_selection_attempts
        else 0.0,
        "answer_eval_count": answer_eval_count,
        "answer_passed_queries": answer_passed,
        "answer_failed_queries": answer_eval_count - answer_passed,
        "answer_pass_rate": round(answer_passed / answer_eval_count, 4) if answer_eval_count else None,
        "answer_failure_reasons": dict(answer_failure_reasons),
        "answer_latency": answer_latency,
        "answer_fallback_count": answer_fallback_count if answer_eval_count else None,
        "answer_fallback_rate": round(answer_fallback_count / answer_eval_count, 4) if answer_eval_count else None,
        "answer_sources": dict(answer_sources),
        "answer_fallback_reasons": dict(answer_fallback_reasons),
        "answer_mean_summary_count": round(sum(answer_summary_counts) / len(answer_summary_counts), 3)
        if answer_summary_counts
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=240)
    parser.add_argument("--max-doc-bytes", type=int, default=90000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--existing-corpus-id", type=str, default=None)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Existing RetrievalEvalCase JSONL dataset to score instead of generating new cases.",
    )
    parser.add_argument(
        "--drop-invalid-saved-cases",
        action="store_true",
        help="When loading --dataset-path, omit stale saved single-step cases whose source chunks fail the current queryworthiness gate.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["http", "direct"],
        default="http",
        help="Use the HTTP API or call the app retriever in-process. Direct mode is intended for fast retrieval-only saved-bank evals.",
    )
    parser.add_argument(
        "--response-mode",
        choices=["retrieval_only", "answer_with_citations"],
        default="retrieval_only",
        help="Score retrieval only or also generate and score final answers with citations.",
    )
    parser.add_argument(
        "--per-query-timeout-seconds",
        type=int,
        default=0,
        help="Abort and score a single query as eval_timeout after this many seconds. 0 disables per-query timeout.",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=0,
        help="Run this many unscored searches before timed evaluation to pay model/retriever startup cost outside the benchmark.",
    )
    parser.add_argument(
        "--warmup-timeout-seconds",
        type=int,
        default=0,
        help="Abort an unscored warmup query after this many seconds. 0 uses max(30, per-query-timeout-seconds * 4).",
    )
    parser.add_argument(
        "--disable-llm-query-generation",
        action="store_true",
        help="Use deterministic eval question generation instead of local LLM rewrites.",
    )
    parser.add_argument(
        "--retrieval-task",
        choices=["single_step_retrieval", "multi_step_retrieval"],
        default="single_step_retrieval",
        help="Generate cases for the selected retrieval task when --dataset-path is not provided.",
    )
    parser.add_argument(
        "--multi-step-case-family",
        choices=["all", "sibling_table_rows", "contextual_section", "warning_step", "cross_document"],
        default="all",
        help="Limit generated multi-step cases to a specific coverage family.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    if args.dataset_path and not args.existing_corpus_id:
        raise SystemExit("--dataset-path requires --existing-corpus-id so saved cases are searched in the intended corpus.")

    if args.existing_corpus_id:
        corpus_id = args.existing_corpus_id
        corpus_documents = fetch_documents_for_corpus(corpus_id)
        ingested_docs = [
            {
                "document_id": row["id"],
                "version_id": row["current_version_id"],
                "filename": row["source_filename"],
                "document": {"title": row["title"], "ingest_status": row["ingest_status"]},
            }
            for row in corpus_documents
            if row["ingest_status"] == "indexed"
        ]
        selected_docs = [Path(item["filename"]) for item in ingested_docs]
        if not ingested_docs:
            raise SystemExit(f"No indexed documents found for corpus {corpus_id}.")
        print(json.dumps({"existing_corpus_id": corpus_id, "indexed_documents": [item["filename"] for item in ingested_docs]}, indent=2), flush=True)
    else:
        corpus_id = f"manuals_eval_{time.strftime('%Y%m%d_%H%M%S')}"
        selected_docs = select_large_documents(args.docs_dir, max_docs=args.max_docs, max_bytes=args.max_doc_bytes)
        if not selected_docs:
            raise SystemExit("No documents selected for ingestion.")
        print(json.dumps({"selected_documents": [path.name for path in selected_docs], "corpus_id": corpus_id}, indent=2), flush=True)
        create_corpus(corpus_id)
        ingested_docs = []
        for path in selected_docs:
            print(json.dumps({"ingesting": path.name, "size_bytes": path.stat().st_size}, indent=2), flush=True)
            ingested_docs.append(upload_and_ingest(path, corpus_id=corpus_id))

    if args.dataset_path:
        cases, rejected_cases = load_eval_cases_and_rejections_from_dataset(
            args.dataset_path,
            max_cases=args.max_queries,
            drop_invalid_cases=args.drop_invalid_saved_cases,
        )
        print(
            json.dumps(
                {
                    "dataset_path": str(args.dataset_path),
                    "loaded_cases": len(cases),
                    "dropped_invalid_cases": len(rejected_cases),
                    "rejected_cases": rejected_cases,
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        rejected_cases = []
        chunk_rows = fetch_chunk_rows([item["document_id"] for item in ingested_docs])
        random.shuffle(chunk_rows)
        if args.retrieval_task == "multi_step_retrieval":
            cases = [
                case.to_dict()
                for case in build_multi_step_eval_cases_from_chunks(
                    chunk_rows,
                    max_cases=args.max_queries,
                    case_family=args.multi_step_case_family,
                )
            ]
        else:
            cases = [
                case.to_dict()
                for case in build_eval_cases_from_chunks(
                    chunk_rows,
                    max_cases=args.max_queries,
                    use_llm_generation=not args.disable_llm_query_generation,
                )
            ]

    warmup_timeout_seconds = args.warmup_timeout_seconds
    if args.warmup_queries > 0 and warmup_timeout_seconds <= 0:
        warmup_timeout_seconds = max(30, args.per_query_timeout_seconds * 4)
    warmups = run_warmup_searches(
        cases,
        corpus_id=corpus_id,
        search_mode=args.search_mode,
        warmup_queries=args.warmup_queries,
        warmup_timeout_seconds=warmup_timeout_seconds,
    )
    for warmup in warmups:
        print(json.dumps({"warmup": warmup}, indent=2), flush=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_path = OUTPUT_DIR / f"retrieval_eval_dataset_{timestamp}.jsonl"
    results_path = OUTPUT_DIR / f"retrieval_eval_results_{timestamp}.jsonl"
    summary_path = OUTPUT_DIR / f"retrieval_eval_summary_{timestamp}.json"
    manifest_path = OUTPUT_DIR / f"retrieval_eval_manifest_{timestamp}.json"

    write_jsonl(dataset_path, cases)

    results: list[dict[str, Any]] = []
    for case in cases:
        start_time = time.time()
        answer: dict[str, Any] | None = None
        answer_evaluation: dict[str, Any] | None = None
        try:
            with query_timeout(args.per_query_timeout_seconds):
                search_payload = run_case_search(
                    case["query"],
                    corpus_id=corpus_id,
                    search_mode=args.search_mode,
                    response_mode=args.response_mode,
                )
            search_results = search_payload.get("top_results", [])
            eval_case = RetrievalEvalCase(**case)
            evaluation = score_search_results(eval_case, search_results)
            evaluation["elapsed_seconds"] = round(time.time() - start_time, 3)
            if args.response_mode == "answer_with_citations":
                answer = dict(search_payload.get("answer") or {})
                answer_evaluation = score_answer_response(eval_case, answer, evaluation)
                answer_evaluation["elapsed_seconds"] = evaluation["elapsed_seconds"]
            elapsed_seconds = enforce_completed_query_timeout(
                start_time=start_time,
                timeout_seconds=args.per_query_timeout_seconds,
            )
            evaluation["elapsed_seconds"] = round(elapsed_seconds, 3)
            if answer_evaluation is not None:
                answer_evaluation["elapsed_seconds"] = evaluation["elapsed_seconds"]
        except Exception as exc:
            if not is_query_timeout_exception(exc):
                raise
            search_results = []
            evaluation = timeout_evaluation(
                case,
                elapsed_seconds=time.time() - start_time,
                timeout_seconds=args.per_query_timeout_seconds,
            )
            if args.response_mode == "answer_with_citations":
                answer = {}
                answer_evaluation = {
                    "passed": False,
                    "failure_reasons": ["eval_timeout"],
                    "expected_document_used": False,
                    "elapsed_seconds": evaluation["elapsed_seconds"],
                }
        result_record: dict[str, Any] = {"case": case, "evaluation": evaluation, "top_results": search_results[:5]}
        if args.response_mode == "answer_with_citations":
            result_record["answer"] = answer or {}
            result_record["answer_evaluation"] = answer_evaluation or {}
        results.append(result_record)
        print(
            json.dumps(
                {
                    "case_id": case["case_id"],
                    "passed": evaluation["passed"],
                    "rank": evaluation["rank"],
                    "failure_category": evaluation.get("failure_category"),
                    "answer_passed": answer_evaluation.get("passed") if answer_evaluation else None,
                    "answer_failure_reasons": answer_evaluation.get("failure_reasons") if answer_evaluation else None,
                    "answer_used_fallback": _answer_trace(answer).get("used_fallback") if answer else None,
                    "answer_source": _answer_trace(answer).get("answer_source") if answer else None,
                    "elapsed_seconds": evaluation.get("elapsed_seconds"),
                },
                indent=2,
            ),
            flush=True,
        )

    write_jsonl(results_path, results)
    summary = summarize(results)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_id": corpus_id,
                "documents": ingested_docs,
                "input_dataset_path": str(args.dataset_path) if args.dataset_path else None,
                "search_mode": args.search_mode,
                "response_mode": args.response_mode,
                "per_query_timeout_seconds": args.per_query_timeout_seconds,
                "warmup_queries": args.warmup_queries,
                "warmup_timeout_seconds": warmup_timeout_seconds,
                "warmups": warmups,
                "dropped_invalid_cases": len(rejected_cases),
                "rejected_cases": rejected_cases,
                "dataset_path": str(dataset_path),
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "selected_documents": [str(path) for path in selected_docs],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({"dataset_path": str(dataset_path), "results_path": str(results_path), "summary_path": str(summary_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
