from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import random
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[3]
MANUALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "evals" / "src"))

from manuals_rag_evals.retrieval_eval import build_eval_cases_from_chunks, score_search_results
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase


API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8600")
ADMIN_TOKEN = os.getenv("LOCAL_ADMIN_TOKEN", "admin-token")
USER_TOKEN = os.getenv("LOCAL_END_USER_TOKEN", "user-token")
DEFAULT_DOCS_DIR = REPO_ROOT / "Technical_Documents" / "Keyence"
OUTPUT_DIR = MANUALS_ROOT / "test_reports"


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
    COPY (
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
    ) TO STDOUT WITH CSV HEADER
    """
    cmd = ["docker", "exec", "-i", "compose-postgres-1", "psql", "-U", "manuals", "-d", "manuals_rag", "-c", sql]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    reader = csv.DictReader(completed.stdout.splitlines())
    rows: list[dict[str, Any]] = []
    for row in reader:
        row["metadata_json"] = json.loads(row.get("metadata_json") or "{}")
        row["page_from"] = int(row["page_from"])
        row["page_to"] = int(row["page_to"])
        rows.append(row)
    return rows


def fetch_documents_for_corpus(corpus_id: str) -> list[dict[str, Any]]:
    sql = f"""
    COPY (
        SELECT id, current_version_id, source_filename, title, ingest_status
        FROM source_documents
        WHERE corpus_id = '{corpus_id}'
        ORDER BY created_at
    ) TO STDOUT WITH CSV HEADER
    """
    cmd = ["docker", "exec", "-i", "compose-postgres-1", "psql", "-U", "manuals", "-d", "manuals_rag", "-c", sql]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    reader = csv.DictReader(completed.stdout.splitlines())
    return [dict(row) for row in reader]


def run_search(query: str, *, corpus_id: str) -> list[dict[str, Any]]:
    payload = {
        "query": query,
        "corpus_ids": [corpus_id],
        "filters": {},
        "response_mode": "answer_with_citations",
    }
    return _json_request(
        f"{API_BASE}/search",
        method="POST",
        payload=payload,
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result["evaluation"]["passed"])
    ranks = [result["evaluation"]["rank"] for result in results if result["evaluation"]["rank"] is not None]
    chunk_type_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    doc_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    failure_categories = Counter()
    benchmark_quality = Counter()
    candidate_recall_hits = 0
    metadata_selection_attempts = 0
    metadata_selection_hits = 0
    metadata_selection_rank_1_hits = 0
    for result in results:
        chunk_type = result["case"]["chunk_type"]
        chunk_type_stats[chunk_type]["total"] += 1
        chunk_type_stats[chunk_type]["passed"] += int(result["evaluation"]["passed"])
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=240)
    parser.add_argument("--max-doc-bytes", type=int, default=90000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--existing-corpus-id", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
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

    chunk_rows = fetch_chunk_rows([item["document_id"] for item in ingested_docs])
    random.shuffle(chunk_rows)
    cases = [case.to_dict() for case in build_eval_cases_from_chunks(chunk_rows, max_cases=args.max_queries)]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_path = OUTPUT_DIR / f"retrieval_eval_dataset_{timestamp}.jsonl"
    results_path = OUTPUT_DIR / f"retrieval_eval_results_{timestamp}.jsonl"
    summary_path = OUTPUT_DIR / f"retrieval_eval_summary_{timestamp}.json"
    manifest_path = OUTPUT_DIR / f"retrieval_eval_manifest_{timestamp}.json"

    write_jsonl(dataset_path, cases)

    results: list[dict[str, Any]] = []
    for case in cases:
        search_results = run_search(case["query"], corpus_id=corpus_id)
        evaluation = score_search_results(RetrievalEvalCase(**case), search_results)
        results.append({"case": case, "evaluation": evaluation, "top_results": search_results[:5]})
        print(json.dumps({"case_id": case["case_id"], "passed": evaluation["passed"], "rank": evaluation["rank"]}, indent=2), flush=True)

    write_jsonl(results_path, results)
    summary = summarize(results)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_id": corpus_id,
                "documents": ingested_docs,
                "dataset_path": str(dataset_path),
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "selected_documents": [str(path) for path in selected_docs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({"dataset_path": str(dataset_path), "results_path": str(results_path), "summary_path": str(summary_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
