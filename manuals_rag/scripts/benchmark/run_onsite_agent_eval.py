#!/usr/bin/env python3
"""Evaluate source-anchored onsite questions through the Agent page's live path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


BACKENDS = ("langgraph_agent", "llamaindex_agent")


def live_request(case: dict, backend: str, corpus: str) -> tuple[dict, list[dict]]:
    """Use the Agent page's /agent/stream contract without external CLI tools."""
    body = {
        "query": case["question"],
        "corpus_ids": [corpus],
        "filters": {},
        "retrieval_orchestrator": backend,
        "max_retrieval_hops": 6,
        "max_retrieval_seconds": 90,
        "max_tool_calls": 4,
        "session_id": f"onsite-eval-{case['case_id']}-{backend}-{time.time_ns()}",
        "conversation_history": [],
        "session_memory": [],
    }
    base = os.getenv("MANUALS_RAG_API_BASE", "http://127.0.0.1:8600").rstrip("/")
    token = os.getenv("MANUALS_RAG_MATRIX_TOKEN", "admin-token")
    request = Request(
        f"{base}/agent/stream",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    events = []
    with urlopen(request, timeout=310) as response:
        for raw in response:
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                events.append({"event": "unparseable_stream_line"})
    result = next((event.get("result") for event in reversed(events)
                   if event.get("event") in {"run_completed", "answer_completed"} and event.get("result")), None)
    if not result:
        failure = next((event for event in reversed(events) if event.get("event") == "run_failed"), {})
        raise RuntimeError(str(failure.get("error") or "Agent stream returned no completed result"))
    return dict(result), events


def load_bank(path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if manifest.get("schema") != "manuals-rag-onsite-agent-eval-v1":
        raise ValueError("Wrong onsite evaluation manifest schema")
    if manifest.get("sha256") != hashlib.sha256(raw).hexdigest() or manifest.get("case_count") != len(rows):
        raise ValueError("Onsite evaluation bank changed after freezing")
    ids = [row.get("case_id") for row in rows]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("Onsite case IDs must be unique and nonempty")
    for row in rows:
        if not all(row.get(key) for key in ("question", "source_document_id", "source_chunk_id", "source_filename", "expected_facets")):
            raise ValueError(f"Incomplete source/answer contract: {row.get('case_id')}")
        if not all(isinstance(facet, list) and facet and all(isinstance(term, str) and term for term in facet) for facet in row["expected_facets"]):
            raise ValueError(f"Invalid answer facets: {row['case_id']}")
    return rows, manifest


def grade(case: dict, backend: str, result: dict, events: list[dict]) -> dict:
    answer = str(result.get("answer") or "")
    citations = result.get("citations") or []
    cited_documents = {str(c.get("document_id") or c.get("source_document_id") or "") for c in citations if isinstance(c, dict)}
    cited_chunks = {str(c.get("chunk_id") or c.get("source_chunk_id") or "") for c in citations if isinstance(c, dict)}
    checks = {
        "completed": bool(answer),
        "supported": not bool(result.get("insufficient_evidence")),
        "cited_expected_document": str(case["source_document_id"]) in cited_documents,
        "answer_facets": all(any(term.casefold() in answer.casefold() for term in facet) for facet in case["expected_facets"]),
    }
    # A citation to a different chunk in the right manual can still be valid.
    # Preserve exact-chunk information for review without making it a false failure.
    trace = result.get("agent_trace") or {}
    calls = [call for call in trace.get("tool_calls") or [] if call.get("action") == "manuals_rag"]
    checks["used_manuals_rag"] = bool(calls)
    return {
        "backend": backend,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "answer": answer,
        "citations": citations,
        "expected_chunk_cited": str(case["source_chunk_id"]) in cited_chunks,
        "rag_calls": len(calls),
        "event_count": len(events),
    }


def run(dataset: Path, manifest_path: Path, output: Path, limit: int | None = None) -> dict:
    rows, manifest = load_bank(dataset, manifest_path)
    selected = rows[:limit] if limit else rows
    run_id = output.stem
    report = {
        "schema": "manuals-rag-onsite-agent-eval-result-v1",
        "run_id": run_id,
        "dataset": str(dataset),
        "dataset_sha256": manifest["sha256"],
        "selected_case_ids": [case["case_id"] for case in selected],
        "backends": list(BACKENDS),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "records": [],
    }
    partial = output.with_suffix(".partial.json")
    partial_tmp = output.with_suffix(".partial.tmp")
    partial.parent.mkdir(parents=True, exist_ok=True)
    def persist_partial() -> None:
        partial_tmp.write_text(json.dumps(report, indent=2) + "\n")
        partial_tmp.replace(partial)

    persist_partial()
    for case in selected:
        record = {"case_id": case["case_id"], "question": case["question"], "backends": {}}
        report["records"].append(record)
        for backend in BACKENDS:
            try:
                result, events = live_request(case, backend, manifest["corpus"])
                record["backends"][backend] = grade(case, backend, result, events)
            except (OSError, RuntimeError, TimeoutError) as error:
                record["backends"][backend] = {"backend": backend, "status": "error", "error": f"{type(error).__name__}: {error}"}
            persist_partial()
            print(json.dumps({"event": "onsite_cell_completed", "case_id": case["case_id"], "backend": backend, "status": record["backends"][backend]["status"]}), flush=True)
    total = len(selected) * len(BACKENDS)
    cells = [cell for row in report["records"] for cell in row["backends"].values()]
    passed = sum(cell["status"] == "pass" for cell in cells)
    failed = sum(cell["status"] == "fail" for cell in cells)
    errors = sum(cell["status"] == "error" for cell in cells)
    report["summary"] = {"passed": passed, "failed": failed, "errors": errors, "total": total}
    report["status"] = "completed" if passed == total else "failed"
    report["exit_code"] = 0 if passed == total else 1
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_tmp = output.with_suffix(".tmp")
    output_tmp.write_text(json.dumps(report, indent=2) + "\n")
    output_tmp.replace(output)
    partial.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        report = run(args.dataset, args.manifest, args.output, args.limit)
    except (OSError, ValueError) as error:
        print(json.dumps({"event": "onsite_failed", "error": str(error)}), flush=True)
        return 2
    print(json.dumps({"event": "onsite_completed", "run_id": report["run_id"], "summary": report["summary"]}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
