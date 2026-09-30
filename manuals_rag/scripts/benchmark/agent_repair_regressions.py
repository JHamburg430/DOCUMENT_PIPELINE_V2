#!/usr/bin/env python3
"""Record verified Agent fixes and replay them from the app database before release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.ui.repair_store import list_cases, record_fix, record_regression, regression_cases, save_case


def evaluate_regression(spec: dict, events: list[dict]) -> tuple[bool, list[str], dict]:
    terminal = next((event for event in reversed(events) if event.get("event") == "run_completed"), None)
    result = terminal.get("result") if terminal else None
    failures: list[str] = []
    if not isinstance(result, dict):
        failures.append("No completed Agent result")
        result = {}
    answer = str(result.get("answer") or "")
    event_types = {str(event.get("event")) for event in events}
    for required in spec.get("required_event_types", []):
        if required not in event_types:
            failures.append(f"Missing event: {required}")
    for term in spec.get("required_answer_terms", []):
        if term.casefold() not in answer.casefold():
            failures.append(f"Missing answer term: {term}")
    citation_ids = {
        str(citation.get("source_document_id") or citation.get("document_id") or "")
        for citation in result.get("citations") or [] if isinstance(citation, dict)
    }
    for document_id in spec.get("required_citation_document_ids", []):
        if document_id not in citation_ids:
            failures.append(f"Missing cited document: {document_id}")
    if bool(result.get("insufficient_evidence")) != spec["expect_abstention"]:
        failures.append("Abstention outcome changed")
    return not failures, failures, {
        "answer": answer,
        "insufficient_evidence": result.get("insufficient_evidence"),
        "citation_document_ids": sorted(citation_ids),
        "event_types": sorted(event_types),
    }


def replay_case(case: dict, *, api_base: str, token: str, timeout: int) -> list[dict]:
    original = case["run_snapshot"]
    spec = case["regression_spec"]
    backends = original.get("backends") or ["langgraph_agent"]
    if len(backends) != 1:
        raise ValueError("Regression replay requires one original Agent backend")
    body = {
        "query": spec.get("replay_query") or original["query"],
        "corpus_ids": original.get("corpus_ids") or ["manuals_vendor_keyence"],
        "retrieval_orchestrator": backends[0],
        "max_retrieval_hops": original.get("max_retrieval_hops") or 6,
        "max_tool_calls": original.get("max_tool_calls") or 4,
        "max_wait_seconds": original.get("max_wait_seconds") or 300,
        "conversation_history": [],
        "session_memory": [],
    }
    request = Request(
        f"{api_base.rstrip('/')}/agent/stream",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "Accept": "application/x-ndjson"},
        method="POST",
    )
    events = []
    with urlopen(request, timeout=timeout) as response:
        for line in response:
            if line.strip():
                events.append(json.loads(line))
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.getenv("POSTGRES_DSN", "postgresql://manuals:manuals@postgres:5432/manuals_rag"))
    subcommands = parser.add_subparsers(dest="command", required=True)
    listing = subcommands.add_parser("list", help="List saved repair cases without exposing full run traces")
    listing.add_argument("--limit", type=int, default=100)
    legacy = subcommands.add_parser("import-legacy", help="Import prior JSON repair packets into PostgreSQL")
    legacy.add_argument("--directory", type=Path, default=Path("test_reports/agent_repair_requests"))
    fix = subcommands.add_parser("record-fix", help="Link a verified code fix and measurable replay contract to a saved case")
    fix.add_argument("--request-id", required=True)
    fix.add_argument("--root-cause", required=True)
    fix.add_argument("--summary", required=True)
    fix.add_argument("--commit", required=True)
    fix.add_argument("--regression-spec", type=Path, required=True, help="JSON assertions: expect_abstention plus answer terms, citation IDs, or event types")
    fix.add_argument("--verification", type=Path, required=True, help="JSON evidence of checks performed")
    check = subcommands.add_parser("check", help="Replay every latest saved fix and record results in PostgreSQL")
    check.add_argument("--revision", required=True)
    check.add_argument("--api-base", default=os.getenv("MANUALS_RAG_API_BASE", "http://api:8600"))
    check.add_argument("--timeout", type=int, default=930)
    check.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(list_cases(args.dsn, limit=args.limit), indent=2))
        return 0
    if args.command == "import-legacy":
        imported = 0
        for path in sorted(args.directory.glob("repair-agent-run-*.json")):
            packet = json.loads(path.read_text(encoding="utf-8"))
            case = save_case(args.dsn, packet["run"])
            if case["source_run_id"] != packet["source_run_id"]:
                raise ValueError(f"Legacy packet did not reconcile: {path}")
            imported += 1
        print(json.dumps({"status": "reconciled", "packets": imported}))
        return 0
    if args.command == "record-fix":
        fix_id = record_fix(
            args.dsn, request_id=args.request_id, root_cause=args.root_cause,
            fix_summary=args.summary, commit_sha=args.commit,
            regression_spec=json.loads(args.regression_spec.read_text()),
            verification=json.loads(args.verification.read_text()),
        )
        print(json.dumps({"fix_id": fix_id, "request_id": args.request_id, "status": "fixed"}))
        return 0
    cases = regression_cases(args.dsn)
    if not cases:
        print(json.dumps({"status": "no_regression_cases", "checked": 0}))
        return 0 if args.allow_empty else 2
    token = os.getenv("MANUALS_RAG_AUTH_TOKEN", "admin-token")
    passed = 0
    for case in cases:
        try:
            events = replay_case(case, api_base=args.api_base, token=token, timeout=args.timeout)
            ok, failures, summary = evaluate_regression(case["regression_spec"], events)
            status = "passed" if ok else "failed"
            error = "; ".join(failures) or None
        except Exception as exc:
            status, summary, error = "blocked", {}, f"{type(exc).__name__}: {exc}"
        record_regression(args.dsn, fix_id=str(case["fix_id"]), source_revision=args.revision,
                          status=status, result=summary, error=error)
        passed += status == "passed"
        print(json.dumps({"request_id": case["request_id"], "status": status, "error": error}))
    print(json.dumps({"checked": len(cases), "passed": passed, "revision": args.revision}))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
