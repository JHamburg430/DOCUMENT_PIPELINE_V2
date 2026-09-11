#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from manuals_rag_common.config import settings


PIPELINE = "evidence_map_reduce_verify_v1"
REQUIRED_STAGES = [
    "map_claims",
    "retrieve_scoped_branches",
    "verify_claims",
    "reduce_confirmed_evidence",
]


def _case_checks(case: dict[str, Any], payload: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    trace = dict(payload.get("retrieval_trace") or {})
    ledger = dict(trace.get("evidence_ledger") or {})
    failures: list[str] = []
    if trace.get("pipeline") != PIPELINE:
        failures.append("wrong_pipeline")
    if trace.get("stages") != REQUIRED_STAGES:
        failures.append("wrong_or_missing_pipeline_stages")
    expected_sufficient = bool(case.get("expect_sufficient", True))
    if bool(trace.get("sufficient")) != expected_sufficient:
        failures.append("unexpected_sufficiency")
    if len(ledger) < int(case.get("min_mapped_claims", 1)):
        failures.append("insufficient_mapped_claims")

    invalid_citations: list[str] = []
    out_of_scope: list[str] = []
    confirmed_chunks: list[str] = []
    supporting_documents: set[str] = set()
    trust_states: dict[str, str] = {}
    for hop_id, entry in ledger.items():
        assessment = dict(entry.get("assessment") or {})
        verification = dict(assessment.get("verification") or {})
        trust_state = str(verification.get("trust_state") or assessment.get("trust_state") or "unresolved")
        trust_states[hop_id] = trust_state
        invalid_citations.extend(str(item) for item in verification.get("invalid_citation_ids") or [])
        out_of_scope.extend(str(item) for item in verification.get("out_of_scope_chunk_ids") or [])
        if trust_state == "confirmed":
            confirmed_chunks.extend(str(item) for item in verification.get("supporting_chunk_ids") or [])
            supporting_documents.update(str(item) for item in assessment.get("supporting_document_ids") or [])
    if invalid_citations:
        failures.append("verifier_emitted_invalid_citations")
    if out_of_scope:
        failures.append("verifier_emitted_out_of_scope_citations")

    expected_documents = {str(item) for item in case.get("expected_document_ids") or []}
    if expected_sufficient and expected_documents and not expected_documents.issubset(supporting_documents):
        failures.append("expected_document_not_confirmed")
    if expected_sufficient and not confirmed_chunks:
        failures.append("no_confirmed_evidence")
    context = dict(trace.get("context_assembly") or {})
    if expected_sufficient and not context.get("all_required_claims_retained"):
        failures.append("confirmed_evidence_not_retained")
    answer = str(payload.get("answer") or "")
    missing_answer_terms = [
        str(term)
        for term in case.get("expected_answer_terms") or []
        if str(term).casefold() not in answer.casefold()
    ]
    if expected_sufficient and missing_answer_terms:
        failures.append("expected_answer_terms_missing")
    if not expected_sufficient and payload.get("insufficient_evidence") is not True:
        failures.append("expected_abstention_not_reported")

    return failures, {
        "pipeline": trace.get("pipeline"),
        "stages": trace.get("stages") or [],
        "sufficient": bool(trace.get("sufficient")),
        "stop_reason": trace.get("stop_reason"),
        "mapped_claim_count": len(ledger),
        "trust_states": trust_states,
        "confirmed_chunk_ids": list(dict.fromkeys(confirmed_chunks)),
        "supporting_document_ids": sorted(supporting_documents),
        "invalid_citation_ids": list(dict.fromkeys(invalid_citations)),
        "out_of_scope_chunk_ids": list(dict.fromkeys(out_of_scope)),
        "context_assembly": context,
        "duration_ms": trace.get("duration_ms"),
        "cost": trace.get("cost") or {},
        "missing_answer_terms": missing_answer_terms,
    }


def run(*, api_base: str, cases: list[dict[str, Any]], backends: list[str], timeout: float) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {settings.local_end_user_token}"}
    with httpx.Client(base_url=api_base, timeout=timeout) as client:
        for case in cases:
            for backend in backends:
                started = datetime.now(UTC)
                try:
                    response = client.post(
                        "/query",
                        headers=headers,
                        json={
                            "query": case["query"],
                            "corpus_ids": case["corpus_ids"],
                            "filters": case.get("filters", {}),
                            "response_mode": "answer_with_citations",
                            "retrieval_orchestrator": backend,
                            "max_retrieval_hops": int(case.get("max_retrieval_hops", 4)),
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    failures, trace_summary = _case_checks(case, payload)
                    results.append(
                        {
                            "case_id": case.get("id") or case["query"],
                            "backend": backend,
                            "query": case["query"],
                            "passed": not failures,
                            "failures": failures,
                            "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                            "answer": payload.get("answer"),
                            "citations": payload.get("citations") or [],
                            "trace": trace_summary,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "case_id": case.get("id") or case["query"],
                            "backend": backend,
                            "query": case["query"],
                            "passed": False,
                            "failures": ["request_failed"],
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                        }
                    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline": PIPELINE,
        "api_base": api_base,
        "case_count": len(cases),
        "backend_count": len(backends),
        "result_count": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate live evidence-first agentic retrieval.")
    parser.add_argument("cases", type=Path, help="JSON array of retrieval validation cases.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8600")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["langgraph_agent", "llamaindex_agent"],
        default=["langgraph_agent"],
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("cases must contain a non-empty JSON array")
    report = run(api_base=args.api_base, cases=cases, backends=args.backends, timeout=args.timeout)
    rendered = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
