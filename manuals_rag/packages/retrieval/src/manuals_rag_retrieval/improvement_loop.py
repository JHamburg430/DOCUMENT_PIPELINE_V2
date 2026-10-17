"""Local, review-first improvement records for agent runs.

This module deliberately does not change prompts or production configuration. It
turns an evaluation result into durable evidence that a human/Codex can review.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


FAILURE_CLASSES = (
    "document_selection",
    "retrieval_recall",
    "evidence_sufficiency",
    "grounded_answer",
    "controller_termination",
    "latency_or_cost",
    "unknown",
)


def classify_failure(record: dict[str, Any]) -> str:
    """Classify from observed failed stages, preserving uncertainty."""
    cells = (record.get("agent_evaluation") or {}).get("cells") or {}
    if cells.get("doc_select", {}).get("status") == "fail":
        return "document_selection"
    if cells.get("candidate_recall", {}).get("status") == "fail":
        return "retrieval_recall"
    if cells.get("evidence_sufficiency", {}).get("status") == "fail":
        return "evidence_sufficiency"
    if cells.get("grounded_answer", {}).get("status") == "fail":
        return "grounded_answer"
    trace = record.get("trace") or {}
    stop_reason = str(trace.get("stop_reason") or "")
    if stop_reason in {"max_hops", "timeout", "budget_exhausted"}:
        return "controller_termination"
    if cells.get("latency_token_cost", {}).get("status") == "fail":
        return "latency_or_cost"
    return "unknown"


def build_improvement_record(
    *,
    run_id: str,
    dataset: str,
    records: list[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures = []
    for record in records:
        evaluation = record.get("agent_evaluation") or {}
        if evaluation.get("passed") is True:
            continue
        failures.append({
            "case_id": record.get("case_id"),
            "failure_class": classify_failure(record),
            "failure_reasons": record.get("failure_reasons") or evaluation.get("failure_reasons") or [],
            "trace_id": record.get("trace_id"),
        })
    counts = Counter(item["failure_class"] for item in failures)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema": "rag-improvement-record.v1",
        "run_id": run_id,
        "dataset": dataset,
        "created_at": now,
        "provenance": provenance or {},
        "summary": {
            "total_cases": len(records),
            "passed_cases": len(records) - len(failures),
            "failed_cases": len(failures),
            "failure_classes": dict(sorted(counts.items())),
        },
        "failures": failures,
        "review": {"status": "pending", "approved_changes": [], "reviewer_notes": []},
        "handoff": {
            "status": "pending_review",
            "recommended_next_step": "Review failure traces, validate the classification, then add a focused regression before changing the harness.",
        },
    }


def render_handoff(record: dict[str, Any]) -> str:
    summary = record.get("summary") or {}
    lines = [
        "# RAG Agent Improvement Handoff",
        "",
        f"- Run: `{record.get('run_id')}`",
        f"- Dataset: `{record.get('dataset')}`",
        f"- Cases: {summary.get('total_cases', 0)}; passed: {summary.get('passed_cases', 0)}; failed: {summary.get('failed_cases', 0)}",
        f"- Review status: `{(record.get('review') or {}).get('status', 'pending')}`",
        "",
        "## Failure clusters",
        "",
    ]
    for name, count in (summary.get("failure_classes") or {}).items():
        lines.append(f"- `{name}`: {count}")
    lines += ["", "## Cases requiring review", ""]
    for failure in record.get("failures") or []:
        reasons = ", ".join(map(str, failure.get("failure_reasons") or [])) or "no reason recorded"
        lines.append(f"- `{failure.get('case_id')}` — `{failure.get('failure_class')}` — {reasons}")
    lines += ["", "## Safety gate", "", "Do not merge, deploy, re-ingest, or change production routing from this handoff until the failure evidence and a focused regression are reviewed.", ""]
    return "\n".join(lines)
