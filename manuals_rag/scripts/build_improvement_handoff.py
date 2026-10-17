#!/usr/bin/env python3
"""Build a review-first improvement record from a completed eval artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manuals_rag_retrieval.improvement_loop import build_improvement_record, render_handoff


def normalize_record(item: dict) -> dict:
    """Normalize retrieval-eval JSONL rows into the improvement schema."""
    evaluation = item.get("evaluation") or {}
    answer = item.get("answer_evaluation") or {}
    failure_reasons = list(answer.get("failure_reasons") or [])
    if not evaluation.get("passed"):
        failure_reasons.append("retrieval_evaluation_failed")
    return {
        "case_id": (item.get("case") or {}).get("case_id") or item.get("case_id"),
        "trace_id": (item.get("query_debug_result") or {}).get("run_id"),
        "agent_evaluation": {
            "passed": bool(evaluation.get("passed") and answer.get("passed")),
            "cells": {
                "doc_select": {"status": "pass" if (not evaluation.get("metadata_document_selection", {}).get("attempted") or evaluation.get("metadata_document_selection", {}).get("passed")) else "fail"},
                "candidate_recall": {"status": "pass" if evaluation.get("candidate_recall") else "fail"},
                "evidence_sufficiency": {"status": "pass" if evaluation.get("evidence_passed") else "fail"},
                "grounded_answer": {"status": "pass" if answer.get("passed") else "fail"},
            },
        },
        "failure_reasons": failure_reasons,
        "trace": (item.get("query_debug_result") or {}).get("retrieval_trace") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("test_reports/improvement_loop"))
    args = parser.parse_args()
    if args.artifact.suffix == ".jsonl":
        records = [normalize_record(json.loads(line)) for line in args.artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = {}
    else:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise SystemExit("artifact must be a list or an object containing records")
    result = build_improvement_record(
        run_id=args.run_id,
        dataset=args.dataset,
        records=records,
        provenance=payload.get("provenance", {}) if isinstance(payload, dict) else {},
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.run_id}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / f"{args.run_id}.md").write_text(render_handoff(result), encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
