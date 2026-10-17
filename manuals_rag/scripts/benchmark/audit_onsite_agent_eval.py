#!/usr/bin/env python3
"""Reconcile a frozen onsite Agent matrix from individual backend cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result: dict, dataset: Path, manifest: Path) -> dict:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    manifest_data = json.loads(manifest.read_text())
    expected_case_ids = [str(row["case_id"]) for row in rows]
    expected_backends = ["langgraph_agent", "llamaindex_agent"]
    expected_keys = [f"{case_id}:{backend}" for case_id in expected_case_ids for backend in expected_backends]

    cells: dict[str, dict] = {}
    duplicate_keys: list[str] = []
    unexpected_keys: list[str] = []
    for record in result.get("records") or []:
        case_id = str(record.get("case_id") or "")
        for backend, cell in (record.get("backends") or {}).items():
            key = f"{case_id}:{backend}"
            if key in cells:
                duplicate_keys.append(key)
            cells[key] = cell
            if key not in expected_keys:
                unexpected_keys.append(key)

    missing_keys = [key for key in expected_keys if key not in cells]
    counts = {
        status: sum(cell.get("status") == status for cell in cells.values())
        for status in ("pass", "fail", "error")
    }
    gate_abstentions = sorted(
        key for key, cell in cells.items()
        if cell.get("insufficient_evidence") is True
        or (cell.get("evidence_gate") or {}).get("status") in {"rejected", "abstained"}
    )
    unsafe_accepted = sorted(
        key for key, cell in cells.items()
        if (cell.get("evidence_gate") or {}).get("status") == "accepted"
        and cell.get("status") != "pass"
    )
    unproven_false_negative_candidates = sorted(
        key for key, cell in cells.items()
        if cell.get("status") == "fail"
        and cell.get("insufficient_evidence") is True
        and (cell.get("retrieval_trace") or {}).get("sufficient") is True
    )

    artifact_summary = result.get("summary") or {}
    recomputed_summary = {
        "passed": counts["pass"],
        "failed": counts["fail"],
        "errors": counts["error"],
        "total": len(expected_keys),
    }
    errors: list[str] = []
    dataset_sha = _sha256(dataset)
    if manifest_data.get("sha256") != dataset_sha:
        errors.append("manifest SHA-256 does not match dataset bytes")
    if result.get("dataset_sha256") != dataset_sha:
        errors.append("artifact dataset SHA-256 does not match dataset bytes")
    if result.get("selected_case_ids") != expected_case_ids:
        errors.append("selected case IDs do not match the frozen dataset order")
    if result.get("backends") != expected_backends:
        errors.append("artifact backend set or order is wrong")
    if missing_keys:
        errors.append("matrix coverage is incomplete")
    if duplicate_keys:
        errors.append("matrix contains duplicate backend cells")
    if unexpected_keys:
        errors.append("matrix contains unexpected backend cells")
    if artifact_summary != recomputed_summary:
        errors.append("artifact summary does not reconcile with individual cells")
    if result.get("status") != "completed" or result.get("exit_code") != 0:
        errors.append("writer did not produce a successful terminal result")
    provenance = result.get("provenance") or {}
    source = provenance.get("source") or {}
    if not source.get("revision") or source.get("dirty") is not False:
        errors.append("matrix was not run from an identified clean revision")
    gate_missing = sorted(
        key for key, cell in cells.items()
        if cell.get("status") == "pass"
        and (cell.get("evidence_gate") or {}).get("status") != "accepted"
    )
    if gate_missing:
        errors.append("passing cells are missing accepted evidence-gate verdicts")
    if counts["fail"] or counts["error"]:
        errors.append("one or more required cells did not pass")
    if unsafe_accepted:
        errors.append("evidence gate accepted one or more cells that failed the source contract")

    return {
        "schema": "manuals-rag-onsite-agent-eval-audit-v1",
        "run_id": result.get("run_id"),
        "dataset_sha256": dataset_sha,
        "expected_case_ids": expected_case_ids,
        "expected_backends": expected_backends,
        "coverage": {
            "expected": len(expected_keys),
            "observed": len(cells),
            "missing_keys": missing_keys,
            "duplicate_keys": sorted(set(duplicate_keys)),
            "unexpected_keys": sorted(set(unexpected_keys)),
        },
        "recomputed_summary": recomputed_summary,
        "gate_abstention_count": len(gate_abstentions),
        "gate_abstention_keys": gate_abstentions,
        "unsafe_accepted_count": len(unsafe_accepted),
        "unsafe_accepted_keys": unsafe_accepted,
        "false_negative_count": 0,
        "false_negative_candidates_unproven": unproven_false_negative_candidates,
        "classification_note": (
            "A false negative is not proven from a post-gate artifact alone; candidates require "
            "the persisted pre-gate answer and answer-bearing final evidence."
        ),
        "errors": errors,
        "accepted_for_production_enablement": not errors,
        "rollout_recommendation": "GO" if not errors else "NO-GO",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.result.read_text()), args.dataset, args.manifest)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"event": "onsite_audit_completed", "accepted": report["accepted_for_production_enablement"]}))
    return 0 if report["accepted_for_production_enablement"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
