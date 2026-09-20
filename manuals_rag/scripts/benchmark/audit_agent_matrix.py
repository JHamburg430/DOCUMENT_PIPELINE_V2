#!/usr/bin/env python3
"""Independently reconcile and classify an agentic-retrieval matrix artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


BACKENDS = ("langgraph", "llamaindex")
STAGES = ("dense", "fusion", "rerank", "final_context")
CELL_ORDER = (
    "tool_selection",
    "candidate_recall",
    "document_retention",
    "hop_dependencies",
    "evidence_sufficiency",
    "grounded_answer",
    "latency_token_cost",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_key(dataset_sha256: str, case_id: str) -> str:
    return f"{dataset_sha256}:{case_id}"


def _dataset_case_ids(path: Path) -> list[str]:
    case_ids: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        case_id = str(payload.get("case_id") or "")
        if not case_id:
            raise ValueError(f"dataset line {line_number} has no case_id")
        case_ids.append(case_id)
    return case_ids


def _cell_status(record: dict[str, Any], name: str) -> str:
    return str(
        ((record.get("agent_evaluation") or {}).get("cells") or {})
        .get(name, {})
        .get("status")
        or "missing"
    )


def _expected_chunk_groups(item: dict[str, Any]) -> list[set[str]]:
    groups: list[set[str]] = []
    graph = item.get("expected_evidence_graph") or {}
    for node in graph.get("nodes") or []:
        if node.get("required", True) is False:
            continue
        chunk_ids = {str(value) for value in node.get("expected_chunk_ids") or [] if value}
        if chunk_ids:
            groups.append(chunk_ids)
    if groups:
        return groups
    chunks = {
        str(value)
        for value in (item.get("expected_source_chunk_ids") or [])
        if value
    }
    return [chunks] if chunks else []


def _equivalent_chunks(record: dict[str, Any], chunk_id: str) -> set[str]:
    equivalents = (
        (((record.get("agent_evaluation") or {}).get("cells") or {})
        .get("candidate_recall", {})
        .get("metrics", {})
        .get("equivalent_chunks", {}))
    )
    values = equivalents.get(chunk_id) if isinstance(equivalents, dict) else None
    return {chunk_id, *(str(value) for value in values or [] if value)}


def _stage_chunks(record: dict[str, Any]) -> tuple[dict[str, set[str]], set[str]]:
    chunks = {stage: set() for stage in STAGES}
    observed: set[str] = set()
    for snapshot in record.get("stage_snapshots") or []:
        raw_stage = str(snapshot.get("stage") or "")
        stage = raw_stage.removeprefix("corrective_")
        if stage not in chunks:
            continue
        observed.add(stage)
        for result in snapshot.get("results") or []:
            chunk_id = str(result.get("chunk_id") or "")
            if chunk_id:
                chunks[stage].add(chunk_id)
    return chunks, observed


def _retention(item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    stage_chunks, observed = _stage_chunks(record)
    expected_groups = _expected_chunk_groups(item)
    retained: dict[str, int] = {}
    missing: dict[str, list[list[str]]] = {}
    for stage in STAGES:
        stage_missing: list[list[str]] = []
        for group in expected_groups:
            alternatives: set[str] = set()
            for chunk_id in group:
                alternatives.update(_equivalent_chunks(record, chunk_id))
            if not alternatives.intersection(stage_chunks[stage]):
                stage_missing.append(sorted(alternatives))
        retained[stage] = len(expected_groups) - len(stage_missing)
        missing[stage] = stage_missing
    return {
        "required_evidence_groups": len(expected_groups),
        "observed_stages": sorted(observed),
        "retained": retained,
        "missing": missing,
    }


def _judge_counts(record: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    ledger = (record.get("trace") or {}).get("evidence_ledger") or {}
    for hop in ledger.values():
        judge = (
            (((hop or {}).get("assessment") or {}).get("verification") or {})
            .get("judge")
            or {}
        )
        mode = str(judge.get("mode") or "missing")
        status = str(judge.get("status") or "missing")
        counts[f"{mode}:{status}"] += 1
        if mode == "llm" and status == "checked":
            attempts = judge.get("attempts") or []
            has_raw = any("raw_response" in attempt for attempt in attempts)
            has_parsed = any("parsed_response" in attempt for attempt in attempts)
            has_normalized = isinstance(judge.get("normalized_verdict"), dict)
            if not (has_raw and has_parsed and has_normalized):
                counts["llm:checked_incomplete_contract"] += 1
    return counts


def _classify_failure(
    item: dict[str, Any], record: dict[str, Any], retention: dict[str, Any]
) -> str:
    if (record.get("agent_evaluation") or {}).get("passed") is True:
        return "pass"
    judge_counts = _judge_counts(record)
    if judge_counts.get("llm:unchecked", 0):
        return "verifier_contract_failure"
    required = retention["required_evidence_groups"]
    retained = retention["retained"]
    observed = set(retention["observed_stages"])
    if required:
        if "dense" in observed and retained["dense"] < required:
            return "dense_retrieval_miss"
        if {"dense", "fusion"}.issubset(observed) and retained["fusion"] < retained["dense"]:
            return "fusion_loss"
        if {"fusion", "rerank"}.issubset(observed) and retained["rerank"] < retained["fusion"]:
            return "reranker_loss"
        if {"rerank", "final_context"}.issubset(observed) and retained["final_context"] < retained["rerank"]:
            return "final_context_loss"
    for cell, label in (
        ("tool_selection", "planning_or_tool_selection"),
        ("candidate_recall", "candidate_anchor_or_scoring"),
        ("document_retention", "document_retention"),
        ("hop_dependencies", "dependency_execution"),
        ("evidence_sufficiency", "evidence_verification"),
        ("grounded_answer", "answer_or_answer_scoring"),
        ("latency_token_cost", "performance_gate"),
    ):
        if _cell_status(record, cell) != "pass":
            return label
    return "unclassified"


def audit_matrix(
    artifact: dict[str, Any], dataset_path: Path, exit_status: int | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    dataset_sha256 = _sha256(dataset_path)
    dataset_case_ids = _dataset_case_ids(dataset_path)
    offset = int(artifact.get("offset") or 0)
    limit = int(artifact.get("limit") or len(dataset_case_ids))
    expected_ids = dataset_case_ids[offset : offset + limit]
    expected_keys = [_case_key(dataset_sha256, case_id) for case_id in expected_ids]
    provenance = artifact.get("provenance") or {}
    launch_dataset = provenance.get("dataset") or {}
    actual_items = artifact.get("items") or []
    actual_ids = [str(item.get("case_id") or "") for item in actual_items]

    if artifact.get("complete") is not True:
        errors.append("artifact complete flag is not true")
    if int(artifact.get("process_exit_status", -1)) != 0:
        errors.append("artifact process_exit_status is not zero")
    if exit_status is not None and exit_status != 0:
        errors.append(f"external exit status is {exit_status}, not zero")
    if str(artifact.get("dataset_sha256") or "") != dataset_sha256:
        errors.append("artifact dataset SHA-256 does not match dataset bytes")
    if launch_dataset.get("ordered_case_keys") != expected_keys:
        errors.append("persisted ordered case keys do not match the selected dataset slice")
    if actual_ids != expected_ids:
        errors.append("artifact case IDs do not match the selected dataset slice in order")
    if len(set(actual_ids)) != len(actual_ids):
        errors.append("artifact contains duplicate case IDs")
    if provenance.get("provenance_complete") is not True:
        errors.append("artifact provenance_complete is not true")

    backend_reports: dict[str, Any] = {}
    for backend in BACKENDS:
        failures: list[dict[str, Any]] = []
        classifications: Counter[str] = Counter()
        judges: Counter[str] = Counter()
        stage_retention = {
            stage: {"observed_groups": 0, "retained_groups": 0}
            for stage in STAGES
        }
        required_groups = 0
        cell_passes = Counter()
        matrix_passes = 0
        for item in actual_items:
            record = item.get(backend)
            if not isinstance(record, dict):
                errors.append(f"{item.get('case_id')}: missing {backend} record")
                continue
            retention = _retention(item, record)
            required_groups += retention["required_evidence_groups"]
            for stage in STAGES:
                if stage in retention["observed_stages"]:
                    stage_retention[stage]["observed_groups"] += retention[
                        "required_evidence_groups"
                    ]
                    stage_retention[stage]["retained_groups"] += retention["retained"][stage]
            judges.update(_judge_counts(record))
            for cell in CELL_ORDER:
                if _cell_status(record, cell) == "pass":
                    cell_passes[cell] += 1
            passed = (record.get("agent_evaluation") or {}).get("passed") is True
            matrix_passes += int(passed)
            classification = _classify_failure(item, record, retention)
            classifications[classification] += 1
            if not passed:
                failures.append(
                    {
                        "case_id": item.get("case_id"),
                        "category": item.get("agent_case_category"),
                        "classification": classification,
                        "stop_reason": record.get("stop_reason"),
                        "failed_cells": [
                            cell for cell in CELL_ORDER if _cell_status(record, cell) != "pass"
                        ],
                        "retention": retention,
                    }
                )
        if judges.get("llm:checked_incomplete_contract", 0):
            errors.append(f"{backend}: checked LLM judge record lacks raw/parsed/normalized data")
        backend_reports[backend] = {
            "cases": len(actual_items),
            "matrix_passes": matrix_passes,
            "matrix_failures": len(actual_items) - matrix_passes,
            "cell_passes": dict(cell_passes),
            "required_evidence_groups": required_groups,
            "stage_retention": stage_retention,
            "judge_counts": dict(judges),
            "failure_classifications": dict(classifications),
            "failures": failures,
        }

    return {
        "accepted_for_diagnostic_adjudication": not errors,
        "accepted_for_production_enablement": False,
        "production_blockers": [
            "source-backed adjudication of failed answers and citations is incomplete",
            "a separate frozen held-out bank has not passed",
            "human visual-PDF adjudication is incomplete",
        ],
        "errors": errors,
        "artifact": {
            "run_id": artifact.get("run_id"),
            "dataset_sha256": dataset_sha256,
            "source": provenance.get("source"),
            "case_count": len(actual_items),
            "offset": offset,
            "limit": limit,
        },
        "backends": backend_reports,
    }


def _markdown(report: dict[str, Any]) -> str:
    artifact = report["artifact"]
    lines = [
        f"# Independent matrix audit: `{artifact['run_id']}`",
        "",
        "## Decision",
        "",
        (
            "The artifact is structurally accepted for diagnostic adjudication."
            if report["accepted_for_diagnostic_adjudication"]
            else "The artifact is rejected because structural reconciliation failed."
        ),
        "It is **not** accepted for production enablement.",
        "",
        "## Reconciliation",
        "",
        f"- Cases: `{artifact['case_count']}`",
        f"- Dataset SHA-256: `{artifact['dataset_sha256']}`",
        f"- Source: `{json.dumps(artifact['source'], sort_keys=True)}`",
    ]
    if report["errors"]:
        lines.extend(["- Errors:", *[f"  - {error}" for error in report["errors"]]])
    for backend, payload in report["backends"].items():
        lines.extend(
            [
                "",
                f"## {backend}",
                "",
                f"- Full matrix passes: `{payload['matrix_passes']}/{payload['cases']}`",
                f"- Required evidence groups: `{payload['required_evidence_groups']}`",
                f"- Stage retention: `{json.dumps(payload['stage_retention'], sort_keys=True)}`",
                f"- Judge states: `{json.dumps(payload['judge_counts'], sort_keys=True)}`",
                f"- Failure classes: `{json.dumps(payload['failure_classifications'], sort_keys=True)}`",
                "",
                "### Failed cases",
                "",
            ]
        )
        for failure in payload["failures"]:
            lines.append(
                f"- `{failure['case_id']}` — `{failure['classification']}`; "
                f"cells={','.join(failure['failed_cells'])}; stop=`{failure['stop_reason']}`"
            )
    lines.extend(["", "## Open production blockers", ""])
    lines.extend(f"- {blocker}" for blocker in report["production_blockers"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exit-file", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    exit_status = int(args.exit_file.read_text().strip()) if args.exit_file else None
    report = audit_matrix(artifact, args.dataset, exit_status)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(_markdown(report))
    return 0 if report["accepted_for_diagnostic_adjudication"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
