from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark" / "audit_agent_matrix.py"
SPEC = importlib.util.spec_from_file_location("audit_agent_matrix", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _stage(stage: str, chunk_ids: list[str]) -> dict:
    return {
        "stage": stage,
        "results": [{"chunk_id": chunk_id} for chunk_id in chunk_ids],
    }


def _record(*, passed: bool, stages: dict[str, list[str]], grounded: str = "pass") -> dict:
    cells = {
        name: {"status": "pass"}
        for name in MODULE.CELL_ORDER
    }
    cells["grounded_answer"]["status"] = grounded
    return {
        "agent_evaluation": {"passed": passed, "cells": cells},
        "stage_snapshots": [_stage(stage, chunks) for stage, chunks in stages.items()],
        "trace": {"evidence_ledger": {}},
        "stop_reason": "sufficient",
    }


def _artifact(dataset: Path, record: dict) -> dict:
    sha = MODULE._sha256(dataset)
    case_id = "case-1"
    item = {
        "case_id": case_id,
        "agent_case_category": "single_hop_control",
        "expected_evidence_graph": {
            "nodes": [{"required": True, "expected_chunk_ids": ["expected"]}]
        },
        "equivalent_result_chunks": {},
        "langgraph": record,
        "llamaindex": record,
    }
    return {
        "run_id": "test-run",
        "complete": True,
        "process_exit_status": 0,
        "dataset_sha256": sha,
        "offset": 0,
        "limit": 1,
        "items": [item],
        "provenance": {
            "provenance_complete": True,
            "dataset": {"ordered_case_keys": [f"{sha}:{case_id}"]},
            "source": {"revision": "abc123", "dirty": False},
        },
    }


def test_audit_accepts_exact_coverage_and_recomputes_totals(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"case_id": "case-1"}) + "\n")
    record = _record(
        passed=True,
        stages={stage: ["expected"] for stage in MODULE.STAGES},
    )
    report = MODULE.audit_matrix(_artifact(dataset, record), dataset, 0)
    assert report["accepted_for_diagnostic_adjudication"] is True
    assert report["accepted_for_production_enablement"] is False
    assert report["backends"]["langgraph"]["matrix_passes"] == 1
    assert report["backends"]["langgraph"]["stage_retention"]["final_context"] == {
        "observed_groups": 1,
        "retained_groups": 1,
    }


def test_audit_classifies_reranker_loss(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"case_id": "case-1"}) + "\n")
    record = _record(
        passed=False,
        grounded="fail",
        stages={
            "dense": ["expected"],
            "fusion": ["expected"],
            "rerank": [],
            "final_context": [],
        },
    )
    report = MODULE.audit_matrix(_artifact(dataset, record), dataset, 0)
    failure = report["backends"]["langgraph"]["failures"][0]
    assert failure["classification"] == "reranker_loss"


def test_audit_rejects_dataset_hash_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"case_id": "case-1"}) + "\n")
    record = _record(
        passed=True,
        stages={stage: ["expected"] for stage in MODULE.STAGES},
    )
    artifact = _artifact(dataset, record)
    artifact["dataset_sha256"] = "wrong"
    report = MODULE.audit_matrix(artifact, dataset, 0)
    assert report["accepted_for_diagnostic_adjudication"] is False
    assert "artifact dataset SHA-256 does not match dataset bytes" in report["errors"]
