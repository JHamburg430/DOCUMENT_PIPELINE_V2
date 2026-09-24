#!/usr/bin/env python3
"""Fail-closed production acceptance gate for a frozen Manuals RAG matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BACKENDS = ("langgraph", "llamaindex")
ARTIFACT_SCHEMA_VERSION = "agentic-retrieval-matrix-v2"
REQUIRED_CELLS = (
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} is not a JSON object")
        records.append(record)
    return records


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _adjudication_status(case: dict[str, Any]) -> str:
    value = case.get("adjudication")
    if isinstance(value, dict):
        return str(value.get("status") or "")
    return str(value or "")


def _expected_outcome(case: dict[str, Any]) -> str:
    return str((case.get("expected_evidence_graph") or {}).get("expected_outcome") or "answerable")


def _referenced_document_ids(value: Any, key: str = "") -> set[str]:
    """Collect document references from the real dataset's nested evidence fields."""
    ids: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            ids.update(_referenced_document_ids(child, str(child_key)))
    elif isinstance(value, list):
        if key in {"source_document_ids", "expected_document_ids"}:
            ids.update(str(item) for item in value if item)
        else:
            for child in value:
                ids.update(_referenced_document_ids(child, key))
    elif value and key in {"source_document_id", "expected_source_document_id"}:
        ids.add(str(value))
    return ids


def _cell(record: dict[str, Any], name: str) -> dict[str, Any]:
    return dict((((record.get("agent_evaluation") or {}).get("cells") or {}).get(name) or {}))


def evaluate(
    artifact: dict[str, Any],
    dataset: list[dict[str, Any]],
    *,
    dataset_sha256: str,
    tuning_document_ids: set[str] | None,
    min_cases: int,
    max_p95_latency_ms: float,
    expected_source_revision: str | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    provenance = artifact.get("provenance") or {}
    source = provenance.get("source") or {}
    items = artifact.get("items") or []
    by_case = {str(case.get("case_id")): case for case in dataset}
    item_ids = [str(item.get("case_id")) for item in items]
    dataset_ids = [str(case.get("case_id")) for case in dataset]

    if artifact.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        blockers.append(f"matrix artifact_schema_version is not {ARTIFACT_SCHEMA_VERSION}")
    run_id = str(artifact.get("run_id") or "")
    if not run_id or str(provenance.get("run_id") or "") != run_id:
        blockers.append("matrix run_id is missing or does not match provenance")

    if artifact.get("complete") is not True or _integer(artifact.get("process_exit_status")) != 0:
        blockers.append("matrix artifact is incomplete or has a nonzero exit status")
    if str(artifact.get("dataset_sha256") or "") != dataset_sha256:
        blockers.append("matrix dataset hash does not match the supplied dataset bytes")
    if item_ids != dataset_ids:
        blockers.append("matrix cases do not exactly match the frozen dataset order")
    if any(not case_id or case_id == "None" for case_id in dataset_ids):
        blockers.append("frozen dataset contains an empty case_id")
    if len(set(dataset_ids)) != len(dataset_ids):
        blockers.append("frozen dataset contains duplicate case_id values")
    expected_keys = [f"{dataset_sha256}:{case_id}" for case_id in dataset_ids]
    launch_dataset = provenance.get("dataset") or {}
    if launch_dataset.get("ordered_case_keys") != expected_keys:
        blockers.append("persisted ordered case keys do not match the frozen dataset")
    if str(launch_dataset.get("sha256") or "") != dataset_sha256:
        blockers.append("launch provenance dataset hash does not match the supplied dataset bytes")
    if _integer(artifact.get("case_count")) != len(dataset):
        blockers.append("matrix case_count does not match the frozen dataset")
    if provenance.get("provenance_complete") is not True or source.get("dirty") is not False:
        blockers.append("source provenance is incomplete or dirty")
    source_revision = str(source.get("revision") or "")
    if not source_revision:
        blockers.append("source revision is missing")
    if expected_source_revision and source_revision != expected_source_revision:
        blockers.append("source revision does not match the expected release revision")
    configuration = provenance.get("configuration")
    required_configuration = {
        "corpus_ids", "backends", "max_hops", "planner", "ollama_url",
        "ollama_embed_url", "ollama_embed_model", "ollama_fast_model",
        "ollama_retrieval_verifier_model", "ollama_answer_model", "qdrant_url",
        "rerank_model", "rerank_device", "result_limit",
        "retrieval_branch_max_workers", "retrieval_qdrant_max_concurrency",
        "case_concurrency",
    }
    if not isinstance(configuration, dict) or any(
        key not in configuration or configuration[key] in (None, "", [])
        for key in required_configuration
    ):
        blockers.append("launch provenance configuration is missing required runtime fields")
    if len(dataset) < min_cases:
        blockers.append(f"held-out dataset has {len(dataset)} cases; policy requires at least {min_cases}")

    non_held_out = [str(case.get("case_id")) for case in dataset if case.get("evaluation_split") != "held_out"]
    if non_held_out:
        blockers.append(f"{len(non_held_out)} cases lack evaluation_split=held_out")
    unadjudicated = [
        str(case.get("case_id"))
        for case in dataset
        if _adjudication_status(case) not in {"human_verified", "source_verified"}
    ]
    if unadjudicated:
        blockers.append(f"{len(unadjudicated)} cases lack source/human-verified adjudication")
    if tuning_document_ids is None:
        blockers.append("no tuning dataset was supplied for document-disjointness verification")
        overlap: list[str] = []
    else:
        held_out_documents = set().union(*(_referenced_document_ids(case) for case in dataset))
        overlap = sorted(held_out_documents & tuning_document_ids)
        if overlap:
            blockers.append(f"held-out/tuning document overlap contains {len(overlap)} documents")

    backend_reports: dict[str, Any] = {}
    for backend in BACKENDS:
        failed_cases: list[str] = []
        category_counts: Counter[str] = Counter()
        category_passes: Counter[str] = Counter()
        citation_failures: list[str] = []
        abstention_failures: list[str] = []
        latencies: list[float] = []
        for item in items:
            case_id = str(item.get("case_id"))
            case = by_case.get(case_id, item)
            category = str(item.get("agent_case_category") or "uncategorized")
            category_counts[category] += 1
            record = item.get(backend) or {}
            cells = [_cell(record, name) for name in REQUIRED_CELLS]
            passed = bool(cells) and all(cell.get("status") == "pass" for cell in cells)
            if passed:
                category_passes[category] += 1
            else:
                failed_cases.append(case_id)

            grounded = _cell(record, "grounded_answer").get("metrics") or {}
            claim_grounding = grounded.get("claim_grounding") or {}
            citations_ok = (
                not grounded.get("invalid_citation_chunks")
                and all(
                    bool(state.get("terms")) and bool(state.get("citation"))
                    for state in claim_grounding.values()
                    if isinstance(state, dict)
                )
            )
            if _expected_outcome(case) == "answerable" and (not claim_grounding or not citations_ok):
                citation_failures.append(case_id)
            if _expected_outcome(case) != "answerable":
                if not bool(grounded.get("insufficient_evidence")):
                    abstention_failures.append(case_id)
            elapsed = record.get("elapsed_ms")
            if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
                latencies.append(float(elapsed))
            else:
                blockers.append(f"{backend}: {case_id} has missing or malformed elapsed_ms")

        missing_categories = sorted(category for category, count in category_counts.items() if count < 1)
        failed_categories = sorted(
            category for category, count in category_counts.items() if category_passes[category] != count
        )
        p95 = _percentile(latencies, 0.95)
        if failed_cases:
            blockers.append(f"{backend}: {len(failed_cases)} cases fail one or more required cells")
        if failed_categories:
            blockers.append(f"{backend}: category gates fail for {', '.join(failed_categories)}")
        if citation_failures:
            blockers.append(f"{backend}: {len(citation_failures)} answerable cases fail claim/citation correctness")
        if abstention_failures:
            blockers.append(f"{backend}: {len(abstention_failures)} unanswerable/conflicting cases fail abstention")
        if p95 is None or p95 > max_p95_latency_ms:
            blockers.append(f"{backend}: p95 latency {p95} exceeds {max_p95_latency_ms} ms")
        backend_reports[backend] = {
            "case_count": len(items),
            "passed_case_count": len(items) - len(failed_cases),
            "failed_case_ids": failed_cases,
            "category_counts": dict(category_counts),
            "category_passes": dict(category_passes),
            "failed_categories": failed_categories,
            "missing_categories": missing_categories,
            "citation_failure_case_ids": citation_failures,
            "abstention_failure_case_ids": abstention_failures,
            "p95_latency_ms": p95,
            "max_p95_latency_ms": max_p95_latency_ms,
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "gate": "manuals_rag_production_acceptance_v1",
        "accepted": not blockers,
        "blockers": blockers,
        "dataset": {
            "sha256": dataset_sha256,
            "case_count": len(dataset),
            "minimum_case_count": min_cases,
            "non_held_out_case_count": len(non_held_out),
            "unadjudicated_case_count": len(unadjudicated),
            "tuning_document_overlap": overlap,
        },
        "source": source,
        "backends": backend_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tuning-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cases", type=int, default=200)
    parser.add_argument("--max-p95-latency-ms", type=float, default=120000)
    parser.add_argument("--expected-source-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite gate artifact: {args.output}")
    try:
        dataset = _load_jsonl(args.dataset)
        tuning_document_ids = None
        if args.tuning_dataset:
            tuning_document_ids = set().union(
                *(_referenced_document_ids(case) for case in _load_jsonl(args.tuning_dataset))
            )
        report = evaluate(
            json.loads(args.artifact.read_text(encoding="utf-8")),
            dataset,
            dataset_sha256=_sha256(args.dataset),
            tuning_document_ids=tuning_document_ids,
            min_cases=args.min_cases,
            max_p95_latency_ms=args.max_p95_latency_ms,
            expected_source_revision=args.expected_source_revision,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "gate": "manuals_rag_production_acceptance_v1",
            "accepted": False,
            "blockers": [f"unverifiable input: {type(exc).__name__}: {exc}"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": report["accepted"], "blockers": report["blockers"]}, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
