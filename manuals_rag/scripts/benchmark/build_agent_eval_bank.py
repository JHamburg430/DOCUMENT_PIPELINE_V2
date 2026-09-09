#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from manuals_rag_evals.agent_eval_schema import attach_expected_evidence_graph
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase


SOURCES: tuple[tuple[str, int, str], ...] = (
    ("test_reports/retrieval_eval_dataset_20260825_0153_curated_single_step_replacements_v2.jsonl", 6, "single_hop_control"),
    ("test_reports/retrieval_eval_dataset_20260824_062714.jsonl", 8, "parallel_multi_part"),
    ("test_reports/retrieval_eval_dataset_20260824_080232.jsonl", 8, "dependent_multi_hop"),
    ("test_reports/retrieval_eval_dataset_20260824_095925.jsonl", 6, "dependent_multi_hop"),
    ("test_reports/retrieval_eval_dataset_20260825_0528_curated_cross_document_v2.jsonl", 10, "cross_document"),
    ("test_reports/retrieval_eval_dataset_20260824_165722.jsonl", 5, "exact_structured_lookup"),
    ("test_reports/retrieval_eval_dataset_20260824_165909.jsonl", 3, "entity_resolution"),
)


def _read_text(repo: Path, relative_path: str, ref: str) -> str:
    path = repo / relative_path
    if path.exists():
        return path.read_text(encoding="utf-8")
    return subprocess.run(
        ["git", "show", f"{ref}:manuals_rag/{relative_path}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _read_cases(repo: Path, relative_path: str, ref: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in _read_text(repo, relative_path, ref).splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case = payload.get("case") if isinstance(payload.get("case"), dict) else payload
        cases.append(dict(case))
    return cases


def _categorized(case: dict[str, Any], category: str, source: str) -> dict[str, Any]:
    metadata = dict(case.get("source_metadata") or {})
    metadata.update({"agent_case_category": category, "agent_bank_source": source})
    case = {**case, "source_metadata": metadata}
    if category in {"dependent_multi_hop", "entity_resolution"}:
        case["retrieval_task"] = "multi_step_retrieval"
    return attach_expected_evidence_graph(case)


def _unanswerable_case() -> dict[str, Any]:
    return attach_expected_evidence_graph(
        {
            "case_id": "agent-unanswerable-invented-controller",
            "query": "What is the quantum flux calibration value for the ZX-9999 controller?",
            "source_document_id": "",
            "document_version_id": "",
            "source_chunk_id": "",
            "source_title": "No supporting manual",
            "source_filename": "",
            "chunk_type": "unknown",
            "section_path": "",
            "page_from": 0,
            "page_to": 0,
            "expected_terms": [],
            "expected_snippet": "No indexed evidence should support this invented product and setting.",
            "generation_method": "curated_unanswerable_control",
            "source_metadata": {
                "agent_case_category": "unanswerable",
                "agent_bank_source": "curated",
                "expected_tools": ["sparse"],
            },
            "benchmark_quality": "validated",
            "anchor_terms": ["zx-9999", "quantum flux"],
            "retrieval_task": "unanswerable",
            "expected_source_chunk_ids": [],
            "expected_evidence": [],
        }
    )


def build_bank(repo: Path, *, ref: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    seen_ids: set[str] = set()
    for relative_path, count, category in SOURCES:
        selected = 0
        for case in _read_cases(repo, relative_path, ref):
            query_key = " ".join(str(case.get("query") or "").lower().split()).rstrip("?")
            case_id = str(case.get("case_id") or "")
            if not query_key or not case_id or query_key in seen_queries or case_id in seen_ids:
                continue
            prepared = _categorized(case, category, relative_path)
            RetrievalEvalCase(**prepared)
            output.append(prepared)
            seen_queries.add(query_key)
            seen_ids.add(case_id)
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise RuntimeError(f"Expected {count} unique cases from {relative_path}, found {selected}.")

    dependent = _read_cases(repo, "tests/fixtures/agentic_dependent_retrieval_eval.jsonl", ref)[0]
    dependent = _categorized(dependent, "dependent_multi_hop", "tests/fixtures/agentic_dependent_retrieval_eval.jsonl")
    RetrievalEvalCase(**dependent)
    output.append(dependent)
    output.append(_unanswerable_case())
    if len(output) != 48:
        raise RuntimeError(f"Agent bank must contain 48 cases, found {len(output)}.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen agentic retrieval evaluation bank.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl"))
    args = parser.parse_args()
    cases = build_bank(args.repo.resolve(), ref=args.ref)
    output = args.output if args.output.is_absolute() else args.repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    counts = Counter((case.get("source_metadata") or {}).get("agent_case_category") for case in cases)
    print(json.dumps({"output": str(output), "cases": len(cases), "categories": counts}, indent=2))


if __name__ == "__main__":
    main()
