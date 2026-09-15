#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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


_BINDING_STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "does", "for", "in", "is",
    "not", "of", "on", "or", "that", "the", "this", "to", "was", "were",
}


def _anchor_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _BINDING_STOPWORDS
    }


def _has_bound_sibling_evidence(case: dict[str, Any]) -> bool:
    """Reject legacy table cases whose siblings came from another table/page.

    Older generated banks joined table cells by row number alone.  Manuals
    reuse row numbers on every page, so a cause from a later table could be
    attached to an unrelated symptom.  A valid sibling's serialized
    ``Row headers`` must retain at least two meaningful terms from the anchor
    cell value.  This is a static bank-integrity check, not retrieval scoring.
    """
    generation = str(case.get("generation_method") or "")
    evidence = list(case.get("expected_evidence") or [])

    # The contextual generator historically paired evidence by a broad
    # section path.  Some manuals reuse that path across several adjacent
    # capture modes, so a table from (for example) 3D Capture could be joined
    # to a Multi-Capture procedure.  When the procedure names a qualifier in
    # parentheses and the sibling is a serialized table cell, require the
    # table headers to retain at least two qualifier terms.  The cell value is
    # deliberately excluded: it may mirror the generated question while
    # still belonging to the wrong adjacent table.
    if generation.startswith("contextual_procedure_plus_section_evidence"):
        if len(evidence) < 2:
            return True
        anchor_snippet = str(evidence[0].get("snippet") or "")
        qualifiers = re.findall(r"\(([^()]*)\)", anchor_snippet)
        if not qualifiers:
            return True
        qualifier_tokens = _anchor_tokens(qualifiers[-1])
        if len(qualifier_tokens) < 2:
            return True
        for sibling in evidence[1:]:
            snippet = str(sibling.get("snippet") or "")
            if "Column headers:" not in snippet or "Row headers:" not in snippet:
                continue
            header_match = re.search(
                r"Column headers:\s*(.*?);\s*Row headers:\s*(.*?)(?:;\s*Cell value:|$)",
                snippet,
                flags=re.I | re.S,
            )
            if not header_match:
                return False
            header_tokens = _anchor_tokens(" ".join(header_match.groups()))
            if len(qualifier_tokens.intersection(header_tokens)) < 2:
                return False
        return True

    if not generation.startswith("table_sibling_"):
        return True
    if len(evidence) < 2:
        return True
    anchor_snippet = str(evidence[0].get("snippet") or "")
    anchor_match = re.search(
        r"Cell value:\s*(.*?)(?:;\s*Row:|$)", anchor_snippet, flags=re.I | re.S
    )
    anchor_tokens = _anchor_tokens(anchor_match.group(1) if anchor_match else anchor_snippet)
    if len(anchor_tokens) < 2:
        return True
    required_overlap = min(2, len(anchor_tokens))
    for sibling in evidence[1:]:
        snippet = str(sibling.get("snippet") or "")
        header_match = re.search(
            r"Row headers:\s*(.*?)(?:;\s*Cell value:|$)", snippet, flags=re.I | re.S
        )
        if not header_match:
            return False
        if len(anchor_tokens.intersection(_anchor_tokens(header_match.group(1)))) < required_overlap:
            return False
    return True


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
    generation = str(case.get("generation_method") or "")
    # These legacy fixtures call retrieval of a procedure header plus its
    # surrounding section "multi-step", but the user asks one self-contained
    # question and the second lookup does not depend on an entity discovered by
    # the first.  Preserve the contextual evidence while classifying the query
    # according to the actual planner contract.
    if generation.startswith("contextual_procedure_plus_section_evidence"):
        category = "single_hop_control"
        context = str((case.get("source_metadata") or {}).get("local_rerank_context") or "")
        query = str(case.get("query") or "")
        mode_match = re.search(r"\(([^()]*\bMode)\)", context, flags=re.I)
        if mode_match and mode_match.group(1).lower() not in query.lower():
            # Contextual legacy fixtures sometimes omitted the capture mode
            # even though the same table relationship is repeated verbatim in
            # adjacent modes. Preserve the source qualifier so the benchmark
            # does not reward cross-mode evidence.
            case = {**case, "query": f"In {mode_match.group(1)}, {query[:1].lower()}{query[1:]}"}
    metadata = dict(case.get("source_metadata") or {})
    metadata.update({"agent_case_category": category, "agent_bank_source": source})
    case = {**case, "source_metadata": metadata}
    if category in {"dependent_multi_hop", "entity_resolution"}:
        case["retrieval_task"] = "multi_step_retrieval"
    elif generation.startswith("contextual_procedure_plus_section_evidence"):
        case["retrieval_task"] = "retrieval"
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
            if not _has_bound_sibling_evidence(case):
                continue
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
