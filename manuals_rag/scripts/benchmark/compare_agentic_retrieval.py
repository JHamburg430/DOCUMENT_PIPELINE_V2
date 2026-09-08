#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from manuals_rag_answering.agentic_retrieval import compare_agentic_backends
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase, score_search_results
from manuals_rag_retrieval.retriever import assess_evidence_sufficiency, build_filters, retrieve


def _read_cases(path: Path, *, limit: int, offset: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            case = record.get("case") if isinstance(record.get("case"), dict) else record
            if isinstance(case, dict):
                cases.append(case)
    return cases[max(0, offset) : max(0, offset) + max(1, limit)]


def _summary(items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    backend_items = [item[backend] for item in items]
    passed = sum(1 for item in backend_items if item["evaluation"]["passed"])
    sufficient = sum(1 for item in backend_items if item["sufficient"])
    return {
        "cases": len(backend_items),
        "retrieval_passed": passed,
        "retrieval_rate": round(passed / len(backend_items), 4) if backend_items else 0.0,
        "agent_sufficient": sufficient,
        "agent_sufficiency_rate": round(sufficient / len(backend_items), 4) if backend_items else 0.0,
        "mean_elapsed_ms": round(mean(item["elapsed_ms"] for item in backend_items), 2) if backend_items else 0.0,
        "mean_completed_hops": round(
            mean(len(item["trace"].get("completed_hops", [])) for item in backend_items), 2
        )
        if backend_items
        else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_cases = _read_cases(args.dataset, limit=args.limit, offset=args.offset)
    items: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        case = RetrievalEvalCase(**raw_case)
        filters = build_filters(case.query, {})
        baseline_started = perf_counter()
        baseline_results = retrieve(case.query, args.corpus_id, filters)
        baseline_elapsed_ms = round((perf_counter() - baseline_started) * 1000, 2)
        baseline_evaluation = score_search_results(
            case,
            [result.model_dump() for result in baseline_results],
            top_k=10,
        )
        baseline_sufficiency = assess_evidence_sufficiency(case.query, baseline_results)
        comparison = compare_agentic_backends(
            case.query,
            args.corpus_id,
            filters,
            max_hops=args.max_hops,
            use_llm=not args.no_llm,
        )
        item: dict[str, Any] = {
            "case_id": case.case_id,
            "query": case.query,
            "retrieval_task": case.retrieval_task,
            "expected_document_ids": sorted(
                {
                    case.source_document_id,
                    *[
                        str(evidence.get("source_document_id") or "")
                        for evidence in case.expected_evidence or []
                        if evidence.get("source_document_id")
                    ],
                }
            ),
            "equivalent_result_chunks": comparison["equivalent_result_chunks"],
            "baseline": {
                "elapsed_ms": baseline_elapsed_ms,
                "sufficient": baseline_sufficiency.sufficient,
                "stop_reason": "single_pass",
                "result_chunk_ids": [result.chunk_id for result in baseline_results],
                "result_document_ids": sorted({result.source_document_id for result in baseline_results}),
                "trace": {"completed_hops": ["baseline"]},
                "evaluation": baseline_evaluation,
            },
        }
        for backend in ("langgraph", "llamaindex"):
            output = comparison[backend]
            evaluation = score_search_results(case, output["results"], top_k=10)
            item[backend] = {
                key: value for key, value in output.items() if key != "results"
            } | {"evaluation": evaluation}
        items.append(item)

    return {
        "dataset": str(args.dataset),
        "offset": args.offset,
        "limit": args.limit,
        "max_hops": args.max_hops,
        "planner": "heuristic" if args.no_llm else "ollama",
        "summary": {
            backend: _summary(items, backend) for backend in ("baseline", "langgraph", "llamaindex")
        },
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LangGraph and LlamaIndex bounded multi-hop retrieval.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-id", action="append", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--no-llm", action="store_true", help="Use the deterministic fallback planner/refiner.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
