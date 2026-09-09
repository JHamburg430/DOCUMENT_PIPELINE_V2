#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from manuals_rag_answering.agentic_retrieval import compare_agentic_backends, insufficient_agent_answer
from manuals_rag_answering.generator import generate_answer
from manuals_rag_common.ollama import capture_ollama_usage, summarize_ollama_usage
from manuals_rag_evals.agent_eval_schema import build_expected_evidence_graph
from manuals_rag_evals.agent_eval import score_agent_run
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase, score_search_results
from manuals_rag_retrieval.retriever import assess_evidence_sufficiency, build_filters, retrieve
from manuals_rag_schemas.documents import SearchResult


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
    layer_rates: dict[str, float] = {}
    if backend != "baseline" and backend_items:
        layer_names = list((backend_items[0].get("agent_evaluation") or {}).get("cells", {}))
        layer_rates = {
            layer: round(
                sum(
                    1
                    for item in backend_items
                    if ((item.get("agent_evaluation") or {}).get("cells", {}).get(layer) or {}).get("status") == "pass"
                )
                / len(backend_items),
                4,
            )
            for layer in layer_names
        }
    recovery_attempts = sum(
        1
        for item in backend_items
        for entry in (item.get("trace", {}).get("evidence_ledger") or {}).values()
        if entry.get("recovery_for")
    )
    successful_recoveries = sum(
        1
        for item in backend_items
        for entry in (item.get("trace", {}).get("evidence_ledger") or {}).values()
        if entry.get("recovery_for") and entry.get("sufficient")
    )
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
        "agent_matrix_passed": sum(
            1 for item in backend_items if (item.get("agent_evaluation") or {}).get("passed")
        ),
        "layer_pass_rates": layer_rates,
        "recovery_attempts": recovery_attempts,
        "successful_recoveries": successful_recoveries,
        "recovery_success_rate": round(successful_recoveries / recovery_attempts, 4) if recovery_attempts else 0.0,
    }


def _category_summary(items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("agent_case_category") or "unknown")].append(item)
    return {category: _summary(category_items, backend) for category, category_items in grouped.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_cases = _read_cases(args.dataset, limit=args.limit, offset=args.offset)
    items: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        case = RetrievalEvalCase(**raw_case)
        evidence_graph = build_expected_evidence_graph(raw_case)
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
            "agent_case_category": evidence_graph.category,
            "expected_evidence_graph": evidence_graph.model_dump(),
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
            with capture_ollama_usage() as answer_usage_events:
                answer = (
                    generate_answer(
                        case.query,
                        [SearchResult.model_validate(result) for result in output["results"]],
                    )
                    if output["sufficient"]
                    else insufficient_agent_answer(case.query, output["trace"])
                ).model_dump()
            answer_usage = summarize_ollama_usage(answer_usage_events)
            retrieval_cost = dict(output["trace"].get("cost") or {})
            retrieval_by_purpose = dict(retrieval_cost.get("by_purpose") or {})
            answer_by_purpose = dict(answer_usage.get("by_purpose") or {})
            output["trace"]["cost"] = {
                **retrieval_cost,
                "retrieval_tokens": int(retrieval_cost.get("total_tokens") or 0),
                "answer_tokens": int(answer_usage.get("total_tokens") or 0),
                "answer_generation": answer_usage,
                "model_calls": int(retrieval_cost.get("model_calls") or 0)
                + int(answer_usage.get("model_calls") or 0),
                "prompt_tokens": int(retrieval_cost.get("prompt_tokens") or 0)
                + int(answer_usage.get("prompt_tokens") or 0),
                "completion_tokens": int(retrieval_cost.get("completion_tokens") or 0)
                + int(answer_usage.get("completion_tokens") or 0),
                "total_tokens": int(retrieval_cost.get("total_tokens") or 0)
                + int(answer_usage.get("total_tokens") or 0),
                "total_duration_ms": round(
                    float(retrieval_cost.get("total_duration_ms") or 0.0)
                    + float(answer_usage.get("total_duration_ms") or 0.0),
                    2,
                ),
                "by_purpose": {**retrieval_by_purpose, **answer_by_purpose},
            }
            agent_evaluation = score_agent_run(
                raw_case,
                trace=output["trace"],
                results=output["results"],
                answer=answer,
                elapsed_ms=output["elapsed_ms"],
            )
            item[backend] = {
                key: value for key, value in output.items() if key != "results"
            } | {
                "evaluation": evaluation,
                "answer": answer,
                "agent_evaluation": agent_evaluation,
            }
        items.append(item)
        if getattr(args, "progress_jsonl", False):
            print(
                json.dumps(
                    {
                        "event": "agent_case_completed",
                        "case_id": case.case_id,
                        "question_number": len(items),
                        "langgraph": item["langgraph"]["agent_evaluation"],
                        "llamaindex": item["llamaindex"]["agent_evaluation"],
                    }
                ),
                flush=True,
            )

    dataset_bytes = args.dataset.read_bytes()
    return {
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "case_count": len(raw_cases),
        "category_counts": dict(Counter(item["agent_case_category"] for item in items)),
        "offset": args.offset,
        "limit": args.limit,
        "max_hops": args.max_hops,
        "planner": "heuristic" if args.no_llm else "ollama",
        "summary": {
            backend: _summary(items, backend) for backend in ("baseline", "langgraph", "llamaindex")
        },
        "category_summary": {
            backend: _category_summary(items, backend) for backend in ("baseline", "langgraph", "llamaindex")
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
    parser.add_argument("--progress-jsonl", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
