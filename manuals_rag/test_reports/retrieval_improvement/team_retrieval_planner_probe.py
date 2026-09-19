#!/usr/bin/env python3
"""Bounded case-6 retrieval/planner comparison. Run under an external timeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from api.debug import execute_query_debug_run
from manuals_rag_answering.agentic_retrieval import (
    RetrievalHop,
    _assess_hop_evidence,
    build_langgraph_agentic_retriever,
    build_llamaindex_agentic_retriever,
    insufficient_agent_answer,
    verify_retrieval_claim,
)
from manuals_rag_answering.generator import generate_answer_with_trace
from manuals_rag_common.config import settings
from manuals_rag_common.ollama import capture_ollama_usage, summarize_ollama_usage
from manuals_rag_retrieval.retriever import build_filters, retrieve
from manuals_rag_schemas.documents import QueryRequest, SearchResult


ROOT = Path("test_reports/retrieval_improvement")
OUTPUT_PATH = ROOT / "team_retrieval_planner_probe.json"
CORPUS_IDS = ["manuals_vendor_keyence"]
COMBINED_QUERY = (
    "For CV-X482 and LJ-X8000, compare what the Condition list and "
    "Standard Angle settings control."
)
SCOPED_QUERIES = [
    (
        "For CV-X482, what does the Condition list setting control?",
        "7361e08a-84bb-5f53-954c-819d2c179a3e",
    ),
    (
        "For LJ-X8000, what does the Standard Angle setting control?",
        "125b3908-de90-53b8-809e-b367bcc83548",
    ),
]
DEPENDENCY_QUERY = (
    "Which encoder head model is compatible with the CA-EN100U, and how is that "
    "encoder head powered?"
)
DEPENDENCY_EXPECTED_CHUNKS = [
    "5441e6e3-1a1c-5f15-b8b9-aa0bee45b4c3",
    "e33ab3cc-a1ff-5f79-b549-1588e1c6398d",
]


def _initial_report() -> dict[str, Any]:
    return {
        "case_id": "curated_cross_document_v2::6",
        "query": COMBINED_QUERY,
        "corpus_ids": CORPUS_IDS,
        "audit_scope": "assistant audit of extracted source text, not PDF/human/held-out",
        "runtime": {
            "ollama_url": settings.ollama_url,
            "answer_model": settings.ollama_answer_model,
            "verifier_model": settings.ollama_retrieval_verifier_model,
            "fast_model": settings.ollama_fast_model,
            "max_hops": 4,
        },
        "scoped_retrieval": [],
        "planner": {},
        "dependency": {},
        "completed_stages": [],
        "completed": False,
    }


def _load_report() -> dict[str, Any]:
    if not OUTPUT_PATH.exists():
        return _initial_report()
    report = json.loads(OUTPUT_PATH.read_text())
    if report.get("case_id") != "curated_cross_document_v2::6":
        raise ValueError("Existing probe report belongs to a different case")
    return report


def _save(report: dict[str, Any]) -> None:
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    return list({result.chunk_id: result for result in results}.values())


def run_scoped(report: dict[str, Any]) -> None:
    report["scoped_retrieval"] = []
    branch_results: list[list[SearchResult]] = []
    for index, (query, expected_chunk_id) in enumerate(SCOPED_QUERIES, start=1):
        print(json.dumps({"stage": "scoped_retrieval", "index": index, "query": query}), flush=True)
        started = perf_counter()
        filters = build_filters(query, {})
        results = retrieve(query, CORPUS_IDS, filters, limit=10)
        sufficient, assessment = _assess_hop_evidence(query, results)
        hop = RetrievalHop(hop_id=f"scoped_{index}", objective=query, query=query)
        verdict = verify_retrieval_claim(hop, query, results, assessment)
        answer, answer_trace = generate_answer_with_trace(query, results)
        result_ids = [result.chunk_id for result in results]
        report["scoped_retrieval"].append(
            {
                "query": query,
                "filters": filters,
                "expected_chunk_id": expected_chunk_id,
                "expected_rank": result_ids.index(expected_chunk_id) + 1 if expected_chunk_id in result_ids else None,
                "result_chunk_ids": result_ids,
                "result_document_ids": sorted({result.source_document_id for result in results}),
                "results": [result.model_dump() for result in results],
                "claim_sufficient": sufficient,
                "assessment": assessment,
                "verdict": verdict,
                "answer": answer.model_dump(),
                "answer_trace": answer_trace,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            }
        )
        branch_results.append(results)
        _save(report)

    interleaved: list[SearchResult] = []
    for rank in range(max(len(results) for results in branch_results)):
        interleaved.extend(results[rank] for results in branch_results if rank < len(results))
    combined_results = _dedupe_results(interleaved)
    print(json.dumps({"stage": "scoped_synthesis", "result_count": len(combined_results)}), flush=True)
    started = perf_counter()
    answer, answer_trace = generate_answer_with_trace(COMBINED_QUERY, combined_results)
    report["scoped_synthesis"] = {
        "result_chunk_ids": [result.chunk_id for result in combined_results],
        "answer": answer.model_dump(),
        "answer_trace": answer_trace,
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
    }
    if "scoped" not in report["completed_stages"]:
        report["completed_stages"].append("scoped")
    _save(report)


def run_planner(report: dict[str, Any]) -> None:
    filters = build_filters(COMBINED_QUERY, {})
    for backend, factory in (
        ("langgraph", build_langgraph_agentic_retriever),
        ("llamaindex", build_llamaindex_agentic_retriever),
    ):
        print(json.dumps({"stage": "planner", "backend": backend}), flush=True)
        started = perf_counter()
        events: list[dict[str, Any]] = []
        with capture_ollama_usage() as usage_events:
            state = factory(use_llm=True, event_callback=events.append).invoke(
                {
                    "query": COMBINED_QUERY,
                    "corpus_ids": CORPUS_IDS,
                    "filters": filters,
                    "max_hops": 4,
                    "max_seconds": 240,
                }
            )
        results = [SearchResult.model_validate(item) for item in state.get("retrieval_results", [])]
        if state.get("sufficient"):
            answer, answer_trace = generate_answer_with_trace(COMBINED_QUERY, results)
        else:
            answer = insufficient_agent_answer(COMBINED_QUERY, dict(state.get("retrieval_trace") or {}))
            answer_trace = {"answer_source": "insufficient_agent_answer"}
        report["planner"][backend] = {
            "sufficient": bool(state.get("sufficient")),
            "stop_reason": state.get("stop_reason"),
            "result_chunk_ids": [result.chunk_id for result in results],
            "result_document_ids": sorted({result.source_document_id for result in results}),
            "events": events,
            "trace": dict(state.get("retrieval_trace") or {}),
            "answer": answer.model_dump(),
            "answer_trace": answer_trace,
            "usage": summarize_ollama_usage(usage_events),
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        }
        _save(report)
    if "planner" not in report["completed_stages"]:
        report["completed_stages"].append("planner")
    _save(report)


def run_cv_debug(report: dict[str, Any]) -> None:
    query = SCOPED_QUERIES[0][0]
    print(json.dumps({"stage": "cv_debug", "query": query}), flush=True)
    started = perf_counter()
    result = execute_query_debug_run(
        QueryRequest(query=query, corpus_ids=CORPUS_IDS, response_mode="answer_with_citations"),
        sample_limit=100,
    )
    result["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
    report["cv_debug"] = result
    if "cv_debug" not in report["completed_stages"]:
        report["completed_stages"].append("cv_debug")
    _save(report)


def run_dependency(report: dict[str, Any]) -> None:
    filters = build_filters(DEPENDENCY_QUERY, {})
    report["dependency"] = {}
    for backend, factory in (
        ("langgraph", build_langgraph_agentic_retriever),
        ("llamaindex", build_llamaindex_agentic_retriever),
    ):
        print(json.dumps({"stage": "dependency", "backend": backend}), flush=True)
        started = perf_counter()
        events: list[dict[str, Any]] = []
        with capture_ollama_usage() as usage_events:
            state = factory(use_llm=True, event_callback=events.append).invoke(
                {
                    "query": DEPENDENCY_QUERY,
                    "corpus_ids": CORPUS_IDS,
                    "filters": filters,
                    "max_hops": 4,
                    "max_seconds": 240,
                }
            )
        results = [SearchResult.model_validate(item) for item in state.get("retrieval_results", [])]
        if state.get("sufficient"):
            answer, answer_trace = generate_answer_with_trace(DEPENDENCY_QUERY, results)
        else:
            answer = insufficient_agent_answer(DEPENDENCY_QUERY, dict(state.get("retrieval_trace") or {}))
            answer_trace = {"answer_source": "insufficient_agent_answer"}
        result_ids = [result.chunk_id for result in results]
        report["dependency"][backend] = {
            "sufficient": bool(state.get("sufficient")),
            "stop_reason": state.get("stop_reason"),
            "expected_chunk_ids": DEPENDENCY_EXPECTED_CHUNKS,
            "expected_chunks_retained": {
                chunk_id: chunk_id in result_ids for chunk_id in DEPENDENCY_EXPECTED_CHUNKS
            },
            "result_chunk_ids": result_ids,
            "events": events,
            "trace": dict(state.get("retrieval_trace") or {}),
            "answer": answer.model_dump(),
            "answer_trace": answer_trace,
            "usage": summarize_ollama_usage(usage_events),
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        }
        _save(report)
    if "dependency" not in report["completed_stages"]:
        report["completed_stages"].append("dependency")
    _save(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("scoped", "cv_debug", "planner", "dependency", "all"),
        default="all",
    )
    args = parser.parse_args()
    report = _load_report()
    try:
        if args.stage in {"scoped", "all"}:
            run_scoped(report)
        if args.stage in {"cv_debug", "all"}:
            run_cv_debug(report)
        if args.stage in {"planner", "all"}:
            run_planner(report)
        if args.stage in {"dependency", "all"}:
            run_dependency(report)
        report["completed"] = {"scoped", "planner", "dependency"}.issubset(
            report["completed_stages"]
        )
        report.pop("error", None)
        _save(report)
    except Exception as exc:
        report["completed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        _save(report)
        raise


if __name__ == "__main__":
    main()
