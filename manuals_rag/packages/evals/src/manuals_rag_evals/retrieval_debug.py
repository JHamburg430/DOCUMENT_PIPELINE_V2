from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from manuals_rag_evals.retrieval_quality import summarize_result_quality
from manuals_rag_retrieval.query_analysis import analyze_query
from manuals_rag_retrieval.qdrant_store import QdrantStore
from manuals_rag_retrieval.retriever import (
    _annotate_completeness,
    _annotate_stage_metadata,
    _apply_family_scoring,
    _apply_query_alignment,
    _dedupe_results,
    _select_family_candidates,
    assemble_context,
    build_filters,
    enrich_candidates_for_rerank,
    fuse_results,
    rerank_results,
    run_dense_search,
    run_sparse_search,
    run_special_search,
)
from manuals_rag_schemas.documents import SearchResult


@dataclass(frozen=True)
class RetrievalDebugStage:
    name: str
    count: int
    results: list[dict[str, Any]]


@dataclass(frozen=True)
class RetrievalDebugCase:
    query: str
    filters: dict[str, Any]
    query_types: list[str]
    preferred_chunk_types: list[str]
    stages: list[RetrievalDebugStage]
    diagnostics: dict[str, Any]


def _serialize_result(result: SearchResult) -> dict[str, Any]:
    quality = summarize_result_quality(result)
    return {
        "chunk_id": result.chunk_id,
        "score": round(float(result.score), 6),
        "chunk_type": result.metadata.get("chunk_type"),
        "retrieval_stage": result.metadata.get("retrieval_stage"),
        "pre_rerank_rank": result.metadata.get("pre_rerank_rank"),
        "post_rerank_rank": result.metadata.get("post_rerank_rank"),
        "family_bucket": result.metadata.get("family_bucket"),
        "semantic_completeness_score": result.metadata.get("semantic_completeness_score"),
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "content_preview": str(result.content)[:220],
        "low_information": quality["low_information"],
        "structured_low_information": quality["structured_low_information"],
        "technical_signal_score": quality["technical_signal_score"],
    }


def _stage(name: str, results: list[SearchResult], *, top_k: int) -> RetrievalDebugStage:
    return RetrievalDebugStage(
        name=name,
        count=len(results),
        results=[_serialize_result(result) for result in results[:top_k]],
    )


def _case_diagnostics(stages: list[RetrievalDebugStage]) -> dict[str, Any]:
    top_results = {stage.name: (stage.results[0] if stage.results else None) for stage in stages}
    low_information_top_stages = [name for name, result in top_results.items() if result and result.get("low_information")]
    structured_low_information_top_stages = [
        name for name, result in top_results.items() if result and result.get("structured_low_information")
    ]
    reranked = top_results.get("reranked")
    family_selected = top_results.get("family_selected")
    diagnostics = {
        "low_information_top_stages": low_information_top_stages,
        "structured_low_information_top_stages": structured_low_information_top_stages,
        "special_empty": not bool(next((stage.results for stage in stages if stage.name == "special"), [])),
        "rerank_promoted_low_information": bool(reranked and reranked.get("low_information")),
        "family_selection_changed_top_chunk": bool(
            family_selected and top_results.get("fused") and family_selected.get("chunk_id") != top_results["fused"].get("chunk_id")
        ),
        "rerank_changed_top_chunk": bool(
            reranked and family_selected and reranked.get("chunk_id") != family_selected.get("chunk_id")
        ),
    }
    return diagnostics


def _report_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "cases_with_low_information_top_hit": 0,
        "cases_with_structured_low_information_top_hit": 0,
        "cases_with_empty_special_stage": 0,
        "cases_where_rerank_promoted_low_information": 0,
        "cases_where_family_selection_changed_top_chunk": 0,
        "cases_where_rerank_changed_top_chunk": 0,
    }
    for case in cases:
        diagnostics = case["diagnostics"]
        summary["cases_with_low_information_top_hit"] += int(bool(diagnostics["low_information_top_stages"]))
        summary["cases_with_structured_low_information_top_hit"] += int(
            bool(diagnostics["structured_low_information_top_stages"])
        )
        summary["cases_with_empty_special_stage"] += int(diagnostics["special_empty"])
        summary["cases_where_rerank_promoted_low_information"] += int(diagnostics["rerank_promoted_low_information"])
        summary["cases_where_family_selection_changed_top_chunk"] += int(diagnostics["family_selection_changed_top_chunk"])
        summary["cases_where_rerank_changed_top_chunk"] += int(diagnostics["rerank_changed_top_chunk"])
    return summary


def debug_retrieval_query(
    *,
    query: str,
    corpus_ids: list[str],
    request_filters: dict[str, Any] | None = None,
    top_k: int = 5,
) -> RetrievalDebugCase:
    request_filters = dict(request_filters or {})
    analysis = analyze_query(query)
    filters = build_filters(query, request_filters)
    store = QdrantStore()

    dense = _annotate_stage_metadata(run_dense_search(store, query, corpus_ids, filters, limit=max(top_k * 2, 10)), "dense")
    sparse = _annotate_stage_metadata(run_sparse_search(store, query, corpus_ids, filters, limit=max(top_k * 2, 10)), "sparse")
    special = _annotate_stage_metadata(run_special_search(store, query, corpus_ids, filters, analysis, limit=max(top_k * 2, 10)), "special")
    fused = _annotate_stage_metadata(fuse_results(store, [dense, sparse, special], limit=max(top_k * 4, 20)), "fused")
    family_scored = _annotate_stage_metadata(_apply_family_scoring(fused, analysis, stage="family_scored"), "family_scored")
    completeness_scored = _annotate_stage_metadata(_annotate_completeness(family_scored), "completeness_scored")
    query_aligned = _annotate_stage_metadata(_apply_query_alignment(completeness_scored, analysis, stage="query_aligned"), "query_aligned")
    family_selected = _annotate_stage_metadata(_select_family_candidates(query_aligned, analysis, limit=max(top_k * 2, 10)), "family_selected")
    enriched = enrich_candidates_for_rerank(family_selected, analysis, limit=max(top_k * 2, 10))
    reranked = _annotate_stage_metadata(rerank_results(enriched, query, limit=max(top_k * 2, 10)), "reranked")
    deduped = _annotate_stage_metadata(_dedupe_results(reranked, analysis), "deduped")
    assembled = _annotate_stage_metadata(assemble_context(deduped, limit=top_k), "assembled")

    return RetrievalDebugCase(
        query=query,
        filters=filters,
        query_types=analysis.query_types,
        preferred_chunk_types=analysis.preferred_chunk_types,
        stages=[
            _stage("dense", dense, top_k=top_k),
            _stage("sparse", sparse, top_k=top_k),
            _stage("special", special, top_k=top_k),
            _stage("fused", fused, top_k=top_k),
            _stage("family_scored", family_scored, top_k=top_k),
            _stage("completeness_scored", completeness_scored, top_k=top_k),
            _stage("query_aligned", query_aligned, top_k=top_k),
            _stage("family_selected", family_selected, top_k=top_k),
            _stage("reranked", reranked, top_k=top_k),
            _stage("assembled", assembled, top_k=top_k),
        ],
        diagnostics=_case_diagnostics(
            [
                _stage("dense", dense, top_k=top_k),
                _stage("sparse", sparse, top_k=top_k),
                _stage("special", special, top_k=top_k),
                _stage("fused", fused, top_k=top_k),
                _stage("family_scored", family_scored, top_k=top_k),
                _stage("completeness_scored", completeness_scored, top_k=top_k),
                _stage("query_aligned", query_aligned, top_k=top_k),
                _stage("family_selected", family_selected, top_k=top_k),
                _stage("reranked", reranked, top_k=top_k),
                _stage("assembled", assembled, top_k=top_k),
            ]
        ),
    )


def debug_retrieval_report(
    *,
    corpus_ids: list[str],
    queries: list[str],
    request_filters: dict[str, Any] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    cases = [
        asdict(
            debug_retrieval_query(
                query=query,
                corpus_ids=corpus_ids,
                request_filters=request_filters,
                top_k=top_k,
            )
        )
        for query in queries
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_ids": corpus_ids,
        "top_k": top_k,
        "query_count": len(queries),
        "cases": cases,
    }
    report["summary"] = _report_summary(cases)
    return report


def debug_report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=True, indent=2)


def debug_report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Debug Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Corpus IDs: `{', '.join(report['corpus_ids'])}`",
        f"- Query count: `{report['query_count']}`",
        f"- Top K per stage: `{report['top_k']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.get("summary", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    for case in report["cases"]:
        lines.extend([
            f"## {case['query']}",
            f"- Query types: `{case['query_types']}`",
            f"- Preferred chunk types: `{case['preferred_chunk_types']}`",
            f"- Filters: `{case['filters']}`",
            f"- Diagnostics: `{case['diagnostics']}`",
            "",
        ])
        for stage in case["stages"]:
            lines.append(f"### {stage['name']}")
            lines.append(f"- Result count: `{stage['count']}`")
            for index, result in enumerate(stage["results"], start=1):
                lines.append(
                    f"- `{index}.` `{result['chunk_type']}` score `{result['score']}` pages `{result['pages']}`: `{result['content_preview']}`"
                )
            lines.append("")
    return "\n".join(lines)
