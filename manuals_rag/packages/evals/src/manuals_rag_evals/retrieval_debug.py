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
    _chunk_search_filters,
    _dedupe_results,
    _promote_comparison_table_candidates,
    _select_family_candidates,
    assemble_context,
    build_filters,
    enrich_candidates_for_rerank,
    fuse_results,
    rerank_results,
    run_contextual_lexical_search,
    run_dense_search,
    run_sparse_search,
    run_special_search,
    run_table_lexical_search,
    run_table_search,
    select_documents_from_metadata,
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
    chunk_search_filters: dict[str, Any]
    query_types: list[str]
    preferred_chunk_types: list[str]
    stages: list[RetrievalDebugStage]
    diagnostics: dict[str, Any]


def _serialize_result(result: SearchResult) -> dict[str, Any]:
    quality = summarize_result_quality(result)
    return {
        "chunk_id": result.chunk_id,
        "score": round(float(result.score), 6),
        "source_document_id": result.source_document_id,
        "document_version_id": result.document_version_id,
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
        "context_preview": str(result.metadata.get("context_window") or "")[:220],
        "table_row_group_context_preview": str(result.metadata.get("table_row_group_context") or "")[:220],
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


def _metadata_document_selection_stage(hits: list[dict[str, Any]], *, top_k: int) -> RetrievalDebugStage:
    return RetrievalDebugStage(
        name="metadata_document_selection",
        count=len(hits),
        results=[
            {
                "source_document_id": hit.get("source_document_id"),
                "score": round(float(hit.get("score", 0.0)), 6),
                "retrieval_stage": hit.get("retrieval_stage"),
                "title": (hit.get("payload") or {}).get("title") if isinstance(hit.get("payload"), dict) else None,
                "source_filename": (hit.get("payload") or {}).get("source_filename") if isinstance(hit.get("payload"), dict) else None,
            }
            for hit in hits[:top_k]
        ],
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _evidence_text(result: dict[str, Any]) -> str:
    return _normalize_text(
        " ".join(
            str(result.get(key) or "")
            for key in (
                "content_preview",
                "context_preview",
                "table_row_group_context_preview",
                "title",
            )
        )
    )


def _term_overlap(result: dict[str, Any], terms: list[str]) -> int:
    text = _evidence_text(result)
    return sum(1 for term in terms if _normalize_text(str(term)) and _normalize_text(str(term)) in text)


def _stage_expected_evidence_ranks(
    stages: list[RetrievalDebugStage],
    expected_evidence: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not expected_evidence:
        return {}
    diagnostics: dict[str, Any] = {}
    for stage in stages:
        if stage.name == "metadata_document_selection":
            continue
        stage_records: list[dict[str, Any]] = []
        for item in expected_evidence:
            chunk_id = str(item.get("chunk_id") or "")
            document_id = str(item.get("source_document_id") or "")
            terms = [str(term) for term in item.get("expected_terms", []) if str(term)]
            exact_rank: int | None = None
            same_document_best_rank: int | None = None
            same_document_best_overlap = 0
            for rank, result in enumerate(stage.results, start=1):
                if chunk_id and str(result.get("chunk_id") or "") == chunk_id:
                    exact_rank = rank
                if document_id and str(result.get("source_document_id") or "") == document_id:
                    overlap = _term_overlap(result, terms)
                    if same_document_best_rank is None or overlap > same_document_best_overlap:
                        same_document_best_rank = rank
                        same_document_best_overlap = overlap
            stage_records.append(
                {
                    "chunk_id": chunk_id,
                    "source_document_id": document_id,
                    "exact_rank": exact_rank,
                    "same_document_best_rank": same_document_best_rank,
                    "same_document_best_overlap": same_document_best_overlap,
                    "expected_term_count": len(terms),
                }
            )
        diagnostics[stage.name] = stage_records
    return diagnostics


def _case_diagnostics(
    stages: list[RetrievalDebugStage],
    *,
    expected_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
        "metadata_document_selection_used": bool(top_results.get("metadata_document_selection")),
        "rerank_promoted_low_information": bool(reranked and reranked.get("low_information")),
        "family_selection_changed_top_chunk": bool(
            family_selected and top_results.get("fused") and family_selected.get("chunk_id") != top_results["fused"].get("chunk_id")
        ),
        "rerank_changed_top_chunk": bool(
            reranked and family_selected and reranked.get("chunk_id") != family_selected.get("chunk_id")
        ),
    }
    if expected_evidence:
        diagnostics["expected_evidence_stage_ranks"] = _stage_expected_evidence_ranks(stages, expected_evidence)
    return diagnostics


def _report_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "cases_with_metadata_document_selection": 0,
        "cases_with_low_information_top_hit": 0,
        "cases_with_structured_low_information_top_hit": 0,
        "cases_with_empty_special_stage": 0,
        "cases_where_rerank_promoted_low_information": 0,
        "cases_where_family_selection_changed_top_chunk": 0,
        "cases_where_rerank_changed_top_chunk": 0,
    }
    for case in cases:
        diagnostics = case["diagnostics"]
        summary["cases_with_metadata_document_selection"] += int(diagnostics["metadata_document_selection_used"])
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
    expected_evidence: list[dict[str, Any]] | None = None,
) -> RetrievalDebugCase:
    request_filters = dict(request_filters or {})
    analysis = analyze_query(query)
    filters = build_filters(query, request_filters)
    store = QdrantStore()
    search_filters, metadata_document_hits = select_documents_from_metadata(store, query, corpus_ids, filters)
    chunk_search_filters = _chunk_search_filters(filters, search_filters, analysis)

    dense = _annotate_stage_metadata(run_dense_search(store, query, corpus_ids, chunk_search_filters, limit=max(top_k * 2, 10)), "dense")
    sparse = _annotate_stage_metadata(run_sparse_search(store, query, corpus_ids, chunk_search_filters, limit=max(top_k * 2, 10)), "sparse")
    table = _annotate_stage_metadata(run_table_search(store, query, corpus_ids, chunk_search_filters, limit=max(top_k * 2, 10)), "table")
    table_lexical = _annotate_stage_metadata(run_table_lexical_search(query, corpus_ids, filters, analysis, limit=max(top_k * 2, 10)), "table_lexical")
    contextual_lexical = _annotate_stage_metadata(
        run_contextual_lexical_search(query, corpus_ids, filters, analysis, limit=max(top_k * 2, 10)),
        "contextual_lexical",
    )
    special = _annotate_stage_metadata(run_special_search(store, query, corpus_ids, chunk_search_filters, analysis, limit=max(top_k * 2, 10)), "special")
    fused = _annotate_stage_metadata(
        fuse_results(store, [dense, sparse, table, table_lexical, contextual_lexical, special], limit=max(top_k * 4, 20)),
        "fused",
    )
    family_scored = _annotate_stage_metadata(_apply_family_scoring(fused, analysis, stage="family_scored"), "family_scored")
    completeness_scored = _annotate_stage_metadata(_annotate_completeness(family_scored), "completeness_scored")
    query_aligned = _annotate_stage_metadata(_apply_query_alignment(completeness_scored, analysis, stage="query_aligned"), "query_aligned")
    family_selected = _annotate_stage_metadata(_select_family_candidates(query_aligned, analysis, filters=search_filters, limit=max(top_k * 2, 10)), "family_selected")
    enriched = enrich_candidates_for_rerank(family_selected, analysis, limit=max(top_k * 2, 10))
    reranked = _annotate_stage_metadata(rerank_results(enriched, query, limit=max(top_k * 2, 10)), "reranked")
    comparison_promoted = _annotate_stage_metadata(
        _promote_comparison_table_candidates(reranked, table_lexical, analysis, limit=max(top_k * 2, 10)),
        "comparison_table_promoted",
    )
    deduped = _annotate_stage_metadata(_dedupe_results(comparison_promoted, analysis), "deduped")
    assembled = _annotate_stage_metadata(assemble_context(deduped, limit=top_k), "assembled")
    stages = [
        _metadata_document_selection_stage(metadata_document_hits, top_k=top_k),
        _stage("dense", dense, top_k=top_k),
        _stage("sparse", sparse, top_k=top_k),
        _stage("table", table, top_k=top_k),
        _stage("table_lexical", table_lexical, top_k=top_k),
        _stage("contextual_lexical", contextual_lexical, top_k=top_k),
        _stage("special", special, top_k=top_k),
        _stage("fused", fused, top_k=top_k),
        _stage("family_scored", family_scored, top_k=top_k),
        _stage("completeness_scored", completeness_scored, top_k=top_k),
        _stage("query_aligned", query_aligned, top_k=top_k),
        _stage("family_selected", family_selected, top_k=top_k),
        _stage("reranked", reranked, top_k=top_k),
        _stage("comparison_table_promoted", comparison_promoted, top_k=top_k),
        _stage("deduped", deduped, top_k=top_k),
        _stage("assembled", assembled, top_k=top_k),
    ]

    return RetrievalDebugCase(
        query=query,
        filters=search_filters,
        chunk_search_filters=chunk_search_filters,
        query_types=analysis.query_types,
        preferred_chunk_types=analysis.preferred_chunk_types,
        stages=stages,
        diagnostics=_case_diagnostics(stages, expected_evidence=expected_evidence),
    )


def debug_retrieval_report(
    *,
    corpus_ids: list[str],
    queries: list[str],
    request_filters: dict[str, Any] | None = None,
    top_k: int = 5,
    expected_evidence_by_query: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    expected_evidence_by_query = expected_evidence_by_query or {}
    cases = [
        asdict(
            debug_retrieval_query(
                query=query,
                corpus_ids=corpus_ids,
                request_filters=request_filters,
                top_k=top_k,
                expected_evidence=expected_evidence_by_query.get(query),
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
                if stage["name"] == "metadata_document_selection":
                    lines.append(
                        f"- `{index}.` document `{result['source_document_id']}` score `{result['score']}` "
                        f"stage `{result['retrieval_stage']}`: `{result.get('source_filename') or result.get('title')}`"
                    )
                else:
                    lines.append(
                        f"- `{index}.` `{result['chunk_type']}` score `{result['score']}` pages `{result['pages']}`: `{result['content_preview']}`"
                    )
            lines.append("")
    return "\n".join(lines)
