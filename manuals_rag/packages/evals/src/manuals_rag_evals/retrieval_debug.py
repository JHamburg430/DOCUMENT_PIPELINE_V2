from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from manuals_rag_common.db import fetch_all
from manuals_rag_evals.retrieval_quality import summarize_result_quality
from manuals_rag_retrieval.query_analysis import analyze_query
from manuals_rag_retrieval.qdrant_store import QdrantStore
from manuals_rag_retrieval.retriever import (
    LEXICAL_TABLE_SCAN_LIMIT,
    _annotate_completeness,
    _annotate_stage_metadata,
    _apply_family_scoring,
    _apply_query_alignment,
    _chunk_search_filters,
    _comparison_table_content_terms,
    _dedupe_results,
    _lexical_table_content_terms,
    _lexical_table_symbol_terms,
    _lexical_table_terms,
    _promote_comparison_table_candidates,
    _result_matches_identifier,
    _select_family_candidates,
    _table_lexical_score,
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

SOURCE_STAGE_NAMES = (
    "dense",
    "sparse",
    "table",
    "table_lexical",
    "contextual_lexical",
    "special",
)


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
        "table_column_headers": result.metadata.get("table_column_headers"),
        "table_row_headers": result.metadata.get("table_row_headers"),
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


def _matched_terms(result: dict[str, Any], terms: list[str]) -> list[str]:
    text = _evidence_text(result)
    matched: list[str] = []
    for term in terms:
        normalized = _normalize_text(str(term))
        if normalized and normalized in text and str(term) not in matched:
            matched.append(str(term))
    return matched


def _stage_same_document_crowding(
    stages: list[RetrievalDebugStage],
    expected_evidence: list[dict[str, Any]],
    ranks_by_stage: dict[str, Any],
    *,
    full_stage_results: dict[str, list[SearchResult]] | None = None,
    top_matches: int = 5,
) -> dict[str, Any]:
    full_stage_results = full_stage_results or {}
    crowding: dict[str, Any] = {}
    for stage in stages:
        if stage.name == "metadata_document_selection":
            continue
        if stage.name in full_stage_results:
            stage_results = [_serialize_result(result) for result in full_stage_results[stage.name]]
        else:
            stage_results = stage.results
        stage_ranks = ranks_by_stage.get(stage.name) or []
        stage_items: list[dict[str, Any]] = []
        for index, item in enumerate(expected_evidence):
            rank_record = stage_ranks[index] if index < len(stage_ranks) else {}
            if rank_record.get("exact_rank") is not None:
                continue
            document_id = str(item.get("source_document_id") or "")
            if not document_id:
                continue
            terms = [str(term) for term in item.get("expected_terms", []) if str(term)]
            candidates: list[dict[str, Any]] = []
            for rank, result in enumerate(stage_results, start=1):
                if str(result.get("source_document_id") or "") != document_id:
                    continue
                matched = _matched_terms(result, terms)
                if not matched:
                    continue
                candidates.append(
                    {
                        "rank": rank,
                        "chunk_id": result.get("chunk_id"),
                        "chunk_type": result.get("chunk_type"),
                        "overlap_terms": len(matched),
                        "matched_terms": matched,
                        "missing_terms": [term for term in terms if term not in matched],
                        "content_preview": result.get("content_preview"),
                        "context_preview": result.get("context_preview"),
                        "table_column_headers": result.get("table_column_headers"),
                        "table_row_headers": result.get("table_row_headers"),
                    }
                )
            if candidates:
                stage_items.append(
                    {
                        "chunk_id": item.get("chunk_id"),
                        "source_document_id": document_id,
                        "expected_terms": terms,
                        "same_document_candidates": candidates[:top_matches],
                    }
                )
        if stage_items:
            crowding[stage.name] = stage_items
    return crowding


def _expected_evidence_lexical_probe(
    query: str,
    corpus_ids: list[str],
    filters: dict[str, Any],
    analysis: Any,
    expected_evidence: list[dict[str, Any]],
    full_stage_results: dict[str, list[SearchResult]],
) -> dict[str, Any]:
    chunk_ids = [str(item.get("chunk_id") or "") for item in expected_evidence if item.get("chunk_id")]
    if not chunk_ids:
        return {}
    try:
        rows = fetch_all(
            """
            select id, document_version_id, source_document_id, title, section_path_text, chunk_type,
                   page_from, page_to, content, metadata_json, priority_score
            from retrieval_chunks
            where id = any(%s)
            """,
            (chunk_ids,),
        )
    except Exception as exc:
        return {"database_probe_error": str(exc), "items": [{"chunk_id": chunk_id, "found_in_database": False} for chunk_id in chunk_ids]}
    rows_by_id = {str(row["id"]): dict(row) for row in rows}
    try:
        lexical_terms = _lexical_table_terms(query, analysis)
    except AttributeError:
        lexical_terms = [
            term
            for term in (re.sub(r"[^a-z0-9]+", "", raw.lower()) for raw in query.split())
            if len(term) >= 3
        ][:16]
    is_comparison = "comparison" in getattr(analysis, "query_types", []) and bool(getattr(analysis, "product_identifiers", []))
    if is_comparison:
        content_prefilter_terms = [
            term
            for term in [*_comparison_table_content_terms(lexical_terms), *_lexical_table_symbol_terms(lexical_terms)]
            if term
        ]
    else:
        content_prefilter_terms = [*_lexical_table_content_terms(lexical_terms), *_lexical_table_symbol_terms(lexical_terms)]
    content_prefilter_terms = list(dict.fromkeys(content_prefilter_terms))
    sql_pool_details: dict[str, dict[str, Any]] = {}
    sql_pool_error: str | None = None
    if content_prefilter_terms:
        try:
            sql_pool_details = _table_lexical_sql_pool_details(
                corpus_ids=corpus_ids,
                filters=filters,
                analysis=analysis,
                terms=lexical_terms,
                content_prefilter_terms=content_prefilter_terms,
                is_comparison=is_comparison,
            )
        except Exception as exc:
            sql_pool_error = str(exc)
    stage_presence: dict[str, dict[str, int]] = {}
    for stage_name, results in full_stage_results.items():
        for rank, result in enumerate(results, start=1):
            if result.chunk_id in chunk_ids:
                stage_presence.setdefault(result.chunk_id, {})[stage_name] = rank
    probe_items: list[dict[str, Any]] = []
    for item in expected_evidence:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        row = rows_by_id.get(chunk_id)
        if row is None:
            probe_items.append({"chunk_id": chunk_id, "found_in_database": False})
            continue
        metadata = dict(row.get("metadata_json") or {})
        content = str(row.get("content") or "")
        compact_content = "".join(char for char in content.lower() if char.isalnum())
        lower_content = content.lower()
        matched_prefilter_terms = [
            term
            for term in content_prefilter_terms
            if (term.lower() in lower_content if is_comparison else term.lower() in compact_content)
        ]
        expected_terms = [str(term) for term in item.get("expected_terms", []) if str(term)]
        expected_terms_matched = [
            term
            for term in expected_terms
            if " ".join(str(term).lower().split()) in " ".join(content.lower().split())
        ]
        identifiers = [str(identifier) for identifier in getattr(analysis, "product_identifiers", []) or [] if str(identifier)]
        result_for_identifier = SearchResult(
            chunk_id=chunk_id,
            score=float(row.get("priority_score") or 0.0),
            title=str(row.get("title") or ""),
            document_version_id=str(row.get("document_version_id") or ""),
            source_document_id=str(row.get("source_document_id") or ""),
            pages=list(range(int(row["page_from"]), int(row["page_to"]) + 1)),
            section_path=[str(part) for part in metadata.get("section_path") or [row.get("section_path_text") or "Document"]],
            content=content,
            metadata={
                **metadata,
                "chunk_id": chunk_id,
                "document_version_id": str(row.get("document_version_id") or ""),
                "source_document_id": str(row.get("source_document_id") or ""),
                "chunk_type": str(row.get("chunk_type") or metadata.get("chunk_type") or "table_record"),
                "title": str(row.get("title") or ""),
                "content": content,
                "content_for_rerank": content,
                "priority_score": float(row.get("priority_score") or 0.0),
            },
        )
        probe_items.append(
            {
                "chunk_id": chunk_id,
                "found_in_database": True,
                "source_document_id": str(row.get("source_document_id") or ""),
                "chunk_type": str(row.get("chunk_type") or metadata.get("chunk_type") or ""),
                "table_column_headers": metadata.get("table_column_headers"),
                "table_row_headers": metadata.get("table_row_headers"),
                "lexical_terms_matched_by_content_prefilter": matched_prefilter_terms,
                "expected_terms_matched_in_content": expected_terms_matched,
                "expected_terms_missing_from_content": [term for term in expected_terms if term not in expected_terms_matched],
                "table_lexical_score": round(_table_lexical_score(row, lexical_terms), 6),
                "table_lexical_sql_pool_rank": (sql_pool_details.get(chunk_id) or {}).get("rank"),
                "table_lexical_sql_pool_limit": LEXICAL_TABLE_SCAN_LIMIT,
                "table_lexical_sql_pool_order_features": (sql_pool_details.get(chunk_id) or {}).get("order_features"),
                "matched_query_identifiers": [
                    identifier for identifier in identifiers if _result_matches_identifier(result_for_identifier, identifier)
                ],
                "stage_exact_ranks": stage_presence.get(chunk_id, {}),
            }
        )
    return {
        "lexical_terms": lexical_terms,
        "content_prefilter_terms": content_prefilter_terms,
        "sql_pool_error": sql_pool_error,
        "items": probe_items,
    }


def _normalized_row_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _row_key_candidates(content: str, metadata: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    setting_match = re.search(r"\bSetting item:\s*([^;\n]+)", content, flags=re.IGNORECASE)
    if setting_match:
        candidates.append({"source": "setting_item", "text": setting_match.group(1).strip()})
    for header in metadata.get("table_row_headers") or []:
        text = str(header).strip()
        if text:
            candidates.append({"source": "table_row_header", "text": text})
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        normalized = _normalized_row_key(candidate["text"])
        if len(normalized) < 3:
            continue
        key = (candidate["source"], normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**candidate, "normalized": normalized})
    return deduped


def _row_key_pool_count(corpus_ids: list[str], normalized_key: str) -> int | None:
    try:
        rows = fetch_all(
            """
            select count(*) as pool_count
            from retrieval_chunks
            where chunk_type = 'table_record'
              and is_active = true
              and metadata_json->>'corpus_id' = any(%s)
              and regexp_replace(lower(content), '[^a-z0-9]+', '', 'g') like %s
            """,
            (corpus_ids, f"%{normalized_key}%"),
        )
    except Exception:
        return None
    if not rows:
        return None
    value = rows[0].get("pool_count", rows[0].get("c"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_evidence_row_key_probe(
    query: str,
    corpus_ids: list[str],
    analysis: Any,
    expected_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk_ids = [str(item.get("chunk_id")) for item in expected_evidence if item.get("chunk_id")]
    if not chunk_ids:
        return {"items": []}
    rows = fetch_all(
        """
        select id, document_version_id, source_document_id, title, section_path_text,
               page_from, page_to, content, metadata_json, priority_score
        from retrieval_chunks
        where id = any(%s)
        """,
        (chunk_ids,),
    )
    rows_by_id = {str(row["id"]): row for row in rows}
    normalized_query = _normalized_row_key(query)
    query_row_keys = [_normalized_row_key(match.group(0)) for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-\s][A-Z0-9][A-Za-z0-9]*)*\b", query)]
    query_row_keys = list(dict.fromkeys(key for key in query_row_keys if len(key) >= 3))
    items: list[dict[str, Any]] = []
    identifiers = [str(identifier) for identifier in getattr(analysis, "product_identifiers", []) or [] if str(identifier)]
    for evidence in expected_evidence:
        chunk_id = str(evidence.get("chunk_id") or "")
        row = rows_by_id.get(chunk_id)
        if row is None:
            items.append({"chunk_id": chunk_id, "found_in_database": False})
            continue
        metadata = dict(row.get("metadata_json") or {})
        content = str(row.get("content") or "")
        row_keys = _row_key_candidates(content, metadata)
        result_for_identifier = SearchResult(
            chunk_id=chunk_id,
            score=float(row.get("priority_score") or 0.0),
            title=str(row.get("title") or ""),
            document_version_id=str(row.get("document_version_id") or ""),
            source_document_id=str(row.get("source_document_id") or ""),
            pages=list(range(int(row["page_from"]), int(row["page_to"]) + 1)),
            section_path=[str(part) for part in metadata.get("section_path") or [row.get("section_path_text") or "Document"]],
            content=content,
            metadata={
                **metadata,
                "chunk_id": chunk_id,
                "document_version_id": str(row.get("document_version_id") or ""),
                "source_document_id": str(row.get("source_document_id") or ""),
                "chunk_type": str(row.get("chunk_type") or metadata.get("chunk_type") or "table_record"),
                "title": str(row.get("title") or ""),
                "content": content,
                "content_for_rerank": content,
                "priority_score": float(row.get("priority_score") or 0.0),
            },
        )
        items.append(
            {
                "chunk_id": chunk_id,
                "found_in_database": True,
                "source_document_id": str(row.get("source_document_id") or ""),
                "matched_query_identifiers": [
                    identifier for identifier in identifiers if _result_matches_identifier(result_for_identifier, identifier)
                ],
                "row_key_candidates": [
                    {
                        **candidate,
                        "mentioned_in_query": candidate["normalized"] in normalized_query,
                        "corpus_pool_count": _row_key_pool_count(corpus_ids, candidate["normalized"]),
                    }
                    for candidate in row_keys
                ],
            }
        )
    return {"query_row_keys": query_row_keys, "items": items}


def _table_lexical_sql_pool_details(
    *,
    corpus_ids: list[str],
    filters: dict[str, Any],
    analysis: Any,
    terms: list[str],
    content_prefilter_terms: list[str],
    is_comparison: bool,
) -> dict[str, int]:
    where = [
        "chunk_type = 'table_record'",
        "is_active = true",
        "metadata_json->>'corpus_id' = any(%s)",
    ]
    params: list[object] = [corpus_ids]
    source_document_ids = filters.get("source_document_id")
    if source_document_ids:
        document_ids = source_document_ids if isinstance(source_document_ids, list) else [source_document_ids]
        where.append("source_document_id = any(%s)")
        params.append([str(item) for item in document_ids])
    document_version_ids = filters.get("document_version_id")
    if document_version_ids:
        version_ids = document_version_ids if isinstance(document_version_ids, list) else [document_version_ids]
        where.append("document_version_id = any(%s)")
        params.append([str(item) for item in version_ids])
    for key in ("product_model", "product_family", "document_kind"):
        value = filters.get(key)
        if value:
            values = value if isinstance(value, list) else [value]
            where.append("metadata_json->>%s = any(%s)")
            params.extend([key, [str(item) for item in values]])
    if is_comparison:
        where.append("(" + " or ".join(["content ilike %s"] * len(content_prefilter_terms)) + ")")
        params.extend([f"%{term}%" for term in content_prefilter_terms])
    else:
        where.append(
            "("
            + " or ".join(["regexp_replace(lower(content), '[^a-z0-9]+', '', 'g') like %s"] * len(content_prefilter_terms))
            + ")"
        )
        params.extend([f"%{term}%" for term in content_prefilter_terms])
    order_by = "order by priority_score desc, id"
    order_params: list[object] = []
    if is_comparison:
        order_terms: list[str] = []
        for term in terms:
            if term not in order_terms:
                order_terms.append(term)
            if len(order_terms) >= 10:
                break
        if order_terms:
            product_patterns = _sql_pool_product_patterns(analysis)
            product_fragments = [
                (
                    "case when metadata_json->>'product_model' ilike %s "
                    "or metadata_json->>'product_family' ilike %s "
                    "or metadata_json->>'product_models' ilike %s "
                    "or metadata_json->>'devices' ilike %s then 3 else 0 end"
                )
                for _ in product_patterns
            ]
            order_by = (
                "order by "
                + " + ".join(
                    [
                        *product_fragments,
                        *(["case when content ilike %s then 1 else 0 end"] * len(order_terms)),
                    ]
                )
                + " desc, priority_score desc, id"
            )
            for pattern in product_patterns:
                order_params.extend([pattern, pattern, pattern, pattern])
            order_params.extend([f"%{term}%" for term in order_terms])
    rows = fetch_all(
        f"""
        select id, content, metadata_json, priority_score
        from retrieval_chunks
        where {" and ".join(where)}
        {order_by}
        limit {LEXICAL_TABLE_SCAN_LIMIT}
        """,
        tuple([*params, *order_params]),
    )
    return {
        str(row["id"]): {
            "rank": index,
            "order_features": _sql_pool_order_features(
                row,
                analysis=analysis,
                terms=terms,
                is_comparison=is_comparison,
            ),
        }
        for index, row in enumerate(rows, start=1)
    }


def _sql_pool_product_patterns(analysis: Any) -> list[str]:
    return [
        f"%{str(identifier).strip()}%"
        for identifier in (getattr(analysis, "product_identifiers", []) or [])[:4]
        if str(identifier).strip()
    ]


def _sql_pool_order_features(
    row: dict[str, Any],
    *,
    analysis: Any,
    terms: list[str],
    is_comparison: bool,
) -> dict[str, Any]:
    metadata = dict(row.get("metadata_json") or {})
    content = str(row.get("content") or "")
    content_lower = content.lower()
    order_terms: list[str] = []
    for term in terms:
        if term not in order_terms:
            order_terms.append(term)
        if len(order_terms) >= 10:
            break
    matched_order_terms = [term for term in order_terms if term.lower() in content_lower]
    identifiers = [str(identifier) for identifier in getattr(analysis, "product_identifiers", []) or [] if str(identifier)]
    product_haystack = " ".join(
        str(part)
        for part in [
            metadata.get("product_model"),
            metadata.get("product_family"),
            metadata.get("product_models"),
            metadata.get("devices"),
        ]
        if part
    ).lower()
    matched_identifiers = [
        identifier
        for identifier in identifiers[:4]
        if str(identifier).strip().lower() in product_haystack
    ]
    return {
        "sql_product_match_score": len(matched_identifiers) * 3 if is_comparison else 0,
        "sql_content_match_score": len(matched_order_terms) if is_comparison else 0,
        "matched_product_identifiers": matched_identifiers,
        "matched_order_terms": matched_order_terms,
        "priority_score": float(row.get("priority_score") or 0.0),
    }


def _stage_expected_evidence_ranks(
    stages: list[RetrievalDebugStage],
    expected_evidence: list[dict[str, Any]] | None,
    *,
    full_stage_results: dict[str, list[SearchResult]] | None = None,
) -> dict[str, Any]:
    if not expected_evidence:
        return {}
    full_stage_results = full_stage_results or {}
    diagnostics: dict[str, Any] = {}
    for stage in stages:
        if stage.name == "metadata_document_selection":
            continue
        if stage.name in full_stage_results:
            stage_results = [_serialize_result(result) for result in full_stage_results[stage.name]]
        else:
            stage_results = stage.results
        stage_records: list[dict[str, Any]] = []
        for item in expected_evidence:
            chunk_id = str(item.get("chunk_id") or "")
            document_id = str(item.get("source_document_id") or "")
            terms = [str(term) for term in item.get("expected_terms", []) if str(term)]
            exact_rank: int | None = None
            same_document_best_rank: int | None = None
            same_document_best_overlap = 0
            for rank, result in enumerate(stage_results, start=1):
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
                    "rank_search_depth": len(stage_results),
                }
            )
        diagnostics[stage.name] = stage_records
    return diagnostics


def _stage_expected_evidence_top_k_outcomes(
    ranks_by_stage: dict[str, Any],
    *,
    cutoffs: list[int],
) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    for stage_name, records in ranks_by_stage.items():
        stage_outcomes: dict[str, Any] = {}
        for cutoff in cutoffs:
            matched_items: list[dict[str, Any]] = []
            missing_items: list[dict[str, Any]] = []
            for record in records:
                required_overlap = max(1, min(2, int(record.get("expected_term_count") or 0)))
                exact_rank = record.get("exact_rank")
                same_document_rank = record.get("same_document_best_rank")
                same_document_overlap = int(record.get("same_document_best_overlap") or 0)
                matched = bool(
                    (exact_rank is not None and exact_rank <= cutoff)
                    or (
                        same_document_rank is not None
                        and same_document_rank <= cutoff
                        and same_document_overlap >= required_overlap
                    )
                )
                item = {
                    "chunk_id": record.get("chunk_id"),
                    "matched": matched,
                    "exact_rank": exact_rank,
                    "same_document_best_rank": same_document_rank,
                    "same_document_best_overlap": same_document_overlap,
                    "required_overlap": required_overlap,
                }
                if matched:
                    matched_items.append(item)
                else:
                    missing_items.append(item)
            stage_outcomes[str(cutoff)] = {
                "passed": bool(records) and not missing_items,
                "matched_count": len(matched_items),
                "missing_count": len(missing_items),
                "matched_evidence": matched_items,
                "missing_evidence": missing_items,
            }
        outcomes[stage_name] = stage_outcomes
    return outcomes


def _expected_evidence_source_stage_contribution(ranks_by_stage: dict[str, Any]) -> dict[str, Any]:
    if not ranks_by_stage:
        return {}
    source_ranks = {stage: ranks_by_stage.get(stage) or [] for stage in SOURCE_STAGE_NAMES}
    evidence_count = max((len(records) for records in source_ranks.values()), default=0)
    exact_hits_by_stage = {stage: 0 for stage in SOURCE_STAGE_NAMES}
    unique_exact_hits_by_stage = {stage: 0 for stage in SOURCE_STAGE_NAMES}
    missed_by_source_stages: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    for index in range(evidence_count):
        source_hits: list[dict[str, Any]] = []
        chunk_id = ""
        source_document_id = ""
        for stage_name in SOURCE_STAGE_NAMES:
            stage_records = source_ranks.get(stage_name) or []
            if index >= len(stage_records):
                continue
            record = stage_records[index]
            chunk_id = chunk_id or str(record.get("chunk_id") or "")
            source_document_id = source_document_id or str(record.get("source_document_id") or "")
            exact_rank = record.get("exact_rank")
            if exact_rank is None:
                continue
            exact_hits_by_stage[stage_name] += 1
            source_hits.append({"stage": stage_name, "exact_rank": exact_rank})
        if len(source_hits) == 1:
            unique_exact_hits_by_stage[source_hits[0]["stage"]] += 1
        if not source_hits:
            missed_by_source_stages.append(
                {
                    "chunk_id": chunk_id,
                    "source_document_id": source_document_id,
                    "evidence_index": index,
                }
            )
        evidence_items.append(
            {
                "chunk_id": chunk_id,
                "source_document_id": source_document_id,
                "source_exact_hits": source_hits,
                "unique_to_stage": source_hits[0]["stage"] if len(source_hits) == 1 else None,
                "missed_by_all_source_stages": not bool(source_hits),
            }
        )
    return {
        "source_stage_names": list(SOURCE_STAGE_NAMES),
        "expected_evidence_count": evidence_count,
        "exact_hits_by_stage": exact_hits_by_stage,
        "unique_exact_hits_by_stage": unique_exact_hits_by_stage,
        "missed_by_source_stage_count": len(missed_by_source_stages),
        "missed_by_source_stages": missed_by_source_stages,
        "evidence_items": evidence_items,
    }


def _case_diagnostics(
    stages: list[RetrievalDebugStage],
    *,
    query: str,
    corpus_ids: list[str],
    filters: dict[str, Any],
    analysis: Any,
    expected_evidence: list[dict[str, Any]] | None = None,
    full_stage_results: dict[str, list[SearchResult]] | None = None,
    stage_timings_seconds: dict[str, float] | None = None,
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
        "stage_timings_seconds": stage_timings_seconds or {},
    }
    if expected_evidence:
        stage_ranks = _stage_expected_evidence_ranks(
            stages,
            expected_evidence,
            full_stage_results=full_stage_results,
        )
        cutoffs = sorted({5, 10, *[stage.count for stage in stages if stage.name != "metadata_document_selection"]})
        diagnostics["expected_evidence_stage_ranks"] = stage_ranks
        diagnostics["expected_evidence_top_k_outcomes"] = _stage_expected_evidence_top_k_outcomes(
            stage_ranks,
            cutoffs=cutoffs,
        )
        diagnostics["expected_evidence_source_stage_contribution"] = _expected_evidence_source_stage_contribution(stage_ranks)
        diagnostics["expected_evidence_same_document_crowding"] = _stage_same_document_crowding(
            stages,
            expected_evidence,
            stage_ranks,
            full_stage_results=full_stage_results,
        )
        diagnostics["expected_evidence_lexical_probe"] = _expected_evidence_lexical_probe(
            query,
            corpus_ids,
            filters,
            analysis,
            expected_evidence,
            full_stage_results or {},
        )
        diagnostics["expected_evidence_row_key_probe"] = _expected_evidence_row_key_probe(
            query,
            corpus_ids,
            analysis,
            expected_evidence,
        )
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
        "stage_timing_totals_seconds": {},
        "stage_timing_max_seconds": {},
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
        for stage_name, elapsed in diagnostics.get("stage_timings_seconds", {}).items():
            summary["stage_timing_totals_seconds"][stage_name] = round(
                float(summary["stage_timing_totals_seconds"].get(stage_name, 0.0)) + float(elapsed),
                6,
            )
            summary["stage_timing_max_seconds"][stage_name] = round(
                max(float(summary["stage_timing_max_seconds"].get(stage_name, 0.0)), float(elapsed)),
                6,
            )
    return summary


def debug_retrieval_query(
    *,
    query: str,
    corpus_ids: list[str],
    request_filters: dict[str, Any] | None = None,
    top_k: int = 5,
    stage_candidate_limit: int | None = None,
    expected_evidence: list[dict[str, Any]] | None = None,
) -> RetrievalDebugCase:
    request_filters = dict(request_filters or {})
    analysis = analyze_query(query)
    filters = build_filters(query, request_filters)
    store = QdrantStore()
    search_filters, metadata_document_hits = select_documents_from_metadata(store, query, corpus_ids, filters)
    chunk_search_filters = _chunk_search_filters(filters, search_filters, analysis)
    stage_limit = max(stage_candidate_limit or max(top_k * 2, 10), top_k)
    fused_limit = max(stage_limit * 2, top_k * 4, 20)

    stage_timings_seconds: dict[str, float] = {}

    def timed_stage(name: str, callback: Any) -> Any:
        started_at = time.perf_counter()
        try:
            return callback()
        finally:
            stage_timings_seconds[name] = round(time.perf_counter() - started_at, 6)

    dense = timed_stage(
        "dense",
        lambda: _annotate_stage_metadata(run_dense_search(store, query, corpus_ids, chunk_search_filters, limit=stage_limit), "dense"),
    )
    sparse = timed_stage(
        "sparse",
        lambda: _annotate_stage_metadata(run_sparse_search(store, query, corpus_ids, chunk_search_filters, limit=stage_limit), "sparse"),
    )
    table = timed_stage(
        "table",
        lambda: _annotate_stage_metadata(run_table_search(store, query, corpus_ids, chunk_search_filters, limit=stage_limit), "table"),
    )
    table_lexical = timed_stage(
        "table_lexical",
        lambda: _annotate_stage_metadata(run_table_lexical_search(query, corpus_ids, filters, analysis, limit=stage_limit), "table_lexical"),
    )
    contextual_lexical = timed_stage(
        "contextual_lexical",
        lambda: _annotate_stage_metadata(
            run_contextual_lexical_search(query, corpus_ids, filters, analysis, limit=stage_limit),
            "contextual_lexical",
        ),
    )
    special = timed_stage(
        "special",
        lambda: _annotate_stage_metadata(run_special_search(store, query, corpus_ids, chunk_search_filters, analysis, limit=stage_limit), "special"),
    )
    fused = timed_stage(
        "fused",
        lambda: _annotate_stage_metadata(
            fuse_results(store, [dense, sparse, table, table_lexical, contextual_lexical, special], limit=fused_limit),
            "fused",
        ),
    )
    family_scored = timed_stage(
        "family_scored",
        lambda: _annotate_stage_metadata(_apply_family_scoring(fused, analysis, stage="family_scored"), "family_scored"),
    )
    completeness_scored = timed_stage(
        "completeness_scored",
        lambda: _annotate_stage_metadata(_annotate_completeness(family_scored), "completeness_scored"),
    )
    query_aligned = timed_stage(
        "query_aligned",
        lambda: _annotate_stage_metadata(_apply_query_alignment(completeness_scored, analysis, stage="query_aligned"), "query_aligned"),
    )
    family_selected = timed_stage(
        "family_selected",
        lambda: _annotate_stage_metadata(
            _select_family_candidates(query_aligned, analysis, filters=search_filters, limit=stage_limit),
            "family_selected",
        ),
    )
    enriched = timed_stage(
        "enrich_candidates_for_rerank",
        lambda: enrich_candidates_for_rerank(family_selected, analysis, limit=stage_limit),
    )
    reranked = timed_stage(
        "reranked",
        lambda: _annotate_stage_metadata(rerank_results(enriched, query, limit=stage_limit), "reranked"),
    )
    comparison_promoted = timed_stage(
        "comparison_table_promoted",
        lambda: _annotate_stage_metadata(
            _promote_comparison_table_candidates(reranked, table_lexical, analysis, limit=stage_limit),
            "comparison_table_promoted",
        ),
    )
    deduped = timed_stage(
        "deduped",
        lambda: _annotate_stage_metadata(_dedupe_results(comparison_promoted, analysis), "deduped"),
    )
    assembled = timed_stage(
        "assembled",
        lambda: _annotate_stage_metadata(assemble_context(deduped, limit=top_k), "assembled"),
    )
    full_stage_results = {
        "dense": dense,
        "sparse": sparse,
        "table": table,
        "table_lexical": table_lexical,
        "contextual_lexical": contextual_lexical,
        "special": special,
        "fused": fused,
        "family_scored": family_scored,
        "completeness_scored": completeness_scored,
        "query_aligned": query_aligned,
        "family_selected": family_selected,
        "reranked": reranked,
        "comparison_table_promoted": comparison_promoted,
        "deduped": deduped,
        "assembled": assembled,
    }
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
        diagnostics=_case_diagnostics(
            stages,
            query=query,
            corpus_ids=corpus_ids,
            filters=filters,
            analysis=analysis,
            expected_evidence=expected_evidence,
            full_stage_results=full_stage_results,
            stage_timings_seconds=stage_timings_seconds,
        ),
    )


def debug_retrieval_report(
    *,
    corpus_ids: list[str],
    queries: list[str],
    request_filters: dict[str, Any] | None = None,
    top_k: int = 5,
    stage_candidate_limit: int | None = None,
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
                stage_candidate_limit=stage_candidate_limit,
                expected_evidence=expected_evidence_by_query.get(query),
            )
        )
        for query in queries
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_ids": corpus_ids,
        "top_k": top_k,
        "stage_candidate_limit": stage_candidate_limit or max(top_k * 2, 10),
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
        f"- Stage candidate limit: `{report.get('stage_candidate_limit', max(report['top_k'] * 2, 10))}`",
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
