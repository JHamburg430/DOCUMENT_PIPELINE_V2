from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable

from manuals_rag_common.config import settings
from manuals_rag_common.db import fetch_all
from manuals_rag_schemas.documents import SearchResult
from manuals_rag_retrieval.embeddings import tokenize
from manuals_rag_retrieval.query_analysis import QueryAnalysis, analyze_query
from manuals_rag_retrieval.qdrant_store import QdrantStore

try:
    from haystack import Document
    from haystack.components.rankers import SentenceTransformersSimilarityRanker
except ImportError as exc:
    Document = None
    SentenceTransformersSimilarityRanker = None
    _RERANK_IMPORT_ERROR = exc
else:
    _RERANK_IMPORT_ERROR = None


logger = logging.getLogger(__name__)
FUSED_CANDIDATE_POOL_LIMIT = 30
DOCUMENT_METADATA_SELECTION_LIMIT = 5
LEXICAL_TABLE_ROW_LIMIT = 24
LEXICAL_TABLE_SCAN_LIMIT = 1500
LEXICAL_CONTEXT_LIMIT = 24
LEXICAL_CONTEXT_SCAN_LIMIT = 1500
LEXICAL_TABLE_STOPWORDS = {
    "about",
    "after",
    "being",
    "causes",
    "check",
    "checked",
    "confirm",
    "corrected",
    "correction",
    "could",
    "error",
    "following",
    "please",
    "series",
    "settings",
    "should",
    "temporarily",
    "there",
    "using",
    "what",
    "when",
    "with",
}
LEXICAL_CONTEXT_STOPWORDS = LEXICAL_TABLE_STOPWORDS.union(
    {
        "detail",
        "related",
        "used",
    }
)

try:
    import torch
except ImportError:
    torch = None


def build_filters(query: str, request_filters: dict[str, object]) -> dict[str, object]:
    filters = dict(request_filters)
    filters.setdefault("is_active", True)
    return filters


def _has_explicit_document_scope(filters: dict[str, object]) -> bool:
    return any(filters.get(key) not in (None, "", [], {}) for key in ("source_document_id", "document_version_id"))


def select_documents_from_metadata(
    store: QdrantStore,
    query: str,
    corpus_ids: list[str],
    filters: dict[str, object],
    *,
    limit: int = DOCUMENT_METADATA_SELECTION_LIMIT,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if _has_explicit_document_scope(filters):
        return filters, []
    hits: list[dict[str, object]] = []
    for corpus_id in corpus_ids:
        hits.extend(store.search_document_metadata(corpus_id=corpus_id, query=query, filters=filters, limit=limit))
    if not hits:
        return filters, []
    hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    document_ids: list[str] = []
    deduped_hits: list[dict[str, object]] = []
    for hit in hits:
        document_id = str(hit.get("source_document_id") or "")
        if not document_id or document_id in document_ids:
            continue
        document_ids.append(document_id)
        deduped_hits.append(hit)
        if len(document_ids) >= limit:
            break
    if not document_ids:
        return filters, []
    return {**filters, "source_document_id": document_ids}, deduped_hits


def _special_route_filters(base_filters: dict[str, object], analysis: QueryAnalysis) -> list[dict[str, object]]:
    routes: list[dict[str, object]] = []
    if analysis.preferred_chunk_types and not analysis.safety_intent:
        routes.append({**base_filters, "chunk_type": analysis.preferred_chunk_types})
    if analysis.safety_intent:
        safety_chunk_types = ["warning_record", "procedure_record"]
        if _safety_action_terms(analysis.raw_query):
            safety_chunk_types.append("atomic_text")
        routes.append({**base_filters, "chunk_type": safety_chunk_types})
    if not analysis.safety_intent and ("how_to" in analysis.query_types or "configuration" in analysis.query_types):
        routes.append({**base_filters, "chunk_type": ["procedure_record", "section_window"]})
    if "comparison" in analysis.query_types or "compatibility" in analysis.query_types:
        routes.append({**base_filters, "chunk_type": ["spec_record", "datasheet_record", "table_record", "section_window"]})
    if "part_lookup" in analysis.query_types:
        routes.append({**base_filters, "chunk_type": ["spec_record", "datasheet_record", "table_record"]})
    if "revision_history" in analysis.query_types:
        routes.append({**base_filters, "chunk_type": ["section_window", "parent_section"], "version_signal": "true"})
    if "brochure_claim" in analysis.query_types:
        routes.append({**base_filters, "chunk_type": ["brochure_fact", "datasheet_record", "spec_record"]})
    if analysis.error_code:
        routes.append({**base_filters, "keywords": [analysis.error_code]})
    seen: set[tuple[tuple[str, str], ...]] = set()
    deduped: list[dict[str, object]] = []
    for route in routes:
        fingerprint_parts = []
        for key, value in route.items():
            if isinstance(value, list):
                fingerprint_parts.append((key, ",".join(sorted(str(item) for item in value))))
            else:
                fingerprint_parts.append((key, str(value)))
        fingerprint = tuple(sorted(fingerprint_parts))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(route)
    return deduped


def run_dense_search(store: QdrantStore, query: str, corpus_ids: list[str], filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
    results: list[SearchResult] = []
    for corpus_id in corpus_ids:
        try:
            results.extend(store.search_dense(corpus_id=corpus_id, query=query, filters=filters, limit=limit))
        except Exception as exc:
            logger.warning("Dense search skipped for corpus_id=%s after embedding/search failure: %s", corpus_id, exc)
    return results


def run_sparse_search(store: QdrantStore, query: str, corpus_ids: list[str], filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
    results: list[SearchResult] = []
    for corpus_id in corpus_ids:
        results.extend(store.search_sparse(corpus_id=corpus_id, query=query, filters=filters, limit=limit))
    return results


def run_table_search(store: QdrantStore, query: str, corpus_ids: list[str], filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
    table_filters = {**filters, "chunk_type": ["table_record"]}
    results: list[SearchResult] = []
    for corpus_id in corpus_ids:
        try:
            dense_results = store.search_dense(corpus_id=corpus_id, query=query, filters=table_filters, limit=limit)
        except Exception as exc:
            logger.warning("Table dense search skipped for corpus_id=%s after embedding/search failure: %s", corpus_id, exc)
            dense_results = []
        sparse_results = store.search_sparse(corpus_id=corpus_id, query=query, filters=table_filters, limit=limit)
        results.extend(store.fuse_rrf([dense_results, sparse_results], limit=limit))
    return results


def _should_run_table_search(analysis: QueryAnalysis) -> bool:
    if "table_record" in analysis.preferred_chunk_types:
        return True
    if {"structured_lookup", "spec_lookup", "part_lookup", "comparison", "compatibility"}.intersection(analysis.query_types):
        return True
    if analysis.safety_intent or {"how_to", "configuration", "operational_flow", "troubleshooting"}.intersection(analysis.query_types):
        return False
    return True


def run_special_search(
    store: QdrantStore,
    query: str,
    corpus_ids: list[str],
    base_filters: dict[str, object],
    analysis: QueryAnalysis,
    limit: int = 40,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for route_filters in _special_route_filters(base_filters, analysis):
        for corpus_id in corpus_ids:
            dense_results = store.search_dense(corpus_id=corpus_id, query=query, filters=route_filters, limit=limit)
            sparse_results = store.search_sparse(corpus_id=corpus_id, query=query, filters=route_filters, limit=limit)
            results.extend(store.fuse_rrf([dense_results, sparse_results], limit=limit))
    return results


def fuse_results(store: QdrantStore, result_sets: list[list[SearchResult]], *, limit: int = 30) -> list[SearchResult]:
    return store.fuse_rrf(result_sets, limit=limit)


def _query_terms(analysis: QueryAnalysis) -> set[str]:
    normalized_terms = getattr(analysis, "normalized_terms", None)
    if normalized_terms is None:
        return set()
    return _term_variants(normalized_terms)


def _term_variants(terms: Iterable[str]) -> set[str]:
    variants: set[str] = set()
    for term in terms:
        token = str(term).strip().lower()
        if not token:
            continue
        variants.add(token)
        compact = re.sub(r"[-/.]", "", token)
        if compact:
            variants.add(compact)
        if not any(char.isdigit() for char in token):
            for piece in re.split(r"[-/.]", token):
                if piece:
                    variants.add(piece)
    return variants


def _text_terms(text: str) -> set[str]:
    return _term_variants(tokenize(text))


def _compact_identifier(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _lexical_table_terms(query: str, analysis: QueryAnalysis) -> list[str]:
    if "structured_lookup" not in analysis.query_types:
        return []
    terms: list[str] = []
    for term in tokenize(query):
        normalized = re.sub(r"[^a-z0-9]+", "", term.lower())
        if len(normalized) < 4 or normalized in LEXICAL_TABLE_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms[:16]


def _lexical_table_content_terms(terms: list[str]) -> list[str]:
    content_terms = [
        term
        for term in terms
        if not any(char.isdigit() for char in term)
        and term not in {"corrective", "corrected", "remedy", "cause"}
    ]
    return sorted(content_terms, key=lambda term: (len(term), term), reverse=True)[:4]


def _structured_prompt_phrase(query: str) -> str:
    match = re.search(
        r"\bwhat\s+causes?\s+(?P<phrase>.+?)\s+for\s+.+?,\s+and\s+(?:what|how)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    phrase = re.sub(r"\s+", " ", match.group("phrase")).strip(" .,:;")
    return re.sub(r"[^a-z0-9]+", "", phrase.lower())


def _table_lexical_score(row: dict[str, object], terms: list[str], prompt_phrase: str = "") -> float:
    content = str(row.get("content") or "")
    metadata = dict(row.get("metadata_json") or {})
    haystack = " ".join(
        str(part)
        for part in [
            content,
            metadata.get("product_model"),
            metadata.get("product_family"),
            " ".join(str(item) for item in metadata.get("product_models") or []),
            " ".join(str(item) for item in metadata.get("devices") or []),
            " ".join(str(item) for item in metadata.get("table_column_headers") or []),
            " ".join(str(item) for item in metadata.get("table_row_headers") or []),
        ]
        if part
    ).lower()
    compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
    overlap = sum(1 for term in terms if term in compact_haystack)
    if overlap <= 0:
        return 0.0
    score = overlap * 0.12
    if prompt_phrase and prompt_phrase in compact_haystack:
        score += 1.0
    if metadata.get("table_row_group"):
        score += 0.28
    if metadata.get("table_cell"):
        score += 0.06
    if re.search(r"\b(?:cause|remedy|corrective action|error messages?|symptom)\b", content, flags=re.IGNORECASE):
        score += 0.16
    return score + float(row.get("priority_score") or 0.0) / 100.0


def run_table_lexical_search(
    query: str,
    corpus_ids: list[str],
    filters: dict[str, object],
    analysis: QueryAnalysis,
    limit: int = LEXICAL_TABLE_ROW_LIMIT,
) -> list[SearchResult]:
    terms = _lexical_table_terms(query, analysis)
    if not terms:
        return []
    prompt_phrase = _structured_prompt_phrase(query)
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
            where.append(f"metadata_json->>%s = any(%s)")
            params.extend([key, [str(item) for item in values]])
    required_terms = _lexical_table_content_terms(terms)
    if required_terms:
        where.extend(["content ilike %s"] * len(required_terms))
        params.extend([f"%{term}%" for term in required_terms])
    else:
        like_terms = terms[:8]
        where.append("(" + " or ".join(["content ilike %s"] * len(like_terms)) + ")")
        params.extend([f"%{term}%" for term in like_terms])
    rows = fetch_all(
        f"""
        select id, document_version_id, source_document_id, title, section_path_text,
               page_from, page_to, content, metadata_json, priority_score
        from retrieval_chunks
        where {" and ".join(where)}
        limit {LEXICAL_TABLE_SCAN_LIMIT}
        """,
        tuple(params),
    )
    ranked: list[tuple[float, SearchResult]] = []
    for row in rows:
        score = _table_lexical_score(dict(row), terms, prompt_phrase)
        if score <= 0:
            continue
        metadata = {
            **dict(row.get("metadata_json") or {}),
            "chunk_id": str(row["id"]),
            "document_version_id": str(row["document_version_id"]),
            "source_document_id": str(row["source_document_id"]),
            "chunk_type": "table_record",
            "chunk_level": 1,
            "title": str(row["title"]),
            "page_from": int(row["page_from"]),
            "page_to": int(row["page_to"]),
            "content": str(row["content"]),
            "content_for_rerank": str(row["content"]),
            "priority_score": float(row.get("priority_score") or 0.0),
        }
        section_path = metadata.get("section_path")
        if not isinstance(section_path, list) or not section_path:
            section_path = [str(row["section_path_text"])]
        ranked.append(
            (
                score,
                SearchResult(
                    chunk_id=str(row["id"]),
                    score=score,
                    title=str(row["title"]),
                    document_version_id=str(row["document_version_id"]),
                    source_document_id=str(row["source_document_id"]),
                    pages=list(range(int(row["page_from"]), int(row["page_to"]) + 1)),
                    section_path=[str(part) for part in section_path],
                    content=str(row["content"]),
                    metadata=metadata,
                ),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked[:limit]]


def _lexical_context_terms(query: str, analysis: QueryAnalysis) -> list[str]:
    if not {"how_to", "configuration", "operational_flow"}.intersection(analysis.query_types):
        return []
    terms: list[str] = []
    for term in tokenize(query):
        normalized = re.sub(r"[^a-z0-9]+", "", term.lower())
        if len(normalized) < 4 or normalized in LEXICAL_CONTEXT_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms[:18]


def _lexical_context_content_terms(terms: list[str]) -> list[str]:
    product_terms = [term for term in terms if any(char.isdigit() for char in term)]
    content_terms = sorted(
        [term for term in terms if not any(char.isdigit() for char in term)],
        key=lambda term: (len(term), term),
        reverse=True,
    )
    return [*product_terms[:3], *content_terms[:6]]


def _context_lexical_score(row: dict[str, object], terms: list[str]) -> float:
    metadata = dict(row.get("metadata_json") or {})
    content = str(row.get("content") or "")
    local_context = str(metadata.get("local_rerank_context") or "")
    haystack = " ".join(
        str(part)
        for part in [
            content,
            local_context,
            metadata.get("product_model"),
            metadata.get("product_family"),
            " ".join(str(item) for item in metadata.get("product_models") or []),
            " ".join(str(item) for item in metadata.get("devices") or []),
            " ".join(str(item) for item in metadata.get("settings") or []),
            " ".join(str(item) for item in metadata.get("parameters") or []),
            " ".join(str(item) for item in metadata.get("document_protocol_terms") or []),
        ]
        if part
    ).lower()
    compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
    overlap = sum(1 for term in terms if term in compact_haystack)
    if overlap <= 0:
        return 0.0
    score = overlap * 0.1
    product_overlap = sum(1 for term in terms if any(char.isdigit() for char in term) and term in compact_haystack)
    score += min(0.28, product_overlap * 0.14)
    family_overlap = sum(1 for term in terms if term in _text_terms(str(metadata.get("product_family") or "")))
    score += min(0.24, family_overlap * 0.08)
    chunk_type = str(row.get("chunk_type") or "")
    if chunk_type == "procedure_record":
        score += 0.24
    elif chunk_type == "section_window":
        score += 0.2
    elif chunk_type == "atomic_text":
        score += 0.14
    if local_context:
        context_overlap = sum(1 for term in terms if term in re.sub(r"[^a-z0-9]+", "", local_context.lower()))
        score += min(0.24, context_overlap * 0.04)
    return score + float(row.get("priority_score") or 0.0) / 100.0


def run_contextual_lexical_search(
    query: str,
    corpus_ids: list[str],
    filters: dict[str, object],
    analysis: QueryAnalysis,
    limit: int = LEXICAL_CONTEXT_LIMIT,
) -> list[SearchResult]:
    terms = _lexical_context_terms(query, analysis)
    if not terms:
        return []
    where = [
        "chunk_type = any(%s)",
        "is_active = true",
        "metadata_json->>'corpus_id' = any(%s)",
    ]
    params: list[object] = [["procedure_record", "atomic_text", "section_window"], corpus_ids]
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
    product_terms = [term for term in terms if any(char.isdigit() for char in term)]
    if product_terms:
        where.append(
            "("
            + " or ".join(
                [
                    "content ilike %s or metadata_json->>'local_rerank_context' ilike %s or metadata_json::text ilike %s"
                ]
                * min(3, len(product_terms))
            )
            + ")"
        )
        for term in product_terms[:3]:
            params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
    like_terms = _lexical_context_content_terms(terms) or terms[:8]
    where.append(
        "("
        + " or ".join(["content ilike %s or metadata_json->>'local_rerank_context' ilike %s"] * len(like_terms))
        + ")"
    )
    for term in like_terms:
        params.extend([f"%{term}%", f"%{term}%"])
    order_terms: list[str] = []
    for term in [*product_terms[:2], *(term for term in like_terms if not any(char.isdigit() for char in term))]:
        if term not in order_terms:
            order_terms.append(term)
        if len(order_terms) >= 6:
            break
    order_fragments: list[str] = []
    order_params: list[object] = []
    for term in order_terms:
        if any(char.isdigit() for char in term):
            order_fragments.append(
                "(case when metadata_json::text ilike %s then 2 else 0 end "
                "+ case when content ilike %s or metadata_json->>'local_rerank_context' ilike %s then 1 else 0 end)"
            )
            order_params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
        else:
            order_fragments.append(
                "(case when content ilike %s or metadata_json->>'local_rerank_context' ilike %s then 1 else 0 end)"
            )
            order_params.extend([f"%{term}%", f"%{term}%"])
    order_by = ""
    if order_fragments:
        order_by = "order by " + " + ".join(order_fragments) + " desc, priority_score desc, id"
    rows = fetch_all(
        f"""
        select id, document_version_id, source_document_id, title, section_path_text,
               page_from, page_to, content, chunk_type, metadata_json, priority_score
        from retrieval_chunks
        where {" and ".join(where)}
        {order_by}
        limit {LEXICAL_CONTEXT_SCAN_LIMIT}
        """,
        tuple([*params, *order_params]),
    )
    ranked: list[tuple[float, SearchResult]] = []
    for row in rows:
        score = _context_lexical_score(dict(row), terms)
        if score <= 0:
            continue
        metadata = {
            **dict(row.get("metadata_json") or {}),
            "chunk_id": str(row["id"]),
            "document_version_id": str(row["document_version_id"]),
            "source_document_id": str(row["source_document_id"]),
            "chunk_type": str(row["chunk_type"]),
            "title": str(row["title"]),
            "page_from": int(row["page_from"]),
            "page_to": int(row["page_to"]),
            "content": str(row["content"]),
            "content_for_rerank": str(row["content"]),
            "priority_score": float(row.get("priority_score") or 0.0),
        }
        section_path = metadata.get("section_path")
        if not isinstance(section_path, list) or not section_path:
            section_path = [str(row["section_path_text"])]
        ranked.append(
            (
                score,
                SearchResult(
                    chunk_id=str(row["id"]),
                    score=score,
                    title=str(row["title"]),
                    document_version_id=str(row["document_version_id"]),
                    source_document_id=str(row["source_document_id"]),
                    pages=list(range(int(row["page_from"]), int(row["page_to"]) + 1)),
                    section_path=[str(part) for part in section_path],
                    content=str(row["content"]),
                    metadata=metadata,
                ),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked[:limit]]


def _query_has_explicit_structure(analysis: QueryAnalysis) -> bool:
    structured_types = {"spec_lookup", "structured_lookup", "part_lookup", "comparison", "compatibility", "revision_history"}
    return bool(structured_types.intersection(analysis.query_types))


def _query_prefers_narrative_prose(analysis: QueryAnalysis) -> bool:
    return (
        "general" in analysis.query_types
        or "operational_flow" in analysis.query_types
        or "troubleshooting" in analysis.query_types
    ) and not _query_has_explicit_structure(analysis)


def _family_score_adjustment(result: SearchResult, analysis: QueryAnalysis) -> float:
    chunk_type = str(result.metadata.get("chunk_type", ""))
    query_terms = _query_terms(analysis)
    adjustment = 0.0
    if "spec_lookup" in analysis.query_types:
        if chunk_type in {"spec_record", "datasheet_record", "table_record"}:
            adjustment += 0.06
        elif chunk_type == "atomic_text":
            adjustment -= 0.01
    if "structured_lookup" in analysis.query_types:
        if chunk_type == "table_record":
            adjustment += 0.07
            if result.metadata.get("table_header"):
                adjustment -= 0.14
        elif chunk_type in {"spec_record", "datasheet_record"}:
            adjustment += 0.04
        elif chunk_type == "section_window":
            adjustment += 0.02
    if "how_to" in analysis.query_types or "configuration" in analysis.query_types:
        if chunk_type == "procedure_record":
            adjustment += 0.06
        elif chunk_type == "section_window":
            adjustment += 0.03
    if "comparison" in analysis.query_types or "compatibility" in analysis.query_types:
        if chunk_type in {"spec_record", "datasheet_record", "table_record"}:
            adjustment += 0.04
    if "revision_history" in analysis.query_types and str(result.metadata.get("version_signal", "false")) == "true":
        adjustment += 0.05
    if result.metadata.get("menu_labels") and any(term in " ".join(result.metadata.get("menu_labels", [])).lower() for term in query_terms):
        adjustment += 0.03
    if result.metadata.get("protocol_terms") and any(term in set(result.metadata.get("protocol_terms", [])) for term in query_terms):
        adjustment += 0.03
    document_protocol_terms = {str(term).lower() for term in result.metadata.get("document_protocol_terms", [])}
    if document_protocol_terms and any(term in document_protocol_terms for term in query_terms):
        adjustment += 0.02
    if result.metadata.get("identifier_tokens") and any(term in " ".join(result.metadata.get("identifier_tokens", [])).lower() for term in query_terms):
        adjustment += 0.02
    document_identifiers = " ".join(
        str(term)
        for term in [
            result.metadata.get("product_model", ""),
            result.metadata.get("product_family", ""),
            *(result.metadata.get("product_models") or []),
            *(result.metadata.get("product_families") or []),
            *(result.metadata.get("devices") or []),
            *(result.metadata.get("part_numbers") or []),
            *(result.metadata.get("settings") or []),
            *(result.metadata.get("parameters") or []),
        ]
    ).lower()
    if document_identifiers:
        identifier_terms = _text_terms(document_identifiers)
        identifier_overlap = query_terms.intersection(identifier_terms)
        if identifier_overlap:
            product_identifier_overlap = {
                term
                for term in identifier_overlap
                if any(char.isdigit() for char in term) or "-" in term or "/" in term
            }
            adjustment += min(0.08, 0.02 * len(identifier_overlap))
            adjustment += min(0.18, 0.09 * len(product_identifier_overlap))
        if analysis.product_model:
            expected_model = _compact_identifier(analysis.product_model)
            result_models = [
                result.metadata.get("product_model", ""),
                *(result.metadata.get("product_models") or []),
                *(result.metadata.get("devices") or []),
                *(result.metadata.get("part_numbers") or []),
            ]
            compact_models = {_compact_identifier(str(item)) for item in result_models if str(item)}
            if expected_model and expected_model in compact_models:
                adjustment += 0.32
            elif compact_models and expected_model and any(
                model.startswith(expected_model[:3]) or expected_model.startswith(model[:3])
                for model in compact_models
                if len(model) >= 3
            ):
                adjustment -= 0.12
    if _query_prefers_narrative_prose(analysis):
        if chunk_type in {"atomic_text", "section_window", "parent_section"}:
            adjustment += 0.03
        elif chunk_type in {"spec_record", "datasheet_record", "table_record"}:
            adjustment -= 0.05
    if chunk_type == "atomic_text" and len(str(result.content).split()) <= 3:
        adjustment -= 0.04
    return adjustment


def _apply_family_scoring(results: list[SearchResult], analysis: QueryAnalysis, *, stage: str) -> list[SearchResult]:
    rescored = [
        result.model_copy(
            update={
                "score": result.score + _family_score_adjustment(result, analysis),
                "metadata": {**result.metadata, "retrieval_stage": stage},
            }
        )
        for result in results
    ]
    return sorted(rescored, key=lambda item: item.score, reverse=True)


def _family_bucket(result: SearchResult) -> str:
    chunk_type = str(result.metadata.get("chunk_type", ""))
    if chunk_type in {"spec_record", "datasheet_record"}:
        return "spec"
    if chunk_type == "table_record":
        return "table"
    if chunk_type == "procedure_record":
        return "procedure"
    if chunk_type in {"section_window", "parent_section"}:
        return "context"
    return "prose"


def _preferred_family_order(analysis: QueryAnalysis) -> list[str]:
    if "revision_history" in analysis.query_types:
        return ["context", "spec", "table", "prose"]
    if "structured_lookup" in analysis.query_types:
        return ["table", "spec", "context", "prose"]
    if "how_to" in analysis.query_types or "configuration" in analysis.query_types:
        return ["procedure", "context", "prose", "table"]
    if "operational_flow" in analysis.query_types:
        return ["context", "procedure", "prose", "table"]
    if "spec_lookup" in analysis.query_types or "part_lookup" in analysis.query_types:
        return ["spec", "table", "prose", "context"]
    if "comparison" in analysis.query_types or "compatibility" in analysis.query_types:
        return ["spec", "table", "context", "prose"]
    return ["prose", "context", "spec", "table"]


def _allowed_families(analysis: QueryAnalysis) -> set[str]:
    if "revision_history" in analysis.query_types:
        return {"context", "spec"}
    if "structured_lookup" in analysis.query_types:
        return {"table", "spec", "context"}
    if "how_to" in analysis.query_types or "configuration" in analysis.query_types:
        return {"procedure", "context", "prose"}
    if "operational_flow" in analysis.query_types:
        return {"prose", "context", "procedure"}
    if "spec_lookup" in analysis.query_types or "part_lookup" in analysis.query_types:
        return {"spec", "table", "context"}
    if "comparison" in analysis.query_types or "compatibility" in analysis.query_types:
        return {"spec", "table", "context"}
    return {"prose", "context"}


def _families_from_chunk_type_filter(filters: dict[str, object]) -> tuple[list[str], set[str]]:
    chunk_type_filter = filters.get("chunk_type")
    if not chunk_type_filter:
        return [], set()
    chunk_types = chunk_type_filter if isinstance(chunk_type_filter, list) else [chunk_type_filter]
    family_order: list[str] = []
    for chunk_type in chunk_types:
        family = {
            "spec_record": "spec",
            "datasheet_record": "spec",
            "table_record": "table",
            "procedure_record": "procedure",
            "section_window": "context",
            "parent_section": "context",
            "atomic_text": "prose",
        }.get(str(chunk_type))
        if family and family not in family_order:
            family_order.append(family)
    return family_order, set(family_order)


def _has_value_pattern(text: str) -> bool:
    return bool(
        re.search(
            r":\s*\S+|\b\d+(?:\.\d+)?\s?(?:v|a|ma|w|kw|mm|cm|m|ms|s|hz|khz|mhz|fps|kg|g|n|mpa|deg|c|°c|%)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _semantic_completeness_score(result: SearchResult) -> float:
    content = str(result.content or "").strip()
    chunk_type = str(result.metadata.get("chunk_type", ""))
    tokens = content.split()
    token_count = len(tokens)
    lowered = content.lower()
    score = 0.0
    if token_count >= 8:
        score += 0.35
    elif token_count >= 5:
        score += 0.15
    if _has_value_pattern(content):
        score += 0.35
    if any(char.isdigit() for char in content):
        score += 0.1
    if any(mark in content for mark in [".", ";", ":"]):
        score += 0.1
    if chunk_type in {"spec_record", "table_record", "procedure_record"}:
        score += 0.15
    heading_like = token_count <= 6 and not _has_value_pattern(content) and not re.search(r"\b(is|are|can|does|set|check|configure|measure|connect|select|display|returns?)\b", lowered)
    if heading_like:
        score -= 0.4
    if token_count <= 3:
        score -= 0.25
    return score


def _annotate_completeness(results: list[SearchResult]) -> list[SearchResult]:
    annotated: list[SearchResult] = []
    for result in results:
        completeness = _semantic_completeness_score(result)
        metadata = {
            **result.metadata,
            "family_bucket": _family_bucket(result),
            "semantic_completeness_score": completeness,
            "is_heading_like": completeness < 0.0,
            "is_value_bearing": _has_value_pattern(str(result.content or "")),
        }
        annotated.append(result.model_copy(update={"metadata": metadata, "score": result.score + completeness * 0.05}))
    return annotated


def _query_alignment_score(result: SearchResult, analysis: QueryAnalysis) -> float:
    query_terms = _query_terms(analysis)
    if not query_terms:
        return 0.0
    content_terms = _text_terms(str(result.content or ""))
    title_terms = _text_terms(str(result.title or ""))
    section_terms = _text_terms(" ".join(result.section_path))
    rerank_terms = _text_terms(str(result.metadata.get("rerank_document") or result.metadata.get("content_for_rerank") or ""))
    content_overlap = len(query_terms.intersection(content_terms))
    title_overlap = len(query_terms.intersection(title_terms))
    section_overlap = len(query_terms.intersection(section_terms))
    rerank_overlap = len(query_terms.intersection(rerank_terms))
    alignment = content_overlap * 0.045 + title_overlap * 0.015 + section_overlap * 0.025 + rerank_overlap * 0.02
    if len(query_terms) >= 2 and content_overlap == 0 and rerank_overlap <= 1:
        alignment -= 0.04
    if len(query_terms) >= 3 and max(content_overlap, rerank_overlap) >= 3:
        alignment += 0.03
    chunk_type = str(result.metadata.get("chunk_type", ""))
    if "spec_lookup" in analysis.query_types and "laser" in query_terms and chunk_type in {"spec_record", "warning_record"}:
        laser_safety_terms = {"radiation", "class", "wavelength", "output"}
        safety_overlap = len(laser_safety_terms.intersection(content_terms.union(rerank_terms)))
        if safety_overlap >= 2:
            alignment += 0.18
    if analysis.safety_intent and "how_to" in analysis.query_types:
        action_terms = _safety_action_terms(analysis.raw_query)
        if action_terms:
            action_overlap = len(action_terms.intersection(content_terms.union(rerank_terms)))
            if chunk_type in {"atomic_text", "procedure_record", "warning_record"} and action_overlap >= 2:
                alignment += min(0.16, action_overlap * 0.04)
            elif chunk_type in {"section_window", "parent_section"} and action_overlap <= 1:
                alignment -= 0.06
    return alignment


def _safety_action_terms(query: str) -> set[str]:
    match = re.search(r"\bapplies\s+when\s+(?P<action>.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if not match:
        return set()
    stopwords = LEXICAL_CONTEXT_STOPWORDS.union(
        {
            "applies",
            "caution",
            "warning",
            "should",
            "about",
        }
    )
    terms: set[str] = set()
    for term in tokenize(match.group("action")):
        normalized = re.sub(r"[^a-z0-9]+", "", term.lower())
        if len(normalized) < 4 or normalized in stopwords:
            continue
        terms.add(normalized)
    return terms


def _apply_query_alignment(results: list[SearchResult], analysis: QueryAnalysis, *, stage: str) -> list[SearchResult]:
    aligned = [
        result.model_copy(
            update={
                "score": result.score + _query_alignment_score(result, analysis),
                "metadata": {**result.metadata, "retrieval_stage": stage},
            }
        )
        for result in results
    ]
    return sorted(aligned, key=lambda item: item.score, reverse=True)


def _select_family_candidates(
    results: list[SearchResult],
    analysis: QueryAnalysis,
    *,
    filters: dict[str, object] | None = None,
    limit: int = 12,
) -> list[SearchResult]:
    if not results:
        return []
    requested_order, requested_families = _families_from_chunk_type_filter(filters or {})
    family_order = requested_order or _preferred_family_order(analysis)
    allowed_families = requested_families or _allowed_families(analysis)
    buckets: dict[str, list[SearchResult]] = {}
    chosen: list[SearchResult] = results[: min(3, limit)] if analysis.query_types == ["general"] else []
    for result in results:
        family = str(result.metadata.get("family_bucket") or _family_bucket(result))
        if family not in allowed_families:
            continue
        buckets.setdefault(family, []).append(result)
    primary_family = next((family for family in family_order if buckets.get(family)), None)
    if primary_family:
        chosen.extend(buckets[primary_family][: max(6, limit // 2)])
    fallback_limit = max(4, limit // 2)
    for fallback_family in family_order:
        if fallback_family == primary_family or fallback_family not in allowed_families or not buckets.get(fallback_family):
            continue
        chosen.extend(buckets[fallback_family][:fallback_limit])
    if not chosen:
        chosen = results[:limit]
    deduped: dict[str, SearchResult] = {}
    for result in chosen:
        deduped[result.chunk_id] = result
    return list(deduped.values())[: max(limit, 10)]


def _resolve_rerank_device() -> object | None:
    configured_device = settings.haystack_rerank_device.strip().lower()
    if configured_device == "auto":
        configured_device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
    if not configured_device:
        return None
    if configured_device == "cpu":
        return None
    if configured_device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        logger.warning("CUDA rerank device requested but CUDA is unavailable; falling back to CPU")
        return None
    try:
        from haystack.utils import ComponentDevice

        return ComponentDevice.from_str(configured_device)
    except Exception as exc:
        logger.warning("Failed to configure rerank device %s; falling back to CPU: %s", configured_device, exc)
        return None


@lru_cache(maxsize=1)
def _get_reranker() -> SentenceTransformersSimilarityRanker:
    if SentenceTransformersSimilarityRanker is None:
        raise RuntimeError("Haystack sentence-transformers ranker is unavailable") from _RERANK_IMPORT_ERROR
    ranker = SentenceTransformersSimilarityRanker(
        model=settings.haystack_rerank_model,
        device=_resolve_rerank_device(),
        top_k=settings.haystack_rerank_top_k,
    )
    ranker.warm_up()
    return ranker


def _to_rerank_documents(results: list[SearchResult]) -> list[Document]:
    if Document is None:
        raise RuntimeError("Haystack Document class is unavailable") from _RERANK_IMPORT_ERROR
    return [
        Document(
            id=result.chunk_id,
            content=str(result.metadata.get("rerank_document") or result.metadata.get("content_for_rerank") or result.content),
            meta={"chunk_id": result.chunk_id, "pre_rerank_rank": result.metadata.get("stage_rank")},
            score=result.score,
        )
        for result in results
    ]


def rerank_results(results: list[SearchResult], query: str, *, limit: int = 12) -> list[SearchResult]:
    if not results:
        return []
    candidate_results = results[: max(limit, settings.haystack_rerank_top_k)]
    try:
        analysis = analyze_query(query)
        documents = _to_rerank_documents(candidate_results)
        reranked_documents = _get_reranker().run(query=query, documents=documents)["documents"]
        result_by_id = {result.chunk_id: result for result in candidate_results}
        reranked_results: list[SearchResult] = []
        for post_rank, document in enumerate(reranked_documents, start=1):
            chunk_id = document.meta.get("chunk_id", document.id)
            result = result_by_id.get(str(chunk_id))
            if result is None:
                continue
            rerank_score = float(document.score or result.score)
            alignment_score = _query_alignment_score(result, analysis)
            blended_score = rerank_score + result.score * 0.35 + alignment_score * 7.0
            reranked_results.append(
                result.model_copy(
                    update={
                        "score": blended_score,
                        "metadata": {
                            **result.metadata,
                            "rerank_score": rerank_score,
                            "rerank_query_alignment": alignment_score,
                            "pre_rerank_rank": document.meta.get("pre_rerank_rank"),
                            "post_rerank_rank": post_rank,
                        },
                    }
                )
            )
        for result in candidate_results:
            alignment_score = _query_alignment_score(result, analysis)
            if alignment_score < 0.1:
                continue
            reranked_results.append(
                result.model_copy(
                    update={
                        "score": result.score + alignment_score * 7.0,
                        "metadata": {
                            **result.metadata,
                            "rerank_score": None,
                            "rerank_query_alignment": alignment_score,
                            "pre_rerank_rank": result.metadata.get("stage_rank"),
                            "post_rerank_rank": None,
                            "rerank_preserved_by_alignment": True,
                        },
                    }
                )
            )
        if reranked_results:
            query_terms = _query_terms(analysis)
            if query_terms and len(query_terms) >= 2 and any(
                result.metadata.get("rerank_query_alignment", 0.0) > 0 for result in reranked_results
            ):
                aligned_results = [
                    result for result in reranked_results if result.metadata.get("rerank_query_alignment", 0.0) > 0
                ]
                if aligned_results:
                    reranked_results = aligned_results
                reranked_results.sort(
                    key=lambda item: (
                        item.metadata.get("rerank_query_alignment", 0.0) > 0,
                        item.score,
                    ),
                    reverse=True,
                )
            deduped_reranked: dict[str, SearchResult] = {}
            for result in reranked_results:
                if result.chunk_id not in deduped_reranked or result.score > deduped_reranked[result.chunk_id].score:
                    deduped_reranked[result.chunk_id] = result
            reranked_results = sorted(deduped_reranked.values(), key=lambda item: item.score, reverse=True)
            return reranked_results[:limit]
    except Exception as exc:
        logger.warning("Haystack rerank failed; falling back to fused order: %s", exc)
    return candidate_results[:limit]


def _chunk_group_key(result: SearchResult, analysis: QueryAnalysis) -> str:
    if analysis.latest_only:
        return "|".join([result.chunk_id, result.document_version_id])
    return result.chunk_id


def _section_path_has_meaningful_signal(section_path: list[str]) -> bool:
    normalized = [part.strip() for part in section_path if part and part.strip()]
    if not normalized:
        return False
    return any(len(re.findall(r"[A-Za-z0-9]", part)) >= 3 for part in normalized)


def _dedupe_results(results: Iterable[SearchResult], analysis: QueryAnalysis) -> list[SearchResult]:
    deduped: dict[str, SearchResult] = {}
    for result in results:
        group_key = _chunk_group_key(result, analysis)
        if group_key not in deduped or result.score > deduped[group_key].score:
            deduped[group_key] = result
    return list(deduped.values())


def _annotate_stage_metadata(results: list[SearchResult], stage: str) -> list[SearchResult]:
    annotated: list[SearchResult] = []
    for index, result in enumerate(results, start=1):
        metadata = {**result.metadata, "retrieval_stage": stage, "stage_rank": index}
        annotated.append(result.model_copy(update={"metadata": metadata}))
    return annotated


def _nearest_lineage_chunk(
    result: SearchResult,
    candidates: list[dict[str, object]],
    *,
    require_flag: str | None = None,
) -> dict[str, object] | None:
    matches = [
        candidate
        for candidate in candidates
        if require_flag is None or bool(candidate.get(require_flag))
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda candidate: abs(int(candidate["page_from"]) - min(result.pages)) + abs(int(candidate["page_to"]) - max(result.pages)),
    )


def enrich_candidates_for_rerank(results: list[SearchResult], analysis: QueryAnalysis, *, limit: int = 30) -> list[SearchResult]:
    if not results:
        return []
    sections = {(result.document_version_id, " / ".join(result.section_path) or "Document") for result in results[:limit]}
    placeholders = ",".join(["(%s,%s)"] * len(sections))
    params: list[object] = []
    for document_version_id, section_path_text in sections:
        params.extend([document_version_id, section_path_text])
    rows = fetch_all(
        f"""
        select document_version_id, section_path_text, chunk_type, chunk_level, page_from, page_to, content, metadata_json
        from retrieval_chunks
        where (document_version_id, section_path_text) in ({placeholders})
        """,
        tuple(params),
    )
    section_map: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["document_version_id"]), str(row["section_path_text"]))
        section_map.setdefault(key, []).append(
            {
                "chunk_type": str(row["chunk_type"]),
                "chunk_level": int(row["chunk_level"]),
                "page_from": int(row["page_from"]),
                "page_to": int(row["page_to"]),
                "content": str(row["content"]),
                **dict(row.get("metadata_json") or {}),
            }
        )
    enriched: list[SearchResult] = []
    for result in results[:limit]:
        section_key = " / ".join(result.section_path) or "Document"
        section_rows = section_map.get((result.document_version_id, section_key), [])
        context_window = next((row["content"] for row in section_rows if int(row["chunk_level"]) == 2 and str(row["chunk_type"]) == "section_window"), None)
        parent_context = next((row["content"] for row in section_rows if int(row["chunk_level"]) == 3 and str(row["chunk_type"]) == "parent_section"), None)
        rerank_parts = [str(result.metadata.get("content_for_rerank") or result.content)]
        chunk_type = str(result.metadata.get("chunk_type", ""))
        if chunk_type == "atomic_text":
            if context_window and context_window not in rerank_parts:
                rerank_parts.append(context_window)
        elif chunk_type == "procedure_record":
            grouped = _nearest_lineage_chunk(result, section_rows, require_flag="grouped_procedure")
            if grouped and grouped["content"] not in rerank_parts:
                rerank_parts.append(str(grouped["content"]))
        elif chunk_type == "table_record":
            grouped = _nearest_lineage_chunk(result, section_rows, require_flag="table_row_group")
            if grouped and grouped["content"] not in rerank_parts:
                rerank_parts.append(str(grouped["content"]))
            summary = _nearest_lineage_chunk(result, section_rows, require_flag="table_summary")
            if summary and summary["content"] not in rerank_parts:
                rerank_parts.append(str(summary["content"]))
        if parent_context and ("revision_history" in analysis.query_types or chunk_type in {"procedure_record", "table_record"}):
            rerank_parts.append(parent_context)
        if chunk_type == "atomic_text" and context_window:
            rerank_parts = rerank_parts[:2]
        elif chunk_type == "spec_record":
            rerank_parts = rerank_parts[:1]
        rerank_document = "\n\n".join(part for part in rerank_parts if part)
        metadata = {
            **result.metadata,
            "context_window": context_window,
            "parent_context": parent_context,
            "rerank_document": rerank_document,
            "rerank_context_strategy": chunk_type,
        }
        enriched.append(result.model_copy(update={"metadata": metadata}))
    return enriched


def assemble_context(results: list[SearchResult], *, limit: int = 10) -> list[SearchResult]:
    if not results:
        return []
    base_results = results[:limit]
    sections = {
        (result.document_version_id, " / ".join(result.section_path) or "Document")
        for result in base_results
        if _section_path_has_meaningful_signal(result.section_path)
    }
    if not sections:
        return [
            result.model_copy(
                update={
                    "metadata": {
                        **result.metadata,
                        "context_window": None,
                        "parent_context": None,
                    }
                }
            )
            if not _section_path_has_meaningful_signal(result.section_path)
            else result
            for result in base_results
        ]
    placeholders = ",".join(["(%s,%s)"] * len(sections))
    params: list[object] = []
    for document_version_id, section_path_text in sections:
        params.extend([document_version_id, section_path_text])
    context_rows = fetch_all(
        f"""
        select document_version_id, section_path_text, chunk_type, chunk_level, page_from, page_to, content, metadata_json
        from retrieval_chunks
        where (document_version_id, section_path_text) in ({placeholders})
          and (
            chunk_level in (2, 3)
            or (chunk_level = 1 and chunk_type = 'table_record')
          )
        """,
        tuple(params),
    )
    context_map: dict[tuple[str, str], dict[int, str]] = {}
    table_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in context_rows:
        key = (str(row["document_version_id"]), str(row["section_path_text"]))
        chunk_level = int(row["chunk_level"])
        chunk_type = str(row["chunk_type"])
        metadata = dict(row.get("metadata_json") or {})
        if chunk_level in {2, 3}:
            bucket = context_map.setdefault(key, {})
            bucket[chunk_level] = str(row["content"])
        if chunk_level == 1 and chunk_type == "table_record" and metadata.get("table_row_group"):
            table_groups.setdefault(key, []).append(
                {
                    "content": str(row["content"]),
                    "page_from": int(row["page_from"]),
                    "page_to": int(row["page_to"]),
                    **metadata,
                }
            )
    assembled: list[SearchResult] = []
    for result in base_results:
        if not _section_path_has_meaningful_signal(result.section_path):
            metadata = {
                **result.metadata,
                "context_window": None,
                "parent_context": None,
            }
            assembled.append(result.model_copy(update={"metadata": metadata}))
            continue
        section_key = " / ".join(result.section_path) or "Document"
        key = (result.document_version_id, section_key)
        context = context_map.get(key, {})
        table_row_group_context = None
        if str(result.metadata.get("chunk_type") or "") == "table_record":
            table_group = _nearest_lineage_chunk(result, table_groups.get(key, []), require_flag="table_row_group")
            if table_group:
                table_row_group_context = str(table_group["content"])
        metadata = {
            **result.metadata,
            "context_window": table_row_group_context or context.get(2),
            "parent_context": context.get(3),
            "table_row_group_context": table_row_group_context,
        }
        assembled.append(result.model_copy(update={"metadata": metadata}))
    return assembled


def retrieve(query: str, corpus_ids: list[str], filters: dict[str, object], limit: int = 10) -> list[SearchResult]:
    store = QdrantStore()
    analysis = analyze_query(query)
    search_filters, metadata_document_hits = select_documents_from_metadata(store, query, corpus_ids, filters)
    dense_results = _annotate_stage_metadata(run_dense_search(store, query, corpus_ids, search_filters), "dense")
    sparse_results = _annotate_stage_metadata(run_sparse_search(store, query, corpus_ids, search_filters), "sparse")
    table_results = (
        _annotate_stage_metadata(run_table_search(store, query, corpus_ids, search_filters), "table")
        if _should_run_table_search(analysis)
        else []
    )
    table_lexical_results = _annotate_stage_metadata(run_table_lexical_search(query, corpus_ids, filters, analysis), "table_lexical")
    contextual_lexical_results = _annotate_stage_metadata(
        run_contextual_lexical_search(query, corpus_ids, filters, analysis),
        "contextual_lexical",
    )
    special_results = _annotate_stage_metadata(run_special_search(store, query, corpus_ids, search_filters, analysis), "special")
    fused = _annotate_stage_metadata(
        fuse_results(
            store,
            [dense_results, sparse_results, table_results, table_lexical_results, contextual_lexical_results, special_results],
            limit=FUSED_CANDIDATE_POOL_LIMIT,
        ),
        "fused",
    )
    rescored = _annotate_stage_metadata(_apply_family_scoring(fused, analysis, stage="family_scored")[:FUSED_CANDIDATE_POOL_LIMIT], "family_scored")
    completed = _annotate_stage_metadata(_annotate_completeness(rescored), "completeness_scored")
    aligned = _annotate_stage_metadata(_apply_query_alignment(completed, analysis, stage="query_aligned"), "query_aligned")
    family_selected = _annotate_stage_metadata(_select_family_candidates(aligned, analysis, filters=search_filters, limit=12), "family_selected")
    enriched = enrich_candidates_for_rerank(family_selected, analysis, limit=12)
    reranked = _annotate_stage_metadata(rerank_results(enriched, query, limit=12), "reranked")
    deduped = _dedupe_results(reranked, analysis)
    assembled = assemble_context(deduped, limit=limit)
    if not metadata_document_hits:
        return assembled
    document_selection = [
        {
            "source_document_id": hit.get("source_document_id"),
            "score": hit.get("score"),
            "retrieval_stage": hit.get("retrieval_stage"),
            "title": (hit.get("payload") or {}).get("title") if isinstance(hit.get("payload"), dict) else None,
            "source_filename": (hit.get("payload") or {}).get("source_filename") if isinstance(hit.get("payload"), dict) else None,
        }
        for hit in metadata_document_hits
    ]
    return [
        result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "document_selection_stage": "metadata_embedding",
                    "selected_document_metadata_hits": document_selection,
                }
            }
        )
        for result in assembled
    ]


def document_versions_for_results(results: list[SearchResult]) -> list[dict[str, object]]:
    version_ids = tuple(result.document_version_id for result in results)
    if not version_ids:
        return []
    placeholders = ",".join(["%s"] * len(version_ids))
    return fetch_all(
        f"""
        select dv.id, dv.version_label, dv.revision_date, sd.id as source_document_id, sd.title
        from document_versions dv
        join source_documents sd on sd.id = dv.source_document_id
        where dv.id in ({placeholders})
        """,
        version_ids,
    )
