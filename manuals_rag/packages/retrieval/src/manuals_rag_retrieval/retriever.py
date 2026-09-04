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
LEXICAL_TABLE_ROW_LIMIT = 48
LEXICAL_TABLE_SCAN_LIMIT = 1500
LEXICAL_CONTEXT_LIMIT = 24
LEXICAL_CONTEXT_SCAN_LIMIT = 1500
LEXICAL_TABLE_STOPWORDS = {
    "about",
    "after",
    "applies",
    "being",
    "built",
    "causes",
    "check",
    "checked",
    "confirm",
    "correspond",
    "corresponds",
    "corrected",
    "correction",
    "could",
    "does",
    "error",
    "following",
    "please",
    "indicate",
    "indicates",
    "series",
    "settings",
    "should",
    "system",
    "and",
    "for",
    "in",
    "is",
    "named",
    "the",
    "at",
    "temporarily",
    "there",
    "using",
    "value",
    "what",
    "when",
    "which",
    "vision",
    "with",
}
LEXICAL_CONTEXT_STOPWORDS = LEXICAL_TABLE_STOPWORDS.union(
    {
        "detail",
        "related",
        "used",
    }
)
LEXICAL_TABLE_FIELD_TERMS = {
    "average",
    "description",
    "message",
    "scaling",
    "specified",
    "summary",
    "symbol",
    "target",
}

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


def _chunk_search_filters(filters: dict[str, object], metadata_filters: dict[str, object], analysis: QueryAnalysis) -> dict[str, object]:
    if _has_explicit_document_scope(filters):
        return metadata_filters
    if "structured_lookup" in analysis.query_types and len(getattr(analysis, "product_identifiers", []) or []) >= 2:
        return filters
    if (
        "structured_lookup" in analysis.query_types
        and analysis.product_family
        and not analysis.product_model
        and not analysis.part_number
    ):
        return filters
    return metadata_filters


def _special_route_filters(base_filters: dict[str, object], analysis: QueryAnalysis) -> list[dict[str, object]]:
    routes: list[dict[str, object]] = []
    if "structured_lookup" in analysis.query_types:
        routes.append({**base_filters, "chunk_type": ["table_record", "spec_record", "section_window"]})
    elif analysis.preferred_chunk_types and not analysis.safety_intent:
        routes.append({**base_filters, "chunk_type": analysis.preferred_chunk_types})
    if analysis.safety_intent:
        safety_chunk_types = ["warning_record", "procedure_record"]
        if _safety_action_terms(analysis.raw_query):
            safety_chunk_types.append("atomic_text")
        routes.append({**base_filters, "chunk_type": safety_chunk_types})
    if (
        "structured_lookup" not in analysis.query_types
        and not analysis.safety_intent
        and ("how_to" in analysis.query_types or "configuration" in analysis.query_types)
    ):
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


def _should_run_broad_vector_search(analysis: QueryAnalysis) -> bool:
    return "structured_lookup" not in analysis.query_types


def _should_run_extra_table_vector_search(analysis: QueryAnalysis) -> bool:
    return _should_run_table_search(analysis)


def _should_run_table_lexical_search(analysis: QueryAnalysis) -> bool:
    diagnostic_codes = _diagnostic_table_code_terms(analysis.raw_query, analysis)
    if _query_has_diagnostic_table_code(analysis.raw_query, analysis) and (
        analysis.error_code
        or len(diagnostic_codes) >= 2
        or (
            bool(diagnostic_codes)
            and "comparison" in analysis.query_types
            and len(analysis.product_identifiers) >= 2
        )
    ):
        return True
    if "structured_lookup" not in analysis.query_types:
        return True
    if analysis.product_model or analysis.product_family or analysis.part_number:
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


def _query_product_identifier_terms(analysis: QueryAnalysis) -> set[str]:
    identifiers = getattr(analysis, "product_identifiers", []) or []
    if not identifiers:
        identifiers = [item for item in [analysis.product_model, analysis.product_family, analysis.part_number] if item]
    terms: set[str] = set()
    for identifier in identifiers:
        compact = _compact_identifier(str(identifier))
        if compact:
            terms.add(compact)
        terms.update(_text_terms(str(identifier)))
    return terms


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
    table_lookup = (
        "structured_lookup" in analysis.query_types
        or "table_record" in analysis.preferred_chunk_types
        or _is_natural_quantity_table_lookup(query, analysis)
    )
    if (
        not table_lookup
        and not ("comparison" in analysis.query_types and analysis.product_identifiers)
        and not _query_has_diagnostic_table_code(query, analysis)
    ):
        return []
    terms: list[str] = []
    query_tokens = tokenize(query)
    compact_tokens = [re.sub(r"[^a-z0-9]+", "", term.lower()) for term in query_tokens]
    for index, term in enumerate(query_tokens):
        normalized = re.sub(r"[^a-z0-9]+", "", term.lower())
        if re.search(r"\d\s*[-–]\s*\d", term) and normalized.isdigit():
            continue
        nearby_numeric = any(
            any(char.isdigit() for char in compact_tokens[neighbor_index])
            for neighbor_index in (index - 1, index + 1)
            if 0 <= neighbor_index < len(compact_tokens)
        )
        short_code_label = normalized in {"id", "pid", "pib", "plc", "ch"} and nearby_numeric
        if (
            len(normalized) < 4
            and not any(char.isdigit() for char in normalized)
            and not short_code_label
        ) or normalized in LEXICAL_TABLE_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
        if "comparison" in analysis.query_types:
            for piece in re.split(r"[-/_.]+", term.lower()):
                piece = re.sub(r"[^a-z0-9]+", "", piece)
                if (
                    (len(piece) >= 4 or any(char.isdigit() for char in piece))
                    and piece not in LEXICAL_TABLE_STOPWORDS
                    and piece not in terms
                ):
                    terms.append(piece)
    for start, end in re.findall(r"\b(\d{2,6})\s*[-–]\s*(\d{2,6})\b", query):
        range_terms = [start, end]
        if len(start) == len(end):
            start_int = int(start)
            end_int = int(end)
            if 0 < end_int - start_int <= 20:
                range_terms = [f"{value:0{len(start)}d}" for value in range(start_int, end_int + 1)]
        for term in range_terms:
            if term not in terms:
                terms.append(term)
    return terms[:16]


def _is_natural_quantity_table_lookup(query: str, analysis: QueryAnalysis) -> bool:
    lowered = query.lower()
    if "how many" not in lowered and not re.search(r"\bhow\s+much\b", lowered):
        return False
    if not {"how_to", "structured_lookup", "spec_lookup"}.intersection(analysis.query_types):
        return False
    return bool(
        re.search(r"\bcount(?:ed|s|ing)?\b", lowered)
        and re.search(r"\b(?:value|quantity|number|total|objects?|items?|lines?)\b", lowered)
    )


def _query_has_diagnostic_table_code(query: str, analysis: QueryAnalysis) -> bool:
    if not {"troubleshooting", "configuration"}.intersection(analysis.query_types):
        return False
    if "table_record" not in analysis.preferred_chunk_types:
        return False
    return bool(_diagnostic_table_code_terms(query, analysis))


def _diagnostic_table_code_terms(query: str, analysis: QueryAnalysis) -> list[str]:
    if not {"troubleshooting", "configuration"}.intersection(analysis.query_types):
        return []
    if "table_record" not in analysis.preferred_chunk_types:
        return []
    product_identifiers = {_compact_identifier(str(analysis.product_model or ""))}
    product_identifiers.update(_compact_identifier(str(item)) for item in getattr(analysis, "product_identifiers", []) or [])
    product_identifiers.discard("")
    model_spans = [
        match.span()
        for match in re.finditer(r"\b[A-Z]{1,5}\d{0,4}(?:-[A-Z0-9]{1,8})+\b", query)
        if any(char.isdigit() for char in match.group(0))
        and _compact_identifier(match.group(0)) in product_identifiers
        and (
            _compact_identifier(match.group(0)) == _compact_identifier(str(analysis.product_model or ""))
            or len(_compact_identifier(match.group(0))) > 4
        )
    ]
    codes: list[str] = []
    if analysis.error_code:
        normalized_error = _compact_identifier(str(analysis.error_code))
        if normalized_error:
            codes.append(normalized_error)
    for match in re.finditer(
        r"\b(?:error|alarm|fault)\s*(?:number|no\.?\s*)?[:#-]?\s*(\d{2,6})\b",
        query,
        flags=re.IGNORECASE,
    ):
        compact = _compact_identifier(match.group(1))
        if compact and compact not in codes:
            codes.append(compact)
    for match in re.finditer(r"\b[A-Z]{1,4}[- ]?\d{1,4}[A-Z]?\b", query, flags=re.IGNORECASE):
        if any(start <= match.start() and match.end() <= end for start, end in model_spans):
            continue
        compact = _compact_identifier(match.group(0))
        if len(compact) >= 2 and any(char.isdigit() for char in compact):
            if compact not in codes:
                codes.append(compact)
    return codes[:4]


def _diagnostic_code_pattern(code: str) -> str:
    compact = _compact_identifier(code)
    if not compact:
        return r"(?!)"
    separated = r"[^a-z0-9]*".join(re.escape(char) for char in compact)
    return rf"(^|[^a-z0-9]){separated}([^a-z0-9]|$)"


def _diagnostic_side_requirements(query: str, identifiers: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    """Bind explicit diagnostic codes and prose symptoms to their product clauses."""
    spans: list[tuple[int, int, str]] = []
    for identifier in identifiers:
        match = re.search(re.escape(identifier), query, flags=re.IGNORECASE)
        if match:
            spans.append((match.start(), match.end(), identifier))
    spans.sort()
    requirements: dict[str, tuple[list[str], list[str]]] = {}
    ignored = LEXICAL_TABLE_STOPWORDS.union(
        {"check", "should", "technician", "error", "fault", "alarm", "what", "when", "they"}
    )
    for index, (_start, end, identifier) in enumerate(spans):
        clause_end = spans[index + 1][0] if index + 1 < len(spans) else len(query)
        clause = query[end:clause_end]
        codes: list[str] = []
        for match in re.finditer(
            r"\b(?:error|alarm|fault)\s*(?:number|no\.?\s*)?[:#-]?\s*(\d{2,6})\b",
            clause,
            flags=re.IGNORECASE,
        ):
            code = _compact_identifier(match.group(1))
            if code and code not in codes:
                codes.append(code)
        context_terms: list[str] = []
        for raw_term in tokenize(clause):
            term = re.sub(r"[^a-z0-9]+", "", raw_term.lower())
            if term in ignored or any(char.isdigit() for char in term) or len(term) < 3:
                continue
            if term not in context_terms:
                context_terms.append(term)
        requirements[identifier] = (codes, context_terms)
    global_codes = [
        _compact_identifier(match.group(1))
        for match in re.finditer(
            r"\b(?:error|alarm|fault)\s*(?:number|no\.?\s*)?[:#-]?\s*(\d{2,6})\b",
            query,
            flags=re.IGNORECASE,
        )
    ]
    list_connector_terms = {
        "a",
        "an",
        "and",
        "model",
        "models",
        "or",
        "series",
        "system",
        "systems",
        "the",
        "versus",
        "vs",
        "with",
    }
    identifiers_form_compact_list = len(spans) >= 2 and all(
        all(
            re.sub(r"[^a-z0-9]+", "", term.lower()) in list_connector_terms
            for term in tokenize(query[spans[index - 1][1] : spans[index][0]])
            if re.sub(r"[^a-z0-9]+", "", term.lower())
        )
        for index in range(1, len(spans))
    )
    if (
        len(set(global_codes)) == 1
        and requirements
        and (
            not any(codes for codes, _terms in requirements.values())
            or identifiers_form_compact_list
        )
    ):
        shared_code = global_codes[0]
        requirements = {
            identifier: ([shared_code], context_terms)
            for identifier, (_codes, context_terms) in requirements.items()
        }
    return requirements


def _diagnostic_prose_context_score(result: SearchResult, context_terms: list[str]) -> int:
    if not context_terms:
        return 0
    evidence_terms = [re.sub(r"[^a-z0-9]+", "", term.lower()) for term in tokenize(str(result.content or ""))]
    positions: list[int] = []
    cursor = 0
    for required in context_terms:
        try:
            position = evidence_terms.index(required, cursor)
        except ValueError:
            return 0
        positions.append(position)
        cursor = position + 1
    if positions[-1] - positions[0] > max(12, len(context_terms) * 3):
        return 0
    return len(context_terms)


def _diagnostic_action_field_score(result: SearchResult, query: str) -> int:
    if not re.search(r"\b(?:what|which)\b.{0,80}\bshould\b.{0,80}\b(?:check|do)\b", query, flags=re.IGNORECASE):
        return 0
    content = str(result.content or "").lower()
    column_headers = " ".join(str(item) for item in result.metadata.get("table_column_headers") or []).lower()
    compact_headers = re.sub(r"[^a-z0-9]+", "", f"{column_headers} {content.split(';', 1)[0]}")
    if "checkpoint" in compact_headers:
        return 3
    if "remedy" in compact_headers or "correctiveaction" in compact_headers:
        return 2
    if re.search(r"\b(?:check point|remedy|corrective action)\s*:", content):
        return 1
    return 0


def _text_contains_diagnostic_code(text: str, code: str) -> bool:
    return bool(re.search(_diagnostic_code_pattern(code), text, flags=re.IGNORECASE))


def _comparison_term_variants(term: str) -> list[str]:
    variants: list[str] = []
    if term.endswith("ies") and len(term) > 5:
        variants.append(f"{term[:-3]}y")
    elif term.endswith("s") and len(term) > 4:
        variants.append(term[:-1])
    if term in {"failure", "failures"}:
        variants.extend(["failed", "fail"])
    if term in {"failed", "failing"}:
        variants.extend(["failure", "fail"])
    if term in {"error", "errors"}:
        variants.extend(["error", "errors"])
    return [variant for variant in variants if variant and variant != term]


def _lexical_table_content_terms(terms: list[str]) -> list[str]:
    content_terms = [
        term
        for term in terms
        if not any(char.isdigit() for char in term)
        and term not in {"corrective", "corrected", "remedy", "cause", "checkpoint", "troubleshooting", "verify"}
    ]
    return sorted(content_terms, key=lambda term: (len(term), term), reverse=True)[:1]


def _comparison_table_content_terms(terms: list[str]) -> list[str]:
    content_terms = [
        term
        for term in terms
        if term
        and len(term) >= 2
        and not (any(char.isdigit() for char in term) and len(term) >= 3)
        and term not in {"compare", "listed", "documentation"}
        and term not in LEXICAL_TABLE_STOPWORDS
    ]
    expanded_terms: list[str] = []
    for term in sorted(content_terms, key=lambda term: (any(char.isdigit() for char in term), len(term), term), reverse=True):
        for variant in [term, *_comparison_term_variants(term)]:
            if variant not in expanded_terms:
                expanded_terms.append(variant)
    return expanded_terms[:16]


def _lexical_table_symbol_terms(terms: list[str]) -> list[str]:
    symbol_terms: list[str] = []
    for term in terms:
        if any(char.isdigit() for char in term) or (
            3 <= len(term) <= 6 and term not in LEXICAL_TABLE_STOPWORDS and term not in LEXICAL_TABLE_FIELD_TERMS
        ):
            symbol_terms.append(term)
    return symbol_terms[:8]


def _is_status_output_table_lookup(query: str) -> bool:
    lowered = query.lower()
    return (
        "output status" in lowered
        and "set value" in lowered
        and re.search(r"\bcount\s+value\b", lowered) is not None
        and re.search(r"\b(?:current|previous|quantity|counted)\b", lowered) is not None
    )


def _status_output_numeric_terms(query: str) -> list[str]:
    query_without_models = re.sub(r"\b[A-Za-z]{1,5}\d{0,4}(?:-[A-Za-z0-9]{1,8})+\b", " ", query)
    terms: list[str] = []
    for term in re.findall(r"\b\d+(?:\.\d+)?\b", query_without_models):
        if term not in terms:
            terms.append(term)
    return terms[:8]


def _status_output_requires_exact_set_value(query: str) -> bool:
    return bool(re.search(r"\b(?:on\s+(?:equals|=)|on\s+when\s*=\s*set\s+value)\b", query, flags=re.IGNORECASE))


def _normalized_status_output_text(text: str) -> str:
    return re.sub(r"[^a-z0-9=]+", "", text.lower())


def _status_output_line_uses_range_set_value(text: str) -> bool:
    return bool(
        re.search(
            r"\bon\s+when\s*(?:>=|=>|\u2265)\s*set\s+value\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _status_output_matching_row_context(query: str, row: dict[str, object], context_rows: list[dict[str, object]]) -> str:
    metadata = dict(row.get("metadata_json") or {})
    if not metadata.get("table_cell"):
        return ""
    if "output status" not in " ".join(str(item) for item in metadata.get("table_column_headers") or []).lower():
        return ""
    cell_match = re.search(r"\bCell value:\s*([^;\n]+)", str(row.get("content") or ""), flags=re.IGNORECASE)
    if not cell_match:
        return ""
    cell_value = cell_match.group(1).strip()
    cell_terms = _text_terms(cell_value)
    row_header_terms = _text_terms(" ".join(str(item) for item in metadata.get("table_row_headers") or []))
    distinctive_row_header_terms = {
        term for term in row_header_terms if term.isdigit() or any(char.isdigit() for char in term) or len(term) >= 5
    }
    required_numbers = set(_status_output_numeric_terms(query))
    if not required_numbers:
        return ""
    query_requires_exact = _status_output_requires_exact_set_value(query)
    for context_row in context_rows:
        for raw_line in str(context_row.get("content") or "").splitlines():
            line = raw_line.strip()
            if "|" not in line:
                continue
            normalized = _normalized_status_output_text(line)
            if "outputstatus" in normalized and not re.search(r"\b(?:latching|one[- ]?shot|does\s+not\s+output)\b", line, flags=re.IGNORECASE):
                continue
            if query_requires_exact and "onwhen=setvalue" not in normalized:
                continue
            if query_requires_exact and _status_output_line_uses_range_set_value(line):
                continue
            line_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", line))
            if not required_numbers.issubset(line_numbers):
                continue
            line_terms = _text_terms(line)
            if distinctive_row_header_terms and not distinctive_row_header_terms.issubset(line_terms):
                continue
            if cell_terms and not cell_terms.issubset(line_terms):
                continue
            return line
    return ""


def _attach_status_output_row_context(query: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows or not _is_status_output_table_lookup(query):
        return rows
    sections = {
        (str(row["document_version_id"]), str(row["section_path_text"]))
        for row in rows
        if dict(row.get("metadata_json") or {}).get("table_cell")
    }
    if not sections:
        return rows
    placeholders = ",".join(["(%s,%s)"] * len(sections))
    params: list[object] = []
    for document_version_id, section_path_text in sections:
        params.extend([document_version_id, section_path_text])
    context_rows = fetch_all(
        f"""
        select document_version_id, section_path_text, content, metadata_json
        from retrieval_chunks
        where (document_version_id, section_path_text) in ({placeholders})
          and chunk_type = 'table_record'
          and metadata_json->>'table_row_group' = 'true'
        """,
        tuple(params),
    )
    contexts_by_section: dict[tuple[str, str], list[dict[str, object]]] = {}
    for context_row in context_rows:
        key = (str(context_row["document_version_id"]), str(context_row["section_path_text"]))
        contexts_by_section.setdefault(key, []).append(context_row)
    updated: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["document_version_id"]), str(row["section_path_text"]))
        matching_context = _status_output_matching_row_context(query, row, contexts_by_section.get(key, []))
        if matching_context:
            row = {
                **row,
                "content": f"{row.get('content')}\nSource row context: {matching_context}",
                "priority_score": float(row.get("priority_score") or 0.0) + 600.0,
            }
        updated.append(row)
    return updated


def _comparison_row_key_terms(terms: list[str]) -> list[str]:
    row_key_terms: list[str] = []
    for term in [*_lexical_table_symbol_terms(terms), *terms]:
        if term in row_key_terms:
            continue
        if term in LEXICAL_TABLE_STOPWORDS or term in {"compare", "corrective", "action", "listed", "documentation"}:
            continue
        if any(char.isdigit() for char in term) or len(term) >= 5:
            row_key_terms.append(term)
        if len(row_key_terms) >= 12:
            break
    return row_key_terms


def _structured_prompt_phrase(query: str) -> str:
    match = re.search(
        r"\bwhat\s+causes?\s+(?P<phrase>.+?)\s+for\s+.+?,\s+and\s+(?:what|how)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\bwhat\s+(?:value|setting|number\s+format|initial\s+value|upper\s+limit(?:\s+value)?|"
            r"lower\s+limit(?:\s+value)?|decimal\s+digits|integer\s+digits|referenceable)\b"
            r".+\b(?:listed|specified|shown|given|configured|set)\s+for\s+(?P<phrase>.+?)\??$",
            query,
            flags=re.IGNORECASE,
        )
    if not match:
        return ""
    phrase = re.sub(r"\s+", " ", match.group("phrase")).strip(" .,:;")
    return re.sub(r"[^a-z0-9]+", "", phrase.lower())


def _structured_lookup_subject_terms(query: str) -> set[str]:
    match = re.search(
        r"\bwhat\s+(?:error\s+)?(?:message|symbol|description|summary|detection)\s+(?P<subject>.+?)\s+(?:applies?\s+(?:to|for)|is\s+specified\s+for)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return set()
    terms: set[str] = set()
    for term in tokenize(match.group("subject")):
        normalized = re.sub(r"[^a-z0-9]+", "", term.lower())
        if len(normalized) < 4 or normalized in LEXICAL_TABLE_STOPWORDS:
            continue
        terms.add(normalized)
    return terms


def _structured_lookup_field_terms(query: str) -> set[str]:
    match = re.search(
        r"\bwhat\s+(?P<field>error\s+message|message|symbol|description|summary|detection)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return set()
    field = re.sub(r"\s+", " ", match.group("field").strip().lower())
    if field == "error message":
        return {"error", "message", "errormessage"}
    return {re.sub(r"[^a-z0-9]+", "", field)}


def _table_lexical_score(row: dict[str, object], terms: list[str], prompt_phrase: str = "") -> float:
    content = str(row.get("content") or "")
    metadata = dict(row.get("metadata_json") or {})
    row_header_text = " ".join(str(item) for item in metadata.get("table_row_headers") or [])
    column_header_text = " ".join(str(item) for item in metadata.get("table_column_headers") or [])
    haystack = " ".join(
        str(part)
        for part in [
            content,
            metadata.get("product_model"),
            metadata.get("product_family"),
            " ".join(str(item) for item in metadata.get("product_models") or []),
            " ".join(str(item) for item in metadata.get("devices") or []),
            column_header_text,
            row_header_text,
        ]
        if part
    ).lower()
    compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
    compact_content = re.sub(r"[^a-z0-9]+", "", content.lower())
    compact_row_headers = re.sub(r"[^a-z0-9]+", "", row_header_text.lower())
    compact_column_headers = re.sub(r"[^a-z0-9]+", "", column_header_text.lower())
    overlap = sum(1 for term in terms if term in compact_haystack)
    if overlap <= 0:
        return 0.0
    score = overlap * 0.12
    content_overlap = sum(1 for term in terms if term in compact_content)
    score += min(0.48, content_overlap * 0.12)
    numeric_anchor_terms = [term for term in terms if term.isdigit() and 2 <= len(term) <= 6]
    numeric_content_overlap = sum(1 for term in numeric_anchor_terms if term in compact_content)
    score += min(1.2, numeric_content_overlap * 0.25)
    if numeric_anchor_terms and numeric_content_overlap == len(numeric_anchor_terms):
        score += 0.6
    code_label_terms = [term for term in terms if term in {"id", "pid", "pib", "piw", "plc", "ch"}]
    if code_label_terms and numeric_anchor_terms:
        code_label_overlap = sum(1 for term in code_label_terms if term in compact_content)
        if code_label_overlap:
            score += min(0.4, code_label_overlap * 0.2)
    symbol_terms = _lexical_table_symbol_terms(terms)
    symbol_overlap = sum(1 for term in symbol_terms if term in compact_haystack)
    score += min(0.9, symbol_overlap * 0.22)
    row_header_symbol_overlap = sum(1 for term in symbol_terms if term and term in compact_row_headers)
    if row_header_symbol_overlap and metadata.get("table_column_headers"):
        score += min(0.6, row_header_symbol_overlap * 0.3)
    header_issue_terms = [
        term
        for term in terms
        if len(term) >= 5
        and not any(char.isdigit() for char in term)
        and term not in {"corrective", "action", "compare", "listed"}
    ]
    row_header_issue_overlap = sum(1 for term in header_issue_terms if term and term in compact_row_headers)
    if row_header_issue_overlap and metadata.get("table_column_headers"):
        score += min(0.45, row_header_issue_overlap * 0.15)
    requested_field_terms = _comparison_requested_field_terms(" ".join(terms))
    if requested_field_terms:
        if _table_result_matches_requested_field_metadata(metadata, requested_field_terms):
            score += 0.7
        elif metadata.get("table_column_headers"):
            score -= 0.7
    if compact_column_headers and any(term in compact_column_headers for term in {"corrective", "action", "description"}):
        score += 0.08
    short_code_overlap = sum(1 for term in terms if any(char.isdigit() for char in term) and len(term) <= 4 and term in compact_haystack)
    score += min(0.45, short_code_overlap * 0.15)
    if {"measured", "data", "format"}.intersection(terms) and "formofmeasureddata" in compact_haystack:
        score += 0.45
    if "symbol" not in terms and re.match(r"\s*(?:symbol|tool):", content, flags=re.IGNORECASE):
        score -= 0.85
    product_haystack = " ".join(
        str(part)
        for part in [
            metadata.get("product_model"),
            metadata.get("product_family"),
            " ".join(str(item) for item in metadata.get("product_models") or []),
            " ".join(str(item) for item in metadata.get("devices") or []),
        ]
        if part
    ).lower()
    product_terms = _text_terms(product_haystack)
    product_overlap = len(set(terms).intersection(product_terms))
    score += min(0.36, product_overlap * 0.09)
    primary_product_terms = _text_terms(
        " ".join(
            str(part)
            for part in [
                metadata.get("product_model"),
                metadata.get("product_family"),
                " ".join(str(item) for item in metadata.get("product_models") or []),
                " ".join(str(item) for item in metadata.get("product_families") or []),
                " ".join(str(item) for item in metadata.get("part_numbers") or []),
            ]
            if part
        )
    )
    primary_product_overlap = len(set(terms).intersection(primary_product_terms))
    score += min(0.72, primary_product_overlap * 0.18)
    if prompt_phrase and prompt_phrase in compact_haystack:
        score += 1.0
    elif prompt_phrase:
        phrase_terms = [
            term
            for term in re.findall(r"[a-z0-9]+", prompt_phrase)
            if len(term) >= 4 and term not in LEXICAL_TABLE_STOPWORDS
        ]
        phrase_overlap = sum(1 for term in phrase_terms if term in compact_haystack)
        score += min(0.75, phrase_overlap * 0.09)
    if metadata.get("table_row_group"):
        score += 0.28
    if metadata.get("table_summary"):
        score += 0.12
    if metadata.get("table_cell"):
        score += 0.06
        score += _table_cell_value_binding_score(content, metadata, terms)
    if metadata.get("table_key_value") and re.search(r"\b(?:status|cause|remedy|error|alarm)\b", content, flags=re.IGNORECASE):
        score += 0.18
    if metadata.get("table_header") and not (
        metadata.get("table_cell") or metadata.get("table_key_value") or metadata.get("table_row_group") or metadata.get("table_summary")
    ):
        score -= 0.35
    if re.search(r"\b(?:cause|remedy|corrective action|error messages?|symptom)\b", content, flags=re.IGNORECASE):
        score += 0.16
    return score + float(row.get("priority_score") or 0.0) / 100.0


def _table_cell_value_binding_score(content: str, metadata: dict[str, object], terms: list[str]) -> float:
    if not metadata.get("table_cell"):
        return 0.0
    cell_match = re.search(r"\bCell value:\s*([^;\n]+)", content, flags=re.IGNORECASE)
    if not cell_match:
        return 0.0
    cell_value = cell_match.group(1).strip()
    cell_terms = _text_terms(cell_value)
    term_set = set(terms)
    numeric_terms = {term for term in term_set if term.isdigit()}
    if not numeric_terms:
        return 0.0

    score = 0.0
    row_terms = _text_terms(" ".join(str(item) for item in metadata.get("table_row_headers") or []))
    column_terms = _text_terms(" ".join(str(item) for item in metadata.get("table_column_headers") or []))
    row_overlap = term_set.intersection(row_terms)
    column_overlap = term_set.intersection(column_terms)
    cell_overlap = term_set.intersection(cell_terms)
    distinctive_row_overlap = {
        term for term in row_overlap if term.isdigit() or any(char.isdigit() for char in term) or len(term) >= 5
    }
    if distinctive_row_overlap and column_overlap and cell_overlap:
        score += 0.9
        score += min(0.45, 0.15 * len(distinctive_row_overlap))
        score += min(0.45, 0.15 * len(column_overlap))
        score += min(0.3, 0.15 * len(cell_overlap))
    elif distinctive_row_overlap and column_overlap and _terms_ask_for_unknown_quantity(term_set):
        score += 1.1
        score += min(0.45, 0.15 * len(distinctive_row_overlap))
        score += min(0.45, 0.15 * len(column_overlap))
    elif distinctive_row_overlap and column_overlap and numeric_terms.intersection(cell_terms):
        score += 0.55
    if {"data", "output"}.issubset(term_set) and {"data", "output", "bit"}.issubset(cell_terms):
        matched_numbers = numeric_terms.intersection(cell_terms)
        if matched_numbers:
            score += 1.8 + min(0.6, 0.2 * len(matched_numbers))
            if {"signal", "description"}.issubset(column_terms):
                score += 1.0
            if any(number in "".join(row_terms) for number in matched_numbers) and any(
                term.startswith(("out", "data", "signal")) for term in row_terms
            ):
                score += 0.45
    return score


def _terms_ask_for_unknown_quantity(terms: set[str]) -> bool:
    return bool(
        {"count", "counted", "quantity", "number", "total"}.intersection(terms)
        and {"many", "much", "how", "objects", "items", "lines", "time"}.intersection(terms)
    )


def _comparison_setting_phrases(query: str) -> list[str]:
    lowered = query.lower()
    phrases: list[str] = []
    for match in re.finditer(
        r"\b(?:compare\s+what\s+the|compare\s+the|what\s+the)\s+(?P<labels>.+?)\s+settings?\s+(?:control|controls|do|does|mean|represent|for|with|listed)\b",
        query,
        flags=re.IGNORECASE,
    ):
        labels = re.split(r"\s+(?:and|with|versus|vs\.?)\s+|,", match.group("labels"))
        for label in labels:
            normalized = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9/_. -]+", " ", label)).strip()
            if not normalized:
                continue
            terms = [
                re.sub(r"[^a-z0-9]+", "", term.lower())
                for term in tokenize(normalized)
                if re.sub(r"[^a-z0-9]+", "", term.lower())
            ]
            if len(terms) < 2:
                continue
            if any(term in {"compare", "what", "setting", "settings", "control", "controls"} for term in terms):
                continue
            phrase = " ".join(terms)
            if phrase and phrase not in phrases and phrase in re.sub(r"[^a-z0-9]+", " ", lowered):
                phrases.append(phrase)
    return phrases[:6]


def _comparison_setting_phrase_score(result: SearchResult, phrases: list[str]) -> float:
    if not phrases:
        return 0.0
    row_header_text = " ".join(str(item) for item in result.metadata.get("table_row_headers") or [])
    column_header_text = " ".join(str(item) for item in result.metadata.get("table_column_headers") or [])
    haystack = re.sub(
        r"[^a-z0-9]+",
        " ",
        " ".join([str(result.content or ""), row_header_text, column_header_text]).lower(),
    )
    score = 0.0
    for phrase in phrases:
        if re.search(rf"\b{re.escape(phrase)}\b", haystack):
            score += 4.0
        else:
            phrase_terms = [term for term in phrase.split() if len(term) >= 4]
            if phrase_terms and all(re.search(rf"\b{re.escape(term)}\b", haystack) for term in phrase_terms):
                score += 2.0
    return score


def _comparison_result_matches_setting_phrase(result: SearchResult, phrases: list[str]) -> bool:
    if not phrases:
        return True
    return _comparison_setting_phrase_score(result, phrases) >= 4.0


def _comparison_setting_phrases_for_identifier(query: str, identifiers: list[str], identifier: str) -> list[str]:
    phrases = _comparison_setting_phrases(query)
    if len(phrases) < 2:
        return phrases
    try:
        index = identifiers.index(identifier)
    except ValueError:
        return phrases
    if index < len(phrases):
        return [phrases[index]]
    return phrases


def _comparison_result_matches_any_side_setting(result: SearchResult, query: str, identifiers: list[str]) -> bool:
    phrases = _comparison_setting_phrases(query)
    if len(phrases) < 2:
        return True
    matched_identifiers = [
        identifier
        for identifier in identifiers
        if _result_matches_primary_identifier(result, identifier)
    ]
    if not matched_identifiers:
        return True
    return any(
        _comparison_result_matches_setting_phrase(
            result,
            _comparison_setting_phrases_for_identifier(query, identifiers, identifier),
        )
        for identifier in matched_identifiers
    )


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
    is_comparison_lookup = "comparison" in analysis.query_types and bool(analysis.product_identifiers)
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
    base_where = [*where]
    base_params = [*params]
    required_terms = [] if is_comparison_lookup else _lexical_table_content_terms(terms)
    symbol_terms = _lexical_table_symbol_terms(terms)
    diagnostic_code_terms = _diagnostic_table_code_terms(query, analysis)
    if diagnostic_code_terms:
        symbol_terms = diagnostic_code_terms
    if is_comparison_lookup and diagnostic_code_terms:
        where.append(
            "("
            + " or ".join(
                [
                    "lower(content) ~ %s or lower(metadata_json::text) ~ %s"
                ]
                * len(diagnostic_code_terms)
            )
            + ")"
        )
        for term in diagnostic_code_terms:
            pattern = _diagnostic_code_pattern(term)
            params.extend([pattern, pattern])
    elif is_comparison_lookup:
        like_terms: list[str] = []
        for term in [*_comparison_table_content_terms(terms), *_lexical_table_symbol_terms(terms)]:
            if term not in like_terms:
                like_terms.append(term)
        if like_terms:
            where.append(
                "("
                + " or ".join(["content ilike %s"] * len(like_terms))
                + ")"
            )
            params.extend([f"%{term}%" for term in like_terms])
    elif required_terms and len(symbol_terms) < 3:
        where.extend(["regexp_replace(lower(content), '[^a-z0-9]+', '', 'g') like %s"] * len(required_terms))
        params.extend([f"%{term}%" for term in required_terms])
    if symbol_terms and not is_comparison_lookup:
        if diagnostic_code_terms:
            where.append(
                "("
                + " or ".join(
                    [
                        "regexp_replace(lower(content), '[^a-z0-9]+', '', 'g') like %s "
                        "or regexp_replace(lower(metadata_json::text), '[^a-z0-9]+', '', 'g') like %s"
                    ]
                    * len(symbol_terms)
                )
                + ")"
            )
            for term in symbol_terms:
                params.extend([f"%{term}%", f"%{term}%"])
        else:
            where.append("(" + " or ".join(["regexp_replace(lower(content), '[^a-z0-9]+', '', 'g') like %s"] * len(symbol_terms)) + ")")
            params.extend([f"%{term}%" for term in symbol_terms])
    if not is_comparison_lookup and not required_terms and not symbol_terms:
        like_terms = terms[:8]
        where.append("(" + " or ".join(["content ilike %s"] * len(like_terms)) + ")")
        params.extend([f"%{term}%" for term in like_terms])
    order_by = "order by priority_score desc, id"
    order_params: list[object] = []
    if is_comparison_lookup:
        order_terms: list[str] = []
        for term in terms:
            if term not in order_terms:
                order_terms.append(term)
            if len(order_terms) >= 10:
                break
        if order_terms:
            product_patterns = [
                f"%{str(identifier).strip()}%"
                for identifier in (analysis.product_identifiers or [])[:4]
                if str(identifier).strip()
            ]
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
    else:
        order_terms = []
        for term in terms:
            if term in {"listed"} or term in order_terms:
                continue
            order_terms.append(term)
            if len(order_terms) >= 12:
                break
        if order_terms:
            order_by = (
                "order by "
                + " + ".join(
                    [
                        "case when content ilike %s "
                        "or metadata_json->>'table_row_headers' ilike %s "
                        "or metadata_json->>'table_column_headers' ilike %s then 1 else 0 end"
                    ]
                    * len(order_terms)
                )
                + " desc, priority_score desc, id"
            )
            for term in order_terms:
                pattern = f"%{term}%"
                order_params.extend([pattern, pattern, pattern])
    rows = fetch_all(
        f"""
        select id, document_version_id, source_document_id, title, section_path_text,
               page_from, page_to, content, metadata_json, priority_score
        from retrieval_chunks
        where {" and ".join(where)}
        {order_by}
        limit {LEXICAL_TABLE_SCAN_LIMIT}
        """,
        tuple([*params, *order_params]),
    )
    if is_comparison_lookup and diagnostic_code_terms and len(analysis.product_identifiers) >= 2:
        side_requirements = _diagnostic_side_requirements(
            query,
            [str(identifier) for identifier in analysis.product_identifiers],
        )
        for identifier, (side_codes, context_terms) in side_requirements.items():
            if side_codes or not context_terms:
                continue
            distinctive_terms = sorted(context_terms, key=lambda term: (len(term), term), reverse=True)[:8]
            side_where = [*base_where]
            side_params = [*base_params]
            product_pattern = f"%{identifier}%"
            side_where.append(
                "(metadata_json->>'product_model' ilike %s "
                "or metadata_json->>'product_family' ilike %s "
                "or metadata_json->>'product_models' ilike %s "
                "or title ilike %s)"
            )
            side_params.extend([product_pattern] * 4)
            side_where.extend(["content ilike %s"] * len(distinctive_terms))
            side_params.extend([f"%{term}%" for term in distinctive_terms])
            rows.extend(
                fetch_all(
                    f"""
                    select id, document_version_id, source_document_id, title, section_path_text,
                           page_from, page_to, content, metadata_json, priority_score
                    from retrieval_chunks
                    where {" and ".join(side_where)}
                    order by priority_score desc, id
                    limit 120
                    """,
                    tuple(side_params),
                )
            )
    if not is_comparison_lookup and _is_status_output_table_lookup(query):
        numeric_terms = _status_output_numeric_terms(query)
        if numeric_terms:
            supplemental_where = [*base_where]
            supplemental_params = [*base_params]
            supplemental_where.append(
                "(content ilike %s or metadata_json->>'table_column_headers' ilike %s)"
            )
            supplemental_params.extend(["%output status%", "%output status%"])
            supplemental_where.append(
                "(content ilike %s or metadata_json->>'table_row_headers' ilike %s)"
            )
            supplemental_params.extend(["%count value%", "%count value%"])
            if _status_output_requires_exact_set_value(query):
                supplemental_where.append("regexp_replace(lower(content), '[^a-z0-9=]+', '', 'g') like %s")
                supplemental_params.append("%onwhen=setvalue%")
            rows.extend(
                fetch_all(
                    f"""
                    select id, document_version_id, source_document_id, title, section_path_text,
                           page_from, page_to, content, metadata_json, priority_score
                    from retrieval_chunks
                    where {" and ".join(supplemental_where)}
                    order by priority_score desc, id
                    limit 160
                    """,
                    tuple(supplemental_params),
                )
            )
    rows = _attach_status_output_row_context(query, rows)
    if is_comparison_lookup and analysis.product_identifiers:
        row_key_terms = _comparison_row_key_terms(terms)
        product_patterns = [f"%{identifier}%" for identifier in analysis.product_identifiers[:4]]
        if row_key_terms and product_patterns:
            supplemental_where = [*where]
            supplemental_params = [*params]
            row_key_order_fragments = [
                "case when metadata_json->>'table_row_headers' ilike %s or content ilike %s then 1 else 0 end"
                for _ in row_key_terms
            ]
            row_key_order_params: list[object] = []
            for term in row_key_terms:
                row_key_order_params.extend([f"%{term}%", f"%{term}%"])
            supplemental_where.append(
                "("
                + " or ".join(
                    ["metadata_json->>'table_row_headers' ilike %s or content ilike %s"] * len(row_key_terms)
                )
                + ")"
            )
            for term in row_key_terms:
                supplemental_params.extend([f"%{term}%", f"%{term}%"])
            supplemental_where.append(
                "("
                + " or ".join(
                    [
                        "metadata_json->>'product_model' ilike %s "
                        "or metadata_json->>'product_family' ilike %s "
                        "or metadata_json->>'product_models' ilike %s "
                        "or metadata_json->>'devices' ilike %s "
                        "or title ilike %s"
                    ]
                    * len(product_patterns)
                )
                + ")"
            )
            for pattern in product_patterns:
                supplemental_params.extend([pattern, pattern, pattern, pattern, pattern])
            rows.extend(
                fetch_all(
                    f"""
                    select id, document_version_id, source_document_id, title, section_path_text,
                           page_from, page_to, content, metadata_json, priority_score
                    from retrieval_chunks
                    where {" and ".join(supplemental_where)}
                    order by {" + ".join(row_key_order_fragments)} desc, priority_score desc, id
                    limit 120
                    """,
                    tuple([*supplemental_params, *row_key_order_params]),
                )
            )
        setting_phrases = _comparison_setting_phrases(query)
        identifiers = [str(identifier) for identifier in analysis.product_identifiers]
        if setting_phrases and product_patterns:
            supplemental_where = [*where]
            supplemental_params = [*params]
            supplemental_where.append(
                "(" + " or ".join(["content ilike %s or metadata_json->>'table_row_headers' ilike %s"] * len(setting_phrases)) + ")"
            )
            for phrase in setting_phrases:
                pattern = f"%{'%'.join(phrase.split())}%"
                supplemental_params.extend([pattern, pattern])
            supplemental_where.append(
                "("
                + " or ".join(
                    [
                        "metadata_json->>'product_model' ilike %s "
                        "or metadata_json->>'product_family' ilike %s "
                        "or metadata_json->>'product_models' ilike %s "
                        "or metadata_json->>'devices' ilike %s "
                        "or title ilike %s"
                    ]
                    * len(product_patterns)
                )
                + ")"
            )
            for pattern in product_patterns:
                supplemental_params.extend([pattern, pattern, pattern, pattern, pattern])
            rows.extend(
                fetch_all(
                    f"""
                    select id, document_version_id, source_document_id, title, section_path_text,
                           page_from, page_to, content, metadata_json, priority_score
                    from retrieval_chunks
                    where {" and ".join(supplemental_where)}
                    order by priority_score desc, id
                    limit 120
                    """,
                    tuple(supplemental_params),
                )
            )
    ranked: list[tuple[float, SearchResult]] = []
    seen_rows: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen_rows:
            continue
        seen_rows.add(row_id)
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
    results = [result for _, result in ranked]
    if is_comparison_lookup and len(analysis.product_identifiers) >= 2:
        identifiers = [str(identifier) for identifier in analysis.product_identifiers]
        results = [
            result
            for result in results
            if _comparison_result_matches_any_side_setting(result, analysis.raw_query, identifiers)
        ]
        setting_phrases = _comparison_setting_phrases(query)
        row_code_terms = _comparison_row_code_terms(
            analysis.raw_query,
            identifiers,
        )
        if len(row_code_terms) < 2:
            row_code_terms = []
        covered_row_codes: set[str] = set()
        promoted: list[SearchResult] = []
        seen_ids: set[str] = set()
        for identifier in analysis.product_identifiers:
            uncovered_row_codes = [term for term in row_code_terms if term not in covered_row_codes]
            if row_code_terms and not uncovered_row_codes:
                break
            side_setting_phrases = _comparison_setting_phrases_for_identifier(analysis.raw_query, identifiers, str(identifier))
            context_terms = _comparison_side_context_terms(analysis.raw_query, identifiers, str(identifier))
            side_row_code_options = _comparison_side_row_code_options(
                uncovered_row_codes or row_code_terms,
                context_terms,
                identifiers=identifiers,
                identifier=str(identifier),
            )
            candidates = [
                result
                for result in results
                if result.chunk_id not in seen_ids and _result_matches_primary_identifier(result, identifier)
                and _comparison_result_matches_setting_phrase(result, side_setting_phrases)
                and any(_result_matches_all_comparison_row_codes(result, option) for option in side_row_code_options)
            ]
            if context_terms:
                candidates.sort(
                    key=lambda result: (
                        _comparison_setting_phrase_score(result, side_setting_phrases),
                        _identifier_context_score(result, context_terms),
                        result.score,
                    ),
                    reverse=True,
                )
            promotion_limit = 1
            for candidate in candidates[:promotion_limit]:
                seen_ids.add(candidate.chunk_id)
                covered_row_codes.update(_matching_comparison_row_codes(candidate, row_code_terms))
                promoted.append(candidate)
        if promoted:
            deduped: list[SearchResult] = []
            for result in [*promoted, *results]:
                if result.chunk_id in seen_ids and result not in promoted:
                    continue
                if any(existing.chunk_id == result.chunk_id for existing in deduped):
                    continue
                deduped.append(result)
            results = deduped
        if diagnostic_code_terms:
            side_requirements = _diagnostic_side_requirements(query, identifiers)
            prose_promoted: list[SearchResult] = []
            for identifier, (side_codes, context_terms) in side_requirements.items():
                if side_codes or not context_terms:
                    continue
                candidates = [
                    result
                    for result in results
                    if _result_matches_primary_identifier(result, identifier)
                    and _diagnostic_prose_context_score(result, context_terms) >= len(context_terms)
                ]
                if candidates:
                    prose_promoted.append(
                        max(
                            candidates,
                            key=lambda result: (
                                _diagnostic_action_field_score(result, query),
                                _diagnostic_prose_context_score(result, context_terms),
                                int(bool(result.metadata.get("table_key_value"))),
                                -len(str(result.content or "")),
                                result.score,
                            ),
                        )
                    )
            if prose_promoted:
                promoted_ids = {result.chunk_id for result in prose_promoted}
                results = [*prose_promoted, *(result for result in results if result.chunk_id not in promoted_ids)]
    return results[:limit]


def _lexical_context_terms(query: str, analysis: QueryAnalysis) -> list[str]:
    if "structured_lookup" in analysis.query_types:
        return []
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


def _explicit_scope_phrase_groups(text: str) -> list[set[str]]:
    phrase_groups: list[set[str]] = []
    for match in re.finditer(r"\b([A-Za-z0-9][A-Za-z0-9/\- ]{0,60}?\s+mode)\b", text, flags=re.IGNORECASE):
        phrase = " ".join(re.findall(r"[a-z0-9]+", match.group(1).lower()))
        words = phrase.split()
        if len(words) < 2:
            continue
        candidates = {
            " ".join(words[-size:])
            for size in range(2, min(4, len(words)) + 1)
            if len(words[-size]) >= 3
        }
        if candidates:
            phrase_groups.append(candidates)
    return phrase_groups


def _explicit_scope_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in re.finditer(r"\b([A-Za-z0-9][A-Za-z0-9/\- ]{0,60}?\s+mode)\b", text, flags=re.IGNORECASE):
        phrase = " ".join(re.findall(r"[a-z0-9]+", match.group(1).lower()))
        if len(phrase.split()) >= 2:
            phrases.append(phrase)
    return phrases


def _scope_phrase_matches_group(phrase: str, group: set[str]) -> bool:
    return any(scope in phrase or phrase in scope for scope in group)


def _has_conflicting_explicit_scope(query_scope_groups: list[set[str]], evidence: str) -> bool:
    if not query_scope_groups:
        return False
    for phrase in _explicit_scope_phrases(evidence):
        if not any(_scope_phrase_matches_group(phrase, group) for group in query_scope_groups):
            return True
    return False


def _result_supports_explicit_scope(query: str, result: SearchResult) -> bool:
    query_scope_groups = _explicit_scope_phrase_groups(query)
    if not query_scope_groups:
        return _result_supports_capture_type_scope(query, result)
    evidence = " ".join(
        str(part)
        for part in [
            result.content,
            result.title,
            " ".join(str(item) for item in result.section_path or []),
            result.metadata.get("content_for_rerank"),
            result.metadata.get("local_rerank_context"),
        ]
        if part
    )
    normalized = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    if _has_conflicting_explicit_scope(query_scope_groups, evidence):
        return False
    return all(any(scope in normalized for scope in group) for group in query_scope_groups) and _evidence_supports_capture_type_scope(
        query, evidence
    )


def _capture_type_scope(text: str) -> str | None:
    scopes = _capture_type_scopes(text)
    if len(scopes) != 1:
        return None
    return next(iter(scopes))


def _capture_type_scopes(text: str) -> set[str]:
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    scopes: set[str] = set()
    if re.search(r"\bline\s+scan(?:\s+cameras?)?\b", normalized) or re.search(
        r"\bline\s+cameras?\b", normalized
    ):
        scopes.add("line_scan")
    if re.search(r"\barea\s+cameras?\b", normalized):
        scopes.add("area_camera")
    return scopes


def _evidence_supports_capture_type_scope(query: str, evidence: str) -> bool:
    query_scope = _capture_type_scope(query)
    if query_scope is None:
        return True
    evidence_scopes = _capture_type_scopes(evidence)
    if len(evidence_scopes) != 1:
        return False
    return query_scope in evidence_scopes


def _result_supports_capture_type_scope(query: str, result: SearchResult) -> bool:
    evidence = " ".join(
        str(part)
        for part in [
            result.content,
            result.title,
            " ".join(str(item) for item in result.section_path or []),
            result.metadata.get("content_for_rerank"),
            result.metadata.get("local_rerank_context"),
        ]
        if part
    )
    return _evidence_supports_capture_type_scope(query, evidence)


def _direct_configuration_query(query: str, analysis: QueryAnalysis) -> bool:
    if "configuration" not in analysis.query_types and "configuration" not in _text_terms(query):
        return False
    query_terms = _text_terms(query)
    anchors = {"camera", "trigger", "light", "lighting", "illumination"}.intersection(query_terms)
    return len(anchors) >= 2


def _direct_configuration_result_score(query: str, result: SearchResult) -> float:
    if not _result_supports_explicit_scope(query, result):
        return 0.0
    evidence = " ".join(
        str(part)
        for part in [
            result.content,
            result.metadata.get("content_for_rerank"),
            result.metadata.get("local_rerank_context"),
            " ".join(str(item) for item in result.section_path or []),
        ]
        if part
    )
    normalized = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    if not re.search(r"\bcamera\s+trigger\s+light\s+configuration\b", normalized):
        return 0.0
    has_trigger_input = re.search(r"\btrigger\s+inputs?\b", normalized) is not None
    has_light_target = (
        re.search(r"\billumination\s+control\s+targets?\b", normalized) is not None
        or re.search(r"\blight\s+control\s+targets?\b", normalized) is not None
    )
    if not (has_trigger_input and has_light_target):
        return 0.0
    score = 4.0
    query_terms = _text_terms(query)
    evidence_terms = _text_terms(evidence)
    score += min(2.0, len(query_terms.intersection(evidence_terms)) * 0.2)
    chunk_type = str(result.metadata.get("chunk_type") or "")
    if chunk_type == "section_window":
        score += 0.3
    elif chunk_type == "atomic_text":
        score += 0.2
    return score + min(1.0, result.score)


def _promote_direct_configuration_candidates(
    primary_results: list[SearchResult],
    supplemental_results: list[SearchResult],
    analysis: QueryAnalysis,
    *,
    limit: int = 12,
) -> list[SearchResult]:
    if not supplemental_results or not _direct_configuration_query(analysis.raw_query, analysis):
        return primary_results
    candidates = [
        (_direct_configuration_result_score(analysis.raw_query, result), index, result)
        for index, result in enumerate(supplemental_results)
    ]
    candidates = [candidate for candidate in candidates if candidate[0] > 0.0]
    if not candidates:
        return primary_results
    best_score, _best_index, best_result = max(candidates, key=lambda item: (item[0], -item[1]))
    promoted = best_result.model_copy(
        update={
            "score": max(best_result.score, primary_results[0].score if primary_results else best_result.score) + 0.01,
            "metadata": {
                **best_result.metadata,
                "retrieval_stage": "direct_configuration_promoted",
                "direct_configuration_support_score": best_score,
            },
        }
    )
    support = _direct_configuration_setup_candidate(analysis.raw_query, best_result, [*supplemental_results, *primary_results])
    promoted_support: SearchResult | None = None
    if support and support.chunk_id != promoted.chunk_id:
        promoted_support = support.model_copy(
            update={
                "score": max(support.score, promoted.score - 0.005),
                "metadata": {
                    **support.metadata,
                    "retrieval_stage": "direct_configuration_setup_promoted",
                },
            }
        )
    combined = [promoted, *([promoted_support] if promoted_support else []), *primary_results]
    deduped: list[SearchResult] = []
    seen_ids: set[str] = set()
    for result in combined:
        if result.chunk_id in seen_ids:
            continue
        seen_ids.add(result.chunk_id)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


def _direct_configuration_setup_candidate(
    query: str,
    configuration_result: SearchResult,
    results: list[SearchResult],
) -> SearchResult | None:
    if not _explicit_scope_phrase_groups(query):
        return None
    scored: list[tuple[float, int, SearchResult]] = []
    for index, result in enumerate(results):
        if result.chunk_id == configuration_result.chunk_id:
            continue
        if result.source_document_id != configuration_result.source_document_id:
            continue
        if not _result_supports_explicit_scope(query, result):
            continue
        evidence = " ".join(
            str(part)
            for part in [
                result.content,
                result.metadata.get("content_for_rerank"),
                result.metadata.get("local_rerank_context"),
                result.metadata.get("context_window"),
                result.metadata.get("parent_context"),
            ]
            if part
        )
        normalized = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
        has_setup_action = (
            re.search(r"\b(change|configure|adjust|set)\b.{0,60}\bsettings?\b", normalized) is not None
            or re.search(r"\bsettings?\b.{0,60}\b(change|configure|adjust|set)\b", normalized) is not None
        )
        has_setup_context = (
            "capture environment" in normalized
            or "line camera setting navigation" in normalized
            or "line scan camera" in normalized
        )
        if not (has_setup_action and has_setup_context):
            continue
        chunk_type = str(result.metadata.get("chunk_type") or "")
        score = 1.0
        if chunk_type == "procedure_record":
            score += 1.0
        elif chunk_type == "section_window":
            score += 0.4
        if "capture environment" in normalized:
            score += 1.0
        if "line camera setting navigation" in normalized:
            score += 0.6
        if configuration_result.pages and result.pages:
            distance = min(abs(config_page - result_page) for config_page in configuration_result.pages for result_page in result.pages)
            score += max(0.0, 0.8 - min(distance, 8) * 0.1)
        scored.append((score, index, result))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], -item[1]))[2]


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
            compact_content = re.sub(
                r"[^a-z0-9]+",
                "",
                str(result.metadata.get("rerank_document") or result.metadata.get("content_for_rerank") or result.content or "").lower(),
            )
            symbol_overlap = sum(
                1
                for term in _lexical_table_symbol_terms(list(query_terms))
                if term and term in compact_content
            )
            adjustment += min(0.9, symbol_overlap * 0.3)
            if result.metadata.get("table_header") and not (
                result.metadata.get("table_cell")
                or result.metadata.get("table_key_value")
                or result.metadata.get("table_row_group")
                or result.metadata.get("table_summary")
            ):
                adjustment -= 0.45
            if result.metadata.get("table_cell") or result.metadata.get("table_key_value") or result.metadata.get("table_summary"):
                adjustment += 0.08
        elif chunk_type in {"spec_record", "datasheet_record"}:
            adjustment += 0.04
            if not _has_value_pattern(str(result.content or "")) and len(str(result.content or "").split()) <= 8:
                adjustment -= 0.55
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
    if chunk_type == "warning_record":
        return "safety"
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
    if analysis.safety_intent:
        return ["safety", "procedure", "prose", "context", "table"]
    if _query_has_diagnostic_table_code(analysis.raw_query, analysis):
        return ["table", "context", "procedure", "prose"]
    if "revision_history" in analysis.query_types:
        return ["context", "spec", "table", "prose"]
    if "structured_lookup" in analysis.query_types:
        return ["table", "spec", "context", "prose"]
    if "spec_lookup" in analysis.query_types or "part_lookup" in analysis.query_types:
        return ["spec", "table", "prose", "context"]
    if "comparison" in analysis.query_types or "compatibility" in analysis.query_types:
        return ["spec", "table", "context", "prose"]
    if "how_to" in analysis.query_types or "configuration" in analysis.query_types:
        return ["procedure", "context", "prose", "table"]
    if "operational_flow" in analysis.query_types:
        return ["context", "procedure", "prose", "table"]
    return ["prose", "context", "spec", "table"]


def _allowed_families(analysis: QueryAnalysis) -> set[str]:
    if analysis.safety_intent:
        return {"safety", "procedure", "prose", "context"}
    if _query_has_diagnostic_table_code(analysis.raw_query, analysis):
        return {"table", "context", "procedure", "prose"}
    if "revision_history" in analysis.query_types:
        return {"context", "spec"}
    if "structured_lookup" in analysis.query_types:
        return {"table", "spec", "context"}
    if "spec_lookup" in analysis.query_types or "part_lookup" in analysis.query_types:
        return {"spec", "table", "context"}
    if "comparison" in analysis.query_types or "compatibility" in analysis.query_types:
        return {"spec", "table", "context"}
    if "how_to" in analysis.query_types or "configuration" in analysis.query_types:
        return {"procedure", "context", "prose"}
    if "operational_flow" in analysis.query_types:
        return {"prose", "context", "procedure"}
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
            "warning_record": "safety",
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
    identifier_terms = _query_product_identifier_terms(analysis)
    if identifier_terms:
        identifier_haystack = " ".join(
            str(part)
            for part in [
                result.content,
                result.title,
                " ".join(result.section_path),
                result.metadata.get("rerank_document"),
                result.metadata.get("content_for_rerank"),
                result.metadata.get("product_model"),
                result.metadata.get("product_family"),
                " ".join(str(item) for item in result.metadata.get("product_models") or []),
                " ".join(str(item) for item in result.metadata.get("product_families") or []),
                " ".join(str(item) for item in result.metadata.get("devices") or []),
            ]
            if part
        )
        compact_identifier_haystack = _compact_identifier(identifier_haystack)
        identifier_overlap = sum(1 for term in identifier_terms if term and term in compact_identifier_haystack)
        if identifier_overlap:
            alignment += min(0.42, identifier_overlap * 0.14)
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
    if "structured_lookup" in analysis.query_types and chunk_type == "table_record":
        subject_terms = _structured_lookup_subject_terms(analysis.raw_query)
        field_terms = _structured_lookup_field_terms(analysis.raw_query)
        compact_evidence = re.sub(
            r"[^a-z0-9]+",
            "",
            str(result.metadata.get("rerank_document") or result.metadata.get("content_for_rerank") or result.content or "").lower(),
        )
        if subject_terms:
            subject_overlap = sum(1 for term in subject_terms if term in compact_evidence)
            if subject_overlap:
                alignment += min(0.6, subject_overlap * 0.3)
            elif content_overlap <= 1 and rerank_overlap <= 1:
                alignment -= 0.08
        if field_terms:
            field_overlap = sum(1 for term in field_terms if term in compact_evidence)
            if field_overlap:
                alignment += min(0.28, field_overlap * 0.14)
            elif subject_terms and any(term in compact_evidence for term in subject_terms):
                alignment -= 0.16
    if analysis.safety_intent and "how_to" in analysis.query_types:
        warning_phrase = _safety_warning_phrase(analysis.raw_query)
        if warning_phrase and chunk_type == "warning_record":
            compact_warning_text = re.sub(
                r"[^a-z0-9]+",
                "",
                str(result.metadata.get("rerank_document") or result.metadata.get("content_for_rerank") or result.content or "").lower(),
            )
            if warning_phrase in compact_warning_text:
                alignment += 0.5
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


def _safety_warning_phrase(query: str) -> str:
    match = re.search(
        r"\b(?:warning|caution)\s+about\s+(?P<warning>.+?)\s+for\s+",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    phrase = re.sub(r"\s+", " ", match.group("warning")).strip(" .,:;")
    return re.sub(r"[^a-z0-9]+", "", phrase.lower())


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


def _result_matches_identifier(result: SearchResult, identifier: str) -> bool:
    expected = _compact_identifier(identifier)
    if not expected:
        return False
    haystack = _compact_identifier(
        " ".join(
            str(part)
            for part in [
                result.metadata.get("product_model"),
                result.metadata.get("product_family"),
                " ".join(str(item) for item in result.metadata.get("product_models") or []),
                " ".join(str(item) for item in result.metadata.get("product_families") or []),
                " ".join(str(item) for item in result.metadata.get("devices") or []),
                result.title,
            ]
            if part
        )
    )
    return expected in haystack


def _result_matches_primary_identifier(result: SearchResult, identifier: str) -> bool:
    expected = _compact_identifier(identifier)
    if not expected:
        return False
    separated = r"[^a-z0-9]*".join(re.escape(char) for char in expected)
    bounded_pattern = re.compile(rf"(?<![a-z0-9]){separated}(?![a-z0-9])", flags=re.IGNORECASE)
    primary_values = [
        result.metadata.get("product_model"),
        result.metadata.get("product_family"),
        *(result.metadata.get("product_models") or []),
        *(result.metadata.get("product_families") or []),
        *(result.metadata.get("part_numbers") or []),
        result.title,
    ]
    return any(bounded_pattern.search(str(value)) for value in primary_values if value)


def _comparison_row_code_terms(query: str, identifiers: list[str]) -> list[str]:
    identifier_parts = {
        _compact_identifier(piece)
        for identifier in identifiers
        for piece in [identifier, *re.split(r"[-/_.\s]+", identifier)]
        if _compact_identifier(piece)
    }
    row_terms: list[str] = []
    for match in re.finditer(r"\b[A-Z0-9][A-Z0-9/_.-]*\b", query):
        raw = match.group(0)
        if not any(char.isupper() for char in raw):
            continue
        compact = _compact_identifier(raw)
        if (
            not compact
            or compact in identifier_parts
            or len(compact) < 2
            or compact in {"angle", "code", "data", "error", "format", "measured", "value"}
            or (not any(char.isdigit() for char in compact) and raw != raw.upper())
            or (not any(char.isdigit() for char in compact) and not (3 <= len(compact) <= 8))
        ):
            continue
        if compact not in row_terms:
            row_terms.append(compact)
    return row_terms


def _result_matches_comparison_row_code(result: SearchResult, row_code_terms: list[str]) -> bool:
    if not row_code_terms:
        return True
    tokens = _comparison_row_code_result_tokens(result)
    return any(term in tokens for term in row_code_terms)


def _result_matches_all_comparison_row_codes(result: SearchResult, row_code_terms: list[str]) -> bool:
    if not row_code_terms:
        return True
    tokens = _comparison_row_code_result_tokens(result)
    return all(term in tokens for term in row_code_terms)


def _matching_comparison_row_codes(result: SearchResult, row_code_terms: list[str]) -> set[str]:
    tokens = _comparison_row_code_result_tokens(result)
    return {term for term in row_code_terms if term in tokens}


def _comparison_side_row_code_options(
    row_code_terms: list[str],
    context_terms: set[str],
    *,
    identifiers: list[str] | None = None,
    identifier: str | None = None,
) -> list[list[str]]:
    side_terms = [term for term in row_code_terms if term in context_terms]
    if side_terms:
        return [side_terms]
    if identifiers and identifier and len(row_code_terms) == len(identifiers) and len(identifiers) >= 2:
        try:
            index = identifiers.index(identifier)
        except ValueError:
            index = -1
        if 0 <= index < len(row_code_terms):
            return [[row_code_terms[index]]]
    return [[term] for term in row_code_terms] or [[]]


def _comparison_row_code_result_tokens(result: SearchResult) -> set[str]:
    tokens: set[str] = set()
    row_headers = [str(item) for item in result.metadata.get("table_row_headers") or []]
    if not row_headers:
        row_header_match = re.search(
            r"\bRow headers?:\s*(?P<headers>[^;]+)",
            str(result.content or ""),
            flags=re.IGNORECASE,
        )
        if row_header_match:
            row_headers = [part.strip() for part in row_header_match.group("headers").split(">")]
    for header in row_headers:
        compact_header = _compact_identifier(header)
        if compact_header:
            tokens.add(compact_header)
        for piece in re.split(r"[^A-Za-z0-9]+", header):
            compact_piece = _compact_identifier(piece)
            if compact_piece:
                tokens.add(compact_piece)
    return tokens


def _identifier_context_terms(query: str, identifier: str) -> set[str]:
    def terms_from_text(text: str) -> set[str]:
        extracted: set[str] = set()
        for term in tokenize(text):
            normalized = re.sub(r"[^a-z0-9]+", "", term.lower())
            if len(normalized) < 4 or normalized in LEXICAL_TABLE_STOPWORDS:
                continue
            extracted.add(normalized)
            for piece in re.split(r"[-/_.]+", term.lower()):
                piece = re.sub(r"[^a-z0-9]+", "", piece)
                if len(piece) >= 4 and piece not in LEXICAL_TABLE_STOPWORDS:
                    extracted.add(piece)
        return extracted

    match = re.search(re.escape(identifier), query, flags=re.IGNORECASE)
    if not match:
        return set()
    left_boundary = max(query.rfind(",", 0, match.start()), query.rfind(" and ", 0, match.start()), query.rfind(" with ", 0, match.start()))
    right_candidates = [
        index
        for index in [
            query.find(",", match.end()),
            query.lower().find(" and ", match.end()),
            query.lower().find(" with ", match.end()),
        ]
        if index != -1
    ]
    right_boundary = min(right_candidates) if right_candidates else len(query)
    clause = query[left_boundary + 1 : right_boundary]
    identifier_terms = _text_terms(identifier)
    ignored_context_terms = {
        "action",
        "compare",
        "correct",
        "corrective",
        "documentation",
        "listed",
        "manual",
        "operation",
        "specification",
        "specifications",
    }
    terms = {
        term
        for term in terms_from_text(clause)
        if term not in identifier_terms and term not in ignored_context_terms and not any(char.isdigit() for char in term)
    }
    if terms:
        return terms
    return {
        term
        for term in terms_from_text(query)
        if term not in identifier_terms and term not in ignored_context_terms and not any(char.isdigit() for char in term)
    }


def _comparison_side_context_terms(query: str, identifiers: list[str], identifier: str) -> set[str]:
    if len(identifiers) < 2:
        return _identifier_context_terms(query, identifier)
    try:
        index = identifiers.index(identifier)
    except ValueError:
        return _identifier_context_terms(query, identifier)
    match = re.search(r"\bcompare\b(?P<body>.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if not match:
        return _identifier_context_terms(query, identifier)
    parts = re.split(r"\s+\bwith\b\s+", match.group("body"), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2 or index >= len(parts):
        return _identifier_context_terms(query, identifier)
    side_text = parts[index]
    for other in identifiers:
        side_text = re.sub(re.escape(other), " ", side_text, flags=re.IGNORECASE)
    side_text = re.sub(
        r"\b(?:the|what|which|is|are|listed|specified|shown|given|for|on|in|entry|entries|"
        r"data|tables?|compare|details?)\b",
        " ",
        side_text,
        flags=re.IGNORECASE,
    )
    identifier_terms = _text_terms(identifier)
    terms = {
        term
        for term in _text_terms(side_text)
        if term not in identifier_terms
        and term not in LEXICAL_TABLE_STOPWORDS
        and not (len(term) < 4 and not any(char.isdigit() for char in term))
    }
    return terms or _identifier_context_terms(query, identifier)


def _identifier_context_score(result: SearchResult, context_terms: set[str]) -> float:
    if not context_terms:
        return 0.0
    row_header_text = " ".join(str(item) for item in result.metadata.get("table_row_headers") or [])
    column_header_text = " ".join(str(item) for item in result.metadata.get("table_column_headers") or [])
    haystack = re.sub(
        r"[^a-z0-9]+",
        "",
        " ".join([str(result.content or ""), row_header_text, column_header_text]).lower(),
    )
    return sum(1.0 for term in context_terms if term in haystack)


def _comparison_requested_field_terms(query: str) -> set[str]:
    fields: set[str] = set()
    lowered = query.lower()
    if re.search(r"\bcauses?\b", lowered):
        fields.add("cause")
    if re.search(r"\b(?:remed(?:y|ies)|corrective\s+actions?|corrected|corrections?|fix(?:ed)?)\b", lowered):
        fields.update({"remedy", "correctiveaction"})
    if re.search(r"\b(?:compare|what|which)\s+(?:is\s+the\s+)?error\s+codes?\b", lowered):
        fields.add("errorcode")
    if re.search(r"\bmessages?\b", lowered):
        fields.add("message")
    if re.search(r"\b(?:descriptions?|summary|symbol)\b", lowered):
        fields.update(re.sub(r"[^a-z0-9]+", "", term) for term in re.findall(r"descriptions?|summary|symbol", lowered))
    return {term for term in fields if term}


def _table_result_matches_requested_field_metadata(metadata: dict[str, object], field_terms: set[str]) -> bool:
    if not field_terms:
        return True
    column_header_text = " ".join(str(item) for item in metadata.get("table_column_headers") or [])
    compact_column_headers = re.sub(r"[^a-z0-9]+", "", column_header_text.lower())
    if compact_column_headers:
        return any(term in compact_column_headers for term in field_terms)
    if metadata.get("table_row_group") or metadata.get("table_summary"):
        return True
    return False


def _comparison_result_satisfies_identifier_context(result: SearchResult, context_terms: set[str], field_terms: set[str] | None = None) -> bool:
    if field_terms and not _table_result_matches_requested_field_metadata(result.metadata, field_terms):
        return False
    if not context_terms:
        return True
    if field_terms and field_terms.intersection({"cause", "remedy", "correctiveaction"}):
        required_overlap = max(1, min(4, len(context_terms) // 2 + 1))
    else:
        required_overlap = min(2, len(context_terms))
    return _identifier_context_score(result, context_terms) >= required_overlap


def _comparison_side_match_score(result: SearchResult, setting_phrases: list[str], context_terms: set[str]) -> tuple[float, float, float]:
    return (
        _comparison_setting_phrase_score(result, setting_phrases),
        _identifier_context_score(result, context_terms),
        result.score,
    )


def _is_structured_comparison_table_result(result: SearchResult) -> bool:
    return bool(
        result.metadata.get("table_column_headers")
        or result.metadata.get("table_row_group")
        or result.metadata.get("table_key_value")
        or result.metadata.get("table_summary")
    )


def _promote_comparison_table_candidates(
    primary_results: list[SearchResult],
    supplemental_results: list[SearchResult],
    analysis: QueryAnalysis,
    *,
    limit: int = 12,
) -> list[SearchResult]:
    identifiers = [str(identifier) for identifier in (analysis.product_identifiers or []) if str(identifier)]
    if "comparison" not in analysis.query_types or len(identifiers) < 2 or not supplemental_results:
        return primary_results
    row_code_terms = _comparison_row_code_terms(analysis.raw_query, identifiers)
    setting_phrases = _comparison_setting_phrases(analysis.raw_query)
    field_terms = _comparison_requested_field_terms(analysis.raw_query)
    if len(row_code_terms) < 2:
        row_code_terms = []
    covered_row_codes: set[str] = set()
    for result in primary_results[:5]:
        covered_row_codes.update(_matching_comparison_row_codes(result, row_code_terms))
    uncovered_row_codes = [term for term in row_code_terms if term not in covered_row_codes]
    if row_code_terms and not uncovered_row_codes:
        return primary_results
    promoted: list[SearchResult] = []
    seen: set[str] = set()
    for identifier in identifiers:
        context_terms = _comparison_side_context_terms(analysis.raw_query, identifiers, identifier)
        required_context_terms = set() if row_code_terms else context_terms
        side_row_code_options = _comparison_side_row_code_options(
            uncovered_row_codes or row_code_terms,
            context_terms,
            identifiers=identifiers,
            identifier=identifier,
        )
        side_setting_phrases = _comparison_setting_phrases_for_identifier(analysis.raw_query, identifiers, identifier)
        if not row_code_terms and not context_terms and not field_terms:
            continue
        candidates = [
            result
            for result in supplemental_results
            if result.chunk_id not in seen
            and str(result.metadata.get("chunk_type") or "") == "table_record"
            and _is_structured_comparison_table_result(result)
            and _result_matches_primary_identifier(result, identifier)
            and _comparison_result_matches_setting_phrase(result, side_setting_phrases)
            and any(_result_matches_all_comparison_row_codes(result, option) for option in side_row_code_options)
            and _comparison_result_satisfies_identifier_context(result, required_context_terms, field_terms)
        ]
        existing_matches = [
            result
            for result in primary_results[:5]
            if _result_matches_primary_identifier(result, identifier)
            and _is_structured_comparison_table_result(result)
            and _comparison_result_matches_setting_phrase(result, side_setting_phrases)
            and _comparison_result_satisfies_identifier_context(result, required_context_terms, field_terms)
            and (
                not row_code_terms
                or _matching_comparison_row_codes(result, uncovered_row_codes)
            )
        ]
        if context_terms:
            candidates.sort(
                key=lambda result: (
                    _comparison_side_match_score(result, side_setting_phrases, context_terms),
                ),
                reverse=True,
            )
            existing_matches.sort(
                key=lambda result: (
                    _comparison_side_match_score(result, side_setting_phrases, context_terms),
                ),
                reverse=True,
            )
        if existing_matches and (not candidates or _comparison_side_match_score(existing_matches[0], side_setting_phrases, context_terms) >= _comparison_side_match_score(candidates[0], side_setting_phrases, context_terms)):
            continue
        promotion_limit = 1
        for candidate in candidates[:promotion_limit]:
            seen.add(candidate.chunk_id)
            promoted.append(
                candidate.model_copy(
                    update={
                        "metadata": {
                            **candidate.metadata,
                            "retrieval_stage": "comparison_table_promoted",
                        }
                    }
                )
            )
    if not promoted:
        return primary_results
    combined = [*promoted, *primary_results]
    deduped: list[SearchResult] = []
    seen_ids: set[str] = set()
    for result in combined:
        if result.chunk_id in seen_ids:
            continue
        seen_ids.add(result.chunk_id)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


def _promote_structured_table_candidates(
    primary_results: list[SearchResult],
    supplemental_results: list[SearchResult],
    analysis: QueryAnalysis,
    *,
    limit: int = 12,
) -> list[SearchResult]:
    table_lookup = (
        "structured_lookup" in analysis.query_types
        or "table_record" in analysis.preferred_chunk_types
        or _is_natural_quantity_table_lookup(analysis.raw_query, analysis)
    )
    if (
        not table_lookup
        or analysis.safety_intent
        or "comparison" in analysis.query_types
        or not supplemental_results
    ):
        return primary_results
    candidates = [
        result
        for result in supplemental_results
        if str(result.metadata.get("chunk_type") or "") == "table_record"
        and _is_structured_comparison_table_result(result)
    ]
    if not candidates:
        return primary_results
    promoted = [
        candidate.model_copy(
            update={
                "metadata": {
                    **candidate.metadata,
                    "retrieval_stage": "structured_table_promoted",
                }
            }
        )
        for candidate in candidates[: min(3, limit)]
    ]
    combined = [*promoted, *primary_results]
    deduped: list[SearchResult] = []
    seen_ids: set[str] = set()
    for result in combined:
        if result.chunk_id in seen_ids:
            continue
        seen_ids.add(result.chunk_id)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


def _promote_diagnostic_table_candidates(
    primary_results: list[SearchResult],
    supplemental_results: list[SearchResult],
    analysis: QueryAnalysis,
    *,
    limit: int = 12,
) -> list[SearchResult]:
    if not _query_has_diagnostic_table_code(analysis.raw_query, analysis) or not supplemental_results:
        return primary_results
    candidates = [
        result
        for result in supplemental_results
        if str(result.metadata.get("chunk_type") or "") == "table_record"
        and _is_structured_comparison_table_result(result)
        and _diagnostic_table_support_score(result, analysis)[1] > 0
    ]
    if not candidates:
        return primary_results
    identifiers = [str(identifier) for identifier in (analysis.product_identifiers or []) if str(identifier)]
    if "comparison" in analysis.query_types and len(identifiers) >= 2:
        side_requirements = _diagnostic_side_requirements(analysis.raw_query, identifiers)
        selected: list[SearchResult] = []
        selected_ids: set[str] = set()
        for identifier in identifiers:
            side_codes, context_terms = side_requirements.get(identifier, ([], []))
            side_candidates = [
                result
                for result in candidates
                if result.chunk_id not in selected_ids
                and _result_matches_primary_identifier(result, identifier)
                and (
                    all(_text_contains_diagnostic_code(str(result.content or ""), code) for code in side_codes)
                    if side_codes
                    else bool(context_terms)
                    and _diagnostic_prose_context_score(result, context_terms) >= len(context_terms)
                )
            ]
            if not side_candidates:
                continue
            best = max(
                side_candidates,
                key=lambda result: (
                    _diagnostic_action_field_score(result, analysis.raw_query),
                    _diagnostic_prose_context_score(result, context_terms),
                    _diagnostic_table_support_score(result, analysis),
                ),
            )
            selected.append(best)
            selected_ids.add(best.chunk_id)
        candidates = selected
    else:
        requested_codes = _diagnostic_table_code_terms(analysis.raw_query, analysis)
        if len(requested_codes) >= 2:
            if identifiers:
                candidates = [
                    result
                    for result in candidates
                    if any(_result_matches_primary_identifier(result, identifier) for identifier in identifiers)
                ]
            uncovered_codes = set(requested_codes)
            selected = []
            remaining = list(candidates)
            while uncovered_codes and remaining and len(selected) < 3:
                best = max(
                    remaining,
                    key=lambda result: (
                        sum(1 for code in uncovered_codes if _text_contains_diagnostic_code(str(result.content or ""), code)),
                        _diagnostic_table_support_score(result, analysis),
                    ),
                )
                covered = {
                    code
                    for code in uncovered_codes
                    if _text_contains_diagnostic_code(str(best.content or ""), code)
                }
                if not covered:
                    break
                selected.append(best)
                uncovered_codes.difference_update(covered)
                remaining = [result for result in remaining if result.chunk_id != best.chunk_id]
            candidates = [] if uncovered_codes else selected
    promoted = [
        candidate.model_copy(
            update={
                "score": max(candidate.score, primary_results[0].score if primary_results else candidate.score) + 0.01 - index * 0.001,
                "metadata": {
                    **candidate.metadata,
                    "retrieval_stage": "diagnostic_table_promoted",
                },
            }
        )
        for index, candidate in enumerate(candidates[: min(3, limit)])
    ]
    combined = [*promoted, *primary_results]
    deduped: list[SearchResult] = []
    seen_ids: set[str] = set()
    for result in combined:
        if result.chunk_id in seen_ids:
            continue
        seen_ids.add(result.chunk_id)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


def _diagnostic_table_support_score(result: SearchResult, analysis: QueryAnalysis) -> tuple[int, int, int, int, float]:
    content = str(result.content or "").lower()
    codes = _diagnostic_table_code_terms(analysis.raw_query, analysis)
    exact_code_count = sum(1 for code in codes if _text_contains_diagnostic_code(content, code))
    evidence_role_count = sum(
        1
        for label in ("error message", "message", "cause", "remedy", "corrective action")
        if re.search(rf"\b{re.escape(label)}s?\s*:", content)
    )
    return (
        exact_code_count,
        evidence_role_count,
        int(bool(result.metadata.get("table_key_value"))),
        int(bool(result.metadata.get("table_row_group"))),
        result.score,
    )


def _requested_troubleshooting_roles(query: str) -> set[str]:
    roles: set[str] = set()
    if re.search(r"\b(?:cause|causes|caused|why|reason)\b", query, flags=re.IGNORECASE):
        roles.add("cause")
    if re.search(r"\b(?:error\s+message|message\s+(?:is|says|shown|displayed))\b", query, flags=re.IGNORECASE):
        roles.add("message")
    if re.search(
        r"\b(?:corrective\s+action|remed(?:y|ies)|fix|what\s+should\b|what\s+(?:do|does|can)\b)\b",
        query,
        flags=re.IGNORECASE,
    ):
        roles.add("action")
    return roles


def _direct_pipe_parent_match_score(result: SearchResult, analysis: QueryAnalysis) -> float:
    """Score a parent only when one contiguous pipe row binds the query to requested roles."""
    if str(result.metadata.get("chunk_type") or "") != "parent_section":
        return 0.0
    roles = _requested_troubleshooting_roles(analysis.raw_query)
    if not roles or "|" not in result.content:
        return 0.0

    requested_identifiers = list(getattr(analysis, "product_identifiers", []) or [])
    if len(requested_identifiers) > 1:
        return 0.0
    if requested_identifiers:
        source_identifiers = {
            _compact_identifier(str(value))
            for key in ("product_model", "product_family", "part_number", "product_models", "product_families", "devices")
            for raw_value in [result.metadata.get(key)]
            for value in (raw_value if isinstance(raw_value, list) else [raw_value])
            if value
        }
        if _compact_identifier(requested_identifiers[0]) not in source_identifiers:
            return 0.0

    role_headers = {
        "cause": {"cause", "reason"},
        "message": {"message", "error message"},
        "action": {"action", "corrective action", "remedy", "remedies", "fix"},
    }
    identity_headers = {
        "condition", "symptom", "fault", "alarm", "error", "error code", "error number",
        "message", "error message", "status",
    }
    recognized_headers = identity_headers.union(*(values for values in role_headers.values()))
    query_terms = _query_terms(analysis).difference(_query_product_identifier_terms(analysis))

    def cells(line: str) -> list[str]:
        values = [value.strip() for value in line.strip().split("|")]
        if values and not values[0]:
            values = values[1:]
        if values and not values[-1]:
            values = values[:-1]
        return values

    headers: list[str] | None = None
    best = 0.0
    for raw_line in result.content.splitlines():
        line = raw_line.strip()
        if not line:
            headers = None
            continue
        if "|" not in line:
            headers = None
            continue
        values = cells(line)
        normalized = [re.sub(r"\s+", " ", value.lower()).strip() for value in values]
        recognized_count = sum(value in recognized_headers for value in normalized if value)
        if (
            len(values) >= 2
            and recognized_count >= 2
            and any(value in identity_headers for value in normalized)
        ):
            headers = normalized
            continue
        if headers is None or len(values) != len(headers):
            continue
        identity_indexes = [index for index, header in enumerate(headers) if header in identity_headers]
        role_indexes = {
            role: [index for index, header in enumerate(headers) if header in role_headers[role]]
            for role in roles
        }
        if not identity_indexes or any(not indexes for indexes in role_indexes.values()):
            continue
        if any(not any(values[index].strip(" -") for index in indexes) for indexes in role_indexes.values()):
            continue
        identity_terms = set().union(*(_text_terms(values[index]) for index in identity_indexes))
        material_identity_terms = identity_terms.difference({"condition", "error", "fault", "message", "status"})
        if not material_identity_terms:
            continue
        overlap = query_terms.intersection(identity_terms)
        required_overlap = len(identity_terms) if len(identity_terms) <= 4 else max(2, len(identity_terms) - 1)
        if required_overlap and len(overlap) >= required_overlap:
            best = max(best, len(overlap) / max(1, len(identity_terms)))
    return best


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
        if result.chunk_id not in deduped:
            deduped[result.chunk_id] = result
    return list(deduped.values())[: max(limit, 10)]


def _promote_direct_pipe_parents(
    primary_results: list[SearchResult],
    candidates: list[SearchResult],
    analysis: QueryAnalysis,
    *,
    limit: int,
) -> list[SearchResult]:
    unique_candidates = {result.chunk_id: result for result in candidates}
    matches = sorted(
        ((score, result) for result in unique_candidates.values() if (score := _direct_pipe_parent_match_score(result, analysis)) > 0),
        key=lambda item: (item[0], item[1].score),
        reverse=True,
    )
    if len(matches) != 1:
        return primary_results[:limit]
    score, parent = matches[0]
    promoted = parent.model_copy(
        update={
            "score": max(parent.score, primary_results[0].score if primary_results else parent.score),
            "metadata": {
                **parent.metadata,
                "retrieval_stage": "direct_pipe_parent_promoted",
                "direct_pipe_parent_match_score": score,
            },
        }
    )
    return [promoted, *(result for result in primary_results if result.chunk_id != promoted.chunk_id)][:limit]


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
        source_context_parts = [
            part
            for part in [
                result.content,
                table_row_group_context,
                context.get(2),
                context.get(3)
                if str(result.metadata.get("chunk_type") or "") in {"atomic_text", "procedure_record", "table_record"}
                else None,
            ]
            if part
        ]
        content_update: dict[str, object] = {}
        if source_context_parts:
            source_context = "\n\n".join(dict.fromkeys(str(part) for part in source_context_parts))
            if source_context and source_context != result.content:
                metadata["content"] = source_context
                if str(result.metadata.get("chunk_type") or "") == "atomic_text" and context.get(3):
                    content_update["content"] = source_context
        assembled.append(result.model_copy(update={"metadata": metadata, **content_update}))
    return assembled


def retrieve(query: str, corpus_ids: list[str], filters: dict[str, object], limit: int = 10) -> list[SearchResult]:
    store = QdrantStore()
    analysis = analyze_query(query)
    search_filters, metadata_document_hits = select_documents_from_metadata(store, query, corpus_ids, filters)
    chunk_search_filters = _chunk_search_filters(filters, search_filters, analysis)
    broad_vector_enabled = _should_run_broad_vector_search(analysis)
    dense_results = (
        _annotate_stage_metadata(run_dense_search(store, query, corpus_ids, chunk_search_filters), "dense")
        if broad_vector_enabled
        else []
    )
    sparse_results = (
        _annotate_stage_metadata(run_sparse_search(store, query, corpus_ids, chunk_search_filters), "sparse")
        if broad_vector_enabled
        else []
    )
    table_results = (
        _annotate_stage_metadata(run_table_search(store, query, corpus_ids, chunk_search_filters), "table")
        if _should_run_extra_table_vector_search(analysis)
        else []
    )
    table_lexical_results = (
        _annotate_stage_metadata(run_table_lexical_search(query, corpus_ids, filters, analysis), "table_lexical")
        if _should_run_table_lexical_search(analysis)
        else []
    )
    contextual_lexical_results = _annotate_stage_metadata(
        run_contextual_lexical_search(query, corpus_ids, filters, analysis),
        "contextual_lexical",
    )
    special_results = _annotate_stage_metadata(run_special_search(store, query, corpus_ids, chunk_search_filters, analysis), "special")
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
    family_selected = _annotate_stage_metadata(_select_family_candidates(aligned, analysis, filters=chunk_search_filters, limit=12), "family_selected")
    enriched = enrich_candidates_for_rerank(family_selected, analysis, limit=12)
    reranked = _annotate_stage_metadata(rerank_results(enriched, query, limit=12), "reranked")
    reranked = _promote_direct_pipe_parents(
        reranked,
        [*family_selected, *dense_results, *sparse_results, *special_results, *contextual_lexical_results],
        analysis,
        limit=12,
    )
    reranked = _promote_structured_table_candidates(reranked, table_lexical_results, analysis, limit=12)
    reranked = _promote_comparison_table_candidates(reranked, table_lexical_results, analysis, limit=12)
    reranked = _promote_direct_configuration_candidates(
        reranked,
        [*contextual_lexical_results, *dense_results, *sparse_results, *special_results],
        analysis,
        limit=12,
    )
    reranked = _promote_diagnostic_table_candidates(reranked, table_lexical_results, analysis, limit=12)
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
