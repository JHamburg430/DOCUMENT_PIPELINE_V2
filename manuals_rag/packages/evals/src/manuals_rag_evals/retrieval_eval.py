from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from manuals_rag_common.config import settings


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "what",
    "when",
    "where",
    "which",
    "does",
    "about",
    "manual",
    "guide",
    "datasheet",
    "document",
    "product",
    "section",
    "step",
    "warning",
    "caution",
    "note",
}

GENERIC_ANCHORS = {
    "use",
    "used",
    "using",
    "click",
    "only",
    "one",
    "value",
    "values",
    "displayed",
    "enter",
    "following",
    "remove",
    "right",
    "left",
    "desired",
    "simply",
    "specified",
    "specifies",
    "shown",
    "show",
    "button",
    "setting",
    "settings",
    "information",
    "data",
    "output",
    "input",
}

TECHNICAL_VERBS = {
    "connect",
    "configure",
    "set",
    "install",
    "select",
    "enable",
    "disable",
    "mount",
    "measure",
    "adjust",
    "trigger",
    "capture",
    "transmit",
    "receive",
    "assign",
    "register",
}

GENERIC_TECHNICAL_TERMS = {
    "warning",
    "caution",
    "parameter",
    "command",
    "protocol",
    "revision",
    "version",
    "voltage",
    "current",
    "pressure",
    "temperature",
    "accuracy",
    "resolution",
    "repeatability",
    "tolerance",
    "configuration",
}

USER_STYLE_QUERY_SYSTEM_PROMPT = """
You generate realistic retrieval benchmark queries for technical documents.

Return strict JSON with this shape:
{"queries":[{"query":"...","intent":"...","reason":"..."}]}

Rules:
- Write queries a real technician, engineer, operator, purchaser, or integrator might type into search.
- Base every query on the provided context, especially the source snippet, structured fields, labels, and extracted terms.
- Do not say "this document", "this manual", "the datasheet", "this section", or similar.
- Do not use meta phrasing like "what specification", "what value is listed", "where does", "what does the document say", or "which step in".
- Do not mirror the source text mechanically or copy long spans verbatim.
- Keep each query concise and natural, usually under 12 words.
- Prefer search-style phrasing:
  model + parameter
  model + task
  parameter + unit
  warning condition
  short natural questions engineers actually ask
- If the snippet contains explicit fields, labels, units, steps, warnings, or settings, use those concrete concepts in the query.
- Prefer concrete terms from the snippet such as field names, units, menu labels, protocol names, settings, or actions.
- Do not invent facts not present in the input.
- Return 2 or 3 diverse queries when the context is strong, otherwise return 1.
- Return at most 3 queries.
""".strip()


@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    query: str
    source_document_id: str
    document_version_id: str
    source_chunk_id: str
    source_title: str
    source_filename: str
    chunk_type: str
    section_path: str
    page_from: int
    page_to: int
    expected_terms: list[str]
    expected_snippet: str
    generation_method: str
    source_metadata: dict[str, Any]
    benchmark_quality: str = "validated"
    anchor_terms: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-/.]+", text.lower())


def _looks_specific(token: str) -> bool:
    if token in STOPWORDS or token in GENERIC_ANCHORS:
        return False
    if len(token) >= 3 and any(char.isdigit() for char in token):
        return True
    if token in GENERIC_TECHNICAL_TERMS:
        return True
    if token in TECHNICAL_VERBS:
        return True
    if re.fullmatch(r"[a-z]{2,}\d+[a-z0-9-]*", token):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?(?:v|a|ma|w|kw|mm|cm|m|ms|s|hz|khz|mhz|fps|kg|g|n|mpa|deg|c|°c|%)", token):
        return True
    if re.fullmatch(r"[a-z]+(?:/[a-z0-9-]+)+", token):
        return True
    if re.fullmatch(r"[a-z]+-[a-z0-9-]+", token):
        return True
    return len(token) >= 5


def _is_high_signal_anchor(token: str) -> bool:
    return (
        any(char.isdigit() for char in token)
        or token in GENERIC_TECHNICAL_TERMS
        or token in TECHNICAL_VERBS
        or "[" in token
        or "/" in token
        or "-" in token
        or re.fullmatch(r"\d+(?:\.\d+)?(?:v|a|ma|w|kw|mm|cm|m|ms|s|hz|khz|mhz|fps|kg|g|n|mpa|deg|c|°c|%)", token) is not None
    )


def extract_anchor_terms(content: str, *, limit: int = 6) -> list[str]:
    candidates = tokenize(content)
    terms: list[str] = []
    for token in candidates:
        if not _looks_specific(token):
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def content_preview(content: str, *, limit: int = 220) -> str:
    preview = re.sub(r"\s+", " ", content).strip()
    return preview[:limit]


def _quoted_menu_labels(content: str) -> list[str]:
    labels = re.findall(r"\[[^\]]+\]", content)
    return [label.strip() for label in labels if len(label.strip()) >= 3]


def _field_value_pairs(content: str) -> list[tuple[str, str]]:
    return re.findall(r"([A-Za-z][A-Za-z0-9 /_()-]{1,40})\s*:\s*([^\n;]{1,80})", content)


def _looks_like_toc_line(content: str) -> bool:
    compact = normalize_text(content)
    if "procedure step" in compact and re.search(r"\d+-\d+\b", compact):
        return True
    if re.search(r"\.{4,}\s*\d+(?:-\d+)?\b", content):
        return True
    return False


def _looks_like_legal_boilerplate(content: str) -> bool:
    compact = normalize_text(content)
    phrases = (
        "other than as stated herein",
        "no other warranties whatsoever",
        "all express implied and statutory warranties",
        "fitness for a particular purpose",
        "merchantability",
        "products/samples are provided",
    )
    return any(phrase in compact for phrase in phrases)


def _has_concrete_technical_signal(content: str, anchors: list[str], chunk_type: str) -> bool:
    field_pairs = _field_value_pairs(content)
    has_units_or_values = any(re.search(r"\b\d+(?:\.\d+)?(?:v|a|ma|w|kw|mm|cm|m|ms|s|hz|khz|mhz|fps|kg|g|n|mpa|°c|c|%)\b", token) for token in tokenize(content))
    has_action = any(token in TECHNICAL_VERBS for token in tokenize(content))
    has_menu_labels = bool(_quoted_menu_labels(content))
    strong_anchor_count = sum(1 for anchor in anchors if _is_high_signal_anchor(anchor))
    if chunk_type in {"spec_record", "datasheet_record", "table_record"}:
        return bool(field_pairs or has_units_or_values or strong_anchor_count >= 1)
    if chunk_type == "procedure_record":
        return bool(has_action and strong_anchor_count >= 1)
    if chunk_type == "warning_record":
        return bool(strong_anchor_count >= 1)
    return bool((field_pairs or has_units_or_values or has_menu_labels or has_action) and strong_anchor_count >= 1)


def chunk_is_queryworthy(chunk: dict[str, Any], anchors: list[str]) -> bool:
    content = str(chunk.get("content", "")).strip()
    chunk_type = str(chunk.get("chunk_type", ""))
    if len(content) < 40:
        return False
    if _looks_like_toc_line(content):
        return False
    if _looks_like_legal_boilerplate(content):
        return False
    if len(anchors) < 1:
        return False
    return _has_concrete_technical_signal(content, anchors, chunk_type)


def _query_specificity_score(query: str, expected_terms: list[str]) -> int:
    score = 0
    tokens = tokenize(query)
    if any(any(char.isdigit() for char in token) for token in tokens):
        score += 2
    if any(token in GENERIC_TECHNICAL_TERMS for token in tokens):
        score += 2
    if len(expected_terms) >= 2:
        score += 2
    if any(_looks_specific(term) for term in expected_terms):
        score += 2
    if len(query) >= 28:
        score += 1
    return score


def _query_looks_document_bound(query: str) -> bool:
    lowered = normalize_text(query)
    banned_phrases = {
        "this document",
        "this manual",
        "the manual",
        "this datasheet",
        "the datasheet",
        "this section",
        "where does",
        "what does",
        "which step in",
        "what procedure in",
        "what warning does",
    }
    return any(phrase in lowered for phrase in banned_phrases)


def _query_looks_meta(query: str) -> bool:
    lowered = normalize_text(query)
    banned_prefixes = (
        "what specification",
        "what value",
        "where does",
        "what does",
        "which step in",
        "what procedure",
        "what warning",
        "in ",
    )
    if any(lowered.startswith(prefix) for prefix in banned_prefixes):
        return True
    banned_fragments = {
        "is listed for",
        "does the table say about",
        "refers to",
        "mentions",
        "discuss",
    }
    return any(fragment in lowered for fragment in banned_fragments)


def validate_eval_case(query: str, chunk: dict[str, Any], anchors: list[str]) -> tuple[bool, str]:
    chunk_type = str(chunk.get("chunk_type", ""))
    if not anchors:
        return False, "no_specific_anchor"
    if _query_looks_document_bound(query):
        return False, "document_bound_query"
    if _query_looks_meta(query):
        return False, "meta_query"
    if chunk_type == "atomic_text":
        if len(anchors) < 2:
            return False, "atomic_requires_two_anchors"
        if not any(_is_high_signal_anchor(anchor) for anchor in anchors):
            return False, "atomic_requires_high_signal_anchor"
    if _query_specificity_score(query, anchors[:4]) < 5:
        return False, "low_specificity"
    return True, "validated"


def build_query_candidates(chunk: dict[str, Any]) -> list[tuple[str, str]]:
    content = str(chunk["content"]).strip()
    title = str(chunk.get("title", "")).strip()
    model = str(chunk.get("product_model") or chunk.get("metadata_json", {}).get("product_model") or "").strip()
    chunk_type = str(chunk.get("chunk_type", ""))
    anchors = extract_anchor_terms(content)
    labels = _quoted_menu_labels(content)
    if labels and labels[0] not in anchors:
        anchors = [labels[0], *anchors][:6]
    if not anchors:
        return []

    label = model or title or "this document"
    primary = anchors[0]
    secondary = anchors[1] if len(anchors) > 1 else primary
    candidates: list[tuple[str, str]] = []

    if chunk_type in {"spec_record", "datasheet_record"}:
        candidates.append((f"{label} {primary}", "spec_primary"))
        candidates.append((f"{primary} for {label}", "spec_value"))
        if secondary != primary:
            candidates.append((f"{label} {primary} {secondary}", "spec_multi"))
    elif chunk_type == "table_record":
        candidates.append((f"{label} {primary}", "table_primary"))
        if secondary != primary:
            candidates.append((f"{primary} {secondary} {label}", "table_multi"))
    elif chunk_type == "procedure_record":
        candidates.append((f"how to {primary} {label}", "procedure_howto"))
        if secondary != primary:
            candidates.append((f"{primary} {secondary} steps {label}", "procedure_step"))
        candidates.append((f"{primary} procedure {label}", "procedure_describe"))
    elif chunk_type == "warning_record":
        candidates.append((f"{label} warning {primary}", "warning_primary"))
        if secondary != primary:
            candidates.append((f"{label} caution {primary} {secondary}", "warning_caution"))
    else:
        if len(anchors) >= 2:
            candidates.append((f"{label} {primary} {secondary}", "general_multi"))
        candidates.append((f"{label} {primary}", "general_primary"))

    return candidates


def _structured_eval_input(chunk: dict[str, Any], anchors: list[str]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata_json", {}))
    content = str(chunk.get("content", ""))
    field_matches = _field_value_pairs(content)
    return {
        "chunk_type": str(chunk.get("chunk_type", "")),
        "title": str(chunk.get("title", "")).strip(),
        "product_model": str(chunk.get("product_model") or metadata.get("product_model") or "").strip(),
        "section_path": str(chunk.get("section_path_text", "")).strip(),
        "anchors": anchors[:6],
        "menu_labels": _quoted_menu_labels(content)[:3],
        "field_value_pairs": [
            {"field": field.strip(), "value": value.strip()}
            for field, value in field_matches[:6]
        ],
        "expected_terms": anchors[:4],
        "snippet": content_preview(content, limit=420),
    }


def _parse_generated_queries(payload: str) -> list[dict[str, str]]:
    data = json.loads(payload)
    queries = data.get("queries", [])
    parsed: list[dict[str, str]] = []
    for item in queries:
        query = str(item.get("query", "")).strip()
        intent = str(item.get("intent", "")).strip() or "llm_user_style"
        reason = str(item.get("reason", "")).strip()
        if query:
            parsed.append({"query": query, "intent": intent, "reason": reason})
    return parsed


def generate_user_style_queries(
    chunk: dict[str, Any],
    *,
    anchors: list[str],
    fallback_candidates: list[tuple[str, str]],
    limit: int,
) -> list[tuple[str, str]]:
    prompt = {
        "document_title": str(chunk.get("title", "")).strip(),
        "source_filename": str(chunk.get("source_filename", "")).strip(),
        "structured_input": _structured_eval_input(chunk, anchors),
        "fallback_examples": [{"query": query, "intent": method} for query, method in fallback_candidates[:3]],
    }
    try:
        with httpx.Client(base_url=settings.ollama_url, timeout=60.0) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": settings.ollama_eval_model,
                    "prompt": f"{USER_STYLE_QUERY_SYSTEM_PROMPT}\n\nInput: {json.dumps(prompt, ensure_ascii=True)}",
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            generated = _parse_generated_queries(str(payload.get("response", "{}")))
    except Exception:
        generated = []

    queries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in generated:
        normalized = normalize_text(item["query"])
        if normalized in seen:
            continue
        is_valid, _ = validate_eval_case(item["query"], chunk, anchors)
        if not is_valid:
            continue
        seen.add(normalized)
        queries.append((item["query"], item["intent"]))
        if len(queries) >= limit:
            return queries
    for query, method in fallback_candidates:
        normalized = normalize_text(query)
        if normalized in seen:
            continue
        seen.add(normalized)
        queries.append((query, method))
        if len(queries) >= limit:
            break
    return queries


def build_eval_cases_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    max_cases: int,
    per_chunk_limit: int = 3,
    use_llm_generation: bool = True,
) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    for chunk in chunks:
        anchors = extract_anchor_terms(str(chunk["content"]))
        if not chunk_is_queryworthy(chunk, anchors):
            continue
        fallback_candidates = build_query_candidates(chunk)[:per_chunk_limit]
        candidates = (
            generate_user_style_queries(
                chunk,
                anchors=anchors,
                fallback_candidates=fallback_candidates,
                limit=per_chunk_limit,
            )
            if use_llm_generation
            else fallback_candidates
        )
        if len(anchors) < 1:
            continue
        for index, (query, method) in enumerate(candidates, start=1):
            is_valid, quality = validate_eval_case(query, chunk, anchors)
            if not is_valid:
                continue
            cases.append(
                RetrievalEvalCase(
                    case_id=f"{chunk['id']}::{index}",
                    query=query,
                    source_document_id=str(chunk["source_document_id"]),
                    document_version_id=str(chunk["document_version_id"]),
                    source_chunk_id=str(chunk["id"]),
                    source_title=str(chunk.get("title", "")),
                    source_filename=str(chunk.get("source_filename", "")),
                    chunk_type=str(chunk.get("chunk_type", "")),
                    section_path=str(chunk.get("section_path_text", "")),
                    page_from=int(chunk.get("page_from", 0)),
                    page_to=int(chunk.get("page_to", 0)),
                    expected_terms=anchors[:4],
                    expected_snippet=content_preview(str(chunk["content"])),
                    generation_method=method,
                    source_metadata=dict(chunk.get("metadata_json", {})),
                    benchmark_quality=quality,
                    anchor_terms=anchors[:4],
                )
            )
            if len(cases) >= max_cases:
                return cases
    return cases


def _result_term_overlap(result_content: str, expected_terms: list[str]) -> int:
    haystack = normalize_text(result_content)
    return sum(1 for term in expected_terms if term and term in haystack)


def score_document_selection(
    case: RetrievalEvalCase,
    results: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    selected_hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results[:top_k]:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        for hit in metadata.get("selected_document_metadata_hits", []) or []:
            document_id = str(hit.get("source_document_id") or "")
            if not document_id or document_id in seen:
                continue
            selected_hits.append(hit)
            seen.add(document_id)

    if not selected_hits:
        return {
            "attempted": False,
            "passed": False,
            "rank": None,
            "expected_source_document_id": case.source_document_id,
            "selected_source_document_ids": [],
            "hit_count": 0,
            "failure_category": "metadata_selection_not_recorded",
        }

    selected_ids = [str(hit.get("source_document_id") or "") for hit in selected_hits]
    rank = next(
        (index for index, document_id in enumerate(selected_ids, start=1) if document_id == case.source_document_id),
        None,
    )
    return {
        "attempted": True,
        "passed": rank is not None and rank <= top_k,
        "rank": rank,
        "expected_source_document_id": case.source_document_id,
        "selected_source_document_ids": selected_ids[:top_k],
        "hit_count": len(selected_hits),
        "failure_category": None if rank is not None and rank <= top_k else "metadata_document_miss",
    }


def score_search_results(
    case: RetrievalEvalCase,
    results: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    document_selection = score_document_selection(case, results, top_k=top_k)
    considered = results[:top_k]
    found_same_document = False
    found_chunk_family = False
    max_overlap = 0
    for rank, result in enumerate(considered, start=1):
        same_document = str(result.get("source_document_id", "")) == case.source_document_id
        same_chunk = str(result.get("chunk_id", "")) == case.source_chunk_id
        same_section = " / ".join(result.get("section_path", [])) == case.section_path
        overlap = _result_term_overlap(str(result.get("content", "")), case.expected_terms)
        max_overlap = max(max_overlap, overlap)
        result_chunk_type = str(result.get("metadata", {}).get("chunk_type") or result.get("chunk_type", ""))
        if same_document:
            found_same_document = True
        if result_chunk_type == case.chunk_type:
            found_chunk_family = True
        if same_chunk:
            return {
                "passed": True,
                "rank": rank,
                "match_reason": "exact_chunk",
                "overlap_terms": overlap,
                "failure_category": None,
                "retrieval_stage": "final_top_k",
                "candidate_recall": True,
                "metadata_document_selection": document_selection,
            }
        if same_document and same_section and overlap >= max(2, min(3, len(case.expected_terms))):
            return {
                "passed": True,
                "rank": rank,
                "match_reason": "same_section_term_overlap",
                "overlap_terms": overlap,
                "failure_category": None,
                "retrieval_stage": "final_top_k",
                "candidate_recall": True,
                "metadata_document_selection": document_selection,
            }
        if same_document and overlap >= max(2, min(3, len(case.expected_terms))):
            return {
                "passed": True,
                "rank": rank,
                "match_reason": "same_document_term_overlap",
                "overlap_terms": overlap,
                "failure_category": None,
                "retrieval_stage": "final_top_k",
                "candidate_recall": True,
                "metadata_document_selection": document_selection,
            }
    failure_category = "candidate_miss"
    if found_same_document:
        failure_category = "ranking_or_context_loss"
    elif found_chunk_family and max_overlap >= 1:
        failure_category = "wrong_document_or_filter_loss"
    return {
        "passed": False,
        "rank": None,
        "match_reason": "no_match",
        "overlap_terms": max_overlap,
        "failure_category": failure_category,
        "retrieval_stage": "final_top_k",
        "candidate_recall": found_same_document,
        "metadata_document_selection": document_selection,
    }
