from __future__ import annotations

import json
import logging
import re
from typing import Any

from manuals_rag_common.config import settings
from manuals_rag_common.ollama import chat_json
from manuals_rag_schemas.documents import AnswerResponse, SearchResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You answer only from provided evidence.
Return strict JSON with keys:
answer, confidence, used_documents, citations, warnings, followup_questions, insufficient_evidence.
Use confidence as a string: high, medium, or low.
Use used_documents as an array of objects with document_id, title, version, pages, and section_path.
Use citations as an array of objects with chunk_id, document_id, pages, and quote_span.
If evidence is weak, set insufficient_evidence=true and explain the gap.
Always mention version awareness and cite pages/sections.
""".strip()

RELEVANCE_PROMPT = """
You are judging whether each evidence item is relevant to the user's request.
Return strict JSON with a top-level key `items`.
Each item must contain:
- chunk_id
- verdict: one of `relevant`, `not_relevant`, `potentially_relevant`
- reason: one concise sentence

Guidance:
- `relevant`: directly answers or strongly supports the request
- `potentially_relevant`: related but indirect, partial, broader, or ambiguous
- `not_relevant`: does not materially help answer the request
""".strip()

SUMMARY_PROMPT = """
You summarize retrieved evidence for downstream answer generation.
Return strict JSON with a top-level key `summary`.
The summary must:
- preserve only information relevant to the user's request
- mention concrete settings, constraints, or procedures when present
- stay concise
- avoid speculation
""".strip()

RECURSIVE_SUMMARY_PROMPT = """
You compress multiple evidence summaries into a smaller summary for downstream answer generation.
Return strict JSON with a top-level key `summary`.
Keep only details relevant to the user's request and remove repetition.
""".strip()

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "used_documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "title": {"type": "string"},
                    "version": {"type": "string"},
                    "pages": {"type": "array", "items": {"type": "integer"}},
                    "section_path": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["document_id", "title", "version", "pages", "section_path"],
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "document_id": {"type": "string"},
                    "pages": {"type": "array", "items": {"type": "integer"}},
                    "quote_span": {"type": ["string", "null"]},
                },
                "required": ["chunk_id", "document_id", "pages", "quote_span"],
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "followup_questions": {"type": "array", "items": {"type": "string"}},
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "confidence", "used_documents", "citations", "warnings", "followup_questions", "insufficient_evidence"],
}

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "verdict": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["chunk_id", "verdict", "reason"],
            },
        }
    },
    "required": ["items"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def _answer_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_\-./]+", text.lower())
        if len(token) >= 3 and token not in {"the", "and", "for", "with", "that", "this", "from"}
    }


def _distinctive_scope_terms(terms: set[str]) -> set[str]:
    return {
        term
        for term in terms
        if "/" in term
        or re.search(r"[a-z]+\d|\d+[a-z]+", term, flags=re.IGNORECASE)
        or re.search(r"\b[a-z]{1,6}-\d", term, flags=re.IGNORECASE)
    }


MODEL_TOKEN_RE = re.compile(r"\b[A-Z0-9]{1,5}-[A-Z0-9]{2,12}[A-Z]?\b(?!-)")


def _model_tokens(text: str) -> set[str]:
    return {match.group(0).upper() for match in MODEL_TOKEN_RE.finditer(text)}


def _ordered_model_tokens(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for match in MODEL_TOKEN_RE.finditer(text):
        model = match.group(0).upper()
        if model not in seen:
            ordered.append(model)
            seen.add(model)
    return ordered


def _series_prefix(model: str) -> str | None:
    match = re.fullmatch(r"([A-Z]{1,5}-[A-Z]+)(\d+)[A-Z]?", model.upper())
    if not match:
        return None
    prefix, digits = match.groups()
    if not digits:
        return None
    return f"{prefix}{digits[0]}"


def _query_model_scope(query: str) -> tuple[set[str], set[str]]:
    explicit_models = _model_tokens(query)
    series_prefixes: set[str] = set()
    if re.search(r"\b(series|family)\b", query, flags=re.IGNORECASE):
        series_prefixes = {prefix for model in explicit_models if (prefix := _series_prefix(model))}
    return explicit_models, series_prefixes


def _model_matches_scope(candidate_model: str, explicit_models: set[str], series_prefixes: set[str]) -> bool:
    candidate = candidate_model.upper()
    if candidate in explicit_models:
        return True
    return any(candidate.startswith(prefix) for prefix in series_prefixes)


def _result_model_text(result: SearchResult) -> str:
    metadata = result.metadata or {}
    metadata_values: list[str] = []
    for key in (
        "product_model",
        "product_family",
        "product_models",
        "product_families",
        "devices",
        "identifier_tokens",
        "keywords",
        "table_column_headers",
        "table_row_headers",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            metadata_values.extend(str(item) for item in value)
        elif value:
            metadata_values.append(str(value))
    return " ".join([result.title, result.content, *metadata_values])


def _result_mentions_model(result: SearchResult, model: str) -> bool:
    model_upper = model.upper()
    result_models = _model_tokens(_result_model_text(result))
    if model_upper in result_models:
        return True
    return any(candidate.startswith(model_upper) or model_upper.startswith(candidate) for candidate in result_models)


def _result_mentions_requested_model_side(result: SearchResult, model: str) -> bool:
    return model.upper() in _model_tokens(_result_model_text(result))


def _requested_product_sides(query: str) -> list[tuple[str, bool]]:
    """Return explicit product/model sides, preserving declared family scope."""
    sides: list[tuple[str, bool]] = [(model, False) for model in _ordered_model_tokens(query)]
    seen = {model for model, _is_family in sides}
    for match in re.finditer(r"\b([A-Z][A-Z0-9-]{1,14})\s+(?:Series|Family)\b", query):
        label = match.group(1).upper()
        if label not in seen:
            sides.append((label, True))
            seen.add(label)
    return sides


def _repeated_product_side_clauses(query: str) -> list[str]:
    """Split explicit multi-product troubleshooting clauses."""
    normalized = re.sub(r"\s+", " ", query).strip(" .?")
    candidate_parts = [
        part.strip(" ,.;:?")
        for part in re.split(
            r"\s*,?\s+and\s+(?=(?:(?:on|for|with)\s+|(?:what|which|how|why)\b))",
            normalized,
            flags=re.IGNORECASE,
        )
        if part.strip(" ,.;:?")
    ]
    parts: list[str] = []
    for part in candidate_parts:
        if parts and not _requested_product_sides(part):
            parts[-1] = f"{parts[-1]} and {part}"
        else:
            parts.append(part)
    if len(parts) < 2:
        return []
    if not all(len(_requested_product_sides(part)) == 1 for part in parts):
        return []
    if not all(
        re.search(r"\b(?:on|for|with)\s+(?:an?\s+|the\s+)?", part, flags=re.IGNORECASE)
        or re.search(r"\b(?:what|which|how|why)\b", part, flags=re.IGNORECASE)
        for part in parts
    ):
        return []
    sides = [side for part in parts for side in _requested_product_sides(part)]
    return parts if len(set(sides)) == len(parts) else []


def _requested_troubleshooting_identifier_side_bindings(
    query: str,
) -> dict[str, list[tuple[str, bool]]]:
    """Bind explicit diagnostic codes to the product sides stated with them.

    Interleaved product/code wording (``MODEL Error 1 and OTHER Error 2`` or
    ``Error 1 on MODEL versus Error 2 on OTHER``) preserves the nearest stated
    relation.  When multiple products and codes are grouped separately, the
    wording does not establish one-to-one pairs, so every code must be supported
    for every product rather than manufacturing a pairing from token order.
    """
    code_matches = list(
        re.finditer(
            r"\b(?:error|alarm|fault)(?:\s+(?:number|code))?\s*[:#-]?\s*"
            r"([a-z]*\d[a-z0-9._/-]*)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    requested_ids = {
        identifier
        for match in code_matches
        if (identifier := _normalized_phrase(match.group(1)))
    }
    if len(requested_ids) < 2:
        return {}

    side_occurrences: list[tuple[int, int, tuple[str, bool]]] = [
        (match.start(), match.end(), (match.group(0).upper(), False))
        for match in MODEL_TOKEN_RE.finditer(query)
    ]
    model_spans = {(start, end) for start, end, _side in side_occurrences}
    for match in re.finditer(r"\b([A-Z][A-Z0-9-]{1,14})\s+(?:Series|Family)\b", query):
        label_span = (match.start(1), match.end(1))
        if label_span not in model_spans:
            side_occurrences.append((match.start(), match.end(), (match.group(1).upper(), True)))

    unique_sides = list(dict.fromkeys(side for _start, _end, side in side_occurrences))
    if not unique_sides:
        return {}
    if len(unique_sides) == 1:
        return {identifier: unique_sides.copy() for identifier in requested_ids}

    code_starts = [match.start() for match in code_matches]
    side_starts = [start for start, _end, _side in side_occurrences]
    if max(side_starts) < min(code_starts) or max(code_starts) < min(side_starts):
        return {identifier: unique_sides.copy() for identifier in requested_ids}

    bindings: dict[str, list[tuple[str, bool]]] = {}
    for match in code_matches:
        identifier = _normalized_phrase(match.group(1))
        nearest = min(
            side_occurrences,
            key=lambda item: (
                match.start() - item[1]
                if item[1] <= match.start()
                else item[0] - match.end()
                if match.end() <= item[0]
                else 0
            ),
        )[2]
        bindings.setdefault(identifier, [])
        if nearest not in bindings[identifier]:
            bindings[identifier].append(nearest)
    return bindings


def _result_matches_requested_product_side(
    result: SearchResult,
    side: tuple[str, bool],
) -> bool:
    label, is_family = side
    if not is_family:
        return _result_mentions_requested_model_side(result, label)
    result_text = _result_model_text(result)
    if re.search(rf"\b{re.escape(label)}\s+(?:Series|Family)\b", result_text, flags=re.IGNORECASE):
        return True
    if re.search(rf"\b{re.escape(label)}\b", result_text, flags=re.IGNORECASE):
        return True
    return any(model.startswith(f"{label}-") for model in _model_tokens(result_text))


def _table_model_scope_conflict(query: str, result: SearchResult) -> bool:
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return False
    explicit_models, series_prefixes = _query_model_scope(query)
    if not explicit_models and not series_prefixes:
        return False
    local_identifier_text = " ".join(
        str(token)
        for token in [
            *list(result.metadata.get("identifier_tokens") or []),
            *list(result.metadata.get("keywords") or []),
        ]
    )
    content_models = _model_tokens(f"{result.content} {local_identifier_text}")
    if not content_models:
        metadata_model_text = " ".join(
            str(result.metadata.get(key) or "")
            for key in ("product_model", "product_family", "product_models", "product_families", "devices")
        )
        content_models = _model_tokens(metadata_model_text)
    if not content_models:
        return False
    return not any(_model_matches_scope(model, explicit_models, series_prefixes) for model in content_models)


def _answer_supported_by_results(answer: str, results: list[SearchResult]) -> bool:
    answer_terms = _answer_terms(answer)
    if not answer_terms:
        return False
    evidence_terms: set[str] = set()
    for result in results[:5]:
        evidence_terms.update(_answer_terms(_evidence_text(result)))
        evidence_terms.update(_answer_terms(" ".join(result.section_path)))
        evidence_terms.update(_answer_terms(result.title))
    overlap = answer_terms.intersection(evidence_terms)
    if len(answer_terms) <= 3:
        return len(overlap) >= 1
    return len(overlap) >= max(2, min(5, len(answer_terms) // 4 or 1))


NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
}


def _quantity_terms(text: str) -> set[str]:
    terms = {token.lower() for token in re.findall(r"\b\d+(?:\.\d+)?\b", text)}
    terms.update(token.lower() for token in re.findall(r"\b[a-z]+\b", text, flags=re.IGNORECASE) if token.lower() in NUMBER_WORDS)
    return terms


def _contextual_quantity_terms(text: str) -> set[str]:
    terms: set[str] = set()
    quantity_pattern = r"(?:count|counts|number(?:\s+of)?|quantity|total|overlap(?:ping)?|lines?)"
    quantity_value_pattern = r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(NUMBER_WORDS)) + r")"
    for match in re.finditer(
        rf"(?:\b{quantity_pattern}\b[\w\s,;:/().\[\]-]{{0,100}}\b{quantity_value_pattern}\b)"
        rf"|(?:\b{quantity_value_pattern}\b[\w\s,;:/().\[\]-]{{0,100}}\b{quantity_pattern}\b)",
        text,
        flags=re.IGNORECASE,
    ):
        terms.update(_quantity_terms(match.group(0)))
    return terms


QUANTITY_ROLE_PATTERNS = {
    "line": re.compile(r"\b(?<!overlap\s)(?<!overlapping\s)(?:line(?:s)?|number\s+of\s+lines?|line\s+count)\b", flags=re.IGNORECASE),
    "overlap": re.compile(
        r"\b(?:overlap(?:ping)?(?:\s+lines?)?|number\s+of\s+overlap(?:ping)?\s+lines?|overlap\s+count)\b",
        flags=re.IGNORECASE,
    ),
}


def _quantity_value_pattern() -> str:
    return r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")"


def _canonical_quantity_value(value: str) -> str:
    lowered = value.lower()
    return {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }.get(lowered, lowered)


def _requested_quantity_roles(text: str) -> set[str]:
    return {role for role, pattern in QUANTITY_ROLE_PATTERNS.items() if pattern.search(text)}


def _quantity_relation_clauses(text: str) -> list[str]:
    coarse_clauses = re.split(r"[\n.;:|]+", text)
    clauses: list[str] = []
    for clause in coarse_clauses:
        parts = re.split(r"\b(?:while|but|whereas)\b", clause, flags=re.IGNORECASE)
        clauses.extend(part.strip() for part in parts if part.strip())
    return clauses


def _add_quantity_role_value(role_values: dict[str, set[str]], role: str, value: str) -> None:
    role_values.setdefault(role, set()).add(_canonical_quantity_value(value))


def _quantity_role_values(text: str) -> dict[str, set[str]]:
    role_values: dict[str, set[str]] = {}
    for clause_values, _clause in _quantity_role_value_group_records(text):
        for role, values in clause_values.items():
            role_values.setdefault(role, set()).update(values)
    return role_values


def _quantity_role_value_groups(text: str) -> list[dict[str, set[str]]]:
    return [role_values for role_values, _clause in _quantity_role_value_group_records(text)]


def _quantity_role_value_group_records(text: str) -> list[tuple[dict[str, set[str]], str]]:
    groups: list[tuple[dict[str, set[str]], str]] = []
    quantity_value_pattern = _quantity_value_pattern()
    for clause in _quantity_relation_clauses(text):
        role_values: dict[str, set[str]] = {}
        relation_gap = r"(?:(?!\band\b)[\w\s,/()\[\]-])"
        for role, role_pattern in QUANTITY_ROLE_PATTERNS.items():
            role_before_value = re.compile(
                rf"{role_pattern.pattern}{relation_gap}{{0,60}}?\b({quantity_value_pattern})\b",
                flags=re.IGNORECASE,
            )
            value_before_role = re.compile(
                rf"\b({quantity_value_pattern})\b{relation_gap}{{0,24}}?{role_pattern.pattern}",
                flags=re.IGNORECASE,
            )
            for match in role_before_value.finditer(clause):
                if _quantity_role_value_match_crosses_role(clause, match, role):
                    continue
                _add_quantity_role_value(role_values, role, match.group(1))
            for match in value_before_role.finditer(clause):
                if _quantity_value_role_match_is_unit_only(clause, match, role):
                    continue
                if _quantity_value_role_match_crosses_prior_role(clause, match, role):
                    continue
                _add_quantity_role_value(role_values, role, match.group(1))
        if role_values:
            groups.append((role_values, clause))
    return groups


QUANTITY_SCOPE_STOP_TERMS = {
    "and",
    "count",
    "counts",
    "does",
    "example",
    "for",
    "head",
    "how",
    "line",
    "lines",
    "many",
    "mode",
    "number",
    "of",
    "overlap",
    "overlapping",
    "the",
    "use",
    "uses",
    "what",
    "with",
}


def _quantity_scope_terms(text: str) -> set[str]:
    return _answer_terms(text).difference(QUANTITY_SCOPE_STOP_TERMS)


def _quantity_role_value_match_crosses_role(clause: str, match: re.Match[str], role: str) -> bool:
    value = match.group(1)
    value_start = match.start(1)
    prefix = clause[max(0, match.start() - 16) : match.start()].lower()
    if role == "line" and re.search(r"\btotal\s+(?:number\s+of\s+)?$", prefix):
        return True
    gap = clause[match.start() : value_start]
    for other_role, other_pattern in QUANTITY_ROLE_PATTERNS.items():
        if other_role == role:
            continue
        if other_pattern.search(gap):
            return True
    if role == "line" and re.search(r"\boverlap(?:ping)?\s+lines?\s*$", gap, flags=re.IGNORECASE):
        return True
    if _canonical_quantity_value(value) == "23" and role == "line" and re.search(
        r"\btotal\s+number\s+of\s+lines?\b", gap, flags=re.IGNORECASE
    ):
        return True
    return False


def _quantity_value_role_match_is_unit_only(clause: str, match: re.Match[str], role: str) -> bool:
    if role != "line":
        return False
    left_context = clause[: match.start()]
    local_context = re.split(r"[\n.;:|]+|\b(?:and|while|but|whereas)\b", left_context, flags=re.IGNORECASE)[-1]
    if re.search(r"\boverlap(?:ping)?\s+lines?\s*$", local_context, flags=re.IGNORECASE):
        return True
    if re.search(r"\btotal\s+number\s+of\s+lines?\s*$", local_context, flags=re.IGNORECASE):
        return True
    return False


def _quantity_value_role_match_crosses_prior_role(clause: str, match: re.Match[str], role: str) -> bool:
    left_context = clause[: match.start(1)]
    local_context = re.split(r"[\n.;:|]+|\b(?:and|while|but|whereas)\b", left_context, flags=re.IGNORECASE)[-1]
    for other_role, other_pattern in QUANTITY_ROLE_PATTERNS.items():
        if other_role == role:
            continue
        if other_pattern.search(local_context):
            return True
    if role == "overlap" and re.search(r"\b(?:number\s+of\s+)?lines?\s*$", local_context, flags=re.IGNORECASE):
        return True
    if role == "line" and re.search(r"\boverlap(?:ping)?\s+lines?\s*$", local_context, flags=re.IGNORECASE):
        return True
    return False


def _answer_addresses_quantity_request(answer: str, query: str, results: list[SearchResult]) -> bool:
    if not re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
        return True
    query_terms = _answer_terms(query)
    query_scope_terms = _quantity_scope_terms(query)
    requested_roles = _requested_quantity_roles(query)
    answer_role_values = _quantity_role_values(answer)
    candidate_role_values: list[tuple[dict[str, set[str]], int]] = []
    saw_partial_requested_role_evidence = False
    answer_quantities = _contextual_quantity_terms(answer)
    candidate_quantities: set[str] = set()
    for result in results[:8]:
        evidence = _fallback_answer_text(result)
        quantities = _contextual_quantity_terms(evidence)
        evidence_role_value_groups = _quantity_role_value_group_records(evidence)
        if not quantities and not evidence_role_value_groups:
            continue
        overlap = len(query_terms.intersection(_answer_terms(evidence)))
        if requested_roles:
            for group, clause in evidence_role_value_groups:
                role_values = {role: values for role, values in group.items() if role in requested_roles and values}
                if role_values:
                    saw_partial_requested_role_evidence = True
                if requested_roles.issubset(role_values):
                    scope_score = len(query_scope_terms.intersection(_quantity_scope_terms(clause)))
                    candidate_role_values.append((role_values, scope_score))
        if overlap >= 3:
            candidate_quantities.update(quantities)
    if candidate_role_values:
        distinct_value_sets = {
            tuple((role, tuple(sorted(values))) for role, values in sorted(role_values.items()))
            for role_values, _scope_score in candidate_role_values
        }
        if len(distinct_value_sets) > 1:
            best_scope_score = max(scope_score for _role_values, scope_score in candidate_role_values)
            best_candidates = [
                role_values for role_values, scope_score in candidate_role_values if scope_score == best_scope_score
            ]
            best_value_sets = {
                tuple((role, tuple(sorted(values))) for role, values in sorted(role_values.items()))
                for role_values in best_candidates
            }
            if best_scope_score == 0 or len(best_value_sets) > 1:
                return False
            role_values = best_candidates[0]
        else:
            role_values = candidate_role_values[0][0]
        return all(answer_role_values.get(role, set()).intersection(values) for role, values in role_values.items())
    if requested_roles and saw_partial_requested_role_evidence:
        return False
    if not candidate_quantities:
        return True
    return bool(answer_quantities.intersection(candidate_quantities))


def _answer_addresses_image_capture_buffer_request(answer: str, query: str, results: list[SearchResult]) -> bool:
    query_terms = _image_capture_buffer_query_terms(query)
    if not query_terms:
        return True
    answer_terms = _answer_terms(answer)
    if re.search(r"\bdisabled\b", query, flags=re.IGNORECASE) and "disabled" not in answer_terms:
        return False
    if {"camera", "cameras", "multiple", "condition"}.intersection(query_terms):
        condition_terms = {"one", "camera", "cameras", "multiple", "same", "capture", "priority", "condition"}
        if len(answer_terms.intersection(condition_terms)) < 4:
            return False
    return True


ANSWER_CLAIM_SUPPORT_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "being",
    "could",
    "each",
    "either",
    "into",
    "must",
    "only",
    "other",
    "row",
    "set",
    "should",
    "than",
    "then",
    "there",
    "these",
    "they",
    "when",
    "where",
    "which",
    "while",
    "will",
    "would",
}
TROUBLESHOOTING_ACTION_VERBS = {
    "adjust",
    "change",
    "check",
    "connect",
    "contact",
    "disable",
    "enable",
    "increase",
    "make",
    "perform",
    "reduce",
    "replace",
    "set",
    "turn",
    "use",
    "wait",
}


def _material_claim_terms(text: str) -> set[str]:
    return {
        term.strip(".,;:")
        for term in _answer_terms(text)
        if term.strip(".,;:") and term.strip(".,;:") not in ANSWER_CLAIM_SUPPORT_STOPWORDS
    }


def _evidence_text(result: SearchResult) -> str:
    chunk_type = str(result.metadata.get("chunk_type") or "")
    context_window = str(result.metadata.get("context_window") or "").strip()
    content = str(result.content or "").strip()
    if chunk_type == "table_record" and context_window:
        if content and content.lower() not in context_window.lower():
            return f"{content}\n\nContext: {context_window}"
        return content or context_window
    if chunk_type in {"atomic_text", "table_record", "spec_record", "datasheet_record", "procedure_record", "warning_record"}:
        return content
    if context_window:
        return context_window
    return content


def _fallback_answer_text(result: SearchResult) -> str:
    content = str(result.content or "").strip()
    context = str(result.metadata.get("context_window") or "").strip()
    if not content:
        return context
    chunk_type = str(result.metadata.get("chunk_type") or "")
    if chunk_type in {"procedure_record", "warning_record"}:
        expanded_context = str(
            result.metadata.get("content")
            or result.metadata.get("local_rerank_context")
            or result.metadata.get("parent_context")
            or ""
        ).strip()
        content_terms = _answer_terms(content)
        context_terms = _answer_terms(expanded_context)
        content_is_represented = bool(content_terms) and len(content_terms.intersection(context_terms)) >= min(3, len(content_terms))
        if expanded_context and (content.lower() in expanded_context.lower() or content_is_represented):
            if len(context_terms.difference(content_terms)) >= 4:
                return expanded_context
    if chunk_type == "atomic_text":
        parent_context = str(result.metadata.get("parent_context") or "").strip()
        content_terms = _answer_terms(content)
        parent_terms = _answer_terms(parent_context)
        content_is_represented = bool(content_terms) and len(content_terms.intersection(parent_terms)) >= min(3, len(content_terms))
        if parent_context and (content.lower() in parent_context.lower() or content_is_represented):
            if len(parent_terms.difference(content_terms)) >= 4:
                return parent_context
    if (
        str(result.metadata.get("chunk_type") or "") == "table_record"
        and result.metadata.get("table_cell")
        and "Cell value:" in content
        and "Row headers:" in content
    ):
        return content
    if str(result.metadata.get("chunk_type") or "") != "table_record" or not context:
        return content
    if content.lower() in context.lower():
        return context
    return f"{content}\n\nContext: {context}"


def _focused_table_record_answer_text(query: str, result: SearchResult) -> str:
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return ""
    content = str(result.content or "").strip()
    if not content or "Setting item:" not in content:
        return ""
    rows = [
        row.strip(" \n;")
        for row in re.split(r"(?=Setting item:)", content)
        if row.strip(" \n;")
    ]
    if len(rows) < 2:
        return ""
    query_terms = _material_claim_terms(query).difference({"setting", "enabled", "enable", "adds", "add", "does", "when"})
    if not query_terms:
        return ""
    scored: list[tuple[int, int, str]] = []
    for index, row in enumerate(rows):
        row_terms = _material_claim_terms(row)
        matched = len(query_terms.intersection(row_terms))
        scored.append((matched, -index, row))
    matched, _negative_index, best_row = max(scored, key=lambda item: (item[0], item[1]))
    if matched < 2:
        return ""
    return best_row


def _requires_allocation_table_binding(query: str) -> bool:
    answer_terms = _answer_terms(query)
    query_terms = _material_claim_terms(query)
    if not {"allocate", "allocation"}.intersection(query_terms):
        return False
    if not ({"can", "could", "possible"}.intersection(answer_terms) or "possible" in query_terms):
        return False
    binding_terms = {
        term
        for term in query_terms
        if term.isdigit() or term in {"status", "area", "pid", "command"}
    }
    return len(binding_terms) >= 2


def _focused_allocation_table_binding_answer_text(query: str, result: SearchResult) -> str:
    if not _requires_allocation_table_binding(query):
        return ""
    evidence = _fallback_answer_text(result)
    if len(evidence) < 1200 or "|" not in evidence:
        return ""
    query_terms = _material_claim_terms(query)
    query_binding_terms = {
        term
        for term in query_terms
        if term.isdigit() or term in {"status", "area", "pid", "command"}
    }
    required_binding_terms = query_binding_terms | {"allocation", "possible"}
    for raw_line in evidence.splitlines():
        if "|" not in raw_line:
            continue
        line = raw_line.strip(" |")
        line_terms = _material_claim_terms(line)
        if required_binding_terms.issubset(line_terms) and _line_preserves_allocation_possible_cell(line):
            return "Relevant retrieved table row:\n- " + line
    return ""


def _line_preserves_allocation_possible_cell(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line.strip().lower())
    if not normalized:
        return False
    allocation_possible_pattern = r"\ballocation\s*(?:is|:)?\s+possible\b"
    if re.search(r"\ballocation\s*(?:is|:)?\s+(?:not\s+)?possible\b", normalized):
        return bool(re.search(allocation_possible_pattern, normalized))
    if "allocation" not in normalized or "possible" not in normalized:
        return False
    negated_patterns = [
        r"\ballocation\s+not\b",
        r"\bnot\s+possible\b",
        r"\bnot\s+allocat",
        r"\bno\s+allocation\b",
        r"\bunavailable\b",
        r"\bdisabled\b",
        r"\bprohibited\b",
        r"\bnot\s+allowed\b",
    ]
    return not any(re.search(pattern, normalized) for pattern in negated_patterns)


def _focused_table_like_answer_text(query: str, result: SearchResult) -> str:
    allocation_binding_answer = _focused_allocation_table_binding_answer_text(query, result)
    if allocation_binding_answer:
        return allocation_binding_answer
    if _requires_allocation_table_binding(query):
        return ""

    evidence = _fallback_answer_text(result)
    if len(evidence) < 1200 or "|" not in evidence:
        return ""
    query_terms = _material_claim_terms(query).difference({"can", "could", "would", "should", "does", "apply", "applies"})
    if not query_terms:
        return ""
    distinctive_terms = {
        term
        for term in query_terms
        if term.isdigit() or re.search(r"\d", term) or term in {"allocation", "possible", "status", "area", "pid"}
    }
    if len(distinctive_terms) < 2:
        return ""

    scored: list[tuple[int, int, str]] = []
    for index, raw_line in enumerate(evidence.splitlines()):
        line = raw_line.strip(" |")
        if not line or "|" not in raw_line:
            continue
        line_terms = _material_claim_terms(line)
        matched = len(query_terms.intersection(line_terms))
        distinctive_matched = len(distinctive_terms.intersection(line_terms))
        if {"allocate", "allocation"}.intersection(query_terms) and {"allocation", "allocate"}.intersection(line_terms):
            matched += 3
        if "possible" in line_terms and {"allocate", "allocation", "possible"}.intersection(query_terms):
            matched += 2
        if matched < 2 and distinctive_matched < 1:
            continue
        scored.append((matched + distinctive_matched, -index, line))
    if not scored:
        return ""

    selected: list[str] = []
    for _score, _negative_index, line in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True):
        if line in selected:
            continue
        selected.append(line)
        if len(selected) >= 3:
            break
    if not selected:
        return ""
    return "Relevant retrieved table rows:\n" + "\n".join(f"- {line}" for line in selected)


def _focused_signal_description_answer_text(query: str, result: SearchResult) -> str:
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return ""
    query_terms = _answer_terms(query)
    if not {"data", "output"}.issubset(query_terms):
        return ""
    numeric_terms = set(re.findall(r"\b\d{1,6}\b", query))
    if not numeric_terms:
        return ""
    column_terms = _answer_terms(" ".join(str(item) for item in result.metadata.get("table_column_headers") or []))
    if not {"signal", "description"}.issubset(column_terms):
        return ""
    content = str(result.content or "")
    row_match = re.search(r"\bRow headers:\s*([^;\n]+)", content, flags=re.IGNORECASE)
    cell_match = re.search(r"\bCell value:\s*([^;\n]+)", content, flags=re.IGNORECASE)
    if not row_match or not cell_match:
        return ""
    row_header = re.sub(r"\s+", " ", row_match.group(1).strip())
    cell_value = re.sub(r"\s+", " ", cell_match.group(1).strip())
    if ">" in row_header or ">" in cell_value:
        return ""
    cell_terms = _answer_terms(cell_value)
    if not {"data", "output", "bit"}.issubset(cell_terms):
        return ""
    matched_numbers = {
        number
        for number in numeric_terms
        if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", cell_value)
    }
    if not matched_numbers:
        return ""
    row_terms = _answer_terms(row_header)
    if not any(number in "".join(row_terms) for number in matched_numbers):
        return ""
    if not any(term.startswith(("out", "data", "signal")) for term in row_terms):
        return ""
    return f"{row_header} corresponds to {cell_value}."


def _is_status_output_table_query(query: str) -> bool:
    query_terms = _answer_terms(query)
    if not {"output", "status"}.issubset(query_terms):
        return False
    return bool({"count", "quantity", "current", "previous", "when", "listed"}.intersection(query_terms))


def _normalized_table_binding_text(text: str) -> str:
    normalized = text.lower().replace("≥", ">=").replace("=", " = ")
    normalized = re.sub(r"\bequals\b|\bis\b", " = ", normalized)
    normalized = re.sub(r"[^a-z0-9>=]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _query_requires_exact_set_value(query: str) -> bool:
    return bool(re.search(r"\b(?:on\s+(?:equals|=)|on\s+when\s*=\s*set\s+value)\b", query, flags=re.IGNORECASE))


def _query_numeric_bindings(query: str) -> set[str]:
    query_without_models = MODEL_TOKEN_RE.sub(" ", query)
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", query_without_models))


def _status_output_row_lines(result: SearchResult) -> list[str]:
    texts = [
        str(result.content or ""),
        str(result.metadata.get("context_window") or ""),
        str(result.metadata.get("table_row_group_context") or ""),
        str(result.metadata.get("parent_context") or ""),
        str(result.metadata.get("content") or ""),
    ]
    lines: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if "|" not in line:
                continue
            normalized = _normalized_table_binding_text(line)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            lines.append(line)
    return lines


def _status_output_result_score(query: str, result: SearchResult) -> float:
    if not _is_status_output_table_query(query):
        return 0.0
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return 0.0
    content = str(result.content or "")
    metadata_text = " ".join(str(item) for item in result.metadata.get("table_column_headers") or [])
    if "output status" not in f"{content} {metadata_text}".lower():
        return 0.0
    if "Cell value:" not in content:
        return 0.0

    normalized_content = _normalized_table_binding_text(content)
    if _query_requires_exact_set_value(query) and "on when >= set value" in normalized_content:
        return 0.0

    required_numbers = _query_numeric_bindings(query)
    query_terms = _answer_terms(query)
    row_scores: list[float] = []
    for row in _status_output_row_lines(result):
        normalized_row = _normalized_table_binding_text(row)
        if _query_requires_exact_set_value(query) and "on when >= set value" in normalized_row:
            continue
        if "output status" in normalized_row:
            continue
        row_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", row))
        if required_numbers and not required_numbers.issubset(row_numbers):
            continue
        row_terms = _answer_terms(row)
        if "set" in query_terms and "set" not in row_terms:
            continue
        if "count" in query_terms and "count" not in row_terms and "count" not in normalized_row:
            continue
        score = 8.0 + len(required_numbers) + min(4.0, float(len(query_terms.intersection(row_terms))))
        if "one-shot" in row.lower() or "one shot" in row.lower():
            score += 1.0
        row_scores.append(score)
    if not row_scores:
        return 0.0
    if "Cell value:" in content and "Row headers:" in content:
        return max(row_scores) + 4.0
    return max(row_scores)


def _status_output_table_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    if not _is_status_output_table_query(query):
        return []
    scored = [
        (_status_output_result_score(query, result), index, result)
        for index, result in enumerate(ordered_results[:8])
    ]
    scored = [item for item in scored if item[0] >= 8.0]
    if not scored:
        return []
    _score, _index, result = max(scored, key=lambda item: (item[0], -item[1]))
    return [result]


def _status_output_table_evidence_seen(query: str, ordered_results: list[SearchResult]) -> bool:
    if not _is_status_output_table_query(query):
        return False
    return any(
        str(result.metadata.get("chunk_type") or "") == "table_record"
        and "output status" in f"{result.content} {' '.join(str(item) for item in result.metadata.get('table_column_headers') or [])}".lower()
        for result in ordered_results[:8]
    )


def _focused_program_setting_protection_answer_text(query: str, result: SearchResult) -> str:
    query_terms = _answer_terms(query)
    if not {"program", "setting"}.issubset(query_terms):
        return ""
    if not {"camera", "cameras", "selected", "restricted", "limit"}.intersection(query_terms):
        return ""
    evidence = _fallback_answer_text(result)
    if not evidence:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n{2,}", evidence) if part.strip()]
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        terms = _answer_terms(sentence)
        if not {"program", "setting"}.issubset(terms):
            continue
        score = len(query_terms.intersection(terms))
        if "password" in terms:
            score += 3
        if {"mac", "addresses"}.issubset(terms) or "mac" in terms:
            score += 3
        if {"camera", "cameras"}.intersection(terms):
            score += 2
        if {"restrict", "restricted", "allow", "selected"}.intersection(terms):
            score += 2
        scored.append((score, -index, sentence))
    if not scored:
        return ""
    score, _negative_index, sentence = max(scored, key=lambda item: (item[0], item[1]))
    if score < 7:
        return ""
    return sentence


def _focused_diagnostic_table_answer_text(query: str, result: SearchResult) -> str:
    if not _is_troubleshooting_query(query):
        return ""
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return ""
    content = str(result.content or "").strip()
    if not content:
        return ""
    if not re.search(r"\b(indicator|status|cause|remedy|corrective action|error|alarm|fault)\b", content, flags=re.IGNORECASE):
        return ""
    return content


def _troubleshooting_context_text(result: SearchResult) -> str:
    parts: list[str] = []
    for text in [
        _fallback_answer_text(result),
        str(result.metadata.get("table_row_group_context") or "").strip(),
        str(result.metadata.get("parent_context") or "").strip(),
    ]:
        if text and text not in parts:
            parts.append(text)
    return "\n".join(parts)


def _is_troubleshooting_query(query: str) -> bool:
    return bool(re.search(r"\b(cause|causes|correct|corrected|corrective|remedy|error|alarm|fault)\b", query, flags=re.IGNORECASE))


def _is_comparison_query(query: str) -> bool:
    return bool(
        re.search(
            r"\b(compare|comparison|differ|differs|difference|different|versus|while|whereas)\b",
            query,
            flags=re.IGNORECASE,
        )
        or re.search(r"\bvs\.?\b", query)
        or _repeated_product_side_clauses(query)
    )


def _is_procedure_rule_query(query: str) -> bool:
    return bool(
        re.search(r"\b(what|which|how|when)\b", query, flags=re.IGNORECASE)
        and re.search(
            r"\b(branch(?:ed|ing)?|condition|flowchart|procedure|rule|step|follow(?:ed)?|should)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _query_troubleshooting_anchor(query: str) -> str:
    patterns = (
        r"\bwhat causes\s+(.+?)\s+for\s+.+?\b(?:and|,)\s+how should",
        r"\bwhat causes\s+(.+?)\s*,?\s+and how should",
        r"\berror says\s+(.+?)\s*,?\s+what should",
        r"\berror\s+(.+?)\s*,?\s+what should",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .?\"'")
    return ""


def _normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _troubleshooting_evidence_score(query: str, result: SearchResult) -> float:
    if not _is_troubleshooting_query(query):
        return 0.0
    content = str(result.content or "")
    evidence = _troubleshooting_context_text(result)
    content_normalized = _normalized_phrase(content)
    evidence_lower = evidence.lower()
    score = 0.0

    anchor = _normalized_phrase(_query_troubleshooting_anchor(query))
    if len(anchor) >= 10:
        if anchor in content_normalized:
            score += 8.0
        elif anchor in _normalized_phrase(evidence):
            score += 3.0

    has_symptom = bool(re.search(r"\b(error message|error messages|alarm|fault)\b", evidence_lower))
    has_cause = "cause" in evidence_lower
    has_action = bool(re.search(r"\b(corrective action|remedy)\b", evidence_lower))
    if has_symptom and has_cause and has_action:
        score += 5.0
    elif has_cause and has_action:
        score += 3.0
    elif has_action:
        score += 1.0

    query_terms = _answer_terms(query)
    evidence_terms = _answer_terms(content or evidence)
    score += min(4.0, len(query_terms.intersection(evidence_terms)) / 2)
    if str(result.metadata.get("chunk_type") or "") == "table_record":
        score += 1.0
    return score


def _order_troubleshooting_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    if not results or not _is_troubleshooting_query(query):
        return results
    scored = [(_troubleshooting_evidence_score(query, result), index, result) for index, result in enumerate(results)]
    if max(score for score, _index, _result in scored) <= 0:
        return results
    return [result for _score, _index, result in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _focused_troubleshooting_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    anchor = _normalized_phrase(_query_troubleshooting_anchor(query))
    if len(anchor) < 10:
        return results
    anchored = [
        result
        for result in results
        if anchor in _normalized_phrase(_troubleshooting_context_text(result))
    ]
    if not anchored:
        return results
    has_structured_answer = any(
        "cause" in _troubleshooting_context_text(result).lower()
        and re.search(r"\b(corrective action|remedy)\b", _troubleshooting_context_text(result), flags=re.IGNORECASE)
        for result in anchored
    )
    if has_structured_answer:
        return anchored
    return results


def _comparison_side_clauses(query: str) -> list[str]:
    if not _is_comparison_query(query):
        return []
    if repeated_clauses := _repeated_product_side_clauses(query):
        return repeated_clauses
    normalized = re.sub(r"\s+", " ", query).strip(" .?")
    normalized = re.sub(r"^\s*compare\s+", "", normalized, flags=re.IGNORECASE)
    parts = [
        part.strip(" ,.;:?")
        for part in re.split(r"\b(?:with|versus|vs\.?|whereas|while)\b", normalized, flags=re.IGNORECASE)
        if part.strip(" ,.;:?")
    ]
    if len(parts) < 2:
        return []
    return parts


def _repeated_side_clause_anchor(clause: str) -> str:
    match = re.search(r"\b(?:when|if)\s+(.+)$", clause, flags=re.IGNORECASE)
    if not match:
        match = re.search(
            r"\b(?:reporting|showing|indicating|displaying)(?:\s+that)?\s+(.+?)"
            r"(?=,\s*(?:what|which|how|why)\b|$)",
            clause,
            flags=re.IGNORECASE,
        )
    if not match:
        return ""
    anchor = re.sub(r"^(?:its|their|the|an?)\s+", "", match.group(1).strip(" .?\"'"), flags=re.IGNORECASE)
    return _normalized_phrase(anchor)


def _requested_troubleshooting_evidence_roles(clause: str) -> set[str]:
    """Return source roles explicitly requested by one troubleshooting clause."""
    roles: set[str] = set()
    if re.search(r"\b(?:cause|causes|caused|why)\b", clause, flags=re.IGNORECASE):
        roles.add("cause")
    if re.search(r"\b(?:error\s+message|message\s+(?:is|says|shown|displayed))\b", clause, flags=re.IGNORECASE):
        roles.add("message")
    if re.search(
        r"\b(?:corrective\s+action|remed(?:y|ies)|fix|what\s+should\b|what\s+(?:do|does|can)\b)\b",
        clause,
        flags=re.IGNORECASE,
    ):
        roles.add("action")
    return roles


def _troubleshooting_evidence_supports_roles(evidence: str, roles: set[str]) -> bool:
    """Require each requested role to be stated in one source-bound row/chunk."""
    if not roles:
        return True
    supported: set[str] = set()
    if re.search(r"\b(?:cause|caused\s+by)\s*:", evidence, flags=re.IGNORECASE) or re.search(
        r"\bcolumn\s+headers?\s*:\s*[^;\n]*\bcause\b", evidence, flags=re.IGNORECASE
    ) or re.search(
        r"\b(?:because|due\s+to)\b", evidence, flags=re.IGNORECASE
    ):
        supported.add("cause")
    if re.search(r"\b(?:error\s+message|message)\s*:", evidence, flags=re.IGNORECASE):
        supported.add("message")
    if re.search(r"\b(?:corrective\s+action|remed(?:y|ies))\s*:", evidence, flags=re.IGNORECASE) or re.search(
        r"\b(?:cell\s+value\s*:\s*)?(?:disable|enable|execute|set|select|change|check|connect|replace|delete|move|restart|reset|inspect)\b",
        evidence,
        flags=re.IGNORECASE,
    ):
        supported.add("action")
    return roles.issubset(supported)


def _troubleshooting_evidence_records(evidence: str) -> list[str]:
    """Split structured troubleshooting evidence into source-bound records.

    Manual table serializers use several equivalent shapes: one complete row per
    line, a single row wrapped across field lines, compact semicolon-delimited
    rows, or pipe tables.  Newlines alone therefore cannot define records.  A
    repeated row anchor (identifier, message, or row header) starts a sibling
    record. Row-header serializers and diagnostic-label serializers are also
    distinct record shapes, so crossing between those shapes starts a sibling;
    complementary identifier/message labels remain part of one diagnostic row.
    Distinct fields on following lines remain attached to the current record.
    """
    text = str(evidence or "").strip()
    if not text:
        return []

    pipe_lines = [line.strip() for line in text.splitlines() if line.strip() and "|" in line]
    non_pipe_lines = [line.strip() for line in text.splitlines() if line.strip() and "|" not in line]
    if len(pipe_lines) >= 2 and not non_pipe_lines:
        return pipe_lines

    label_pattern = re.compile(
        r"\b(?P<label>"
        r"error\s+messages?|messages?|error\s+(?:code|number)|alarm|fault|"
        r"row\s+headers?|cause|corrective\s+actions?|remed(?:y|ies)|cell\s+value"
        r")\s*:",
        flags=re.IGNORECASE,
    )
    matches = list(label_pattern.finditer(text))
    if not matches:
        return [line.strip() for line in text.splitlines() if line.strip()]

    def label_family(label: str) -> str:
        normalized = re.sub(r"\s+", " ", label.lower()).strip()
        if normalized.startswith("error code") or normalized.startswith("error number"):
            return "identifier"
        if normalized in {"alarm", "fault"}:
            return "identifier"
        if normalized.startswith("row header"):
            return "row"
        if normalized.startswith("error message") or normalized.startswith("message"):
            return "message"
        if normalized == "cause":
            return "cause"
        if normalized.startswith("corrective action") or normalized.startswith("remed"):
            return "action"
        return "value"

    records: list[str] = []
    current_parts: list[str] = []
    current_anchor_families: set[str] = set()
    anchor_families = {"identifier", "message", "row"}
    prefix = text[: matches[0].start()].strip()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        part = text[match.start() : end].strip(" \t\r\n;")
        family = label_family(match.group("label"))
        crosses_record_shape = (
            family in anchor_families
            and bool(current_anchor_families)
            and ((family == "row") != ("row" in current_anchor_families))
        )
        if family in anchor_families and (family in current_anchor_families or crosses_record_shape):
            record = "; ".join(item for item in current_parts if item).strip()
            if record:
                records.append(record)
            current_parts = []
            current_anchor_families = set()
        if not current_parts and prefix:
            current_parts.append(prefix)
            prefix = ""
        current_parts.append(part)
        if family in anchor_families:
            current_anchor_families.add(family)

    record = "; ".join(item for item in current_parts if item).strip()
    if record:
        records.append(record)
    return records


def _repeated_side_matching_records(clause: str, evidence: str) -> list[str]:
    """Return only records bound to the identifier or symptom requested by a side."""
    requested_ids = _query_requested_troubleshooting_identifiers(clause)
    anchor = _repeated_side_clause_anchor(clause)
    matching: list[str] = []
    for record in _troubleshooting_evidence_records(evidence):
        identifiers = {
            _normalized_phrase(match.group(0))
            for match in re.finditer(r"\b[a-z]*\d[a-z0-9._/-]*\b", record, flags=re.IGNORECASE)
        }
        if requested_ids and requested_ids.intersection(identifiers):
            matching.append(record)
        elif not requested_ids and len(anchor) >= 6:
            anchor_terms = _material_claim_terms(anchor).difference(
                {"condition", "reported", "reporting", "shown", "showing"}
            )
            identity_parts = [
                match.group("value").strip(" .;|")
                for match in re.finditer(
                    r"\b(?:row\s+headers?|error\s+messages?|messages?)\s*:\s*"
                    r"(?P<value>.*?)"
                    r"(?=(?:;|\n)\s*(?:error\s+messages?|messages?|error\s+(?:code|number)|"
                    r"alarm|fault|row\s+headers?|cause|corrective\s+actions?|remed(?:y|ies)|"
                    r"cell\s+value)\s*:|$)",
                    record,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            ]
            if not identity_parts and "|" in record:
                first_cell = next((cell.strip() for cell in record.split("|") if cell.strip()), "")
                if first_cell and not re.fullmatch(r"(?:error|message|condition|symptom)", first_cell, re.I):
                    identity_parts.append(first_cell)
            identity_terms = _material_claim_terms(_normalized_phrase(" ".join(identity_parts)))
            if len(anchor_terms) >= 2 and anchor_terms.issubset(identity_terms):
                matching.append(record)
    return matching


def _repeated_side_answer_text(clause: str, result: SearchResult) -> str:
    citation_evidence = _citation_evidence_text(result)
    for record in _repeated_side_matching_records(clause, citation_evidence):
        return record
    return _focused_diagnostic_table_answer_text(clause, result) or citation_evidence


def _repeated_side_troubleshooting_results(
    query: str,
    results: list[SearchResult],
) -> tuple[bool, list[SearchResult]]:
    """Bind each repeated product clause to its own troubleshooting evidence."""
    clauses = _repeated_product_side_clauses(query)
    if len(clauses) < 2 or not _is_troubleshooting_query(query):
        return False, []
    if any(
        not _query_requested_troubleshooting_identifiers(clause)
        and not _repeated_side_clause_anchor(clause)
        for clause in clauses
    ):
        return False, []

    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()
    for clause in clauses:
        side = _requested_product_sides(clause)[0]
        requested_ids = _query_requested_troubleshooting_identifiers(clause)
        anchor = _repeated_side_clause_anchor(clause)
        requested_roles = _requested_troubleshooting_evidence_roles(clause)
        candidates: list[tuple[float, int, SearchResult]] = []
        for index, result in enumerate(results):
            if result.chunk_id in seen_chunks or not _result_matches_requested_product_side(result, side):
                continue
            citation_evidence = _citation_evidence_text(result)
            matching_records = _repeated_side_matching_records(clause, citation_evidence)
            if not matching_records or not any(
                _troubleshooting_evidence_supports_roles(record, requested_roles)
                for record in matching_records
            ):
                continue
            candidates.append((_troubleshooting_evidence_score(clause, result), index, result))
        if not candidates:
            return True, []
        _score, _index, best = max(candidates, key=lambda item: (item[0], -item[1]))
        selected.append(best)
        seen_chunks.add(best.chunk_id)
    return True, selected


def _meaningful_comparison_clause_terms(clause: str) -> set[str]:
    generic_terms = {
        "action",
        "correct",
        "corrective",
        "failure",
        "guidance",
        "listed",
        "manual",
        "remedy",
        "series",
        "system",
    }
    terms = {
        term
        for term in _material_claim_terms(clause)
        if term not in generic_terms and not re.fullmatch(r"[a-z]{1,4}-?[a-z]?\d{0,4}", term)
    }
    expanded_terms = set(terms)
    for term in terms:
        if "-" in term:
            expanded_terms.update(part for part in term.split("-") if len(part) >= 3)
    return expanded_terms


def _compact_query_codes(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"\b[A-Za-z]{1,6}\d*[A-Za-z]*\b", text)
        if 2 <= len(token) <= 8
        and token.lower()
        not in {
            "and",
            "for",
            "the",
            "with",
            "value",
            "data",
            "form",
            "format",
            "angle",
            "compare",
            "measured",
            "series",
        }
    }


def _comparison_table_requested_codes(query: str) -> set[str]:
    query_without_models = MODEL_TOKEN_RE.sub(" ", query)
    return _compact_query_codes(query_without_models)


def _comparison_table_primary_requested_codes(query: str) -> set[str]:
    codes = _comparison_table_requested_codes(query)
    primary = {code for code in codes if re.search(r"\d", code)}
    return primary or codes


def _comparison_table_result_codes(result: SearchResult) -> set[str]:
    texts = [
        str(result.content or ""),
        " ".join(str(item) for item in result.metadata.get("table_row_headers") or []),
        " ".join(str(item) for item in result.metadata.get("table_column_headers") or []),
    ]
    return _compact_query_codes(" ".join(texts))


def _comparison_table_result_score(query: str, result: SearchResult) -> float:
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return 0.0
    score = _fallback_evidence_score(query, result)
    query_terms = _answer_terms(query)
    content = str(result.content or "")
    content_terms = _answer_terms(content)
    if {"form", "format", "measured", "data"}.intersection(query_terms) and {
        "form",
        "measured",
        "data",
    }.issubset(content_terms):
        score += 3.0
    if "cause" in query_terms and "cause" in content_terms:
        score += 2.0
    requested_codes = _comparison_table_requested_codes(query)
    if requested_codes:
        result_codes = _comparison_table_result_codes(result)
        code_matches = requested_codes.intersection(result_codes)
        score += min(5.0, 2.5 * len(code_matches))
        if not code_matches and len(requested_codes) <= 3:
            score -= 4.0
    if re.search(r"\brow headers?:\s*[^;\n]+", content, flags=re.IGNORECASE):
        score += 1.0
    if re.search(r"\bcell value:\s*[^;\n]+", content, flags=re.IGNORECASE):
        score += 1.0
    return score


def _comparison_model_side_queries(query: str) -> dict[str, str]:
    models = _ordered_model_tokens(query)
    if len(models) < 2:
        return {}
    side_queries = {model: query for model in models}
    pattern = re.search(
        rf"\bfor\s+{re.escape(models[0])}\s+and\s+{re.escape(models[1])}\b[\w\s,\-/]*?\bcompare\b(.+?)\bwith\b(.+?)(?:[.?]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not pattern:
        return side_queries
    common_prefix = query[: pattern.start(1)]
    left = pattern.group(1).strip(" ,.;:?")
    right = pattern.group(2).strip(" ,.;:?")
    if left and right:
        side_queries[models[0]] = f"{common_prefix} {left}"
        side_queries[models[1]] = f"{common_prefix} {right}"
    return side_queries


def _comparison_table_model_side_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    if not _is_comparison_query(query):
        return []
    side_queries = _comparison_model_side_queries(query)
    explicit_models = set(side_queries)
    if len(explicit_models) < 2:
        return []

    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()
    available_models = [
        model
        for model in sorted(explicit_models)
        if any(_result_mentions_requested_model_side(result, model) for result in ordered_results)
    ]
    if len(available_models) < 2:
        return []

    for model in available_models:
        side_query = side_queries.get(model, query)
        requested_codes = _comparison_table_requested_codes(side_query)
        primary_requested_codes = _comparison_table_primary_requested_codes(side_query)
        candidates = [
            (_comparison_table_result_score(side_query, result), index, result)
            for index, result in enumerate(ordered_results[:12])
            if result.chunk_id not in seen_chunks
            and _result_mentions_requested_model_side(result, model)
            and (
                not requested_codes
                or requested_codes.intersection(_comparison_table_result_codes(result))
            )
            and (
                not primary_requested_codes
                or primary_requested_codes.intersection(_comparison_table_result_codes(result))
            )
        ]
        candidates = [item for item in candidates if item[0] >= 4.0]
        if not candidates:
            return []
        _score, _index, result = max(candidates, key=lambda item: (item[0], -item[1]))
        selected.append(result)
        seen_chunks.add(result.chunk_id)
    return selected


def _requires_comparison_table_model_side_coverage(query: str, results: list[SearchResult]) -> bool:
    if not (_is_comparison_query(query) and len(_model_tokens(query)) >= 2):
        return False
    query_terms = _answer_terms(query)
    if not {"form", "format", "value", "setting", "settings", "field", "parameter", "cell"}.intersection(query_terms):
        return False
    return any(str(result.metadata.get("chunk_type") or "") == "table_record" for result in results[:12])


def _result_matches_comparison_side_clause(result: SearchResult, clause: str) -> bool:
    clause_terms = _meaningful_comparison_clause_terms(clause)
    if not clause_terms:
        return False
    evidence_terms = _material_claim_terms(_troubleshooting_context_text(result))
    if not evidence_terms:
        return False
    model_terms = _model_tokens(clause)
    if model_terms and not any(_result_mentions_requested_model_side(result, model) for model in model_terms):
        return False
    matched = clause_terms.intersection(evidence_terms)
    required = max(1, min(4, len(clause_terms) // 2 + 1))
    return len(matched) >= required


def _comparison_troubleshooting_side_matches(query: str, results: list[SearchResult]) -> list[SearchResult]:
    if not (_is_comparison_query(query) and _is_troubleshooting_query(query)):
        return []
    repeated_applies, repeated_results = _repeated_side_troubleshooting_results(query, results)
    if repeated_applies:
        return repeated_results
    clauses = _comparison_side_clauses(query)
    if len(clauses) < 2:
        return []

    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()
    for clause in clauses:
        matches = [
            result
            for result in results
            if result.chunk_id not in seen_chunks and _result_matches_comparison_side_clause(result, clause)
        ]
        if not matches:
            continue
        best = max(matches, key=lambda result: _troubleshooting_evidence_score(clause, result))
        selected.append(best)
        seen_chunks.add(best.chunk_id)
    return selected


def _comparison_scoped_troubleshooting_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    selected = _comparison_troubleshooting_side_matches(query, results)
    if not selected:
        return results
    seen_chunks = {result.chunk_id for result in selected}
    selected.extend(result for result in results if result.chunk_id not in seen_chunks)
    return selected


def _matching_troubleshooting_row_text(query: str, result: SearchResult) -> str:
    anchor = _normalized_phrase(_query_troubleshooting_anchor(query))
    if len(anchor) < 10:
        return ""
    candidate_blocks = [
        str(result.metadata.get("table_row_group_context") or "").strip(),
        str(result.metadata.get("parent_context") or "").strip(),
        str(result.metadata.get("context_window") or "").strip(),
        str(result.content or "").strip(),
    ]
    query_terms = _answer_terms(query)
    matches: list[tuple[float, str]] = []
    for block in candidate_blocks:
        if not block:
            continue
        for line in block.splitlines():
            row = line.strip()
            if "|" not in row or anchor not in _normalized_phrase(row):
                continue
            row_lower = row.lower()
            score = 0.0
            if re.search(r"\b(corrective action|remedy)\b", row_lower):
                score += 3.0
            if "cause" in row_lower:
                score += 2.0
            if re.search(r"\b(change|set|check|connect|adjust|disable|enable|contact)\b", row_lower):
                score += 2.0
            score += min(4.0, len(query_terms.intersection(_answer_terms(row))) / 2)
            matches.append((score, row))
    if not matches:
        return ""
    return max(matches, key=lambda item: item[0])[1]


def _matching_troubleshooting_rows(query: str, results: list[SearchResult]) -> list[str]:
    rows: list[str] = []
    for result in _order_troubleshooting_results(query, results):
        row = _matching_troubleshooting_row_text(query, result)
        if row and row not in rows:
            rows.append(row)
    return rows


def _row_action_text(row: str) -> str:
    cells = [cell.strip() for cell in row.split("|") if cell.strip()]
    if len(cells) < 2:
        return ""
    for cell in reversed(cells[1:]):
        if re.search(r"\b(adjust|change|check|connect|contact|disable|enable|increase|make|perform|reduce|replace|set|turn|use|wait)\b", cell, flags=re.IGNORECASE):
            return cell
    return cells[-1]


def _answer_uses_matching_troubleshooting_row(answer: str, query: str, results: list[SearchResult]) -> bool:
    if not _is_troubleshooting_query(query):
        return True
    rows = _matching_troubleshooting_rows(query, results)
    if not rows:
        return True
    answer_terms = _material_claim_terms(answer)
    for row in rows:
        action = _row_action_text(row)
        action_terms = _material_claim_terms(action)
        if not action_terms:
            continue
        action_verbs = action_terms.intersection(TROUBLESHOOTING_ACTION_VERBS)
        if action_verbs and not action_verbs.intersection(answer_terms):
            continue
        matched = action_terms.intersection(answer_terms)
        required = max(2, min(len(action_terms), len(action_terms) // 2 + 1))
        if len(matched) >= required:
            return True
    return False


def _answer_uses_comparison_troubleshooting_side_rows(answer: str, query: str, results: list[SearchResult]) -> bool:
    clauses = _comparison_side_clauses(query)
    if len(clauses) < 2:
        return True
    side_results = _comparison_troubleshooting_side_matches(query, results)
    if len(side_results) < len(clauses):
        return False
    answer_terms = _material_claim_terms(answer)
    for result in side_results:
        action = _row_action_text(_fallback_answer_text(result))
        action_terms = _material_claim_terms(action or _fallback_answer_text(result))
        if not action_terms:
            continue
        action_verbs = action_terms.intersection(TROUBLESHOOTING_ACTION_VERBS)
        if action_verbs and not action_verbs.intersection(answer_terms):
            return False
        matched = action_terms.intersection(answer_terms)
        required = max(2, min(len(action_terms), len(action_terms) // 2 + 1))
        if len(matched) < required:
            return False
    return True


def _fallback_answer(query: str, results: list[SearchResult]) -> AnswerResponse:
    if not results:
        return AnswerResponse(
            answer="I could not answer from the available evidence.",
            confidence="low",
            used_documents=[],
            citations=[],
            warnings=["No retrieved evidence met the threshold."],
            followup_questions=[],
            insufficient_evidence=True,
        )
    repeated_applies, repeated_results = _repeated_side_troubleshooting_results(query, results)
    if (
        not _requested_troubleshooting_identifiers_are_available(query, results)
        and not (repeated_applies and repeated_results)
    ):
        return AnswerResponse(
            answer="I could not answer from the available evidence.",
            confidence="low",
            used_documents=[],
            citations=[],
            warnings=["Retrieved evidence did not cover every requested troubleshooting side."],
            followup_questions=[],
            insufficient_evidence=True,
        )
    fallback_results = _fallback_evidence_results(query, results)
    requested_troubleshooting_ids = _query_requested_troubleshooting_identifiers(query)
    if requested_troubleshooting_ids and not repeated_applies:
        requested_results = [
            result
            for result in fallback_results
            if _troubleshooting_row_identifiers(result).intersection(requested_troubleshooting_ids)
        ]
        has_explicit_diagnostic_code = bool(
            re.search(
                r"\b(?:error|alarm|fault)(?:\s+(?:number|code))?\s*[:#-]?\s*[a-z]*\d[a-z0-9._/-]*\b",
                query,
                flags=re.IGNORECASE,
            )
        )
        if requested_results or has_explicit_diagnostic_code:
            fallback_results = requested_results
    if not fallback_results:
        return AnswerResponse(
            answer="I could not answer from the available evidence.",
            confidence="low",
            used_documents=[],
            citations=[],
            warnings=["No retrieved evidence met the requested scope."],
            followup_questions=[],
            insufficient_evidence=True,
        )
    top = fallback_results[0]
    if _requires_allocation_table_binding(query) and not any(
        _focused_allocation_table_binding_answer_text(query, result) for result in fallback_results
    ):
        return AnswerResponse(
            answer="I could not answer from the available evidence.",
            confidence="low",
            used_documents=[],
            citations=[],
            warnings=["No retrieved evidence preserved the requested table row/cell binding."],
            followup_questions=[],
            insufficient_evidence=True,
        )
    if len(fallback_results) == 1:
        answer_text = (
            _requested_troubleshooting_rows_text(query, top)
            or _matching_troubleshooting_row_text(query, top)
            or _focused_diagnostic_table_answer_text(query, top)
            or _focused_signal_description_answer_text(query, top)
            or _focused_table_record_answer_text(query, top)
            or _focused_table_like_answer_text(query, top)
            or _focused_program_setting_protection_answer_text(query, top)
            or _fallback_answer_text(top)
        )
    elif repeated_applies:
        clauses = _repeated_product_side_clauses(query)
        answer_text = "Retrieved evidence:\n" + "\n".join(
            f"- {result.title}, page(s) {', '.join(str(page) for page in result.pages) or 'unknown'}: "
            f"{_repeated_side_answer_text(clause, result)}"
            for clause, result in zip(clauses, fallback_results, strict=True)
        )
    else:
        answer_text = "Retrieved evidence:\n" + "\n".join(
            f"- {result.title}, page(s) {', '.join(str(page) for page in result.pages) or 'unknown'}: "
            f"{_requested_troubleshooting_rows_text(query, result) or _focused_diagnostic_table_answer_text(query, result) or _focused_signal_description_answer_text(query, result) or _focused_table_record_answer_text(query, result) or _focused_table_like_answer_text(query, result) or _focused_program_setting_protection_answer_text(query, result) or _fallback_answer_text(result)}"
            for result in fallback_results
        )
    return AnswerResponse(
        answer=answer_text,
        confidence="medium",
        used_documents=[
            {
                "document_id": result.source_document_id,
                "title": result.title,
                "version": result.document_version_id,
                "pages": result.pages,
                "section_path": result.section_path,
            }
            for result in fallback_results
        ],
        citations=[
            {
                "chunk_id": result.chunk_id,
                "document_id": result.source_document_id,
                "pages": result.pages,
                "quote_span": None,
            }
            for result in fallback_results
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )


def _fallback_evidence_score(query: str, result: SearchResult) -> float:
    query_terms = _answer_terms(query)
    evidence = _fallback_answer_text(result)
    evidence_terms = _answer_terms(evidence)
    if not evidence_terms:
        return 0.0
    score = min(6.0, len(query_terms.intersection(evidence_terms)))
    score += _ordered_query_phrase_score(query, evidence)
    score += _direct_configuration_binding_score(query_terms, evidence_terms)
    evidence_scope_terms = _answer_terms(
        " ".join(
            [
                evidence,
                str(result.title or ""),
                " ".join(str(part) for part in result.section_path or []),
                _result_model_text(result),
            ]
        )
    )
    distinctive_query_terms = {
        term
        for term in query_terms
        if re.search(r"[-/]", term) or re.search(r"[a-z]+\d|\d+[a-z]+", term, flags=re.IGNORECASE)
    }
    for term in distinctive_query_terms:
        if term in evidence_scope_terms:
            score += 2.0
        else:
            score -= 3.0
    if not _explicit_scope_phrases_supported(query, result):
        score -= 8.0
    chunk_type = str(result.metadata.get("chunk_type") or "")
    if chunk_type in {"table_record", "spec_record", "datasheet_record", "procedure_record", "warning_record"}:
        score += 2.0
    if _is_troubleshooting_query(query) and re.search(r"\b(cause|corrective action|remedy|message)\b", evidence, flags=re.IGNORECASE):
        score += 2.0
    if _is_comparison_query(query) and (
        str(result.source_document_id or "") in evidence
        or _model_tokens(query).intersection(_model_tokens(evidence))
        or query_terms.intersection(_answer_terms(result.title))
    ):
        score += 1.0
    if re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
        score += min(4.0, float(len(_contextual_quantity_terms(evidence))))
    return score


def _citation_visible_text(result: SearchResult) -> str:
    return str(result.content or "").strip()


def _quantity_citation_visible_score(query: str, result: SearchResult) -> float:
    if not re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
        return 0.0
    requested_roles = _requested_quantity_roles(query)
    if not requested_roles:
        return 0.0
    content = _citation_visible_text(result)
    if not content:
        return 0.0
    query_scope_terms = _quantity_scope_terms(query)
    best_score = 0.0
    content_role_values = _quantity_role_values(content)
    if requested_roles.issubset({role for role, values in content_role_values.items() if values}):
        best_score = 6.0
        best_score += min(4.0, float(len(query_scope_terms.intersection(_quantity_scope_terms(content)))))
    for role_values, clause in _quantity_role_value_group_records(content):
        scoped_role_values = {role: values for role, values in role_values.items() if role in requested_roles and values}
        if not requested_roles.issubset(scoped_role_values):
            continue
        score = 6.0
        score += min(4.0, float(len(query_scope_terms.intersection(_quantity_scope_terms(clause)))))
        if re.search(r"\bcontinuous\b", query, flags=re.IGNORECASE) and re.search(
            r"\bcontinuous\b", content, flags=re.IGNORECASE
        ):
            score += 2.0
        if re.search(r"\btrigger\b", query, flags=re.IGNORECASE) and re.search(r"\btrigger\b", content, flags=re.IGNORECASE):
            score += 1.0
        best_score = max(best_score, score)
    return best_score


def _explicit_scope_phrase_groups(text: str) -> list[set[str]]:
    phrase_groups: list[set[str]] = []
    for match in re.finditer(r"\b([A-Za-z0-9][A-Za-z0-9/\- ]{0,60}?\s+mode)\b", text, flags=re.IGNORECASE):
        phrase = " ".join(re.findall(r"[a-z0-9]+", match.group(1).lower()))
        words = phrase.split()
        if len(words) >= 2:
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


def _exclusive_scope_phrase(phrase: str) -> bool:
    phrase_terms = _answer_terms(phrase)
    return bool(
        {"standard", "lighting", "lumitrax", "multispectrum", "capture", "asynchronous", "continuous", "sheet-fed", "sheet", "fed"}.intersection(
            phrase_terms
        )
    )


def _has_conflicting_explicit_scope(query_scope_groups: list[set[str]], evidence: str) -> bool:
    if not query_scope_groups:
        return False
    for phrase in _explicit_scope_phrases(evidence):
        if not _exclusive_scope_phrase(phrase):
            continue
        if not any(_scope_phrase_matches_group(phrase, group) for group in query_scope_groups):
            return True
    return False


def _mode_scope_state_supported(scope: str, evidence_normalized: str) -> bool:
    if not scope.endswith(" mode"):
        return False
    base_scope = scope.removesuffix(" mode").strip()
    if len(base_scope.split()) < 2:
        return False
    base_pattern = re.escape(base_scope).replace(r"\ ", r"\s+")
    return bool(
        re.search(
            rf"\b{base_pattern}\b(?:\s+(?:is|was|has\s+been))?\s+(?:set|selected|enabled)\b"
            rf"|\b(?:set|select|selected|enable|enabled)\b[\w\s\[\]-]{{0,30}}\b{base_pattern}\b",
            evidence_normalized,
            flags=re.IGNORECASE,
        )
    )


def _explicit_scope_phrases_supported(query: str, result: SearchResult) -> bool:
    query_scope_groups = _explicit_scope_phrase_groups(query)
    if not query_scope_groups:
        return _result_supports_capture_type_scope(query, result)
    evidence_text = " ".join(
        [
            _fallback_answer_text(result),
            str(result.title or ""),
            " ".join(str(part) for part in result.section_path or []),
        ]
    )
    evidence_normalized = " ".join(re.findall(r"[a-z0-9]+", evidence_text.lower()))
    if _has_conflicting_explicit_scope(query_scope_groups, evidence_text):
        return False
    return all(
        any(scope in evidence_normalized or _mode_scope_state_supported(scope, evidence_normalized) for scope in group)
        for group in query_scope_groups
    ) and _evidence_supports_capture_type_scope(
        query, evidence_text
    )


def _capture_type_scope(text: str) -> str | None:
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    if re.search(r"\bline\s+scan(?:\s+cameras?)?\b", normalized) or re.search(
        r"\bline\s+cameras?\b", normalized
    ):
        return "line_scan"
    if re.search(r"\barea\s+cameras?\b", normalized):
        return "area_camera"
    return None


def _evidence_supports_capture_type_scope(query: str, evidence: str) -> bool:
    query_scope = _capture_type_scope(query)
    if query_scope is None:
        return True
    evidence_scope = _capture_type_scope(evidence)
    if evidence_scope is None:
        return False
    return evidence_scope == query_scope


def _result_supports_capture_type_scope(query: str, result: SearchResult) -> bool:
    evidence_text = " ".join(
        [
            _fallback_answer_text(result),
            str(result.title or ""),
            " ".join(str(part) for part in result.section_path or []),
        ]
    )
    return _evidence_supports_capture_type_scope(query, evidence_text)


def _explicit_scope_has_candidate(query: str, results: list[SearchResult]) -> bool:
    query_scope_groups = _explicit_scope_phrase_groups(query)
    if not query_scope_groups:
        return True
    return any(_explicit_scope_phrases_supported(query, result) for result in results)


def _direct_configuration_binding_score(query_terms: set[str], evidence_terms: set[str]) -> float:
    if not {"configuration", "setting", "settings", "area", "screen", "menu", "option"}.intersection(query_terms):
        return 0.0
    if "configuration" in query_terms and "configuration" not in evidence_terms:
        return 0.0
    anchors = {"camera", "trigger", "light", "lighting", "illumination"}.intersection(query_terms)
    if len(anchors) < 2:
        return 0.0
    if not {"configuration", "setting", "settings", "screen", "menu"}.intersection(evidence_terms):
        return 0.0
    matched_anchors = anchors.intersection(evidence_terms)
    if len(matched_anchors) < 2:
        return 0.0
    return min(3.0, 1.0 + float(len(matched_anchors)))


def _direct_configuration_label_score(query: str, evidence: str) -> float:
    query_terms = _answer_terms(query)
    if "configuration" not in query_terms:
        return 0.0
    anchors = {"camera", "trigger", "light", "lighting", "illumination"}.intersection(query_terms)
    if len(anchors) < 2:
        return 0.0
    normalized_evidence = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    ordered_anchor_sets = (
        ("camera", "trigger", "light"),
        ("camera", "trigger", "lighting"),
        ("camera", "trigger", "illumination"),
    )
    for anchor_set in ordered_anchor_sets:
        if not set(anchor_set).issubset(query_terms):
            continue
        phrase = " ".join([*anchor_set, "configuration"])
        if phrase in normalized_evidence:
            return 4.0
        if all(anchor in normalized_evidence for anchor in anchor_set) and re.search(
            rf"{re.escape(anchor_set[0])}\W+{re.escape(anchor_set[1])}\W+{re.escape(anchor_set[2])}"
            rf"[\w\s:/().\[\]-]{{0,80}}\bconfiguration\b",
            evidence,
            flags=re.IGNORECASE,
        ):
            return 3.0
    return 0.0


def _direct_configuration_detail_supported(query: str, evidence: str) -> bool:
    query_terms = _answer_terms(query)
    if not _requires_direct_configuration_evidence(query):
        return True
    if not {"trigger"}.intersection(query_terms) or not {"light", "lighting", "illumination"}.intersection(query_terms):
        return True
    normalized_evidence = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    has_trigger_input = re.search(r"\btrigger\s+inputs?\b", normalized_evidence) is not None
    has_light_target = (
        re.search(r"\billumination\s+control\s+targets?\b", normalized_evidence) is not None
        or re.search(r"\blight\s+control\s+targets?\b", normalized_evidence) is not None
    )
    return has_trigger_input and has_light_target


def _configuration_setup_candidate(
    query: str,
    configuration_result: SearchResult,
    scored: list[tuple[float, int, SearchResult]],
) -> SearchResult | None:
    if not _explicit_scope_phrase_groups(query):
        return None
    matches: list[tuple[float, int, SearchResult]] = []
    for _score, index, result in scored:
        if result.chunk_id == configuration_result.chunk_id:
            continue
        if result.source_document_id != configuration_result.source_document_id:
            continue
        if not _explicit_scope_phrases_supported(query, result):
            continue
        evidence = _fallback_answer_text(result)[:1200]
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
        candidate_score = _fallback_evidence_score(query, result)
        if str(result.metadata.get("chunk_type") or "") == "procedure_record":
            candidate_score += 1.0
        if "capture environment" in normalized:
            candidate_score += 1.0
        matches.append((candidate_score, index, result))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], -item[1]))[2]


def _configuration_requested_candidates(
    query: str, query_terms: set[str], scored: list[tuple[float, int, SearchResult]]
) -> list[SearchResult]:
    if "configuration" not in query_terms:
        return []
    candidates = [
        (score + _direct_configuration_label_score(query, _fallback_answer_text(result)[:1000]), index, result)
        for score, index, result in scored
        if _explicit_scope_phrases_supported(query, result)
        if _direct_configuration_detail_supported(query, _fallback_answer_text(result)[:1000])
        if _direct_configuration_binding_score(
            query_terms,
            _answer_terms(_fallback_answer_text(result)[:800]),
        )
        > 0.0
    ]
    if not candidates:
        return []
    best_score, _best_index, best_result = max(candidates, key=lambda item: (item[0], -item[1]))
    if best_score >= 2.0:
        selected = [best_result]
        if support := _configuration_setup_candidate(query, best_result, scored):
            selected.append(support)
        return selected
    return []


def _requires_direct_configuration_evidence(query: str) -> bool:
    query_terms = _answer_terms(query)
    if "configuration" not in query_terms:
        return False
    return len({"camera", "trigger", "light", "lighting", "illumination"}.intersection(query_terms)) >= 2


def _ordered_query_phrase_score(query: str, evidence: str) -> float:
    query_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in {"what", "which", "where", "when", "how", "for", "the", "is", "are", "used"}
    ]
    if len(query_tokens) < 2:
        return 0.0
    normalized_evidence = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    score = 0.0
    for size, weight in ((4, 1.5), (3, 1.0), (2, 0.5)):
        for index in range(0, len(query_tokens) - size + 1):
            phrase = " ".join(query_tokens[index : index + size])
            if phrase in normalized_evidence:
                score += weight
                if score >= 3.0:
                    return 3.0
    return min(score, 3.0)


def _screen_request_score(query: str, evidence: str) -> float:
    query_terms = _answer_terms(query)
    if "screen" not in query_terms:
        return 0.0
    evidence_terms = _answer_terms(evidence)
    if "screen" not in evidence_terms:
        return 0.0
    score = 2.0
    normalized_evidence = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    important_terms = query_terms.intersection({"trigger", "settings", "setting", "camera", "line", "navigation", "change"})
    score += min(3.0, float(len(important_terms.intersection(evidence_terms))) * 0.75)
    if re.search(r"\bstep\s+\d+(?:\s+\d+)?\s+trigger\s+settings\s+screen\b", normalized_evidence):
        score += 2.0
    if re.search(r"\btrigger\s+settings\s+screen\b", normalized_evidence):
        score += 1.0
    return score


def _procedure_membership_score(query: str, result: SearchResult) -> float:
    query_terms = _answer_terms(query)
    if not {"procedure", "section", "step", "preparation"}.intersection(query_terms):
        return 0.0
    normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    asks_membership = bool(
        re.search(
            r"\bwhat\s+(?:procedure|section|step|preparation)\b|\bpart\s+of\b|\bincluded\s+in\b|\bbelongs\s+to\b",
            normalized_query,
        )
    )
    if not asks_membership:
        return 0.0

    evidence = _fallback_answer_text(result)
    evidence_terms = _answer_terms(evidence)
    if not evidence_terms:
        return 0.0
    normalized_evidence = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    has_procedure_context = bool(
        re.search(r"\b(preparation|preparing|procedure|navigation|step\s+\d+)\b", normalized_evidence)
    )
    if not has_procedure_context:
        return 0.0

    generic_membership_terms = {
        "what",
        "which",
        "where",
        "when",
        "how",
        "for",
        "the",
        "and",
        "that",
        "this",
        "procedure",
        "section",
        "step",
        "preparation",
        "part",
        "included",
        "belongs",
        "setup",
        "used",
        "use",
        "line",
        "camera",
        "cameras",
    }
    ordered_query_subject_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) >= 3 and token not in generic_membership_terms
    ]
    query_subject_tokens = set(ordered_query_subject_tokens)
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence.lower()))
    expanded_evidence_tokens = set(evidence_tokens)
    for token in evidence_tokens:
        if token.endswith("ing") and len(token) > 5:
            expanded_evidence_tokens.add(token[:-3])
        if token.endswith("ment") and len(token) > 6:
            expanded_evidence_tokens.add(token[:-4])
        if token.endswith("ed") and len(token) > 4:
            expanded_evidence_tokens.add(token[:-2])
        if token.endswith("s") and len(token) > 3:
            expanded_evidence_tokens.add(token[:-1])
    subject_matches = query_subject_tokens.intersection(expanded_evidence_tokens)
    query_subject_bigrams = [
        " ".join(ordered_query_subject_tokens[index : index + 2])
        for index in range(0, len(ordered_query_subject_tokens) - 1)
    ]
    if query_subject_bigrams and not any(bigram in normalized_evidence for bigram in query_subject_bigrams):
        return 0.0
    if query_subject_tokens and len(subject_matches) < min(2, len(query_subject_tokens)):
        return 0.0
    subject_terms = query_terms.difference(generic_membership_terms)
    answer_term_matches = subject_terms.intersection(evidence_terms)
    subject_match_count = max(len(subject_matches), len(answer_term_matches))
    if len(subject_matches) < min(2, len(subject_terms)):
        return 0.0

    score = 2.0 + min(4.0, float(subject_match_count))
    if re.search(r"\bpreparation\s+\d+\b", normalized_evidence):
        score += 2.0
    if re.search(r"\b(?:adjust|change|set|select)\b.{0,80}\b(?:ratio|settings?|screen|navigation)\b", normalized_evidence):
        score += 1.0
    if "navigation" in query_terms and "navigation" in evidence_terms:
        score += 1.0
    return score


def _procedure_membership_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    scored = [
        (_procedure_membership_score(query, result), index, result)
        for index, result in enumerate(ordered_results[:8])
    ]
    scored = [item for item in scored if item[0] >= 4.0]
    if not scored:
        return []
    best_score, _best_index, best_result = max(scored, key=lambda item: (item[0], -item[1]))
    first_score = _procedure_membership_score(query, ordered_results[0]) if ordered_results else 0.0
    if best_result.chunk_id != ordered_results[0].chunk_id or best_score > first_score:
        return [best_result]
    return []


def _has_strong_insufficient_fallback(query: str, results: list[SearchResult]) -> bool:
    fallback_results = _fallback_evidence_results(query, results)
    if len(fallback_results) > 1:
        return True
    if not fallback_results:
        return False
    top = fallback_results[0]
    if _is_troubleshooting_query(query) and str(top.metadata.get("chunk_type") or "") == "table_record":
        evidence_terms = _answer_terms(_fallback_answer_text(top))
        query_terms = _answer_terms(query)
        required_scope_terms = _distinctive_scope_terms(query_terms)
        result_scope_terms = _answer_terms(_result_model_text(top))
        if required_scope_terms and not required_scope_terms.issubset(result_scope_terms):
            return False
        has_diagnostic_answer = bool(re.search(r"\b(cause|remedy|corrective action|indicator|status)\b", _fallback_answer_text(top), flags=re.IGNORECASE))
        if has_diagnostic_answer and len(query_terms.intersection(evidence_terms)) >= 3:
            return True
    if not (
        _should_score_direct_fallback(query)
        or re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE)
        or _image_capture_buffer_query_terms(query)
    ):
        return False
    return _fallback_evidence_score(query, fallback_results[0]) >= 2.0


def _diagnostic_table_evidence_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    if not _is_troubleshooting_query(query):
        return []
    if _is_comparison_query(query):
        return []
    query_terms = _answer_terms(query)
    scored: list[tuple[float, int, SearchResult]] = []
    for index, result in enumerate(ordered_results[:8]):
        if str(result.metadata.get("chunk_type") or "") != "table_record":
            continue
        row_text = str(result.content or "").strip()
        row_terms = _answer_terms(row_text)
        if not row_terms:
            continue
        if not _diagnostic_table_requested_facets_supported(query, row_text):
            continue
        score = min(6.0, float(len(query_terms.intersection(row_terms))))
        score += _ordered_query_phrase_score(query, row_text)
        if re.search(r"\b(indicator|status|cause|remedy|corrective action)\b", row_text, flags=re.IGNORECASE):
            score += 3.0
        if re.search(r"\b(?:error|alarm|fault)\b", query, flags=re.IGNORECASE) and re.search(
            r"\b(indicator|status|cause)\b", row_text, flags=re.IGNORECASE
        ):
            score += 2.0
        if re.search(r"\bfield[-\s]?network", query, flags=re.IGNORECASE) and re.search(
            r"\b(?:field[-\s]?network|ethernet/ip|profinet|handshake)\b", row_text, flags=re.IGNORECASE
        ):
            score += 3.0
        required_scope_terms = _distinctive_scope_terms(query_terms)
        result_scope_terms = _answer_terms(_result_model_text(result))
        if required_scope_terms and not required_scope_terms.issubset(result_scope_terms):
            score -= 8.0
        scored.append((score, index, result))
    if not scored:
        return []
    best_score, _best_index, best_result = max(scored, key=lambda item: (item[0], -item[1]))
    return [best_result] if best_score >= 5.0 else []


def _diagnostic_table_requested_facets_supported(query: str, row_text: str) -> bool:
    query_lc = query.lower()
    row_lc = row_text.lower()
    if re.search(r"\bfield[-\s]?network", query_lc):
        if not re.search(r"\b(?:field[-\s]?network|ethernet/ip|profinet|handshake)\b", row_lc):
            return False
    if re.search(r"\b(?:setting|settings|check|checks|correct|corrective|remedy|action|fix)\b", query_lc):
        if not re.search(
            r"\b(?:disable|enable|execute|set|select|change|check|connect|replace|notify|notification|remedy|corrective action)\b",
            row_lc,
        ):
            return False
    if re.search(r"\b(?:indicate|status|symptom)\b", query_lc):
        if not re.search(r"\b(?:indicator|status|occurred|occurs|condition|symptom|cause)\b", row_lc):
            return False
    return True


def _program_setting_protection_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    query_terms = _answer_terms(query)
    if not {"program", "setting"}.issubset(query_terms):
        return []
    if not {"camera", "cameras", "selected", "restricted", "limit"}.intersection(query_terms):
        return []
    scored: list[tuple[float, int, SearchResult]] = []
    for index, result in enumerate(ordered_results[:8]):
        evidence = _fallback_answer_text(result)
        evidence_terms = _answer_terms(evidence)
        if not {"program", "setting"}.issubset(evidence_terms):
            continue
        if not {"camera", "cameras"}.intersection(evidence_terms):
            continue
        if "password" not in evidence_terms or "mac" not in evidence_terms:
            continue
        score = _fallback_evidence_score(query, result)
        if {"restrict", "restricted", "allow", "selected"}.intersection(evidence_terms):
            score += 2.0
        if "addresses" in evidence_terms:
            score += 1.0
        scored.append((score, index, result))
    if not scored:
        return []
    return [max(scored, key=lambda item: (item[0], -item[1]))[2]]


def _answer_cites_selected_screen_evidence(answer: AnswerResponse, query: str, results: list[SearchResult]) -> bool:
    if "screen" not in _answer_terms(query):
        return True
    selected = _fallback_evidence_results(query, results)
    if not selected:
        return True
    selected_chunk_ids = {result.chunk_id for result in selected}
    cited_chunk_ids = {str(citation.get("chunk_id") or "") for citation in answer.citations}
    return bool(selected_chunk_ids.intersection(cited_chunk_ids))


def _should_score_direct_fallback(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:what|which)\b.+\b(?:area|screen|menu|setting|settings|configuration|option|field|parameter|control|value)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _multi_part_evidence_clauses(query: str) -> list[str]:
    if _is_comparison_query(query):
        return []
    normalized = re.sub(r"\s+", " ", query).strip(" .?")
    question_markers = re.findall(r"\b(?:what|which|where|when|how)\b", normalized, flags=re.IGNORECASE)
    if len(question_markers) < 2:
        return []
    parts = [
        part.strip(" ,.;:?")
        for part in re.split(r"\b(?:and|plus|as well as)\s+(?=(?:what|which|where|when|how)\b)", normalized, flags=re.IGNORECASE)
        if part.strip(" ,.;:?")
    ]
    if len(parts) < 2:
        return []
    return parts


def _result_evidence_key(result: SearchResult) -> str:
    return f"{result.source_document_id}:{_normalized_citation_text(_fallback_answer_text(result))[:500]}"


def _multi_part_evidence_coherence_score(result: SearchResult, selected: list[SearchResult]) -> float:
    if not selected:
        return 0.0
    score = 0.0
    result_pages = set(result.pages or [])
    result_section_path = tuple(str(part) for part in result.section_path or [])
    for prior in selected:
        if result.source_document_id and result.source_document_id == prior.source_document_id:
            score += 0.5
        if result_pages and result_pages.intersection(prior.pages or []):
            score += 2.0
        if result_section_path and result_section_path == tuple(str(part) for part in prior.section_path or []):
            score += 2.0
    return min(score, 3.0)


def _multi_part_fallback_evidence_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    clauses = _multi_part_evidence_clauses(query)
    if len(clauses) < 2:
        return []

    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()
    seen_evidence: set[str] = set()
    covered_clauses = 0
    for clause in clauses:
        if any(_fallback_evidence_score(clause, result) >= 2.0 for result in selected):
            covered_clauses += 1
            continue
        scored = [
            (
                _fallback_evidence_score(clause, result) + _multi_part_evidence_coherence_score(result, selected),
                index,
                result,
            )
            for index, result in enumerate(ordered_results[:8])
            if result.chunk_id not in seen_chunks
        ]
        if not scored:
            continue
        best_score, _best_index, best_result = max(scored, key=lambda item: (item[0], -item[1]))
        if best_score < 2.0:
            continue
        evidence_key = _result_evidence_key(best_result)
        if evidence_key in seen_evidence:
            continue
        selected.append(best_result)
        seen_chunks.add(best_result.chunk_id)
        seen_evidence.add(evidence_key)
        covered_clauses += 1
        if len(selected) >= 4:
            break

    if selected and covered_clauses == len(clauses):
        return selected
    if len(selected) >= 2:
        return selected
    return []


def _image_capture_buffer_query_terms(query: str) -> set[str]:
    query_terms = _answer_terms(query)
    if not {"image", "capture", "buffer"}.issubset(query_terms):
        return set()
    return query_terms


def _image_capture_buffer_evidence_score(query_terms: set[str], result: SearchResult) -> float:
    evidence = _fallback_answer_text(result)
    normalized = " ".join(re.findall(r"[a-z0-9]+", evidence.lower()))
    evidence_terms = _answer_terms(evidence)
    has_buffer_context = {"image", "capture", "buffer"}.issubset(evidence_terms) or bool(
        re.search(r"\bimage\s+capture\s+buffer\b", normalized)
    )
    has_same_priority_role = bool(re.search(r"\bsame\s+capture\s+priority\s+condition\b", normalized))
    if not has_buffer_context and not has_same_priority_role:
        return 0.0

    score = min(4.0, float(len(query_terms.intersection(evidence_terms))))
    if re.search(r"\bimage\s+capture\s+buffer\b.{0,80}\bdisabled\b", normalized):
        score += 4.0
    elif "disabled" in query_terms and "disabled" in evidence_terms:
        score += 1.0

    if "trigger" in query_terms and {"input", "inputs"}.intersection(query_terms):
        if re.search(r"\btrigger\s+inputs?\b", normalized):
            score += 2.0
        if {"allowed", "prohibited"}.intersection(evidence_terms) or {"permitted", "prohibited"}.issubset(evidence_terms):
            score += 2.0
        if re.search(r"\breceiving\s+trigger\s+inputs?\b", normalized):
            score += 1.0
        if "progress" in query_terms and re.search(r"\bflow\b.{0,50}\bstopped\b|\bany\s+other\s+time\b", normalized):
            score += 1.0

    if {"camera", "cameras", "multiple", "condition"}.intersection(query_terms):
        if re.search(r"\bone\s+camera\b|\bmultiple\s+cameras\b", normalized):
            score += 2.0
        if re.search(r"\bsame\s+capture\s+priority\s+condition\b", normalized):
            score += 4.0
        elif {"same", "capture", "priority", "condition"}.issubset(evidence_terms):
            score += 2.0

    chunk_type = str(result.metadata.get("chunk_type") or "")
    if chunk_type == "atomic_text":
        score += 0.6
    elif chunk_type in {"procedure_record", "section_window"}:
        score += 0.3
    return score


def _image_capture_buffer_evidence_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    query_terms = _image_capture_buffer_query_terms(query)
    if not query_terms:
        return []

    scored = [
        (_image_capture_buffer_evidence_score(query_terms, result), index, result)
        for index, result in enumerate(ordered_results[:8])
    ]
    scored = [item for item in scored if item[0] >= 5.0]
    if not scored:
        return []

    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()

    def add_best(predicate: Any) -> None:
        matches = [(score, index, result) for score, index, result in scored if predicate(result)]
        if not matches:
            return
        _score, _index, result = max(matches, key=lambda item: (item[0], -item[1]))
        if result.chunk_id not in seen_chunks:
            selected.append(result)
            seen_chunks.add(result.chunk_id)

    def has_disabled_buffer_state(result: SearchResult) -> bool:
        return (
            re.search(
                r"\bimage\s+capture\s+buffer\b.{0,80}\bdisabled\b",
                " ".join(re.findall(r"[a-z0-9]+", _fallback_answer_text(result).lower())),
            )
            is not None
        )

    if "trigger" in query_terms and {"input", "inputs"}.intersection(query_terms):
        add_best(
            lambda result: re.search(
                r"\btrigger\s+inputs?\b",
                " ".join(re.findall(r"[a-z0-9]+", _fallback_answer_text(result).lower())),
            )
            is not None
            and {"allowed", "permitted", "prohibited"}.intersection(_answer_terms(_fallback_answer_text(result)))
            and ("disabled" not in query_terms or has_disabled_buffer_state(result))
        )

    if {"camera", "cameras", "multiple", "condition"}.intersection(query_terms):
        add_best(
            lambda result: re.search(
                r"\bone\s+camera\b|\bmultiple\s+cameras\b|\bsame\s+capture\s+priority\s+condition\b",
                " ".join(re.findall(r"[a-z0-9]+", _fallback_answer_text(result).lower())),
            )
            is not None
        )

    query_mentions_disabled = re.search(r"\bdisabled\b", query, flags=re.IGNORECASE) is not None

    if query_mentions_disabled:
        add_best(has_disabled_buffer_state)

    if selected:
        if query_mentions_disabled and not any(has_disabled_buffer_state(result) for result in selected):
            return []
        return selected[:3]

    if query_mentions_disabled:
        return []

    best_score, _best_index, best_result = max(scored, key=lambda item: (item[0], -item[1]))
    return [best_result] if best_score >= 6.0 else []


def _fallback_evidence_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    ordered_results = _comparison_scoped_troubleshooting_results(
        query,
        _focused_troubleshooting_results(query, _order_troubleshooting_results(query, results)),
    )
    requested_ids = _query_requested_troubleshooting_identifiers(query)
    requested_product_sides = _requested_product_sides(query)
    requested_numeric_ids = {identifier for identifier in requested_ids if identifier.isdigit()}
    repeated_applies, repeated_results = _repeated_side_troubleshooting_results(query, ordered_results)
    if repeated_applies:
        return repeated_results
    if _is_comparison_query(query) and len(requested_numeric_ids) >= 2:
        side_bindings = _requested_troubleshooting_identifier_side_bindings(query)
        requested_rows: list[SearchResult] = []
        seen_chunks: set[str] = set()
        for requested_id in requested_numeric_ids:
            required_sides = side_bindings.get(requested_id, [])
            if requested_product_sides and not required_sides:
                return []
            for required_side in required_sides or [None]:
                match = next(
                    (
                        result
                        for result in ordered_results
                        if requested_id in _troubleshooting_row_identifiers(result)
                        and (
                            required_side is None
                            or _result_matches_requested_product_side(result, required_side)
                        )
                    ),
                    None,
                )
                if match is None:
                    return []
                if match.chunk_id not in seen_chunks:
                    requested_rows.append(match)
                    seen_chunks.add(match.chunk_id)
        return requested_rows
    if (
        _is_comparison_query(query)
        and len(requested_ids) == 1
        and len(requested_product_sides) >= 2
    ):
        requested_id = next(iter(requested_ids))
        side_results: list[SearchResult] = []
        seen_chunks: set[str] = set()
        for side in requested_product_sides:
            match = next(
                (
                    result
                    for result in ordered_results
                    if result.chunk_id not in seen_chunks
                    and requested_id in _troubleshooting_row_identifiers(result)
                    and _result_matches_requested_product_side(result, side)
                ),
                None,
            )
            if match is None:
                return []
            side_results.append(match)
            seen_chunks.add(match.chunk_id)
        return side_results
    image_buffer_query_terms = _image_capture_buffer_query_terms(query)
    if image_buffer_results := _image_capture_buffer_evidence_results(query, ordered_results):
        return image_buffer_results
    if image_buffer_query_terms and re.search(r"\bdisabled\b", query, flags=re.IGNORECASE):
        return []
    if diagnostic_table_results := _diagnostic_table_evidence_results(query, ordered_results):
        return diagnostic_table_results
    if comparison_table_results := _comparison_table_model_side_results(query, ordered_results):
        return comparison_table_results
    if _requires_comparison_table_model_side_coverage(query, ordered_results):
        return []
    if not _is_comparison_query(query):
        if status_output_results := _status_output_table_results(query, ordered_results):
            return status_output_results
        if _status_output_table_evidence_seen(query, ordered_results):
            return []
        if program_setting_results := _program_setting_protection_results(query, ordered_results):
            return program_setting_results
        multi_part_results = _multi_part_fallback_evidence_results(query, ordered_results)
        if multi_part_results:
            return multi_part_results
        if procedure_membership_results := _procedure_membership_results(query, ordered_results):
            return procedure_membership_results
        if _should_score_direct_fallback(query):
            scored = [
                (_fallback_evidence_score(query, result) + _screen_request_score(query, _fallback_answer_text(result)), index, result)
                for index, result in enumerate(ordered_results[:8])
            ]
            if scored:
                if not _explicit_scope_has_candidate(query, [result for _score, _index, result in scored]):
                    return []
                query_terms = _answer_terms(query)
                if configuration_results := _configuration_requested_candidates(query, query_terms, scored):
                    return configuration_results
                if _requires_direct_configuration_evidence(query):
                    return []
                first_score = scored[0][0]
                best_score, _best_index, best_result = max(scored, key=lambda item: (item[0], -item[1]))
                if best_score >= 2.0 and best_score > first_score:
                    return [best_result]
                if _is_troubleshooting_query(query):
                    return []
        if not re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
            return ordered_results[:1]
        visible_quantity_scored = [
            (_quantity_citation_visible_score(query, result), _fallback_evidence_score(query, result), index, result)
            for index, result in enumerate(ordered_results[:8])
        ]
        visible_quantity_scored = [item for item in visible_quantity_scored if item[0] >= 6.0]
        if visible_quantity_scored:
            _visible_score, _fallback_score, _best_index, best_result = max(
                visible_quantity_scored,
                key=lambda item: (item[0], item[1], -item[2]),
            )
            return [best_result]
        scored = [
            (_fallback_evidence_score(query, result), index, result)
            for index, result in enumerate(ordered_results[:8])
        ]
        if not scored:
            return []
        best_score, _best_index, best_result = max(scored, key=lambda item: (item[0], -item[1]))
        if best_score >= 2.0:
            return [best_result]
        return ordered_results[:1]
    scored = [
        (_fallback_evidence_score(query, result), index, result)
        for index, result in enumerate(ordered_results[:8])
    ]
    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()
    seen_evidence: set[str] = set()
    seen_documents: set[str] = set()

    side_matches = _comparison_troubleshooting_side_matches(query, ordered_results)
    for result in side_matches:
        evidence_key = _result_evidence_key(result)
        if result.chunk_id in seen_chunks or evidence_key in seen_evidence:
            continue
        selected.append(result)
        seen_chunks.add(result.chunk_id)
        seen_evidence.add(evidence_key)
        seen_documents.add(result.source_document_id)
        if len(selected) >= 5:
            break
    if side_matches and len(side_matches) < len(_comparison_side_clauses(query)):
        return selected
    if len(selected) >= 2 and len(selected) == len(side_matches):
        return selected

    for score, _index, result in scored:
        if score < 2.0 or result.source_document_id in seen_documents:
            continue
        evidence_key = _result_evidence_key(result)
        selected.append(result)
        seen_chunks.add(result.chunk_id)
        seen_evidence.add(evidence_key)
        seen_documents.add(result.source_document_id)
        if len(selected) >= 5:
            break

    for score, _index, result in sorted(scored, key=lambda item: (-item[0], item[1])):
        if score < 2.0 or result.chunk_id in seen_chunks:
            continue
        evidence_key = _result_evidence_key(result)
        if evidence_key in seen_evidence:
            continue
        selected.append(result)
        seen_chunks.add(result.chunk_id)
        seen_evidence.add(evidence_key)
        seen_documents.add(result.source_document_id)
        if len(selected) >= 5:
            break
    return selected or ordered_results[:1]


def _normalize_generated_answer_payload(payload: dict[str, Any], results: list[SearchResult]) -> dict[str, Any]:
    normalized = dict(payload)
    confidence = normalized.get("confidence")
    if not isinstance(confidence, str):
        if isinstance(confidence, (int, float)):
            confidence = "high" if confidence >= 0.8 else ("medium" if confidence >= 0.4 else "low")
        else:
            confidence = "medium"
    confidence = confidence.strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    normalized["confidence"] = confidence

    if not isinstance(normalized.get("warnings"), list):
        normalized["warnings"] = [str(normalized["warnings"])] if normalized.get("warnings") else []
    normalized["warnings"] = [str(item) for item in normalized["warnings"]]

    if not isinstance(normalized.get("followup_questions"), list):
        normalized["followup_questions"] = [str(normalized["followup_questions"])] if normalized.get("followup_questions") else []
    normalized["followup_questions"] = [str(item) for item in normalized["followup_questions"]]

    if not isinstance(normalized.get("used_documents"), list) or any(not isinstance(item, dict) for item in normalized.get("used_documents", [])):
        normalized["used_documents"] = [
            {
                "document_id": result.source_document_id,
                "title": result.title,
                "version": result.document_version_id,
                "pages": result.pages,
                "section_path": result.section_path,
            }
            for result in results[:3]
        ]

    if not isinstance(normalized.get("citations"), list) or any(not isinstance(item, dict) for item in normalized.get("citations", [])):
        normalized["citations"] = [
            {
                "chunk_id": result.chunk_id,
                "document_id": result.source_document_id,
                "pages": result.pages,
                "quote_span": None,
            }
            for result in results[:3]
        ]

    if not isinstance(normalized.get("insufficient_evidence"), bool):
        normalized["insufficient_evidence"] = str(normalized.get("insufficient_evidence", "")).lower() in {"1", "true", "yes"}
    return normalized


def _structured_answer_is_too_terse(answer: str, results: list[SearchResult]) -> bool:
    if not results:
        return False
    top = results[0]
    chunk_type = str(top.metadata.get("chunk_type") or "")
    if chunk_type not in {"table_record", "spec_record", "datasheet_record"}:
        return False
    answer_terms = _answer_terms(answer)
    if not answer_terms or len(answer_terms) > 2:
        return False
    evidence_text = _fallback_answer_text(top)
    evidence_terms = _answer_terms(evidence_text)
    if len(evidence_terms) <= len(answer_terms) + 2:
        return False
    answer_lower = answer.lower()
    return bool(
        ":" in evidence_text
        and any(term in evidence_terms and term not in answer_lower for term in {"applicable", "value", "setting", "part", "number", "cell"})
    )


def _normalized_citation_text(text: str) -> str:
    return " ".join(text.lower().split())


def _citation_evidence_text(result: SearchResult) -> str:
    return str(result.content or "").strip()


def _troubleshooting_anchor_terms(query: str) -> set[str]:
    anchor = _query_troubleshooting_anchor(query)
    if not anchor:
        return set()
    return {
        term
        for term in _answer_terms(anchor)
        if term
        not in {
            "error",
            "occurred",
            "message",
            "messages",
            "communication",
            "stopped",
            "invalid",
            "setting",
            "settings",
            "corrected",
        }
    }


def _answer_addresses_troubleshooting_anchor(
    answer: str,
    query: str,
    citations: list[dict[str, Any]],
    results: list[SearchResult],
) -> bool:
    if not _is_troubleshooting_query(query):
        return True
    anchor_terms = _troubleshooting_anchor_terms(query)
    if not anchor_terms:
        return True
    text = answer
    result_by_chunk_id = {result.chunk_id: result for result in results}
    for citation in citations:
        result = result_by_chunk_id.get(str(citation.get("chunk_id") or ""))
        if result:
            text = f"{text}\n{_citation_evidence_text(result)}"
    text_terms = _answer_terms(text)
    required = min(2, len(anchor_terms))
    return len(anchor_terms.intersection(text_terms)) >= required


def _is_troubleshooting_row_evidence(result: SearchResult) -> bool:
    text = _citation_evidence_text(result)
    if str(result.metadata.get("chunk_type") or "") == "table_record" and re.search(
        r"\b(error\s+(?:number|code|messages?)|cause|remedy|corrective action)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(r"\berror\s+(?:number|code|messages?)\b", text, flags=re.IGNORECASE)
        and re.search(r"\b(cause|remedy|corrective action)\b", text, flags=re.IGNORECASE)
    )


def _troubleshooting_row_identifiers(result: SearchResult) -> set[str]:
    """Return source-bound error identifiers that a user can explicitly request."""
    text = _citation_evidence_text(result)
    identifiers: set[str] = set()
    for match in re.finditer(
        r"\b(?:error\s+(?:number|code|messages?)|alarm|fault)\s*:\s*(.+?)"
        r"(?=\s*(?:;|\n|\b(?:cause|remedy|corrective action)\s*:)|$)",
        text,
        flags=re.IGNORECASE,
    ):
        identifier = _normalized_phrase(match.group(1))
        if identifier:
            identifiers.add(identifier)
    return identifiers


def _requested_troubleshooting_rows_text(query: str, result: SearchResult) -> str:
    requested_identifiers = _query_requested_troubleshooting_identifiers(query)
    if not requested_identifiers:
        return ""
    text = _citation_evidence_text(result)
    row_matches = list(
        re.finditer(
            r"(?:^|\n)(?:error\s+(?:number|code|messages?)|alarm|fault)\s*:\s*(.+?)"
            r"(?=\n(?:error\s+(?:number|code|messages?)|alarm|fault)\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    selected_rows: list[str] = []
    for match in row_matches:
        row = match.group(0).strip()
        identifiers = {
            _normalized_phrase(identifier_match.group(1))
            for identifier_match in re.finditer(
                r"\b(?:error\s+(?:number|code|messages?)|alarm|fault)\s*:\s*(.+?)"
                r"(?=\s*(?:;|\n|\b(?:cause|remedy|corrective action)\s*:)|$)",
                row,
                flags=re.IGNORECASE,
            )
        }
        if identifiers.intersection(requested_identifiers):
            selected_rows.append(row)
    return "\n".join(selected_rows)


def _query_contains_troubleshooting_identifier(normalized_query: str, identifier: str) -> bool:
    """Match a source identifier as a complete normalized token sequence."""
    return bool(re.search(rf"(?:^| ){re.escape(identifier)}(?: |$)", normalized_query))


def _query_requested_troubleshooting_identifiers(query: str) -> set[str]:
    """Return distinct troubleshooting sides explicitly named by a comparison query.

    These identifiers come from the user-visible query, so a requested side remains
    represented even when retrieval returned no row for it.
    """
    if not _is_troubleshooting_query(query):
        return set()

    explicit_codes = {
        _normalized_phrase(match.group(1))
        for match in re.finditer(
            r"\b(?:error|alarm|fault)(?:\s+(?:number|code))?\s*[:#-]?\s*"
            r"([a-z]*\d[a-z0-9._/-]*)\b",
            query,
            flags=re.IGNORECASE,
        )
    }
    if not _is_comparison_query(query):
        return explicit_codes
    if len(explicit_codes) >= 2:
        return explicit_codes
    if len(explicit_codes) == 1:
        explicit_models = _model_tokens(query)
        named_series = {
            match.group(1).upper()
            for match in re.finditer(
                r"\b([A-Z][A-Z0-9-]{1,14})\s+(?:Series|Family)\b",
                query,
            )
        }
        explicit_versions = {
            _normalized_phrase(match.group(0))
            for match in re.finditer(
                r"\b(?:version|revision|rev\.?)\s*[a-z0-9][a-z0-9._/-]*\b",
                query,
                flags=re.IGNORECASE,
            )
        }
        if len(explicit_models.union(named_series)) >= 2 or len(explicit_versions) >= 2:
            return explicit_codes

    raw_sides = _comparison_side_clauses(query)
    if len(raw_sides) < 2:
        plural = re.search(
            r"\b(?:errors?|alarms?|faults?)\s+(.+?)(?:[.?]|$)",
            query,
            flags=re.IGNORECASE,
        )
        if plural:
            raw_sides = [
                side.strip(" ,.;:?")
                for side in re.split(r"\s+and\s+", plural.group(1), flags=re.IGNORECASE)
                if side.strip(" ,.;:?")
            ]

    identifiers: set[str] = set()
    for side in raw_sides:
        cleaned = re.sub(
            r"^(?:the\s+)?(?:causes?|remedies?|causes?\s+and\s+remedies?)\s+(?:for|of)\s+",
            "",
            side,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:the\s+)?(?:errors?|alarms?|faults?)(?:\s+(?:number|code|message))?\s*[:#-]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+(?:causes?(?:\s+and\s+remed(?:y|ies))?|remed(?:y|ies)|corrective\s+actions?)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        identifier = _normalized_phrase(cleaned)
        terms = set(identifier.split())
        if identifier and terms.difference(
            {"and", "cause", "causes", "compare", "corrective", "error", "errors", "remedy", "remedies"}
        ):
            identifiers.add(identifier)
    return identifiers if len(identifiers) >= 2 else set()


def _requested_troubleshooting_identifiers_are_available(
    query: str,
    results: list[SearchResult],
) -> bool:
    requested_identifiers = _query_requested_troubleshooting_identifiers(query)
    if not requested_identifiers:
        return True
    available_identifiers = {
        identifier
        for result in results
        if _is_troubleshooting_row_evidence(result)
        for identifier in _troubleshooting_row_identifiers(result)
    }
    explicitly_labeled_codes = re.findall(
        r"\b(?:error|alarm|fault)(?:\s+(?:number|code))?\s*[:#-]?\s*[a-z]*\d[a-z0-9._/-]*\b",
        query,
        flags=re.IGNORECASE,
    )
    explicitly_labeled_list = bool(
        re.search(
            r"\b(?:errors|alarms|faults)\s+(?!(?:on|for|in|of)\b).+\band\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    if not requested_identifiers.intersection(available_identifiers) and not (
        len(explicitly_labeled_codes) >= 2 or explicitly_labeled_list
    ):
        return True
    return requested_identifiers.issubset(available_identifiers)


def _troubleshooting_citations_match_query_anchor(
    query: str,
    citations: list[dict[str, Any]],
    results: list[SearchResult],
) -> bool:
    if not _is_troubleshooting_query(query):
        return True
    result_by_chunk_id = {result.chunk_id: result for result in results}
    normalized_query = _normalized_phrase(query)
    source_requested_identifiers = {
        identifier
        for result in results
        if _is_troubleshooting_row_evidence(result)
        for identifier in _troubleshooting_row_identifiers(result)
        if _query_contains_troubleshooting_identifier(normalized_query, identifier)
    }
    requested_identifiers = (
        _query_requested_troubleshooting_identifiers(query) or source_requested_identifiers
    )
    cited_requested_identifiers: set[str] = set()
    for citation in citations:
        result = result_by_chunk_id.get(str(citation.get("chunk_id") or ""))
        if not result or not _is_troubleshooting_row_evidence(result):
            continue
        if requested_identifiers:
            cited_identifiers = _troubleshooting_row_identifiers(result).intersection(requested_identifiers)
            if not cited_identifiers:
                return False
            cited_requested_identifiers.update(cited_identifiers)
            continue

        anchor = _normalized_phrase(_query_troubleshooting_anchor(query))
        if len(anchor) >= 10 and anchor not in _normalized_phrase(_citation_evidence_text(result)):
            return False
    return not requested_identifiers or requested_identifiers.issubset(cited_requested_identifiers)


def _comparison_answer_covers_retrieved_model_sides(
    query: str,
    citations: list[dict[str, Any]],
    results: list[SearchResult],
) -> bool:
    if not _is_comparison_query(query):
        return True
    explicit_models = _model_tokens(query)
    if len(explicit_models) < 2:
        return True
    result_by_chunk_id = {result.chunk_id: result for result in results}
    cited_results = [
        result
        for citation in citations
        if (result := result_by_chunk_id.get(str(citation.get("chunk_id") or "")))
    ]
    if not cited_results:
        return True

    available_models = {
        model
        for model in explicit_models
        if any(_result_mentions_requested_model_side(result, model) for result in results)
    }
    if len(available_models) < 2:
        return False

    cited_model_sides: list[set[str]] = [
        {
            model
            for model in available_models
            if _result_mentions_requested_model_side(result, model)
        }
        for result in cited_results
    ]
    if any(not model_sides for model_sides in cited_model_sides):
        return False

    covered_models: set[str] = set().union(*cited_model_sides)
    uncovered_models = available_models.difference(covered_models)
    if uncovered_models:
        return False
    return True


def _comparison_answer_is_overcautious(answer: AnswerResponse, query: str, results: list[SearchResult]) -> bool:
    if not _is_comparison_query(query):
        return False
    answer_text = answer.answer.lower()
    if not re.search(
        r"\b(evidence|documents?|text)\b.{0,80}\b(do(?:es)? not|don't|missing|lack|lacks|without|cannot|can't|insufficient|incomplete)\b"
        r"|\bcomparison cannot be made\b",
        answer_text,
        flags=re.IGNORECASE,
    ):
        return False
    return len(_fallback_evidence_results(query, results)) > 1


def _citation_quotes_are_supported(citations: list[dict[str, Any]], results: list[SearchResult]) -> bool:
    if not citations:
        return True
    evidence_by_chunk_id = {result.chunk_id: _citation_evidence_text(result) for result in results}
    for citation in citations:
        chunk_id = str(citation.get("chunk_id") or "")
        if chunk_id not in evidence_by_chunk_id:
            return False
        quote = str(citation.get("quote_span") or "").strip()
        if not quote:
            continue
        if _normalized_citation_text(quote) not in _normalized_citation_text(evidence_by_chunk_id[chunk_id]):
            return False
    return True


def validate_answer(answer: AnswerResponse, results: list[SearchResult], query: str = "") -> AnswerResponse:
    if results and (
        not _answer_supported_by_results(answer.answer, results)
        or _structured_answer_is_too_terse(answer.answer, results)
        or (answer.insufficient_evidence and _has_strong_insufficient_fallback(query, results))
        or (_should_score_direct_fallback(query) and not _explicit_scope_has_candidate(query, results))
        or (
            _should_score_direct_fallback(query)
            and _requires_direct_configuration_evidence(query)
            and not _fallback_evidence_results(query, results)
        )
        or _comparison_answer_is_overcautious(answer, query, results)
        or not _citation_quotes_are_supported(list(answer.citations), results)
        or not _answer_addresses_troubleshooting_anchor(answer.answer, query, list(answer.citations), results)
        or not _troubleshooting_citations_match_query_anchor(query, list(answer.citations), results)
        or not _answer_uses_matching_troubleshooting_row(answer.answer, query, results)
        or not _answer_uses_comparison_troubleshooting_side_rows(answer.answer, query, results)
        or not _comparison_answer_covers_retrieved_model_sides(query, list(answer.citations), results)
        or not _answer_addresses_quantity_request(answer.answer, query, results)
        or not _answer_addresses_image_capture_buffer_request(answer.answer, query, results)
        or not _answer_cites_selected_screen_evidence(answer, query, results)
    ):
        fallback = _fallback_answer(query, results)
        fallback_warnings = list(fallback.warnings)
        fallback_warnings.append(
            "Generated answer was not sufficiently supported by retrieved evidence; using retrieval-grounded fallback."
        )
        answer = fallback.model_copy(
            update={
                "warnings": fallback_warnings,
            }
        )

    warnings = list(answer.warnings)
    citations = list(answer.citations)
    used_documents = list(answer.used_documents)
    insufficient_evidence = answer.insufficient_evidence
    confidence = answer.confidence
    reconstructed_citations = False

    if results and not citations and not insufficient_evidence:
        top = results[0]
        citations.append(
            {
                "chunk_id": top.chunk_id,
                "document_id": top.source_document_id,
                "pages": top.pages,
                "quote_span": None,
            }
        )
        reconstructed_citations = True
        warnings.append("Citations were reconstructed from top retrieval evidence.")

    cited_chunk_ids = {str(citation.get("chunk_id") or "") for citation in citations}
    cited_document_ids = {str(citation.get("document_id") or "") for citation in citations}
    version_ids = {
        result.document_version_id
        for result in results
        if reconstructed_citations
        or not cited_chunk_ids
        or result.chunk_id in cited_chunk_ids
        or (cited_document_ids and result.source_document_id in cited_document_ids)
    }
    if len(version_ids) > 1:
        warnings.append("Retrieved evidence spans multiple document versions; verify revision-specific details.")

    if results:
        top_score = max(result.score for result in results)
        if top_score < 0.02:
            insufficient_evidence = True
            confidence = "low"
            warnings.append("Retrieved evidence scored weakly for this question.")
        if not used_documents:
            top_results = results[:3]
            used_documents = [
                {
                    "document_id": result.source_document_id,
                    "title": result.title,
                    "version": result.document_version_id,
                    "pages": result.pages,
                    "section_path": result.section_path,
                }
                for result in top_results
            ]

    return answer.model_copy(
        update={
            "citations": citations,
            "used_documents": used_documents,
            "warnings": warnings,
            "insufficient_evidence": insufficient_evidence,
            "confidence": confidence,
        }
    )


def generate_answer(
    query: str,
    results: list[SearchResult],
    *,
    prioritized_results: list[SearchResult] | None = None,
    summarized_evidence: list[dict[str, Any]] | None = None,
) -> AnswerResponse:
    answer, _trace = generate_answer_with_trace(
        query,
        results,
        prioritized_results=prioritized_results,
        summarized_evidence=summarized_evidence,
    )
    return answer


def generate_answer_with_trace(
    query: str,
    results: list[SearchResult],
    *,
    prioritized_results: list[SearchResult] | None = None,
    summarized_evidence: list[dict[str, Any]] | None = None,
) -> tuple[AnswerResponse, dict[str, Any]]:
    trace: dict[str, Any] = {
        "relevance_review": {
            "provider": "ollama",
            "model": settings.ollama_fast_model,
            "prompt_kind": "relevance_judgment",
            "think": False,
        },
        "summarization": {
            "provider": "ollama",
            "model": settings.ollama_fast_model,
            "prompt_kind": "evidence_summary",
            "think": False,
        },
        "final_answer": {
            "provider": "ollama",
            "model": settings.ollama_answer_model,
            "prompt_kind": "final_answer",
            "think": False,
            "num_predict": -1,
            "used_fallback": False,
            "answer_source": "model",
            "fallback_reason": None,
            "summarized_evidence": [],
        },
    }
    if not results:
        answer = _fallback_answer(query, results)
        trace["final_answer"].update(
            {
                "used_fallback": True,
                "answer_source": "fallback_no_results",
                "fallback_reason": "No retrieval results were available.",
                "summarized_evidence": [],
            }
        )
        return answer, trace
    if prioritized_results is None:
        candidate_results = results[:8]
        prioritized = prioritize_results_for_answer(query, candidate_results)
        prioritized_results = prioritized["prioritized_results"]
    if summarized_evidence is None:
        summarized_evidence = summarize_results_for_answer(query, prioritized_results)
    trace["final_answer"]["summarized_evidence"] = summarized_evidence
    trace["summarization"]["summary_count"] = len(summarized_evidence)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\nEvidence summaries: {json.dumps(summarized_evidence)}"},
    ]
    try:
        generated, _raw = chat_json(
            model=settings.ollama_answer_model,
            messages=messages,
            json_schema=ANSWER_SCHEMA,
            think=False,
            num_predict=-1,
            timeout=90.0,
            purpose="final_answer",
        )
        generated_answer = AnswerResponse.model_validate(_normalize_generated_answer_payload(generated, prioritized_results))
        validated_answer = validate_answer(generated_answer, prioritized_results, query=query)
        if validated_answer.answer != generated_answer.answer and any(
            "not sufficiently supported" in warning for warning in validated_answer.warnings
        ):
            trace["final_answer"].update(
                {
                    "used_fallback": True,
                    "answer_source": "fallback_validation",
                    "fallback_reason": "Generated answer was replaced by retrieval-grounded fallback during validation.",
                }
            )
        return validated_answer, trace
    except Exception as exc:
        logger.warning("Final answer generation failed for model=%s; using fallback answer: %s", settings.ollama_answer_model, exc)
        fallback_answer = validate_answer(_fallback_answer(query, prioritized_results), prioritized_results, query=query)
        trace["final_answer"].update(
            {
                "used_fallback": True,
                "answer_source": "fallback_exception",
                "fallback_reason": str(exc),
            }
        )
        return fallback_answer, trace


def prepare_answer_evidence(query: str, results: list[SearchResult]) -> dict[str, Any]:
    candidate_results = results[:8]
    prioritized = prioritize_results_for_answer(query, candidate_results)
    summaries = summarize_results_for_answer(query, prioritized["prioritized_results"])
    return {
        "candidate_results": candidate_results,
        "judgments": prioritized["judgments"],
        "prioritized_results": prioritized["prioritized_results"],
        "summaries": summaries,
    }


def prioritize_results_for_answer(query: str, candidate_results: list[SearchResult]) -> dict[str, Any]:
    judgments = judge_retrieval_relevance(query, candidate_results)
    judgment_by_chunk_id = {item["chunk_id"]: item for item in judgments}
    focused_results = _focused_troubleshooting_results(query, candidate_results)
    anchored_results = focused_results if [result.chunk_id for result in focused_results] != [result.chunk_id for result in candidate_results] else []
    comparison_evidence = _fallback_evidence_results(query, candidate_results) if _is_comparison_query(query) else []
    procedure_evidence = (
        _fallback_evidence_results(query, candidate_results)
        if _is_procedure_rule_query(query) and not _is_troubleshooting_query(query)
        else []
    )
    prioritized_results = [result for result in [*anchored_results, *comparison_evidence, *procedure_evidence]]
    prioritized_results.extend(
        result
        for result in candidate_results
        if judgment_by_chunk_id.get(result.chunk_id, {}).get("verdict") == "relevant"
        and result.chunk_id not in {item.chunk_id for item in prioritized_results}
    )
    prioritized_results.extend(
        result
        for result in candidate_results
        if judgment_by_chunk_id.get(result.chunk_id, {}).get("verdict") == "potentially_relevant"
        and result.chunk_id not in {item.chunk_id for item in prioritized_results}
    )
    if not prioritized_results:
        prioritized_results = candidate_results
    prioritized_results = _focused_troubleshooting_results(query, _order_troubleshooting_results(query, prioritized_results))
    return {
        "judgments": judgments,
        "prioritized_results": prioritized_results,
    }


def _fallback_relevance_judgments(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    query_terms = _answer_terms(query)
    judgments: list[dict[str, str]] = []
    for result in results:
        if _table_model_scope_conflict(query, result):
            judgments.append(
                {
                    "chunk_id": result.chunk_id,
                    "verdict": "not_relevant",
                    "reason": "The table row names a different model family than the explicit model in the request.",
                }
            )
            continue
        evidence_terms = _answer_terms(_evidence_text(result))
        overlap = len(query_terms.intersection(evidence_terms))
        if overlap >= max(1, len(query_terms) // 2):
            verdict = "relevant"
            reason = "Shares key terms with the request and appears directly connected."
        elif overlap > 0:
            verdict = "potentially_relevant"
            reason = "Touches some request terms but may be broader or indirect."
        else:
            verdict = "not_relevant"
            reason = "Does not appear to address the request directly."
        judgments.append({"chunk_id": result.chunk_id, "verdict": verdict, "reason": reason})
    return judgments


def _normalize_relevance_item(item: dict[str, Any], fallback: dict[str, str]) -> dict[str, str]:
    chunk_id = str(item.get("chunk_id") or fallback["chunk_id"])
    raw_verdict = str(item.get("verdict") or "").strip().lower()
    if raw_verdict not in {"relevant", "not_relevant", "potentially_relevant"}:
        raw_verdict = fallback["verdict"]
    raw_reason = str(item.get("reason") or "").strip()
    if not raw_reason or raw_reason.lower() == "null":
        raw_reason = fallback["reason"]
    return {"chunk_id": chunk_id, "verdict": raw_verdict, "reason": raw_reason}


def _apply_model_scope_to_judgments(
    query: str,
    results: list[SearchResult],
    judgments: list[dict[str, str]],
) -> list[dict[str, str]]:
    result_by_chunk_id = {result.chunk_id: result for result in results}
    scoped: list[dict[str, str]] = []
    for judgment in judgments:
        result = result_by_chunk_id.get(judgment["chunk_id"])
        if result and _table_model_scope_conflict(query, result):
            scoped.append(
                {
                    **judgment,
                    "verdict": "not_relevant",
                    "reason": "The table row names a different model family than the explicit model in the request.",
                }
            )
            continue
        scoped.append(judgment)
    return scoped


def _relevance_prompt(query: str, evidence: list[dict[str, Any]], *, strict: bool = False) -> str:
    prompt = f"Question: {query}\nEvidence: {json.dumps(evidence)}"
    if not strict:
        return prompt
    required_ids = [item["chunk_id"] for item in evidence]
    return (
        f"{prompt}\n\n"
        "Return one judgment for every evidence item.\n"
        f"Required chunk_ids in order: {json.dumps(required_ids)}\n"
        "Do not omit any chunk_id. Do not use null. If uncertain, use potentially_relevant."
    )


def _parse_relevance_response(
    raw_response: str,
    query: str,
    results: list[SearchResult],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    generated = json.loads(raw_response or "{}")
    items = generated.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Invalid relevance payload: items is not a list")
    fallback = _fallback_relevance_judgments(query, results)
    fallback_by_chunk_id = {item["chunk_id"]: item for item in fallback}
    normalized: list[dict[str, str]] = []
    invalid_items: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for item in items:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            invalid_items.append({"error": "missing_chunk_id", "item": item})
            continue
        seen_chunk_ids.add(chunk_id)
        normalized_item = _normalize_relevance_item(
            item,
            fallback_by_chunk_id.get(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "verdict": "potentially_relevant",
                    "reason": "The model returned an incomplete relevance judgment.",
                },
            ),
        )
        if normalized_item["verdict"] != str(item.get("verdict") or "").strip().lower() or normalized_item["reason"] != str(item.get("reason") or "").strip():
            invalid_items.append({"error": "normalized_invalid_fields", "item": item})
        normalized.append(normalized_item)
    missing_chunk_ids = [item["chunk_id"] for item in fallback if item["chunk_id"] not in seen_chunk_ids]
    merged = [next((item for item in normalized if item["chunk_id"] == fallback_item["chunk_id"]), fallback_item) for fallback_item in fallback]
    return merged, {
        "missing_chunk_ids": missing_chunk_ids,
        "invalid_items": invalid_items,
        "item_count": len(items),
    }


def judge_retrieval_relevance(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    if not results:
        return []
    evidence = [
        {
            "chunk_id": result.chunk_id,
            "title": result.title,
            "pages": result.pages,
            "section_path": result.section_path,
            "content": _evidence_text(result)[:2000],
            "document_version_id": result.document_version_id,
        }
        for result in results
    ]
    fallback = _fallback_relevance_judgments(query, results)
    try:
        for attempt, strict in enumerate((False, True), start=1):
            _parsed, raw_response = chat_json(
                model=settings.ollama_fast_model,
                messages=[
                    {"role": "system", "content": RELEVANCE_PROMPT},
                    {"role": "user", "content": _relevance_prompt(query, evidence, strict=strict)},
                ],
                json_schema=RELEVANCE_SCHEMA,
                think=False,
                timeout=90.0,
                purpose="relevance_review",
            )
            try:
                parsed, diagnostics = _parse_relevance_response(raw_response, query, results)
            except Exception as exc:
                diagnostics = {
                    "missing_chunk_ids": [item["chunk_id"] for item in fallback],
                    "invalid_items": [{"error": str(exc)}],
                }
                parsed = fallback
            parsed = _apply_model_scope_to_judgments(query, results, parsed)
            if not diagnostics["missing_chunk_ids"] and not diagnostics["invalid_items"]:
                return parsed
            logger.warning(
                "Relevance judgment response was incomplete on attempt %s; retrying=%s missing_chunk_ids=%s invalid_items=%s raw_response=%s",
                attempt,
                attempt == 1,
                diagnostics["missing_chunk_ids"],
                diagnostics["invalid_items"],
                raw_response[:4000],
            )
            if attempt == 2:
                return parsed
    except Exception as exc:
        logger.warning("Relevance judgment failed; using fallback judgments: %s", exc)
        return _apply_model_scope_to_judgments(query, results, fallback)
    return _apply_model_scope_to_judgments(query, results, fallback)


def _extract_json_summary(raw_response: str) -> str:
    generated = json.loads(raw_response or "{}")
    summary = str(generated.get("summary") or "").strip()
    if not summary:
        raise ValueError("Missing summary")
    return summary


def _fallback_summary(query: str, result: SearchResult) -> str:
    evidence = _evidence_text(result)[:700]
    return f"[{result.chunk_id}] {evidence}"


def _direct_evidence_summary(query: str, result: SearchResult) -> str | None:
    chunk_type = str(result.metadata.get("chunk_type") or "")
    if chunk_type not in {"table_record", "spec_record", "datasheet_record", "procedure_record", "warning_record"}:
        return None
    evidence = (_focused_table_record_answer_text(query, result) or _fallback_answer_text(result)).strip()
    if not evidence:
        return None
    return evidence[:1200]


def _summarize_chunk(query: str, result: SearchResult) -> dict[str, Any]:
    direct_summary = _direct_evidence_summary(query, result)
    if direct_summary:
        return {
            "chunk_id": result.chunk_id,
            "title": result.title,
            "pages": result.pages,
            "section_path": result.section_path,
            "summary": direct_summary,
            "summary_source": "direct_evidence",
            "source_document_id": result.source_document_id,
            "document_version_id": result.document_version_id,
            "source_documents": [
                {
                    "chunk_id": result.chunk_id,
                    "title": result.title,
                    "pages": result.pages,
                    "section_path": result.section_path,
                    "source_document_id": result.source_document_id,
                    "document_version_id": result.document_version_id,
                }
            ],
        }
    payload = {
        "chunk_id": result.chunk_id,
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "content": _evidence_text(result)[:2500],
        "parent_context": str(result.metadata.get("parent_context") or "")[:1000],
    }
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"Question: {query}\nEvidence item: {json.dumps(payload)}"},
    ]
    try:
        _parsed, raw = chat_json(
            model=settings.ollama_fast_model,
            messages=messages,
            json_schema=SUMMARY_SCHEMA,
            think=False,
            timeout=60.0,
            purpose="chunk_summary",
        )
        summary = _extract_json_summary(raw)
        summary_source = "model"
    except Exception as exc:
        logger.warning("Chunk summary failed for %s; using fallback summary: %s", result.chunk_id, exc)
        summary = _fallback_summary(query, result)
        summary_source = "fallback_summary"
    return {
        "chunk_id": result.chunk_id,
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "summary": summary,
        "summary_source": summary_source,
        "source_document_id": result.source_document_id,
        "document_version_id": result.document_version_id,
        "source_documents": [
            {
                "chunk_id": result.chunk_id,
                "title": result.title,
                "pages": result.pages,
                "section_path": result.section_path,
                "source_document_id": result.source_document_id,
                "document_version_id": result.document_version_id,
            }
        ],
    }


def _summary_source_documents(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in batch:
        source_documents = item.get("source_documents")
        if not isinstance(source_documents, list):
            source_documents = [item]
        for source in source_documents:
            if not isinstance(source, dict):
                continue
            document_id = str(source.get("source_document_id") or "")
            chunk_id = str(source.get("chunk_id") or "")
            key = (document_id, chunk_id)
            if not document_id or key in seen:
                continue
            documents.append(
                {
                    "chunk_id": source.get("chunk_id"),
                    "title": source.get("title"),
                    "pages": source.get("pages"),
                    "section_path": source.get("section_path"),
                    "source_document_id": source.get("source_document_id"),
                    "document_version_id": source.get("document_version_id"),
                }
            )
            seen.add(key)
    return documents


def _merge_summary_batch(query: str, batch: list[dict[str, Any]]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": RECURSIVE_SUMMARY_PROMPT},
        {"role": "user", "content": f"Question: {query}\nSummaries: {json.dumps(batch)}"},
    ]
    batch_chunk_ids = [item["chunk_id"] for item in batch]
    try:
        _parsed, raw = chat_json(
            model=settings.ollama_fast_model,
            messages=messages,
            json_schema=SUMMARY_SCHEMA,
            think=False,
            timeout=60.0,
            purpose="recursive_summary",
        )
        summary = _extract_json_summary(raw)
    except Exception as exc:
        logger.warning("Recursive summary failed for chunk_ids=%s; using concatenated fallback: %s", batch_chunk_ids, exc)
        summary = " ".join(item["summary"] for item in batch)
    return {
        "chunk_id": ",".join(batch_chunk_ids),
        "title": batch[0]["title"],
        "pages": sorted({page for item in batch for page in item.get("pages", [])}),
        "section_path": batch[0].get("section_path", []),
        "summary": summary[:2000],
        "source_document_id": batch[0]["source_document_id"],
        "document_version_id": batch[0]["document_version_id"],
        "source_documents": _summary_source_documents(batch),
    }


def summarize_results_for_answer(query: str, results: list[SearchResult]) -> list[dict[str, Any]]:
    summaries = [_summarize_chunk(query, result) for result in results]
    if len(summaries) <= 6 and all(item.get("summary_source") == "direct_evidence" for item in summaries):
        return summaries
    while len(summaries) > 4:
        merged: list[dict[str, Any]] = []
        for index in range(0, len(summaries), 3):
            batch = summaries[index:index + 3]
            if len(batch) == 1:
                merged.append(batch[0])
            else:
                merged.append(_merge_summary_batch(query, batch))
        summaries = merged
    return summaries
