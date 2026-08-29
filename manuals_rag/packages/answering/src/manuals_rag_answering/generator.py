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


MODEL_TOKEN_RE = re.compile(r"\b[A-Z0-9]{1,5}-[A-Z0-9]{2,12}[A-Z]?\b(?!-)")


def _model_tokens(text: str) -> set[str]:
    return {match.group(0).upper() for match in MODEL_TOKEN_RE.finditer(text)}


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
    for clause_values in _quantity_role_value_groups(text):
        for role, values in clause_values.items():
            role_values.setdefault(role, set()).update(values)
    return role_values


def _quantity_role_value_groups(text: str) -> list[dict[str, set[str]]]:
    groups: list[dict[str, set[str]]] = []
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
                _add_quantity_role_value(role_values, role, match.group(1))
            for match in value_before_role.finditer(clause):
                _add_quantity_role_value(role_values, role, match.group(1))
        if role_values:
            groups.append(role_values)
    return groups


def _answer_addresses_quantity_request(answer: str, query: str, results: list[SearchResult]) -> bool:
    if not re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
        return True
    query_terms = _answer_terms(query)
    requested_roles = _requested_quantity_roles(query)
    answer_role_values = _quantity_role_values(answer)
    candidate_role_values: list[dict[str, set[str]]] = []
    saw_partial_requested_role_evidence = False
    answer_quantities = _contextual_quantity_terms(answer)
    candidate_quantities: set[str] = set()
    for result in results[:8]:
        evidence = _fallback_answer_text(result)
        quantities = _contextual_quantity_terms(evidence)
        evidence_role_values = _quantity_role_values(evidence)
        evidence_role_value_groups = _quantity_role_value_groups(evidence)
        if not quantities and not evidence_role_values:
            continue
        overlap = len(query_terms.intersection(_answer_terms(evidence)))
        if requested_roles:
            for group in evidence_role_value_groups:
                role_values = {role: values for role, values in group.items() if role in requested_roles and values}
                if role_values:
                    saw_partial_requested_role_evidence = True
                if requested_roles.issubset(role_values):
                    candidate_role_values.append(role_values)
        if overlap >= 3:
            candidate_quantities.update(quantities)
    if candidate_role_values:
        role_values = candidate_role_values[0]
        return all(answer_role_values.get(role, set()).intersection(values) for role, values in role_values.items())
    if requested_roles and saw_partial_requested_role_evidence:
        return False
    if not candidate_quantities:
        return True
    return bool(answer_quantities.intersection(candidate_quantities))


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
    return bool(re.search(r"\b(compare|comparison|differ|differs|difference|different|versus|vs\.?|while|whereas)\b", query, flags=re.IGNORECASE)) or bool(
        re.search(r"\bwhat\b.+\band what\b", query, flags=re.IGNORECASE)
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
    fallback_results = _fallback_evidence_results(query, results)
    top = fallback_results[0]
    if len(fallback_results) == 1:
        answer_text = (
            _matching_troubleshooting_row_text(query, top)
            or _focused_table_record_answer_text(query, top)
            or _fallback_answer_text(top)
        )
    else:
        answer_text = "Retrieved evidence:\n" + "\n".join(
            f"- {result.title}, page(s) {', '.join(str(page) for page in result.pages) or 'unknown'}: "
            f"{_focused_table_record_answer_text(query, result) or _fallback_answer_text(result)}"
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


def _multi_part_fallback_evidence_results(query: str, ordered_results: list[SearchResult]) -> list[SearchResult]:
    clauses = _multi_part_evidence_clauses(query)
    if len(clauses) < 2:
        return []

    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()
    seen_evidence: set[str] = set()
    for clause in clauses:
        scored = [
            (_fallback_evidence_score(clause, result), index, result)
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
        if len(selected) >= 4:
            break

    if len(selected) >= 2:
        return selected
    return []


def _fallback_evidence_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    ordered_results = _comparison_scoped_troubleshooting_results(
        query,
        _focused_troubleshooting_results(query, _order_troubleshooting_results(query, results)),
    )
    if not _is_comparison_query(query):
        multi_part_results = _multi_part_fallback_evidence_results(query, ordered_results)
        if multi_part_results:
            return multi_part_results
        if not re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
            return ordered_results[:1]
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
    content = str(result.content or "").strip()
    metadata_content = str(result.metadata.get("content") or "").strip()
    if metadata_content and metadata_content != content:
        return f"{content}\n{metadata_content}" if content else metadata_content
    return content


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
        or (answer.insufficient_evidence and len(_fallback_evidence_results(query, results)) > 1)
        or _comparison_answer_is_overcautious(answer, query, results)
        or not _citation_quotes_are_supported(list(answer.citations), results)
        or not _answer_addresses_troubleshooting_anchor(answer.answer, query, list(answer.citations), results)
        or not _answer_uses_matching_troubleshooting_row(answer.answer, query, results)
        or not _answer_uses_comparison_troubleshooting_side_rows(answer.answer, query, results)
        or not _comparison_answer_covers_retrieved_model_sides(query, list(answer.citations), results)
        or not _answer_addresses_quantity_request(answer.answer, query, results)
    ):
        fallback = _fallback_answer(query, results)
        answer = fallback.model_copy(
            update={
                "warnings": [
                    *answer.warnings,
                    "Generated answer was not sufficiently supported by retrieved evidence; using retrieval-grounded fallback.",
                ]
            }
        )

    warnings = list(answer.warnings)
    citations = list(answer.citations)
    used_documents = list(answer.used_documents)
    insufficient_evidence = answer.insufficient_evidence
    confidence = answer.confidence

    if results and not citations:
        top = results[0]
        citations.append(
            {
                "chunk_id": top.chunk_id,
                "document_id": top.source_document_id,
                "pages": top.pages,
                "quote_span": None,
            }
        )
        warnings.append("Citations were reconstructed from top retrieval evidence.")

    version_ids = {result.document_version_id for result in results}
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
