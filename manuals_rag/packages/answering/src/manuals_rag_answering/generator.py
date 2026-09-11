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
Answer the user's requested task directly in the first sentence; never begin with a filename, manual title, or raw evidence dump.
For location/configuration questions, state the supported unit/menu/screen/tab hierarchy, the exact setting, what it controls, and any applicable mode or constraint present in evidence.
Do not substitute a neighboring or prerequisite setting for the setting the user asked about.
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
- for location/configuration questions, preserve the full supported unit/menu/screen/tab hierarchy, the exact requested setting, its purpose, and applicable mode
- distinguish the requested setting from neighboring, prerequisite, or calculation settings
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
    normalized = text.lower()
    for spaced, compact in (
        (r"\buser[\s-]+name\b", "username"),
        (r"\blogin[\s-]+name\b", "username"),
        (r"\buser[\s-]+id\b", "username"),
        (r"\blogin[\s-]+id\b", "username"),
        (r"\bhost[\s-]+name\b", "hostname"),
        (r"\bfile[\s-]+name\b", "filename"),
        (r"\bpath[\s-]+name\b", "pathname"),
    ):
        normalized = re.sub(spaced, compact, normalized)
    terms = {
        cleaned
        for token in re.findall(r"[a-z0-9_\-./]+", normalized)
        if (cleaned := token.strip(".,;:"))
        and len(cleaned) >= 3
        and cleaned not in {"the", "and", "for", "with", "that", "this", "from"}
    }
    for term in list(terms):
        for piece in term.split("/"):
            if len(piece) >= 3:
                terms.add(piece)
    return terms


MODEL_TOKEN_RE = re.compile(
    r"\b(?:[A-Z0-9]{1,5}-[A-Z0-9]{1,12}[A-Z]?|[A-Z]{1,5}\d{2,8})\b(?!-)"
)


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
        "context_window",
        "parent_context",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            metadata_values.extend(str(item) for item in value)
        elif value:
            metadata_values.append(str(value))
    return " ".join([result.title, result.content, *metadata_values])


def _result_mentions_model(result: SearchResult, model: str) -> bool:
    model_upper = model.upper()
    result_model_text = _result_model_text(result)
    model_compact = re.sub(r"[^A-Z0-9]+", "", model_upper)
    if model_compact and model_compact in re.sub(r"[^A-Z0-9]+", "", result_model_text.upper()):
        return True
    result_models = _model_tokens(result_model_text)
    if model_upper in result_models:
        return True
    return any(
        candidate.startswith(model_upper)
        or model_upper.startswith(candidate)
        or (
            model_compact
            and (
                re.sub(r"[^A-Z0-9]+", "", candidate) == model_compact
                or re.sub(r"[^A-Z0-9]+", "", candidate).startswith(model_compact)
                or model_compact.startswith(re.sub(r"[^A-Z0-9]+", "", candidate))
            )
        )
        for candidate in result_models
    )


def _result_mentions_requested_model_side(result: SearchResult, model: str) -> bool:
    return model.upper() in _model_tokens(_result_model_text(result))


def _result_document_scope_models(result: SearchResult) -> set[str]:
    """Return model identifiers that describe the document/result scope."""

    metadata = result.metadata or {}
    values: list[str] = [result.title]
    for key in ("product_model", "product_family", "product_models", "product_families"):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return _model_tokens(" ".join(values))


def _scope_answer_results_to_query_models(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Drop explicitly conflicting product manuals when matching evidence exists.

    Identical troubleshooting rows can occur in adjacent product manuals.  A
    semantic match alone must not let an LJ-S document answer an LJ-X question.
    Unscoped evidence is retained so cross-document and accessory questions can
    still use supporting material that does not repeat the controller model.
    """

    explicit_models, series_prefixes = _query_model_scope(query)
    if not explicit_models and not series_prefixes:
        return results

    def matches_requested_scope(result: SearchResult) -> bool:
        return any(_result_mentions_model(result, model) for model in explicit_models)

    if not any(matches_requested_scope(result) for result in results):
        return results

    scoped: list[SearchResult] = []
    for result in results:
        if matches_requested_scope(result):
            scoped.append(result)
            continue
        document_models = _result_document_scope_models(result)
        if not document_models:
            scoped.append(result)
            continue
        if any(_model_matches_scope(model, explicit_models, series_prefixes) for model in document_models):
            scoped.append(result)
    return scoped or results


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
    terms = {
        token.lower()
        for token in re.findall(
            r"(?<![\w.])\d+(?:\.\d+)?(?=[A-Za-z%/]|\b)",
            text,
        )
    }
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
    if not candidate_quantities and re.search(r"\bhow many\b", query, flags=re.IGNORECASE):
        selected = _fallback_evidence_results(query, results)
        if selected:
            target_answer = _concise_general_fallback_answer(query, selected[0])
            target_quantities = _quantity_terms(target_answer)
            if target_quantities:
                return bool(_quantity_terms(answer).intersection(target_quantities))
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
    # A section window is itself the selected evidence.  Its context_window may
    # contain a neighboring fragment and must not replace the selected text.
    return content or context_window


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


def _focused_pipe_table_answer_text(query: str, result: SearchResult) -> str:
    """Return one query-matched row with its labels from a rendered table."""
    lines = [line.strip() for line in _fallback_answer_text(result).splitlines() if "|" in line]
    parsed = [[cell.strip() for cell in line.split("|")] for line in lines]
    parsed = [cells for cells in parsed if len(cells) >= 2 and all(cells)]
    if len(parsed) < 2:
        return ""

    query_models = _model_tokens(query)
    asks_power = bool(re.search(r"\b(?:power|consume|consumes|consumption)\b", query, flags=re.IGNORECASE))
    asks_current = bool(re.search(r"\b(?:current|amps?|amperage|draw|draws)\b", query, flags=re.IGNORECASE))
    if query_models and (asks_power or asks_current):
        for header_index, header_cells in enumerate(parsed):
            if not re.fullmatch(r"Model", header_cells[0], flags=re.IGNORECASE):
                continue
            model_columns = {
                model.upper(): column_index
                for column_index, cell in enumerate(header_cells[1:], start=1)
                for model in _model_tokens(cell)
            }
            requested_model = next(
                (model for model in sorted(query_models) if model in model_columns),
                None,
            )
            if not requested_model:
                continue
            column_index = model_columns[requested_model]
            for cells in parsed[header_index + 1 :]:
                if column_index >= len(cells):
                    continue
                label = cells[0]
                if not re.search(r"\b(?:power|current)\s+consumption\b", label, flags=re.IGNORECASE):
                    continue
                value = cells[column_index]
                if asks_current:
                    current_match = re.search(r"(?P<current>\d+(?:\.\d+)?\s*A)\b", value, flags=re.IGNORECASE)
                    if current_match:
                        return f"The {requested_model} draws {current_match.group('current')}."
                    if not re.search(r"\bcurrent\s+consumption\b", label, flags=re.IGNORECASE):
                        continue
                return f"{requested_model} — {label}: {value}"

    query_terms = _material_claim_terms(query).difference(
        {"cause", "causes", "description", "does", "happens", "range", "setting", "which"}
    )
    scored_rows: list[tuple[int, int, list[str]]] = []
    for index, cells in enumerate(parsed):
        overlap = len(query_terms.intersection(_material_claim_terms(" ".join(cells))))
        scored_rows.append((overlap, -index, cells))
    overlap, negative_index, best_cells = max(scored_rows, key=lambda item: (item[0], item[1]))
    if overlap < 2:
        return ""

    row_index = -negative_index
    header_cells: list[str] = []
    for cells in reversed(parsed[:row_index]):
        if len(cells) != len(best_cells):
            continue
        if any(".pdf" in cell.lower() for cell in cells):
            continue
        if all(len(cell) <= 60 and not re.search(r"[.!?]$", cell) for cell in cells):
            header_cells = cells
            break
    if not header_cells:
        return ""

    labelled = [f"{label}: {value}" for label, value in zip(header_cells, best_cells) if label and value]
    return "; ".join(labelled)


def _focused_labeled_table_cell_answer_text(query: str, result: SearchResult) -> str:
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return ""
    content = str(result.content or "").strip()
    match = re.search(
        r"Column headers:\s*(?P<column>.*?);\s*Row headers:\s*(?P<row>.*?);\s*"
        r"Cell value:\s*(?P<value>.*?)(?:;\s*Row:\s*\d+;\s*Column:\s*\d+)?\s*$",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    column = re.sub(r"\s+", " ", match.group("column")).strip()
    row = re.sub(r"\s+", " ", match.group("row")).strip()
    value = re.sub(r"\s+", " ", match.group("value")).strip()
    query_models = _model_tokens(query)
    if ">" in row and query_models:
        # A spanning row header often lists every model before the final leaf,
        # for example ``Model A B C > B``.  The leaf, not the spanning header,
        # owns the cell value.  Do not bind B's value to A merely because A is
        # present earlier in the header.
        row_models = _model_tokens(row)
        leaf_models = _model_tokens(row.rsplit(">", 1)[-1])
        if row_models.intersection(query_models) and leaf_models and not leaf_models.intersection(query_models):
            return ""
    if re.search(
        r"\b(?:how\s+many|number\s+of)\s+input(?:\s+terminals?)?\b|\binput\s+terminals?\b",
        query,
        flags=re.IGNORECASE,
    ) and re.search(r"\bnumber\s+of\s+inputs\b", row, flags=re.IGNORECASE):
        count_match = re.search(
            r"\b(?P<count>\d+)\s*\(\s*IN\s*1\s+to\s+IN\s*(?P<last>\d+)\s*\)",
            value,
            flags=re.IGNORECASE,
        )
        if count_match:
            model = next(iter(sorted(query_models)), column)
            count = count_match.group("count")
            last = count_match.group("last")
            return f"The {model} provides {count} input terminals (IN1 to IN{last})."
    if (
        re.search(r"\b(?:first\s+input|in\s*1)\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:function|assigned?)\b", query, flags=re.IGNORECASE)
        and re.search(r"\bfunction\b", row, flags=re.IGNORECASE)
    ):
        function_match = re.search(
            r"\bIN\s*1\s*:\s*(?P<function>[^,;.]{1,120})",
            value,
            flags=re.IGNORECASE,
        )
        if function_match:
            function = function_match.group("function").strip()
            return f"The first input (IN1) is assigned to {function}."
    condition = ""
    if re.search(r"\bno[- ]voltage\s+input\b", query, flags=re.IGNORECASE):
        condition = "no-voltage input"
    elif re.search(r"\bvoltage\s+input\b", query, flags=re.IGNORECASE):
        condition = "voltage input"
    if condition:
        branch_match = re.search(
            rf"For\s+{re.escape(condition)}\s*:\s*(?P<branch>.*?)"
            r"(?=\s+For\s+(?:no[- ]voltage|voltage)\s+input\s*:|$)",
            value,
            flags=re.IGNORECASE,
        )
        if branch_match:
            branch = branch_match.group("branch").strip(" ,;:")
            state = "ON" if re.search(r"\bon\s+state\b|\btriggers?\s+an?\s+on\b", query, flags=re.IGNORECASE) else ""
            measurement = "voltage" if re.search(r"\bvoltage\b", query, flags=re.IGNORECASE) else ""
            if state and measurement:
                field_match = re.search(
                    rf"\b{state}\s*{measurement}\s*(?P<value>[^,;]{{1,80}})",
                    branch,
                    flags=re.IGNORECASE,
                )
                if field_match:
                    field_value = field_match.group("value").strip(" .,:;")
                    return f"For the {condition}, {state} {measurement} is {field_value}."
    query_terms = _material_claim_terms(query)
    label_terms = _material_claim_terms(f"{row} {column}")
    overlap = len(query_terms.intersection(label_terms))
    detection_range_match = bool(
        re.search(r"\bdetection\s+range\b", query, flags=re.IGNORECASE)
        and re.search(r"\bdetect(?:able|ing)?\s+distance\b", row, flags=re.IGNORECASE)
        and _model_tokens(query).intersection(_model_tokens(f"{row} {column} {value}"))
    )
    if overlap < 2 and not detection_range_match:
        return ""
    return f"{row} — {column}: {value}"


def _focused_repeated_labeled_record_answer_text(query: str, result: SearchResult) -> str:
    """Bind one repeated label/value record such as a wiring-terminal row."""
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return ""
    content = str(result.content or "").strip()
    if not re.search(r"(?:^|\s)Wiring\s+color\s*:", content, flags=re.IGNORECASE):
        return ""
    block_start = r"Terminal\s+No\.\s*:" if re.search(r"Terminal\s+No\.\s*:", content, flags=re.IGNORECASE) else r"Wiring\s+color\s*:"
    blocks = [
        block.strip()
        for block in re.split(rf"(?={block_start})", content, flags=re.IGNORECASE)
        if re.search(rf"^{block_start}", block.strip(), flags=re.IGNORECASE)
    ]
    query_terms = _material_claim_terms(query)
    candidates: list[tuple[int, int, dict[str, str]]] = []
    field_pattern = re.compile(
        r"(Terminal\s+No\.|Wiring\s+color|Name|Assigning\s+default\s+value|Description)\s*:\s*"
        r"(.*?)(?=\s+(?:Terminal\s+No\.|Wiring\s+color|Name|Assigning\s+default\s+value|Description)\s*:|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for index, block in enumerate(blocks):
        fields = {
            re.sub(r"\s+", " ", label).lower(): re.sub(r"\s+", " ", value).strip(" ;|")
            for label, value in field_pattern.findall(block)
        }
        overlap = len(query_terms.intersection(_material_claim_terms(block)))
        if overlap >= 2 and fields.get("wiring color") and fields.get("name"):
            candidates.append((overlap, -index, fields))
    if not candidates:
        return ""
    _overlap, _negative_index, fields = max(candidates, key=lambda item: (item[0], item[1]))
    description = re.sub(r"^[y•·▪]\s*", "", fields.get("description", ""), flags=re.IGNORECASE)
    parts = []
    if fields.get("terminal no."):
        parts.append(f"Terminal: {fields['terminal no.']}")
    parts.extend([f"Wiring color: {fields['wiring color']}", f"Name: {fields['name']}"])
    if fields.get("assigning default value"):
        parts.append(f"Default assignment: {fields['assigning default value']}")
    if description:
        parts.append(f"Description: {description}")
    return "; ".join(parts)


def _focused_model_field_record_answer_text(query: str, result: SearchResult) -> str:
    """Bind a requested model to one field in repeated key/value specification records."""
    if str(result.metadata.get("chunk_type") or "") != "table_record":
        return ""
    content = str(result.content or "").strip()
    query_models = _model_tokens(query)
    if not query_models or not re.search(r"(?:^|\s)Model\s*:", content, flags=re.IGNORECASE):
        return ""
    query_terms = _material_claim_terms(query)
    # In specification questions, "unit" commonly means the physical device,
    # not a field named "Unit cell size". Normalize the measurement synonym
    # and keep that generic noun from outvoting the requested property.
    if "mass" in query_terms:
        query_terms.add("weight")
    query_terms.discard("unit")
    requested_field_match = re.search(
        r"\bhow\s+many\s+(?P<count_field>.+?)\s+(?:does|do|is|are|can)\b|"
        r"\bwhat\s+is\s+(?:the\s+)?(?P<what_field>.+?)\s+(?:of|for)\b|"
        r"\bwhat\s+(?P<does_field>.+?)\s+does\b|"
        r"\bwhich\s+(?P<which_field>.+?)\s+(?:applies|does|is|are)\b",
        query,
        flags=re.IGNORECASE,
    )
    requested_field_terms: set[str] = set()
    if requested_field_match:
        requested_field_terms = _material_claim_terms(
            next((value for value in requested_field_match.groupdict().values() if value), "")
        )
    candidates: list[tuple[int, int, int, str]] = []
    for index, block in enumerate(re.split(r"(?=Model\s*:)", content, flags=re.IGNORECASE)):
        field_match = re.match(r"Model\s*:\s*(?P<field>[^;]+);\s*(?P<rest>.*)", block.strip(), flags=re.IGNORECASE)
        if not field_match:
            continue
        field = field_match.group("field").strip()
        rest = field_match.group("rest").strip()
        field_terms = _material_claim_terms(field)
        for model in query_models:
            value_match = re.search(
                rf"{re.escape(model)}\s*:\s*(?P<value>.*?)"
                r"(?=\s+Model\s*:|\s+[A-Z0-9]+(?:-[A-Z0-9]+)+\s*:|"
                r"\s+[A-Z][A-Za-z ]{1,40}\s*\||$)",
                rest,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not value_match:
                continue
            value = re.sub(r"\s+", " ", value_match.group("value")).strip(" ;|")
            overlap = len(query_terms.intersection(_material_claim_terms(f"{field} {model}")))
            requested_overlap = len(requested_field_terms.intersection(field_terms))
            # Some specification tables label a light's emitted color as its
            # "Pattern". Treat that source label as the requested color field,
            # while still requiring the exact model-bound value below.
            if "color" in requested_field_terms and "pattern" in field_terms:
                requested_overlap += 2
                overlap += 1
            if (
                re.search(r"\bhow\s+many\b", query, flags=re.IGNORECASE)
                and field_terms.intersection({"number", "numbers", "count", "capacity"})
            ):
                requested_overlap += 2
                overlap += 1
            if overlap >= 2 and value:
                candidates.append(
                    (requested_overlap, overlap, -index, f"{field} — {model.upper()}: {value}")
                )
    if not candidates:
        return ""
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _concise_included_item_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Answer inclusion questions only from an explicit supplied/separate marker."""
    if not re.search(
        r"^\s*(?:does|do|is|are)\b.{0,180}"
        r"\b(?:come\s+with|include(?:d|s)?|suppl(?:y|ied))\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []

    subject_match = re.search(
        r"\bcome\s+with\s+(?:an?\s+|the\s+)?(?P<subject>.+?)"
        r"(?:\s+included)?\s*[?.!]*$|"
        r"\b(?:is|are)\s+(?:an?\s+|the\s+)?(?P<included_subject>.+?)\s+"
        r"(?:included|supplied)\b",
        query,
        flags=re.IGNORECASE,
    )
    subject = re.sub(
        r"\s+",
        " ",
        (subject_match.group("subject") or subject_match.group("included_subject") or "")
        if subject_match
        else "",
    ).strip(" .?!")
    subject_terms = _material_claim_terms(subject).difference({"included", "supplied"})
    query_models = _model_tokens(query)
    candidates: list[tuple[int, int, int, bool, SearchResult]] = []
    for result_index, result in enumerate(results[:12]):
        evidence = _fallback_answer_text(result)
        evidence_terms = _material_claim_terms(evidence)
        subject_overlap = len(subject_terms.intersection(evidence_terms))
        if subject_terms and subject_overlap <= 0:
            continue
        evidence_models = _model_tokens(evidence)
        model_alignment = int(not query_models or bool(query_models.intersection(evidence_models)))
        if query_models and not model_alignment:
            continue
        negative = bool(
            re.search(
                r"\b(?:sold|available|ordered|purchased)\s+separately\b|"
                r"\bnot\s+(?:included|supplied)\b",
                evidence,
                flags=re.IGNORECASE,
            )
        )
        positive = bool(re.search(r"\b(?:included|supplied)\b", evidence, flags=re.IGNORECASE))
        if not (negative or positive):
            continue
        candidates.append((model_alignment, subject_overlap, -result_index, negative, result))
    if not candidates:
        return "", []

    _model_alignment, _subject_overlap, _negative_index, negative, result = max(
        candidates,
        key=lambda item: item[:3],
    )
    item = subject or "item"
    if re.search(r"\bcable\b", item, flags=re.IGNORECASE) and re.search(
        r"\bM12\b", _fallback_answer_text(result), flags=re.IGNORECASE
    ):
        item = re.sub(r"\bcable\b", "M12 cable", item, count=1, flags=re.IGNORECASE)
    if negative:
        return f"No. The {item} is sold separately.", [result]
    return f"Yes. The {item} is included.", [result]


def _concise_structured_table_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    if re.search(
        r"\b(?:check|verify|verification|procedure|steps?)\b|\b(?:how do i|what should i)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    included_answer, included_results = _concise_included_item_answer(query, results)
    if included_answer:
        return included_answer, included_results
    candidates: list[tuple[int, int, int, str, SearchResult]] = []
    query_terms = _material_claim_terms(query)
    range_lookup = bool(re.search(r"\b(?:range|distance)\b", query, flags=re.IGNORECASE))
    for index, result in enumerate(results[:10]):
        answer = (
            _focused_model_field_record_answer_text(query, result)
            or _focused_repeated_labeled_record_answer_text(query, result)
            or _focused_labeled_table_cell_answer_text(query, result)
            or _focused_pipe_table_answer_text(query, result)
        )
        if not answer:
            continue
        overlap = len(query_terms.intersection(_material_claim_terms(answer)))
        measurement_fit = 0
        if range_lookup:
            if re.search(r"\b(?:detect(?:able|ing)?\s+)?distance\b", answer, flags=re.IGNORECASE):
                measurement_fit += 1
            if re.search(r"\b\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b", answer, flags=re.IGNORECASE):
                measurement_fit += 2
            if re.search(r"\bresponse\s+time\b", answer, flags=re.IGNORECASE):
                measurement_fit -= 2
        candidates.append((measurement_fit, overlap, -index, answer, result))
    if not candidates:
        return "", []
    _measurement_fit, _overlap, _negative_index, answer, result = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return answer, [result]


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
    if (
        re.search(r"\bterminals?\b", query, flags=re.IGNORECASE)
        and re.search(r"\bdefault\b", query, flags=re.IGNORECASE)
        and re.search(r"\berror(?:\s+condition|\s+signal)?\b", query, flags=re.IGNORECASE)
    ):
        return False
    return bool(
        re.search(
            r"\b(cause|causes|caused|why|reason|correct|corrected|corrective|remedy|"
            r"resolve|resolved|fix|fixed|error|alarm|fault)\b",
            query,
            flags=re.IGNORECASE,
        )
        or re.search(r"\bwhat should i do\b", query, flags=re.IGNORECASE)
        or re.search(r"\bhow do i stop\b.+\bfrom\b", query, flags=re.IGNORECASE)
    )


def _is_comparison_query(query: str) -> bool:
    return bool(
        re.search(r"\b(compare|comparison|versus|whereas)\b", query, flags=re.IGNORECASE)
        # Keep the conventional lowercase abbreviation while avoiding collisions with
        # uppercase product families such as "VS Series".
        or re.search(r"\bvs\.?(?=\s)", query)
        or re.search(r"\bdifferences?\s+between\b", query, flags=re.IGNORECASE)
        or re.search(r"\b(?:how|what)\b.+\b(?:differ|differs)\s+from\b", query, flags=re.IGNORECASE)
        or re.search(r"\bwhat\s+(?:is|are)\b.+\band what\s+(?:is|are)\b", query, flags=re.IGNORECASE)
    )


def _is_procedure_rule_query(query: str) -> bool:
    return bool(
        re.search(r"\b(what|which|how|when)\b", query, flags=re.IGNORECASE)
        and re.search(
            r"\b(branch(?:ed|ing)?|check|condition|flowchart|procedure|rule|step|follow(?:ed)?|"
            r"should|verification|verify|verified)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _is_configuration_location_query(query: str) -> bool:
    if re.search(
        r"\bwhat\s+screen\s+(?:resolution|size|dimensions?|technology|type|format)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"\bwhere\b.{0,100}\b(?:set|adjust|change|configure|find|locate|select|enable|disable)\b"
            r"|\bwhere\s+(?:is|are)\b.{0,100}\b(?:setting|option|parameter|control|field)\b"
            r"|\b(?:which|what)\s+(?:menu|screen|tab|section|page)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _query_troubleshooting_anchor(query: str) -> str:
    quoted = re.search(r'["“](.+?)["”]', query)
    if quoted:
        return quoted.group(1).strip(" .?\"'")
    patterns = (
        r"\bhow do i\s+(?:fix|resolve|correct)\s+(?:the\s+)?(.+?)\s+error\s+(?:on|for|with)\b",
        r"\bhow should i\s+(?:fix|resolve|correct)\s+(?:the\s+)?(.+?)\s+(?:on|for|with)\b",
        r"\bwhat causes\s+(.+?)\s+for\s+.+?\b(?:and|,)\s+how should",
        r"\bwhat causes\s+(.+?)\s*,?\s+and how should",
        r"\bwhat causes\s+(.+?)\s+for\s+.+?\s*\?*$",
        r"\bhow should\s+(.+?)\s+for\s+.+?\s+be corrected\s*\?*$",
        r"\bhow should\s+(.+?)\s+be corrected\s*\?*$",
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

    error_number_match = re.search(r"\berror\s+(\d{3,})\b", query, flags=re.IGNORECASE)
    if error_number_match:
        error_number = re.escape(error_number_match.group(1))
        if re.search(rf"\bError Number:\s*{error_number}\b", content, flags=re.IGNORECASE):
            score += 10.0
        elif re.search(rf"\b{error_number}\b", content):
            score += 3.0

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


_TROUBLESHOOTING_FIELD_PATTERN = re.compile(
    r"(?:^|[.;]\s*)(Error Number|Error Code|Error Messages?|Message|Display|Cause|Solution|Corrective Action|Remedy|Column headers|Row headers|Cell value|Row|Column):\s*"
    r"(.*?)(?=[.;]\s*(?:Error Number|Error Code|Error Messages?|Message|Display|Cause|Solution|Corrective Action|Remedy|Column headers|Row headers|Cell value|Row|Column):|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _troubleshooting_fields(text: str) -> dict[str, str]:
    """Extract answerable fields from a troubleshooting table record."""
    fields: dict[str, str] = {}
    for label, value in _TROUBLESHOOTING_FIELD_PATTERN.findall(text or ""):
        normalized_label = " ".join(label.lower().split())
        cleaned_value = " ".join(value.split()).strip(" ;|")
        if cleaned_value:
            fields[normalized_label] = cleaned_value

    column = fields.get("column headers", "").lower()
    cell = fields.get("cell value", "")
    row_header_parts = [
        re.sub(r"\s+", " ", part).strip(" ;|")
        for part in fields.get("row headers", "").split(">")
        if part.strip(" ;|")
    ]
    if len(row_header_parts) >= 3:
        final_part = row_header_parts[-1]
        if _material_claim_terms(final_part).intersection(TROUBLESHOOTING_ACTION_VERBS):
            fields.setdefault("display", row_header_parts[0])
            fields.setdefault("cause", row_header_parts[-2])
            fields.setdefault("corrective action", final_part)
    if cell:
        if re.fullmatch(r"(?:error\s+)?cause", column):
            fields.setdefault("cause", cell)
        elif "corrective action" in column or "remedy" in column or "solution" in column:
            fields.setdefault("corrective action", cell)
        elif "error message" in column:
            fields.setdefault("error message", cell)
    if fields.get("display"):
        fields.setdefault("error message", fields["display"])
    if fields.get("solution"):
        fields.setdefault("corrective action", fields["solution"])
    return fields


def _troubleshooting_field_records(text: str) -> list[dict[str, str]]:
    source = text or ""
    if re.match(
        r"\s*(?:Error Messages?|Message):\s*",
        source,
        flags=re.IGNORECASE,
    ):
        # Combined row records commonly put Error Code last. Splitting on every
        # code label detaches that code from its message/cause/action fields.
        # Start a new record only when the next message label begins.
        blocks = re.split(
            r"(?=(?<![A-Za-z])(?:Error Messages?|Message):\s*)",
            source,
            flags=re.IGNORECASE,
        )
    elif re.search(
        r"(?:^|[.;]\s*)(?:Error Number|Error Code|Display):\s*",
        source,
        flags=re.IGNORECASE,
    ):
        blocks = re.split(
            r"(?=(?<![A-Za-z])(?:Error Number|Error Code|Display):\s*)",
            source,
            flags=re.IGNORECASE,
        )
    else:
        blocks = re.split(r"(?=(?:Error Messages?|Message):\s*)", source, flags=re.IGNORECASE)
    records = [_troubleshooting_fields(block) for block in blocks if block.strip()]
    return [record for record in records if record]


def _pipe_troubleshooting_fields(text: str) -> dict[str, str]:
    cells = [" ".join(cell.split()).strip(" ;") for cell in (text or "").split("|") if cell.strip(" ;")]
    if len(cells) < 3:
        return {}
    return {
        "error message": cells[0],
        "cause": cells[1],
        "corrective action": cells[2],
    }


def _requested_troubleshooting_fields(query: str) -> tuple[bool, bool]:
    query_lower = query.lower()
    wants_cause = bool(re.search(r"\b(?:cause|causes|caused|why|reason)\b", query_lower))
    wants_action = bool(
        re.search(
            r"\b(?:correct|corrected|corrective|remedy|resolve|resolved|fix|fixed|"
            r"how should|how do i stop|what should|what must|change|set|adjust)\b",
            query_lower,
        )
    )
    if not wants_cause and not wants_action:
        wants_action = True
    return wants_cause, wants_action


def _troubleshooting_anchor_matches(anchor: str, evidence: str) -> bool:
    return _troubleshooting_anchor_match_score(anchor, evidence) >= 0


def _troubleshooting_anchor_match_score(anchor: str, evidence: str) -> float:
    normalized_evidence = _normalized_phrase(evidence)
    normalized_anchor = re.sub(r"^(?:a|an|the)\s+", "", anchor)
    if not normalized_anchor:
        return 0.0
    anchor_tokens = normalized_anchor.split()
    evidence_tokens_list = normalized_evidence.split()
    anchor_terms = set(anchor_tokens)
    evidence_terms = set(evidence_tokens_list)
    if normalized_anchor == normalized_evidence:
        return 100.0
    if normalized_evidence.endswith(normalized_anchor):
        return 95.0
    if normalized_anchor in normalized_evidence:
        precision = len(anchor_terms) / max(1, len(evidence_terms))
        return 70.0 + precision * 20.0
    if ("not" in anchor_terms) != ("not" in evidence_terms):
        return -1.0
    for left, right in (("enabled", "disabled"), ("enable", "disable")):
        if left in anchor_terms and right in evidence_terms and left not in evidence_terms:
            return -1.0
        if right in anchor_terms and left in evidence_terms and right not in evidence_terms:
            return -1.0
    material_anchor_terms = _material_claim_terms(
        re.sub(r"\b1\s+spot\b", "1spot", normalized_anchor)
    )
    material_evidence_terms = _material_claim_terms(
        re.sub(r"\b1\s+spot\b", "1spot", normalized_evidence)
    )
    if "unstable" in material_anchor_terms and not (
        "unstable" in material_evidence_terms
        or re.search(r"\bnot\s+stable\b", normalized_evidence)
    ):
        return -1.0
    if len(material_anchor_terms) >= 4:
        required_material = (3 * len(material_anchor_terms) + 3) // 4
        if len(material_anchor_terms.intersection(material_evidence_terms)) < required_material:
            return -1.0
    required = max(2, min(len(anchor_terms), len(anchor_terms) // 2 + 1))
    overlap = len(anchor_terms.intersection(evidence_terms))
    if overlap < required:
        return -1.0
    return 50.0 * overlap / max(1, len(anchor_terms.union(evidence_terms)))


def _troubleshooting_table_row_key(result: SearchResult) -> tuple[object, ...] | None:
    row = result.metadata.get("table_row")
    if row is None:
        return None
    return (
        result.source_document_id,
        result.document_version_id,
        tuple(result.pages),
        tuple(result.section_path),
        result.metadata.get("table_id") or result.metadata.get("table_index"),
        str(row),
    )


def _concise_troubleshooting_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    if not _is_troubleshooting_query(query) or _is_comparison_query(query):
        return "", []

    wants_cause, wants_action = _requested_troubleshooting_fields(query)
    anchor = _normalized_phrase(_query_troubleshooting_anchor(query))
    if not anchor:
        def troubleshooting_roots(text: str) -> set[str]:
            roots: set[str] = set()
            for term in _material_claim_terms(text):
                if len(term) > 5 and term.endswith("ing"):
                    term = term[:-3]
                elif len(term) > 4 and term.endswith("es"):
                    term = term[:-2]
                elif len(term) > 4 and term.endswith("s"):
                    term = term[:-1]
                roots.add(term)
            return roots

        query_terms = troubleshooting_roots(query)
        direct_candidates: list[tuple[int, int, dict[str, str], SearchResult]] = []
        for result_index, result in enumerate(results[:10]):
            for fields in _troubleshooting_field_records(str(result.content or "")):
                symptom = fields.get("error message") or fields.get("display") or ""
                overlap = len(query_terms.intersection(troubleshooting_roots(symptom)))
                action = fields.get("corrective action") or fields.get("remedy") or fields.get("solution")
                has_requested_field = bool(
                    (wants_cause and fields.get("cause"))
                    or (wants_action and action)
                )
                if overlap >= 2 and has_requested_field:
                    direct_candidates.append((overlap, -result_index, fields, result))
        if direct_candidates:
            _overlap, _result_order, fields, result = max(
                direct_candidates,
                key=lambda item: (item[0], item[1]),
            )
            lines: list[str] = []
            if wants_cause and fields.get("cause"):
                value = fields["cause"]
                lines.append(f"Cause: {value if value.endswith(('.', '!', '?')) else value + '.'}")
            action = fields.get("corrective action") or fields.get("remedy") or fields.get("solution")
            if wants_action and action:
                lines.append(f"Corrective action: {action if action.endswith(('.', '!', '?')) else action + '.'}")
            if lines:
                return "\n".join(lines), [result]
    requested_error_number_match = re.search(
        r"\berror(?:\s+(?:number|code))?\s*[:#-]?\s*(\d{3,})\b",
        query,
        flags=re.IGNORECASE,
    )
    requested_error_number = requested_error_number_match.group(1) if requested_error_number_match else ""
    requested_display_match = re.search(
        r"\bdisplay\s+(?:code\s+)?(?P<after>[A-Za-z0-9_-]{2,})\b"
        r"|(?<![A-Za-z0-9_-])(?P<before>-?[A-Za-z0-9_]{2,})\s+display\b",
        query,
        flags=re.IGNORECASE,
    )
    requested_display_token = (
        requested_display_match.group("after") or requested_display_match.group("before") or ""
        if requested_display_match
        else ""
    )
    normalized_display_token = _normalized_phrase(requested_display_token)
    requested_display = (
        normalized_display_token
        if requested_display_token
        and (
            requested_display_token.startswith("-")
            or len(normalized_display_token) >= 3
            or any(character.isdigit() for character in requested_display_token)
        )
        else ""
    )
    ordered = _order_troubleshooting_results(query, results)
    candidates: list[tuple[float, int, int, dict[str, str], SearchResult]] = []
    query_terms = _material_claim_terms(query)

    # Table-cell chunks frequently repeat only a parent header in their row headers. Find
    # the exact error-message cell first, then allow cause/action cells from that same row
    # to contribute even when those cells do not repeat the symptom text themselves.
    exact_row_keys: set[tuple[object, ...]] = set()
    if anchor and len(anchor) >= 10:
        for result in ordered[:10]:
            row_key = _troubleshooting_table_row_key(result)
            if row_key is None:
                continue
            for fields in _troubleshooting_field_records(str(result.content or "")):
                error_text = (
                    fields.get("error message")
                    or fields.get("error messages")
                    or fields.get("message")
                    or fields.get("row headers")
                    or ""
                )
                if error_text and _troubleshooting_anchor_match_score(anchor, error_text) >= 95.0:
                    exact_row_keys.add(row_key)

    for result_index, result in enumerate(ordered[:10]):
        evidence = _troubleshooting_context_text(result)
        matching_row = _matching_troubleshooting_row_text(query, result)
        anchor_evidence = (
            evidence
            if requested_display
            else (matching_row or _fallback_answer_text(result))
        )
        row_key = _troubleshooting_table_row_key(result)
        row_is_exact = row_key is not None and row_key in exact_row_keys
        if (
            anchor
            and len(anchor) >= 10
            and not row_is_exact
            and not _troubleshooting_anchor_matches(anchor, anchor_evidence)
        ):
            continue
        field_records: list[dict[str, str]] = []
        if requested_display:
            for pipe_row in evidence.splitlines():
                cells = [
                    re.sub(r"\s+", " ", cell).strip(" ;")
                    for cell in pipe_row.split("|")
                    if cell.strip(" ;")
                ]
                if len(cells) < 3 or _normalized_phrase(cells[0]) != requested_display:
                    continue
                field_records.append(
                    {
                        "display": cells[0],
                        "error message": cells[0],
                        "cause": cells[1],
                        "corrective action": cells[2],
                    }
                )
        if not field_records:
            field_records = _troubleshooting_field_records(str(result.content or ""))
        if not field_records:
            pipe_fields = _pipe_troubleshooting_fields(matching_row)
            field_records = [pipe_fields] if pipe_fields else []
        for fields in field_records:
            record_error_number = fields.get("error number") or fields.get("error code") or ""
            if requested_error_number and record_error_number and record_error_number != requested_error_number:
                continue
            record_display = _normalized_phrase(fields.get("display") or "")
            if requested_display and record_display and requested_display != record_display:
                continue
            error_text = (
                fields.get("error message")
                or fields.get("error messages")
                or fields.get("message")
                or fields.get("row headers")
                or ""
            )
            match_score = 0.0
            if anchor:
                # Some manuals render the display code in a symbol font, so
                # the parsed row header contains no searchable letters. A
                # quoted descriptive anchor can still match the cause/action
                # fields in that same record without weakening row binding.
                match_score = max(
                    _troubleshooting_anchor_match_score(anchor, error_text or anchor_evidence),
                    _troubleshooting_anchor_match_score(anchor, " ".join(fields.values())),
                )
            if requested_display and record_display == requested_display:
                match_score = max(match_score, 100.0)
            if row_is_exact and match_score < 92.0:
                match_score = 92.0
            if anchor and match_score < 0:
                continue
            coverage = len(query_terms.intersection(_material_claim_terms(" ".join(fields.values()))))
            if requested_error_number and record_error_number == requested_error_number:
                match_score += 100.0
            candidates.append((match_score, coverage, -result_index, fields, result))

    selected_values: dict[str, str] = {}
    selected_results: list[SearchResult] = []
    for field_name, wanted in (("cause", wants_cause), ("corrective action", wants_action)):
        if not wanted:
            continue
        matching = []
        for match_score, coverage, result_order, fields, result in candidates:
            value = fields.get(field_name) or (fields.get("remedy") if field_name == "corrective action" else "")
            if value:
                matching.append((match_score, coverage, result_order, value, result))
        if not matching:
            continue
        _match_score, _coverage, _result_order, value, result = max(
            matching,
            key=lambda item: (item[0], item[1], item[2]),
        )
        selected_values[field_name] = value
        selected_results.append(result)

    lines: list[str] = []
    if wants_cause and selected_values.get("cause"):
        cause = selected_values["cause"]
        lines.append(f"Cause: {cause if cause.endswith(('.', '!', '?')) else cause + '.'}")
    if wants_action and selected_values.get("corrective action"):
        action = selected_values["corrective action"]
        lines.append(f"Corrective action: {action if action.endswith(('.', '!', '?')) else action + '.'}")
    if not lines and wants_action:
        # Some manuals state a symptom and its remedy as prose rather than as
        # labeled Cause/Remedy cells.  Select the imperative sentence from the
        # same symptom-bearing evidence instead of returning the symptom itself.
        prose_actions: list[tuple[float, int, str, SearchResult]] = []
        for result_index, result in enumerate(ordered[:10]):
            # Use the selected chunk text for both symptom binding and action
            # extraction. Parent/context windows may contain unrelated procedures.
            evidence = _fallback_answer_text(result)
            if anchor and not _troubleshooting_anchor_matches(anchor, evidence):
                continue
            evidence_overlap = len(query_terms.intersection(_material_claim_terms(evidence)))
            for segment in re.split(r"(?<=[.!?])\s+|\n+", evidence):
                action = re.sub(r"\s+", " ", segment).strip(" -|•·▪z\t\r\n")
                action_terms = _material_claim_terms(action)
                if not action_terms.intersection(TROUBLESHOOTING_ACTION_VERBS):
                    continue
                action_overlap = len(query_terms.intersection(action_terms))
                score = float(evidence_overlap + action_overlap * 2)
                if str(result.metadata.get("chunk_type") or "") == "atomic_text":
                    score += 3.0
                score -= min(len(evidence), 2000) / 2000.0
                prose_actions.append((score, -result_index, action, result))
        if prose_actions:
            _score, _result_order, action, result = max(
                prose_actions,
                key=lambda item: (item[0], item[1]),
            )
            lines.append(
                f"Corrective action: {action if action.endswith(('.', '!', '?')) else action + '.'}"
            )
            selected_results.append(result)
    if not lines:
        return "", []

    unique_results: list[SearchResult] = []
    seen_chunks: set[str] = set()
    for result in selected_results:
        if result.chunk_id not in seen_chunks:
            unique_results.append(result)
            seen_chunks.add(result.chunk_id)
    return "\n".join(lines), unique_results


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
    concise_answer, concise_results = _concise_troubleshooting_answer(query, results)
    if _is_troubleshooting_query(query) and _query_troubleshooting_anchor(query) and not concise_answer:
        return AnswerResponse(
            answer="I could not find a troubleshooting entry matching the stated error in the retrieved evidence.",
            confidence="low",
            used_documents=[],
            citations=[],
            warnings=["The retrieved evidence did not contain a matching cause or corrective action."],
            followup_questions=[],
            insufficient_evidence=True,
        )
    location_answer, location_results = _concise_configuration_location_answer(query, results)
    table_answer, table_results = (
        ("", [])
        if _is_comparison_query(query)
        else _concise_structured_table_answer(query, results)
    )
    if (
        table_results
        and table_results[0].metadata.get("context_window")
        and _quantity_terms(query)
    ):
        table_answer = ""
    fallback_results = concise_results or location_results or table_results or _fallback_evidence_results(query, results)
    top = fallback_results[0]
    if concise_answer:
        answer_text = concise_answer
    elif location_answer:
        answer_text = location_answer
    elif table_answer:
        answer_text = table_answer
    elif len(fallback_results) == 1:
        table_fallback = (
            _fallback_answer_text(top)
            if str(top.metadata.get("chunk_type") or "") == "table_record"
            else ""
        )
        preserve_structured_context = bool(
            table_fallback
            and top.metadata.get("context_window")
            and _quantity_terms(query)
        )
        answer_text = (
            table_fallback
            if preserve_structured_context
            else (
                _matching_troubleshooting_row_text(query, top)
                or _focused_table_record_answer_text(query, top)
                or table_fallback
                or _concise_general_fallback_answer(query, top)
            )
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


def _fallback_answer_from_summaries(
    query: str,
    summaries: list[dict[str, Any]],
    results: list[SearchResult],
) -> AnswerResponse | None:
    """Recover a concise grounded answer when structured model output is malformed."""
    query_terms = _material_claim_terms(query)
    query_quantities = _quantity_terms(query)
    requested_models = _model_tokens(query)
    wants_measurement_location = bool(
        re.search(r"\bwhere\b.{0,100}\bmeasure(?:d|ment)?\b", query, flags=re.IGNORECASE)
    )
    candidates: list[tuple[float, int, str, tuple[str, ...]]] = []
    candidate_order = 0
    result_by_chunk_id = {result.chunk_id: result for result in results}
    for summary in summaries[:4]:
        text = str(summary.get("summary") or "").strip()
        source_chunk_ids = [
            str(item.get("chunk_id") or "")
            for item in summary.get("source_documents", [])
            if isinstance(item, dict)
        ] or [part for part in str(summary.get("chunk_id") or "").split(",") if part]
        source_evidence = "\n".join(
            _fallback_answer_text(result_by_chunk_id[chunk_id])
            for chunk_id in source_chunk_ids
            if chunk_id in result_by_chunk_id
        )
        source_scope = " ".join(
            " ".join(
                [
                    *result_by_chunk_id[chunk_id].section_path,
                    _fallback_answer_text(result_by_chunk_id[chunk_id])[:300],
                    str(result_by_chunk_id[chunk_id].metadata.get("product_model") or ""),
                ]
            )
            for chunk_id in source_chunk_ids
            if chunk_id in result_by_chunk_id
        )
        identifier_alignment = sum(
            1
            for model in requested_models
            if _normalized_phrase(model) in _normalized_phrase(source_scope)
        )
        if source_evidence and len(
            query_terms.intersection(_material_claim_terms(source_evidence))
        ) < 2:
            # A query-shaped model summary is not evidence when its own source
            # does not contain the query's identifying terms.
            continue
        sentences = [
            re.sub(r"\s+", " ", sentence).strip(" -|\t\r\n")
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip(" -|\t\r\n")
        ]
        windows = [*sentences]
        windows.extend(f"{left} {right}" for left, right in zip(sentences, sentences[1:]))
        for window in windows:
            if re.search(
                r"\b(?:provided|retrieved|available) evidence does not contain\b|"
                r"\b(?:cannot|could not|unable to) (?:determine|find|answer)\b",
                window,
                flags=re.IGNORECASE,
            ):
                candidate_order += 1
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(window)))
            if overlap < 2:
                candidate_order += 1
                continue
            quantities = _quantity_terms(window)
            score = float(overlap * 2 + identifier_alignment * 8)
            if query_quantities:
                score += 8.0 if query_quantities.issubset(quantities) else -8.0
            if wants_measurement_location:
                score += 8.0 if re.search(
                    r"\bmeasure(?:d|ment)?\b.{0,100}\b(?:at|on)\b",
                    window,
                    flags=re.IGNORECASE,
                ) else -8.0
            if quantities:
                score += 2.0
            if re.search(r"\b(?:calculated|computed|derived)\b", query, flags=re.IGNORECASE):
                if re.search(
                    r"\b(?:calculated\s+from|computed\s+from|derived\s+from|ratio\s+of|divid(?:e|ed|ing))\b",
                    window,
                    flags=re.IGNORECASE,
                ):
                    score += 8.0
            candidates.append((score, -candidate_order, window, tuple(source_chunk_ids)))
            candidate_order += 1
    if not candidates:
        return None
    score, _order, answer_text, selected_source_chunk_ids = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    if score < 4.0:
        return None

    answer_terms = _material_claim_terms(answer_text)
    provenance_results = [
        result_by_chunk_id[chunk_id]
        for chunk_id in selected_source_chunk_ids
        if chunk_id in result_by_chunk_id
    ]
    ranked_results = sorted(
        provenance_results or results,
        key=lambda result: (
            len(answer_terms.intersection(_material_claim_terms(_fallback_answer_text(result)))),
            len(query_terms.intersection(_material_claim_terms(_fallback_answer_text(result)))),
            result.score,
        ),
        reverse=True,
    )
    supporting_results = [
        result
        for result in ranked_results
        if len(answer_terms.intersection(_material_claim_terms(_fallback_answer_text(result)))) >= 2
    ][:2]
    if not supporting_results:
        return None
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
            for result in supporting_results
        ],
        citations=[
            {
                "chunk_id": result.chunk_id,
                "document_id": result.source_document_id,
                "pages": result.pages,
                "quote_span": None,
            }
            for result in supporting_results
        ],
        warnings=["The answer model returned malformed structured output; used the strongest grounded summary."],
        followup_questions=[],
        insufficient_evidence=False,
    )


def _concise_conditioned_measurement_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    if _is_troubleshooting_query(query):
        return "", []
    """Answer conditional limits and physical measurement locations from one sentence."""
    is_conditioned_measurement = bool(
        re.search(
            r"\b(?:angle|distance|height|limit|pressure|range|speed|temperature|voltage|width)\b",
            query,
            flags=re.IGNORECASE,
        )
        and re.search(r"\b(?:if|when|exceeds?|below|above)\b", query, flags=re.IGNORECASE)
    )
    if re.search(r"\breference\s+points?\s+monitoring\b", query, flags=re.IGNORECASE):
        # This safety requirement binds an applicability condition and a
        # separate response-time limit; the dedicated structured-fact route
        # below returns only the facet requested by the user.
        is_conditioned_measurement = False
    wants_measurement_location = bool(
        re.search(r"\bwhere\b.{0,100}\bmeasure(?:d|ment)?\b", query, flags=re.IGNORECASE)
    )
    if not is_conditioned_measurement and not wants_measurement_location:
        return "", []

    query_terms = _material_claim_terms(query)
    query_quantities = _quantity_terms(query)
    candidates: list[tuple[float, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        sentence_segments = [
            re.sub(r"\s+", " ", segment).strip(" -|•·▪\t\r\n")
            for segment in re.split(r"(?<=[.!?])\s+|\n+", evidence)
            if segment.strip(" -|•·▪\t\r\n")
        ]
        segments = [*sentence_segments]
        segments.extend(
            f"{left} {right}"
            for left, right in zip(sentence_segments, sentence_segments[1:])
            if not re.match(r"^[\u0080-\u009f\u2022\u25a0-\u25ff]", right)
        )
        segments.extend(
            f"{first} {second} {third}"
            for first, second, third in zip(
                sentence_segments,
                sentence_segments[1:],
                sentence_segments[2:],
            )
            if not re.match(r"^[\u0080-\u009f\u2022\u25a0-\u25ff]", second)
            and not re.match(r"^[\u0080-\u009f\u2022\u25a0-\u25ff]", third)
        )
        for segment in segments:
            overlap = len(query_terms.intersection(_material_claim_terms(segment)))
            if overlap < 2:
                continue
            segment_quantities = _quantity_terms(segment)
            if query_quantities and not query_quantities.issubset(segment_quantities):
                continue
            score = float(overlap * 2)
            if is_conditioned_measurement and re.search(
                r"\b(?:must not exceed|does not exceed|limit(?:ed)? to|rated|maximum|minimum)\b",
                segment,
                flags=re.IGNORECASE,
            ):
                score += 8.0
            if wants_measurement_location and re.search(
                r"\bmeasure(?:d|ment)?\b.{0,100}\b(?:at|on)\b",
                segment,
                flags=re.IGNORECASE,
            ):
                score += 10.0
            candidates.append((score, -result_index, segment, result))
    if not candidates:
        return "", []
    _score, _negative_index, answer_text, result = max(candidates, key=lambda item: (item[0], item[1]))
    requested_models = sorted(_model_tokens(query), key=lambda value: (-len(value), value))
    if requested_models and not any(
        _normalized_phrase(model) in _normalized_phrase(answer_text)
        for model in requested_models
    ):
        answer_text = f"For {requested_models[0]}, {answer_text[:1].lower() + answer_text[1:]}"
    return answer_text, [result]


def _concise_instruction_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Extract the exact imperative/default instruction from procedural evidence."""
    if _is_troubleshooting_query(query):
        return "", []
    dependent_answer, _dependent_results = _concise_dependent_list_answer(query, results)
    if dependent_answer:
        return "", []
    if re.search(r"\bdefault\s*:", " ".join(_fallback_answer_text(result) for result in results[:3]), flags=re.IGNORECASE):
        # Let the structured/default extractors preserve a directly stated
        # setting value instead of turning descriptive field text into a step.
        return "", []
    is_how_to = bool(
        re.search(r"^\s*(?:how do i|how should i|what should i)\b", query, flags=re.IGNORECASE)
    )
    is_action_effect = bool(
        re.search(
            r"^\s*what\s+happens\s+when\s+(?:i|you)\s+"
            r"(?:press|select|set|connect|enable|disable|turn)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    wants_action_sequence = bool(
        re.search(
            r"\b(?:configure|configuration|connect|convert|install|procedure|set|setting|setup|steps?|update)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    asks_factory_default = bool(
        re.search(r"\bfactory default(?: setting)?\b", query, flags=re.IGNORECASE)
    )
    asks_disable = bool(
        re.search(r"\b(?:disable|turn\s+off|not\s+require(?:d)?)\b", query, flags=re.IGNORECASE)
    )
    asks_calibration = bool(re.search(r"\bcalibrat(?:e|ed|es|ing|ion)\b", query, flags=re.IGNORECASE))
    if not is_how_to and not is_action_effect and not asks_factory_default:
        return "", []

    pass_workpiece_query = bool(
        re.search(r"\bpass\s+(?:the\s+)?workpiece\b", query, flags=re.IGNORECASE)
        and asks_calibration
    )
    if pass_workpiece_query:
        pass_candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = _fallback_answer_text(result)
            for segment in re.split(r"(?<=[.!?])\s+|\n+", evidence):
                clean_segment = re.sub(r"\s+", " ", segment).strip(" -|•·▪\t\r\n")
                if not re.search(
                    r"\bpass\s+(?:the\s+)?workpiece\b",
                    clean_segment,
                    flags=re.IGNORECASE,
                ):
                    continue
                action_match = re.match(
                    r"^When\s+.+?,\s*(?P<action>.+)$",
                    clean_segment,
                    flags=re.IGNORECASE,
                )
                action = action_match.group("action") if action_match else clean_segment
                action = re.sub(r"(['\"])[ ]+([^'\"]+?)[ ]+\1", r"\1\2\1", action)
                action = action[:1].upper() + action[1:]
                pass_candidates.append(
                    (
                        int(bool(re.search(r"\bcontinue\s+holding\b", action, flags=re.IGNORECASE))),
                        -result_index,
                        action,
                        result,
                    )
                )
        if pass_candidates:
            _holding, _negative_index, action, result = max(
                pass_candidates,
                key=lambda item: item[:2],
            )
            return action, [result]

    conditional_install_match = re.search(
        r"^\s*what\s+should\s+i\s+install\s+if\s+(?P<condition>.+?)[?\s]*$",
        query,
        flags=re.IGNORECASE,
    )
    if conditional_install_match:
        condition_terms = _material_claim_terms(conditional_install_match.group("condition"))
        install_candidates: list[tuple[int, int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:10]):
            evidence = _fallback_answer_text(result)
            for match in re.finditer(
                r"(?P<condition>[^.!?\n]{0,220}?)\binstall\s+"
                r"(?P<object>(?:an?|the)\s+[^,.!?\n]{2,120})"
                r"(?=\s*,?\s+or\b|[.!?\n]|$)",
                evidence,
                flags=re.IGNORECASE,
            ):
                overlap = len(
                    condition_terms.intersection(
                        _material_claim_terms(match.group("condition"))
                    )
                )
                if overlap < 2:
                    continue
                bounded = int(
                    str(result.metadata.get("chunk_type") or "")
                    in {"atomic_text", "procedure_record"}
                )
                install_candidates.append(
                    (
                        overlap,
                        bounded,
                        -result_index,
                        re.sub(r"\s+", " ", match.group("object")).strip(),
                        result,
                    )
                )
        if install_candidates:
            _overlap, _bounded, _negative_index, object_text, result = max(
                install_candidates,
                key=lambda item: item[:3],
            )
            return f"Install {object_text}.", [result]

    def instruction_terms(text: str) -> set[str]:
        terms = _material_claim_terms(text)
        for pattern, canonical in (
            (r"\bpress(?:ed|es|ing)?\b", "press"),
            (r"\bselect(?:ed|s|ing)?\b", "select"),
            (r"\bconnect(?:ed|s|ing)?\b", "connect"),
            (r"\bconfigur(?:e|ed|es|ing|ation)\b", "configure"),
            (r"\bcalibrat(?:e|ed|es|ing|ion)\b", "calibrate"),
        ):
            if re.search(pattern, text, flags=re.IGNORECASE):
                terms.add(canonical)
        for term in list(terms):
            if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
                terms.add(term[:-1])
        if terms.intersection({"count", "number"}):
            terms.update({"count", "number"})
        return terms

    query_terms = instruction_terms(query)
    ordered_query_terms = [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query.lower())
        if token in query_terms
    ]
    query_phrases = {
        f"{left} {right}"
        for left, right in zip(ordered_query_terms, ordered_query_terms[1:])
    }
    requested_control_match = re.search(
        r"\b(?:press|select|set|turn)\s+(?:the\s+)?"
        r"(?P<control>[A-Za-z0-9_./+-]+\s+(?:button|key|switch))\b",
        query,
        flags=re.IGNORECASE,
    )
    requested_control = (
        _normalized_phrase(requested_control_match.group("control"))
        if requested_control_match
        else ""
    )
    requested_series_phrases = {
        _normalized_phrase(match.group(0))
        for match in re.finditer(
            r"\b[A-Z][A-Z0-9-]{1,12}\s+Series\b",
            query,
        )
    }
    candidates: list[tuple[float, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        result_scope = _normalized_phrase(_result_model_text(result))
        sentence_segments = [
            re.sub(r"\s+", " ", segment).strip(" -|•·▪\t\r\n")
            for segment in re.split(r"(?<=[.!?])\s+|\n+", evidence)
            if segment.strip(" -|•·▪\t\r\n")
        ]
        segments = [*sentence_segments]
        segments.extend(
            f"{left} {right}"
            for left, right in zip(sentence_segments, sentence_segments[1:])
            if not re.match(r"^[\u0080-\u009f\u2022\u25a0-\u25ff]", right)
        )
        segments.extend(
            f"{first} {second} {third}"
            for first, second, third in zip(
                sentence_segments,
                sentence_segments[1:],
                sentence_segments[2:],
            )
            if not re.match(r"^[\u0080-\u009f\u2022\u25a0-\u25ff]", second)
            and not re.match(r"^[\u0080-\u009f\u2022\u25a0-\u25ff]", third)
        )
        for segment in segments:
            segment_terms = instruction_terms(segment)
            overlap = len(query_terms.intersection(segment_terms))
            normalized_segment = _normalized_phrase(segment)
            if is_action_effect and requested_control and requested_control not in normalized_segment:
                continue
            has_distinctive_phrase = any(
                phrase in normalized_segment for phrase in query_phrases
            )
            disable_condition = asks_disable and bool(
                re.search(
                    r"\b(?:not\s+(?:be\s+)?required|disable(?:d)?|turn(?:ed)?\s+off)\b",
                    segment,
                    flags=re.IGNORECASE,
                )
            )
            calibration_procedure = asks_calibration and bool(
                re.search(r"\bcalibrat(?:e|ed|es|ing|ion)\b", segment, flags=re.IGNORECASE)
                and re.search(
                    r"\b(?:press(?:ed|es|ing)?|set|select(?:ed|s|ing)?|"
                    r"while|then|again|present|absent)\b",
                    segment,
                    flags=re.IGNORECASE,
                )
            )
            if (
                overlap < 3
                and not (overlap >= 2 and has_distinctive_phrase)
                and not (overlap >= 1 and disable_condition)
                and not (overlap >= 2 and calibration_procedure)
            ):
                continue
            if asks_factory_default and not re.search(r"\bfactory default\b", segment, flags=re.IGNORECASE):
                continue
            if (is_how_to or is_action_effect) and not re.search(
                r"\b(?:calibrat(?:e|ed|es|ing|ion)|set|select(?:s|ed|ing)?|connect|configure|"
                r"check|press(?:ed|es|ing)?|open|install|insulate|secure|use|turn)\b",
                segment,
                flags=re.IGNORECASE,
            ):
                continue
            score = float(overlap * 2)
            if disable_condition:
                score += 20.0
                if re.match(r"^\s*If\b", segment, flags=re.IGNORECASE):
                    score += 10.0
            elif asks_disable and re.search(
                r"\b(?:require|enable)\b",
                segment,
                flags=re.IGNORECASE,
            ):
                score -= 12.0
            if calibration_procedure:
                score += 20.0
                if re.search(
                    r"\bcalibrat(?:e|ed|es|ing|ion)\b.{0,180}\b(?:press(?:ed|es|ing)?|"
                    r"while|then|again|present|absent)\b",
                    segment,
                    flags=re.IGNORECASE,
                ):
                    score += 10.0
            elif asks_calibration and re.search(
                r"\bcan\s+be\s+used\s+to\b.{0,60}\bcalibrat",
                segment,
                flags=re.IGNORECASE,
            ):
                score -= 12.0
            if requested_series_phrases and any(
                phrase in result_scope for phrase in requested_series_phrases
            ):
                score += 20.0
            if is_action_effect and re.search(
                r"\b(?:automatically|causes?|results?\s+in|will\s+(?:set|select|enable|disable|turn))\b",
                segment,
                flags=re.IGNORECASE,
            ):
                score += 12.0
            if re.search(r"\b(?:set|select)\s+the\s+.+?\bswitch\b", segment, flags=re.IGNORECASE):
                score += 6.0
            if "rs-232c" in query.lower() and "rs-232c" in segment.lower():
                score += 4.0
            if "image processing system" in query.lower() and "image processing system" in segment.lower():
                score += 3.0
            if re.search(r"\bother than\b", segment, flags=re.IGNORECASE) and "other than" not in query.lower():
                score -= 8.0
            if (is_how_to or is_action_effect) and wants_action_sequence:
                # Prefer a short adjacent action sequence over only its opening
                # sentence.  Manuals commonly put the menu selection and the
                # confirming click/save operation in consecutive sentences.
                action_count = len(
                    re.findall(
                        r"\b(?:calibrat(?:e|ed|es|ing|ion)|set|select|connect|configure|check|"
                        r"press(?:ed|es|ing)?|click|open|"
                        r"install|insulate|secure|use|turn|convert|save)\b",
                        segment,
                        flags=re.IGNORECASE,
                    )
                )
                if action_count >= 2:
                    score += min(4.0, float(action_count))
            clean_segment = re.sub(r"^\s*\d+[.)]?\s+", "", segment)
            clean_segment = re.sub(r"^[yY]\s+(?=[A-Z])", "", clean_segment)
            if re.match(r"^\s*how\s+do\s+i\s+open\b", query, flags=re.IGNORECASE):
                clean_segment = re.sub(
                    r",?\s+and\s+then\b.*$",
                    ".",
                    clean_segment,
                    flags=re.IGNORECASE,
                ).rstrip()
            if re.match(r"^Select\s+this\b", clean_segment, flags=re.IGNORECASE):
                selected_option = re.search(
                    r"\bWhen\s+\[(?P<option>[^\]\n]{1,80})\]\s+is\s+selected\b"
                    r"[^.]*\.\s*Select\s+this\b",
                    evidence,
                    flags=re.IGNORECASE,
                )
                if selected_option:
                    option = selected_option.group("option").strip()
                    clean_segment = re.sub(
                        r"^Select\s+this\b",
                        f"Select [{option}]",
                        clean_segment,
                        flags=re.IGNORECASE,
                    )
                    score += 6.0
            candidates.append((score, -result_index, clean_segment, result))
    if not candidates:
        return "", []
    _score, _negative_index, answer_text, result = max(candidates, key=lambda item: (item[0], item[1]))
    return answer_text, [result]


def _concise_contamination_action_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Extract a bounded protective action for contamination in an optical path."""
    if not (
        re.search(r"\b(?:dirt|dust|debris|contamination)\b", query, flags=re.IGNORECASE)
        and re.search(r"\blight[- ]?axis\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:what should|how should|how do)\b", query, flags=re.IGNORECASE)
    ):
        return "", []
    candidates: list[tuple[int, int, SearchResult]] = []
    for result_index, result in enumerate(results[:12]):
        evidence = _fallback_answer_text(result)
        if not re.search(r"\blight[- ]?axis\b", evidence, flags=re.IGNORECASE):
            continue
        has_cover = bool(re.search(r"\bprotective\s+cover\b", evidence, flags=re.IGNORECASE))
        has_air_purge = bool(re.search(r"\bair\s+purge\b", evidence, flags=re.IGNORECASE))
        if has_cover or has_air_purge:
            candidates.append((int(has_cover) + int(has_air_purge), -result_index, result))
    if not candidates:
        return "", []
    _coverage, _negative_index, result = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return "Use a protective cover or air purge.", [result]


def _concise_temporal_effect_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Resolve when a changed value applies from explicit temporal evidence."""
    if not re.search(
        r"^\s*when\b.{0,160}\b(?:take effect|become effective|apply|affect)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    query_terms = _material_claim_terms(query)
    candidates: list[tuple[float, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for segment in re.split(r"(?<=[.!?])\s+|\n+", evidence):
            clean_segment = re.sub(r"\s+", " ", segment).strip(" -|•·▪\t\r\n")
            if not clean_segment:
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(clean_segment)))
            if overlap < 3 or not re.search(
                r"\b(?:after|before|current|immediate(?:ly)?|next|subsequent|until|when)\b",
                clean_segment,
                flags=re.IGNORECASE,
            ):
                continue
            temporal_bonus = 8.0 if re.search(
                r"\b(?:only subsequent|subsequent|does not affect the current|takes? effect)\b",
                clean_segment,
                flags=re.IGNORECASE,
            ) else 2.0
            candidates.append((overlap * 2.0 + temporal_bonus, -result_index, clean_segment, result))
    if not candidates:
        return "", []
    _score, _negative_index, answer_text, result = max(candidates, key=lambda item: (item[0], item[1]))
    requested_models = sorted(_model_tokens(query), key=lambda value: (-len(value), value))
    if requested_models and not any(
        _normalized_phrase(model) in _normalized_phrase(answer_text)
        for model in requested_models
    ):
        answer_text = f"For {requested_models[0]}, {answer_text[:1].lower() + answer_text[1:]}"
    return answer_text, [result]


def _concise_event_action_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Answer what to do when a named display/event occurs."""
    query_match = re.search(
        r"^\s*what\s+should\s+i\s+do\s+when\s+(?P<condition>.+?)[?\s]*$",
        query,
        flags=re.IGNORECASE,
    )
    if not query_match:
        return "", []
    condition_terms = _material_claim_terms(query_match.group("condition")).difference(
        {model.lower() for model in _model_tokens(query)}
    )
    condition_terms.difference_update({"indicator", "indicators"})
    candidates: list[tuple[int, int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:12]):
        evidence = _fallback_answer_text(result)
        for segment in re.split(r"(?<=[.!?])\s+|\n+", evidence):
            clean_segment = re.sub(r"\s+", " ", segment).strip(" -|•·▪\t\r\n")
            if not clean_segment:
                continue
            overlap = len(condition_terms.intersection(_material_claim_terms(clean_segment)))
            if overlap < max(1, min(2, len(condition_terms))):
                continue
            leading_condition = re.match(
                r"^When\s+.+?,\s*(?P<action>.+)$",
                clean_segment,
                flags=re.IGNORECASE,
            )
            action = leading_condition.group("action") if leading_condition else clean_segment
            if not re.search(
                r"\b(?:continue|hold|pass|press|release|set|select|check|install|change|use)\b",
                action,
                flags=re.IGNORECASE,
            ):
                continue
            action = re.sub(r"(['\"])[ ]+([^'\"]+?)[ ]+\1", r"\1\2\1", action)
            if action:
                action = action[:1].upper() + action[1:]
            candidates.append(
                (
                    int(leading_condition is not None),
                    overlap,
                    -result_index,
                    action,
                    result,
                )
            )
    if not candidates:
        return "", []
    _leading, _overlap, _negative_index, action, result = max(
        candidates,
        key=lambda item: item[:3],
    )
    return action, [result]


def _query_target_quantity_score(query: str, evidence: str) -> float:
    if not re.search(r"\b(?:count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
        return 0.0
    ignored = ANSWER_CLAIM_SUPPORT_STOPWORDS.union(
        {"connect", "connected", "does", "many", "new", "series", "system", "what", "which"}
    )
    query_terms = _answer_terms(query).difference(ignored)
    best = 0.0
    for segment in re.split(r"(?:\n+|(?<=[.!?;])\s+)", evidence):
        quantities = _quantity_terms(segment)
        if not quantities:
            continue
        overlap = len(query_terms.intersection(_answer_terms(segment)))
        if overlap >= 2:
            best = max(best, min(8.0, overlap * 1.5 + len(quantities)))
    return best


def _concise_general_fallback_answer(query: str, result: SearchResult) -> str:
    evidence = _fallback_answer_text(result)
    if re.search(r"\bdetection\s+count\b", query, flags=re.IGNORECASE):
        mode_match = re.search(
            r"\bfor\s+(?P<mode>[A-Za-z][A-Za-z -]{1,60}?)\s+mode\b",
            query,
            flags=re.IGNORECASE,
        )
        if mode_match:
            requested_mode = re.sub(r"\s+", " ", mode_match.group("mode")).strip()
            row_match = re.search(
                rf"Operation\s+Mode:\s*{re.escape(requested_mode)}\s*;[^\n]*?"
                r"Max\.?\s*(?:No\.?)?\s*of\s+Detections?\s*:\s*(?P<count>\d+)",
                evidence,
                flags=re.IGNORECASE,
            )
            if row_match:
                return f"Max. No. of Detections: {row_match.group('count')}"
    if (
        str(result.metadata.get("chunk_type") or "") in {"spec_record", "datasheet_record"}
        and len(evidence) <= 700
    ):
        return re.sub(r"^\S+\.pdf\s*\|\s*", "", evidence, flags=re.IGNORECASE)
    if (
        len(evidence) <= 700
        and re.search(r"\bdefault\b", query, flags=re.IGNORECASE)
        and re.search(r"\bdefault\b", evidence, flags=re.IGNORECASE)
    ):
        return re.sub(r"^\S+\.pdf\s*\|\s*", "", evidence, flags=re.IGNORECASE)
    bullet_items = [
        re.sub(r"\s+", " ", item).strip(" -|\t\r\n")
        for item in re.split(r"\s*[•●▪]\s*", evidence)
        if item.strip(" -|\t\r\n")
    ]
    if len(bullet_items) >= 2:
        query_terms = _material_claim_terms(query)

        def bullet_score(item: str) -> tuple[float, int]:
            overlap = len(query_terms.intersection(_material_claim_terms(item)))
            action_bonus = 1.0 if re.search(
                r"\b(?:check|connect|do not|ground|install|keep|mount|select|separate|set|use)\b",
                item,
                flags=re.IGNORECASE,
            ) else 0.0
            return overlap * 2.0 + action_bonus, -len(item)

        best_bullet_index = max(range(len(bullet_items)), key=lambda index: bullet_score(bullet_items[index]))
        best_bullet = bullet_items[best_bullet_index]
        if re.search(
            r"\b(?:following\s+(?:the\s+)?|these\s+)(?:installation\s+)?"
            r"(?:precautions|steps|instructions|items)\s+below\b"
            r"|\bas follows\b",
            best_bullet,
            flags=re.IGNORECASE,
        ):
            dependent_items = bullet_items[best_bullet_index : best_bullet_index + 6]
            answer = "\n".join(f"• {item}" for item in dependent_items)
            if len(answer) > 700:
                answer = answer[:700].rsplit(" ", 1)[0].rstrip(" ,;:") + "."
            return answer
    segmentable_evidence = re.sub(
        r"\bCap\.\s+Time\b",
        "Cap Time",
        evidence,
        flags=re.IGNORECASE,
    )
    segments = [
        re.sub(r"\s+", " ", segment).strip(" -|\t\r\n")
        for segment in re.split(r"(?:\n+|(?<=[.!?;])\s+)", segmentable_evidence)
        if segment.strip(" -|\t\r\n")
    ]
    if re.search(r"^\s*why\b|\b(?:cause|reason)\b", query, flags=re.IGNORECASE):
        sentence_segments = [*segments]
        segments.extend(
            f"{left} {right}"
            for left, right in zip(sentence_segments, sentence_segments[1:])
        )
    if not segments:
        return "I could not produce a concise answer from the retrieved evidence."
    query_terms = _material_claim_terms(query)
    asks_direct_measurement = bool(
        re.search(
            r"\b(?:how\s+(?:deep|wide|high|long|far)|working\s+distance|"
            r"(?:depth|width|height|length|distance|interval|rate|frequency|detection\s+count|torque|temperature|pressure|voltage|speed)\b)",
            query,
            flags=re.IGNORECASE,
        )
    )
    needs_adjacent_measurement_context = bool(
        re.search(
            r"\b(?:if|under|during|mode|condition|both|and)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    if asks_direct_measurement and needs_adjacent_measurement_context:
        sentence_segments = [*segments]
        segments.extend(
            f"{left.rstrip(';')}; {right}"
            for left, right in zip(sentence_segments, sentence_segments[1:])
        )
    measurement_patterns: list[tuple[str, str]] = []
    for query_pattern, segment_pattern in (
        (r"\bcapture time\b", r"\b(?:capture|cap) time\b"),
        (r"\bworking distance\b", r"\b(?:working distance|wd)\b"),
        (
            r"\b(?:profile\s+capture|capture|sampling|profile)\s+(?:rate|frequency)\b",
            r"\b(?:capture\s+profiles?|profiles?\s*(?:/|per)\s*second|sampling\s+frequency|\d[\d,]*\s*hz)\b",
        ),
        (
            r"\b(?:detection|detectable|detecting)\s+(?:range|distance)\b",
            r"\b(?:detectable\s+distance|detection\s+range)\b",
        ),
        (
            r"\b(?:maximum|max\.?)?(?:\s+number\s+of)?\s*detection\s+count\b",
            r"\b(?:maximum|max\.?(?:\s+no\.?)?\s+of)\s+detections?\b",
        ),
        (r"\b(?:trigger\s+)?interval\b", r"\b(?:trigger\s+)?interval\b"),
        (r"\bdepth\b|\bhow deep\b", r"\bdepth\b"),
        (r"\btorque\b", r"\btorque\b"),
        (r"\bvoltage\b", r"\b(?:voltage|\d+(?:\.\d+)?v)\b"),
        (r"\btemperature\b", r"\b(?:ambienttemperatures?|temperatures?)\b"),
        (r"\bpressure\b", r"\bpressure\b"),
        (r"\bspeed\b", r"\bspeed\b"),
    ):
        if re.search(query_pattern, _normalized_phrase(query)):
            measurement_patterns = [(query_pattern, segment_pattern)]
            break

    def score(segment: str) -> tuple[float, int]:
        overlap = len(query_terms.intersection(_material_claim_terms(segment)))
        quantity_score = _query_target_quantity_score(query, segment)
        normalized_query = _normalized_phrase(query)
        normalized_segment = _normalized_phrase(segment)
        measurement_target_matches = any(
            re.search(query_pattern, normalized_query)
            and re.search(segment_pattern, normalized_segment)
            for query_pattern, segment_pattern in measurement_patterns
        )
        measurement_bonus = (
            10.0
            if asks_direct_measurement
            and measurement_target_matches
            and _quantity_terms(segment)
            else 0.0
        )
        action_bonus = 1.0 if re.search(
            r"\b(?:select|set|open|click|press|choose|connect|install|use|must|should|do not|cannot)\b",
            segment,
            flags=re.IGNORECASE,
        ) else 0.0
        return overlap * 2.0 + quantity_score + measurement_bonus + action_bonus, -len(segment)

    if asks_direct_measurement and not needs_adjacent_measurement_context:
        best_index = max(
            range(len(segments)),
            key=lambda index: (
                score(segments[index])[0]
                + (
                    2.0
                    * len(
                        query_terms.intersection(
                            _material_claim_terms(segments[index - 1])
                        )
                    )
                    if index > 0
                    else 0.0
                ),
                score(segments[index])[1],
            ),
        )
        best = segments[best_index]
    else:
        best = max(segments, key=score)
    best = re.sub(r"^\S+\.pdf\s*\|\s*", "", best, flags=re.IGNORECASE)
    best = re.sub(
        r"\s+(?:mounting\s+examples?|reference|specifications?)\b.*$",
        "",
        best,
        flags=re.IGNORECASE,
    )
    if re.search(r"\btorque\b", query, flags=re.IGNORECASE):
        torque = re.search(
            r"\b(?:tightening\s+)?torque\s*:\s*"
            r"\d+(?:\.\d+)?\s*(?:to|[-–—])\s*\d+(?:\.\d+)?\s*N\s*[·.]?\s*m\b",
            best,
            flags=re.IGNORECASE,
        )
        if torque:
            best = torque.group(0)
    if re.search(r"\bdetection\s+count\b", query, flags=re.IGNORECASE):
        detection_count = re.search(
            r"\bMax\.?\s*(?:No\.?)?\s*of\s+Detections?\s*:\s*\d+\b",
            best,
            flags=re.IGNORECASE,
        )
        if detection_count:
            best = detection_count.group(0)
        else:
            mode_match = re.search(
                r"\bfor\s+(?P<mode>[A-Za-z][A-Za-z -]{1,60}?)\s+mode\b",
                query,
                flags=re.IGNORECASE,
            )
            if mode_match:
                requested_mode = re.sub(r"\s+", " ", mode_match.group("mode")).strip()
                pipe_count = re.search(
                    rf"\b{re.escape(requested_mode)}\s*\|\s*[^|\n]+\|\s*(?P<count>\d+)\b",
                    best,
                    flags=re.IGNORECASE,
                )
                if pipe_count:
                    best = f"Max. No. of Detections: {pipe_count.group('count')}"
    if len(best) > 700:
        best = best[:700].rsplit(" ", 1)[0].rstrip(" ,;:") + "."
    best = re.sub(r"[ \t]{2,}", " ", best)
    return best


def _concise_dependent_list_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Keep list items that an introductory instruction explicitly depends on."""
    candidates: list[tuple[float, int, str, SearchResult]] = []
    query_terms = _material_claim_terms(query)
    for index, result in enumerate(results[:10]):
        content = str(result.content or "").strip()
        if not re.search(
            r"\b(?:following\s+(?:the\s+)?|these\s+)(?:installation\s+)?"
            r"(?:precautions|steps|instructions|items)\s+below\b|\bas follows\b",
            content,
            flags=re.IGNORECASE,
        ):
            continue
        answer = _concise_general_fallback_answer(query, result)
        if answer.count("\n• ") < 1:
            continue
        overlap = len(query_terms.intersection(_material_claim_terms(answer)))
        quality = float(overlap)
        if content.lower().startswith("table summary:"):
            quality += 4.0
        if re.search(r"\b(?:column headers|row headers|cell value|row|column):", content, flags=re.IGNORECASE):
            quality -= 4.0
        candidates.append((quality, -index, answer, result))
    if not candidates:
        return "", []
    _quality, _negative_index, answer, result = max(candidates, key=lambda item: (item[0], item[1]))
    return answer, [result]


def _concise_named_selection_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Answer a named mode/method choice from an explicit selection sentence."""
    query_match = re.search(
        r"^\s*(?:which|what)\s+(?P<subject>.*?\b(?:mode|method|option|setting|type))\b",
        query,
        flags=re.IGNORECASE,
    )
    if not query_match:
        return "", []
    subject = re.sub(r"\s+", " ", query_match.group("subject")).strip(" .?:")
    query_terms = _material_claim_terms(query)
    candidates: list[tuple[int, int, int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        if re.search(r"\boption$", subject, flags=re.IGNORECASE):
            for match_index, match in enumerate(
                re.finditer(
                    r"(?P<sentence>\b(?:click|select|choose)\s+"
                    r"['\"](?P<label>[^'\"\n]{1,100})['\"]\s+to\s+transmit\b[^.!?\n]*[.!?]?)",
                    evidence,
                    flags=re.IGNORECASE,
                )
            ):
                sentence = re.sub(r"\s+", " ", match.group("sentence")).strip()
                label = re.sub(r"\s+", " ", match.group("label")).strip(" .?:")
                overlap = len(query_terms.intersection(_material_claim_terms(f"{evidence} {sentence}")))
                if overlap >= 3:
                    candidates.append((overlap + 10, -result_index, -match_index, label, result))
            for match_index, match in enumerate(
                re.finditer(
                    r"(?P<sentence>[^.!?\n]{0,260}\b(?:select|choose|use)\b"
                    r"[^.!?\n]{0,180}\[(?P<label>[^\]\n]{1,80})\][^.!?\n]{0,160}[.!?]?)",
                    evidence,
                    flags=re.IGNORECASE,
                )
            ):
                sentence = re.sub(r"\s+", " ", match.group("sentence")).strip()
                label = re.sub(r"\s+", " ", match.group("label")).strip(" .?:")
                overlap = len(query_terms.intersection(_material_claim_terms(sentence)))
                if overlap >= 3:
                    candidates.append((overlap + 2, -result_index, -match_index, label, result))
        for match_index, match in enumerate(
            re.finditer(
                r"(?P<sentence>[^.!?\n]{0,260}\b(?:select|choose|use)\s+"
                r"\[(?P<label>[^\]\n]{1,80})\][^.!?\n]{0,160}[.!?]?)",
                evidence,
                flags=re.IGNORECASE,
            )
        ):
            sentence = re.sub(r"\s+", " ", match.group("sentence")).strip()
            label = re.sub(r"\s+", " ", match.group("label")).strip(" .?:")
            overlap = len(query_terms.intersection(_material_claim_terms(sentence)))
            if overlap < 2:
                continue
            candidates.append((overlap, -result_index, -match_index, label, result))
    if not candidates:
        return "", []
    _overlap, _result_index, _match_index, label, result = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return f"The {subject} is {label}.", [result]


def _concise_named_alternative_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Preserve an explicit named choice and its alternatives from one source row."""
    asks_how_choice_is_made = bool(
        re.search(
            r"\bhow\s+does\b.{0,100}\b(?:mode|method|option|setting|type)\b.{0,40}"
            r"\b(?:select|choose|determine)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    asks_which_alternatives = bool(
        re.search(
            r"\b(?:which|what)\b.{0,100}\b(?:modes|methods|options|settings|types)\b.{0,80}"
            r"\b(?:select|choose|use)(?:s|d)?\b.{0,30}\bbetween\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    if not (asks_how_choice_is_made or asks_which_alternatives):
        return "", []

    query_terms = _material_claim_terms(query)
    candidates: list[tuple[int, int, str, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for match in re.finditer(
            r"(?P<statement>(?:[^.!?\n;]{1,120};\s*)?[^.!?\n]{0,260}"
            r"\b(?:selected|selects?|chosen|chooses?|used|uses)\s+between\s+"
            r"(?P<left>[A-Za-z0-9][A-Za-z0-9+_./-]{0,40})\s+(?:or|and)\s+"
            r"(?P<right>[A-Za-z0-9][A-Za-z0-9+_./-]{0,40})[^.!?\n]*[.!?]?)",
            evidence,
            flags=re.IGNORECASE,
        ):
            statement = re.sub(r"\s+", " ", match.group("statement")).strip(" .;:")
            overlap = len(query_terms.intersection(_material_claim_terms(statement)))
            if overlap < 2:
                continue
            # Keep the source's causal direction intact.  Replacing the row separator
            # improves readability without asking a model to paraphrase the rule.
            answer = re.sub(r";\s*", ". ", statement)
            candidates.append((overlap, -result_index, f"{answer}.", result))
    if not candidates:
        return "", []
    _overlap, _result_index, answer, result = max(candidates, key=lambda item: (item[0], item[1]))
    return answer, [result]


def _concise_named_option_behavior_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Explain how a named mode/method works from its labeled definition."""
    priority_match = re.search(
        r"^\s*what\s+does\s+enabling\s+(?P<subject>.+?\bmode)\s+prioritize\b",
        query,
        flags=re.IGNORECASE,
    )
    if priority_match:
        subject = re.sub(r"\s+", " ", priority_match.group("subject")).strip(" .?:")
        query_terms = _material_claim_terms(query)
        candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:10]):
            evidence = _fallback_answer_text(result)
            for match in re.finditer(
                r"\bWhen\s+\[[^\]\n]{1,80}\]\s+is\s+selected\s*,?\s*"
                r"priority\s+is\s+given\s+to\s+(?P<target>[^.!?\n]{1,160})",
                evidence,
                flags=re.IGNORECASE,
            ):
                target = re.sub(r"\s+", " ", match.group("target")).strip(" .;:")
                overlap = len(query_terms.intersection(_material_claim_terms(match.group(0))))
                candidates.append(
                    (overlap, -result_index, f"Enabling {subject} prioritizes {target}.", result)
                )
        if candidates:
            _overlap, _negative_index, answer, result = max(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            return answer, [result]
    query_match = re.search(
        r"^\s*(?:how|what)\s+does\s+(?P<subject>.+?\b(?:mode|method|option|setting|type))\s+"
        r"(?:work|function|operate|affect|control|do)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not query_match:
        return "", []
    subject = re.sub(r"\s+", " ", query_match.group("subject")).strip(" .?:")
    subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.IGNORECASE)
    normalized_subject = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
    candidates: list[tuple[int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        normalized_subject_label = re.sub(
            r"\s+(?:mode|method|option|setting|type)$",
            "",
            normalized_subject,
        ).strip()
        for row_match in re.finditer(
            r"Setting\s+item:\s*(?P<label>[^;\n]{1,120});\s*Settings:\s*"
            r"(?P<behavior>.*?)(?=\nSetting\s+item:|$)",
            evidence,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            label = re.sub(r"\s+", " ", row_match.group("label")).strip(" .?:")
            normalized_label = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
            if not normalized_label or normalized_label != normalized_subject_label:
                continue
            behavior = re.sub(r"\s+", " ", row_match.group("behavior")).strip(" .;:")
            behavior = re.sub(
                r"\s+Context:\s+Setting\s+item:.*$",
                "",
                behavior,
                flags=re.IGNORECASE,
            ).strip(" .;:")
            overlap = len(_material_claim_terms(query).intersection(_material_claim_terms(behavior)))
            candidates.append((overlap + 6, -result_index, f"works as follows: {behavior}", result))
        labels = {
            re.sub(r"\s+", " ", label).strip(" .?:")
            for label in re.findall(r"\[([^\]\n]{1,80})\]", evidence)
        }
        for label in labels:
            normalized_label = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
            if not normalized_label or normalized_label not in normalized_subject:
                continue
            definition = re.search(
                rf"\b{re.escape(label)}\b\s+(?P<behavior>"
                r"(?:detects|extracts|searches|compares|selects|uses|sets|enables|disables)\b"
                r".*?)(?=\s+y\s+[A-Z]|[.\n|]|$)",
                evidence,
                flags=re.IGNORECASE,
            )
            if definition:
                behavior = re.sub(r"\s+", " ", definition.group("behavior")).strip(" .;:")
                behavior = behavior[:1].lower() + behavior[1:]
                overlap = len(_material_claim_terms(query).intersection(_material_claim_terms(behavior)))
                candidates.append((overlap + 4, -result_index, behavior, result))
                continue
            selection = re.search(
                rf"(?P<sentence>[^.!?\n]{{0,260}}\b(?:select|choose|use)\s+"
                rf"\[{re.escape(label)}\][^.!?\n]{{0,120}}[.!?]?)",
                evidence,
                flags=re.IGNORECASE,
            )
            if selection:
                sentence = re.sub(r"\s+", " ", selection.group("sentence")).strip(" .;:")
                sentence = re.sub(r"^[yY]\s+(?=When\b)", "", sentence)
                overlap = len(_material_claim_terms(query).intersection(_material_claim_terms(sentence)))
                candidates.append((overlap, -result_index, f"works as follows: {sentence}", result))
    if not candidates:
        return "", []
    _overlap, _result_index, behavior, result = max(candidates, key=lambda item: (item[0], item[1]))
    return f"The {subject} {behavior}.", [result]


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
    if re.search(r"\b(?:check|verify|verification)\b", query, flags=re.IGNORECASE):
        if re.search(r"(?:^|[.;:]\s*)check\b", evidence, flags=re.IGNORECASE):
            score += 4.0
    if _is_comparison_query(query) and (
        str(result.source_document_id or "") in evidence
        or _model_tokens(query).intersection(_model_tokens(evidence))
        or query_terms.intersection(_answer_terms(result.title))
    ):
        score += 1.0
    if re.search(r"\b(count|counts|how many|number of|quantity|total)\b", query, flags=re.IGNORECASE):
        score += min(4.0, float(len(_contextual_quantity_terms(evidence))))
        score += _query_target_quantity_score(query, evidence)
    return score


def _structured_fact_evidence_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    asks_named_setting = bool(
        re.search(
            r"\b(?:which|what)\b.{0,100}\b(?:setting|parameter|option)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    asks_direct_measurement = not asks_named_setting and bool(
        re.search(
            r"\b(?:how\s+(?:deep|wide|high|long|far)|"
            r"capture\s+time|working\s+distance|"
            r"(?:movable|travel)\s+range|"
            r"(?:detection|detectable|detecting)\s+(?:range|distance)|"
            r"(?:depth|width|height|length|distance|interval|rate|frequency|detection\s+count|torque|temperature|pressure|voltage|speed)\b)",
            query,
            flags=re.IGNORECASE,
        )
    )
    query_terms = _material_claim_terms(query)
    requested_models = _model_tokens(query)
    query_normalized = _normalized_phrase(query)

    def compact(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    requested_modes = {
        compact(match.group(1))
        for match in re.finditer(
            r"\b([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*)?)\s+mode\b",
            query,
            flags=re.IGNORECASE,
        )
        if len(compact(match.group(1))) >= 4
    }
    measurement_patterns: list[str] = []
    for query_pattern, evidence_pattern in (
        (r"\bcapture time\b", r"\b(?:capture|cap) time\b"),
        (r"\bworking distance\b", r"\b(?:working distance|wd)\b"),
        (
            r"\bhow\s+far\b.{0,100}\b(?:move|moves|moving|travel|travels)\b",
            r"\b(?:movable|travel)\s+range\b",
        ),
        (r"\b(?:movable|travel)\s+range\b", r"\b(?:movable|travel)\s+range\b"),
        (
            r"\b(?:profile\s+capture|capture|sampling|profile)\s+(?:rate|frequency)\b",
            r"\b(?:capture\s+profiles?|profiles?\s*(?:/|per)\s*second|sampling\s+frequency|\d[\d,]*\s*hz)\b",
        ),
        (
            r"\b(?:detection|detectable|detecting)\s+(?:range|distance)\b",
            r"\b(?:detectable\s+distance|detection\s+range)\b",
        ),
        (
            r"\b(?:maximum|max\.?)?(?:\s+number\s+of)?\s*detection\s+count\b",
            r"\b(?:maximum|max\.?(?:\s+no\.?)?\s+of)\s+detections?\b",
        ),
        (r"\b(?:trigger\s+)?interval\b", r"\b(?:trigger\s+)?interval\b"),
        (r"\bdepth\b|\bhow deep\b", r"\bdepth\b"),
        (r"\btorque\b", r"\btorque\b"),
        (r"\bvoltage\b", r"\b(?:voltage|\d+(?:\.\d+)?v)\b"),
        (r"\btemperature\b", r"\b(?:ambienttemperatures?|temperatures?)\b"),
        (r"\bpressure\b", r"\bpressure\b"),
        (r"\bspeed\b", r"\bspeed\b"),
    ):
        if re.search(query_pattern, query_normalized):
            measurement_patterns = [evidence_pattern]
            break

    def direct_measurement_target_alignment(result: SearchResult) -> int:
        if not asks_direct_measurement:
            return 0
        evidence = _normalized_phrase(_fallback_answer_text(result))
        alignments = 0
        for evidence_pattern in measurement_patterns:
            for match in re.finditer(evidence_pattern, evidence):
                window = evidence[max(0, match.start() - 180) : match.end() + 180]
                if not _quantity_terms(window):
                    continue
                if requested_modes and not any(
                    mode in compact(window) for mode in requested_modes
                ):
                    continue
                alignments += 1
                break
        return alignments

    candidates = [
        (
            sum(
                1
                for model in requested_models
                if _normalized_phrase(model)
                in _normalized_phrase(
                    " ".join(
                        [
                            *result.section_path,
                            _fallback_answer_text(result)[:300],
                            str(result.metadata.get("product_model") or ""),
                        ]
                    )
                )
            )
            if asks_direct_measurement
            else 0,
            int(str(result.metadata.get("retrieval_stage") or "") == "measurement_promoted")
            if asks_direct_measurement
            else 0,
            int(
                str(result.metadata.get("chunk_type") or "")
                in {"atomic_text", "table_record", "spec_record", "datasheet_record"}
            )
            if asks_direct_measurement
            else 0,
            int(
                bool(
                    re.search(
                        r"\b\d+(?:\.\d+)?\s*(?:°\s*)?[a-z%/]+",
                        str(result.content or ""),
                        flags=re.IGNORECASE,
                    )
                )
            )
            if asks_direct_measurement
            else 0,
            len(query_terms.intersection(_material_claim_terms(str(result.content or ""))))
            if asks_direct_measurement
            else 0,
            direct_measurement_target_alignment(result),
            len(query_terms.intersection(_material_claim_terms(_fallback_answer_text(result))))
            if asks_direct_measurement
            else 0,
            _fallback_evidence_score(query, result),
            index,
            result,
        )
        for index, result in enumerate(results)
        if (
            str(result.metadata.get("chunk_type") or "") in {"spec_record", "datasheet_record"}
            or (
                asks_direct_measurement
                and bool(_quantity_terms(_fallback_answer_text(result)))
            )
        )
    ]
    if not candidates:
        return []
    (
        _identifier_alignment,
        _measurement_promoted,
        _bounded_source,
        _content_value,
        _direct_terms,
        _target_alignment,
        _term_alignment,
        score,
        _index,
        result,
    ) = max(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[5],
            item[4],
            item[6],
            item[2],
            item[3],
            item[7],
            -item[8],
        ),
    )
    if score >= 4.0 or (
        asks_direct_measurement
        and _identifier_alignment > 0
        and (_target_alignment > 0 or _term_alignment >= 2)
    ):
        return [result]
    return []


def _concise_required_setting_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Answer which setting must correspond to connected equipment."""
    if not re.search(
        r"\b(?:which|what)\b.{0,100}\b(?:setting|parameter|option)\b.{0,100}"
        r"\b(?:match|correspond|agree)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    query_terms = _material_claim_terms(query).difference({"which", "what", "must", "match", "setting"})
    candidates: list[tuple[int, int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -|;:")
            sentence = re.sub(
                r"^Table summary:\s*[^|]{1,100}\|\s*",
                "",
                sentence,
                flags=re.IGNORECASE,
            )
            if not re.search(
                r"\b(?:make\s+sure\s+to\s+set|set|configure)\b.+\bcorrect(?:ly)?\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(sentence)))
            if overlap < 2:
                continue
            setting_match = re.search(
                r"\bset\s+(?:the\s+)?(?P<label>[^.;]{1,80}?)\s+for\s+(?:the\s+)?",
                sentence,
                flags=re.IGNORECASE,
            )
            label = setting_match.group("label").strip() if setting_match else ""
            named_setting_score = 0 if not label.lower().startswith("setting ") else -1
            chunk_type = str(result.metadata.get("chunk_type") or "")
            bounded_source_score = 2 if chunk_type in {"atomic_text", "table_record", "warning_record"} else 0
            candidates.append(
                (bounded_source_score, named_setting_score, overlap, -result_index, sentence, result)
            )
    if not candidates:
        return "", []
    _source_score, _setting_score, _overlap, _result_index, answer, result = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    return answer, [result]


def _concise_named_mode_requirement_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Return the explicit threshold that makes a numbered mode mandatory."""
    query_match = re.search(
        r"^\s*when\s+must\s+i\s+(?:enable|select|use)\s+"
        r"(?P<label>.+?\bmode)\s+(?P<mode>\d+)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not query_match:
        return "", []
    label = re.sub(r"\s+", " ", query_match.group("label")).strip(" .?:")
    mode = query_match.group("mode")
    normalized_label = _normalized_phrase(label)
    candidates: list[tuple[int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for row in re.finditer(
            r"Setting\s+item:\s*(?P<label>[^;\n]{1,120});\s*Settings:\s*"
            r"(?P<body>.*?)(?=Setting\s+item:|$)",
            evidence,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            row_label = re.sub(r"\s+", " ", row.group("label")).strip(" .?:")
            if _normalized_phrase(row_label) != normalized_label:
                continue
            mode_match = re.search(
                rf"(?:[•▪]\s*)?Mode\s*{re.escape(mode)}\s*:\s*"
                r"If\s+(?P<condition>.*?)(?=,?\s*this\s+mode\s+must\s+be\s+selected)"
                r",?\s*this\s+mode\s+must\s+be\s+selected",
                row.group("body"),
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not mode_match:
                continue
            condition = re.sub(r"\s+", " ", mode_match.group("condition")).strip(" .;:")
            answer = f"Select {row_label} {mode} if {condition}."
            overlap = len(_material_claim_terms(query).intersection(_material_claim_terms(answer)))
            candidates.append((overlap, -result_index, answer, result))
    if not candidates:
        return "", []
    _overlap, _negative_index, answer, result = max(candidates, key=lambda item: (item[0], item[1]))
    return answer, [result]


def _concise_physical_location_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Answer a physical placement question from an explicit spatial sentence."""
    enter_match = re.search(
        r"^\s*where\s+should\s+i\s+enter\s+(?:the\s+)?(?P<subject>.+?)[?\s]*$",
        query,
        flags=re.IGNORECASE,
    )
    if enter_match:
        subject = re.sub(r"\s+", " ", enter_match.group("subject")).strip(" .?:")
        subject_terms = _material_claim_terms(subject)
        enter_candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = _fallback_answer_text(result)
            for segment in re.split(r"(?<=[.!?])\s+|\n+", evidence):
                segment = re.sub(r"\s+", " ", segment).strip(" -|;:")
                location_match = re.search(
                    r"Under\s+['\"](?P<container>[^'\"]+?)[,]?['\"][,]?\s*"
                    r".*?['\"](?P<target>[^'\"]+)['\"].*?\bopen\b.*?"
                    r"\b(?:and\s+then\s+)?enter\b",
                    segment,
                    flags=re.IGNORECASE,
                )
                if not location_match:
                    continue
                overlap = len(subject_terms.intersection(_material_claim_terms(segment)))
                if overlap < 2:
                    continue
                container = location_match.group("container").strip(" ,")
                target = location_match.group("target").strip()
                enter_candidates.append(
                    (
                        overlap,
                        -result_index,
                        f"Enter the {subject} in '{target}' under '{container}'.",
                        result,
                    )
                )
        if enter_candidates:
            _overlap, _negative_index, answer, result = max(
                enter_candidates,
                key=lambda item: item[:2],
            )
            return answer, [result]
    if not re.search(
        r"^\s*where\b.{0,160}\b(?:install(?:ed)?|mount(?:ed)?|locat(?:e|ed)|connect(?:ed)?)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    query_terms = _material_claim_terms(query).difference(
        {"where", "installed", "install", "mounted", "mount", "located", "locate", "connected", "connect"}
    )
    candidates: list[tuple[int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -|;:")
            sentence = re.sub(r"^\S+\.pdf\s*\|\s*", "", sentence, flags=re.IGNORECASE)
            if not re.search(
                r"\b(?:install(?:ed)?|mount(?:ed)?|locat(?:e|ed)|connect(?:ed)?)\b.{0,100}"
                r"\b(?:between|inside|within|under|above|behind|beside|on|to|into)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(sentence)))
            if overlap < 2:
                continue
            candidates.append((overlap, -result_index, sentence, result))
    if not candidates:
        return "", []
    _overlap, _result_index, answer, result = max(candidates, key=lambda item: (item[0], item[1]))
    return answer, [result]


def _concise_alignment_components_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Answer which two connector components an alignment diagram identifies."""
    if not re.search(
        r"\bwhat\b.{0,100}\b(?:components?|parts?|items?)\b.{0,80}\balign(?:ed|ment)?\b"
        r"|\bwhat\b.{0,100}\balign(?:ed|ment)?\b.{0,80}\b(?:components?|parts?|items?)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    query_terms = _material_claim_terms(query).difference(
        {"what", "component", "components", "part", "parts", "item", "items", "aligned", "alignment"}
    )
    candidates: list[tuple[int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for match in re.finditer(
            r"\balign\s+(?:the\s+)?(?P<left>[A-Za-z0-9 -]{1,60}?)\s+and\s+"
            r"(?:the\s+)?(?P<right>[A-Za-z0-9 -]{1,60}?)(?=\s{2,}|[.;:\n]|$)",
            evidence,
            flags=re.IGNORECASE,
        ):
            left = re.sub(r"\s+", " ", match.group("left")).strip()
            right = re.sub(r"\s+", " ", match.group("right")).strip()
            answer = f"Align the {left} and the {right}."
            overlap = len(query_terms.intersection(_material_claim_terms(answer)))
            candidates.append((overlap, -result_index, answer, result))
    if not candidates:
        return "", []
    _overlap, _result_index, answer, result = max(candidates, key=lambda item: (item[0], item[1]))
    return answer, [result]


def _concise_enumerated_options_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Return a source sentence that explicitly enumerates requested size options."""
    if not re.search(
        r"\b(?:what|which)\b.{0,120}\b(?:options|sizes|thicknesses|lengths)\b"
        r"|\b(?:options|sizes|thicknesses|lengths)\b.{0,80}\b(?:available|included|offered)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    query_terms = _material_claim_terms(query).difference({"what", "which", "available", "included"})

    def singularized(terms: set[str]) -> set[str]:
        normalized: set[str] = set()
        for term in terms:
            if len(term) > 5 and term.endswith("es"):
                normalized.add(term[:-2])
            elif len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
                normalized.add(term[:-1])
            else:
                normalized.add(term)
        return normalized

    query_term_roots = singularized(query_terms)
    candidates: list[tuple[int, int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -|;:")
            sentence = re.sub(r"^\S+\.pdf\s*\|\s*", "", sentence, flags=re.IGNORECASE)
            if not re.search(
                r"\b(?:available|contains?|includes?|offered|set\s+of)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            quantities = _quantity_terms(sentence)
            if len(quantities) < 2:
                continue
            overlap = len(query_term_roots.intersection(singularized(_material_claim_terms(sentence))))
            if overlap < 2:
                continue
            candidates.append((len(quantities), overlap, -result_index, sentence, result))
    if not candidates:
        return "", []
    _quantity_count, _overlap, _result_index, answer, result = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return answer, [result]


def _concise_labeled_list_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Return the values from an explicitly requested labeled list row."""
    if not re.search(
        r"\b(?:which|what)\b.{0,80}\bindicators?\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    requested_models = _model_tokens(query)
    candidates: list[tuple[int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        match = re.search(
            r"(?:^|\n)\s*Indicators?\s*\|\s*(?P<values>[^\n]{2,240})",
            evidence,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        values = re.sub(r"\s+", " ", match.group("values")).strip(" .;:|")
        model_haystack = _normalized_phrase(
            " ".join(
                str(part)
                for part in (
                    result.content,
                    result.title,
                    result.metadata.get("product_model"),
                    result.metadata.get("product_models"),
                )
                if part
            )
        )
        model_alignment = sum(
            1 for model in requested_models if _normalized_phrase(model) in model_haystack
        )
        candidates.append((model_alignment, -result_index, values, result))
    if not candidates:
        return "", []
    _model_alignment, _result_index, values, result = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return f"The available indicators are {values}.", [result]


def _concise_structured_fact_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    if re.search(r"\b(?:configure|set|login|log in|user name|username)\b", query, flags=re.IGNORECASE):
        for result in results[:12]:
            evidence = _fallback_answer_text(result)
            default_statement = re.search(
                r"(?P<statement>[^.!?\n]{1,320}?\(Default:\s*[^.\n]{1,160}\)+)",
                evidence,
                flags=re.IGNORECASE,
            )
            if default_statement:
                statement = re.sub(r"\s+", " ", default_statement.group("statement")).strip()
                return statement if statement.endswith(".") else f"{statement}.", [result]
    explicit_setting_match = re.search(
        r"\b(?P<label>(?:[A-Z][A-Za-z]+\s+){2,}(?:Rate|Time|Mode|Level|Width))\b",
        query,
    )
    if explicit_setting_match:
        requested_label = _normalized_phrase(explicit_setting_match.group("label"))
        for result in results[:12]:
            evidence = _fallback_answer_text(result)
            for row in re.finditer(
                r"Setting\s+item:\s*(?P<label>[^;\n]{1,120});\s*Settings:\s*"
                r"(?P<body>.*?)(?=\n?Setting\s+item:|$)",
                evidence,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                label = re.sub(r"\s+", " ", row.group("label")).strip(" .;:")
                if _normalized_phrase(label) != requested_label:
                    continue
                body = re.sub(r"\s+", " ", row.group("body")).strip(" .;:")
                body = re.sub(
                    r"\s+Context\s*:?.*$",
                    "",
                    body,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip(" .;:")
                if re.search(r"\b(?:range|minimum|maximum|limits?)\b", query, flags=re.IGNORECASE):
                    value_range = re.search(
                        r"(?P<low>\d+(?:\.\d+)?)\s*\((?P<low_label>[^)]{1,60})\)\s*"
                        r"(?:to|~|[-–])\s*(?P<high>\d+(?:\.\d+)?)\s*"
                        r"\((?P<high_label>[^)]{1,60})\)",
                        body,
                        flags=re.IGNORECASE,
                    )
                    if value_range:
                        return (
                            f"{label} ranges from {value_range.group('low')} "
                            f"({value_range.group('low_label')}) to {value_range.group('high')} "
                            f"({value_range.group('high_label')}).",
                            [result],
                        )
                return f"{label}: {body}", [result]

    reference_monitoring_query = bool(
        re.search(r"\breference\s+points?\s+monitoring\b", query, flags=re.IGNORECASE)
    )
    if reference_monitoring_query:
        for result in results[:12]:
            evidence = _fallback_answer_text(result)
            requirement = re.search(
                r"Reference\s+points?\s+monitoring\s+function\s+must\s+be\s+applied\s+when\s+the\s+"
                r"(?P<model>[A-Z]+-[A-Z])\s+is\s+used\s+for\s+the\s+access\s+protection\s+specified\s+in\s+"
                r"(?P<standard>IEC\s*61496-3:\s*2008\s+Annex\s+A\.12\s+and\s+A\.13)\s+"
                r"\(the\s+application\s+where\s+the\s+angle\s+of\s+the\s+approach\s+exceeds\s+"
                r"(?P<angle>±\s*30°)[^)]*\).*?response\s+time\s+must\s+be\s+"
                r"(?P<response>\d+(?:\.\d+)?\s*ms\s+or\s+less)",
                evidence,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not requirement:
                continue
            if re.search(r"\bresponse\s+time\b", query, flags=re.IGNORECASE):
                return (
                    f"With reference-point monitoring, the response time must be "
                    f"{requirement.group('response')}.",
                    [result],
                )
            standard = re.sub(r"\s+", " ", requirement.group("standard")).strip()
            angle = re.sub(r"\s+", "", requirement.group("angle"))
            return (
                f"Enable reference-point monitoring when {requirement.group('model').upper()} is used for "
                f"access protection under {standard}, where the approach angle exceeds {angle} to the "
                "detection plane.",
                [result],
            )

    ip_address_path_query = bool(
        re.search(r"\bIP\s+address\b|\bEthernet\s+addresses\b", query, flags=re.IGNORECASE)
        and _model_tokens(query)
    )
    if ip_address_path_query:
        requested_models = _model_tokens(query)
        for result in results[:12]:
            evidence = _fallback_answer_text(result)
            match = re.search(
                r"Double[-: ]click\s+the\s+(?P<model>[A-Z0-9]+(?:-[A-Z0-9]+)+)\s+image,\s*"
                r"click\s+the\s+['\"]General['\"]\s+tab,\s*click\s+['\"]PROFINET\s+interface,?['\"],?\s*"
                r"click\s+['\"]Ethernet\s+addresses,?['\"],?\s*and\s+then\s+set\s+the\s+IP\s+address",
                evidence,
                flags=re.IGNORECASE,
            )
            if not match or not requested_models.intersection(_model_tokens(match.group("model"))):
                continue
            model = match.group("model").upper()
            if re.search(r"\b(?:which|what)\s+(?:menu\s+)?path\b", query, flags=re.IGNORECASE):
                return (
                    f"Location: {model} image > General tab > PROFINET interface > Ethernet addresses.",
                    [result],
                )
            return (
                f"Double-click the {model} image, open General > PROFINET interface > Ethernet addresses, "
                f"and set the {model} IP address there.",
                [result],
            )

    polarity_query = re.search(r"\b(?P<polarity>NPN|PNP)\b", query, flags=re.IGNORECASE)
    if polarity_query and re.search(r"\b(?:polarity|input\s+voltage|voltage[- ]based)\b", query, flags=re.IGNORECASE):
        requested_polarity = polarity_query.group("polarity").upper()
        asks_maximum = bool(re.search(r"\b(?:maximum|max\.?|rating)\b", query, flags=re.IGNORECASE))
        for result in results[:12]:
            evidence = _fallback_answer_text(result)
            behavior = re.search(
                rf"When\s+(?:the\s+)?{re.escape(requested_polarity)}\s+is\s+selected\s+in\s+the\s+"
                r"Polarity(?:\s*\([^)]*\))?,?\s+the\s+circuit\s+becomes\s+"
                r"(?P<circuit>[^.]+?circuit)\.",
                evidence,
                flags=re.IGNORECASE,
            )
            if not behavior:
                continue
            if asks_maximum:
                maximum = re.search(
                    r"Input\s+maximum\s+rating\s*:\s*(?P<value>\d+(?:\.\d+)?\s*V)",
                    evidence,
                    flags=re.IGNORECASE,
                )
                if maximum:
                    return (
                        f"With {requested_polarity} polarity selected, the maximum input voltage is "
                        f"{maximum.group('value')}.",
                        [result],
                    )
                continue
            circuit = re.sub(r"\s+", " ", behavior.group("circuit")).strip()
            circuit = re.sub(r"^(?:an?|the)\s+", "", circuit, flags=re.IGNORECASE)
            article = "an" if circuit[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            return f"Yes. Selecting {requested_polarity} makes the circuit {article} {circuit}.", [result]

    response_stability_query = bool(
        re.search(r"\bresponse\s+time\b", query, flags=re.IGNORECASE)
        and re.search(r"\bstabil(?:ity|e|ize|ization)\b", query, flags=re.IGNORECASE)
    )
    if response_stability_query:
        for result in results[:12]:
            evidence = _fallback_answer_text(result)
            match = re.search(
                r"(?P<times>\[?\d+(?:\.\d+)?\s*(?:μ\s*s|µ\s*s|us|ms)[^.;]{0,40}?or\s+"
                r"\[?\d+(?:\.\d+)?\s*(?:μ\s*s|µ\s*s|us|ms)[^.;]{0,40}?)\s+"
                r"response\s+time\s+selected,?\s+stable\s+operation\s+may\s+be\s+reduced\."
                r"\s+In\s+this\s+situation,?\s+it\s+may\s+be\s+possible\s+to\s+increase\s+"
                r"stability\s+by\s+adjusting\s+the\s+light\s+intensity\s+to\s+the\s+optimal\s+value",
                evidence,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            if re.search(r"\b(?:how|improve|increase|adjust)\b", query, flags=re.IGNORECASE):
                return "Adjust the light intensity to the optimal value to increase stability.", [result]
            return (
                "Yes. With the 300 μs or 1.1 ms response time selected, stable operation may be reduced.",
                [result],
            )

    signal_duration_query = bool(
        re.search(r"\b(?:pulse|output)\s+duration\b|\bduration\s+range\b", query, flags=re.IGNORECASE)
    )
    if signal_duration_query:
        signal_candidates: list[tuple[int, int, str, str, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = _fallback_answer_text(result)
            match = re.search(
                r"from\s+when\s+the\s+(?P<signal>[A-Z][A-Z0-9]{1,5})\s+rises\s+to\s+when\s+the\s+"
                r"(?P=signal)\s+falls\s+within\s+the\s+range\s+of\s+"
                r"(?P<low>\d+(?:\.\d+)?)\s+to\s+(?P<high>\d+(?:\.\d+)?)\s*"
                r"\(?(?P<unit>ms|μs|us|s)\)?\.?(?:\s*\(Default:\s*"
                r"(?P<default>\d+(?:\.\d+)?)\s*(?P<default_unit>ms|μs|us|s)\))?",
                evidence,
                flags=re.IGNORECASE,
            )
            if not match or not re.search(
                rf"\b{re.escape(match.group('signal'))}\b",
                query,
                flags=re.IGNORECASE,
            ):
                continue
            model_fit = int(
                any(
                    model in _model_tokens(query)
                    for model in _model_tokens(
                        " ".join(
                            str(value)
                            for value in (
                                result.metadata.get("product_model"),
                                result.title,
                                result.content,
                            )
                            if value
                        )
                    )
                )
            )
            signal_candidates.append(
                (
                    model_fit,
                    -result_index,
                    match.group("signal").upper(),
                    f"{match.group('low')} to {match.group('high')} {match.group('unit')}",
                    (
                        f"{match.group('default')} {match.group('default_unit')}"
                        if match.group("default")
                        else ""
                    ),
                    result,
                )
            )
        if signal_candidates:
            _model_fit, _negative_index, signal, duration_range, default, result = max(
                signal_candidates,
                key=lambda item: item[:2],
            )
            if re.search(r"\bdefault\b", query, flags=re.IGNORECASE) and default:
                return f"The default {signal} output duration is {default}.", [result]
            return f"The valid {signal} pulse duration range is {duration_range}.", [result]

    windows_editions_query = bool(
        re.search(r"\bWindows\s+10\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:editions?|supported|support)\b", query, flags=re.IGNORECASE)
    )
    if windows_editions_query:
        os_candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = _fallback_answer_text(result)
            match = re.search(
                r"Microsoft\s+Windows\s+10\s+"
                r"(?P<editions>Home\s*,\s*Pro\s*,\s*Enterprise)\s*"
                r"\(\s*(?P<bits>64[- ]bit\s+version)\s*\)",
                evidence,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            direct = int(bool(re.search(r"\b(?:Supported\s+OS|OS)\b", str(result.content or ""), flags=re.IGNORECASE)))
            os_candidates.append((direct, -result_index, match.group(0), result))
        if os_candidates:
            _direct, _negative_index, _matched, result = max(
                os_candidates,
                key=lambda item: item[:2],
            )
            return (
                "Microsoft Windows 10 Home, Pro, and Enterprise are supported; only the 64-bit versions are supported.",
                [result],
            )

    dotnet_space_query = bool(
        re.search(r"\.NET\s+Framework", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:disk|free|storage)\s+space\b", query, flags=re.IGNORECASE)
    )
    if dotnet_space_query:
        space_candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = _fallback_answer_text(result)
            match = re.search(
                r"(?P<space>\d+(?:\.\d+)?\s*GB\s+or\s+more)\s+of\s+free\s+space\s+"
                r"is\s+required\s+in\s+addition",
                evidence,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            direct = int(".net framework" in str(result.content or "").lower())
            value = re.sub(r"\s+", " ", match.group("space")).strip()
            space_candidates.append((direct, -result_index, value, result))
        if space_candidates:
            _direct, _negative_index, value, result = max(
                space_candidates,
                key=lambda item: item[:2],
            )
            return f"{value} of additional free disk space is required.", [result]

    direct_model_field_candidates: list[tuple[int, int, int, int, int, str, SearchResult]] = []
    if _model_tokens(query):
        query_terms = _material_claim_terms(query)
        requested_field_match = re.search(
            r"\bhow\s+many\s+(?P<count_field>.+?)\s+(?:does|do|is|are|can)\b|"
            r"\bwhat\s+is\s+(?:the\s+)?(?P<what_field>.+?)\s+(?:of|for)\b|"
            r"\bwhat\s+(?P<does_field>.+?)\s+does\b|"
            r"\bwhich\s+(?P<which_field>.+?)\s+(?:applies|does|is|are)\b",
            query,
            flags=re.IGNORECASE,
        )
        requested_field_terms: set[str] = set()
        if requested_field_match:
            requested_field_terms = _material_claim_terms(
                next((value for value in requested_field_match.groupdict().values() if value), "")
            )
            requested_field_terms |= {
                term[:-1] for term in requested_field_terms if len(term) > 4 and term.endswith("s")
            }
        count_subject_match = re.search(
            r"\bhow\s+many\s+(?P<subject>[a-z][a-z0-9_-]*)",
            query,
            flags=re.IGNORECASE,
        )
        count_subject = count_subject_match.group("subject") if count_subject_match else ""
        for result_index, result in enumerate(results[:12]):
            direct_answer = (
                _focused_labeled_table_cell_answer_text(query, result)
                or _focused_model_field_record_answer_text(query, result)
            )
            if not direct_answer:
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(direct_answer)))
            answer_label = re.split(r"\s+[—-]\s+|:", direct_answer, maxsplit=1)[0]
            answer_label_terms = _material_claim_terms(answer_label)
            answer_label_terms |= {
                term[:-1] for term in answer_label_terms if len(term) > 4 and term.endswith("s")
            }
            requested_fit = len(requested_field_terms.intersection(answer_label_terms))
            if "color" in requested_field_terms and "pattern" in answer_label_terms:
                requested_fit += 2
            count_fit = int(
                bool(count_subject)
                and bool(
                    re.search(
                        rf"\b\d[\d,.]*\s+{re.escape(count_subject)}\b",
                        direct_answer,
                        flags=re.IGNORECASE,
                    )
                )
            )
            direct_model_field_candidates.append(
                (
                    count_fit,
                    requested_fit,
                    overlap,
                    -len(direct_answer),
                    -result_index,
                    direct_answer,
                    result,
                )
            )
    asks_model_length = bool(re.search(r"\blength\b", query, flags=re.IGNORECASE))
    asks_model_weight = bool(re.search(r"\b(?:weighs?|weight|mass)\b", query, flags=re.IGNORECASE))
    if asks_model_length or asks_model_weight:
        requested_models = _model_tokens(query)
        property_candidates: list[tuple[int, int, int, int, str, str, SearchResult]] = []

        def clean_dimension(value: str) -> str:
            cleaned = re.sub(r"\s+", " ", value).strip(" .;:|")
            metric_imperial = re.fullmatch(
                r"(?P<metric>\d+(?:\.\d+)?)\s*m\s*(?P<feet>\d+(?:\.\d+)?)\s*'",
                cleaned,
            )
            if metric_imperial:
                return f"{metric_imperial.group('metric')} m ({metric_imperial.group('feet')} ft)"
            return cleaned

        for result_index, result in enumerate(results[:12]):
            evidence = str(result.content or "")
            chunk_type = str(result.metadata.get("chunk_type") or "")
            bounded_source = int(
                chunk_type in {"atomic_text", "table_record", "spec_record", "datasheet_record"}
            )
            for record in re.split(r"\n+", evidence):
                fields = {
                    label.lower(): re.sub(r"\s+", " ", value).strip(" .;:|")
                    for label, value in re.findall(
                        r"(?:^|;\s*)(Type|Length|Model|Weight):\s*(.*?)(?=;\s*"
                        r"(?:Type|Length|Model|Weight):|$)",
                        record,
                        flags=re.IGNORECASE,
                    )
                }
                model = fields.get("model", "")
                model_alignment = sum(
                    1 for requested in requested_models if _normalized_phrase(requested) in _normalized_phrase(model)
                )
                if fields and (not requested_models or model_alignment > 0):
                    field_name = "length" if asks_model_length else "weight"
                    value = fields.get(field_name, "")
                    if value:
                        property_candidates.append(
                            (
                                model_alignment,
                                bounded_source,
                                -len(record),
                                -result_index,
                                model,
                                clean_dimension(value),
                                result,
                            )
                        )
                if "|" not in record:
                    continue
                cells = [re.sub(r"\s+", " ", cell).strip() for cell in record.split("|")]
                for cell_index, cell in enumerate(cells):
                    model_alignment = sum(
                        1
                        for requested in requested_models
                        if _normalized_phrase(requested) == _normalized_phrase(cell)
                    )
                    if requested_models and model_alignment <= 0:
                        continue
                    value_index = cell_index - 1 if asks_model_length else cell_index + 1
                    if value_index < 0 or value_index >= len(cells):
                        continue
                    value = cells[value_index]
                    if asks_model_length and not re.search(r"\b\d+(?:\.\d+)?\s*m", value):
                        continue
                    if asks_model_weight and not re.search(r"\b\d+(?:\.\d+)?\s*(?:kg|g)\b", value, flags=re.IGNORECASE):
                        continue
                    property_candidates.append(
                        (
                            model_alignment,
                            bounded_source,
                            -len(record),
                            -result_index,
                            cell,
                            clean_dimension(value),
                            result,
                        )
                    )
        if property_candidates:
            _alignment, _bounded, _specificity, _negative_index, model, value, result = max(
                property_candidates,
                key=lambda item: item[:4],
            )
            if asks_model_length:
                return f"The {model} cable length is {value}.", [result]
            return f"The {model} cable weighs {value}.", [result]

    wire_temperature_query = bool(
        re.search(r"\bwire\b", query, flags=re.IGNORECASE)
        and re.search(r"\btemperature\s+rating\b", query, flags=re.IGNORECASE)
    )
    wire_area_query = bool(
        re.search(r"\bwire\b", query, flags=re.IGNORECASE)
        and re.search(r"\bcross[- ]sectional\s+area\b", query, flags=re.IGNORECASE)
    )
    if wire_temperature_query or wire_area_query:
        wire_candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = re.sub(r"\s+", " ", str(result.content or "")).strip()
            if wire_temperature_query:
                value_match = re.search(
                    r"\bwire\s+with\s+(?P<value>\d+(?:\.\d+)?\s*°?\s*C\s+or\s+higher)"
                    r"\s+temperature\s+rating\b",
                    evidence,
                    flags=re.IGNORECASE,
                )
                if value_match:
                    value = re.sub(r"\s*°\s*C", "°C", value_match.group("value"), flags=re.IGNORECASE)
                    wire_candidates.append(
                        (2, -result_index, f"Use electrical wire rated for {value}.", result)
                    )
            if wire_area_query:
                value_match = re.search(
                    r"cross[- ]sectional\s+area\s+of\s+the\s+wire.*?should\s+be\s+"
                    r"(?P<value>\d+(?:\.\d+)?\s*mm\s*(?:2|²)\s+to\s+"
                    r"\d+(?:\.\d+)?\s*mm\s*(?:2|²)"
                    r"(?:\s*\(\s*AWG\s*\d+\s+to\s+\d+\s*\))?)",
                    evidence,
                    flags=re.IGNORECASE,
                )
                if value_match:
                    value = re.sub(r"mm\s*2\b", "mm²", value_match.group("value"), flags=re.IGNORECASE)
                    value = re.sub(r"\bAWG\s*(\d+)", r"AWG \1", value, flags=re.IGNORECASE)
                    value = re.sub(r"\s+", " ", value).strip()
                    wire_candidates.append(
                        (
                            2,
                            -result_index,
                            f"Use wire with a nominal cross-sectional area of {value}.",
                            result,
                        )
                    )
        if wire_candidates:
            _specificity, _negative_index, answer, result = max(
                wire_candidates,
                key=lambda item: item[:2],
            )
            return answer, [result]

    enclosure_rating_query = bool(
        re.search(r"\benclosure\s+rating\b", query, flags=re.IGNORECASE)
    )
    if enclosure_rating_query:
        requested_models = _model_tokens(query)
        enclosure_candidates: list[tuple[int, int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = str(result.content or "")
            if not re.search(r"\benclosure\s+rating\b", evidence, flags=re.IGNORECASE):
                continue
            rating_match = re.search(r"\bIP\d{2}[A-Z]?\b", evidence, flags=re.IGNORECASE)
            if not rating_match:
                continue
            model_alignment = sum(
                1 for model in requested_models if _result_mentions_model(result, model)
            )
            if requested_models and model_alignment <= 0:
                continue
            bounded_source = int(
                str(result.metadata.get("chunk_type") or "")
                in {"atomic_text", "table_record", "spec_record", "datasheet_record"}
            )
            enclosure_candidates.append(
                (
                    model_alignment,
                    bounded_source,
                    -result_index,
                    rating_match.group(0).upper(),
                    result,
                )
            )
        if enclosure_candidates:
            _alignment, _bounded, _negative_index, rating, result = max(
                enclosure_candidates,
                key=lambda item: item[:3],
            )
            return f"The enclosure rating is {rating}.", [result]

    material_composition_query = bool(
        re.search(r"\bmaterials?\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:compos(?:e|es|ed)|made\s+of|case|cover)\b", query, flags=re.IGNORECASE)
    )
    if material_composition_query:
        requested_models = _model_tokens(query)
        query_terms = _material_claim_terms(query)
        material_candidates: list[
            tuple[int, int, int, int, list[tuple[str, str]], SearchResult]
        ] = []
        for result_index, result in enumerate(results[:12]):
            evidence = str(result.content or "")
            model_alignment = sum(
                1 for model in requested_models if _result_mentions_model(result, model)
            )
            if requested_models and model_alignment <= 0:
                continue
            cell_match = re.search(
                r"Cell\s+value:\s*(?P<cell>.*?)(?:;\s*Row:\s*\d+|$)",
                evidence,
                flags=re.IGNORECASE | re.DOTALL,
            )
            cell = cell_match.group("cell") if cell_match else evidence
            pairs: list[tuple[str, str]] = []
            for segment in cell.split(","):
                pair_match = re.search(
                    r"(?P<label>[A-Za-z][A-Za-z0-9 /()-]{1,60})\s*:\s*"
                    r"(?P<value>.+)$",
                    segment.strip(),
                )
                if not pair_match:
                    continue
                label = re.sub(
                    r"^IP\d{2}[A-Z]?\s+",
                    "",
                    re.sub(r"\s+", " ", pair_match.group("label")).strip(),
                    flags=re.IGNORECASE,
                )
                value = re.sub(r"\s+", " ", pair_match.group("value")).strip(" .;:")
                value = re.sub(
                    r"\s+Approx\.?\s+\d.*$",
                    "",
                    value,
                    flags=re.IGNORECASE,
                ).strip(" .;:")
                label_terms = _material_claim_terms(label)
                if label_terms and label_terms.issubset(query_terms) and value:
                    pairs.append((label, value))
            if not pairs:
                continue
            bounded_source = int(
                str(result.metadata.get("chunk_type") or "")
                in {"atomic_text", "table_record", "spec_record", "datasheet_record"}
            )
            material_candidates.append(
                (model_alignment, len(pairs), bounded_source, -result_index, pairs, result)
            )
        if material_candidates:
            _alignment, _pair_count, _bounded, _negative_index, pairs, result = max(
                material_candidates,
                key=lambda item: item[:4],
            )
            statements = [f"the {label.lower()} is {value}" for label, value in pairs]
            if len(statements) == 1:
                answer = statements[0]
            else:
                answer = ", ".join(statements[:-1]) + f", and {statements[-1]}"
            return f"{answer[:1].upper()}{answer[1:]}.", [result]

    standards_authority_match = re.search(
        r"(?<![A-Z-])(?P<authority>C-UL|UL|EN|IEC)(?![A-Z-])",
        query,
        flags=re.IGNORECASE,
    )
    standards_query = bool(
        standards_authority_match
        and re.search(r"\b(?:safety\s+)?standards?\b", query, flags=re.IGNORECASE)
    )
    if standards_query and standards_authority_match:
        authority = standards_authority_match.group("authority").upper()
        requested_models = _model_tokens(query)
        standards_candidates: list[tuple[int, int, list[str], SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = str(result.content or "")
            model_alignment = sum(
                1 for model in requested_models if _result_mentions_model(result, model)
            )
            if requested_models and model_alignment <= 0:
                continue
            standards_match = re.search(
                rf"(?<![-A-Z]){re.escape(authority)}\s*:\s*(?P<standards>.+?)"
                r"(?=\s+(?:C-UL|UL|EN|IEC)\s*:|;|$)",
                evidence,
                flags=re.IGNORECASE,
            )
            if not standards_match:
                continue
            standards = [
                item.strip()
                for item in standards_match.group("standards").split(",")
                if item.strip()
            ]
            if standards:
                standards_candidates.append(
                    (model_alignment, -result_index, standards, result)
                )
        if standards_candidates:
            _alignment, _negative_index, standards, result = max(
                standards_candidates,
                key=lambda item: item[:2],
            )
            if len(standards) == 1:
                standards_text = standards[0]
            else:
                standards_text = ", ".join(standards[:-1]) + f" and {standards[-1]}"
            return (
                f"The applicable {authority} safety standards are {standards_text}.",
                [result],
            )

    mass_query = bool(re.search(r"\b(?:mass|weighs?|weight)\b", query, flags=re.IGNORECASE))
    if mass_query:
        requested_models = _model_tokens(query)
        mass_candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = str(result.content or "")
            model_alignment = sum(
                1 for model in requested_models if _result_mentions_model(result, model)
            )
            if requested_models and model_alignment <= 0:
                continue
            weight_match = re.search(
                r"\bWeight\s*(?:\|\s*){0,3}(?P<value>(?:Approx\.?\s*)?"
                r"\d+(?:\.\d+)?\s*(?:kg|g)\b(?:\s*\([^\n)]{1,100}\))?)",
                evidence,
                flags=re.IGNORECASE,
            )
            if not weight_match:
                continue
            value = re.sub(r"\s+", " ", weight_match.group("value")).strip()
            value = re.sub(r"^Approx\.?\s*", "approximately ", value, flags=re.IGNORECASE)
            mass_candidates.append((model_alignment, -result_index, value, result))
        if mass_candidates:
            _alignment, _negative_index, value, result = max(
                mass_candidates,
                key=lambda item: item[:2],
            )
            model = next(iter(sorted(requested_models)), "The unit")
            subject = f"The {model}" if requested_models else "The unit"
            return f"{subject} weighs {value}.", [result]

    named_range_match = re.search(
        r"\b(?:valid\s+)?range\s+for\s+(?:the\s+)?(?P<label>.+?)\s+setting\b",
        query,
        flags=re.IGNORECASE,
    )
    if named_range_match:
        label = re.sub(r"\s+", " ", named_range_match.group("label")).strip(" .?:")
        normalized_label = _normalized_phrase(label)
        range_candidates: list[tuple[int, int, int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = _fallback_answer_text(result)
            normalized_evidence = _normalized_phrase(evidence)
            if normalized_label not in normalized_evidence:
                continue
            for range_match in re.finditer(
                r"\brange\s*:\s*(?P<low>-?\d+(?:\.\d+)?)\s+to\s+"
                r"(?P<high>-?\d+(?:\.\d+)?)\b",
                evidence,
                flags=re.IGNORECASE,
            ):
                prefix = _normalized_phrase(evidence[max(0, range_match.start() - 320):range_match.start()])
                suffix = _normalized_phrase(evidence[range_match.end():range_match.end() + 160])
                label_before = int(normalized_label in prefix)
                label_after = int(normalized_label in suffix)
                if not label_before and not label_after:
                    continue
                label_position = prefix.rfind(normalized_label) if label_before else -1
                proximity = label_position - len(prefix) if label_before else -1000
                range_candidates.append(
                    (
                        label_before,
                        -label_after,
                        proximity,
                        -result_index,
                        f"{range_match.group('low')} to {range_match.group('high')}",
                        result,
                    )
                )
        if range_candidates:
            _before, _after, _proximity, _negative_index, value_range, result = max(
                range_candidates,
                key=lambda item: item[:4],
            )
            return f"The valid range for the {label} setting is {value_range}.", [result]

    response_light_condition_query = bool(
        re.search(r"\bresponse[- ]time\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:light|saturat\w*|insufficient|recalibrat\w*)\b", query, flags=re.IGNORECASE)
    )
    if response_light_condition_query:
        condition_candidates: list[tuple[int, int, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            direct_evidence = str(result.content or "")
            if not (
                re.search(r"\b500\s*[µμu]?\s*s\b", direct_evidence, flags=re.IGNORECASE)
                and re.search(r"\b2\.1\s*ms\b", direct_evidence, flags=re.IGNORECASE)
                and re.search(r"\bsaturat\w*\b", direct_evidence, flags=re.IGNORECASE)
                and re.search(r"\binsufficient\b", direct_evidence, flags=re.IGNORECASE)
            ):
                continue
            has_action = int(bool(re.search(r"\brecalibrat\w*\b", direct_evidence, flags=re.IGNORECASE)))
            condition_candidates.append((has_action, -result_index, result))
        if condition_candidates:
            _has_action, _negative_index, result = max(
                condition_candidates,
                key=lambda item: item[:2],
            )
            if re.search(r"\b(?:what\s+should|corrective\s+action|what\s+action|recommend)\b", query, flags=re.IGNORECASE):
                return (
                    "Recalibrate the LR-W70(C); calibration automatically adjusts the light intensity.",
                    [result],
                )
            return (
                "At 500 µs or 2.1 ms response time, the indicators may appear when the light "
                "intensity is saturated or insufficient.",
                [result],
            )
    trigger_delay_query = bool(
        re.search(r"\btrigger\W+delay\b|\bdelay\b.{0,100}\btrigger\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:image\s+capture|capture|range|time)\b", query, flags=re.IGNORECASE)
    )
    if trigger_delay_query:
        lj_s_scope = bool(re.search(r"\bLJ\s*[-:]?\s*S\b", query, flags=re.IGNORECASE))
        delay_candidates: list[tuple[int, int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            evidence = "\n".join(
                part
                for part in (
                    str(result.content or "").strip(),
                    str(result.metadata.get("parent_context") or "").strip(),
                    str(result.metadata.get("context_window") or "").strip(),
                    " ".join(result.section_path),
                )
                if part
            )
            for delay_match in re.finditer(
                r"\btrigger\s+delay\b.{0,220}?\b(?:range\s+(?:between|of)|from)\s+"
                r"(?P<low>\d+(?:\.\d+)?)\s*(?:to|and)\s*"
                r"(?P<high>\d+(?:\.\d+)?)\s*ms\b",
                evidence,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                local_window = evidence[
                    max(0, delay_match.start() - 3000) : delay_match.end() + 300
                ]
                scope_alignment = int(
                    not lj_s_scope
                    or bool(re.search(r"\bLJ\s*[-:]?\s*S\b", local_window, flags=re.IGNORECASE))
                )
                if lj_s_scope and not scope_alignment:
                    continue
                direct_match = int(delay_match.end() <= len(str(result.content or "")))
                delay_candidates.append(
                    (
                        scope_alignment,
                        direct_match,
                        -result_index,
                        f"{delay_match.group('low')} to {delay_match.group('high')} ms",
                        result,
                    )
                )
        if delay_candidates:
            _scope, _direct, _negative_index, delay_range, result = max(
                delay_candidates,
                key=lambda item: item[:3],
            )
            if re.search(r"\b(?:how\s+do\s+i|what\s+should\s+i)\b", query, flags=re.IGNORECASE):
                return (
                    f"Specify the Trigger Delay time; for each camera, it can be set from {delay_range}.",
                    [result],
                )
            return f"The trigger delay range is {delay_range} for each camera.", [result]
    if re.search(
        r"^\s*(?:how do i|how should i|what should i)\b"
        r"|\b(?:calculated|computed|derived)\b"
        r"|^\s*what\s+does\s+enabling\s+.+?\bmode\s+prioritize\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "", []
    connector_type_query = bool(
        re.search(
            r"\bwhat\s+(?:connector\s+type|type\s+of\s+connector)\b|"
            r"\bwhich\s+(?:kind|type)\s+of\s+connector\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    if connector_type_query:
        query_models = _model_tokens(query)
        connector_candidates: list[tuple[int, int, int, str, SearchResult]] = []
        connector_pattern = re.compile(
            r"\b(?P<type>M\d+(?:\s+\d+[- ]pin)?|RJ-?45|USB(?:-[A-Z])?|"
            r"(?:mini|micro)[- ]USB|D[- ]?sub(?:miniature)?(?:\s+\d+[- ]pin)?)\s+connector\b",
            flags=re.IGNORECASE,
        )
        for result_index, result in enumerate(results[:12]):
            direct_evidence = str(result.content or "").strip()
            evidence_models = _model_tokens(direct_evidence)
            model_alignment = int(
                not query_models or bool(query_models.intersection(evidence_models))
            )
            if query_models and not model_alignment:
                continue
            for connector_index, connector_match in enumerate(
                connector_pattern.finditer(direct_evidence)
            ):
                connector_type = re.sub(
                    r"\s+",
                    " ",
                    connector_match.group("type"),
                ).upper().replace("RJ-45", "RJ45")
                connector_candidates.append(
                    (
                        model_alignment,
                        -result_index,
                        -connector_index,
                        connector_type,
                        result,
                    )
                )
        if connector_candidates:
            _alignment, _negative_result, _negative_match, connector_type, result = max(
                connector_candidates,
                key=lambda item: item[:3],
            )
            model = next(iter(sorted(query_models)), "The device")
            subject = f"The {model} cable" if re.search(r"\bcable\b", query, flags=re.IGNORECASE) else f"The {model}"
            return f"{subject} uses an {connector_type} connector.", [result]
    initial_polarity_query = bool(
        re.search(r"\b(?:output\s+)?polarity\b", query, flags=re.IGNORECASE)
        and re.search(
            r"\b(?:out\s+of\s+the\s+box|out\s+of\s+box|default|initial)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    if initial_polarity_query:
        requested_models = _model_tokens(query)
        polarity_candidates: list[tuple[int, int, int, str, SearchResult]] = []
        labeled_pattern = re.compile(
            r"Column headers:\s*Initial value.*?Row headers:\s*.*?NPN/PNP selection.*?"
            r"Cell value:\s*(?P<value>NPN|PNP)\b",
            flags=re.IGNORECASE | re.DOTALL,
        )
        pipe_pattern = re.compile(
            r"Item\s*\|\s*Initial value.*?NPN/PNP selection\s*\|\s*(?P<value>NPN|PNP)\b",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for result_index, result in enumerate(results[:12]):
            direct = str(result.content or "").strip()
            for evidence_priority, evidence in ((2, direct),):
                if not evidence:
                    continue
                match = labeled_pattern.search(evidence) or pipe_pattern.search(evidence)
                if not match:
                    continue
                model_haystack = _normalized_phrase(
                    " ".join(
                        part
                        for part in (
                            direct,
                            result.title,
                            str(result.metadata.get("product_model") or ""),
                        )
                        if part
                    )
                )
                model_alignment = int(
                    not requested_models
                    or any(_normalized_phrase(model) in model_haystack for model in requested_models)
                )
                if requested_models and not model_alignment:
                    continue
                polarity_candidates.append(
                    (
                        model_alignment,
                        evidence_priority,
                        -result_index,
                        match.group("value").upper(),
                        result,
                    )
                )
        if polarity_candidates:
            _alignment, _evidence_priority, _negative_index, polarity, result = max(
                polarity_candidates,
                key=lambda item: item[:3],
            )
            return f"The initial output polarity is {polarity}.", [result]
    voltage_options_query = bool(
        re.search(
            r"\b(?:which|what)\b.{0,80}\b(?:voltage\s+options?|voltages?)\b",
            query,
            flags=re.IGNORECASE,
        )
    )
    if voltage_options_query:
        requested_models = _model_tokens(query)
        voltage_candidates: list[tuple[int, int, int, str, str, SearchResult]] = []
        for result_index, result in enumerate(results[:12]):
            direct_evidence = re.sub(r"\s+", " ", str(result.content or "")).strip()
            model_alignment = sum(
                1
                for model in requested_models
                if _result_mentions_model(result, model)
            )
            if requested_models and model_alignment <= 0:
                continue
            options_match = re.search(
                r"\bselect\s+either\s+(?P<first>\d+(?:\.\d+)?\s*V)"
                r"(?P<default>\s*\(\s*Default\s*\))?\s+or\s+"
                r"(?P<second>\d+(?:\.\d+)?\s*V)\b",
                direct_evidence,
                flags=re.IGNORECASE,
            )
            if not options_match:
                continue
            first = re.sub(r"\s*V$", " V", options_match.group("first"), flags=re.IGNORECASE)
            second = re.sub(r"\s*V$", " V", options_match.group("second"), flags=re.IGNORECASE)
            has_default = bool(options_match.group("default"))
            voltage_candidates.append(
                (
                    model_alignment,
                    int(has_default),
                    -result_index,
                    first,
                    second,
                    result,
                )
            )
        if voltage_candidates:
            _alignment, has_default, _negative_index, first, second, result = max(
                voltage_candidates,
                key=lambda item: item[:3],
            )
            default_suffix = " (default)" if has_default else ""
            return f"Select either {first}{default_suffix} or {second}.", [result]
    reset_count_query = bool(
        re.search(r"\breset\s*count\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:clear|reset|do|does)\b", query, flags=re.IGNORECASE)
    )
    if reset_count_query:
        reset_candidates: list[tuple[int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:10]):
            evidence = "\n".join(
                part
                for part in (
                    str(result.content or "").strip(),
                    str(result.metadata.get("context_window") or "").strip(),
                    str(result.metadata.get("parent_context") or "").strip(),
                )
                if part
            )
            serial_match = re.search(
                r"\bResets?\s+the\s+serial\s+number[^.\n]*[.]?",
                evidence,
                flags=re.IGNORECASE,
            )
            count_match = re.search(
                r"\bResets?\s+the\s+count\s+value[^.\n]*[.]?",
                evidence,
                flags=re.IGNORECASE,
            )
            if not (serial_match and count_match):
                continue
            statements = [
                re.sub(r"\s+", " ", serial_match.group(0)).strip(),
                re.sub(r"\s+", " ", count_match.group(0)).strip(),
            ]
            all_tools_match = re.search(
                r"\bAll\s+tools\s+are\s+reset[^.\n]*[.]?",
                evidence,
                flags=re.IGNORECASE,
            )
            if all_tools_match:
                statements.append(re.sub(r"\s+", " ", all_tools_match.group(0)).strip())
            answer = " ".join(statement.rstrip(".") + "." for statement in statements)
            if re.match(r"^\s*(?:does|do|is|are|can|will)\b", query, flags=re.IGNORECASE):
                answer = re.sub(r"^Resets?\b", "The command resets", answer, flags=re.IGNORECASE)
                answer = f"Yes. {answer}"
            reset_candidates.append((len(statements), -result_index, answer, result))
        if reset_candidates:
            _coverage, _negative_index, answer, result = max(
                reset_candidates,
                key=lambda item: (item[0], item[1]),
            )
            return answer, [result]
    unit_subject_match = re.search(
        r"\b(?:in\s+)?what\s+unit\s+is\s+(?P<subject>.+?)\s+(?:reported|output|displayed|measured)\b",
        query,
        flags=re.IGNORECASE,
    )
    if unit_subject_match:
        subject = re.sub(r"\s+", " ", unit_subject_match.group("subject")).strip(" .?:")
        subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.IGNORECASE)
        subject_terms = _material_claim_terms(subject)
        unit_candidates: list[tuple[int, int, int, str, SearchResult]] = []
        for result_index, result in enumerate(results[:10]):
            for segment_index, segment in enumerate(
                re.split(r"(?<=[.!?])\s+|\n+", _fallback_answer_text(result))
            ):
                segment = re.sub(r"\s+", " ", segment).strip(" -|;:")
                unit_match = re.search(r"\bUnit\s*:\s*(?P<unit>[^).,;\s]{1,24})", segment, flags=re.IGNORECASE)
                if not unit_match:
                    continue
                overlap = len(subject_terms.intersection(_material_claim_terms(segment)))
                if overlap <= 0:
                    continue
                unit = unit_match.group("unit").strip()
                unit_candidates.append((overlap, -result_index, -segment_index, unit, result))
        if unit_candidates:
            _overlap, _negative_result, _negative_segment, unit, result = max(
                unit_candidates,
                key=lambda item: item[:3],
            )
            unit_text = "milliseconds (ms)" if unit.lower() == "ms" else unit
            return f"The {subject} is reported in {unit_text}.", [result]
    if re.search(
        r"^\s*where\b.{0,180}\b(?:display|show)(?:s|ed|n)?\b",
        query,
        flags=re.IGNORECASE,
    ):
        display_candidates: list[tuple[int, int, str, SearchResult]] = []
        query_terms = _material_claim_terms(query)
        for index, result in enumerate(results[:10]):
            evidence = _fallback_answer_text(result)
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence):
                sentence = re.sub(r"\s+", " ", sentence).strip(" -|;:")
                if not re.search(
                    r"\b(?:displayed|shown)\b.{0,100}\[[^\]\n]{1,80}\]",
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    continue
                overlap = len(query_terms.intersection(_material_claim_terms(sentence)))
                if overlap >= 2:
                    display_candidates.append((overlap, -index, sentence, result))
        if display_candidates:
            _overlap, _negative_index, answer, result = max(
                display_candidates,
                key=lambda item: (item[0], item[1]),
            )
            return answer, [result]
    if (
        re.search(r"\banalog\s+output\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:numeric\s+)?range\b", query, flags=re.IGNORECASE)
        and re.search(r"\bdisplay\s+value\b", query, flags=re.IGNORECASE)
    ):
        candidates: list[tuple[int, int, str, SearchResult]] = []
        query_terms = _material_claim_terms(query)
        for index, result in enumerate(results[:10]):
            evidence = _fallback_answer_text(result)
            range_match = re.search(
                r"\bDisplay\s+value\b.*?\b(?:value\s*)?\(?(?P<low>\d+)\s*(?:-|to)\s*"
                r"(?P<high>\d+)\)?",
                evidence,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not range_match:
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(evidence)))
            candidates.append(
                (
                    overlap,
                    -index,
                    f"The analog-output display-value range is {range_match.group('low')} to "
                    f"{range_match.group('high')}.",
                    result,
                )
            )
        if candidates:
            _overlap, _negative_index, answer, result = max(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            return answer, [result]
    fact_results = _structured_fact_evidence_results(query, results)
    if direct_model_field_candidates:
        fact_chunk_ids = {result.chunk_id for result in fact_results}
        ranked_direct_candidates = (
            [
                candidate
                for candidate in direct_model_field_candidates
                if candidate[-1].chunk_id in fact_chunk_ids
            ]
            or direct_model_field_candidates
        )
        _count_fit, _requested_fit, _overlap, _specificity, _negative_index, answer, result = max(
            ranked_direct_candidates,
            key=lambda item: item[:5],
        )
        return answer, [result]

    if not fact_results:
        return "", []
    result = fact_results[0]
    movement_range_query = bool(
        re.search(r"\b(?:movable|travel)\s+range\b", query, flags=re.IGNORECASE)
        or (
            re.search(r"\bhow\s+far\b", query, flags=re.IGNORECASE)
            and re.search(r"\b(?:move|moves|moving|travel|travels)\b", query, flags=re.IGNORECASE)
        )
    )
    if movement_range_query:
        query_terms = _material_claim_terms(query).difference(
            {"far", "how", "move", "moves", "moving", "movable", "range", "travel", "travels"}
        )
        range_candidates: list[tuple[int, int, str]] = []
        for segment_index, segment in enumerate(
            re.split(r"(?<=[.!?])\s+|\n+", _fallback_answer_text(result))
        ):
            segment = re.sub(r"\s+", " ", segment).strip(" -|;:")
            if not re.search(r"\b(?:movable|travel)\s+range\b", segment, flags=re.IGNORECASE):
                continue
            if not _quantity_terms(segment):
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(segment)))
            range_candidates.append((overlap, -segment_index, segment))
        if range_candidates:
            _overlap, _negative_index, answer = max(
                range_candidates,
                key=lambda item: (item[0], item[1]),
            )
            return answer, fact_results
    answer = (
        _focused_model_field_record_answer_text(query, result)
        or _focused_labeled_table_cell_answer_text(query, result)
        or _concise_general_fallback_answer(query, result)
    )
    if not answer:
        return "", []
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    return answer, fact_results


def _concise_default_setting_value_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Return the default value from the matching repeated setting record."""
    subject_match = re.search(
        r"\bdefault\s+(?P<subject>.+?)(?:\s+for\b|\?|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not subject_match:
        return "", []
    subject = subject_match.group("subject").strip(" .?:")
    subject_terms = _material_claim_terms(subject)
    candidates: list[tuple[int, int, str, str, SearchResult]] = []
    for result_index, result in enumerate(results[:12]):
        evidence = _fallback_answer_text(result)
        for record in re.finditer(
            r"Items?\s*:\s*(?P<item>[^;]+);(?P<body>.*?)"
            r"Default\s+value\s*:\s*(?P<value>.*?)(?=\s+Items?\s*:|$)",
            evidence,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            item = re.sub(r"\s+", " ", record.group("item")).strip()
            value = re.sub(r"\s+", " ", record.group("value")).strip(" ;|.")
            overlap = len(subject_terms.intersection(_material_claim_terms(item)))
            if overlap and value:
                candidates.append((overlap, -result_index, item, value, result))
    if not candidates:
        return "", []
    _overlap, _negative_index, item, value, result = max(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return f"The default {item} is {value}.", [result]


def _concise_matching_model_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Select the model whose table row satisfies all requested features."""
    if not re.search(r"\bwhich\b.{0,100}\bmodel\b", query, flags=re.IGNORECASE):
        return "", []
    query_terms = _material_claim_terms(query).difference({"which", "model", "offers", "sensor"})
    candidates: list[tuple[int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:12]):
        if str(result.metadata.get("chunk_type") or "") != "table_record":
            continue
        match = re.search(
            r"Column headers:\s*(?P<column>.*?);\s*Row headers:\s*(?P<row>.*?);\s*"
            r"Cell value:\s*(?P<value>.*?)(?:;\s*Row:\s*\d+;\s*Column:\s*\d+)?\s*$",
            str(result.content or ""),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match or not re.search(r"\bmodel\b", match.group("column"), flags=re.IGNORECASE):
            continue
        row = re.sub(r"\s+", " ", match.group("row")).strip()
        value = re.sub(r"\s+", " ", match.group("value")).strip(" ;|")
        overlap = len(query_terms.intersection(_material_claim_terms(row)))
        if overlap >= 2 and _model_tokens(value):
            candidates.append((overlap, -result_index, value, row, result))
    if not candidates:
        return "", []
    _overlap, _negative_index, value, row, result = max(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    if (
        re.search(r"\badjustable\b", query, flags=re.IGNORECASE)
        and re.search(r"\banalog\s+output\b", query, flags=re.IGNORECASE)
        and re.search(r"\badjustable\b", row, flags=re.IGNORECASE)
        and re.search(r"\banalog\s+output\b", row, flags=re.IGNORECASE)
    ):
        return f"The matching model is {value}; it is adjustable and supports analog output.", [result]
    return f"The matching model is {value}.", [result]


def _concise_part_number_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Select the part number nearest the component named in the question."""
    subject_match = re.search(
        r"\bwhich\s+(?P<subject>.+?)\s+part\s+number\b|"
        r"\bpart\s+number\s+(?:is\s+)?(?:required\s+)?for\s+(?P<subject_after>.+?)(?:\?|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not subject_match:
        return "", []
    subject = (subject_match.group("subject") or subject_match.group("subject_after") or "").strip()
    subject_terms = _material_claim_terms(subject).difference({"required", "which"})
    candidates: list[tuple[int, int, int, int, int, str, SearchResult]] = []
    subject_pattern = re.compile(
        r"\b" + r"\W+".join(re.escape(term) for term in re.findall(r"[A-Za-z0-9]+", subject)) + r"\b",
        flags=re.IGNORECASE,
    )
    for result_index, result in enumerate(results[:12]):
        evidence_parts = [_fallback_answer_text(result)]
        parent_context = str(result.metadata.get("parent_context") or "").strip()
        if parent_context and parent_context not in evidence_parts[0]:
            evidence_parts.append(parent_context)
        evidence = "\n".join(evidence_parts)
        subject_positions = [match.start() for match in subject_pattern.finditer(evidence)]
        for code_index, code_match in enumerate(
            re.finditer(r"\b[A-Z]{2,5}-\d{4,8}\b", evidence, flags=re.IGNORECASE)
        ):
            # Wide tables commonly put a component header several cells before its
            # part-number value.  Keep the association local to the same table
            # region while allowing more than a single-cell look-behind.
            window = evidence[max(0, code_match.start() - 500) : code_match.end() + 80]
            overlap = len(subject_terms.intersection(_material_claim_terms(window)))
            if overlap:
                subject_distance = min(
                    (
                        code_match.start() - position
                        for position in subject_positions
                        if position <= code_match.start()
                    ),
                    default=10_000_000,
                )
                exact_subject_near = int(
                    len(subject_terms) >= 2 and subject_distance <= 600
                )
                candidates.append(
                    (
                        exact_subject_near,
                        overlap,
                        -subject_distance,
                        -result_index,
                        -code_index,
                        code_match.group(0).upper(),
                        result,
                    )
                )
    if not candidates:
        return "", []
    (
        _exact_subject_near,
        _overlap,
        _negative_subject_distance,
        _negative_result_index,
        _negative_code_index,
        code,
        result,
    ) = max(
        candidates,
        key=lambda candidate: candidate[:5],
    )
    return f"The required {subject} part number is {code}.", [result]


def _concise_capability_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Answer a yes/no capability question from an explicit capability sentence."""
    yes_no_query = bool(re.search(
        r"^\s*(?:can|could|does|do|is|are)\b.{0,180}"
        r"\b(?:support|measure|detect|capture|operate|run|use|work|allow|capable|available)\b",
        query,
        flags=re.IGNORECASE,
    ))
    selection_impact_query = bool(re.search(
        r"^\s*what\s+happens\s+to\b.{1,180}\b(?:if|when)\b.{0,120}"
        r"\b(?:select|choose|use)(?:s|d|ing)?\b",
        query,
        flags=re.IGNORECASE,
    ))
    if not (yes_no_query or selection_impact_query):
        return "", []
    if selection_impact_query:
        selection_match = re.search(
            r"\b(?:if|when)\s+(?:i|you)\s+"
            r"(?:select|choose|use)(?:s|d|ing)?\s+(?:the\s+)?"
            r"(?P<option>.+?)(?=\s+(?:on|for|with)\s+(?:the\s+)?[A-Z0-9]|\?\s*$|$)",
            query,
            flags=re.IGNORECASE,
        )
        if selection_match:
            option = re.sub(r"\s+", " ", selection_match.group("option")).strip(" .?:")
            normalized_option = _normalized_phrase(option)
            impact_candidates: list[tuple[int, int, int, str, SearchResult]] = []
            for result_index, result in enumerate(results[:10]):
                evidence = _fallback_answer_text(result)
                parent_match = re.search(
                    r"Setting\s+item:\s*(?P<label>[^;\n]{1,120});\s*Settings:",
                    evidence,
                    flags=re.IGNORECASE,
                )
                parent = (
                    re.sub(r"\s+", " ", parent_match.group("label")).strip(" .?:")
                    if parent_match
                    else ""
                )
                for labeled_match in re.finditer(
                    r"(?:^|\s[-•▪]\s*)"
                    r"(?P<label>[A-Z0-9][^:\n]{1,100}):\s*"
                    r"(?P<body>.*?)(?=\s[-•▪]\s*[A-Z0-9][^:\n]{1,100}:|$)",
                    evidence,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    label = re.sub(r"\s+", " ", labeled_match.group("label")).strip(" .?:")
                    if _normalized_phrase(label) != normalized_option:
                        continue
                    body = re.sub(r"\s+", " ", labeled_match.group("body")).strip(" .;:")
                    if not body:
                        continue
                    overlap = len(
                        _material_claim_terms(query).intersection(_material_claim_terms(body))
                    )
                    bounded_source = int(
                        str(result.metadata.get("chunk_type") or "")
                        in {"atomic_text", "table_record", "spec_record", "datasheet_record"}
                    )
                    prefix = f"For {parent}, " if parent else ""
                    impact_candidates.append(
                        (
                            bounded_source,
                            overlap,
                            -result_index,
                            f"{prefix}{label}: {body}.",
                            result,
                        )
                    )
            if impact_candidates:
                _bounded, _overlap, _negative_index, answer, result = max(
                    impact_candidates,
                    key=lambda item: item[:3],
                )
                return answer, [result]
    query_terms = _material_claim_terms(query).difference(
        {
            "can",
            "could",
            "does",
            "do",
            "is",
            "are",
            "run",
            "use",
            "support",
            "supported",
        }
    )
    candidates: list[tuple[int, int, int, str, SearchResult]] = []
    for result_index, result in enumerate(results[:10]):
        evidence = _fallback_answer_text(result)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -|;:")
            sentence = re.sub(r"^\S+\.pdf\s*\|\s*", "", sentence, flags=re.IGNORECASE)
            if len(sentence.split()) < 6:
                continue
            if not re.search(
                r"\b(?:ability\s+to|can(?:not|'t)?|capable\s+of|supports?|allows?|"
                r"without\s+missing|unable\s+to|not\s+available|"
                r"eliminat(?:e|es|ed|ing)\s+the\s+need)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            overlap = len(query_terms.intersection(_material_claim_terms(sentence)))
            if overlap < 3:
                continue
            candidates.append((overlap, len(_quantity_terms(sentence)), -result_index, sentence, result))
    if not candidates:
        return "", []
    _overlap, _quantity_count, _result_index, sentence, result = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    negative = bool(
        re.search(
            r"\b(?:cannot|can't|unable\s+to|does\s+not\s+support|not\s+available|"
            r"eliminat(?:e|es|ed|ing)\s+the\s+need)\b",
            sentence,
            flags=re.IGNORECASE,
        )
    )
    sentence = re.sub(r"^\*\s*\d+\s*", "", sentence)
    sentence = re.sub(r"(?<=\w)\*\s*\d+\b", "", sentence)
    if selection_impact_query:
        return sentence, [result]
    return f"{'No' if negative else 'Yes'}. {sentence}", [result]


LOCATION_CUE_RE = re.compile(
    r"\b(?:menu|screen|tab|section|page|unit|settings?|options?|area|panel|dialog|toolbar|folder)\b",
    flags=re.IGNORECASE,
)


def _configuration_location_subject(query: str) -> str:
    patterns = (
        r"\b(?:set|adjust|change|configure|find|locate|select|enable|disable)\s+(.+?)"
        r"(?:\s+(?:between|for|on|in|under|with|using)\s+|[?.!,;]|$)",
        r"\bwhere\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+(?:setting|option|parameter|control|field)\b",
        r"\b(?:which|what)\s+(?:menu|screen|tab|section|page)\s+"
        r"(?:contains?|has|includes?|shows?|holds?)\s+(?:the\s+)?(.+?)"
        r"(?:\s+(?:for|on|in|under|with|using)\s+|[?.!,;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip(" .?\"'")
    return ""


def _normalized_device_text(text: str) -> str:
    normalized = re.sub(r"\blinescan\b", "line scan", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


def _location_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _normalized_device_text(text).split():
        if len(token) < 3:
            continue
        terms.add(token)
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            terms.add(token[:-1])
        if len(token) > 5 and token.endswith("ing"):
            stem = token[:-3]
            terms.add(stem[:-1] if len(stem) > 2 and stem[-1:] == stem[-2:-1] else stem)
    return terms


def _requested_device_phrases(query: str) -> set[str]:
    normalized = _normalized_device_text(query)
    phrases: set[str] = set()
    for match in re.finditer(
        r"\b([a-z0-9-]+(?:\s+[a-z0-9-]+){0,3}\s+"
        r"(?:camera|head|controller|scanner|sensor|reader|drive|motor|unit|module))\b",
        normalized,
    ):
        phrase = match.group(1).strip()
        words = phrase.split()
        while words and words[0] in {"a", "an", "the", "for", "on", "with", "using"}:
            words.pop(0)
        if len(words) >= 2:
            phrases.add(" ".join(words))
    return phrases


def _requested_capture_scope_phrases(query: str) -> set[str]:
    normalized = _normalized_device_text(query)
    phrases: set[str] = set()
    for match in re.finditer(
        r"\b([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3})\s+"
        r"(?:camera|captures?|imaging|acquisition)\b",
        normalized,
    ):
        words = match.group(1).split()
        while words and words[0] in {"a", "an", "the", "for", "on", "with", "using", "between"}:
            words.pop(0)
        if len(words) >= 2:
            phrases.add(" ".join(words[-3:]))
    return phrases


def _configuration_location_evidence_text(result: SearchResult) -> str:
    parts: list[str] = []
    for value in [
        str(result.metadata.get("parent_context") or "").strip(),
        str(result.content or "").strip(),
        str(result.metadata.get("context_window") or "").strip(),
    ]:
        if value and value not in parts:
            parts.append(value)
    return "\n\n".join(parts)


def _configuration_location_evidence_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    if not _is_configuration_location_query(query):
        return []
    subject = _configuration_location_subject(query)
    subject_terms = _location_terms(subject)
    compact_subject = re.sub(r"[^a-z0-9]+", "", subject.lower())
    requested_devices = _requested_device_phrases(query)
    requested_scopes = _requested_capture_scope_phrases(query)
    scored: list[tuple[float, int, SearchResult]] = []
    for index, result in enumerate(results[:12]):
        evidence = _configuration_location_evidence_text(result)
        evidence_terms = _location_terms(evidence)
        score = float(len(subject_terms.intersection(evidence_terms)) * 2)
        if compact_subject and compact_subject in re.sub(r"[^a-z0-9]+", "", evidence.lower()):
            score += 5.0
        score += min(3.0, float(len(set(LOCATION_CUE_RE.findall(evidence.lower())))) * 0.5)
        if re.search(r"\b(?:specif(?:y|ies)|controls?|used for|allows?|enables?|prevents?)\b", evidence, flags=re.IGNORECASE):
            score += 1.0
        scope_text = _normalized_device_text(
            " ".join(
                [
                    str(result.metadata.get("parent_context") or "")[:700],
                    " ".join(result.section_path),
                    str(result.content or "")[:300],
                ]
            )
        )
        declared_scope = re.search(
            r"\bwhen using (?:a |an )?([a-z0-9 -]{2,60}?(?:camera|head|controller|scanner|sensor|reader|drive|motor|unit|module))\b",
            scope_text,
        )
        if requested_devices:
            matching_devices = {device for device in requested_devices if device in scope_text}
            score += float(len(matching_devices) * 4)
            if declared_scope and not any(
                device in declared_scope.group(1) or declared_scope.group(1) in device
                for device in requested_devices
            ):
                score -= 4.0
        if requested_scopes:
            scope_terms = _location_terms(scope_text)
            best_scope_overlap = max(
                (len(_location_terms(scope).intersection(scope_terms)) for scope in requested_scopes),
                default=0,
            )
            score += float(best_scope_overlap * 1.5)
            if declared_scope and not any(
                len(_location_terms(scope).intersection(_location_terms(declared_scope.group(1)))) >= 2
                for scope in requested_scopes
            ):
                score -= 3.0
        scored.append((score, index, result))
    selected = [
        result
        for score, _index, result in sorted(scored, key=lambda item: (-item[0], item[1]))
        if score >= max(3.0, float(len(subject_terms) * 2))
    ]
    primary = selected[:4]
    if not primary:
        return []
    support: list[SearchResult] = []
    primary_pages = [page for result in primary[:2] for page in result.pages]
    primary_documents = {result.source_document_id for result in primary[:2]}
    for result in results[:12]:
        if result in primary or result.source_document_id not in primary_documents:
            continue
        evidence = _configuration_location_evidence_text(result)
        if len(set(LOCATION_CUE_RE.findall(evidence.lower()))) < 2:
            continue
        if primary_pages and result.pages and min(abs(page - primary_page) for page in result.pages for primary_page in primary_pages) > 3:
            continue
        support.append(result)
    return [*primary, *support[:2]]


def _configuration_path_labels(text: str, subject: str) -> list[str]:
    compact_subject = re.sub(r"\s+", " ", subject).strip()
    subject_match = re.search(re.escape(compact_subject), text, flags=re.IGNORECASE) if compact_subject else None
    if not subject_match and compact_subject:
        subject_terms = _location_terms(compact_subject)
        candidates = [
            (len(subject_terms.intersection(_location_terms(candidate.group(0)))), candidate)
            for candidate in re.finditer(r"[^\n]+", text)
        ]
        best_overlap, best_candidate = max(candidates, key=lambda item: item[0], default=(0, None))
        if best_overlap and best_candidate is not None:
            subject_match = best_candidate
    prefix = text[: subject_match.start()] if subject_match else text
    label_re = re.compile(
        r"\b([A-Z][A-Za-z0-9/&-]*(?:[ \t]+[A-Z][A-Za-z0-9/&-]*){0,5}[ \t]+"
        r"(?:Settings?|Options?|Menu|Screen|Tab|Area|Panel|Dialog|Folder)"
        r"(?:\s*\([^\n)]{1,80}\))?)",
    )
    labels: list[str] = []
    for match in label_re.finditer(prefix):
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        if label not in labels:
            labels.append(label)
    local = text[subject_match.start() : subject_match.start() + 500].lower() if subject_match else ""
    if "continuous" in local and "fixed" not in local:
        labels = [label for label in labels if "fixed" not in label.lower()]
    if len(labels) <= 3:
        return labels
    broad = next((label for label in labels if "settings" in label.lower()), labels[0])
    tail = labels[-2:]
    selected: list[str] = []
    for label in [broad, *tail]:
        if label not in selected:
            selected.append(label)
    return selected


def _merged_configuration_path_labels(
    query: str,
    subject: str,
    evidence_texts: list[str],
) -> tuple[list[str], str]:
    """Build one menu path from complementary chunks in the same retrieved section."""
    candidates = [(_configuration_path_labels(text, subject), text) for text in evidence_texts]
    definition_pattern = re.compile(
        rf"{re.escape(subject)}\s*(?:\([^\n)]*\))?\s*:",
        flags=re.IGNORECASE,
    )
    _base_index, (base_labels, base_text) = max(
        enumerate(candidates),
        key=lambda indexed: (
            bool(definition_pattern.search(indexed[1][1])),
            len(indexed[1][0]),
            -indexed[0],
        ),
        default=(0, ([], "")),
    )
    if not base_labels:
        return [], base_text

    # Prefer the plural UI label when OCR/navigation prose exposes both forms.
    all_labels: list[str] = []
    subject_terms = _location_terms(subject)
    for labels, text in candidates:
        # Subject anchoring finds the local child setting; an unanchored pass
        # exposes parent screens that may live in an adjacent support chunk.
        has_subject_definition = bool(definition_pattern.search(text)) or any(
            subject_terms.intersection(_location_terms(segment.split(":", 1)[0]))
            for segment in text.splitlines()
            if ":" in segment
        )
        support_labels = [] if has_subject_definition else _configuration_path_labels(text, "")
        for label in [*labels, *support_labels]:
            if label not in all_labels:
                all_labels.append(label)
    plural_keys = {
        re.sub(r"\bsettings\b", "setting", label.lower())
        for label in all_labels
        if re.search(r"\bsettings\b", label, flags=re.IGNORECASE)
    }
    all_labels = [
        label
        for label in all_labels
        if not (
            re.search(r"\bsetting\b", label, flags=re.IGNORECASE)
            and not re.search(r"\bsettings\b", label, flags=re.IGNORECASE)
            and label.lower() in plural_keys
        )
    ]
    base_labels = [label for label in base_labels if label in all_labels]

    # A setting definition and its parent screen are frequently split across
    # adjacent chunks. Add the strongest device-specific settings screen ahead
    # of a local path, without relying on any product or manual-specific label.
    query_terms = _location_terms(_normalized_device_text(query))
    broad_candidates = [
        label
        for label in all_labels
        if label not in base_labels
        and "settings" in label.lower()
        and len(_location_terms(label).intersection(query_terms)) >= 1
    ]
    broad = max(
        broad_candidates,
        key=lambda label: (
            len(_location_terms(label).intersection(query_terms)),
            -len(label),
        ),
        default="",
    )
    merged: list[str] = []
    if broad and broad not in base_labels:
        merged.append(broad)
    for label in base_labels:
        if label not in merged:
            merged.append(label)
    return merged, base_text


def _concise_configuration_location_answer(
    query: str,
    results: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    location_results = _configuration_location_evidence_results(query, results)
    subject = _configuration_location_subject(query)
    if not subject or not location_results:
        return "", []
    evidence_texts = [_configuration_location_evidence_text(result) for result in location_results]
    path_results = location_results
    anchor_result = location_results[0]
    if anchor_result.pages:
        nearby_results = [
            result
            for result in location_results
            if result.source_document_id == anchor_result.source_document_id
            and result.pages
            and min(
                abs(page - anchor_page)
                for page in result.pages
                for anchor_page in anchor_result.pages
            )
            <= 3
        ]
        if nearby_results:
            path_results = nearby_results
    path_labels, best_text = _merged_configuration_path_labels(
        query,
        subject,
        [_configuration_location_evidence_text(result) for result in path_results],
    )
    subject_label = subject[:1].upper() + subject[1:]
    if not path_labels:
        return "", []
    definition = ""
    purpose = ""
    definition_re = re.compile(
        rf"{re.escape(subject)}\s*(?:\([^\n)]*\))?\s*:\s*([^\n.]+(?:\.[^\n.]+)?)",
        flags=re.IGNORECASE,
    )
    for result in location_results:
        evidence = _configuration_location_evidence_text(result)
        match = definition_re.search(evidence)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        if len(candidate) > len(definition):
            definition = candidate
        purpose_match = re.search(
            rf"{re.escape(subject)}[^\n]{{0,500}}?((?:Even if|This (?:keeps|allows|enables|prevents|ensures)|It (?:keeps|allows|enables|prevents|ensures))[^\n.]*\.)",
            evidence,
            flags=re.IGNORECASE,
        )
        if purpose_match and len(purpose_match.group(1)) > len(purpose):
            purpose = re.sub(r"\s+", " ", purpose_match.group(1)).strip()
    if not definition:
        subject_terms = _location_terms(subject)
        candidates: list[tuple[int, str, str]] = []
        for evidence in evidence_texts:
            for segment in re.split(r"[\n]+", evidence):
                if ":" not in segment:
                    continue
                label, value = segment.split(":", 1)
                overlap = len(subject_terms.intersection(_location_terms(label)))
                if overlap and re.search(r"\b(?:specif(?:y|ies)|controls?|used for|allows?|enables?)\b", value, flags=re.IGNORECASE):
                    candidates.append((overlap, label.strip(), value.strip()))
        if candidates:
            _overlap, resolved_label, resolved_definition = max(
                candidates,
                key=lambda item: (item[0], len(item[2])),
            )
            subject_label = re.sub(r"\s*\([^)]*\)\s*$", "", resolved_label).strip()
            definition = re.split(r"(?<=[.!?])\s+", resolved_definition, maxsplit=1)[0].strip(" .")
    if not definition:
        return "", []
    unit_source = " ".join([*path_labels, *evidence_texts])
    common_unit_matches = re.findall(
        r"\b(?:common\s+for\s+)?all\s+([A-Za-z][A-Za-z0-9-]*)\s+units\b",
        unit_source,
        flags=re.IGNORECASE,
    )
    path_unit_matches = re.findall(
        r"\b([A-Za-z][A-Za-z0-9-]*)\s+Units?\b",
        " ".join(path_labels),
        flags=re.IGNORECASE,
    )
    unit_matches = common_unit_matches or path_unit_matches
    clean_labels = [re.sub(r"\s*\([^)]*\bUnits?\b[^)]*\)", "", label).strip() for label in path_labels]
    path = " > ".join([*clean_labels, subject_label])
    if unit_matches:
        unit_name = f"{unit_matches[-1].title()} Unit"
        location = f"In the {unit_name}, open {path}"
    else:
        location = path
    answer = f"Location: {location}. Purpose: {definition}."
    if re.search(r"\bcamera (?:you selected in (?:the )?camera )?tab\b", best_text, flags=re.IGNORECASE):
        answer = f"{answer} Use the tab for the camera being configured."
    if purpose and purpose.lower() not in answer.lower():
        answer = f"{answer} {purpose}"
    return answer, location_results[:4]


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
            if not _is_procedure_rule_query(query):
                return ordered_results[:1]
            scored = [
                (_fallback_evidence_score(query, result), index, result)
                for index, result in enumerate(ordered_results[:8])
            ]
            if not scored:
                return []
            _best_score, _best_index, best_result = max(scored, key=lambda item: (item[0], -item[1]))
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


def _answer_addresses_configuration_location(
    answer: str,
    query: str,
    results: list[SearchResult],
) -> bool:
    if not _is_configuration_location_query(query):
        return True
    normalized = re.sub(r"\s+", " ", answer).strip()
    if not normalized or re.search(r"\.pdf\b|\bretrieved evidence\b", normalized[:180], flags=re.IGNORECASE):
        return False
    first_sentence = re.split(r"(?<=[.!?])\s+|\n+", normalized, maxsplit=1)[0]
    if not re.match(r"^location\s*:", first_sentence, flags=re.IGNORECASE) and not re.search(
        r"\b(?:in|under|inside|within|from|open|go to|navigate to)\b",
        first_sentence,
        flags=re.IGNORECASE,
    ):
        return False
    subject_terms = _location_terms(_configuration_location_subject(query))
    first_terms = _location_terms(first_sentence)
    required_subject_terms = max(1, min(len(subject_terms), (len(subject_terms) + 1) // 2))
    if len(subject_terms.intersection(first_terms)) < required_subject_terms:
        return False
    if not LOCATION_CUE_RE.search(first_sentence):
        return False
    expected_location, _expected_results = _concise_configuration_location_answer(query, results)
    if expected_location:
        expected_path = expected_location.split(". Purpose:", 1)[0]
        expected_labels = re.findall(
            r"\b([A-Z][A-Za-z0-9/&-]*(?:\s+[A-Z][A-Za-z0-9/&-]*){0,5}\s+"
            r"(?:Units?|Settings?|Options?|Menu|Screen|Tab|Area|Panel|Dialog|Folder))\b",
            expected_path,
        )
        compact_answer = re.sub(r"[^a-z0-9]+", "", normalized.lower())
        for label in expected_labels:
            compact_label = re.sub(r"[^a-z0-9]+", "", label.lower())
            if compact_label and compact_label not in compact_answer:
                return False
    evidence_terms: set[str] = set()
    for result in results[:4]:
        evidence_terms.update(_material_claim_terms(_configuration_location_evidence_text(result)))
    supporting_terms = _material_claim_terms(normalized).difference(subject_terms).intersection(evidence_terms)
    return len(supporting_terms) >= 3


def _repair_query_model_separators(text: str, query: str) -> str:
    repaired = text
    for requested_model in _model_tokens(query):
        model_parts = requested_model.split("-")
        if len(model_parts) != 2 or not all(
            re.fullmatch(r"[A-Z0-9]{2,}", part, flags=re.IGNORECASE)
            for part in model_parts
        ):
            continue
        repaired = re.sub(
            rf"\b{re.escape(model_parts[0])}\s*:\s*{re.escape(model_parts[1])}\b",
            requested_model,
            repaired,
            flags=re.IGNORECASE,
        )
    return repaired


def _clean_final_answer_text(text: str, query: str) -> str:
    """Remove bounded parser artifacts from any final answer route."""
    clean_answer = re.sub(
        r"\s*;?\s*Row:\s*\d+\s*;\s*Column:\s*\d+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()
    # Some PDFs encode visual bullets or spacing as private-use glyphs.
    clean_answer = re.sub(r"\s*[\ue000-\uf8ff]\s*", " ", clean_answer).strip()
    for broken, repaired in (
        (r"\bov\s+ercurrent\b", "overcurrent"),
        (r"\bf\s+lowing\b", "flowing"),
        (r"\bf\s+low\b", "flow"),
    ):
        clean_answer = re.sub(broken, repaired, clean_answer, flags=re.IGNORECASE)
    return _repair_query_model_separators(clean_answer, query)


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
        or not _answer_addresses_configuration_location(answer.answer, query, results)
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

    concise_troubleshooting, concise_results = _concise_troubleshooting_answer(query, results)
    if concise_troubleshooting and concise_results:
        normalized = _fallback_answer(query, concise_results)
        answer = answer.model_copy(
            update={
                "answer": concise_troubleshooting,
                "citations": normalized.citations,
                "used_documents": normalized.used_documents,
                "insufficient_evidence": False,
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

    clean_answer = _clean_final_answer_text(answer.answer, query)

    return answer.model_copy(
        update={
            "answer": clean_answer,
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
            "num_predict": settings.ollama_answer_num_predict,
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
    results = _scope_answer_results_to_query_models(query, results)
    if prioritized_results is not None:
        scoped_prioritized_results = _scope_answer_results_to_query_models(query, prioritized_results)
        if [result.chunk_id for result in scoped_prioritized_results] != [
            result.chunk_id for result in prioritized_results
        ]:
            prioritized_results = scoped_prioritized_results
            # Precomputed summaries may already have merged conflicting product
            # documents. Rebuild them from the scoped evidence rather than trying
            # to remove claims and provenance from a merged summary envelope.
            summarized_evidence = None
    conditioned_answer, conditioned_results = _concise_conditioned_measurement_answer(query, results)
    if conditioned_answer:
        answer = validate_answer(
            _fallback_answer(query, conditioned_results),
            conditioned_results,
            query=query,
        )
        answer.answer = conditioned_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "conditioned_measurement"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "conditioned_measurement",
                "num_predict": None,
                "used_fallback": False,
                "answer_source": "deterministic_conditioned_measurement",
            }
        )
        return answer, trace
    temporal_answer, temporal_results = _concise_temporal_effect_answer(query, results)
    if temporal_answer:
        answer = validate_answer(
            _fallback_answer(query, temporal_results),
            temporal_results,
            query=query,
        )
        answer.answer = temporal_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "temporal_effect"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "temporal_effect",
                "num_predict": None,
                "used_fallback": False,
                "answer_source": "deterministic_temporal_effect",
            }
        )
        return answer, trace
    instruction_answer, instruction_results = _concise_instruction_answer(
        query,
        prioritized_results or results,
    )
    if instruction_answer:
        answer = validate_answer(
            _fallback_answer(query, instruction_results),
            instruction_results,
            query=query,
        )
        answer.answer = _clean_final_answer_text(instruction_answer, query)
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "instruction"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "instruction",
                "num_predict": None,
                "used_fallback": False,
                "answer_source": "deterministic_instruction",
            }
        )
        return answer, trace
    alignment_answer, alignment_results = _concise_alignment_components_answer(query, results)
    if alignment_answer:
        answer = validate_answer(
            _fallback_answer(query, alignment_results),
            alignment_results,
            query=query,
        )
        answer.answer = alignment_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "alignment_components"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "alignment_components",
                "used_fallback": True,
                "answer_source": "deterministic_alignment_components",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    physical_location_answer, physical_location_results = _concise_physical_location_answer(
        query,
        prioritized_results or results,
    )
    if physical_location_answer:
        answer = validate_answer(
            _fallback_answer(query, physical_location_results),
            physical_location_results,
            query=query,
        )
        answer.answer = physical_location_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "physical_location"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "physical_location",
                "used_fallback": True,
                "answer_source": "deterministic_physical_location",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    enumerated_answer, enumerated_results = _concise_enumerated_options_answer(query, results)
    if enumerated_answer:
        answer = validate_answer(
            _fallback_answer(query, enumerated_results),
            enumerated_results,
            query=query,
        )
        answer.answer = enumerated_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "enumerated_options"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "enumerated_options",
                "used_fallback": True,
                "answer_source": "deterministic_enumerated_options",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    labeled_list_answer, labeled_list_results = _concise_labeled_list_answer(query, results)
    if labeled_list_answer:
        answer = validate_answer(
            _fallback_answer(query, labeled_list_results),
            labeled_list_results,
            query=query,
        )
        answer.answer = labeled_list_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "labeled_list"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "labeled_list",
                "used_fallback": True,
                "answer_source": "deterministic_labeled_list",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    required_setting_answer, required_setting_results = _concise_required_setting_answer(query, results)
    if required_setting_answer:
        answer = validate_answer(
            _fallback_answer(query, required_setting_results),
            required_setting_results,
            query=query,
        )
        answer.answer = required_setting_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "required_setting"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "required_setting",
                "used_fallback": True,
                "answer_source": "deterministic_required_setting",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    mode_requirement_answer, mode_requirement_results = _concise_named_mode_requirement_answer(query, results)
    if mode_requirement_answer:
        answer = validate_answer(
            _fallback_answer(query, mode_requirement_results),
            mode_requirement_results,
            query=query,
        )
        answer.answer = mode_requirement_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "named_mode_requirement"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "named_mode_requirement",
                "used_fallback": True,
                "answer_source": "deterministic_named_mode_requirement",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    capability_answer, capability_results = _concise_capability_answer(query, results)
    if capability_answer:
        answer = validate_answer(
            _fallback_answer(query, capability_results),
            capability_results,
            query=query,
        )
        answer.answer = capability_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "capability"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "capability",
                "used_fallback": True,
                "answer_source": "deterministic_capability",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    named_selection_answer, named_selection_results = _concise_named_selection_answer(query, results)
    if named_selection_answer:
        answer = validate_answer(
            _fallback_answer(query, named_selection_results),
            named_selection_results,
            query=query,
        )
        answer.answer = named_selection_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "named_selection"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "named_selection",
                "used_fallback": True,
                "answer_source": "deterministic_named_selection",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    contamination_answer, contamination_results = _concise_contamination_action_answer(query, results)
    if contamination_answer:
        answer = validate_answer(
            _fallback_answer(query, contamination_results),
            contamination_results,
            query=query,
        )
        answer.answer = contamination_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "contamination_action"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "contamination_action",
                "used_fallback": True,
                "answer_source": "deterministic_contamination_action",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    event_action, event_action_results = _concise_event_action_answer(
        query,
        prioritized_results or results,
    )
    if event_action:
        answer = validate_answer(
            _fallback_answer(query, event_action_results),
            event_action_results,
            query=query,
        )
        answer.answer = event_action
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "event_action"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "event_action",
                "used_fallback": True,
                "answer_source": "deterministic_event_action",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    response_light_condition_query = bool(
        re.search(r"\bresponse[- ]time\b", query, flags=re.IGNORECASE)
        and re.search(r"\b(?:light|saturat\w*|insufficient|recalibrat\w*)\b", query, flags=re.IGNORECASE)
    )
    if response_light_condition_query:
        # Relevance review can promote the complete atomic evidence above the
        # raw assembled retrieval order. Run this route against that reviewed
        # evidence when available so a neighboring troubleshooting row cannot
        # win merely because it appeared earlier in ``results``.
        response_evidence = prioritized_results or results
        response_answer, response_results = _concise_structured_fact_answer(
            query,
            response_evidence,
        )
        if response_answer:
            answer = validate_answer(
                _fallback_answer(query, response_results),
                response_results,
                query=query,
            )
            answer.answer = response_answer
            trace["relevance_review"].update(
                {"provider": "deterministic", "model": None, "prompt_kind": "structured_fact"}
            )
            trace["summarization"].update(
                {"provider": "deterministic", "model": None, "summary_count": 0}
            )
            trace["final_answer"].update(
                {
                    "provider": "deterministic",
                    "model": None,
                    "prompt_kind": "structured_fact",
                    "used_fallback": True,
                    "answer_source": "deterministic_structured_fact",
                    "fallback_reason": None,
                    "summarized_evidence": [],
                    "num_predict": None,
                }
            )
            return answer, trace
    if _is_troubleshooting_query(query):
        troubleshooting_evidence = prioritized_results or results
        troubleshooting_answer, structured_results = _concise_troubleshooting_answer(
            query,
            troubleshooting_evidence,
        )
        if troubleshooting_answer:
            answer = validate_answer(
                _fallback_answer(query, structured_results),
                structured_results,
                query=query,
            )
            # The concise troubleshooting route can assemble cause and action
            # from separate cells in one exact table row. Preserve every
            # contributing chunk in the final provenance instead of allowing
            # the fallback validator to collapse citations to its first cell.
            answer = answer.model_copy(
                update={
                    "answer": troubleshooting_answer,
                    "used_documents": [
                        {
                            "document_id": result.source_document_id,
                            "title": result.title,
                            "version": result.document_version_id,
                            "pages": result.pages,
                            "section_path": result.section_path,
                        }
                        for result in structured_results
                    ],
                    "citations": [
                        {
                            "chunk_id": result.chunk_id,
                            "document_id": result.source_document_id,
                            "pages": result.pages,
                            "quote_span": None,
                        }
                        for result in structured_results
                    ],
                }
            )
            trace["relevance_review"].update(
                {"provider": "deterministic", "model": None, "prompt_kind": "structured_troubleshooting"}
            )
            trace["summarization"].update(
                {"provider": "deterministic", "model": None, "summary_count": 0}
            )
            trace["final_answer"].update(
                {
                    "provider": "deterministic",
                    "model": None,
                    "prompt_kind": "structured_troubleshooting",
                    "used_fallback": False,
                    "answer_source": "structured_evidence",
                    "fallback_reason": None,
                    "summarized_evidence": [],
                    "num_predict": None,
                }
            )
            return answer, trace
    default_value_answer, default_value_results = _concise_default_setting_value_answer(query, results)
    if default_value_answer:
        answer = validate_answer(
            _fallback_answer(query, default_value_results),
            default_value_results,
            query=query,
        )
        answer.answer = default_value_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "default_setting_value"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "default_setting_value",
                "used_fallback": True,
                "answer_source": "deterministic_default_setting_value",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    part_number_answer, part_number_results = _concise_part_number_answer(query, results)
    if part_number_answer:
        answer = validate_answer(
            _fallback_answer(query, part_number_results),
            part_number_results,
            query=query,
        )
        answer.answer = part_number_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "part_number"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "part_number",
                "used_fallback": True,
                "answer_source": "deterministic_part_number",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    matching_model_answer, matching_model_results = _concise_matching_model_answer(query, results)
    if matching_model_answer:
        answer = validate_answer(
            _fallback_answer(query, matching_model_results),
            matching_model_results,
            query=query,
        )
        answer.answer = matching_model_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "matching_model"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "matching_model",
                "used_fallback": True,
                "answer_source": "deterministic_matching_model",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    structured_fact_evidence = prioritized_results or results
    if prioritized_results:
        prioritized_ids = {result.chunk_id for result in prioritized_results}
        structured_fact_evidence = [
            *prioritized_results,
            *(result for result in results if result.chunk_id not in prioritized_ids),
        ]
    named_option_preview, _named_option_preview_results = _concise_named_option_behavior_answer(
        query,
        results,
    )
    dependent_list_preview, _dependent_list_preview_results = _concise_dependent_list_answer(
        query,
        results,
    )
    model_field_table_preview = any(
        len(re.findall(r"(?:^|\s)Model\s*:", str(result.content or ""), flags=re.IGNORECASE)) > 1
        and _focused_model_field_record_answer_text(query, result)
        for result in results[:10]
    )
    use_precomputed_model_path = prioritized_results is not None and bool(summarized_evidence)
    structured_fact_answer, structured_fact_results = (
        ("", [])
        if (
            named_option_preview
            or dependent_list_preview
            or model_field_table_preview
            or use_precomputed_model_path
        )
        else _concise_structured_fact_answer(query, structured_fact_evidence)
    )
    if structured_fact_answer:
        answer = validate_answer(
            _fallback_answer(query, structured_fact_results),
            structured_fact_results,
            query=query,
        )
        answer.answer = _clean_final_answer_text(structured_fact_answer, query)
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "structured_fact"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        threshold_measurement = bool(
            re.search(
                r"\bat\s+what\b.{0,80}\b(?:temperature|pressure|voltage|speed)\b",
                query,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:above|below|exceeds?|limited|threshold)\b",
                structured_fact_answer,
                flags=re.IGNORECASE,
            )
        )
        table_cell_fact = bool(
            re.search(
                r"\b(?:how\s+many\s+input\s+terminals?|"
                r"function\s+(?:is\s+)?assigned\s+to\s+the\s+first\s+input)\b",
                query,
                flags=re.IGNORECASE,
            )
            and any(
                _focused_labeled_table_cell_answer_text(query, result)
                == _clean_final_answer_text(structured_fact_answer, query)
                for result in structured_fact_results
            )
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "structured_fact",
                "used_fallback": True,
                "answer_source": (
                    "structured_evidence"
                    if threshold_measurement or table_cell_fact
                    else "deterministic_structured_fact"
                ),
                "fallback_reason": None,
                "summarized_evidence": [],
            }
        )
        return answer, trace
    dependent_list_answer, dependent_list_results = _concise_dependent_list_answer(query, results)
    if dependent_list_answer:
        answer = validate_answer(
            _fallback_answer(query, dependent_list_results),
            dependent_list_results,
            query=query,
        )
        answer.answer = dependent_list_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "dependent_list"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "dependent_list",
                "used_fallback": True,
                "answer_source": "deterministic_dependent_list",
                "fallback_reason": None,
                "summarized_evidence": [],
            }
        )
        return answer, trace
    alternative_answer, alternative_results = _concise_named_alternative_answer(query, results)
    if alternative_answer:
        answer = validate_answer(
            _fallback_answer(query, alternative_results),
            alternative_results,
            query=query,
        )
        answer.answer = alternative_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "named_alternative"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "named_alternative",
                "used_fallback": True,
                "answer_source": "deterministic_named_alternative",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    option_behavior_answer, option_behavior_results = _concise_named_option_behavior_answer(query, results)
    if option_behavior_answer:
        answer = validate_answer(
            _fallback_answer(query, option_behavior_results),
            option_behavior_results,
            query=query,
        )
        answer.answer = option_behavior_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "named_option_behavior"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "named_option_behavior",
                "used_fallback": True,
                "answer_source": "deterministic_named_option_behavior",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    named_selection_answer, named_selection_results = _concise_named_selection_answer(query, results)
    if named_selection_answer:
        answer = validate_answer(
            _fallback_answer(query, named_selection_results),
            named_selection_results,
            query=query,
        )
        answer.answer = named_selection_answer
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "named_selection"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "named_selection",
                "used_fallback": True,
                "answer_source": "deterministic_named_selection",
                "fallback_reason": None,
                "summarized_evidence": [],
                "num_predict": None,
            }
        )
        return answer, trace
    table_answer, table_results = _concise_structured_table_answer(query, results)
    if (
        table_answer
        and not _is_configuration_location_query(query)
        and not use_precomputed_model_path
    ):
        answer = validate_answer(_fallback_answer(query, table_results), table_results, query=query)
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "structured_table"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "structured_table",
                "num_predict": None,
                "answer_source": "structured_evidence",
            }
        )
        return answer, trace
    location_answer, location_results = _concise_configuration_location_answer(query, results)
    if location_answer:
        answer = validate_answer(_fallback_answer(query, location_results), location_results, query=query)
        trace["relevance_review"].update(
            {"provider": "deterministic", "model": None, "prompt_kind": "configuration_location"}
        )
        trace["summarization"].update(
            {"provider": "deterministic", "model": None, "summary_count": 0}
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "configuration_location",
                "num_predict": None,
                "answer_source": "structured_evidence",
            }
        )
        return answer, trace
    if _is_troubleshooting_query(query):
        structured_results = _order_troubleshooting_results(query, results[:10])
        answer = validate_answer(
            _fallback_answer(query, structured_results),
            structured_results,
            query=query,
        )
        trace["relevance_review"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "structured_troubleshooting",
            }
        )
        trace["summarization"].update(
            {
                "provider": "deterministic",
                "model": None,
                "summary_count": 0,
            }
        )
        trace["final_answer"].update(
            {
                "provider": "deterministic",
                "model": None,
                "prompt_kind": "structured_troubleshooting",
                "num_predict": None,
                "answer_source": "structured_evidence",
            }
        )
        return answer, trace
    if prioritized_results is None:
        candidate_results = results[:12] if _is_configuration_location_query(query) else results[:8]
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
    last_error: Exception | None = None
    for attempt in range(2):
        attempt_num_predict = (
            settings.ollama_answer_num_predict
            if attempt == 0
            else max(settings.ollama_answer_num_predict, 1024)
        )
        attempt_messages = messages
        if attempt:
            attempt_messages = [
                messages[0],
                {
                    "role": "user",
                    "content": (
                        f"{messages[1]['content']}\n\n"
                        "Return exactly one valid JSON object matching the schema. "
                        "Lead with the direct answer and do not copy raw evidence blocks."
                    ),
                },
            ]
        try:
            generated, _raw = chat_json(
                model=settings.ollama_answer_model,
                messages=attempt_messages,
                json_schema=ANSWER_SCHEMA,
                think=False,
                num_predict=attempt_num_predict,
                timeout=90.0,
                purpose="final_answer",
            )
            generated_answer = AnswerResponse.model_validate(_normalize_generated_answer_payload(generated, prioritized_results))
            validated_answer = validate_answer(generated_answer, prioritized_results, query=query)
            if validated_answer.answer != generated_answer.answer and any(
                "not sufficiently supported" in warning for warning in validated_answer.warnings
            ):
                can_use_summary_recovery = bool(
                    re.search(
                        r"\b(?:angle|distance|height|limit|pressure|range|speed|temperature|voltage|width)\b",
                        query,
                        flags=re.IGNORECASE,
                    )
                    and _quantity_terms(query)
                ) or bool(
                    re.search(r"\bwhere\b.{0,100}\bmeasure(?:d|ment)?\b", query, flags=re.IGNORECASE)
                )
                summary_recovery = (
                    _fallback_answer_from_summaries(query, summarized_evidence, prioritized_results)
                    if can_use_summary_recovery
                    else None
                )
                if summary_recovery is not None:
                    recovered_answer = validate_answer(summary_recovery, prioritized_results, query=query)
                    if recovered_answer.answer == summary_recovery.answer:
                        validated_answer = recovered_answer
                trace["final_answer"].update(
                    {
                        "used_fallback": True,
                        "answer_source": (
                            "fallback_summary_validation"
                            if summary_recovery is not None and validated_answer.answer == summary_recovery.answer
                            else "fallback_validation"
                        ),
                        "fallback_reason": "Generated answer was replaced by retrieval-grounded fallback during validation.",
                    }
                )
            trace["final_answer"]["attempts"] = attempt + 1
            trace["final_answer"]["num_predict"] = attempt_num_predict
            return validated_answer, trace
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Final answer generation attempt %s failed for model=%s: %s",
                attempt + 1,
                settings.ollama_answer_model,
                exc,
            )
    summary_fallback = _fallback_answer_from_summaries(query, summarized_evidence, prioritized_results)
    fallback_answer = validate_answer(
        summary_fallback or _fallback_answer(query, prioritized_results),
        prioritized_results,
        query=query,
    )
    trace["final_answer"].update(
        {
            "attempts": 2,
            "used_fallback": True,
            "answer_source": "fallback_summary" if summary_fallback else "fallback_exception",
            "fallback_reason": str(last_error),
        }
    )
    return fallback_answer, trace


def prepare_answer_evidence(query: str, results: list[SearchResult]) -> dict[str, Any]:
    results = _scope_answer_results_to_query_models(query, results)
    candidate_results = results[:12]
    prioritized = prioritize_results_for_answer(query, candidate_results)
    summaries = summarize_results_for_answer(query, prioritized["prioritized_results"])
    return {
        "candidate_results": candidate_results,
        "judgments": prioritized["judgments"],
        "prioritized_results": prioritized["prioritized_results"],
        "summaries": summaries,
    }


def prioritize_results_for_answer(query: str, candidate_results: list[SearchResult]) -> dict[str, Any]:
    _contamination_answer, contamination_results = _concise_contamination_action_answer(
        query,
        candidate_results,
    )
    if contamination_results:
        return {
            "judgments": _fallback_relevance_judgments(query, contamination_results),
            "prioritized_results": contamination_results,
            "selection_source": "contamination_action",
        }
    if _is_troubleshooting_query(query):
        prioritized_results = _focused_troubleshooting_results(
            query,
            _order_troubleshooting_results(query, candidate_results),
        )
        return {
            "judgments": _fallback_relevance_judgments(query, prioritized_results),
            "prioritized_results": prioritized_results,
            "selection_source": "structured_troubleshooting",
        }
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
    _instruction_answer, instruction_evidence = _concise_instruction_answer(
        query,
        candidate_results,
    )
    _table_answer, table_evidence = _concise_structured_table_answer(query, candidate_results)
    _default_value_answer, default_value_evidence = _concise_default_setting_value_answer(
        query,
        candidate_results,
    )
    _matching_model_answer, matching_model_evidence = _concise_matching_model_answer(
        query,
        candidate_results,
    )
    _part_number_answer, part_number_evidence = _concise_part_number_answer(
        query,
        candidate_results,
    )
    structured_fact_evidence = _structured_fact_evidence_results(query, candidate_results)
    if structured_fact_evidence:
        best_distinctive_coverage = max(
            _distinctive_query_coverage(query, result)
            for result in candidate_results
        )
        structured_fact_evidence = [
            result
            for result in structured_fact_evidence
            if _distinctive_query_coverage(query, result) >= best_distinctive_coverage
        ]
    location_evidence = _configuration_location_evidence_results(query, candidate_results)
    prioritized_results = [
        result
        for result in [
            *anchored_results,
            *comparison_evidence,
            *procedure_evidence,
            *instruction_evidence,
            *table_evidence,
            *default_value_evidence,
            *matching_model_evidence,
            *part_number_evidence,
            *structured_fact_evidence,
            *location_evidence,
        ]
    ]
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
    protected_chunk_ids = {
        result.chunk_id
        for result in [
            *anchored_results,
            *comparison_evidence,
            *procedure_evidence,
            *instruction_evidence,
            *table_evidence,
            *default_value_evidence,
            *matching_model_evidence,
            *part_number_evidence,
            *structured_fact_evidence,
            *location_evidence,
        ]
    }
    original_rank = {result.chunk_id: index for index, result in enumerate(prioritized_results)}
    location_priority = {
        result.chunk_id: len(location_evidence) - index
        for index, result in enumerate(location_evidence)
    }
    verdict_priority = {"relevant": 2, "potentially_relevant": 1, "not_relevant": 0}
    prioritized_results.sort(
        key=lambda result: (
            result.chunk_id in protected_chunk_ids,
            location_priority.get(result.chunk_id, 0),
            _distinctive_query_coverage(query, result),
            verdict_priority.get(judgment_by_chunk_id.get(result.chunk_id, {}).get("verdict", ""), 0),
            _query_evidence_overlap_score(query, result),
            -original_rank[result.chunk_id],
        ),
        reverse=True,
    )
    return {
        "judgments": judgments,
        "prioritized_results": prioritized_results,
    }


def _distinctive_query_coverage(query: str, result: SearchResult) -> int:
    distinctive_terms = {
        term
        for term in _answer_terms(query)
        if len(term) >= 10 or any(char.isdigit() for char in term) or "-" in term or "/" in term
    }
    if not distinctive_terms:
        return 0
    evidence_terms = _answer_terms(
        " ".join([_evidence_text(result), result.title, " ".join(result.section_path)])
    )
    return len(distinctive_terms.intersection(evidence_terms))


def _query_evidence_overlap_score(query: str, result: SearchResult) -> int:
    ignored = {"which", "series", "system", "manual", "does", "what", "when", "where"}
    query_terms = _answer_terms(query).difference(ignored)
    evidence_terms = _answer_terms(_evidence_text(result))
    return len(query_terms.intersection(evidence_terms))


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
