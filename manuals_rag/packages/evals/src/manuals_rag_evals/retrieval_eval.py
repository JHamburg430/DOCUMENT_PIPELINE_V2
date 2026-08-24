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

QUERY_DEDUPE_FILLER = STOPWORDS.union(
    {
        "a",
        "an",
        "are",
        "be",
        "before",
        "do",
        "how",
        "is",
        "required",
        "should",
        "specified",
        "applies",
        "apply",
        "described",
        "give",
        "given",
        "other",
        "to",
        "you",
    }
)

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
    "cell",
    "column",
    "columns",
    "header",
    "headers",
    "row",
    "rows",
    "setting",
    "settings",
    "table",
    "information",
    "data",
    "output",
    "input",
    "store",
    "stores",
    "stored",
    "number",
    "numbers",
    "priority",
    "checking",
}

TECHNICAL_VERBS = {
    "connect",
    "disconnect",
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
    "store",
    "stores",
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
    "device",
    "devices",
    "plc",
    "network",
    "power",
    "supply",
    "ethernet/ip",
    "laser",
    "wavelength",
}

USER_STYLE_QUERY_SYSTEM_PROMPT = """
You generate realistic retrieval benchmark queries for technical documents.

Return strict JSON with this shape:
{"queries":[{"query":"...","intent":"...","reason":"..."}]}

Rules:
- Write concise question-form queries a real technician, engineer, operator, purchaser, or integrator might ask.
- Base every query on the provided context, especially the source snippet, structured fields, labels, and extracted terms.
- Make each query answerable from the source snippet itself, not merely from a surrounding section.
- Include enough discriminating terms from the source snippet that the intended row, warning, step, or spec can be found without reading adjacent context.
- For compact specs or table rows, include the label plus one concrete value/unit/class/action when available.
- Do not say "this document", "this manual", "the datasheet", "this section", or similar.
- Do not use meta phrasing like "what specification", "what value is listed", "where does", "what does the document say", or "which step in".
- Do not mirror the source text mechanically or copy long spans verbatim.
- Keep each query concise and natural, usually under 14 words.
- End each query with a question mark.
- Prefer direct technical questions:
  What/which setting applies to model?
  What value/unit/class is specified for model?
  Which devices/actions are required?
  How should a procedure step be performed?
- If the snippet contains explicit fields, labels, units, steps, warnings, or settings, use those concrete concepts in the query.
- Prefer concrete terms from the snippet such as field names, units, menu labels, protocol names, settings, or actions.
- Avoid vague storage-only phrasing such as "stores number" unless the query also includes the specific field/action name, for example "command number" or "specified-command".
- If previous questions are provided, do not repeat them or make close paraphrases. Ask about a different concrete facet of the same snippet.
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
    retrieval_task: str = "single_step_retrieval"
    expected_source_chunk_ids: list[str] | None = None
    expected_evidence: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalized_query_key(query: str) -> str:
    key = normalize_text(query)
    key = re.sub(r"\?$", "", key).strip()
    return key


def tokenize(text: str) -> list[str]:
    return [
        token.strip(".,;:")
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-/.]+", text.lower())
        if token.strip(".,;:")
    ]


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


def _meaningful_table_field_value_pairs(content: str) -> list[tuple[str, str]]:
    ignored_fields = {"column headers", "row headers", "cell value", "table header", "header role", "row", "column"}
    return [
        (field, value)
        for field, value in _field_value_pairs(content)
        if normalize_text(field) not in ignored_fields
    ]


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
    metadata = dict(chunk.get("metadata_json", {}))
    if len(content) < 40:
        return False
    if (
        chunk_type == "table_record"
        and metadata.get("table_cell")
        and not _metadata_list(metadata, "table_row_headers")
        and not _meaningful_table_field_value_pairs(content)
    ):
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


def _query_context_terms(query: str, chunk: dict[str, Any]) -> set[str]:
    metadata = dict(chunk.get("metadata_json", {}))
    content_terms = set(tokenize(str(chunk.get("content", ""))))
    section_terms = set(tokenize(str(chunk.get("section_path_text", ""))))
    title_terms = set(tokenize(str(chunk.get("title", ""))))
    model_terms = set(tokenize(str(chunk.get("product_model") or metadata.get("product_model") or "")))
    allowed = content_terms.union(section_terms, title_terms, model_terms)
    ignored = STOPWORDS.union(GENERIC_ANCHORS).union({"new", "series"})
    return {
        token
        for token in tokenize(query)
        if token not in ignored and (token in allowed or _is_high_signal_anchor(token))
    }


def _query_source_affinity_score(query: str, chunk: dict[str, Any], anchors: list[str]) -> int:
    query_terms = _query_context_terms(query, chunk)
    if not query_terms:
        return 0
    content_terms = set(tokenize(str(chunk.get("content", ""))))
    anchor_terms = set(anchors)
    score = 0
    score += sum(1 for term in query_terms if term in content_terms)
    score += sum(1 for term in query_terms if term in anchor_terms)
    score += sum(1 for term in query_terms if _is_high_signal_anchor(term))
    return score


def _query_source_content_overlap(query: str, chunk: dict[str, Any]) -> int:
    ignored = STOPWORDS.union(GENERIC_ANCHORS).union({"new", "series"})
    content_terms = set(tokenize(str(chunk.get("content", ""))))
    return sum(1 for term in tokenize(query) if term not in ignored and term in content_terms)


def _query_has_discriminating_source_term(query: str, chunk: dict[str, Any]) -> bool:
    metadata = dict(chunk.get("metadata_json", {}))
    content_terms = set(tokenize(str(chunk.get("content", ""))))
    model_terms = set(tokenize(str(chunk.get("product_model") or metadata.get("product_model") or "")))
    title_terms = set(tokenize(str(chunk.get("title", ""))))
    section_terms = set(tokenize(str(chunk.get("section_path_text", ""))))
    ignored = STOPWORDS.union(GENERIC_ANCHORS).union(model_terms).union(title_terms).union(section_terms).union({"new", "series"})
    for token in tokenize(query):
        if token in ignored or token not in content_terms:
            continue
        if _is_high_signal_anchor(token) or token in GENERIC_TECHNICAL_TERMS or token in TECHNICAL_VERBS or len(token) >= 7:
            return True
    return False


def _filename_artifact_terms(chunk: dict[str, Any]) -> set[str]:
    filename = str(chunk.get("source_filename", ""))
    if not filename:
        return set()
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
    raw_parts = [part.lower() for part in re.split(r"[_\W]+", stem) if part]
    content_terms = set(tokenize(str(chunk.get("content", ""))))
    section_terms = set(tokenize(str(chunk.get("section_path_text", ""))))
    artifact_terms: set[str] = set()
    for part in raw_parts:
        if part in content_terms or part in section_terms:
            continue
        if re.fullmatch(r"[a-z]\d{2,}[a-z]{2,}", part) or re.fullmatch(r"\d{3,}[a-z]*", part):
            artifact_terms.add(part)
    return artifact_terms


def _query_uses_filename_artifact(query: str, chunk: dict[str, Any]) -> bool:
    artifacts = _filename_artifact_terms(chunk)
    if not artifacts:
        return False
    return any(token in artifacts for token in tokenize(query))


def _query_looks_document_bound(query: str) -> bool:
    lowered = normalize_text(query)
    banned_phrases = {
        "this document",
        "this manual",
        "the manual",
        "this datasheet",
        "the datasheet",
        "this section",
        "where does the document",
        "where does the manual",
        "where does this document",
        "where does this manual",
        "what does the document",
        "what does the manual",
        "what does this document",
        "what does this manual",
        "which step in",
        "what procedure in",
        "what warning does",
    }
    return any(phrase in lowered for phrase in banned_phrases)


def _query_looks_meta(query: str) -> bool:
    lowered = normalize_text(query)
    banned_prefixes = (
        "what specification",
        "where does the document",
        "where does the manual",
        "where does this document",
        "where does this manual",
        "what does the document",
        "what does the manual",
        "what does this document",
        "what does this manual",
        "which step in",
        "what procedure",
        "what warning",
        "in ",
    )
    if any(lowered.startswith(prefix) for prefix in banned_prefixes):
        return True
    banned_fragments = {
        "does the table say about",
        "refers to",
        "mentions",
        "discuss",
    }
    return any(fragment in lowered for fragment in banned_fragments)


def _query_looks_like_question(query: str) -> bool:
    return query.strip().endswith("?")


def _query_dedupe_terms(query: str) -> set[str]:
    return {
        token
        for token in tokenize(query)
        if token not in QUERY_DEDUPE_FILLER and token not in GENERIC_ANCHORS
    }


def _queries_are_near_duplicates(query: str, existing_query: str) -> bool:
    if _normalized_query_key(query) == _normalized_query_key(existing_query):
        return True
    query_terms = _query_dedupe_terms(query)
    existing_terms = _query_dedupe_terms(existing_query)
    if not query_terms or not existing_terms:
        return False
    overlap = len(query_terms.intersection(existing_terms))
    if overlap < 2:
        return False
    smaller = min(len(query_terms), len(existing_terms))
    larger = max(len(query_terms), len(existing_terms))
    containment = overlap / smaller
    jaccard = overlap / len(query_terms.union(existing_terms))
    return (overlap >= 3 and containment >= 0.8 and larger <= smaller + 2) or jaccard >= 0.75


def _has_near_duplicate_query(query: str, existing_queries: list[str]) -> bool:
    return any(_queries_are_near_duplicates(query, existing_query) for existing_query in existing_queries)


def validate_eval_case(query: str, chunk: dict[str, Any], anchors: list[str]) -> tuple[bool, str]:
    chunk_type = str(chunk.get("chunk_type", ""))
    if not anchors:
        return False, "no_specific_anchor"
    if not _query_looks_like_question(query):
        return False, "not_question_form"
    if _query_looks_document_bound(query):
        return False, "document_bound_query"
    if _query_looks_meta(query):
        return False, "meta_query"
    if _query_uses_filename_artifact(query, chunk):
        return False, "filename_artifact_query"
    if chunk_type == "atomic_text":
        if len(anchors) < 2:
            return False, "atomic_requires_two_anchors"
        if not any(_is_high_signal_anchor(anchor) for anchor in anchors):
            return False, "atomic_requires_high_signal_anchor"
    if _query_specificity_score(query, anchors[:4]) < 5:
        return False, "low_specificity"
    affinity_score = _query_source_affinity_score(query, chunk, anchors)
    required_affinity = 4 if chunk_type in {"spec_record", "datasheet_record", "table_record"} else 3
    if affinity_score < required_affinity:
        return False, "weak_source_affinity"
    if chunk_type in {"spec_record", "datasheet_record", "table_record"} and _query_source_content_overlap(query, chunk) < 2:
        return False, "weak_source_affinity"
    if chunk_type in {"spec_record", "datasheet_record", "table_record"} and not _query_has_discriminating_source_term(query, chunk):
        return False, "weak_source_discriminator"
    return True, "validated"


def _safe_query_label(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata_json", {}))
    model = str(chunk.get("product_model") or metadata.get("product_model") or "").strip()
    if model and len(model) <= 60 and model.count("/") <= 3 and not _query_uses_filename_artifact(model, chunk):
        return model
    family = str(metadata.get("product_family") or "").strip()
    if family and len(family) <= 80 and family.count("/") <= 1 and not _query_uses_filename_artifact(family, chunk):
        return family
    title = str(chunk.get("title", "")).strip()
    filename = str(chunk.get("source_filename", "")).strip()
    if title and title != filename and not title.lower().endswith(".pdf"):
        return title
    return ""


def _for_label(label: str) -> str:
    return f" for {label}" if label else ""


def _to_label(label: str) -> str:
    return f" to {label}" if label else ""


def build_query_candidates(chunk: dict[str, Any]) -> list[tuple[str, str]]:
    content = str(chunk["content"]).strip()
    chunk_type = str(chunk.get("chunk_type", ""))
    anchors = extract_anchor_terms(content)
    labels = _quoted_menu_labels(content)
    if labels and labels[0] not in anchors:
        anchors = [labels[0], *anchors][:6]
    if not anchors:
        return []

    label = _safe_query_label(chunk)
    primary = anchors[0]
    secondary = anchors[1] if len(anchors) > 1 else primary
    tertiary = anchors[2] if len(anchors) > 2 else secondary
    candidates: list[tuple[str, str]] = []

    if "disconnect" in anchors and "devices" in anchors:
        candidates.append((f"Which other devices should be disconnected{_for_label(label)}?", "disconnect_devices_question"))
        if label:
            candidates.append((f"For {label}, which other devices should be disconnected before connection checks?", "disconnect_context_question"))
        candidates.append((f"Which devices should be disconnected before checking the EtherNet/IP connection{_for_label(label)}?", "disconnect_ethernetip_question"))
    if "specified-command" in anchors and "command" in anchors:
        candidates.append((f"What does the PLC store in Command Number{_for_label(label)}?", "command_number_question"))
        if label:
            candidates.append((f"For {label}, what specified-command number does the PLC store?", "specified_command_question"))
        else:
            candidates.append(("What specified-command number does the PLC store?", "specified_command_question"))

    if chunk_type in {"spec_record", "datasheet_record"}:
        if secondary != primary:
            candidates.append((f"What {primary} {secondary} is specified{_for_label(label)}?", "spec_primary_multi"))
        if tertiary not in {primary, secondary}:
            candidates.append((f"What {primary} {secondary} {tertiary} is specified{_for_label(label)}?", "spec_context_multi"))
        candidates.append((f"What {primary} {secondary} applies{_to_label(label)}?", "spec_value"))
        if secondary != primary:
            candidates.append((f"What {primary} {secondary} is specified{_for_label(label)}?", "spec_multi"))
    elif chunk_type == "table_record":
        candidates.extend(_build_table_query_candidates(chunk, label))
        if secondary != primary:
            candidates.append((f"What {primary} {secondary} applies{_to_label(label)}?", "table_primary_multi"))
        if tertiary not in {primary, secondary}:
            candidates.append((f"What {primary} {secondary} {tertiary} applies{_to_label(label)}?", "table_context_multi"))
        candidates.append((f"What {primary} applies{_to_label(label)}?", "table_primary"))
    elif chunk_type == "procedure_record":
        candidates.append((f"How do you {primary}{_for_label(label)}?", "procedure_howto"))
        if secondary != primary:
            candidates.append((f"What {primary} {secondary} steps apply{_to_label(label)}?", "procedure_step"))
        candidates.append((f"What {primary} procedure applies{_to_label(label)}?", "procedure_describe"))
    elif chunk_type == "warning_record":
        candidates.append((f"What warning applies to {primary}{_for_label(label)}?", "warning_primary"))
        if secondary != primary:
            candidates.append((f"What caution applies to {primary} {secondary}{_for_label(label)}?", "warning_caution"))
    else:
        if len(anchors) >= 2:
            candidates.append((f"What {primary} {secondary} is described{_for_label(label)}?", "general_multi"))
        candidates.append((f"What {primary} is described{_for_label(label)}?", "general_primary"))

    return candidates


def _structured_eval_input(chunk: dict[str, Any], anchors: list[str]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata_json", {}))
    content = str(chunk.get("content", ""))
    field_matches = _field_value_pairs(content)
    return {
        "chunk_type": str(chunk.get("chunk_type", "")),
        "title": _safe_query_label(chunk),
        "product_model": _safe_query_label(chunk),
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


def _metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key, [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _table_question_terms(chunk: dict[str, Any]) -> tuple[list[str], list[str]]:
    metadata = dict(chunk.get("metadata_json", {}))
    row_headers = _metadata_list(metadata, "table_row_headers")
    column_headers = _metadata_list(metadata, "table_column_headers")
    content = str(chunk.get("content", ""))
    cell_match = re.search(r"Cell value:\s*([^;]+)", content)
    cell_terms = extract_anchor_terms(cell_match.group(1) if cell_match else content, limit=3)
    query_terms = [*row_headers[:2], *column_headers[:2]]
    expected_terms = extract_anchor_terms(" ".join([*cell_terms, *query_terms]), limit=6)
    return query_terms, expected_terms


def _build_table_query_candidates(chunk: dict[str, Any], label: str) -> list[tuple[str, str]]:
    query_terms, expected_terms = _table_question_terms(chunk)
    field_pairs = _meaningful_table_field_value_pairs(str(chunk.get("content", "")))
    if len(field_pairs) >= 2:
        answer_field, _ = field_pairs[0]
        context_field, context_value = field_pairs[1]
        context_anchor = " ".join(extract_anchor_terms(context_value, limit=3)) or context_value.strip()
        key_value_subject = f"{answer_field.strip()} for {context_field.strip()} {context_anchor}".strip()
        if label:
            return [
                (f"For {label}, what {key_value_subject}?", "table_key_value_lookup"),
                (f"What {answer_field.strip()} is listed for {context_field.strip()} {context_anchor} on {label}?", "table_key_value_reverse_lookup"),
            ]
        return [(f"What {key_value_subject}?", "table_key_value_lookup")]
    if not query_terms or len(expected_terms) < 2:
        return []
    subject = " ".join(query_terms[:3])
    cell_anchor = expected_terms[0] if expected_terms and _is_high_signal_anchor(expected_terms[0]) else ""
    anchored_subject = f"{cell_anchor} {subject}".strip()
    if label:
        return [
            (f"What {anchored_subject} value applies to {label}?", "table_row_column_value"),
            (f"For {label}, what value is listed for {anchored_subject}?", "table_row_column_lookup"),
            (f"Which {anchored_subject} value should be used for {label}?", "table_row_column_engineer_lookup"),
        ]
    return [(f"What value is listed for {anchored_subject}?", "table_row_column_lookup")]


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
    previous_questions: list[str] | None = None,
    limit: int,
) -> list[tuple[str, str]]:
    previous_questions = previous_questions or []
    prompt = {
        "document_title": _safe_query_label(chunk),
        "structured_input": _structured_eval_input(chunk, anchors),
        "previous_questions_for_this_chunk": previous_questions[:10],
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
    seen: set[str] = {_normalized_query_key(question) for question in previous_questions}
    accepted_query_texts: list[str] = list(previous_questions)
    for item in generated:
        normalized = _normalized_query_key(item["query"])
        if normalized in seen:
            continue
        if _has_near_duplicate_query(item["query"], accepted_query_texts):
            continue
        is_valid, _ = validate_eval_case(item["query"], chunk, anchors)
        if not is_valid:
            continue
        seen.add(normalized)
        accepted_query_texts.append(item["query"])
        queries.append((item["query"], item["intent"]))
        if len(queries) >= limit:
            return queries
    for query, method in fallback_candidates:
        normalized = _normalized_query_key(query)
        if normalized in seen:
            continue
        if _has_near_duplicate_query(query, accepted_query_texts):
            continue
        seen.add(normalized)
        accepted_query_texts.append(query)
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
    previous_questions_by_chunk_id: dict[str, list[str]] | None = None,
) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    previous_questions_by_chunk_id = previous_questions_by_chunk_id or {}
    for chunk in chunks:
        anchors = extract_anchor_terms(str(chunk["content"]))
        if str(chunk.get("chunk_type", "")) == "table_record":
            _, table_expected_terms = _table_question_terms(chunk)
            if table_expected_terms:
                anchors = table_expected_terms
        if not chunk_is_queryworthy(chunk, anchors):
            continue
        previous_questions = previous_questions_by_chunk_id.get(str(chunk.get("id")), [])
        fallback_candidates = build_query_candidates(chunk)[:per_chunk_limit]
        candidates = (
            generate_user_style_queries(
                chunk,
                anchors=anchors,
                fallback_candidates=fallback_candidates,
                previous_questions=previous_questions,
                limit=per_chunk_limit,
            )
            if use_llm_generation
            else fallback_candidates
        )
        if len(anchors) < 1:
            continue
        accepted_query_texts: list[str] = list(previous_questions)
        for index, (query, method) in enumerate(candidates, start=1):
            if _has_near_duplicate_query(query, accepted_query_texts):
                continue
            is_valid, quality = validate_eval_case(query, chunk, anchors)
            if not is_valid:
                continue
            accepted_query_texts.append(query)
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


def _cell_value(content: str) -> str:
    match = re.search(r"Cell value:\s*([^;]+)", content)
    if match:
        return match.group(1).strip()
    return content.strip()


def _column_field(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata_json", {}))
    headers = _metadata_list(metadata, "table_column_headers")
    return normalize_text(" ".join(headers))


def _row_group_key(chunk: dict[str, Any]) -> tuple[str, str, str, str] | None:
    metadata = dict(chunk.get("metadata_json", {}))
    table_row = metadata.get("table_row")
    if table_row is None:
        return None
    return (
        str(chunk.get("source_document_id", "")),
        str(chunk.get("document_version_id", "")),
        str(chunk.get("section_path_text", "")),
        str(table_row),
    )


def _good_multi_step_anchor(value: str) -> bool:
    compact = normalize_text(value)
    if len(compact) < 18:
        return False
    if re.search(r"\.{4,}", compact):
        return False
    return bool(extract_anchor_terms(value, limit=2))


def _row_header_text(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata_json", {}))
    return normalize_text(" ".join(_metadata_list(metadata, "table_row_headers")))


def _best_related_cell(cells: list[dict[str, Any]], field_terms: set[str], anchor_value: str) -> dict[str, Any] | None:
    anchor_terms = set(extract_anchor_terms(anchor_value, limit=6))
    best: tuple[int, dict[str, Any]] | None = None
    for cell in cells:
        field = _column_field(cell)
        if not any(term in field for term in field_terms):
            continue
        evidence_text = f"{_row_header_text(cell)} {cell.get('content', '')}"
        score = sum(1 for term in anchor_terms if _term_matches_evidence(term, evidence_text))
        score += 2 if normalize_text(anchor_value)[:60] in normalize_text(evidence_text) else 0
        if best is None or score > best[0]:
            best = (score, cell)
    if best and best[0] > 0:
        return best[1]
    return None


def _short_answer_anchor(value: str) -> str:
    clean = re.sub(r"\s+", " ", value).strip()
    clean = re.sub(r"\s*\([^)]{18,}\)", "", clean).strip()
    if len(clean) <= 90:
        return clean
    sentence = re.split(r"(?<=[.!?])\s+", clean)[0].strip()
    return sentence[:90].strip() if sentence else clean[:90].strip()


def _multi_step_expected_evidence(*cells: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for cell in cells:
        value = _cell_value(str(cell.get("content", "")))
        terms = extract_anchor_terms(value, limit=4)
        if not terms:
            terms = extract_anchor_terms(str(cell.get("content", "")), limit=4)
        evidence.append(
            {
                "chunk_id": str(cell.get("id", "")),
                "field": _column_field(cell),
                "expected_terms": terms[:4],
                "snippet": content_preview(str(cell.get("content", "")), limit=180),
            }
        )
    return evidence


def _procedure_subject(content: str) -> str:
    subject = re.sub(r"^Procedure step\s*\d+\s*:\s*", "", content, flags=re.IGNORECASE).strip()
    subject = re.sub(r"^\d+\.\s*", "", subject).strip()
    return _short_answer_anchor(subject)


def _good_contextual_procedure_subject(subject: str) -> bool:
    compact = normalize_text(subject)
    if len(compact) < 18:
        return False
    if re.fullmatch(r"\d+\)?", compact):
        return False
    if re.match(r"^\d+\)?\s*$", compact):
        return False
    terms = [term for term in extract_anchor_terms(subject, limit=5) if term not in {"procedure", "typical"}]
    if len(terms) < 2:
        return False
    return bool(any(term in TECHNICAL_VERBS or _is_high_signal_anchor(term) for term in terms) or len(compact.split()) >= 5)


def _support_subject(chunk: dict[str, Any]) -> str:
    content = str(chunk.get("content", ""))
    field_pairs = _meaningful_table_field_value_pairs(content)
    if field_pairs:
        field, value = field_pairs[0]
        value_anchor = _short_answer_anchor(value)
        return f"{field.strip()} {value_anchor}".strip()
    labels = _quoted_menu_labels(content)
    if labels:
        return " ".join(labels[:2])
    anchors = extract_anchor_terms(content, limit=3)
    return " ".join(anchors)


def _support_chunks_for_contextual_multi_step(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    support: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_type = str(chunk.get("chunk_type", ""))
        if chunk_type not in {"atomic_text", "table_record", "spec_record", "datasheet_record", "warning_record"}:
            continue
        content = str(chunk.get("content", ""))
        anchors = extract_anchor_terms(content, limit=4)
        if chunk_type == "table_record":
            _, table_terms = _table_question_terms(chunk)
            anchors = table_terms or anchors
        if _looks_like_toc_line(content) or _looks_like_legal_boilerplate(content):
            continue
        if chunk_type != "atomic_text" and not chunk_is_queryworthy(chunk, anchors):
            continue
        if len(anchors) < 2:
            continue
        support.append(chunk)
    return support


def _chunks_are_contextually_linked(procedure: dict[str, Any], support: dict[str, Any]) -> bool:
    procedure_terms = [
        term
        for term in extract_anchor_terms(str(procedure.get("content", "")), limit=5)
        if term not in {"procedure", "typical"}
    ]
    support_terms = extract_anchor_terms(str(support.get("content", "")), limit=5)
    procedure_context = str(dict(procedure.get("metadata_json", {})).get("local_rerank_context") or "")
    support_context = str(dict(support.get("metadata_json", {})).get("local_rerank_context") or "")
    if not procedure_context and not support_context:
        return True
    procedure_hits = sum(1 for term in procedure_terms if _term_matches_evidence(term, support_context))
    support_hits = sum(1 for term in support_terms if _term_matches_evidence(term, procedure_context))
    return procedure_hits >= min(2, len(procedure_terms)) or support_hits >= min(2, len(support_terms))


def _build_contextual_multi_step_cases(
    chunks: list[dict[str, Any]],
    *,
    max_cases: int,
    seen_queries: list[str],
) -> list[RetrievalEvalCase]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for chunk in chunks:
        if int(chunk.get("chunk_level", 1) or 1) != 1:
            continue
        key = (
            str(chunk.get("source_document_id", "")),
            str(chunk.get("document_version_id", "")),
            str(chunk.get("section_path_text", "")),
        )
        grouped.setdefault(key, []).append(chunk)

    cases: list[RetrievalEvalCase] = []
    for section_chunks in grouped.values():
        procedures = [
            chunk
            for chunk in section_chunks
            if str(chunk.get("chunk_type", "")) == "procedure_record"
            and chunk_is_queryworthy(chunk, extract_anchor_terms(str(chunk.get("content", "")), limit=4))
        ]
        support_chunks = _support_chunks_for_contextual_multi_step(section_chunks)
        for procedure in procedures:
            procedure_subject = _procedure_subject(str(procedure.get("content", "")))
            if not procedure_subject or not _good_contextual_procedure_subject(procedure_subject):
                continue
            procedure_terms = extract_anchor_terms(str(procedure.get("content", "")), limit=4)
            for support in support_chunks:
                if support.get("id") == procedure.get("id"):
                    continue
                if not _chunks_are_contextually_linked(procedure, support):
                    continue
                support_subject = _support_subject(support)
                if not support_subject:
                    continue
                support_terms = extract_anchor_terms(str(support.get("content", "")), limit=4)
                if len(set(procedure_terms).intersection(support_terms)) >= min(2, len(support_terms)):
                    continue
                label = _safe_query_label(procedure) or _safe_query_label(support)
                query = (
                    f"When {procedure_subject}{_for_label(label)}, "
                    f"what related {support_subject} detail should be used?"
                )
                if _has_near_duplicate_query(query, seen_queries):
                    continue
                evidence = _multi_step_expected_evidence(procedure, support)
                expected_terms = []
                for item in evidence:
                    for term in item["expected_terms"]:
                        if term not in expected_terms:
                            expected_terms.append(term)
                        if len(expected_terms) >= 6:
                            break
                    if len(expected_terms) >= 6:
                        break
                if len(expected_terms) < 4:
                    continue
                seen_queries.append(query)
                cases.append(
                    RetrievalEvalCase(
                        case_id=f"{procedure['id']}::contextual_multi_step::{len(cases) + 1}",
                        query=query,
                        source_document_id=str(procedure["source_document_id"]),
                        document_version_id=str(procedure["document_version_id"]),
                        source_chunk_id=str(procedure["id"]),
                        source_title=str(procedure.get("title", "")),
                        source_filename=str(procedure.get("source_filename", "")),
                        chunk_type=str(procedure.get("chunk_type", "")),
                        section_path=str(procedure.get("section_path_text", "")),
                        page_from=int(procedure.get("page_from", 0)),
                        page_to=int(procedure.get("page_to", 0)),
                        expected_terms=expected_terms[:6],
                        expected_snippet=" | ".join(item["snippet"] for item in evidence),
                        generation_method="contextual_procedure_plus_section_evidence",
                        source_metadata=dict(procedure.get("metadata_json", {})),
                        benchmark_quality="validated",
                        anchor_terms=expected_terms[:6],
                        retrieval_task="multi_step_retrieval",
                        expected_source_chunk_ids=[item["chunk_id"] for item in evidence],
                        expected_evidence=evidence,
                    )
                )
                if len(cases) >= max_cases:
                    return cases
    return cases


def build_multi_step_eval_cases_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    max_cases: int,
    case_family: str = "all",
) -> list[RetrievalEvalCase]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for chunk in chunks:
        if str(chunk.get("chunk_type", "")) != "table_record":
            continue
        metadata = dict(chunk.get("metadata_json", {}))
        if not metadata.get("table_cell"):
            continue
        key = _row_group_key(chunk)
        if key is None:
            continue
        grouped.setdefault(key, []).append(chunk)

    cases: list[RetrievalEvalCase] = []
    seen_queries: list[str] = []
    if case_family in {"all", "sibling_table_rows"}:
        for cells in grouped.values():
            prompt_cells = [
                cell
                for cell in cells
                if any(term in _column_field(cell) for term in {"error message", "symptom"})
                and _good_multi_step_anchor(_cell_value(str(cell.get("content", ""))))
            ]
            for prompt_cell in prompt_cells:
                prompt_value = _cell_value(str(prompt_cell.get("content", "")))
                cause_cell = _best_related_cell(cells, {"cause"}, prompt_value)
                action_cell = _best_related_cell(cells, {"corrective action", "countermeasure", "remedy"}, prompt_value)
                if not cause_cell or not action_cell or len({prompt_cell["id"], cause_cell["id"], action_cell["id"]}) < 3:
                    continue
                label = _safe_query_label(prompt_cell)
                prompt_anchor = _short_answer_anchor(prompt_value)
                if not prompt_anchor:
                    continue
                if "symptom" in _column_field(prompt_cell):
                    query = f"What causes {prompt_anchor}{_for_label(label)}, and what should be checked or corrected?"
                else:
                    query = f"What causes {prompt_anchor}{_for_label(label)}, and how should it be corrected?"
                if _has_near_duplicate_query(query, seen_queries):
                    continue
                evidence = _multi_step_expected_evidence(prompt_cell, cause_cell, action_cell)
                expected_terms = []
                for item in evidence:
                    for term in item["expected_terms"]:
                        if term not in expected_terms:
                            expected_terms.append(term)
                        if len(expected_terms) >= 6:
                            break
                    if len(expected_terms) >= 6:
                        break
                if len(expected_terms) < 4:
                    continue
                seen_queries.append(query)
                cases.append(
                    RetrievalEvalCase(
                        case_id=f"{prompt_cell['id']}::multi_step::{len(cases) + 1}",
                        query=query,
                        source_document_id=str(prompt_cell["source_document_id"]),
                        document_version_id=str(prompt_cell["document_version_id"]),
                        source_chunk_id=str(prompt_cell["id"]),
                        source_title=str(prompt_cell.get("title", "")),
                        source_filename=str(prompt_cell.get("source_filename", "")),
                        chunk_type=str(prompt_cell.get("chunk_type", "")),
                        section_path=str(prompt_cell.get("section_path_text", "")),
                        page_from=int(prompt_cell.get("page_from", 0)),
                        page_to=int(prompt_cell.get("page_to", 0)),
                        expected_terms=expected_terms[:6],
                        expected_snippet=" | ".join(item["snippet"] for item in evidence),
                        generation_method="table_sibling_error_cause_action",
                        source_metadata=dict(prompt_cell.get("metadata_json", {})),
                        benchmark_quality="validated",
                        anchor_terms=expected_terms[:6],
                        retrieval_task="multi_step_retrieval",
                        expected_source_chunk_ids=[item["chunk_id"] for item in evidence],
                        expected_evidence=evidence,
                    )
                )
                if len(cases) >= max_cases:
                    return cases
    if case_family in {"all", "contextual_section"} and len(cases) < max_cases:
        cases.extend(
            _build_contextual_multi_step_cases(
                chunks,
                max_cases=max_cases - len(cases),
                seen_queries=seen_queries,
            )
        )
    if case_family not in {"all", "sibling_table_rows", "contextual_section"}:
        raise ValueError(f"Unsupported multi-step case family: {case_family}")
    return cases


def _term_matches_evidence(term: str, evidence_text: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    haystack = normalize_text(evidence_text)
    if normalized_term in haystack:
        return True
    evidence_tokens = set(tokenize(evidence_text))
    term_tokens = tokenize(normalized_term)
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        token = term_tokens[0]
        if "/" in token:
            parts = [part for part in token.split("/") if part and part not in STOPWORDS]
            return bool(parts) and all(part in evidence_tokens for part in parts)
        return token in evidence_tokens
    return all(token in evidence_tokens for token in term_tokens if token not in STOPWORDS)


def _result_evidence_text(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    section_path = result.get("section_path", [])
    if isinstance(section_path, list):
        section_text = " ".join(str(part) for part in section_path)
    else:
        section_text = str(section_path)
    metadata_values = [
        metadata.get("context_window"),
        metadata.get("table_row_group_context"),
        metadata.get("parent_context"),
        metadata.get("product_model"),
        metadata.get("chunk_type"),
        metadata.get("row_header"),
        metadata.get("column_header"),
        metadata.get("table_title"),
    ]
    return " ".join(
        str(part)
        for part in [
            result.get("content", ""),
            result.get("title", ""),
            section_text,
            *metadata_values,
        ]
        if part
    )


def _result_term_overlap(result: dict[str, Any], expected_terms: list[str]) -> int:
    evidence_text = _result_evidence_text(result)
    return sum(1 for term in expected_terms if _term_matches_evidence(term, evidence_text))


def _score_multi_step_search_results(
    case: RetrievalEvalCase,
    results: list[dict[str, Any]],
    *,
    top_k: int,
    document_selection: dict[str, Any],
) -> dict[str, Any]:
    considered = results[:top_k]
    expected_evidence = case.expected_evidence or []
    if not expected_evidence:
        expected_evidence = [
            {"chunk_id": chunk_id, "expected_terms": case.expected_terms}
            for chunk_id in (case.expected_source_chunk_ids or [case.source_chunk_id])
        ]
    found_same_document = any(str(result.get("source_document_id", "")) == case.source_document_id for result in considered)
    matched_items: list[dict[str, Any]] = []
    missing_items: list[dict[str, Any]] = []
    best_rank: int | None = None
    for item in expected_evidence:
        chunk_id = str(item.get("chunk_id") or "")
        terms = [str(term) for term in item.get("expected_terms", []) if str(term)]
        matched = False
        item_rank: int | None = None
        max_overlap = 0
        for rank, result in enumerate(considered, start=1):
            same_chunk = chunk_id and str(result.get("chunk_id", "")) == chunk_id
            same_document = str(result.get("source_document_id", "")) == case.source_document_id
            overlap = _result_term_overlap(result, terms)
            max_overlap = max(max_overlap, overlap)
            required_overlap = max(1, min(2, len(terms)))
            if same_chunk or (same_document and overlap >= required_overlap):
                matched = True
                item_rank = rank
                best_rank = rank if best_rank is None else min(best_rank, rank)
                break
        record = {"chunk_id": chunk_id, "matched": matched, "rank": item_rank, "overlap_terms": max_overlap}
        if matched:
            matched_items.append(record)
        else:
            missing_items.append(record)
    passed = bool(expected_evidence) and not missing_items
    failure_category = None if passed else ("ranking_or_context_loss" if found_same_document else "candidate_miss")
    return {
        "passed": passed,
        "rank": best_rank if passed else None,
        "match_reason": "multi_step_expected_evidence" if passed else "multi_step_missing_expected_evidence",
        "matched_evidence": matched_items,
        "missing_evidence": missing_items,
        "failure_category": failure_category,
        "retrieval_stage": "final_top_k",
        "candidate_recall": found_same_document,
        "metadata_document_selection": document_selection,
    }


def _query_evidence_overlap(query: str, result: dict[str, Any]) -> int:
    ignored = STOPWORDS.union(GENERIC_ANCHORS).union({"new", "series"})
    evidence_text = _result_evidence_text(result)
    query_terms = []
    for token in tokenize(query):
        if token in ignored:
            continue
        if token not in query_terms:
            query_terms.append(token)
    return sum(1 for term in query_terms if _term_matches_evidence(term, evidence_text))


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
    if case.retrieval_task == "multi_step_retrieval":
        return _score_multi_step_search_results(case, results, top_k=top_k, document_selection=document_selection)
    considered = results[:top_k]
    found_same_document = False
    found_chunk_family = False
    max_overlap = 0
    max_query_overlap = 0
    for rank, result in enumerate(considered, start=1):
        same_document = str(result.get("source_document_id", "")) == case.source_document_id
        same_chunk = str(result.get("chunk_id", "")) == case.source_chunk_id
        same_section = " / ".join(result.get("section_path", [])) == case.section_path
        overlap = _result_term_overlap(result, case.expected_terms)
        query_overlap = _query_evidence_overlap(case.query, result)
        max_overlap = max(max_overlap, overlap)
        max_query_overlap = max(max_query_overlap, query_overlap)
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
        if same_document and overlap >= 2 and query_overlap >= 2:
            return {
                "passed": True,
                "rank": rank,
                "match_reason": "same_document_answerable_evidence",
                "overlap_terms": overlap,
                "query_overlap_terms": query_overlap,
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
        "query_overlap_terms": max_query_overlap,
        "failure_category": failure_category,
        "retrieval_stage": "final_top_k",
        "candidate_recall": found_same_document,
        "metadata_document_selection": document_selection,
    }
