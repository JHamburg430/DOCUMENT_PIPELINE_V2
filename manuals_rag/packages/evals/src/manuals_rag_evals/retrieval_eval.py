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
    "assigned",
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
    "check",
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

ANSWER_SCORING_GENERIC_TERMS = QUERY_DEDUPE_FILLER.union(
    GENERIC_ANCHORS,
    {
        "as",
        "by",
        "description",
        "descriptions",
        "detail",
        "details",
        "flag",
        "whether",
        "executed",
        "having",
        "inputting",
        "operation",
        "operations",
        "pr",
        "procedure",
        "purpose",
        "purposes",
        "same",
        "typical",
        "type",
    },
)

ANSWER_SCORING_NUMBER_WORDS = {
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
}

ANSWER_SCORING_QUANTITY_QUERY_TERMS = {
    "count",
    "counts",
    "how many",
    "line count",
    "number",
    "numbers",
    "quantity",
    "quantities",
    "value",
    "values",
}

ANSWER_SCORING_SENTENCE_SKIP_TERMS = {
    "minimum",
    "maximum",
    "min",
    "max",
    "range",
    "ranges",
}

ANSWER_SCORING_ACTION_VERBS = {
    "adjust",
    "change",
    "check",
    "close",
    "connect",
    "delete",
    "disable",
    "enable",
    "execute",
    "extend",
    "format",
    "initialize",
    "increase",
    "reduce",
    "register",
    "remove",
    "replace",
    "restart",
    "review",
    "select",
    "set",
    "verify",
    "wait",
}

ANSWER_SCORING_STATE_TERMS = {
    "off",
    "offline",
    "on",
    "online",
    "disable",
    "disabled",
    "enable",
    "enabled",
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
    "measurement",
    "range",
    "symbol",
    "identifier",
    "mac",
    "password",
}

USER_STYLE_QUERY_SYSTEM_PROMPT = """
You generate realistic retrieval benchmark queries for technical documents.

Return strict JSON with this shape:
{"queries":[{"query":"...","intent":"...","reason":"..."}]}

Rules:
- Write concise question-form queries a real technician, engineer, operator, purchaser, or integrator might ask.
- Base every query on the provided context, especially the source snippet, structured fields, labels, and extracted terms.
- Make each query answerable from the source snippet itself, not merely from a surrounding section.
- Represent the kind of question a user would ask before seeing the answer text; do not turn source wording into a keyword query.
- Include enough fair discriminators that the intended row, warning, step, or spec can be found without reading adjacent context.
- Use product names, model numbers, protocol names, units, and standardized technical terms as anchors when needed.
- Treat source field labels, table headers, row headers, and UI labels as concepts to paraphrase, not text to copy verbatim.
- If a label or snippet contains a compound phrase, break it apart, reorder it, or replace part of it with a natural synonym.
- If the source uses bracketed UI labels like [Output Setting], rewrite them into natural user wording. Do not include square brackets in the query.
- Do not copy any other exact sentence, clause, or two-or-more-word phrase from the snippet into the question.
- Paraphrase awkward or document-authored wording into natural user language; for example, do not copy phrases like "obtained authentication" or "there manners".
- For compact specs or table rows, ask about the field, setting, action, or constraint in natural language and include one concrete value/unit/class/action when useful.
- Do not say "this document", "this manual", "the datasheet", "this section", or similar.
- Do not use meta phrasing like "what specification", "what value is listed", "where does", "what does the document say", or "which step in".
- Do not mirror the source text mechanically.
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

USER_STYLE_QUERY_FEW_SHOT_EXAMPLES = """
Generic examples to imitate. These examples are source-neutral and are not about the current document.

Example 1:
Input facts: Product: MODEL-A. Source says: Power supply voltage: 24 VDC.
Bad query: What power supply voltage is listed for MODEL-A?
Good query: What voltage does MODEL-A need for power?

Example 2:
Input facts: Source says: Disconnect all other devices before checking EtherNet/IP communication.
Bad query: Which disconnect all other devices detail is needed?
Good query: What should stay unplugged while I check EtherNet/IP communication?

Example 3:
Input facts: Table row is MODEL-B. Column is measurement range. Cell value is 72 mm.
Bad query: Which measurement range entry applies to MODEL-B?
Good query: What measurement range does MODEL-B cover?

Example 4:
Input facts: UI label is [Output Format]. Value is binary.
Bad query: How is [Output Format] configured?
Good query: Which output format should I choose for binary transfer?

Example 5:
Input facts: Source explains that the endian option controls PLC byte order.
Bad query: What Endian is a method of uses byte data in the PLC data?
Good query: Which endian setting matches my PLC byte order?

Example 6:
Input facts: UI label is Reference Detection Position. Source says it displays a detection coordinate used as the reference position.
Bad query: What purpose does the displayed detection coordinate serve?
Good query: Which coordinate becomes the position reference?

Example 7:
Input facts: Source says the tag PLC1: I.Data[0].1 is ON when the Command error flag is assigned.
Bad query: Is PLC1: I.Data0.1 the correct indicator for Command errors?
Good query: Which tag indicates whether a command error occurred?

Pattern:
- The good query sounds like a person asking before they have seen the answer.
- It keeps necessary anchors such as model names, protocols, units, and settings, but avoids copying exact addresses, tag prefixes, or code-like identifiers unless the user would already know that identifier.
- It does not copy source sentence structure, labels, bracket syntax, or awkward phrasing.
- It avoids reusing full source noun phrases such as "displayed detection coordinate" when shorter wording can ask the same thing.
- It never uses "detail is needed", "is listed", "entry applies", "purpose does", or source-like grammar.
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


def _strip_bracket_label(label: str) -> str:
    return re.sub(r"^\[|\]$", "", label.strip()).strip()


def _unbracket_source_labels(content: str) -> str:
    return re.sub(r"\[([^\]]+)\]", r"\1", content)


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
    if re.match(r"^\[[^\]]+\]\s*\(page\s+\d+(?:-\d+)?\)", compact):
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


def _table_cell_value(content: str) -> str:
    match = re.search(r"Cell value:\s*([^;]+)", content)
    return match.group(1).strip() if match else ""


def _looks_like_placeholder_table_cell(content: str) -> bool:
    value = _table_cell_value(content)
    if not value:
        return False
    normalized = normalize_text(value)
    if normalized in {"-", "--", "n/a", "na", "none", "not applicable"}:
        return True
    if not re.search(r"[A-Za-z0-9]", value):
        return True
    stripped = re.sub(r"[\s\u2713\u2714\u25cb\u25cf\u25ce\u25a0\u25a1\u25b3\u25b2\u25bd\u25bc\u25c7\u25c6\uf0fc]+", "", value)
    return stripped == ""


def _looks_like_cross_reference_only(content: str) -> bool:
    compact = normalize_text(content)
    if not re.search(r"\b(?:refer to|see)\b", compact):
        return False
    if re.search(r"\b(?:set|connect|install|select|enter|enable|disable|measure|adjust|capture|trigger|store|stores)\b", compact):
        return False
    if re.search(r"\b\d+(?:\.\d+)?(?:v|a|ma|w|kw|mm|cm|m|ms|s|hz|khz|mhz|fps|kg|g|n|mpa|°c|c|%)\b", compact):
        return False
    return len(compact) < 180


def _looks_like_numbered_step_fragment(content: str) -> bool:
    compact_tokens = [token for token in tokenize(content) if token not in STOPWORDS]
    if len(compact_tokens) > 18:
        return False
    joined = " ".join(compact_tokens)
    return bool(re.search(r"\b\d+[a-z]{3,}\b", joined) and re.search(r"\b(?:left-click|click|restart|ok)\b", joined))


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
        and metadata.get("table_header")
        and not metadata.get("table_cell")
        and not metadata.get("table_key_value")
        and not metadata.get("table_row_group")
    ):
        return False
    if (
        chunk_type == "table_record"
        and metadata.get("table_cell")
        and not _metadata_list(metadata, "table_row_headers")
        and not _meaningful_table_field_value_pairs(content)
    ):
        return False
    if (
        chunk_type == "table_record"
        and metadata.get("table_cell")
        and _looks_like_placeholder_table_cell(content)
        and not _meaningful_table_field_value_pairs(content)
    ):
        return False
    if _looks_like_toc_line(content):
        return False
    if _looks_like_legal_boilerplate(content):
        return False
    if chunk_type == "atomic_text" and _looks_like_cross_reference_only(content):
        return False
    if chunk_type == "atomic_text" and _looks_like_numbered_step_fragment(content):
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


def _allowed_copied_source_phrases(chunk: dict[str, Any]) -> set[tuple[str, ...]]:
    metadata = dict(chunk.get("metadata_json", {}))
    allowed_texts: list[str] = [
        str(chunk.get("product_model") or metadata.get("product_model") or ""),
        str(metadata.get("product_family") or ""),
        _safe_query_label(chunk),
    ]

    allowed: set[tuple[str, ...]] = set()
    for text in allowed_texts:
        tokens = tuple(tokenize(text))
        for size in (2, 3, 4, 5):
            for index in range(0, max(0, len(tokens) - size + 1)):
                allowed.add(tokens[index : index + size])
    return allowed


def _is_allowed_exact_source_ngram(ngram: tuple[str, ...], allowed_phrases: set[tuple[str, ...]]) -> bool:
    if ngram in allowed_phrases:
        return True
    if any(re.search(r"\d", token) for token in ngram):
        return True
    if any("/" in token for token in ngram):
        return True
    return False


def _query_copies_unfair_source_phrase(query: str, chunk: dict[str, Any]) -> bool:
    query_tokens = tokenize(query)
    content_tokens = tokenize(str(chunk.get("content", "")))
    if len(query_tokens) < 2 or len(content_tokens) < 2:
        return False

    allowed_phrases = _allowed_copied_source_phrases(chunk)

    copied_pairs: set[tuple[str, str]] = set()
    meaningful_content = [
        token
        for token in content_tokens
        if token not in STOPWORDS
        and token not in GENERIC_ANCHORS
        and not _is_high_signal_anchor(token)
        and len(token) >= 4
    ]
    for index, first in enumerate(meaningful_content):
        for second in meaningful_content[index + 1 : index + 3]:
            copied_pairs.add((first, second))

    for index in range(0, len(query_tokens) - 1):
        pair = tuple(query_tokens[index : index + 2])
        if _is_allowed_exact_source_ngram(pair, allowed_phrases):
            continue
        if pair in copied_pairs:
            return True
    return False


def _query_uses_bracketed_source_label(query: str, chunk: dict[str, Any]) -> bool:
    if re.search(r"\[[^\]]+\]", query):
        return True
    query_tokens = tokenize(query)
    if not query_tokens:
        return False
    for label in _quoted_menu_labels(str(chunk.get("content", ""))):
        label_tokens = tokenize(_strip_bracket_label(label))
        if len(label_tokens) < 2:
            continue
        for index in range(0, len(query_tokens) - len(label_tokens) + 1):
            if query_tokens[index : index + len(label_tokens)] == label_tokens:
                return True
    return False


def _query_uses_source_address_syntax(query: str, chunk: dict[str, Any]) -> bool:
    content = str(chunk.get("content", ""))
    if not re.search(r"\b(?:tag|address|bit|flag|register|command|plc|i/o|input|output)\b", content, flags=re.IGNORECASE):
        return False
    metadata = dict(chunk.get("metadata_json", {}))
    allowed_label_tokens = set(tokenize(_safe_query_label(chunk))).union(tokenize(str(metadata.get("product_model") or "")))
    query_tokens = set(tokenize(query))
    copied_prefixes = {
        prefix.lower()
        for prefix in re.findall(r"\b([A-Z][A-Z0-9_-]{2,})\s*:\s*[A-Za-z][A-Za-z0-9_]*(?:\.|\[|\d)", content)
    }
    if any(prefix in query_tokens and prefix not in allowed_label_tokens for prefix in copied_prefixes):
        return True
    address_patterns = (
        r"\b[A-Z][A-Z0-9_-]*\s*:\s*[A-Za-z][A-Za-z0-9_]*(?:\.|\[|\d)[A-Za-z0-9_.\[\]]*",
        r"\b[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_.]*\b",
        r"\b[A-Za-z]\.[A-Za-z0-9_]+(?:\d|\[[0-9]+\]|\.)[A-Za-z0-9_.\[\]]*\b",
    )
    return any(re.search(pattern, query) for pattern in address_patterns)


def _query_uses_table_artifact_syntax(query: str, chunk: dict[str, Any]) -> bool:
    if str(chunk.get("chunk_type", "")) != "table_record":
        return False
    lowered = normalize_text(query)
    if re.search(r"\b(?:row|column)\s+\d+\b", lowered):
        return True
    table_artifact_phrases = {
        "cell value",
        "column header",
        "column headers",
        "description measurement",
        "description of",
        "description of measurement",
        "row header",
        "row headers",
        "row number",
        "table header",
    }
    return any(phrase in lowered for phrase in table_artifact_phrases)


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


def _query_looks_mechanical(query: str) -> bool:
    tokens = tokenize(query)
    if len(tokens) > 24:
        return True
    compact_tokens = [token for token in tokens if token not in STOPWORDS and token not in GENERIC_ANCHORS]
    if len(compact_tokens) > 18:
        return True
    source_like_tokens = [
        token
        for token in compact_tokens
        if len(token) >= 12 and (re.search(r"\d", token) or token.isupper())
    ]
    if len(source_like_tokens) >= 2:
        return True
    if len(re.findall(r"\b[A-Z][A-Z0-9]*\[\]", query)) >= 3:
        return True
    if re.search(r"\.\s*\.\s*\.", query):
        return True
    if len(re.findall(r"\.(?:bmp|jpg|jpeg|png|pdf|csv)\b", query, flags=re.IGNORECASE)) >= 2:
        return True
    if re.search(r"\b[A-Z0-9]{4,}\*", query):
        return True
    compact = normalize_text(query)
    if re.search(r"\bwhat\s+items?\b", compact) or re.search(r"\bwhat\s+[^?]{1,40}\s+for\s+items?\b", compact):
        return True
    if (
        compact.startswith("what ")
        and compact.endswith(" applies")
        and len(compact_tokens) <= 3
        and not any(re.search(r"\d", token) for token in compact_tokens)
    ):
        return True
    if re.search(r"\bwhat\s+\S+\s+is\s+described\s+for\b", compact):
        described_subject = compact.split(" is described for ", 1)[0].removeprefix("what ").strip()
        described_terms = [token for token in tokenize(described_subject) if token not in STOPWORDS and token not in GENERIC_ANCHORS]
        if len(described_terms) <= 1:
            return True
    if re.search(r"\bwhat\s+.+\s+is\s+described\s+for\b", compact):
        return True
    applies_match = re.search(r"\bwhat\s+(.+?)\s+applies\s+(?:to|for)\b", compact)
    if applies_match:
        subject_terms = [token for token in tokenize(applies_match.group(1)) if token not in STOPWORDS and token not in GENERIC_ANCHORS]
        has_domain_term = any(token in GENERIC_TECHNICAL_TERMS or token in TECHNICAL_VERBS for token in subject_terms)
        if len(subject_terms) <= 3 and not has_domain_term and not any(re.search(r"\d|/|-", token) for token in subject_terms):
            return True
    if re.search(r"\bwhat\s+.+\s+value\s+applies\s+(?:to|for)\b", compact):
        before_value = compact.split(" value applies ", 1)[0].removeprefix("what ").strip()
        before_terms = [token for token in tokenize(before_value) if token not in STOPWORDS]
        generic_table_terms = {
            "area",
            "bit",
            "contents",
            "description",
            "field",
            "identifier",
            "status",
            "symbol",
            "word",
        }
        if len(before_terms) >= 5 or sum(1 for token in before_terms if token in generic_table_terms) >= 2:
            return True
        if len(before_terms) >= 4 and len(before_terms) != len(set(before_terms)):
            return True
    if re.search(r"\bwhat\s+\[[^\]]+\]\s+\w+\s+applies\s+(?:to|for)\b", query, flags=re.IGNORECASE):
        return True
    if re.search(r"\bwhat\s+[\d.]+\s+[a-z]+-?\d+\s+applies\s+(?:to|for)\b", compact):
        return True
    if re.search(r"\bwhat\s+\w+\s+when\s*=\s*.+\s+value\s+applies\s+(?:to|for)\b", compact):
        return True
    bracketed_phrases = re.findall(r"\[[^\]]+\]", query)
    if sum(len(tokenize(phrase)) for phrase in bracketed_phrases) > 6:
        return True
    return False


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
    if _query_looks_mechanical(query):
        return False, "mechanical_query"
    if _query_uses_filename_artifact(query, chunk):
        return False, "filename_artifact_query"
    if _query_uses_bracketed_source_label(query, chunk):
        return False, "bracketed_source_label_query"
    if _query_uses_source_address_syntax(query, chunk):
        return False, "source_address_syntax_query"
    if _query_uses_table_artifact_syntax(query, chunk):
        return False, "table_artifact_syntax_query"
    if _query_copies_unfair_source_phrase(query, chunk):
        return False, "copied_source_phrase"
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
    generic_model_label = re.search(
        r"\b(?:user'?s?\s+manual|instruction\s+manual|setup\s+guide|installation\s+guide|operation\s+manual)\b",
        model,
        flags=re.IGNORECASE,
    )
    if model and not generic_model_label and len(model) <= 60 and model.count("/") <= 3 and not _query_uses_filename_artifact(model, chunk):
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
    prompt_content = _unbracket_source_labels(content)
    field_matches = _field_value_pairs(content)
    return {
        "chunk_type": str(chunk.get("chunk_type", "")),
        "title": _safe_query_label(chunk),
        "product_model": _safe_query_label(chunk),
        "section_path": str(chunk.get("section_path_text", "")).strip(),
        "anchors": anchors[:6],
        "menu_labels": [_strip_bracket_label(label) for label in _quoted_menu_labels(content)[:3]],
        "field_value_pairs": [
            {"field": field.strip(), "value": value.strip()}
            for field, value in field_matches[:6]
        ],
        "expected_terms": anchors[:4],
        "snippet": content_preview(prompt_content, limit=420),
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
    metadata = dict(chunk.get("metadata_json", {}))
    row_headers = _metadata_list(metadata, "table_row_headers")
    column_headers = _metadata_list(metadata, "table_column_headers")
    content = normalize_text(str(chunk.get("content", "")))
    candidates: list[tuple[str, str]] = []
    if "symbol" in expected_terms and "identifier" in expected_terms and "enabled" in expected_terms:
        candidates.append(
            (
                f"Which setting adds a symbol identifier{_for_label(label)}?",
                "table_symbol_identifier_setting",
            )
        )
    if any("decimal" in normalize_text(header) and "digit" in normalize_text(header) for header in column_headers):
        subject = "LumiTrax capture" if "lumitrax" in content else "the selected setting"
        candidates.append(
            (
                f"How many decimal places are used for {subject}{_for_label(label)}?",
                "table_decimal_places_setting",
            )
        )
    if "luminance" in expected_terms and "analog" in expected_terms and "voltage" in expected_terms:
        candidates.append(
            (
                f"Which analog signal voltage should {label or 'the device'} output?",
                "table_luminance_output_signal",
            )
        )
    if len(field_pairs) >= 2:
        answer_field, _ = field_pairs[0]
        context_field, context_value = field_pairs[1]
        context_anchor = " ".join(extract_anchor_terms(context_value, limit=3)) or context_value.strip()
        answer_field_text = answer_field.strip()
        context_field_text = context_field.strip()
        if label:
            candidates.extend(
                [
                    (
                        f"For {label}, which {answer_field_text} is used with {context_field_text} {context_anchor}?",
                        "table_key_value_lookup",
                    ),
                    (
                        f"Which {answer_field_text} matches {context_field_text} {context_anchor} on {label}?",
                        "table_key_value_reverse_lookup",
                    ),
                ]
            )
            return candidates
        candidates.append((f"Which {answer_field_text} is used with {context_field_text} {context_anchor}?", "table_key_value_lookup"))
        return candidates
    if not query_terms or len(expected_terms) < 2:
        return candidates
    subject = " ".join(row_headers[:1] or query_terms[:1])
    column_label = " ".join(column_headers[:2]) or (query_terms[-1] if query_terms else "value")
    cell_anchor = expected_terms[0] if expected_terms and _is_high_signal_anchor(expected_terms[0]) else ""
    if label:
        if "description of measurement" in normalize_text(column_label):
            candidates.extend(
                [
                    (f"What measurement does {subject} represent{_for_label(label)}?", "table_measurement_description_lookup"),
                    (f"For {label}, what does {subject} measure?", "table_measurement_subject_lookup"),
                ]
            )
            return candidates
        candidates.extend(
            [
                (f"What value does {label} use for {subject} {column_label}?", "table_row_column_lookup"),
                (f"For {label}, which {subject} setting uses {column_label}?", "table_row_column_engineer_lookup"),
            ]
        )
        if cell_anchor:
            candidates.append((f"For {label}, which {subject} entry reports {cell_anchor}?", "table_row_column_reverse_lookup"))
        return candidates
    candidates.append((f"What value is used for {subject} {column_label}?", "table_row_column_lookup"))
    return candidates


def _parse_generated_queries(payload: str) -> list[dict[str, str]]:
    data = _loads_generated_query_payload(payload)
    if isinstance(data, list):
        queries = data
    else:
        queries = data.get("queries", [])
    parsed: list[dict[str, str]] = []
    for item in queries:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        intent = str(item.get("intent", "")).strip() or "llm_user_style"
        reason = str(item.get("reason", "")).strip()
        if query:
            parsed.append({"query": query, "intent": intent, "reason": reason})
    return parsed


def _loads_generated_query_payload(payload: str) -> dict[str, Any] | list[Any]:
    text = _strip_generated_json_wrappers(payload)
    if not text:
        raise ValueError("Model returned an empty generated-query response")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = json.loads(_extract_balanced_json(text))
    if not isinstance(loaded, (dict, list)):
        raise ValueError("Generated-query response must be a JSON object or array")
    return loaded


def _strip_generated_json_wrappers(payload: str) -> str:
    text = payload.strip()
    if not text:
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    fence = re.fullmatch(r"(?is)```(?:json)?\s*(.*?)\s*```", text)
    if fence:
        return fence.group(1).strip()
    return text


def _extract_balanced_json(text: str) -> str:
    starts = [index for index, char in enumerate(text) if char in "[{"]
    for start in starts:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    raise ValueError("Generated-query response did not contain a valid JSON object")


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
    }
    try:
        with httpx.Client(base_url=settings.ollama_url, timeout=20.0) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": settings.ollama_eval_model,
                    "prompt": (
                        f"{USER_STYLE_QUERY_SYSTEM_PROMPT}\n\n"
                        f"{USER_STYLE_QUERY_FEW_SHOT_EXAMPLES}\n\n"
                        f"Current input: {json.dumps(prompt, ensure_ascii=True)}"
                    ),
                    "stream": False,
                    "format": "json",
                    "think": False,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            response_text = str(payload.get("response") or payload.get("thinking") or "")
            generated = _parse_generated_queries(response_text)
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
        is_valid, _ = validate_eval_case(query, chunk, anchors)
        if not is_valid:
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
        fallback_candidates = build_query_candidates(chunk)[: max(per_chunk_limit * 4, per_chunk_limit)]
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


def _looks_like_parser_artifact(text: str) -> bool:
    if "_" in text or "*" in text:
        return True
    if re.search(r"[\ue000-\uf8ff\ufffd]", text):
        return True
    tokens = tokenize(text)
    if sum(1 for token in tokens if "[]" in token or "_" in token) >= 2:
        return True
    if re.search(r"\b[A-Za-z0-9_]+\[\]\.?", text):
        return True
    return False


def _looks_like_page_reference_fragment(text: str) -> bool:
    compact = normalize_text(text)
    page_refs = re.findall(r"\bpage\s+\d+(?:-\d+)?\b", compact)
    if page_refs:
        return True
    return bool(re.search(r"\b\d+-\d+\b", compact) and re.search(r"\b(?:page|section)\b", compact))


def _looks_like_truncated_source_fragment(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return False
    if stripped.count("(") != stripped.count(")") or stripped.count('"') % 2 == 1 or stripped.count("\u201c") != stripped.count("\u201d"):
        return True
    if re.match(r"^[A-Za-z]\s+[A-Z]", stripped):
        return True
    if len(stripped) >= 85 and not re.search(r"[.!?)]$", stripped):
        return True
    compact = normalize_text(stripped)
    if compact.endswith((" c", " im", " po", " to", " wit", " cannot", "distribu", "increas")):
        return True
    if len(tokenize(stripped)) > 14:
        return True
    return False


def _multi_step_expected_evidence(*cells: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for cell in cells:
        metadata = dict(cell.get("metadata_json", {}))
        value = _cell_value(str(cell.get("content", "")))
        terms = extract_anchor_terms(value, limit=4)
        if not terms:
            terms = extract_anchor_terms(str(cell.get("content", "")), limit=4)
        evidence.append(
            {
                "chunk_id": str(cell.get("id", "")),
                "source_document_id": str(cell.get("source_document_id", "")),
                "field": _column_field(cell),
                "label": _safe_query_label(cell),
                "product_identifiers": sorted(_metadata_product_identifiers(metadata)),
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


def _operational_step_subject(chunk: dict[str, Any]) -> str:
    content = str(chunk.get("content", ""))
    if str(chunk.get("chunk_type", "")) == "procedure_record":
        return _procedure_subject(content)
    subject = re.sub(r"^\d+\s+", "", content).strip()
    return _short_answer_anchor(subject)


def _warning_step_action_phrase(subject: str) -> str:
    phrase = re.sub(r"\s+", " ", subject).strip()
    if not phrase:
        return phrase
    phrase = phrase[0].lower() + phrase[1:]
    return phrase.rstrip(".")


def _good_operational_step_chunk(chunk: dict[str, Any]) -> bool:
    chunk_type = str(chunk.get("chunk_type", ""))
    if chunk_type not in {"procedure_record", "atomic_text"}:
        return False
    content = str(chunk.get("content", ""))
    anchors = extract_anchor_terms(content, limit=5)
    if chunk_type == "procedure_record":
        return chunk_is_queryworthy(chunk, anchors)
    if _looks_like_toc_line(content) or _looks_like_legal_boilerplate(content):
        return False
    compact = normalize_text(content)
    if re.match(r"^(?:do not|never|caution|warning|notice|if|failing to)\b", compact):
        return False
    has_step_shape = re.match(r"^\d+\s+[a-z]", compact) is not None
    has_action = any(term in TECHNICAL_VERBS for term in tokenize(content))
    return len(content) >= 45 and len(anchors) >= 2 and (has_step_shape or has_action)


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
    return _short_answer_anchor(content)


def _natural_contextual_fragment(subject: str) -> str:
    fragment = re.sub(r"\bPage\s+\d+(?:-\d+)?\b", "", subject, flags=re.IGNORECASE)
    fragment = re.sub(r"\b(?:Row|Column)\s*:?\s*\d+\b", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"\b(?:Cell value|Column headers?|Row headers?)\s*:?", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"\s+", " ", fragment).strip(" ;:-,.")
    fragment = re.sub(r"^(?:the|a|an|on a|on an)\s+", "", fragment, flags=re.IGNORECASE)
    tokens = fragment.split()
    if len(tokens) > 10:
        fragment = " ".join(tokens[:10])
    if fragment:
        fragment = fragment[0].lower() + fragment[1:]
    return fragment


def _contextual_multi_step_query(procedure_subject: str, label: str, support_subject: str) -> str:
    procedure_fragment = _natural_contextual_fragment(procedure_subject)
    support_fragment = _natural_contextual_fragment(support_subject)
    if not procedure_fragment or not support_fragment:
        return ""
    label_fragment = _for_label(label).removeprefix(" for ").strip()
    procedure_action = _warning_step_action_phrase(procedure_fragment)
    if procedure_action.lower().startswith("when "):
        procedure_action = procedure_action[5:].strip()
        procedure_clause = f"when {procedure_action}"
    else:
        procedure_clause = f"when you need to {procedure_action}"
    if label_fragment:
        return f"For {label_fragment}, {procedure_clause}, what should be checked about {support_fragment}?"
    return f"{procedure_clause.capitalize()}, what should be checked about {support_fragment}?"


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


def _warning_subject(content: str) -> str:
    subject = re.sub(r"^(?:Warning|Caution|Notice)\s*:\s*", "", content, flags=re.IGNORECASE).strip()
    return _short_answer_anchor(subject)


def _good_warning_subject(subject: str) -> bool:
    compact = normalize_text(subject)
    if len(compact) < 12:
        return False
    if re.fullmatch(r"warning(?: code| number)?\s*\d+", compact):
        return False
    terms = extract_anchor_terms(subject, limit=5)
    return len(terms) >= 1 and any(_is_high_signal_anchor(term) or term in TECHNICAL_VERBS or len(term) >= 6 for term in terms)


def _chunks_are_warning_step_linked(warning: dict[str, Any], procedure: dict[str, Any]) -> bool:
    if str(warning.get("source_document_id", "")) != str(procedure.get("source_document_id", "")):
        return False
    if str(warning.get("document_version_id", "")) != str(procedure.get("document_version_id", "")):
        return False
    try:
        page_distance = abs(int(warning.get("page_from", 0) or 0) - int(procedure.get("page_from", 0) or 0))
    except (TypeError, ValueError):
        page_distance = 999
    if page_distance > 2:
        return False
    warning_terms = [
        term
        for term in extract_anchor_terms(str(warning.get("content", "")), limit=6)
        if term not in {"warning", "caution"}
    ]
    procedure_terms = [
        term
        for term in extract_anchor_terms(str(procedure.get("content", "")), limit=6)
        if term not in {"procedure", "typical"}
    ]
    if set(warning_terms).intersection(procedure_terms):
        return True
    warning_context = str(dict(warning.get("metadata_json", {})).get("local_rerank_context") or "")
    procedure_context = str(dict(procedure.get("metadata_json", {})).get("local_rerank_context") or "")
    warning_hits = sum(1 for term in warning_terms if _term_matches_evidence(term, procedure_context))
    procedure_hits = sum(1 for term in procedure_terms if _term_matches_evidence(term, warning_context))
    return warning_hits >= min(2, len(warning_terms)) or procedure_hits >= min(2, len(procedure_terms))


def _cross_document_lookup_field(chunk: dict[str, Any]) -> str:
    chunk_type = str(chunk.get("chunk_type", ""))
    if chunk_type == "table_record":
        field = _column_field(chunk)
        if field:
            return field
    field_pairs = _meaningful_table_field_value_pairs(str(chunk.get("content", "")))
    if field_pairs:
        return normalize_text(field_pairs[0][0])
    anchors = extract_anchor_terms(str(chunk.get("content", "")), limit=3)
    return " ".join(anchors[:2])


def _cross_document_lookup_value(chunk: dict[str, Any]) -> str:
    content = str(chunk.get("content", ""))
    if str(chunk.get("chunk_type", "")) == "table_record":
        value = _cell_value(content)
        if value:
            return value
    field_pairs = _meaningful_table_field_value_pairs(content)
    if field_pairs:
        return field_pairs[0][1]
    return _short_answer_anchor(content)


def _good_cross_document_field(field: str) -> bool:
    compact = normalize_text(field)
    if len(compact) < 4:
        return False
    if compact in STOPWORDS or compact in GENERIC_ANCHORS:
        return False
    if compact.startswith("description "):
        return False
    if compact in {
        "cell value",
        "value",
        "values",
        "description",
        "remarks",
        "reference",
        "page",
        "item",
        "items",
        "setting item",
        "model",
        "model name",
        "name",
    }:
        return False
    terms = [term for term in tokenize(compact) if term not in STOPWORDS and term not in GENERIC_ANCHORS]
    return bool(terms)


def _good_cross_document_label(label: str) -> bool:
    compact = normalize_text(label)
    if len(compact) < 4:
        return False
    if compact in STOPWORDS or compact in GENERIC_ANCHORS:
        return False
    tokens = tokenize(label)
    if len(tokens) == 1 and len(tokens[0]) < 4 and not any(char.isdigit() for char in tokens[0]):
        return False
    return bool(any(_is_high_signal_anchor(token) or len(token) >= 4 for token in tokens))


def _good_cross_document_value(value: str) -> bool:
    compact = normalize_text(value)
    if len(compact) < 2:
        return False
    if compact in {"-", "--", "n/a", "na", "none", "not applicable"}:
        return False
    return bool(extract_anchor_terms(value, limit=1) or re.search(r"\d", value))


def _cross_document_lookup_subject(chunk: dict[str, Any], *, field: str, value: str) -> str:
    row_subject = _row_header_text(chunk)
    row_subject = re.sub(r"\brow headers?\b", "", row_subject, flags=re.IGNORECASE).strip()
    candidates = [row_subject]
    for pair_field, pair_value in _meaningful_table_field_value_pairs(str(chunk.get("content", ""))):
        if normalize_text(pair_field) != normalize_text(field):
            candidates.append(f"{pair_field.strip()} {pair_value.strip()}".strip())
    candidates.append(_short_answer_anchor(value))
    for candidate in candidates:
        subject = re.sub(r"\s+", " ", candidate).strip(" ;:,-")
        if _good_cross_document_subject(subject, field=field, value=value):
            return subject
    return ""


def _good_cross_document_subject(subject: str, *, field: str, value: str) -> bool:
    compact = normalize_text(subject)
    if len(compact) < 4:
        return False
    if len(subject) > 80:
        return False
    if compact in STOPWORDS or compact in GENERIC_ANCHORS:
        return False
    if compact == normalize_text(field) or compact == normalize_text(value):
        return False
    if compact in {"row", "column", "item", "items", "value", "values", "description"}:
        return False
    if compact.startswith(("item ", "items ", "setting ", "settings ")):
        return False
    if _looks_like_parser_artifact(subject):
        return False
    if _looks_like_page_reference_fragment(subject):
        return False
    if _looks_like_truncated_source_fragment(subject):
        return False
    if sum(1 for token in tokenize(subject) if re.search(r"[a-z]+-\d|\d+[a-z]+|/", token)) >= 3:
        return False
    if re.search(r"\b(?:can be set|specify|specifies|set this option|select this option)\b", compact):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?%?", compact):
        return False
    terms = [term for term in extract_anchor_terms(subject, limit=4) if term not in {"row", "column"}]
    return bool(terms)


def _build_cross_document_multi_step_cases(
    chunks: list[dict[str, Any]],
    *,
    max_cases: int,
    seen_queries: list[str],
) -> list[RetrievalEvalCase]:
    candidates_by_field: dict[str, list[dict[str, Any]]] = {}
    cases: list[RetrievalEvalCase] = []
    for chunk in chunks:
        if int(chunk.get("chunk_level", 1) or 1) != 1:
            continue
        chunk_type = str(chunk.get("chunk_type", ""))
        if chunk_type not in {"table_record", "spec_record", "datasheet_record"}:
            continue
        metadata = dict(chunk.get("metadata_json", {}))
        if chunk_type == "table_record" and not metadata.get("table_cell"):
            continue
        if chunk_type == "table_record" and not _metadata_list(metadata, "table_row_headers"):
            continue
        anchors = extract_anchor_terms(str(chunk.get("content", "")), limit=4)
        if not chunk_is_queryworthy(chunk, anchors):
            continue
        field = _cross_document_lookup_field(chunk)
        value = _cross_document_lookup_value(chunk)
        label = _safe_query_label(chunk)
        if (
            not _good_cross_document_label(label)
            or not _good_cross_document_field(field)
            or not _good_cross_document_value(value)
        ):
            continue
        grouped = dict(chunk)
        grouped["_cross_doc_field"] = field
        grouped["_cross_doc_value"] = value
        grouped["_cross_doc_label"] = label
        candidates = candidates_by_field.setdefault(field, [])
        for left in candidates:
            left_label = str(left.get("_cross_doc_label", ""))
            left_value = str(left.get("_cross_doc_value", ""))
            left_terms = set(extract_anchor_terms(left_value, limit=4))
            if str(left.get("source_document_id", "")) == str(grouped.get("source_document_id", "")):
                continue
            if left_label == label:
                continue
            right_terms = set(extract_anchor_terms(value, limit=4))
            if left_terms and right_terms and left_terms == right_terms:
                continue
            left_subject = _cross_document_lookup_subject(left, field=field, value=left_value)
            right_subject = _cross_document_lookup_subject(grouped, field=field, value=value)
            if not left_subject or not right_subject:
                continue
            query = (
                f"For {left_label} and {label}, what {field} entries are listed for "
                f"{left_subject} and {right_subject}?"
            )
            if _has_near_duplicate_query(query, seen_queries):
                continue
            evidence = _multi_step_expected_evidence(left, grouped)
            expected_terms = []
            for item in evidence:
                for term in item["expected_terms"]:
                    if term not in expected_terms:
                        expected_terms.append(term)
                    if len(expected_terms) >= 6:
                        break
                if len(expected_terms) >= 6:
                    break
            if len(expected_terms) < 3:
                continue
            seen_queries.append(query)
            cases.append(
                RetrievalEvalCase(
                    case_id=f"{left['id']}::{grouped['id']}::cross_document_multi_step::{len(cases) + 1}",
                    query=query,
                    source_document_id=str(left["source_document_id"]),
                    document_version_id=str(left["document_version_id"]),
                    source_chunk_id=str(left["id"]),
                    source_title=str(left.get("title", "")),
                    source_filename=str(left.get("source_filename", "")),
                    chunk_type=str(left.get("chunk_type", "")),
                    section_path=str(left.get("section_path_text", "")),
                    page_from=int(left.get("page_from", 0)),
                    page_to=int(left.get("page_to", 0)),
                    expected_terms=expected_terms[:6],
                    expected_snippet=" | ".join(item["snippet"] for item in evidence),
                    generation_method="cross_document_same_field_evidence",
                    source_metadata=dict(left.get("metadata_json", {})),
                    benchmark_quality="validated",
                    anchor_terms=expected_terms[:6],
                    retrieval_task="multi_step_retrieval",
                    expected_source_chunk_ids=[item["chunk_id"] for item in evidence],
                    expected_evidence=evidence,
                )
            )
            if len(cases) >= max_cases:
                return cases
        if len(candidates) < 24 and not any(
            str(existing.get("_cross_doc_label", "")) == label
            and str(existing.get("source_document_id", "")) == str(grouped.get("source_document_id", ""))
            for existing in candidates
        ):
            candidates.append(grouped)
    return cases


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
                query = _contextual_multi_step_query(procedure_subject, label, support_subject)
                if not query:
                    continue
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


def _build_warning_step_multi_step_cases(
    chunks: list[dict[str, Any]],
    *,
    max_cases: int,
    seen_queries: list[str],
) -> list[RetrievalEvalCase]:
    by_document: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for chunk in chunks:
        if int(chunk.get("chunk_level", 1) or 1) != 1:
            continue
        key = (str(chunk.get("source_document_id", "")), str(chunk.get("document_version_id", "")))
        by_document.setdefault(key, []).append(chunk)

    cases: list[RetrievalEvalCase] = []
    for document_chunks in by_document.values():
        warnings = [
            chunk
            for chunk in document_chunks
            if str(chunk.get("chunk_type", "")) == "warning_record"
            and not _looks_like_toc_line(str(chunk.get("content", "")))
            and not _looks_like_legal_boilerplate(str(chunk.get("content", "")))
            and _good_warning_subject(_warning_subject(str(chunk.get("content", ""))))
        ]
        operational_steps = [
            chunk
            for chunk in document_chunks
            if _good_operational_step_chunk(chunk)
        ]
        for warning in warnings:
            warning_subject = _warning_subject(str(warning.get("content", "")))
            if not _good_warning_subject(warning_subject):
                continue
            for step in operational_steps:
                if not _chunks_are_warning_step_linked(warning, step):
                    continue
                step_subject = _operational_step_subject(step)
                if not _good_contextual_procedure_subject(step_subject):
                    continue
                label = _safe_query_label(step) or _safe_query_label(warning)
                action_phrase = _warning_step_action_phrase(step_subject)
                query = (
                    f"What warning or caution about {warning_subject}{_for_label(label)} "
                    f"applies when {action_phrase}?"
                )
                if _has_near_duplicate_query(query, seen_queries):
                    continue
                evidence = _multi_step_expected_evidence(step, warning)
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
                        case_id=f"{step['id']}::warning_step_multi_step::{len(cases) + 1}",
                        query=query,
                        source_document_id=str(step["source_document_id"]),
                        document_version_id=str(step["document_version_id"]),
                        source_chunk_id=str(step["id"]),
                        source_title=str(step.get("title", "")),
                        source_filename=str(step.get("source_filename", "")),
                        chunk_type=str(step.get("chunk_type", "")),
                        section_path=str(step.get("section_path_text", "")),
                        page_from=int(step.get("page_from", 0)),
                        page_to=int(step.get("page_to", 0)),
                        expected_terms=expected_terms[:6],
                        expected_snippet=" | ".join(item["snippet"] for item in evidence),
                        generation_method="warning_plus_step_evidence",
                        source_metadata=dict(step.get("metadata_json", {})),
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
    if case_family in {"all", "warning_step"} and len(cases) < max_cases:
        cases.extend(
            _build_warning_step_multi_step_cases(
                chunks,
                max_cases=max_cases - len(cases),
                seen_queries=seen_queries,
            )
        )
    if case_family in {"all", "cross_document"} and len(cases) < max_cases:
        cases.extend(
            _build_cross_document_multi_step_cases(
                chunks,
                max_cases=max_cases - len(cases),
                seen_queries=seen_queries,
            )
        )
    if case_family not in {"all", "sibling_table_rows", "contextual_section", "warning_step", "cross_document"}:
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


def _result_column_field(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    headers = _metadata_list(metadata, "table_column_headers")
    if headers:
        return normalize_text(" ".join(headers))
    pairs = _meaningful_table_field_value_pairs(str(result.get("content", "")))
    if pairs:
        return normalize_text(pairs[0][0])
    match = re.search(r"Column headers:\s*([^;]+)", str(result.get("content", "")))
    return normalize_text(match.group(1)) if match else ""


def _case_product_identifiers(case: RetrievalEvalCase) -> set[str]:
    metadata = dict(case.source_metadata or {})
    return _metadata_product_identifiers(metadata)


def _compact_eval_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _metadata_product_identifiers(metadata: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for key in ("product_model", "product_family"):
        value = metadata.get(key)
        if value:
            identifiers.add(str(value))
    for key in ("devices", "product_models", "product_families"):
        for value in metadata.get(key) or []:
            if value:
                identifiers.add(str(value))
    return {_compact_eval_identifier(value) for value in identifiers if _compact_eval_identifier(value)}


def _result_product_identifiers(result: dict[str, Any]) -> set[str]:
    metadata = dict(result.get("metadata") or {})
    return _metadata_product_identifiers(metadata)


def _query_product_identifier_hits(query: str, result: dict[str, Any]) -> set[str]:
    compact_query = _compact_eval_identifier(query)
    return {identifier for identifier in _result_product_identifiers(result) if identifier and identifier in compact_query}


def _result_is_applicable_equivalent(case: RetrievalEvalCase, result: dict[str, Any], overlap: int, query_overlap: int) -> bool:
    if str(result.get("source_document_id", "")) == case.source_document_id:
        return False
    result_chunk_type = str(result.get("metadata", {}).get("chunk_type") or result.get("chunk_type", ""))
    if result_chunk_type != case.chunk_type:
        return False
    required_overlap = max(2, min(3, len(case.expected_terms)))
    if overlap < required_overlap or query_overlap < 3:
        return False
    case_identifiers = _case_product_identifiers(case)
    result_identifiers = _result_product_identifiers(result)
    return bool(case_identifiers and result_identifiers and case_identifiers.intersection(result_identifiers))


def _result_matches_cross_document_evidence_item(
    case: RetrievalEvalCase,
    item: dict[str, Any],
    result: dict[str, Any],
    *,
    overlap: int,
    required_overlap: int,
) -> bool:
    if case.generation_method != "cross_document_same_field_evidence":
        return False
    result_chunk_type = str(result.get("metadata", {}).get("chunk_type") or result.get("chunk_type", ""))
    if result_chunk_type != case.chunk_type:
        return False
    expected_field = normalize_text(str(item.get("field") or ""))
    if not expected_field or _result_column_field(result) != expected_field:
        return False
    if overlap < required_overlap:
        return False
    item_identifiers = {
        str(identifier)
        for identifier in item.get("product_identifiers", []) or []
        if str(identifier)
    }
    result_query_identifiers = _query_product_identifier_hits(case.query, result)
    if item_identifiers:
        return bool(item_identifiers.intersection(_result_product_identifiers(result)))
    return bool(result_query_identifiers)


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
            expected_document_id = str(item.get("source_document_id") or case.source_document_id)
            same_document = str(result.get("source_document_id", "")) == expected_document_id
            overlap = _result_term_overlap(result, terms)
            max_overlap = max(max_overlap, overlap)
            required_overlap = max(1, min(2, len(terms)))
            if (
                same_chunk
                or (same_document and overlap >= required_overlap)
                or _result_matches_cross_document_evidence_item(
                    case,
                    item,
                    result,
                    overlap=overlap,
                    required_overlap=required_overlap,
                )
            ):
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
        if _result_is_applicable_equivalent(case, result, overlap, query_overlap):
            return {
                "passed": True,
                "rank": rank,
                "match_reason": "applicable_equivalent_answer_evidence",
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


def _answer_document_ids(answer: dict[str, Any], key: str) -> set[str]:
    document_ids: set[str] = set()
    for item in answer.get(key, []) or []:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id") or item.get("source_document_id") or "")
        if document_id:
            document_ids.add(document_id)
    return document_ids


def _expected_answer_document_ids(case: RetrievalEvalCase) -> set[str]:
    document_ids = {case.source_document_id}
    for item in case.expected_evidence or []:
        document_id = str(item.get("source_document_id") or "")
        if document_id:
            document_ids.add(document_id)
    return document_ids


def _answer_contains_expected_terms(
    answer: dict[str, Any],
    expected_terms: list[str],
    *,
    required_terms: list[str] | None = None,
) -> dict[str, Any]:
    answer_text = str(answer.get("answer") or "")
    answer_text_lower = answer_text.lower()
    answer_tokens = set(tokenize(answer_text))
    expected = [term for term in expected_terms if term]
    matched = [
        term
        for term in expected
        if _expected_term_matches_text(term, answer_text_lower, answer_tokens)
    ]
    material_expected = [term for term in required_terms or [] if term]
    material_matched = [
        term
        for term in material_expected
        if _expected_term_matches_text(term, answer_text_lower, answer_tokens)
    ]
    required = min(2, len(expected))
    material_required = len(material_expected)
    material_passed = len(material_matched) >= material_required if material_required else True
    base_terms_passed = len(matched) >= required if required else False
    return {
        "passed": base_terms_passed and material_passed,
        "matched_terms": matched,
        "expected_terms": expected,
        "required_terms": required,
        "material_expected_terms": material_expected,
        "material_matched_terms": material_matched,
        "material_required_terms": material_required,
    }


def _answer_scoring_terms(case: RetrievalEvalCase) -> tuple[list[str], str]:
    if not case.expected_evidence:
        specific_terms = _scorable_answer_terms(case.expected_terms, case.expected_snippet)
        if specific_terms:
            return specific_terms, "case_expected_terms"
        return [], "no_scorable_case_expected_terms"
    evidence_terms: list[str] = []
    seen: set[str] = set()

    def add_term(term: Any) -> None:
        normalized = str(term).strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        evidence_terms.append(normalized)

    for item in case.expected_evidence:
        for term in item.get("expected_terms") or []:
            add_term(term)
        for token in tokenize(str(item.get("snippet") or "")):
            if len(token) < 3 and not any(char.isdigit() for char in token):
                continue
            if token.lower() not in ANSWER_SCORING_GENERIC_TERMS:
                add_term(token)
    specific_terms = [
        term
        for term in evidence_terms
        if term.lower().strip() not in ANSWER_SCORING_GENERIC_TERMS
    ]
    if len(specific_terms) >= 2:
        return specific_terms, "expected_evidence_specific_terms"
    if evidence_terms:
        return evidence_terms, "expected_evidence_terms"
    return case.expected_terms, "case_expected_terms"


def _scorable_answer_terms(terms: list[str], source_text: str = "") -> list[str]:
    source_address_terms = _source_address_terms(source_text)
    scorable: list[str] = []
    for term in terms:
        normalized = str(term).strip()
        key = normalized.lower()
        if not normalized:
            continue
        if key in ANSWER_SCORING_GENERIC_TERMS:
            continue
        if key in source_address_terms:
            continue
        scorable.append(normalized)
    return scorable


def _source_address_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Z0-9-]*\s*:\s*[A-Za-z][A-Za-z0-9_.]*(?:\[[^\]]+\])?(?:\.\d+)?", text):
        address = match.group(0)
        terms.update(tokenize(address))
        if ":" in address:
            lhs, rhs = address.split(":", 1)
            terms.add(lhs.strip().lower())
            rhs_base = re.sub(r"\[[^\]]+\](?:\.\d+)?", "", rhs.strip().lower())
            if rhs_base:
                terms.add(rhs_base)
    return {term for term in terms if term}


def _is_quantity_answer_query(query: str) -> bool:
    normalized = normalize_text(query)
    return any(term in normalized for term in ANSWER_SCORING_QUANTITY_QUERY_TERMS)


def _answer_material_term_key(term: str) -> str:
    return ANSWER_SCORING_NUMBER_WORDS.get(term.lower(), term.lower())


def _answer_overlap_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in tokenize(text)
        if token not in STOPWORDS and token not in GENERIC_ANCHORS and token not in ANSWER_SCORING_GENERIC_TERMS
    }
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
        elif len(token) > 2:
            expanded.add(f"{token}s")
    return expanded


def _add_answer_material_term(terms: list[str], seen: set[str], term: str) -> None:
    normalized = term.strip()
    if not normalized:
        return
    key = _answer_material_term_key(normalized)
    if key in seen:
        return
    seen.add(key)
    terms.append(normalized)


def _material_answer_terms_from_sentence(sentence: str) -> list[str]:
    material: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\b[A-Za-z]*\d+(?:\.\d+)?[A-Za-z/%.-]*\b|\b[A-Z]{2,}\d+[A-Z0-9-]*\b", sentence):
        _add_answer_material_term(material, seen, token)
    for token in tokenize(sentence):
        if token in ANSWER_SCORING_NUMBER_WORDS:
            _add_answer_material_term(material, seen, token)
    return material


def _answer_action_source_text(answer: dict[str, Any], item: dict[str, Any]) -> str:
    snippets = [str(item.get("snippet") or "")]
    expected_chunk_id = str(item.get("chunk_id") or "")
    expected_terms = {
        normalize_text(str(term))
        for term in item.get("expected_terms") or []
        if str(term).strip()
    }
    for citation in answer.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        quote = str(citation.get("quote_span") or "")
        if not quote:
            continue
        citation_chunk_id = str(citation.get("chunk_id") or "")
        quote_tokens = _answer_overlap_tokens(quote)
        expected_overlap = {
            term
            for term in expected_terms
            if term and _expected_term_matches_text(term, normalize_text(quote), quote_tokens)
        }
        if (
            citation_chunk_id == expected_chunk_id
            or "corrective action" in normalize_text(quote)
            or len(expected_overlap) >= 2
        ):
            snippets.append(quote)
    return " ".join(snippets)


def _without_table_row_header_metadata(text: str) -> str:
    without_row_headers = re.sub(r"Row headers:\s*[^;]*(?:;|$)", "", text)
    return re.sub(r"\b(?:Row|Column):\s*\d+\b", "", without_row_headers)


def _answer_required_action_terms(case: RetrievalEvalCase, answer: dict[str, Any]) -> tuple[list[str], str]:
    if case.generation_method != "table_sibling_error_cause_action" or not case.expected_evidence:
        return [], "none"
    required_terms: list[str] = []
    seen: set[str] = set()
    query_terms = _answer_overlap_tokens(case.query)
    action_fields = {"action", "corrective action", "countermeasure", "remedy"}
    for item in case.expected_evidence:
        field = normalize_text(str(item.get("field") or ""))
        if field not in action_fields:
            continue
        candidates: list[str] = []
        source_text = _answer_action_source_text(answer, item)
        action_text = _without_table_row_header_metadata(source_text)
        source_tokens = _answer_overlap_tokens(action_text)
        source_action_verbs = [
            token
            for token in tokenize(action_text)
            if token in ANSWER_SCORING_ACTION_VERBS
        ]
        for term in item.get("expected_terms") or []:
            normalized = str(term).strip()
            key = normalize_text(normalized)
            if key in ANSWER_SCORING_STATE_TERMS and key not in query_terms:
                candidates.append(normalized)
        for term in source_action_verbs:
            candidates.append(term)
        for term in item.get("expected_terms") or []:
            normalized = str(term).strip()
            if not normalized:
                continue
            key = normalize_text(normalized)
            if key in ANSWER_SCORING_GENERIC_TERMS or key in query_terms:
                continue
            if key in ANSWER_SCORING_SENTENCE_SKIP_TERMS and key not in query_terms:
                continue
            if key not in source_tokens and source_action_verbs:
                continue
            candidates.append(normalized)
        for token in _material_answer_terms_from_sentence(action_text):
            candidates.append(token)
        for term in candidates:
            _add_answer_material_term(required_terms, seen, term)
            if len(required_terms) >= 2:
                return required_terms, "troubleshooting_action_terms"
    return required_terms, "troubleshooting_action_terms" if required_terms else "none"


def _answer_required_material_terms(case: RetrievalEvalCase, answer: dict[str, Any]) -> tuple[list[str], str]:
    if not case.expected_evidence:
        return [], "none"
    action_terms, action_source = _answer_required_action_terms(case, answer)
    query_tokens = _answer_overlap_tokens(case.query)
    if not _is_quantity_answer_query(case.query):
        return action_terms, action_source
    required_terms: list[str] = []
    seen: set[str] = set()
    for term in action_terms:
        _add_answer_material_term(required_terms, seen, term)
    for item in case.expected_evidence:
        snippet = str(item.get("snippet") or "")
        for sentence in re.split(r"[.!?;|]\s*", snippet):
            sentence_tokens = _answer_overlap_tokens(sentence)
            if not sentence_tokens:
                continue
            if sentence_tokens.intersection(ANSWER_SCORING_SENTENCE_SKIP_TERMS) and not (
                sentence_tokens.intersection(ANSWER_SCORING_SENTENCE_SKIP_TERMS).intersection(query_tokens)
            ):
                continue
            if len(sentence_tokens.intersection(query_tokens)) < 2:
                continue
            for term in _material_answer_terms_from_sentence(sentence):
                _add_answer_material_term(required_terms, seen, term)
            if len(required_terms) >= 4:
                break
        if len(required_terms) >= 4:
            break
    if required_terms and action_terms:
        return required_terms, "troubleshooting_action_and_quantity_terms"
    return required_terms, "quantity_evidence_terms" if required_terms else "none"


def _expected_term_matches_text(term: str, text_lower: str, text_tokens: set[str]) -> bool:
    term_lower = term.lower().strip()
    if not term_lower:
        return False
    if term_lower in text_lower or term_lower in text_tokens:
        return True
    number = ANSWER_SCORING_NUMBER_WORDS.get(term_lower)
    if number and (number in text_tokens or re.search(rf"(?<![\w.-]){re.escape(number)}(?![\w.-])", text_lower)):
        return True
    for word, value in ANSWER_SCORING_NUMBER_WORDS.items():
        if term_lower == value and word in text_tokens:
            return True
    if term_lower.endswith("s") and len(term_lower) > 3 and term_lower[:-1] in text_tokens:
        return True
    if f"{term_lower}s" in text_tokens:
        return True
    if term_lower.endswith("ing") and len(term_lower) > 5:
        stem = term_lower[:-3]
        if stem in text_tokens or f"{stem}s" in text_tokens:
            return True
    if term_lower.endswith("ed") and len(term_lower) > 4:
        stem = term_lower[:-2]
        if stem in text_tokens or f"{stem}s" in text_tokens:
            return True
    if "/" in term_lower:
        parts = [part for part in term_lower.split("/") if part]
        return bool(parts) and all(part in text_tokens for part in parts)
    return False


def score_answer_response(
    case: RetrievalEvalCase,
    answer: dict[str, Any],
    retrieval_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    citation_document_ids = _answer_document_ids(answer, "citations")
    used_document_ids = _answer_document_ids(answer, "used_documents")
    answer_document_ids = citation_document_ids.union(used_document_ids)
    expected_document_ids = _expected_answer_document_ids(case)
    missing_document_ids = sorted(expected_document_ids.difference(answer_document_ids))
    expected_terms, term_source = _answer_scoring_terms(case)
    material_terms, material_source = _answer_required_material_terms(case, answer)
    terms = _answer_contains_expected_terms(answer, expected_terms, required_terms=material_terms)
    terms["term_source"] = term_source
    terms["material_term_source"] = material_source
    answer_text = str(answer.get("answer") or "").strip()
    passed = bool(
        answer_text
        and not answer.get("insufficient_evidence")
        and not missing_document_ids
        and terms["passed"]
    )
    failure_reasons: list[str] = []
    if not answer_text:
        failure_reasons.append("empty_answer")
    if answer.get("insufficient_evidence"):
        failure_reasons.append("insufficient_evidence")
    if missing_document_ids:
        failure_reasons.append("expected_document_not_cited_or_used")
    if not terms["passed"]:
        failure_reasons.append("expected_terms_missing")
    if retrieval_evaluation and not retrieval_evaluation.get("passed"):
        failure_reasons.append("retrieval_not_passed")
    return {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "expected_document_ids": sorted(expected_document_ids),
        "citation_document_ids": sorted(citation_document_ids),
        "used_document_ids": sorted(used_document_ids),
        "missing_document_ids": missing_document_ids,
        "expected_document_used": not missing_document_ids,
        "term_check": terms,
    }
