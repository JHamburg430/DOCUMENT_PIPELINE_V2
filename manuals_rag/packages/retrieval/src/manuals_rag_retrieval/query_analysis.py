from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class QueryAnalysis:
    raw_query: str
    query_types: list[str] = field(default_factory=list)
    normalized_terms: list[str] = field(default_factory=list)
    product_family: str | None = None
    product_model: str | None = None
    product_identifiers: list[str] = field(default_factory=list)
    part_number: str | None = None
    error_code: str | None = None
    requested_doc_kind: str | None = None
    safety_intent: bool = False
    latest_only: bool = False
    preferred_chunk_types: list[str] = field(default_factory=list)
    filter_strictness: str = "balanced"
    preferred_metadata_filters: dict[str, list[str] | str] = field(default_factory=dict)


def analyze_query(query: str) -> QueryAnalysis:
    lowered = query.lower()
    types: list[str] = []
    preferred_chunk_types: list[str] = []
    preferred_metadata_filters: dict[str, list[str] | str] = {}
    normalized_terms = sorted(
        {
            term
            for term in re.findall(r"[a-z0-9][a-z0-9\-/.]*", lowered)
            if len(term) >= 2 or any(char.isdigit() for char in term) or term in {"x", "y", "z"}
        }
    )
    menu_labels = [label.strip() for label in re.findall(r"\[[^\]]+\]", query)]
    if any(word in lowered for word in ["warning", "danger", "safety", "hazard", "caution"]):
        types.append("safety")
        preferred_chunk_types.append("warning_record")
    location_configuration = bool(
        re.search(
            r"\bwhere\b.{0,80}\b(?:set|adjust|change|configure|find|locate|select|enable|disable)\b"
            r"|\bwhere\s+(?:is|are)\b.{0,80}\b(?:setting|option|parameter|control|field)\b"
            r"|\b(?:which|what)\s+(?:menu|screen|tab|section|page)\b",
            lowered,
        )
    )
    if any(word in lowered for word in ["how", "steps", "configure", "setup", "install"]) or location_configuration:
        types.append("how_to")
        preferred_chunk_types.append("procedure_record")
    if any(word in lowered for word in ["configure", "configuration", "setting", "parameter", "menu"]) or location_configuration:
        types.append("configuration")
        preferred_chunk_types.extend(["procedure_record", "section_window", "table_record"])
    if any(word in lowered for word in ["command", "timing", "flow", "handshake", "flag", "procedure"]):
        types.append("operational_flow")
        preferred_chunk_types.extend(["section_window", "procedure_record"])
    structured_lookup_field = re.search(
        r"\b(?:address|values?|items?|setting\s+(?:item|range)|word\s+device|number\s+of\s+image\s+pixels|causes?|error\s+code|message|symbol|description|detection|index|sub\s+index|stored\s+data|error\s+message|summary|data\s*\d+)\b",
        lowered,
    )
    structured_lookup_shape = re.search(r"\b(?:appl(?:y|ies)\s+to|appl(?:y|ies)\s+for)\b", lowered)
    structured_reverse_lookup_shape = re.search(
        r"\bwhat\s+"
        r"(?:address|values?|items?|setting\s+(?:item|range)|word\s+device|number\s+of\s+image\s+pixels|"
        r"cause|error\s+code|message|symbol|description|detection|index|sub\s+index|stored\s+data|"
        r"error\s+message|summary|data\s*\d+)"
        r"\b.+\b(?:selects?|sets?|specif(?:y|ies)|measures?|displays?|indicates?|corresponds?)\b",
        lowered,
    )
    structured_value_lookup_shape = re.search(
        r"\bwhat\s+(?:value|setting|number\s+format|initial\s+value|upper\s+limit(?:\s+value)?|"
        r"lower\s+limit(?:\s+value)?|decimal\s+digits|integer\s+digits|referenceable)\b"
        r".+\b(?:listed|specified|shown|given|configured|set)\s+for\b",
        lowered,
    )
    structured_table_field = re.search(
        r"\b(?:number\s+format|decimal\s+digits|integer\s+digits|referenceable|initial\s+value|"
        r"upper\s+limit(?:\s+value)?|lower\s+limit(?:\s+value)?|setting\s+(?:item|range)|"
        r"row|column|cell\s+value)\b",
        lowered,
    )
    if structured_lookup_field and (structured_lookup_shape or structured_reverse_lookup_shape):
        types.append("structured_lookup")
        preferred_chunk_types.extend(["table_record", "spec_record", "section_window"])
    if structured_value_lookup_shape and structured_table_field:
        types.append("structured_lookup")
        preferred_chunk_types.extend(["table_record", "spec_record", "section_window"])
    requested_doc_kind = None
    if "datasheet" in lowered:
        requested_doc_kind = "datasheet"
    elif "brochure" in lowered:
        requested_doc_kind = "brochure"
    elif "safety" in lowered:
        requested_doc_kind = "safety_bulletin"
    elif "troubleshoot" in lowered or "error" in lowered or "fault" in lowered:
        requested_doc_kind = "troubleshooting_guide"
    elif "install" in lowered:
        requested_doc_kind = "installation_guide"
    elif "setup" in lowered:
        requested_doc_kind = "setup_guide"
    if "manual" in lowered and requested_doc_kind is None:
        requested_doc_kind = "manual"
    spec_terms = {
        "voltage",
        "current",
        "dimension",
        "laser",
        "radiation",
        "wavelength",
        "output",
        "class",
        "enclosure",
        "rating",
        "resistance",
        "shock",
        "vibration",
        "humidity",
        "temperature",
    }
    explicit_spec_lookup = "specification" in lowered or re.search(r"\bspecs?\b", lowered) is not None
    if requested_doc_kind in {None, "manual"} and (
        explicit_spec_lookup
        or any(word in lowered for word in spec_terms)
        or (re.search(r"\bspecified\s+for\b", lowered) and re.search(r"\b[A-Z]{1,5}\d{0,4}(?:-[A-Z0-9]{1,8})+\b", query))
    ):
        types.append("spec_lookup")
        preferred_chunk_types.extend(["datasheet_record", "spec_record", "table_record"])
    if any(word in lowered for word in ["error", "alarm", "troubleshoot", "fault"]):
        types.append("troubleshooting")
    if (
        re.search(
            r"\bwhat\s+(?:causes?|caused|(?:is|are)\s+the\s+(?:likely\s+)?(?:cause|reason))\b",
            lowered,
        )
        or re.search(r"\bhow\s+should\b.+\b(?:corrected|fixed|resolved)\b", lowered)
        or re.search(r"\bwhy\s+(?:does|is)\b.+\b(?:show|shows|report|reporting)\b", lowered)
        or re.search(r"\bwhat should i do when\b.+\b(?:show|shows|report|reports)\b", lowered)
        or re.search(r"\bhow can i resolve\b", lowered)
    ):
        types.append("troubleshooting")
        types.append("structured_lookup")
        preferred_chunk_types.extend(["table_record", "section_window"])
    if re.search(r"\b(?:compare|difference|versus)\b", lowered) or re.search(r"\bvs\.?\b", lowered) and not re.search(
        r"\bvs\.?\s+series\b", lowered
    ):
        types.append("comparison")
        preferred_chunk_types.extend(["spec_record", "datasheet_record", "table_record"])
    if any(word in lowered for word in ["compatib", "support", "supported", "works with"]):
        types.append("compatibility")
        preferred_chunk_types.extend(["spec_record", "table_record", "section_window"])
    if any(word in lowered for word in ["part number", "part no", "sku"]):
        types.append("part_lookup")
        preferred_chunk_types.extend(["spec_record", "datasheet_record", "table_record"])
    if any(word in lowered for word in ["claim", "feature", "advantage", "benefit"]):
        types.append("brochure_claim")
        preferred_chunk_types.extend(["brochure_fact", "datasheet_record", "spec_record"])
    if any(word in lowered for word in ["revision", "version", "legacy", "old version", "superseded"]):
        types.append("revision_history")
        preferred_metadata_filters["version_signal"] = "true"
    if any(word in lowered for word in ["step", "list", "row", "entry"]):
        preferred_chunk_types.append("table_record")
    if "table" in lowered:
        preferred_chunk_types.append("table_record")
    if menu_labels:
        preferred_metadata_filters["menu_labels"] = menu_labels
    if not types:
        types.append("general")
    model_match_spans: list[tuple[int, int]] = []
    model_matches: list[tuple[int, str]] = []
    for match in re.finditer(r"\b[A-Z]{1,5}\d{0,4}(?:-[A-Z0-9]{1,8})+\b", query):
        if not any(char.isdigit() for char in match.group(0)):
            continue
        model_matches.append((match.start(), match.group(0)))
        model_match_spans.append(match.span())
    if (
        len(model_matches) >= 2
        and re.search(r"\b(?:and|with)\b", lowered)
        and re.search(
            r"\b(?:listed|value|rating|resistance|format|cause|remedy|corrective|setting|specifications?|specs?)\b",
            lowered,
        )
    ):
        types.append("comparison")
        preferred_chunk_types.extend(["spec_record", "datasheet_record", "table_record"])
    comparison_identifier_matches: list[tuple[int, str]] = []
    if "comparison" in types or "compatibility" in types:
        comparison_identifier_matches = [
            (match.start(), match.group(0))
            for match in re.finditer(
                r"\b(?:[A-Z]{2,5}\d{1,5}|[A-Z]{1,5}-[A-Z]{1,8})\b",
                query,
            )
            if not re.fullmatch(r"[A-Z]\d{1,5}", match.group(0))
            and not any(
                start <= match.start() and match.end() <= end and (match.start(), match.end()) != (start, end)
                for start, end in model_match_spans
            )
        ]
    model_match = re.search(r"\b[A-Z]{1,5}\d{0,4}(?:-[A-Z0-9]{1,8})+\b", query)
    if model_match and not any(char.isdigit() for char in model_match.group(0)):
        model_match = None
    family_matches = [
        (match.start(1), match.group(1).upper())
        for match in re.finditer(
            r"\b([A-Z]{1,5}-[A-Z0-9]{1,8}|[A-Z]{1,5}\d{2,8}|[A-Z]{2,5})\s+(?:series|family)\b",
            query,
            flags=re.IGNORECASE,
        )
    ]
    family_match = re.search(
        r"\b([A-Z]{1,5}-[A-Z0-9]{1,8}|[A-Z]{1,5}\d{2,8}|[A-Z]{2,5})\s+(?:series|family)\b",
        query,
        flags=re.IGNORECASE,
    )
    part_match = re.search(r"\b(?:OP|CA|SZ|GL|SR|IV|LJ|LR|KV|XG|VS|WM|VJ)-[A-Z0-9]{2,12}[A-Z0-9-]*\b", query)
    error_match = re.search(r"\b[A-Z]\d{2,4}\b(?!\s+(?:series|family))", query, flags=re.IGNORECASE)
    if error_match and model_match:
        model_span = model_match.span()
        if model_span[0] <= error_match.start() and error_match.end() <= model_span[1]:
            error_match = None
    explicit_identifier_count = int(bool(model_match)) + int(bool(error_match))
    filter_strictness = "strict" if explicit_identifier_count >= 2 else ("balanced" if explicit_identifier_count == 1 else "loose")
    product_identifiers: list[str] = []
    for _start, identifier in sorted([*model_matches, *family_matches, *comparison_identifier_matches], key=lambda item: item[0]):
        if identifier and identifier not in product_identifiers:
            product_identifiers.append(identifier)
    return QueryAnalysis(
        raw_query=query,
        query_types=sorted(set(types)),
        normalized_terms=normalized_terms,
        product_family=family_match.group(1).upper() if family_match and not model_match else None,
        product_model=model_match.group(0) if model_match else None,
        product_identifiers=product_identifiers,
        part_number=part_match.group(0) if part_match else None,
        error_code=error_match.group(0) if error_match and not part_match else None,
        requested_doc_kind=requested_doc_kind,
        safety_intent="safety" in types,
        latest_only=any(term in lowered for term in ["latest", "newest", "current revision", "most recent"]),
        preferred_chunk_types=sorted(set(preferred_chunk_types)),
        filter_strictness=filter_strictness,
        preferred_metadata_filters=preferred_metadata_filters,
    )
