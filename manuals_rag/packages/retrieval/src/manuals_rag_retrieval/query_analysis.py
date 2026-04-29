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
    if any(word in lowered for word in ["how", "steps", "configure", "setup", "install"]):
        types.append("how_to")
        preferred_chunk_types.append("procedure_record")
    if any(word in lowered for word in ["configure", "configuration", "setting", "menu"]):
        types.append("configuration")
        preferred_chunk_types.extend(["procedure_record", "section_window", "table_record"])
    if any(word in lowered for word in ["command", "timing", "flow", "handshake", "flag", "procedure"]):
        types.append("operational_flow")
        preferred_chunk_types.extend(["section_window", "procedure_record"])
    if any(word in lowered for word in ["error", "alarm", "troubleshoot", "fault"]):
        types.append("troubleshooting")
    if any(word in lowered for word in ["compare", "difference", "versus", "vs "]):
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
    if menu_labels:
        preferred_metadata_filters["menu_labels"] = menu_labels
    if not types:
        types.append("general")
    model_match = re.search(r"\b[A-Z]{1,5}(?:-[A-Z0-9]{1,8})+\b", query)
    if model_match and not any(char.isdigit() for char in model_match.group(0)):
        model_match = None
    family_match = re.search(r"\b([A-Z]{1,5}-[A-Z0-9]{1,6}|[A-Z]{2,5})\s+(?:series|family)\b", query, flags=re.IGNORECASE)
    part_match = None
    error_match = re.search(r"\b[A-Z]\d{2,4}\b", query)
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
    explicit_identifier_count = int(bool(model_match)) + int(bool(error_match))
    filter_strictness = "strict" if explicit_identifier_count >= 2 else ("balanced" if explicit_identifier_count == 1 else "loose")
    return QueryAnalysis(
        raw_query=query,
        query_types=sorted(set(types)),
        normalized_terms=normalized_terms,
        product_family=family_match.group(1).upper() if family_match and not model_match else None,
        product_model=model_match.group(0) if model_match else None,
        part_number=part_match.group(0) if part_match else None,
        error_code=error_match.group(0) if error_match and not part_match else None,
        requested_doc_kind=requested_doc_kind,
        safety_intent="safety" in types,
        latest_only=any(term in lowered for term in ["latest", "newest", "current revision", "most recent"]),
        preferred_chunk_types=sorted(set(preferred_chunk_types)),
        filter_strictness=filter_strictness,
        preferred_metadata_filters=preferred_metadata_filters,
    )
