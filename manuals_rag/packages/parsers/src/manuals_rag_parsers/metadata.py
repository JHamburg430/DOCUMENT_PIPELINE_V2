from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from manuals_rag_common.config import settings
from manuals_rag_common.ollama import chat_json
from manuals_rag_schemas.enums import DocumentKind


logger = logging.getLogger(__name__)

NON_ENTITY_TERMS = {
    "caution",
    "danger",
    "important",
    "notice",
    "point",
    "reference",
    "warning",
}

PROTOCOL_TERMS = {
    "ethernet/ip",
    "ethercat",
    "profinet",
    "modbus",
    "tcp/ip",
    "udp",
    "rs-232",
    "rs-232c",
    "rs-485",
    "usb",
    "io-link",
    "canopen",
    "cc-link",
}

SCOPED_METADATA_KINDS = {
    "company",
    "product_family",
    "product_model",
    "device",
    "part_number",
    "protocol",
    "firmware_version",
    "software_name",
    "software_version",
    "document_revision",
}
SCOPED_METADATA_RELATIONS = {
    "primary_manufacturer",
    "primary_product",
    "applies_to",
    "compatible_with",
    "accessory_for",
    "external_reference",
    "mentioned",
    "document_revision",
}
DEFAULT_METADATA_SEGMENT_CHARS = 12000


@dataclass(frozen=True)
class DocumentMetadata:
    manufacturer: str
    companies: list[str]
    product_family: str | None
    product_model: str | None
    product_families: list[str]
    product_models: list[str]
    devices: list[str]
    part_numbers: list[str]
    protocol_terms: list[str]
    settings: list[str]
    parameters: list[str]
    menu_labels: list[str]
    document_topics: list[str]
    title: str
    document_kind: DocumentKind
    revision_date: date | None
    effective_date: date | None
    metadata_schema_version: int = 1
    metadata_evidence: list[dict[str, Any]] = field(default_factory=list)
    normalized_identifier_aliases: list[str] = field(default_factory=list)
    routing_product_models: list[str] = field(default_factory=list)
    routing_part_numbers: list[str] = field(default_factory=list)
    routing_protocol_terms: list[str] = field(default_factory=list)
    firmware_applicability: list[dict[str, Any]] = field(default_factory=list)
    software_applicability: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MetadataSourceSegment:
    text: str
    page_from: int | None = None
    page_to: int | None = None
    section_path: tuple[str, ...] = ()


class ScopedMetadataCandidate(BaseModel):
    value: str
    kind: str
    relation: str = "mentioned"
    subject: str | None = None
    source_quote: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_entity_type_alias(cls, value: Any) -> Any:
        if isinstance(value, dict) and "kind" not in value and "entity_type" in value:
            return {**value, "kind": value["entity_type"]}
        return value


class ScopedMetadataExtraction(BaseModel):
    entities: list[ScopedMetadataCandidate] = Field(default_factory=list)


class MetadataExtraction(BaseModel):
    title: str | None = None
    document_kind: DocumentKind = DocumentKind.manual
    manufacturer: str | None = None
    companies: list[str] = Field(default_factory=list)
    product_family: str | None = None
    product_model: str | None = None
    product_families: list[str] = Field(default_factory=list)
    product_models: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    part_numbers: list[str] = Field(default_factory=list)
    protocol_terms: list[str] = Field(default_factory=list)
    settings: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    menu_labels: list[str] = Field(default_factory=list)
    document_topics: list[str] = Field(default_factory=list)
    revision_date: date | None = None
    effective_date: date | None = None

    @field_validator(
        "companies",
        "product_families",
        "product_models",
        "devices",
        "part_numbers",
        "protocol_terms",
        "settings",
        "parameters",
        "menu_labels",
        "document_topics",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    @field_validator("revision_date", "effective_date", mode="before")
    @classmethod
    def _coerce_optional_date(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return value


class ScalarMetadataExtraction(BaseModel):
    title: str | None = None
    document_kind: DocumentKind = DocumentKind.manual
    manufacturer: str | None = None
    product_family: str | None = None
    product_model: str | None = None
    revision_date: date | None = None
    effective_date: date | None = None

    @field_validator("revision_date", "effective_date", mode="before")
    @classmethod
    def _coerce_optional_date(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return value


def infer_document_kind(filename: str) -> DocumentKind:
    extraction = _extract_scalar_metadata(filename=filename, text="")
    return extraction.document_kind


def _normalize_title(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return stem.replace("_", " ").strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        fingerprint = normalized.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(normalized)
    return deduped


def _source_text(filename: str, text: str) -> str:
    return f"FILENAME:\n{filename}\n\nTEXT:\n{text[:12000]}"


def _scalar_prompt_messages(filename: str, text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a metadata classification function. Return only JSON matching the schema. "
                "Use null for unknown scalar values. Do not invent identifiers. "
                "document_kind must use the enum value from the schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_source_text(filename, text)}\n\n"
                "Classify title, document_kind, manufacturer, primary product_family, primary product_model, "
                "revision_date, and effective_date. Return JSON only."
            ),
        },
    ]


def _scoped_prompt_messages(filename: str, text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract retrieval metadata from one page-aware manual excerpt. Return only JSON. "
                "Every entity must include an exact short source_quote copied from the excerpt. "
                "Do not infer an entity from a filename or from general knowledge. "
                "Classify references to another controller, PLC, accessory, example vendor, or host software "
                "as external_reference unless the excerpt explicitly says it applies to the manual's primary product. "
                "Never attach a firmware or software version to a product unless the quote establishes that scope."
            ),
        },
        {
            "role": "user",
            "content": (
                f"FILENAME (context only; not evidence): {filename}\n\n{text}\n\n"
                "Extract only high-value routing metadata. Allowed kinds: company, product_family, product_model, "
                "device, part_number, protocol, firmware_version, software_name, software_version, document_revision. "
                "Allowed relations: primary_manufacturer, primary_product, applies_to, compatible_with, accessory_for, "
                "external_reference, mentioned, document_revision. For firmware_version and software_version, subject "
                "must name the product or software that the quote binds the version to; omit the entity if scope is unclear. "
                "Use primary_manufacturer or primary_product only when the excerpt explicitly identifies the document owner/product."
            ),
        },
    ]


def _scoped_metadata_schema() -> dict[str, Any]:
    schema = ScopedMetadataExtraction.model_json_schema()
    schema["additionalProperties"] = False
    entity_schema = schema.get("$defs", {}).get("ScopedMetadataCandidate")
    if isinstance(entity_schema, dict):
        entity_schema["additionalProperties"] = False
    return schema


LIST_FIELD_INSTRUCTIONS = {
    "companies": "Copy company or manufacturer names from the source. Examples of the kind of value: ACME CONTROLS, NORTHRIDGE AUTOMATION. Return exact source text only.",
    "product_families": "Extract product family names or series names from the source. Examples of the kind of value: AX series, QN family, Model 700 platform. Use short values.",
    "product_models": "Copy product model identifiers from the source. Examples of the kind of value: AX-1200, QN-42A, MTR-700. Do not include accessory part numbers.",
    "devices": "Copy device, product, or equipment names from the source. Use concise exact source phrases.",
    "part_numbers": "Copy accessory, option, cable, or part/order numbers from the source. Examples of the kind of value: ACC-88310, CBL-2040. Do not include product model identifiers.",
    "protocol_terms": "Copy industrial communication protocol names from the source. Examples of the kind of value: EtherNet/IP, EtherCAT, PROFINET, Modbus, RS-232C.",
    "settings": "Copy UI setting names, menu setting names, setup labels, or configurable setting names from the source. Use concise exact source phrases.",
    "parameters": "Copy parameter names, parameter labels, or named numeric/configuration parameters from the source. Use concise exact source phrases.",
    "menu_labels": "Copy only UI labels that are enclosed in square brackets in the source, preserving the brackets. If none appear, return an empty list.",
    "document_topics": "Classify concise lowercase topics supported by the source, such as setup, wiring, safety, communications, specifications, troubleshooting, maintenance.",
}

GROUNDED_LIST_FIELDS = {
    "companies",
    "product_families",
    "product_models",
    "devices",
    "part_numbers",
    "protocol_terms",
    "settings",
    "parameters",
    "menu_labels",
    "document_topics",
}


def _list_prompt_messages(field_name: str, filename: str, text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a one-field extraction function. Return only JSON matching the schema. "
                "Do not explain. Do not copy the instructions. Use [] when the source has no evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Field: {field_name}\n"
                f"Instruction: {LIST_FIELD_INSTRUCTIONS[field_name]}\n\n"
                f"{_source_text(filename, text)}\n\n"
                f"Return JSON only with the key {field_name}."
            ),
        },
    ]


def _list_field_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {field_name: {"type": "array", "items": {"type": "string"}}},
        "required": [field_name],
    }


def _scalar_metadata_schema() -> dict[str, Any]:
    schema = ScalarMetadataExtraction.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def _extract_scalar_metadata(filename: str, text: str) -> ScalarMetadataExtraction:
    try:
        parsed, _raw = chat_json(
            model=settings.ollama_metadata_model,
            messages=_scalar_prompt_messages(filename, text),
            json_schema=_scalar_metadata_schema(),
            think=False,
            purpose="metadata_extraction",
            num_predict=240,
        )
        return ScalarMetadataExtraction.model_validate(parsed)
    except Exception as exc:
        logger.warning("Scalar metadata extraction failed for %s; using empty scalar metadata: %s", filename, exc)
        return ScalarMetadataExtraction(title=_normalize_title(filename))


def _value_is_grounded(value: str, source: str) -> bool:
    normalized_value = " ".join(value.casefold().split())
    normalized_source = " ".join(source.casefold().split())
    return bool(normalized_value) and normalized_value in normalized_source


def _ground_values(field_name: str, values: list[str], filename: str, text: str) -> list[str]:
    if field_name not in GROUNDED_LIST_FIELDS:
        return values
    source = _source_text(filename, text)
    grounded = []
    filename_stems = {
        filename.rsplit(".", 1)[0].casefold(),
        _normalize_title(filename).casefold(),
    }
    for value in values:
        stripped = value.strip()
        lowered = stripped.casefold()
        if lowered in NON_ENTITY_TERMS:
            continue
        if "|" in stripped:
            continue
        if field_name in {"companies", "devices"} and len(stripped.split()) > 6:
            continue
        if field_name == "protocol_terms" and lowered not in PROTOCOL_TERMS:
            continue
        if field_name == "product_models" and not (any(char.isdigit() for char in stripped) and re.search(r"[A-Z]{1,5}-?[A-Z0-9]", stripped)):
            continue
        if field_name == "product_families" and stripped.upper() in {"PLC"}:
            continue
        if field_name == "menu_labels" and not (stripped.startswith("[") and stripped.endswith("]")):
            continue
        if field_name == "devices" and stripped.casefold() in {filename.casefold(), *filename_stems}:
            continue
        if field_name in {"part_numbers", "product_models"} and ("_" in stripped or stripped.lower().endswith(".pdf")):
            continue
        if field_name == "parameters" and "parameter" not in stripped.casefold():
            continue
        if field_name == "settings" and not any(term in stripped.casefold() for term in ("setting", "setup", "menu", "select", "configure")):
            continue
        if _value_is_grounded(stripped, source):
            grounded.append(stripped)
    return grounded


def _ground_date(value: date | None, filename: str, text: str) -> date | None:
    if value is None:
        return None
    source = _source_text(filename, text)
    candidates = {
        value.isoformat(),
        value.strftime("%Y/%m/%d"),
        value.strftime("%m/%d/%Y"),
    }
    if any(candidate in source for candidate in candidates):
        return value
    return None


def _extract_list_field(field_name: str, filename: str, text: str) -> list[str]:
    try:
        parsed, _raw = chat_json(
            model=settings.ollama_metadata_model,
            messages=_list_prompt_messages(field_name, filename, text),
            json_schema=_list_field_schema(field_name),
            think=False,
            purpose=f"metadata_extraction.{field_name}",
            num_predict=160,
        )
    except Exception as exc:
        logger.warning("List metadata extraction failed for %s field=%s; using empty list: %s", filename, field_name, exc)
        return []
    values = MetadataExtraction._coerce_list(parsed.get(field_name))
    return _dedupe_preserve_order(_ground_values(field_name, values, filename, text))


def _extract_metadata_with_model(filename: str, text: str) -> MetadataExtraction:
    scalar = _extract_scalar_metadata(filename, text)
    lists = {field_name: _extract_list_field(field_name, filename, text) for field_name in LIST_FIELD_INSTRUCTIONS}
    return MetadataExtraction(
        **scalar.model_dump(),
        **lists,
    )


def _to_document_metadata(filename: str, text: str, extraction: MetadataExtraction) -> DocumentMetadata:
    companies = _dedupe_preserve_order(extraction.companies)
    manufacturer = companies[0] if companies else None
    if manufacturer is None and extraction.manufacturer:
        candidate = extraction.manufacturer.strip()
        if (
            candidate.casefold() not in NON_ENTITY_TERMS
            and "|" not in candidate
            and len(candidate.split()) <= 6
            and _value_is_grounded(candidate, _source_text(filename, text))
        ):
            manufacturer = candidate
    manufacturer = manufacturer or "Unknown"
    product_models = _dedupe_preserve_order(extraction.product_models)
    product_model = product_models[0] if product_models else None
    if (
        product_model is None
        and extraction.product_model
        and extraction.product_model.casefold() not in NON_ENTITY_TERMS
        and any(char.isdigit() for char in extraction.product_model)
        and _value_is_grounded(extraction.product_model, _source_text(filename, text))
    ):
        product_model = extraction.product_model
    product_families = _dedupe_preserve_order(extraction.product_families)
    product_family = product_families[0] if product_families else None
    if product_family is None and extraction.product_family and extraction.product_family not in product_models:
        product_family = extraction.product_family if _value_is_grounded(extraction.product_family, _source_text(filename, text)) else None
    title = (extraction.title or _normalize_title(filename)).strip()
    revision_date = _ground_date(extraction.revision_date, filename, text)
    effective_date = _ground_date(extraction.effective_date, filename, text)
    return DocumentMetadata(
        manufacturer=manufacturer,
        companies=companies,
        product_family=product_family,
        product_model=product_model,
        product_families=product_families,
        product_models=product_models,
        devices=_dedupe_preserve_order(extraction.devices),
        part_numbers=_dedupe_preserve_order(extraction.part_numbers),
        protocol_terms=sorted({term.strip().lower() for term in extraction.protocol_terms if term.strip()}),
        settings=_dedupe_preserve_order(extraction.settings),
        parameters=_dedupe_preserve_order(extraction.parameters),
        menu_labels=_dedupe_preserve_order(extraction.menu_labels),
        document_topics=sorted({topic.strip().lower() for topic in extraction.document_topics if topic.strip()}),
        title=title,
        document_kind=extraction.document_kind,
        revision_date=revision_date,
        effective_date=effective_date or revision_date,
    )


def infer_document_metadata(filename: str, text: str) -> DocumentMetadata:
    try:
        extraction = _extract_metadata_with_model(filename, text)
    except (ValidationError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Metadata extraction failed for {filename}: {exc}") from exc
    return _to_document_metadata(filename, text, extraction)


def _compact_identifier(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _segment_text(segment: MetadataSourceSegment) -> str:
    page_label = "unknown" if segment.page_from is None else str(segment.page_from)
    if segment.page_to is not None and segment.page_to != segment.page_from:
        page_label = f"{page_label}-{segment.page_to}"
    section = " > ".join(segment.section_path)
    header = f"[PAGE {page_label}]"
    if section:
        header += f" [SECTION {section}]"
    return f"{header}\n{segment.text.strip()}"


def pack_metadata_source_segments(
    segments: list[MetadataSourceSegment],
    *,
    max_chars: int = DEFAULT_METADATA_SEGMENT_CHARS,
) -> list[list[MetadataSourceSegment]]:
    """Pack page-aware sources without discarding late-document evidence."""
    batches: list[list[MetadataSourceSegment]] = []
    current: list[MetadataSourceSegment] = []
    current_chars = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        rendered_chars = len(_segment_text(segment)) + 2
        if current and current_chars + rendered_chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        if rendered_chars <= max_chars:
            current.append(segment)
            current_chars += rendered_chars
            continue
        for offset in range(0, len(text), max_chars):
            piece = replace(segment, text=text[offset : offset + max_chars])
            if current:
                batches.append(current)
                current = []
                current_chars = 0
            batches.append([piece])
    if current:
        batches.append(current)
    return batches


def _quote_location(quote: str, segments: list[MetadataSourceSegment]) -> MetadataSourceSegment | None:
    normalized_quote = " ".join(quote.casefold().split())
    if not normalized_quote:
        return None
    for segment in segments:
        if normalized_quote in " ".join(segment.text.casefold().split()):
            return segment
    return None


def _extract_scoped_metadata(
    filename: str,
    segments: list[MetadataSourceSegment],
) -> list[dict[str, Any]]:
    rendered = "\n\n".join(_segment_text(segment) for segment in segments)
    try:
        parsed, _raw = chat_json(
            model=settings.ollama_metadata_model,
            messages=_scoped_prompt_messages(filename, rendered),
            json_schema=_scoped_metadata_schema(),
            think=False,
            purpose="metadata_extraction.scoped_entities",
            num_predict=1200,
        )
        extraction = ScopedMetadataExtraction.model_validate(parsed)
    except Exception as exc:
        logger.warning("Scoped metadata extraction failed for %s; skipping batch: %s", filename, exc)
        return []

    external_subjects = {
        _compact_identifier(candidate.value)
        for candidate in extraction.entities
        if candidate.relation.strip().lower() == "external_reference" and candidate.value.strip()
    }
    grounded: list[dict[str, Any]] = []
    for candidate in extraction.entities:
        value = candidate.value.strip()
        quote = candidate.source_quote.strip()
        kind = candidate.kind.strip().lower()
        relation = candidate.relation.strip().lower()
        located = _quote_location(quote, segments)
        if (
            not value
            or kind not in SCOPED_METADATA_KINDS
            or relation not in SCOPED_METADATA_RELATIONS
            or located is None
            or not _value_is_grounded(value, quote)
        ):
            continue
        subject = candidate.subject.strip() if candidate.subject else None
        if kind in {"firmware_version", "software_version"} and not subject:
            continue
        if kind in {"firmware_version", "software_version"} and _compact_identifier(subject or "") in external_subjects:
            relation = "external_reference"
        grounded.append(
            {
                "value": value,
                "kind": kind,
                "relation": relation,
                "subject": subject,
                "source_quote": quote[:500],
                "page_from": located.page_from,
                "page_to": located.page_to,
                "section_path": list(located.section_path),
                "confidence": candidate.confidence,
                "grounded": True,
                "source": "page_aware_model_extraction",
            }
        )
    return grounded


def _dedupe_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int | None]] = set()
    for item in evidence:
        fingerprint = (
            str(item.get("kind") or ""),
            _compact_identifier(str(item.get("value") or "")),
            str(item.get("relation") or ""),
            _compact_identifier(str(item.get("subject") or "")),
            item.get("page_from"),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(item)
    return deduped


def _first_primary(evidence: list[dict[str, Any]], kind: str, relation: str) -> str | None:
    matches = [
        item
        for item in evidence
        if item.get("kind") == kind and item.get("relation") == relation and item.get("grounded") is True
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (-float(item.get("confidence") or 0.0), int(item.get("page_from") or 10**9)))
    return str(matches[0]["value"])


def _values_for_routing(evidence: list[dict[str, Any]], kind: str) -> list[str]:
    allowed_relations = {"primary_product", "applies_to", "compatible_with", "accessory_for", "mentioned"}
    if kind == "company":
        allowed_relations.add("primary_manufacturer")
    return _dedupe_preserve_order(
        [
            str(item["value"])
            for item in evidence
            if item.get("kind") == kind
            and item.get("relation") in allowed_relations
            and item.get("grounded") is True
        ]
    )


def _applicability_records(evidence: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [
        {
            "version": item["value"],
            "subject": item.get("subject"),
            "relation": item.get("relation"),
            "page_from": item.get("page_from"),
            "page_to": item.get("page_to"),
            "section_path": item.get("section_path") or [],
            "source_quote": item.get("source_quote"),
            "confidence": item.get("confidence"),
            "grounded": True,
        }
        for item in evidence
        if item.get("kind") == kind
        and item.get("relation") != "external_reference"
        and item.get("subject")
        and item.get("grounded") is True
    ]


def _base_metadata_evidence(base: DocumentMetadata, segments: list[MetadataSourceSegment]) -> list[dict[str, Any]]:
    fields = {
        "company": [base.manufacturer, *base.companies],
        "product_family": [base.product_family, *base.product_families],
        "product_model": [base.product_model, *base.product_models],
        "device": base.devices,
        "part_number": base.part_numbers,
        "protocol": base.protocol_terms,
    }
    evidence: list[dict[str, Any]] = []
    for kind, values in fields.items():
        for value in _dedupe_preserve_order([str(item) for item in values if item]):
            located = _quote_location(value, segments)
            if located is None:
                continue
            evidence.append(
                {
                    "value": value,
                    "kind": kind,
                    "relation": "mentioned",
                    "subject": None,
                    "source_quote": value,
                    "page_from": located.page_from,
                    "page_to": located.page_to,
                    "section_path": list(located.section_path),
                    "confidence": 0.5,
                    "grounded": True,
                    "source": "legacy_flat_extraction",
                }
            )
    return evidence


def infer_document_metadata_from_segments(
    filename: str,
    segments: list[MetadataSourceSegment],
    *,
    max_segment_chars: int = DEFAULT_METADATA_SEGMENT_CHARS,
) -> DocumentMetadata:
    """Extract flat compatibility fields plus grounded, scoped metadata from the whole document."""
    nonempty = [segment for segment in segments if segment.text.strip()]
    if not nonempty:
        return infer_document_metadata(filename, "")
    front_text = "\n\n".join(_segment_text(segment) for segment in nonempty)[:DEFAULT_METADATA_SEGMENT_CHARS]
    base = infer_document_metadata(filename, front_text)
    evidence: list[dict[str, Any]] = []
    for batch in pack_metadata_source_segments(nonempty, max_chars=max_segment_chars):
        evidence.extend(_extract_scoped_metadata(filename, batch))
    scoped_evidence = _dedupe_evidence(evidence)
    evidence = _dedupe_evidence(scoped_evidence + _base_metadata_evidence(base, nonempty))

    verified_product_models = _values_for_routing(scoped_evidence, "product_model")
    verified_part_numbers = _values_for_routing(scoped_evidence, "part_number")
    verified_protocols = [value.lower() for value in _values_for_routing(scoped_evidence, "protocol")]
    product_models = _dedupe_preserve_order(base.product_models + verified_product_models)
    product_families = _dedupe_preserve_order(base.product_families + _values_for_routing(scoped_evidence, "product_family"))
    part_numbers = _dedupe_preserve_order(base.part_numbers + verified_part_numbers)
    devices = _dedupe_preserve_order(base.devices + _values_for_routing(scoped_evidence, "device"))
    protocols = _dedupe_preserve_order(base.protocol_terms + verified_protocols)
    routing_product_models = verified_product_models or ([base.product_model] if base.product_model else [])
    routing_part_numbers = verified_part_numbers or base.part_numbers
    routing_protocols = verified_protocols or base.protocol_terms
    identifiers = routing_product_models + routing_part_numbers + routing_protocols
    normalized_aliases = _dedupe_preserve_order(
        [alias for value in identifiers for alias in (value, _compact_identifier(value)) if alias]
    )

    primary_manufacturer = _first_primary(evidence, "company", "primary_manufacturer")
    primary_product = _first_primary(evidence, "product_model", "primary_product")
    return replace(
        base,
        manufacturer=primary_manufacturer or base.manufacturer,
        companies=_dedupe_preserve_order(base.companies + _values_for_routing(evidence, "company")),
        product_model=primary_product or base.product_model,
        product_models=product_models,
        product_family=base.product_family or (product_families[0] if product_families else None),
        product_families=product_families,
        devices=devices,
        part_numbers=part_numbers,
        protocol_terms=protocols,
        metadata_schema_version=2,
        metadata_evidence=evidence,
        normalized_identifier_aliases=normalized_aliases,
        routing_product_models=routing_product_models,
        routing_part_numbers=routing_part_numbers,
        routing_protocol_terms=routing_protocols,
        firmware_applicability=_applicability_records(scoped_evidence, "firmware_version"),
        software_applicability=_applicability_records(scoped_evidence, "software_version"),
    )
