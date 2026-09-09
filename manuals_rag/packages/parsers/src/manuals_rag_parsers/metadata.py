from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
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
METADATA_EXTRACTION_ATTEMPTS = 3
PRIMARY_ENTITY_MIN_CONFIDENCE = 0.8

DOCUMENT_KIND_ALIASES = {
    "user_manual": "manual",
    "user_guide": "manual",
    "instruction_manual": "manual",
    "release_notes": "release_note",
    "release_notes_document": "release_note",
    "data_sheet": "datasheet",
    "specification_sheet": "spec_sheet",
}

VERSION_SIGNAL_PATTERNS = {
    "firmware_version": re.compile(
        r"\b(?:firmware|fw)\b.{0,80}?\b(?:v(?:er(?:sion)?)?\.?\s*)?\d+(?:\.\d+){0,3}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "software_version": re.compile(
        r"\b(?:software|application|tool|studio|explorer|twincat|sysmac)\b.{0,80}?"
        r"\b(?:v(?:er(?:sion)?)?\.?\s*)\d+(?:\.\d+){0,3}\b",
        re.IGNORECASE | re.DOTALL,
    ),
}


class MetadataExtractionIncomplete(RuntimeError):
    """Raised when critical metadata evidence cannot be extracted safely."""


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
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_entity_type_alias(cls, value: Any) -> Any:
        if isinstance(value, dict):
            normalized = dict(value)
            if "value" not in normalized:
                for alias in ("name", "entity"):
                    if normalized.get(alias) not in (None, ""):
                        normalized["value"] = normalized[alias]
                        break
            if "kind" not in normalized and "entity_type" in normalized:
                normalized["kind"] = normalized["entity_type"]
            if "source_quote" not in normalized:
                for alias in ("quote", "evidence"):
                    if normalized.get(alias) not in (None, ""):
                        normalized["source_quote"] = normalized[alias]
                        break
            return normalized
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
        if value in (None, "") or str(value).strip().casefold() in {"null", "none", "unknown", "n/a"}:
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

    @field_validator("document_kind", mode="before")
    @classmethod
    def _normalize_document_kind(cls, value: Any) -> Any:
        if value in (None, ""):
            return DocumentKind.manual
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
        return DOCUMENT_KIND_ALIASES.get(normalized, normalized)

    @field_validator("revision_date", "effective_date", mode="before")
    @classmethod
    def _coerce_optional_date(cls, value: Any) -> Any:
        if value in (None, "") or str(value).strip().casefold() in {"null", "none", "unknown", "n/a"}:
            return None
        if isinstance(value, date):
            return value
        normalized = str(value).strip()
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(normalized, pattern).date()
            except ValueError:
                continue
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
                "Never attach a firmware or software version to a product unless the quote establishes that scope. "
                "Every entity must include a calibrated confidence from 0 to 1; do not use a fixed default."
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


def _version_prompt_messages(filename: str, text: str, expected_kinds: set[str]) -> list[dict[str, str]]:
    labels = ", ".join(sorted(expected_kinds))
    return [
        {
            "role": "system",
            "content": (
                "You extract version applicability from a page-aware manual excerpt. Return only JSON. "
                "Extract every explicit firmware/software version statement, including minimums, maximums, "
                "unsupported ranges, requirements, and external PLC/controller dependencies. Each item needs an "
                "exact source_quote, a subject, a calibrated confidence, and the correct relationship. "
                "External product requirements must use external_reference. Do not invent a subject."
            ),
        },
        {
            "role": "user",
            "content": (
                f"FILENAME (context only; not evidence): {filename}\n\n{text}\n\n"
                f"The source has lexical signals for: {labels}. Return all grounded firmware_version and "
                "software_version entities. Use relation applies_to, compatible_with, external_reference, or mentioned."
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


def _normalize_object_response(parsed: Any, *, collection_key: str | None = None) -> dict[str, Any]:
    """Tolerate common model JSON shape drift without weakening field validation."""
    if isinstance(parsed, dict):
        if collection_key and collection_key not in parsed:
            for alias in ("items", "results", "metadata", "entities"):
                if isinstance(parsed.get(alias), list):
                    return {collection_key: parsed[alias]}
        return parsed
    if isinstance(parsed, list):
        if collection_key:
            return {collection_key: parsed}
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
    raise ValueError(f"Expected a JSON object, received {type(parsed).__name__}")


def _extract_scalar_metadata(filename: str, text: str) -> ScalarMetadataExtraction:
    last_error: Exception | None = None
    for attempt in range(1, METADATA_EXTRACTION_ATTEMPTS + 1):
        try:
            parsed, _raw = chat_json(
                model=settings.ollama_metadata_model,
                messages=_scalar_prompt_messages(filename, text),
                json_schema=_scalar_metadata_schema(),
                think=False,
                purpose="metadata_extraction",
                num_predict=320,
            )
            return ScalarMetadataExtraction.model_validate(_normalize_object_response(parsed))
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Scalar metadata extraction attempt %s/%s failed for %s: %s",
                attempt,
                METADATA_EXTRACTION_ATTEMPTS,
                filename,
                exc,
            )
    logger.warning("Scalar metadata extraction exhausted retries for %s; using filename title: %s", filename, last_error)
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
    last_error: Exception | None = None
    for attempt in range(1, METADATA_EXTRACTION_ATTEMPTS + 1):
        try:
            parsed, _raw = chat_json(
                model=settings.ollama_metadata_model,
                messages=_list_prompt_messages(field_name, filename, text),
                json_schema=_list_field_schema(field_name),
                think=False,
                purpose=f"metadata_extraction.{field_name}",
                num_predict=240,
            )
            if isinstance(parsed, list):
                parsed = {field_name: parsed}
            normalized = _normalize_object_response(parsed)
            values = MetadataExtraction._coerce_list(normalized.get(field_name))
            return _dedupe_preserve_order(_ground_values(field_name, values, filename, text))
        except Exception as exc:
            last_error = exc
            logger.warning(
                "List metadata extraction attempt %s/%s failed for %s field=%s: %s",
                attempt,
                METADATA_EXTRACTION_ATTEMPTS,
                filename,
                field_name,
                exc,
            )
    logger.warning("List metadata extraction exhausted retries for %s field=%s: %s", filename, field_name, last_error)
    return []


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


def _expected_version_kinds(segments: list[MetadataSourceSegment]) -> set[str]:
    source = "\n".join(segment.text for segment in segments)
    return {kind for kind, pattern in VERSION_SIGNAL_PATTERNS.items() if pattern.search(source)}


def _call_scoped_model(
    filename: str,
    messages: list[dict[str, str]],
    *,
    purpose: str,
) -> ScopedMetadataExtraction:
    last_error: Exception | None = None
    for attempt in range(1, METADATA_EXTRACTION_ATTEMPTS + 1):
        try:
            parsed, _raw = chat_json(
                model=settings.ollama_metadata_model,
                messages=messages,
                json_schema=_scoped_metadata_schema(),
                think=False,
                purpose=purpose,
                num_predict=1800,
            )
            normalized = _normalize_object_response(parsed, collection_key="entities")
            return ScopedMetadataExtraction.model_validate(normalized)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Scoped metadata extraction attempt %s/%s failed for %s purpose=%s: %s",
                attempt,
                METADATA_EXTRACTION_ATTEMPTS,
                filename,
                purpose,
                exc,
            )
    raise MetadataExtractionIncomplete(
        f"Scoped metadata extraction exhausted retries for {filename} purpose={purpose}: {last_error}"
    )


def _ground_scoped_candidates(
    extraction: ScopedMetadataExtraction,
    segments: list[MetadataSourceSegment],
) -> list[dict[str, Any]]:
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


def _bisect_metadata_segments(
    segments: list[MetadataSourceSegment],
) -> tuple[list[MetadataSourceSegment], list[MetadataSourceSegment]] | None:
    if len(segments) > 1:
        midpoint = len(segments) // 2
        return segments[:midpoint], segments[midpoint:]
    if not segments or len(segments[0].text) < 2000:
        return None
    segment = segments[0]
    midpoint = len(segment.text) // 2
    newline = segment.text.rfind("\n", 0, midpoint)
    if newline < midpoint // 2:
        newline = segment.text.find("\n", midpoint)
    split_at = newline if newline >= 0 else midpoint
    if split_at <= 0 or split_at >= len(segment.text):
        return None
    return (
        [replace(segment, text=segment.text[:split_at])],
        [replace(segment, text=segment.text[split_at:])],
    )


def _extract_scoped_metadata(
    filename: str,
    segments: list[MetadataSourceSegment],
    *,
    _split_depth: int = 0,
) -> list[dict[str, Any]]:
    rendered = "\n\n".join(_segment_text(segment) for segment in segments)
    try:
        extraction = _call_scoped_model(
            filename,
            _scoped_prompt_messages(filename, rendered),
            purpose="metadata_extraction.scoped_entities",
        )
        grounded = _ground_scoped_candidates(extraction, segments)
        expected_versions = _expected_version_kinds(segments)
        found_versions = {item["kind"] for item in grounded if item["kind"] in expected_versions}
        missing_versions = expected_versions - found_versions
        if missing_versions:
            focused = _call_scoped_model(
                filename,
                _version_prompt_messages(filename, rendered, missing_versions),
                purpose="metadata_extraction.version_applicability",
            )
            grounded.extend(_ground_scoped_candidates(focused, segments))
            found_versions = {item["kind"] for item in grounded if item["kind"] in expected_versions}
            missing_versions = expected_versions - found_versions
        if missing_versions:
            raise MetadataExtractionIncomplete(
                f"Version-bearing batch for {filename} is missing grounded {sorted(missing_versions)} evidence"
            )
        return _dedupe_evidence(grounded)
    except MetadataExtractionIncomplete:
        split = _bisect_metadata_segments(segments) if _split_depth < 5 else None
        if split is not None:
            logger.warning(
                "Bisecting failed metadata batch for %s at depth %s (%s source characters)",
                filename,
                _split_depth + 1,
                len(rendered),
            )
            left, right = split
            return _dedupe_evidence(
                _extract_scoped_metadata(filename, left, _split_depth=_split_depth + 1)
                + _extract_scoped_metadata(filename, right, _split_depth=_split_depth + 1)
            )
        raise
    except Exception as exc:
        raise MetadataExtractionIncomplete(f"Scoped metadata extraction failed for {filename}: {exc}") from exc


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


def _repeated_short_line_values(segments: list[MetadataSourceSegment]) -> set[str]:
    pages_by_line: dict[str, set[int | None]] = {}
    for segment in segments:
        for raw_line in segment.text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line or len(line) > 100:
                continue
            pages_by_line.setdefault(line.casefold(), set()).add(segment.page_from)
    repeated_lines = {line for line, pages in pages_by_line.items() if len(pages) >= 3}
    return {_compact_identifier(line) for line in repeated_lines if _compact_identifier(line)}


def _unsafe_routing_value(value: str, *, repeated_lines: set[str]) -> bool:
    stripped = value.strip()
    compact = _compact_identifier(stripped)
    if not compact or "_" in stripped or stripped.casefold().endswith(".pdf"):
        return True
    if compact in repeated_lines:
        return True
    if re.search(r"(?:^|[-_ ])(?:UM|IM|RM|MANUAL)(?:[-_ ]?[A-Z])?$", stripped, re.IGNORECASE):
        return True
    return False


def _first_primary(
    evidence: list[dict[str, Any]],
    kind: str,
    relation: str,
    *,
    repeated_lines: set[str] | None = None,
) -> str | None:
    repeated_lines = repeated_lines or set()
    matches = [
        item
        for item in evidence
        if item.get("kind") == kind and item.get("relation") == relation and item.get("grounded") is True
        and float(item.get("confidence") or 0.0) >= PRIMARY_ENTITY_MIN_CONFIDENCE
        and not (kind in {"product_model", "part_number"} and _unsafe_routing_value(str(item.get("value") or ""), repeated_lines=repeated_lines))
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (-float(item.get("confidence") or 0.0), int(item.get("page_from") or 10**9)))
    return str(matches[0]["value"])


def _values_for_routing(
    evidence: list[dict[str, Any]],
    kind: str,
    *,
    repeated_lines: set[str] | None = None,
) -> list[str]:
    repeated_lines = repeated_lines or set()
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
            and float(item.get("confidence") or 0.0) >= PRIMARY_ENTITY_MIN_CONFIDENCE
            and not (kind in {"product_model", "part_number"} and _unsafe_routing_value(str(item.get("value") or ""), repeated_lines=repeated_lines))
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
        and float(item.get("confidence") or 0.0) >= PRIMARY_ENTITY_MIN_CONFIDENCE
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
    repeated_lines = _repeated_short_line_values(nonempty)

    verified_product_models = _values_for_routing(scoped_evidence, "product_model", repeated_lines=repeated_lines)
    verified_part_numbers = _values_for_routing(scoped_evidence, "part_number", repeated_lines=repeated_lines)
    verified_protocols = [value.lower() for value in _values_for_routing(scoped_evidence, "protocol")]
    safe_base_product_models = [
        value
        for value in base.product_models
        if not _unsafe_routing_value(value, repeated_lines=repeated_lines)
    ]
    product_models = _dedupe_preserve_order(safe_base_product_models + verified_product_models)
    product_families = _dedupe_preserve_order(base.product_families + _values_for_routing(scoped_evidence, "product_family"))
    part_numbers = _dedupe_preserve_order(base.part_numbers + verified_part_numbers)
    devices = _dedupe_preserve_order(base.devices + _values_for_routing(scoped_evidence, "device"))
    protocols = _dedupe_preserve_order(base.protocol_terms + verified_protocols)
    # Schema-v2 routing is evidence-gated. Legacy flat values remain searchable metadata,
    # but cannot become hard-routing keys without scoped, high-confidence evidence.
    routing_product_models = verified_product_models
    routing_part_numbers = verified_part_numbers
    routing_protocols = verified_protocols
    identifiers = routing_product_models + routing_part_numbers + routing_protocols
    normalized_aliases = _dedupe_preserve_order(
        [alias for value in identifiers for alias in (value, _compact_identifier(value)) if alias]
    )

    primary_manufacturer = _first_primary(scoped_evidence, "company", "primary_manufacturer")
    primary_product = _first_primary(
        scoped_evidence,
        "product_model",
        "primary_product",
        repeated_lines=repeated_lines,
    )
    selected_product = primary_product or (verified_product_models[0] if verified_product_models else None)
    if selected_product is None and base.product_model and not _unsafe_routing_value(
        base.product_model,
        repeated_lines=repeated_lines,
    ):
        selected_product = base.product_model
    return replace(
        base,
        manufacturer=primary_manufacturer or base.manufacturer,
        companies=_dedupe_preserve_order(base.companies + _values_for_routing(evidence, "company")),
        product_model=selected_product,
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
