from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

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
        if candidate.casefold() not in NON_ENTITY_TERMS and "|" not in candidate and len(candidate.split()) <= 6:
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
