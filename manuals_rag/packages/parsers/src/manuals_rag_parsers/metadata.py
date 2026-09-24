from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
import json
import logging
import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
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
    "ethernet",
    "poe",
    "bluetooth",
}

PROTOCOL_ALIASES = {
    "ethernetip": "ethernet/ip",
    "ethercat": "ethercat",
    "profinet": "profinet",
    "modbus": "modbus",
    "tcpip": "tcp/ip",
    "udp": "udp",
    "rs232": "rs-232",
    "rs232c": "rs-232c",
    "rs422": "rs-422",
    "rs485": "rs-485",
    "usb": "usb",
    "iolink": "io-link",
    "canopen": "canopen",
    "cclink": "cc-link",
    "ethernet": "ethernet",
    "poe": "poe",
    "bluetooth": "bluetooth",
}
PROTOCOL_PATTERN = re.compile(
    r"\b(?:EtherNet/IP|Ethernet|EtherCAT|PROFINET|Modbus|TCP/IP|UDP|RS[- ]?232C?|RS[- ]?422|RS[- ]?485|"
    r"IO[- ]?Link|CANopen|CC[- ]?Link|Bluetooth|USB|PoE)\b",
    re.IGNORECASE,
)

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
DEFAULT_METADATA_SEGMENT_CHARS = 3000
MAX_FLAT_LIST_ITEMS = 12
MAX_SCOPED_ENTITIES = 10
MIN_SCOPED_SPLIT_CHARS = 500
METADATA_NUM_CTX = 16384
METADATA_EXTRACTION_ATTEMPTS = 3
PRIMARY_ENTITY_MIN_CONFIDENCE = 0.8
TITLE_PAGE_LIMIT = 2
TITLE_SOURCE_MAX_CHARS = 12000
CLAIM_VERIFICATION_BATCH_SIZE = 8
METADATA_MAP_MAX_CONCURRENCY = 2
MAX_HARVESTED_CANDIDATES = 80
METADATA_SCOPED_NUM_PREDICT = 5000
IDENTIFIER_CANDIDATE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{1,8}(?:[-:]\s*[A-Z0-9]{1,16})+|"
    r"[A-Z]{1,8}[A-Z-]*\d+[A-Z0-9-]*)(?![A-Za-z0-9])"
)
METADATA_PIPELINE_VERSION = "evidence_map_reduce_verify_v5"

DOCUMENT_KIND_ALIASES = {
    "user_manual": "manual",
    "user_guide": "manual",
    "instruction_manual": "manual",
    "release_notes": "release_note",
    "release_notes_document": "release_note",
    "data_sheet": "datasheet",
    "specification_sheet": "spec_sheet",
    "product_specification": "spec_sheet",
    "product_specifications": "spec_sheet",
    "technical_specification": "spec_sheet",
    "technical_specifications": "spec_sheet",
    "product_brochure": "brochure",
}

VERSION_SIGNAL_PATTERNS = {
    "firmware_version": re.compile(
        r"\b(?:firmware|fw)\b.{0,80}?\b(?:v(?:er(?:sion)?)?\.?\s*)?\d+(?:\.\d+){0,3}\b",
        re.IGNORECASE,
    ),
    "software_version": re.compile(
        r"(?:\b(?:software|application|tool|studio|explorer|[a-z]*editor|runtime|interpreter)\b"
        r"|\b[a-z]+\s+version\s+used\b).{0,80}?"
        r"\b(?:v(?:er(?:sion)?)?\.?\s*)\d+(?:\.\d+){0,3}\b",
        re.IGNORECASE,
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
    metadata_claims: list[dict[str, Any]] = field(default_factory=list)
    metadata_pipeline_version: str = "legacy"


@dataclass(frozen=True)
class MetadataSourceSegment:
    text: str
    page_from: int | None = None
    page_to: int | None = None
    section_path: tuple[str, ...] = ()


class MetadataClaim(BaseModel):
    """Framework-neutral, evidence-bearing intermediate metadata record."""

    value: str
    normalized_value: str
    kind: str
    relation: str
    subject: str | None = None
    source_quote: str
    page_from: int | None = None
    page_to: int | None = None
    section_path: list[str] = Field(default_factory=list)
    source_method: str
    verification_status: Literal["confirmed", "probable", "unresolved", "conflicting", "rejected"]
    confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool = True
    support_pages: list[int] = Field(default_factory=list)


class ScopedMetadataCandidate(BaseModel):
    value: str
    kind: str
    relation: str = "mentioned"
    subject: str | None = None
    source_quote: str
    # Candidate confidence is advisory only.  Reconciliation resets it and the
    # independent verifier must promote the grounded relationship to 0.8+.
    # Defaulting model omissions avoids discarding an otherwise valid verifier
    # response before that independent trust decision can be made.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_entity_type_alias(cls, value: Any) -> Any:
        if isinstance(value, dict):
            normalized = dict(value)
            if "value" not in normalized:
                for alias in (
                    "name",
                    "entity",
                    "entity_value",
                    "identifier",
                    "model",
                    "version",
                ):
                    if normalized.get(alias) not in (None, ""):
                        normalized["value"] = normalized[alias]
                        break
                if "value" not in normalized:
                    for key, candidate in normalized.items():
                        key_normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
                        if (
                            candidate not in (None, "")
                            and isinstance(candidate, (str, int, float))
                            and (key_normalized.endswith("_value") or key_normalized in {"entity_name", "item"})
                        ):
                            normalized["value"] = candidate
                            break
            if "kind" not in normalized:
                for alias in ("entity_kind", "entity_type", "type"):
                    if normalized.get(alias) not in (None, ""):
                        normalized["kind"] = normalized[alias]
                        break
            if "source_quote" not in normalized:
                for alias in (
                    "quote",
                    "evidence",
                    "evidence_quote",
                    "evidence_text",
                    "source_text",
                    "source_excerpt",
                    "excerpt",
                ):
                    if normalized.get(alias) not in (None, ""):
                        normalized["source_quote"] = normalized[alias]
                        break
                if "source_quote" not in normalized:
                    for key, candidate in normalized.items():
                        key_normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
                        if (
                            candidate not in (None, "")
                            and isinstance(candidate, str)
                            and ("quote" in key_normalized or "evidence" in key_normalized or key_normalized.endswith("_excerpt"))
                        ):
                            normalized["source_quote"] = candidate
                            break
            if "value" not in normalized and isinstance(normalized.get("source_quote"), str):
                quote = str(normalized["source_quote"]).casefold()
                excluded = {
                    "entity_type",
                    "kind",
                    "relation",
                    "relationship",
                    "subject",
                    "source_quote",
                    "confidence",
                    "score",
                }
                candidates: list[tuple[int, str | int | float]] = []
                for key, candidate in normalized.items():
                    key_normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
                    if key_normalized in excluded or not isinstance(candidate, (str, int, float)):
                        continue
                    rendered = str(candidate).strip()
                    if rendered and rendered.casefold() in quote:
                        candidates.append((len(rendered), candidate))
                if candidates:
                    normalized["value"] = min(candidates, key=lambda item: item[0])[1]
            if "relation" not in normalized and normalized.get("relationship") not in (None, ""):
                normalized["relation"] = normalized["relationship"]
            if "confidence" not in normalized and normalized.get("score") not in (None, ""):
                normalized["confidence"] = normalized["score"]
            confidence = normalized.get("confidence")
            if isinstance(confidence, str):
                rendered_confidence = confidence.strip()
                if rendered_confidence.endswith("%"):
                    try:
                        normalized["confidence"] = float(rendered_confidence[:-1]) / 100.0
                    except ValueError:
                        pass
            elif isinstance(confidence, (int, float)) and 1 < confidence <= 100:
                normalized["confidence"] = float(confidence) / 100.0
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
        aliased = DOCUMENT_KIND_ALIASES.get(normalized)
        if aliased:
            return aliased
        words = set(normalized.split("_"))
        if "release" in words and ({"note", "notes"} & words):
            return "release_note"
        if ({"data", "datasheet"} & words) and ({"sheet", "datasheet"} & words):
            return "datasheet"
        if {"specification", "specifications", "spec"} & words:
            return "spec_sheet"
        if "brochure" in words:
            return "brochure"
        if "troubleshooting" in words:
            return "troubleshooting_guide"
        if "installation" in words:
            return "installation_guide"
        if "service" in words:
            return "service_guide"
        if "setup" in words:
            return "setup_guide"
        if "safety" in words:
            return "safety_bulletin"
        if "parts" in words and ({"catalog", "catalogue"} & words):
            return "parts_catalog"
        if {"catalog", "catalogue"} & words:
            return "parts_catalog"
        if {"manual", "handbook", "guide"} & words:
            return "manual"
        return normalized

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


class TitleMetadataExtraction(BaseModel):
    title: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_title_aliases(cls, value: Any) -> Any:
        if value is None:
            return {"title": None}
        normalized = _normalize_object_response(value)
        if normalized.get("title") in (None, ""):
            for alias in ("document_title", "publication_title", "manual_title"):
                if normalized.get(alias) not in (None, ""):
                    normalized["title"] = normalized[alias]
                    break
        return normalized


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
                "For title, copy the publication title printed in TEXT; never use or rewrite FILENAME as the title. "
                "Set revision_date or effective_date only when the source explicitly labels the date as a revision, "
                "edition, effective, issued, or publication date; an unlabeled footer date is not sufficient. "
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


def _title_prompt_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You identify the title printed on the opening pages of a technical publication. "
                "Return a JSON object with exactly one title field. "
                "Copy the document's overarching publication title exactly, joining wrapped title lines with spaces. "
                "If there is no formal title, use the most prominent descriptive heading on the first page. "
                "Do not choose a section heading, feature caption, footer, document code, revision string, or page number. "
                "Use null only when neither opening page contains any meaningful title or cover heading."
            ),
        },
        {
            "role": "user",
            "content": f"OPENING PAGES:\n{text}\n\nReturn the printed publication title as title, or null.",
        },
    ]


def _scoped_prompt_messages(
    filename: str,
    text: str,
    harvested_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    candidate_json = json.dumps(harvested_candidates or [], ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You extract retrieval metadata from one page-aware manual excerpt. Return only JSON. "
                "Every entity must include an exact short source_quote copied from the excerpt. "
                "Do not infer an entity from a filename or from general knowledge. "
                "Classify references to another controller, PLC, accessory, example vendor, or host software "
                "as external_reference unless the excerpt explicitly says it applies to the manual's primary product. "
                "An accessory can itself be the product described by a catalog or datasheet: a bracket or cable "
                "with its own specifications is not external merely because it is an accessory. "
                "In a model/specification table, retain every model in the model column; distinguish those "
                "from models in a compatible-models column. Preserve the table heading in source_quote "
                "when it is needed to establish that relationship. "
                "Never attach a firmware or software version to a product unless the quote establishes that scope. "
                "Every entity must include a calibrated confidence from 0 to 1; do not use a fixed default."
                " A deterministic candidate harvester supplies recall-oriented candidates. Classify candidates only "
                "when the excerpt supports them. You may add an omitted value only by copying it exactly from the excerpt."
                f" Return at most {MAX_SCOPED_ENTITIES} highest-value distinct entities from this excerpt."
            ),
        },
        {
            "role": "user",
            "content": (
                f"FILENAME (context only; not evidence): {filename}\n\n"
                f"HARVESTED CANDIDATES (untrusted until classified):\n{candidate_json}\n\n{text}\n\n"
                "Extract only high-value routing metadata. Allowed kinds: company, product_family, product_model, "
                "device, part_number, protocol, firmware_version, software_name, software_version, document_revision. "
                "Allowed relations: primary_manufacturer, primary_product, applies_to, compatible_with, accessory_for, "
                "external_reference, mentioned, document_revision. For firmware_version and software_version, subject "
                "must name the product or software that the quote binds the version to; omit the entity if scope is unclear. "
                "For accessory_for, applies_to, and compatible_with, include subject naming the related "
                "product exactly as printed in source_quote. Never leave a known relationship subject null. "
                'Return {"entities":[{"value":"...","kind":"...","relation":"...",'
                '"subject":null,"source_quote":"...","confidence":0.9}]}; replace null with '
                "the explicit subject for relational claims. "
                "Use primary_manufacturer or primary_product only when the excerpt explicitly identifies the document owner/product."
            ),
        },
    ]


def _verification_prompt_messages(
    filename: str,
    claims: list[dict[str, Any]],
    segments: list[MetadataSourceSegment],
) -> list[dict[str, str]]:
    evidence_windows: list[str] = []
    for claim in claims:
        quote = str(claim.get("source_quote") or "")
        located = _quote_location(quote, segments)
        if located is None:
            continue
        text = located.text
        offset = text.casefold().find(quote.casefold())
        if offset < 0:
            window = text[:1200]
        else:
            window = text[max(0, offset - 400) : offset + len(quote) + 400]
        rendered = _segment_text(replace(located, text=window))
        if rendered not in evidence_windows:
            evidence_windows.append(rendered)
    source = "\n\n".join(evidence_windows)
    candidate_payload = [
        {
            key: claim.get(key)
            for key in ("value", "kind", "relation", "subject", "source_quote")
        }
        for claim in claims
    ]
    for index, candidate in enumerate(candidate_payload):
        candidate["claim_id"] = f"claim_{index + 1}"
        if candidate["kind"] in {"product_model", "product_family"} and candidate["relation"] == "applies_to":
            candidate["assertion"] = f"The documented function or manual applies to {candidate['value']}."
        elif candidate["relation"] == "primary_product":
            candidate["assertion"] = f"{candidate['value']} is a primary product or product family described by this document."
        else:
            candidate["assertion"] = f"{candidate['kind']} {candidate['value']} has relation {candidate['relation']} to {candidate.get('subject') or 'the documented system'}."
        # Required by the shared structured-output schema. This value is deliberately
        # ignored; publication confidence is derived from verifier agreement and evidence.
        candidate["confidence"] = 1.0
    return [
        {
            "role": "system",
            "content": (
                "You independently verify grounded metadata claims. Return a decision for EVERY claim_id, "
                "with supported true or false and a short evidence-based reason. Judge the explicit assertion; "
                "the typed fields are bookkeeping. Do not rewrite the claim. Reject when the "
                "quote does not prove the typed relationship, its subject is ambiguous, a version qualifier is lost, "
                "or an external/example device is classified as the primary product. A specification-table row "
                "explicitly labelled Model is direct primary-product evidence unless nearby text identifies it as an "
                "accessory or example. An opening-page publisher/manufacturer heading is direct manufacturer evidence "
                "when the same page describes the primary model. For product_model/product_family claims, "
                "applies_to with a null subject means this manual or documented function applies to the named "
                "product value: an explicit statement that a function is for those controllers supports it. "
                "Do not demand a second subject for this product-applicability relationship. Firmware/software "
                "version claims still require an explicit subject binding. Do not repair or add claims. "
                'Return valid JSON in exactly this shape: {"decisions":[{"claim_id":"claim_1",'
                '"supported":false,"reason":"Evidence-based reason"}]}. '
                "The example is a format template, not a verdict. Use double-quoted keys and strings, "
                "JSON booleans, no Markdown, and include each supplied claim_id exactly once."
            ),
        },
        {
            "role": "user",
            "content": (
                f"FILENAME (context only; not evidence): {filename}\n\n"
                f"SOURCE EVIDENCE:\n{source}\n\n"
                f"CLAIMS TO VERIFY:\n{json.dumps(candidate_payload, ensure_ascii=False)}\n\n"
                "Return decisions: [{claim_id, supported, reason}] for every supplied claim."
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
                "External product requirements must use external_reference. Do not invent a subject. "
                'Return exactly {"entities":[{"kind":"software_version","value":"1.2",'
                '"subject":"Exact software name","relation":"mentioned",'
                '"source_quote":"Exact quote containing software name and version",'
                '"confidence":0.9}]}. This example is format only, never evidence. '
                "Use kind firmware_version only for firmware. The value is the version number; "
                "preserve minimum/maximum qualifiers in source_quote. Do not emit extracted_versions, "
                "firmware_version or software_version as object keys, or omit subject."
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


def _metadata_thinking() -> bool:
    # Qwen3.5/Ollama's thinking workaround exhausted 8192 tokens without content
    # on source extraction. Use bounded non-thinking responses, validated here;
    # never rely on server-side format enforcement alone (upstream #14645).
    return False


def _metadata_token_budget(requested: int) -> int:
    return requested


def _scoped_metadata_schema() -> dict[str, Any]:
    schema = ScopedMetadataExtraction.model_json_schema()
    schema["additionalProperties"] = False
    entity_schema = schema.get("$defs", {}).get("ScopedMetadataCandidate")
    if isinstance(entity_schema, dict):
        entity_schema["additionalProperties"] = False
        entity_schema["required"] = list(dict.fromkeys([*entity_schema.get("required", []), "subject"]))
        properties = entity_schema.get("properties", {})
        if isinstance(properties.get("kind"), dict):
            properties["kind"]["enum"] = sorted(SCOPED_METADATA_KINDS)
        if isinstance(properties.get("relation"), dict):
            properties["relation"]["enum"] = sorted(SCOPED_METADATA_RELATIONS)
        for field_name, max_length in {
            "value": 160,
            "subject": 160,
            "source_quote": 500,
        }.items():
            if isinstance(properties.get(field_name), dict):
                properties[field_name]["maxLength"] = max_length
    entities_schema = schema.get("properties", {}).get("entities")
    if isinstance(entities_schema, dict):
        entities_schema["maxItems"] = MAX_SCOPED_ENTITIES
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
                f"Return JSON only with the key {field_name}. Return at most {MAX_FLAT_LIST_ITEMS} "
                "highest-value distinct items. Keep every item concise."
            ),
        },
    ]


def _list_field_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            field_name: {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "maxItems": MAX_FLAT_LIST_ITEMS,
            }
        },
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
                think=_metadata_thinking(),
                timeout=settings.ollama_metadata_timeout_seconds,
                purpose="metadata_extraction",
                num_predict=_metadata_token_budget(320),
                num_ctx=METADATA_NUM_CTX,
                num_batch=settings.ollama_metadata_num_batch,
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


def _extract_printed_title(text: str) -> str | None:
    last_error: Exception | None = None
    for attempt in range(1, METADATA_EXTRACTION_ATTEMPTS + 1):
        try:
            parsed, _raw = chat_json(
                model=settings.ollama_metadata_model,
                messages=_title_prompt_messages(text),
                json_schema=TitleMetadataExtraction.model_json_schema(),
                think=_metadata_thinking(),
                timeout=settings.ollama_metadata_timeout_seconds,
                purpose="metadata_extraction.document_title",
                num_predict=_metadata_token_budget(160),
                num_ctx=METADATA_NUM_CTX,
                num_batch=settings.ollama_metadata_num_batch,
            )
            candidate = TitleMetadataExtraction.model_validate(parsed).title
            if candidate is None:
                return None
            candidate = " ".join(candidate.split()).strip()
            return candidate if _value_is_grounded(candidate, text) else None
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Printed title extraction attempt %s/%s failed: %s",
                attempt,
                METADATA_EXTRACTION_ATTEMPTS,
                exc,
            )
    logger.warning("Printed title extraction exhausted retries: %s", last_error)
    return None


def _value_is_grounded(value: str, source: str) -> bool:
    normalized_value = " ".join(value.casefold().split())
    normalized_source = " ".join(source.casefold().split())
    return bool(normalized_value) and normalized_value in normalized_source


def _identifier_is_grounded(value: str, source: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(stripped)}(?![A-Za-z0-9])",
        source,
        re.IGNORECASE,
    ) is not None


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
        if field_name == "menu_labels":
            if not (stripped.startswith("[") and stripped.endswith("]")):
                continue
            if re.fullmatch(r"\[(?:PAGE|SECTION)(?:\s+[^\]]+)?\]", stripped, re.IGNORECASE):
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


def _ground_date(
    value: date | None,
    filename: str,
    text: str,
    *,
    label_pattern: str | None = None,
) -> date | None:
    if value is None:
        return None
    source = _source_text(filename, text)
    candidates = {
        value.isoformat(),
        value.strftime("%Y/%m/%d"),
        value.strftime("%m/%d/%Y"),
    }
    for candidate in candidates:
        for match in re.finditer(re.escape(candidate), source, flags=re.IGNORECASE):
            if label_pattern is None:
                return value
            context = source[max(0, match.start() - 80) : min(len(source), match.end() + 80)]
            if re.search(label_pattern, context, flags=re.IGNORECASE):
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
                think=_metadata_thinking(),
                timeout=settings.ollama_metadata_timeout_seconds,
                purpose=f"metadata_extraction.{field_name}",
                num_predict=_metadata_token_budget(1024),
                num_ctx=METADATA_NUM_CTX,
                num_batch=settings.ollama_metadata_num_batch,
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
    if (
        product_model is None
        and product_family
        and any(char.isdigit() for char in product_family)
        and re.fullmatch(r"[A-Za-z]{1,8}(?:[-: ]?[A-Za-z0-9]+)+", product_family)
    ):
        # Models are sometimes returned in the adjacent family field. Promote only
        # grounded, compact alphanumeric identifiers; descriptive family names and
        # ungrounded values remain families rather than retrieval identities.
        product_model = product_family
        product_models = _dedupe_preserve_order([*product_models, product_family])
    proposed_title = " ".join((extraction.title or "").split()).strip()
    title = proposed_title if proposed_title and _value_is_grounded(proposed_title, text) else _normalize_title(filename)
    revision_date = _ground_date(
        extraction.revision_date,
        filename,
        text,
        label_pattern=r"\b(?:revision|revised|rev\.?|edition)\b",
    )
    effective_date = _ground_date(
        extraction.effective_date,
        filename,
        text,
        label_pattern=r"\b(?:effective|issued|published|publication\s+date)\b",
    )
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


def harvest_metadata_candidates(segments: list[MetadataSourceSegment]) -> list[dict[str, Any]]:
    """Harvest broad, corpus-general candidates before asking the model to classify them."""
    candidates: list[dict[str, Any]] = []
    version_pattern = re.compile(
        r"\b(?:firmware|software|version|ver\.?|revision|rev\.?)\b[^\n]{0,100}?\b\d+(?:\.\d+){0,3}\b",
        re.IGNORECASE,
    )
    seen: set[tuple[str, str, int | None]] = set()
    for segment in segments:
        for raw_line in segment.text.splitlines():
            quote = " ".join(raw_line.split()).strip()
            if not quote:
                continue
            for kind, pattern in (
                ("identifier", IDENTIFIER_CANDIDATE_PATTERN),
                ("version_statement", version_pattern),
                ("protocol", PROTOCOL_PATTERN),
            ):
                for match in pattern.finditer(quote):
                    value = match.group(0).strip()
                    fingerprint = (kind, _compact_identifier(value), segment.page_from)
                    if not value or fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    candidates.append(
                        {
                            "candidate_kind": kind,
                            "value": value,
                            "source_quote": quote[:240],
                            "page_from": segment.page_from,
                            "page_to": segment.page_to,
                            "section_path": list(segment.section_path),
                        }
                    )
                    if len(candidates) >= MAX_HARVESTED_CANDIDATES:
                        return candidates
    return candidates


def _opening_title_identifier_evidence(
    selected_title: str,
    segments: list[MetadataSourceSegment],
) -> list[dict[str, Any]]:
    """Turn grounded identifiers in the selected printed title into verifiable claims."""
    opening_segments = _opening_page_segments(segments)
    located = _quote_location(selected_title, opening_segments)
    if located is None:
        return []
    evidence: list[dict[str, Any]] = []
    for match in IDENTIFIER_CANDIDATE_PATTERN.finditer(selected_title):
        value = re.sub(r"\s*([-:])\s*", r"\1", match.group(0).strip())
        canonical = _canonical_routing_identifier(value, repeated_lines=set()) or value
        kind = "part_number" if re.fullmatch(r"OP-\d+[A-Z0-9-]*", canonical, re.IGNORECASE) else "product_model"
        evidence.append(
            {
                "value": canonical,
                "kind": kind,
                "relation": "mentioned",
                "subject": None,
                "source_quote": selected_title,
                "page_from": located.page_from,
                "page_to": located.page_to,
                "section_path": list(located.section_path),
                "confidence": 0.45,
                "grounded": True,
                "source": "opening_title_candidate",
            }
        )
    return evidence


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


def _missing_explicit_software_versions(segments, evidence) -> set[str]:
    """Check each explicit version in software-bearing source units, not just kind presence."""
    expected = {
        match.group(1)
        for segment in segments
        for line in segment.text.splitlines()
        if VERSION_SIGNAL_PATTERNS["software_version"].search(line)
        for match in re.finditer(r"\bVer(?:sion)?\.?\s*(\d+(?:\.\d+){0,3})\b", line, re.I)
    }
    # A flattened table/paragraph can contain both firmware and software. Keep
    # its already-grounded firmware claim from becoming a false software gap.
    found = set()
    for item in evidence:
        if item.get("kind") not in {"software_version", "firmware_version"}:
            continue
        value = str(item["value"]).strip()
        match = re.fullmatch(r"(?:Ver(?:sion)?\.?\s*)?(\d+(?:\.\d+){0,3})", value, re.I)
        found.add(match.group(1) if match else value)
    return expected - found


def _parenthesized_version_mentions(text: str) -> list[tuple[str, str]]:
    """Read a named subject's explicit version list, never its applicability range."""
    mentions = []
    for match in re.finditer(
        r"(?P<subject>[A-Z][A-Za-z0-9+_.-]*(?:\s+[A-Z][A-Za-z0-9+_.-]*){0,3})"
        r"\s*\((?P<body>[^()\n]{1,160})\)", text
    ):
        body = match.group("body")
        version_pattern = r"\bVer(?:sion)?\.?\s*(\d+(?:\.\d+){0,3})\b"
        versions = re.findall(version_pattern, body, re.I)
        residue = re.sub(version_pattern, "", body, flags=re.I)
        residue = re.sub(r"\b(?:or|and|later|earlier|above|below)\b|[,;/\s]+", "", residue, flags=re.I)
        if versions and not residue:
            mentions.extend((match.group("subject"), value) for value in versions)
    return mentions


def _deterministic_version_evidence(
    segments: list[MetadataSourceSegment],
    expected_kinds: set[str],
) -> list[dict[str, Any]]:
    """Recover explicit same-line version statements without inferring applicability."""
    patterns: dict[str, re.Pattern[str]] = {}
    if "software_version" in expected_kinds:
        patterns["software_version"] = re.compile(
            r"(?P<subject>[A-Za-z][A-Za-z0-9+_.-]*(?:\s+[A-Za-z][A-Za-z0-9+_.-]*){0,3})"
            r"\s*Ver(?:sion)?\.?\s*(?P<version>\d+(?:\.\d+){0,3})\b",
            re.IGNORECASE,
        )
    recovered: list[dict[str, Any]] = []
    for segment in segments:
        for raw_line in segment.text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if "software_version" in expected_kinds and VERSION_SIGNAL_PATTERNS["software_version"].search(line):
                for subject, version in _parenthesized_version_mentions(line):
                    recovered.append({
                        "value": version, "kind": "software_version", "relation": "mentioned",
                        "subject": subject, "source_quote": line, "page_from": segment.page_from,
                        "page_to": segment.page_to, "section_path": list(segment.section_path),
                        "confidence": 0.95, "grounded": True, "source": "deterministic_explicit_version",
                    })
            for kind, pattern in patterns.items():
                for match in pattern.finditer(line):
                    subject = " ".join(match.group("subject").split()).strip(" |,;:")
                    version = match.group("version")
                    if not subject or not version:
                        continue
                    if re.search(r"\b(?:is|are|was|used|this|using)\b", subject, re.IGNORECASE):
                        continue
                    recovered.append(
                        {
                            "value": version,
                            "kind": kind,
                            "relation": "mentioned",
                            "subject": subject,
                            "source_quote": line,
                            "page_from": segment.page_from,
                            "page_to": segment.page_to,
                            "section_path": list(segment.section_path),
                            "confidence": 0.95,
                            "grounded": True,
                            "source": "deterministic_explicit_version",
                        }
                    )
    return _dedupe_evidence(recovered)


def _call_scoped_model(
    filename: str,
    messages: list[dict[str, str]],
    *,
    purpose: str,
) -> ScopedMetadataExtraction:
    verification_candidates: dict[str, dict[str, Any]] = {}
    schema = _scoped_metadata_schema()
    if purpose == "metadata_extraction.claim_verification":
        content = messages[1]["content"]
        candidates = json.loads(content.rsplit("CLAIMS TO VERIFY:\n", 1)[1].split("\n\n", 1)[0])
        verification_candidates = {item["claim_id"]: item for item in candidates}
        schema = {
            "type": "object", "additionalProperties": False, "required": ["decisions"],
            "properties": {"decisions": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["claim_id", "supported", "reason"], "properties": {
                    "claim_id": {"type": "string", "enum": list(verification_candidates)},
                    "supported": {"type": "boolean"}, "reason": {"type": "string"},
                },
            }}},
        }
    last_error: Exception | None = None
    for attempt in range(1, METADATA_EXTRACTION_ATTEMPTS + 1):
        try:
            parsed, _raw = chat_json(
                model=settings.ollama_metadata_model,
                messages=messages,
                json_schema=schema,
                think=_metadata_thinking(),
                timeout=settings.ollama_metadata_timeout_seconds,
                purpose=purpose,
                num_predict=_metadata_token_budget(METADATA_SCOPED_NUM_PREDICT),
                num_ctx=METADATA_NUM_CTX,
                num_batch=settings.ollama_metadata_num_batch,
            )
            if verification_candidates and isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "claim_id" in parsed[0]:
                parsed = {"decisions": parsed}
            if verification_candidates and isinstance(parsed, dict) and "claim_id" in parsed:
                parsed = {"decisions": [parsed]}
            if verification_candidates and isinstance(parsed, dict) and "decisions" in parsed:
                decisions = parsed["decisions"]
                if not isinstance(decisions, list):
                    raise ValueError("Verifier decisions must be an array")
                seen: set[str] = set()
                accepted = []
                for decision in decisions:
                    if not isinstance(decision, dict):
                        raise ValueError("Invalid verifier decision")
                    claim_id = decision.get("claim_id")
                    if claim_id not in verification_candidates or claim_id in seen:
                        raise ValueError("Verifier returned unknown or duplicate claim ID")
                    if type(decision.get("supported")) is not bool or not str(decision.get("reason") or "").strip():
                        raise ValueError("Verifier decision requires boolean support and a reason")
                    seen.add(claim_id)
                    if decision["supported"]:
                        accepted.append(verification_candidates[claim_id])
                if seen != set(verification_candidates):
                    raise ValueError("Verifier omitted claim decisions")
                parsed = {"entities": accepted}
            normalized = _normalize_object_response(parsed, collection_key="entities")
            if "entities" not in normalized:
                raise ValueError("Scoped response omitted its entities or decisions collection")
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
        if kind in {"product_model", "part_number"} and not _identifier_is_grounded(value, quote):
            continue
        if kind == "part_number" and re.search(r"copyright|printed in", quote, re.I) and not re.search(r"\b(?:part|order|model|accessory)\b", quote, re.I):
            # Publication/footer codes are not component part numbers.
            continue
        if kind in {"product_model", "product_family"} and relation == "primary_product":
            if re.search(
                r"\b(?:bracket|column|cable|adapter|accessory)\b.{0,100}\bfor\s+" + re.escape(value), quote, re.I
            ):
                # The receiver of an accessory is only mentioned here; this quote
                # does not establish it as the product being specified.
                relation = "mentioned"
            elif not any(char.isdigit() for char in value) and not re.search(
                r"\b(?:series|family|product|model|controller|sensor|scanner|camera|unit|system|"
                r"manual|datasheet|data\s+sheet|bracket|column|cable|adapter|accessory)\b",
                quote,
                re.IGNORECASE,
            ):
                # Short alphabetic publication/region codes can look like
                # product families (for example "KA-US 2114-1 689034").  A
                # code-only footer cannot establish a primary product.
                relation = "mentioned"
        subject = candidate.subject.strip() if candidate.subject else None
        if subject and not _value_is_grounded(subject, quote):
            continue
        if kind in {"firmware_version", "software_version"} and not subject:
            continue
        if relation in {"applies_to", "compatible_with", "accessory_for"}:
            if subject and subject.casefold() in {"product", "device", "system", "manual"}:
                continue
            if subject and not _value_is_grounded(subject, quote):
                continue
        if kind == "firmware_version" and not re.search(r"\b(?:firmware|fw)\b", quote, re.IGNORECASE):
            # A bare software/runtime "version" is not evidence of firmware.
            if not subject or _canonical_routing_identifier(subject, repeated_lines=set()) is None:
                continue
            if not re.search(r"\bversion\b", quote, re.IGNORECASE):
                continue
        if kind == "software_version" and not (
            re.search(r"\b(?:software|application|tool|version|ver\.?|studio|explorer)\b", quote, re.IGNORECASE)
            or (subject and _value_is_grounded(subject, quote) and re.search(r"\d", quote))
        ):
            continue
        if kind in {"firmware_version", "software_version"} and _compact_identifier(subject or "") in external_subjects:
            relation = "external_reference"
        grounded.append(
            {
                "value": value,
                "kind": kind,
                "relation": relation,
                "subject": subject,
                # Grounding was checked against this entire quote. Truncating it
                # afterward can remove the very identifier/version just verified.
                "source_quote": quote,
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
    if not segments or len(segments[0].text) < MIN_SCOPED_SPLIT_CHARS:
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
    # The response schema caps entities. A dense table can exceed that cap while
    # still fitting the character budget; a successful response then silently
    # loses its later rows. Split before extraction, not only after JSON errors.
    candidates = harvest_metadata_candidates(segments)
    if len(candidates) > MAX_SCOPED_ENTITIES and _split_depth < 5:
        split = _bisect_metadata_segments(segments)
        if split is not None:
            return _dedupe_evidence([
                item
                for part in split
                for item in _extract_scoped_metadata(filename, part, _split_depth=_split_depth + 1)
            ])
    try:
        extraction = _call_scoped_model(
            filename,
            _scoped_prompt_messages(filename, rendered, candidates),
            purpose="metadata_extraction.scoped_entities",
        )
        grounded = _ground_scoped_candidates(extraction, segments)
        expected_versions = _expected_version_kinds(segments)
        found_versions = {item["kind"] for item in grounded if item["kind"] in expected_versions}
        missing_versions = expected_versions - found_versions
        if missing_versions or _missing_explicit_software_versions(segments, grounded):
            focused = _call_scoped_model(
                filename,
                _version_prompt_messages(filename, rendered, expected_versions),
                purpose="metadata_extraction.version_applicability",
            )
            grounded.extend(_ground_scoped_candidates(focused, segments))
            found_versions = {item["kind"] for item in grounded if item["kind"] in expected_versions}
            missing_versions = expected_versions - found_versions
        if missing_versions:
            grounded.extend(_deterministic_version_evidence(segments, missing_versions))
            found_versions = {item["kind"] for item in grounded if item["kind"] in expected_versions}
            missing_versions = expected_versions - found_versions
        if missing_versions:
            raise MetadataExtractionIncomplete(
                f"Version-bearing batch for {filename} is missing grounded {sorted(missing_versions)} evidence"
            )
        missing_values = _missing_explicit_software_versions(segments, grounded)
        if missing_values:
            raise MetadataExtractionIncomplete(f"Missing explicit software versions: {sorted(missing_values)}")
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
    seen: dict[tuple[str, str, str, str, int | None], int] = {}
    provenance_priority = {
        "upload_identity_page_grounded": 3,
        "opening_title_candidate": 4,
    }
    for item in evidence:
        fingerprint = (
            str(item.get("kind") or ""),
            _compact_identifier(str(item.get("value") or "")),
            str(item.get("relation") or ""),
            _compact_identifier(str(item.get("subject") or "")),
            item.get("page_from"),
        )
        if fingerprint in seen:
            existing_index = seen[fingerprint]
            existing = deduped[existing_index]
            item_source = str(item.get("source_method") or item.get("source") or "")
            existing_source = str(existing.get("source_method") or existing.get("source") or "")
            if provenance_priority.get(item_source, 0) > provenance_priority.get(existing_source, 0):
                # Preserve deterministic identity provenance when the model
                # independently emits the same mentioned identifier. Routing
                # policy depends on this provenance, not on raw confidence.
                deduped[existing_index] = item
            continue
        seen[fingerprint] = len(deduped)
        deduped.append(item)
    return deduped


def _claim_fingerprint(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(item.get("kind") or "").casefold(),
        _compact_identifier(str(item.get("value") or "")),
        str(item.get("relation") or "").casefold(),
        _compact_identifier(str(item.get("subject") or "")),
        " ".join(str(item.get("source_quote") or "").casefold().split()),
    )


def _claim_group_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    """Group equivalent page claims while retaining distinct relationships."""
    return (
        str(item.get("kind") or "").casefold(),
        _normalized_claim_value(item),
        str(item.get("relation") or "").casefold(),
        _compact_identifier(str(item.get("subject") or "")),
    )


def _normalized_claim_value(item: dict[str, Any]) -> str:
    value = str(item.get("value") or "")
    if item.get("kind") in {"product_model", "part_number", "product_family", "protocol"}:
        return _compact_identifier(value)
    return " ".join(value.casefold().split())


def _claim_scope_key(item: dict[str, Any]) -> tuple[str, str, str]:
    kind = str(item.get("kind") or "")
    subject = (
        _compact_identifier(str(item.get("subject") or ""))
        if kind in {"firmware_version", "software_version"}
        else ""
    )
    return kind, _normalized_claim_value(item), subject


def reconcile_metadata_claims(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce page claims into a document ledger without product-specific knowledge."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for raw_item in evidence:
        item = _sanitize_claim_shape(raw_item)
        if item is not None:
            groups.setdefault(_claim_group_key(item), []).append(item)
    reduced: list[dict[str, Any]] = []
    for items in groups.values():
        representative = dict(items[0])
        pages = sorted(
            {
                int(item["page_from"])
                for item in items
                if item.get("page_from") is not None
            }
        )
        representative["support_pages"] = pages
        representative["source_method"] = str(
            representative.pop(
                "source",
                representative.get("source_method", "page_aware_model_extraction"),
            )
        )
        representative["normalized_value"] = _normalized_claim_value(representative)
        representative["verification_status"] = "unresolved"
        representative["confidence"] = 0.45 if representative.get("grounded") is True else 0.0
        reduced.append(representative)

    relations_by_entity: dict[tuple[str, str, str], set[str]] = {}
    for item in reduced:
        key = _claim_scope_key(item)
        relations_by_entity.setdefault(key, set()).add(str(item.get("relation") or ""))
    for item in reduced:
        key = _claim_scope_key(item)
        relations = relations_by_entity[key]
        # Being compatible with a protocol/device and mentioning it in an
        # external example are not mutually exclusive. Only primary identity
        # versus external identity is a document-wide role contradiction.
        if "external_reference" in relations and "primary_product" in relations:
            item["verification_status"] = "conflicting"
    return reduced


_RELATIONS_BY_CLAIM_KIND: dict[str, set[str]] = {
    "company": {"primary_manufacturer", "external_reference", "mentioned"},
    "product_family": {"primary_product", "applies_to", "compatible_with", "external_reference", "mentioned"},
    "product_model": {"primary_product", "applies_to", "compatible_with", "external_reference", "mentioned"},
    "device": {"primary_product", "applies_to", "compatible_with", "accessory_for", "external_reference", "mentioned"},
    "part_number": {"applies_to", "compatible_with", "accessory_for", "external_reference", "mentioned"},
    "protocol": {"applies_to", "compatible_with", "external_reference", "mentioned"},
    "firmware_version": {"applies_to", "compatible_with", "external_reference", "mentioned"},
    "software_name": {"applies_to", "compatible_with", "external_reference", "mentioned"},
    "software_version": {"applies_to", "compatible_with", "external_reference", "mentioned"},
    "document_revision": {"document_revision", "mentioned"},
}


def _sanitize_claim_shape(raw_item: dict[str, Any]) -> dict[str, Any] | None:
    """Reject prose-shaped entities and impossible kind/relation pairs before verification."""
    item = dict(raw_item)
    kind = str(item.get("kind") or "").strip().casefold()
    relation = str(item.get("relation") or "").strip().casefold()
    value = " ".join(str(item.get("value") or "").split()).strip(" |,.;:")
    if not value or kind not in _RELATIONS_BY_CLAIM_KIND:
        return None
    if relation not in _RELATIONS_BY_CLAIM_KIND[kind]:
        relation = "mentioned"
    if kind == "company":
        value = re.split(r"\b(?:all rights reserved|printed in)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
        value = re.sub(r"^copyright\s*(?:\([cC]\)|©)?\s*\d{4}\s*", "", value, flags=re.IGNORECASE)
        value = value.strip(" |,.;:")
        if not _plausible_company_name(value):
            return None
    if kind in {"product_model", "part_number"}:
        if len(value) > 80 or len(value.split()) > 6:
            return None
        if not _expand_routing_identifiers(value, repeated_lines=set()):
            return None
    if kind in {"product_family", "device", "software_name"} and (
        len(value) > 120 or len(value.split()) > 12
    ):
        return None
    if kind in {"firmware_version", "software_version"} and relation != "external_reference" and not item.get("subject"):
        return None
    item["kind"] = kind
    item["relation"] = relation
    item["value"] = value
    return item


def _segments_for_claims(
    claims: list[dict[str, Any]],
    segments: list[MetadataSourceSegment],
) -> list[MetadataSourceSegment]:
    pages = {claim.get("page_from") for claim in claims}
    selected = [segment for segment in segments if segment.page_from in pages]
    return selected or segments


def _literal_opening_title_claim_is_confirmed(claim: dict[str, Any]) -> bool:
    """Confirm literal model or part identifiers harvested from an opening-page title."""
    if (
        claim.get("source_method") != "opening_title_candidate"
        or claim.get("kind") not in {"product_model", "part_number"}
        or claim.get("relation") != "mentioned"
        or claim.get("grounded") is not True
        or int(claim.get("page_from") or 10**9) > TITLE_PAGE_LIMIT
    ):
        return False
    value = _compact_identifier(str(claim.get("value") or ""))
    quote = _compact_identifier(str(claim.get("source_quote") or ""))
    return bool(value) and value in quote


def _literal_upload_identity_claim_is_confirmed(claim: dict[str, Any]) -> bool:
    """Confirm a filename identifier only when the same identifier is on an opening page."""
    if (
        claim.get("source_method") != "upload_identity_page_grounded"
        or claim.get("kind") not in {"product_model", "part_number"}
        or claim.get("relation") != "mentioned"
        or claim.get("grounded") is not True
        or int(claim.get("page_from") or 10**9) > 3
    ):
        return False
    value = _compact_identifier(str(claim.get("value") or ""))
    if _identifier_is_grounded(
        str(claim.get("value") or ""),
        str(claim.get("source_quote") or ""),
    ):
        return True
    quote_values = {
        _compact_identifier(candidate)
        for candidate in _expand_routing_identifiers(
            str(claim.get("source_quote") or ""), repeated_lines=set()
        )
    }
    return bool(value) and value in quote_values


def _literal_protocol_mention_is_confirmed(claim: dict[str, Any]) -> bool:
    """Confirm only the fact that an explicit protocol token occurs in the quote."""
    if (
        claim.get("source_method") != "deterministic_protocol_mention"
        or claim.get("kind") != "protocol"
        or claim.get("relation") != "mentioned"
        or claim.get("grounded") is not True
    ):
        return False
    expected = _canonical_protocol(str(claim.get("value") or ""))
    observed = {
        _canonical_protocol(match.group(0))
        for match in PROTOCOL_PATTERN.finditer(str(claim.get("source_quote") or ""))
    }
    return bool(expected) and expected in observed


def _literal_deterministic_version_claim_is_confirmed(claim: dict[str, Any]) -> bool:
    """Confirm an exact same-line version mention without assigning applicability."""
    if (
        claim.get("source_method") != "deterministic_explicit_version"
        or claim.get("kind") not in {"firmware_version", "software_version"}
        or claim.get("relation") != "mentioned"
        or claim.get("grounded") is not True
    ):
        return False
    subject = " ".join(str(claim.get("subject") or "").split())
    value = " ".join(str(claim.get("value") or "").split())
    quote = " ".join(str(claim.get("source_quote") or "").split())
    if not subject or not value or not quote:
        return False
    if claim.get("kind") == "software_version" and (subject, value) in _parenthesized_version_mentions(quote):
        return True
    return re.search(
        rf"{re.escape(subject)}\s+Ver(?:sion)?\.?\s*{re.escape(value)}\b",
        quote,
        re.IGNORECASE,
    ) is not None


def _literal_compatible_model_column_claim_is_confirmed(
    claim: dict[str, Any],
) -> bool:
    """Confirm a model relationship only when the quote includes the table contract and row."""
    if (
        claim.get("source_method") != "deterministic_compatible_model_column"
        or claim.get("kind") != "product_model"
        or claim.get("relation") != "compatible_with"
        or claim.get("grounded") is not True
    ):
        return False
    quote = str(claim.get("source_quote") or "")
    return (
        "compatible" in quote.casefold()
        and "model" in quote.casefold()
        and _identifier_is_grounded(str(claim.get("subject") or ""), quote)
        and _identifier_is_grounded(str(claim.get("value") or ""), quote)
    )


def verify_metadata_claims(
    filename: str,
    claims: list[dict[str, Any]],
    segments: list[MetadataSourceSegment],
) -> list[dict[str, Any]]:
    """Independently verify typed relationships and derive confidence from evidence."""
    verified: list[dict[str, Any]] = []
    for offset in range(0, len(claims), CLAIM_VERIFICATION_BATCH_SIZE):
        batch = claims[offset : offset + CLAIM_VERIFICATION_BATCH_SIZE]
        local_segments = _segments_for_claims(batch, segments)
        accepted: set[tuple[str, str, str, str, str]] = {
            _claim_fingerprint(claim)
            for claim in batch
            if _literal_opening_title_claim_is_confirmed(claim)
            or _literal_upload_identity_claim_is_confirmed(claim)
            or _literal_protocol_mention_is_confirmed(claim)
            or _literal_deterministic_version_claim_is_confirmed(claim)
            or _literal_compatible_model_column_claim_is_confirmed(claim)
        }
        verification_completed = False
        try:
            extraction = _call_scoped_model(
                filename,
                _verification_prompt_messages(filename, batch, local_segments),
                purpose="metadata_extraction.claim_verification",
            )
            accepted_evidence = _ground_scoped_candidates(extraction, local_segments)
            accepted.update(_claim_fingerprint(item) for item in accepted_evidence)
            verification_completed = True
        except MetadataExtractionIncomplete as exc:
            # Verification is a publication gate, not a best-effort enrichment step.
            # Persisting claims from a partially verified document can turn a transient
            # model failure into authoritative routing metadata. Quarantine the entire
            # document so a later run can retry it cleanly.
            raise MetadataExtractionIncomplete(
                f"Independent claim verification did not complete for {filename}: {exc}"
            ) from exc

        # Deterministically harvested title/version claims are especially important to
        # routing and completeness. If a crowded verifier batch omits one, retry that
        # exact grounded claim alone before treating the omission as a rejection.
        for claim in batch:
            fingerprint = _claim_fingerprint(claim)
            if (
                fingerprint in accepted
                or (claim.get("kind") not in {"product_model", "product_family"}
                    and claim.get("source_method") not in {
                    "deterministic_explicit_version",
                    "opening_title_candidate",
                    "upload_identity_page_grounded",
                    "deterministic_protocol_mention",
                })
            ):
                continue
            claim_segments = _segments_for_claims([claim], segments)
            try:
                retry = _call_scoped_model(
                    filename,
                    _verification_prompt_messages(filename, [claim], claim_segments),
                    purpose="metadata_extraction.claim_verification",
                )
                accepted.update(
                    _claim_fingerprint(item)
                    for item in _ground_scoped_candidates(retry, claim_segments)
                )
                verification_completed = True
            except MetadataExtractionIncomplete as exc:
                logger.warning(
                    "Focused claim verification failed safely for %s: %s",
                    filename,
                    exc,
                )

        for claim in batch:
            item = dict(claim)
            if item.get("verification_status") == "conflicting":
                item["confidence"] = 0.35
            elif _claim_fingerprint(item) in accepted:
                score = 0.80
                if len(item.get("support_pages") or []) > 1:
                    score += 0.05
                if item.get("subject") and _value_is_grounded(
                    str(item["subject"]), str(item.get("source_quote") or "")
                ):
                    score += 0.05
                if int(item.get("page_from") or 10**9) <= TITLE_PAGE_LIMIT:
                    score += 0.05
                item["verification_status"] = "confirmed"
                item["confidence"] = min(score, 1.0)
            elif verification_completed:
                item["verification_status"] = "rejected"
                item["confidence"] = 0.0
            else:
                item["verification_status"] = "unresolved"
                item["confidence"] = min(float(item.get("confidence") or 0.0), 0.45)
            verified.append(MetadataClaim.model_validate(item).model_dump())
    return verified


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
    if "|" in stripped:
        return True
    if compact in repeated_lines:
        return True
    if re.search(r"(?:^|[-_ ])(?:UM|IM|RM|MANUAL)(?:[-_ ]?[A-Z])?$", stripped, re.IGNORECASE):
        return True
    return False


def _canonical_routing_identifier(value: str, *, repeated_lines: set[str]) -> str | None:
    """Return one exact identifier from a model-produced phrase, or reject it."""
    stripped = value.strip()
    if _unsafe_routing_value(stripped, repeated_lines=repeated_lines):
        if "|" not in stripped:
            return None
        stripped = stripped.rsplit("|", 1)[-1].strip()
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", stripped).strip()
    stripped = re.sub(r"\s*([-:])\s*", r"\1", stripped)
    stripped = re.sub(r"\s+(?:series|family)$", "", stripped, flags=re.IGNORECASE).strip()
    if re.search(r"\s", stripped):
        return None
    if "/" in stripped and stripped.casefold() not in {"ethernet/ip", "tcp/ip"}:
        return None
    matches = re.findall(
        r"(?<![A-Z0-9])(?:[A-Z]{1,8}(?:[-:][A-Z0-9]+)+|[A-Z]{1,8}[A-Z-]*\d+[A-Z0-9-]*)(?![A-Z0-9])",
        stripped.upper(),
    )
    matches = [
        match.strip("-:")
        for match in matches
        if any(char.isalpha() for char in match)
        and (
            any(char.isdigit() for char in match)
            or re.fullmatch(r"[A-Z]{2,4}[-:][A-Z]{1,2}", match) is not None
        )
    ]
    if len(matches) != 1:
        return None
    # OCR commonly renders the narrow hyphen in cover-page model names as a
    # colon (for example ``KV: X`` and ``LJ: X8000``).  Model identifiers in
    # this corpus use hyphens, so keep the punctuation-insensitive alias while
    # publishing a stable canonical spelling for routing and display.
    candidate = matches[0].replace(":", "-")
    return candidate


def _expand_routing_identifiers(value: str, *, repeated_lines: set[str]) -> list[str]:
    """Expand compact grouped model notation such as SR-2000/1000 and CV-X302/X322."""
    if "/" not in value:
        candidate = _canonical_routing_identifier(value, repeated_lines=repeated_lines)
        return [candidate] if candidate else []
    raw_parts = [part.strip().strip("()[]{}.,;") for part in value.split("/") if part.strip()]
    if not raw_parts:
        return []
    first = _canonical_routing_identifier(raw_parts[0], repeated_lines=repeated_lines)
    if first is None:
        return []
    expanded = [first]
    prefix_match = re.match(r"^(.*-)([A-Z]?\d[A-Z0-9-]*)$", first)
    prefix = prefix_match.group(1) if prefix_match else ""
    for raw_part in raw_parts[1:]:
        candidate = None
        if prefix and "-" not in raw_part and re.fullmatch(r"[A-Z]?\d[A-Z0-9-]*", raw_part.upper()):
            candidate = _canonical_routing_identifier(prefix + raw_part, repeated_lines=set())
        if candidate is None:
            candidate = _canonical_routing_identifier(raw_part, repeated_lines=set())
        if candidate:
            expanded.append(candidate)
    return _dedupe_preserve_order(expanded)


def _canonical_protocol(value: str) -> str | None:
    cleaned = re.sub(r"[™®©]", "", value)
    compact = _compact_identifier(cleaned).casefold()
    if compact.startswith("bluetooth"):
        return "bluetooth"
    if compact.startswith("usb"):
        return "usb"
    return PROTOCOL_ALIASES.get(compact)


def _deterministic_protocol_evidence(
    segments: list[MetadataSourceSegment],
) -> list[dict[str, Any]]:
    """Record exact protocol mentions without inferring compatibility or applicability."""
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment in segments:
        for raw_line in segment.text.splitlines():
            quote = " ".join(raw_line.split()).strip()
            for match in PROTOCOL_PATTERN.finditer(quote):
                protocol = _canonical_protocol(match.group(0))
                if not protocol or protocol in seen:
                    continue
                seen.add(protocol)
                evidence.append(
                    {
                        "value": protocol,
                        "kind": "protocol",
                        "relation": "mentioned",
                        "subject": None,
                        "source_quote": quote,
                        "page_from": segment.page_from,
                        "page_to": segment.page_to,
                        "section_path": list(segment.section_path),
                        "confidence": 0.45,
                        "grounded": True,
                        "source": "deterministic_protocol_mention",
                    }
                )
    return evidence


def _filename_grounded_identifier_evidence(
    filename: str,
    segments: list[MetadataSourceSegment],
) -> list[dict[str, Any]]:
    """Recover upload-identity identifiers only when the same text is grounded near the document front."""
    stem = filename.rsplit(".", 1)[0]
    candidates = _dedupe_preserve_order(
        re.findall(r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9]{0,7}(?:-[A-Za-z0-9]+)+)(?![A-Za-z0-9])", stem)
    )
    evidence: list[dict[str, Any]] = []
    for candidate in candidates:
        is_short_series_code = re.fullmatch(r"[A-Za-z]{2,4}-[A-Za-z]{1,2}", candidate) is not None
        if (
            not any(char.isdigit() for char in candidate)
            and not is_short_series_code
        ):
            continue
        candidate_compact = _compact_identifier(candidate)
        located: MetadataSourceSegment | None = None
        quote: str | None = None
        for segment in segments:
            if segment.page_from is None or segment.page_from > 3:
                continue
            for line in segment.text.splitlines():
                line_identifiers = _expand_routing_identifiers(line, repeated_lines=set())
                if candidate_compact and (
                    _identifier_is_grounded(candidate, line)
                    or candidate_compact in {_compact_identifier(item) for item in line_identifiers}
                ):
                    located = segment
                    quote = " ".join(line.split())[:500]
                    break
            if located is not None:
                break
        if located is None or quote is None:
            continue
        evidence.append(
            {
                "value": candidate.upper(),
                "kind": "part_number" if candidate.upper().startswith("OP-") else "product_model",
                "relation": "mentioned",
                "subject": None,
                "source_quote": quote,
                "page_from": located.page_from,
                "page_to": located.page_to,
                "section_path": list(located.section_path),
                "confidence": 0.45,
                "grounded": True,
                "source": "upload_identity_page_grounded",
            }
        )
    return evidence


def _plausible_company_name(value: str) -> bool:
    """Reject identifiers and prose that the model occasionally labels as companies."""
    stripped = " ".join(value.split()).strip(" |,.;:")
    if not stripped or len(stripped) > 100 or any(char.isdigit() for char in stripped):
        return False
    if _canonical_routing_identifier(stripped, repeated_lines=set()) is not None:
        return False
    words = stripped.split()
    if words[0].casefold() in {"a", "an", "the", "based", "following", "this", "these", "using"}:
        return False
    if sum(len(word.strip(".,")) == 1 for word in words) >= max(3, len(words) // 2):
        return False
    legal_markers = {
        "corp", "corporation", "company", "co", "inc", "incorporated", "ltd", "limited",
        "llc", "gmbh", "ag", "plc", "electric", "electronics", "automation", "industries",
    }
    normalized_words = {re.sub(r"[^a-z]", "", word.casefold()) for word in words}
    return bool(normalized_words & legal_markers) or (len(words) == 1 and stripped.isupper()) or len(words) >= 2


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
        and item.get("verification_status", "confirmed") == "confirmed"
        and float(item.get("confidence") or 0.0) >= PRIMARY_ENTITY_MIN_CONFIDENCE
        and not (kind in {"product_model", "part_number"} and _unsafe_routing_value(str(item.get("value") or ""), repeated_lines=repeated_lines))
    ]
    if not matches:
        return None
    if kind in {"product_model", "product_family"} and relation == "primary_product":
        matches = [
            item
            for item in matches
            if (item.get("source_method") or item.get("source")) == "upload_identity_page_grounded"
            or int(item.get("page_from") or 10**9) <= 3
        ]
        if not matches:
            return None
    if kind == "company":
        matches = [item for item in matches if _plausible_company_name(str(item.get("value") or ""))]
        if not matches:
            return None
        trusted_markers = ("copyright", "all rights reserved", "www.", "warrant")
        trusted = [
            item for item in matches
            if any(marker in str(item.get("source_quote") or "").casefold() for marker in trusted_markers)
        ]
        if trusted:
            matches = trusted
        else:
            matches = [item for item in matches if int(item.get("page_from") or 10**9) <= 3]
            if not matches:
                return None
    matches.sort(key=lambda item: (int(item.get("page_from") or 10**9), -float(item.get("confidence") or 0.0)))
    return str(matches[0]["value"])


def _values_for_routing(
    evidence: list[dict[str, Any]],
    kind: str,
    *,
    repeated_lines: set[str] | None = None,
    routing_subjects: list[str] | None = None,
) -> list[str]:
    repeated_lines = repeated_lines or set()
    routing_subject_keys = {
        _compact_identifier(value) for value in (routing_subjects or []) if value
    }
    allowed_relations = {"primary_product", "applies_to", "compatible_with", "accessory_for"}
    if kind in {"company", "device", "protocol", "software_name"}:
        allowed_relations.add("mentioned")
    if kind == "company":
        allowed_relations.add("primary_manufacturer")
    routed: list[str] = []
    for item in evidence:
        upload_identity = (
            (item.get("source_method") or item.get("source")) == "upload_identity_page_grounded"
            and kind in {"product_model", "part_number"}
            and item.get("relation") == "mentioned"
        )
        opening_part_identity = (
            (item.get("source_method") or item.get("source")) == "opening_title_candidate"
            and kind == "part_number"
            and item.get("relation") == "mentioned"
            and int(item.get("page_from") or 10**9) <= TITLE_PAGE_LIMIT
        )
        if (
            item.get("kind") != kind
            or (
                item.get("relation") not in allowed_relations
                and not upload_identity
                and not opening_part_identity
            )
            or item.get("grounded") is not True
            or item.get("verification_status", "confirmed") != "confirmed"
            or float(item.get("confidence") or 0.0) < PRIMARY_ENTITY_MIN_CONFIDENCE
        ):
            continue
        if (
            kind == "product_model"
            and not upload_identity
            and int(item.get("page_from") or 10**9) > 3
        ):
            # Models found deep in compatibility/accessory tables are useful
            # searchable metadata, but they are not safe document-scope keys.
            continue
        if (
            kind == "part_number"
            and not upload_identity
            and not opening_part_identity
            and (
                item.get("relation") != "accessory_for"
                or not str(item.get("subject") or "").strip()
                or (
                    routing_subject_keys
                    and _compact_identifier(str(item.get("subject") or ""))
                    not in routing_subject_keys
                )
            )
        ):
            # A bare identifier in a bill of materials or compatibility table
            # is not sufficient to scope the entire document. Keep it in flat
            # metadata/chunk text, but require an explicit subject-bound
            # accessory relationship before publishing a hard routing key.
            continue
        value = str(item.get("value") or "")
        if kind == "company" and not _plausible_company_name(value):
            continue
        if kind in {"product_model", "part_number"}:
            value_repeated_lines = (
                set()
                if (item.get("source_method") or item.get("source")) == "upload_identity_page_grounded"
                else repeated_lines
            )
            expanded_values = _expand_routing_identifiers(value, repeated_lines=value_repeated_lines)
            routed.extend(expanded_values)
            continue
        elif kind == "protocol":
            value = _canonical_protocol(value) or ""
        if value:
            routed.append(value)
    return _dedupe_preserve_order(routed)


def _opening_title_identity_models(
    evidence: list[dict[str, Any]],
    selected_title: str,
    *,
    repeated_lines: set[str],
) -> list[str]:
    """Promote verified cover-title identifiers without trusting a model relation label."""
    title_key = _compact_identifier(selected_title)
    if not title_key:
        return []
    upload_identity_keys = {
        _compact_identifier(value)
        for item in evidence
        if (item.get("source_method") or item.get("source")) == "upload_identity_page_grounded"
        and item.get("kind") == "product_model"
        and item.get("verification_status", "confirmed") == "confirmed"
        for value in _expand_routing_identifiers(
            str(item.get("value") or ""), repeated_lines=set()
        )
    }
    routed: list[str] = []
    for item in evidence:
        if (
            item.get("kind") != "product_model"
            or item.get("relation") != "mentioned"
            or item.get("grounded") is not True
            or item.get("verification_status", "confirmed") != "confirmed"
            or float(item.get("confidence") or 0.0) < PRIMARY_ENTITY_MIN_CONFIDENCE
            or int(item.get("page_from") or 10**9) > 2
        ):
            continue
        for value in _expand_routing_identifiers(
            str(item.get("value") or ""),
            repeated_lines=repeated_lines,
        ):
            value_key = _compact_identifier(value)
            if upload_identity_keys and value_key not in upload_identity_keys:
                continue
            if value_key and value_key in title_key:
                routed.append(value)
    return _dedupe_preserve_order(routed)


def _family_identifiers_for_routing(
    evidence: list[dict[str, Any]],
    *,
    repeated_lines: set[str],
) -> list[str]:
    routed: list[str] = []
    for item in evidence:
        if (
            item.get("kind") != "product_family"
            or item.get("relation") not in {"primary_product", "applies_to", "compatible_with"}
            or item.get("grounded") is not True
            or item.get("verification_status", "confirmed") != "confirmed"
            or float(item.get("confidence") or 0.0) < PRIMARY_ENTITY_MIN_CONFIDENCE
        ):
            continue
        if int(item.get("page_from") or 10**9) > 3:
            # A late family mention normally describes a compatible or
            # referenced product, not the document's authoritative scope.
            continue
        routed.extend(_expand_routing_identifiers(str(item.get("value") or ""), repeated_lines=repeated_lines))
    return _dedupe_preserve_order(routed)


def _applicability_records(
    evidence: list[dict[str, Any]],
    kind: str,
    *,
    applicable_subjects: list[str] | None = None,
) -> list[dict[str, Any]]:
    subject_aliases = {_compact_identifier(value) for value in (applicable_subjects or []) if value}
    def subject_is_product_identifier(item: dict[str, Any]) -> bool:
        subject = str(item.get("subject") or "")
        return (
            _compact_identifier(subject) in subject_aliases
            or _canonical_routing_identifier(subject, repeated_lines=set()) is not None
        )

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
        and item.get("relation") in {"applies_to", "compatible_with"}
        and item.get("subject")
        and (kind != "firmware_version" or subject_is_product_identifier(item))
        and item.get("grounded") is True
        and item.get("verification_status", "confirmed") == "confirmed"
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


def _opening_page_segments(segments: list[MetadataSourceSegment]) -> list[MetadataSourceSegment]:
    """Return all text blocks belonging to the first two physical pages present."""
    page_keys = sorted({segment.page_from for segment in segments if segment.page_from is not None})[
        :TITLE_PAGE_LIMIT
    ]
    if not page_keys:
        return segments[:]
    indexed = [
        (index, segment)
        for index, segment in enumerate(segments)
        if segment.page_from in page_keys
    ]
    indexed.sort(key=lambda item: (int(item[1].page_from or 10**9), item[0]))
    return [segment for _index, segment in indexed]


def _title_evidence(title: str, segments: list[MetadataSourceSegment]) -> dict[str, Any] | None:
    located = _quote_location(title, segments)
    if located is None:
        return None
    return {
        "value": title,
        "kind": "document_title",
        "relation": "printed_title",
        "subject": None,
        "source_quote": title,
        "page_from": located.page_from,
        "page_to": located.page_to,
        "section_path": list(located.section_path),
        "confidence": 0.95,
        "grounded": True,
        "source": "opening_page_title",
    }


def _select_document_title(
    filename: str,
    proposed_title: str,
    segments: list[MetadataSourceSegment],
) -> tuple[str, dict[str, Any] | None]:
    def plausible_title(value: str) -> bool:
        if "|" in value:
            # Pipe-delimited text is a parsed table row, never a publication title.
            return False
        if value.lstrip().startswith(("■", "●", "•", "・")):
            return False
        return not re.search(
            r"\b(?:download|click|tap|scan)\b.*\b(?:file|manual|image|text|details?|more)\b|"
            r"\bfor (?:a )?(?:larger|full) (?:image|text|view)\b|"
            r"\b(?:learn|read|see) more\b|"
            r"\bplease\s+read\b|\bread\b.*\bcarefully\b",
            value,
            re.IGNORECASE,
        )

    opening_segments = _opening_page_segments(segments)
    opening_text = "\n\n".join(_segment_text(segment) for segment in opening_segments)[:TITLE_SOURCE_MAX_CHARS]
    grounded_filename_identity_keys = {
        _compact_identifier(str(item.get("value") or ""))
        for item in _filename_grounded_identifier_evidence(filename, opening_segments)
    }
    descriptive_candidates: list[tuple[int, int, str]] = []
    title_kind_pattern = re.compile(
        r"\b(?:user|instruction|configuration|installation|operation|reference|service)?\s*"
        r"(?:manual|guide|datasheet|data\s+sheet|catalog|brochure|handbook|specifications?)\b",
        re.IGNORECASE,
    )
    product_type_pattern = re.compile(
        r"\b(?:adapter|amplifier|bracket|camera|controller|encoder|laser|module|reader|scanner|"
        r"sensor|system|unit)\b",
        re.IGNORECASE,
    )
    for segment in opening_segments:
        for raw_line in segment.text.splitlines():
            line = " ".join(raw_line.split()).strip(" |")
            has_identifier = IDENTIFIER_CANDIDATE_PATTERN.search(line) is not None
            line_identifier_keys = {
                _compact_identifier(match.group(0))
                for match in IDENTIFIER_CANDIDATE_PATTERN.finditer(line)
            }
            if (
                grounded_filename_identity_keys
                and line_identifier_keys
                and not (grounded_filename_identity_keys & line_identifier_keys)
                and not title_kind_pattern.search(line)
            ):
                # Covers often advertise a compatible accessory in a small
                # callout. Do not let that competing identifier outrank the
                # filename-grounded primary identity printed on the same page.
                continue
            has_kind = title_kind_pattern.search(line) is not None
            has_series_identity = has_identifier and re.search(r"\bseries\b", line, re.IGNORECASE)
            page_one_identifier_heading = (
                int(segment.page_from or 10**9) == 1
                and has_identifier
                and "|" not in raw_line
                and 2 <= len(line.split()) <= 12
            )
            if not (12 <= len(line) <= 180) or not (
                has_kind or has_series_identity or page_one_identifier_heading
            ):
                continue
            if re.search(r"\b(?:copyright|all rights reserved|https?://|www\.)\b", line, re.IGNORECASE):
                continue
            if not plausible_title(line):
                continue
            words = line.split()
            if not 2 <= len(words) <= 18:
                continue
            match = title_kind_pattern.search(line)
            score = 4 if match and match.end() == len(line) else 0
            if has_identifier:
                score += 3
            if has_series_identity:
                score += 2
            if page_one_identifier_heading:
                score += 2
            score += max(0, 3 - int(segment.page_from or 3))
            descriptive_candidates.append((score, -len(line), line))
    if descriptive_candidates:
        descriptive_candidates.sort(reverse=True)
        deterministic_title = descriptive_candidates[0][2]
        return deterministic_title, _title_evidence(deterministic_title, opening_segments)
    proposed = " ".join(proposed_title.split()).strip()
    proposed_has_identifier = IDENTIFIER_CANDIDATE_PATTERN.search(proposed) is not None
    proposed_identifier_keys = {
        _compact_identifier(match.group(0))
        for match in IDENTIFIER_CANDIDATE_PATTERN.finditer(proposed)
    }
    proposed_identity_conflict = bool(
        grounded_filename_identity_keys
        and proposed_identifier_keys
        and not (grounded_filename_identity_keys & proposed_identifier_keys)
        and not title_kind_pattern.search(proposed)
    )
    proposed_has_kind = title_kind_pattern.search(proposed) is not None
    proposed_has_series_identity = proposed_has_identifier and re.search(
        r"\bseries\b", proposed, re.IGNORECASE
    )
    if (
        proposed
        and not proposed_identity_conflict
        and _value_is_grounded(proposed, opening_text)
        and plausible_title(proposed)
        and (
            proposed_has_identifier
            or proposed_has_kind
            or proposed_has_series_identity
            or (grounded_filename_identity_keys and product_type_pattern.search(proposed))
        )
    ):
        return proposed, _title_evidence(proposed, opening_segments)
    printed_title = _extract_printed_title(opening_text) if opening_text else None
    if printed_title:
        printed_identifier_keys = {
            _compact_identifier(match.group(0))
            for match in IDENTIFIER_CANDIDATE_PATTERN.finditer(printed_title)
        }
        printed_identity_conflict = bool(
            grounded_filename_identity_keys
            and printed_identifier_keys
            and not (grounded_filename_identity_keys & printed_identifier_keys)
            and not title_kind_pattern.search(printed_title)
        )
        if not printed_identity_conflict and plausible_title(printed_title) and (
            IDENTIFIER_CANDIDATE_PATTERN.search(printed_title)
            or title_kind_pattern.search(printed_title)
            or (len(printed_title) >= 20 and len(printed_title.split()) >= 4)
        ):
            return printed_title, _title_evidence(printed_title, opening_segments)
    return _normalize_title(filename), None


def _demote_competing_cover_callouts(
    evidence: list[dict[str, Any]], upload_identity: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Demote a competing short-series cover callout beside the upload identity."""
    identity_keys = {
        _compact_identifier(str(item.get("value") or ""))
        for item in upload_identity
        if re.fullmatch(r"[A-Za-z]{2,4}[-:][A-Za-z]{1,2}", str(item.get("value") or ""))
    }
    if not identity_keys:
        return evidence
    adjusted: list[dict[str, Any]] = []
    for raw_item in evidence:
        item = dict(raw_item)
        value = str(item.get("value") or "")
        if (
            item.get("kind") in {"product_model", "product_family"}
            and item.get("relation") == "primary_product"
            and int(item.get("page_from") or 10**9) <= TITLE_PAGE_LIMIT
            and re.fullmatch(r"[A-Za-z]{2,4}[-:][A-Za-z]{1,2}", value)
            and _compact_identifier(value) not in identity_keys
        ):
            item["relation"] = "mentioned"
        adjusted.append(item)
    return adjusted


def _materialize_verified_metadata(
    base: DocumentMetadata,
    selected_title: str,
    printed_title_evidence: dict[str, Any] | None,
    claims: list[dict[str, Any]],
    segments: list[MetadataSourceSegment],
) -> DocumentMetadata:
    scoped_evidence = claims
    evidence = _dedupe_evidence(
        scoped_evidence
        + _base_metadata_evidence(base, segments)
        + ([printed_title_evidence] if printed_title_evidence else [])
    )
    repeated_lines = _repeated_short_line_values(segments)

    verified_product_models = _values_for_routing(scoped_evidence, "product_model", repeated_lines=repeated_lines)
    opening_title_models = _opening_title_identity_models(
        scoped_evidence,
        selected_title,
        repeated_lines=repeated_lines,
    )
    verified_product_models = _dedupe_preserve_order(verified_product_models + opening_title_models)
    verified_family_identifiers = _family_identifiers_for_routing(
        scoped_evidence,
        repeated_lines=repeated_lines,
    )
    verified_part_numbers = _values_for_routing(
        scoped_evidence,
        "part_number",
        repeated_lines=repeated_lines,
        routing_subjects=verified_product_models + verified_family_identifiers,
    )
    verified_protocols = [value.lower() for value in _values_for_routing(scoped_evidence, "protocol")]
    safe_base_product_models = [
        value
        for value in base.product_models
        if not _unsafe_routing_value(value, repeated_lines=repeated_lines)
    ]
    product_models = _dedupe_preserve_order(safe_base_product_models + verified_product_models)
    product_families = _dedupe_preserve_order(_values_for_routing(scoped_evidence, "product_family"))
    part_numbers = _dedupe_preserve_order(base.part_numbers + verified_part_numbers)
    devices = _dedupe_preserve_order(base.devices + _values_for_routing(scoped_evidence, "device"))
    protocols = _dedupe_preserve_order(base.protocol_terms + verified_protocols)
    # Schema-v2 routing is evidence-gated. Legacy flat values remain searchable metadata,
    # but cannot become hard-routing keys without scoped, high-confidence evidence.
    routing_product_models = _dedupe_preserve_order(verified_product_models + verified_family_identifiers)
    routing_part_numbers = verified_part_numbers
    routing_protocols = verified_protocols
    identifiers = routing_product_models + routing_part_numbers + routing_protocols
    normalized_aliases = _dedupe_preserve_order(
        [alias for value in identifiers for alias in (value, _compact_identifier(value)) if alias]
    )

    primary_manufacturer = _first_primary(scoped_evidence, "company", "primary_manufacturer") or _first_primary(
        scoped_evidence,
        "company",
        "mentioned",
    )
    primary_product = _first_primary(
        scoped_evidence,
        "product_model",
        "primary_product",
        repeated_lines=repeated_lines,
    )
    primary_family = _first_primary(
        scoped_evidence,
        "product_family",
        "primary_product",
        repeated_lines=repeated_lines,
    )
    upload_identity_models = _dedupe_preserve_order(
        [
            value
            for item in scoped_evidence
            if (item.get("source_method") or item.get("source")) == "upload_identity_page_grounded"
            and item.get("kind") == "product_model"
            and item.get("verification_status", "confirmed") == "confirmed"
            for value in _expand_routing_identifiers(
                str(item.get("value") or ""), repeated_lines=set()
            )
        ]
    )
    title_parts = [value for value in verified_part_numbers if _identifier_is_grounded(value, selected_title)]
    selected_product = (
        (title_parts[0] if len(title_parts) == 1 else None)
        or (opening_title_models[0] if len(opening_title_models) == 1 else None)
        or primary_product
        or (opening_title_models[0] if opening_title_models else None)
        or (upload_identity_models[0] if upload_identity_models else None)
    )
    return replace(
        base,
        title=selected_title,
        manufacturer=primary_manufacturer or "Unknown",
        companies=_dedupe_preserve_order(_values_for_routing(scoped_evidence, "company")),
        product_model=selected_product,
        product_models=product_models,
        product_family=primary_family,
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
        firmware_applicability=_applicability_records(
            scoped_evidence,
            "firmware_version",
            applicable_subjects=routing_product_models + product_families,
        ),
        software_applicability=_applicability_records(scoped_evidence, "software_version"),
        metadata_claims=scoped_evidence,
        metadata_pipeline_version=METADATA_PIPELINE_VERSION,
    )


class MetadataWorkflowState(TypedDict, total=False):
    filename: str
    segments: list[MetadataSourceSegment]
    max_segment_chars: int
    batches: list[list[MetadataSourceSegment]]
    batch: list[MetadataSourceSegment]
    base: DocumentMetadata
    selected_title: str
    printed_title_evidence: dict[str, Any] | None
    mapped_evidence: Annotated[list[dict[str, Any]], operator.add]
    claims: list[dict[str, Any]]
    verified_claims: list[dict[str, Any]]
    result: DocumentMetadata


def _prepare_metadata_workflow(state: MetadataWorkflowState) -> dict[str, Any]:
    indexed_segments = [
        (index, segment)
        for index, segment in enumerate(state["segments"])
        if segment.text.strip()
    ]
    indexed_segments.sort(
        key=lambda item: (
            int(item[1].page_from or 10**9),
            int(item[1].page_to or item[1].page_from or 10**9),
            item[0],
        )
    )
    segments = [segment for _index, segment in indexed_segments]
    opening_segments = _opening_page_segments(segments)
    front_text = "\n\n".join(_segment_text(segment) for segment in opening_segments)[
        :DEFAULT_METADATA_SEGMENT_CHARS
    ]
    base = infer_document_metadata(state["filename"], front_text)
    selected_title, printed_title_evidence = _select_document_title(
        state["filename"], base.title, segments
    )
    return {
        "segments": segments,
        "base": base,
        "selected_title": selected_title,
        "printed_title_evidence": printed_title_evidence,
        "batches": pack_metadata_source_segments(
            segments, max_chars=state["max_segment_chars"]
        ),
        "mapped_evidence": [],
    }


def _dispatch_metadata_batches(state: MetadataWorkflowState) -> list[Send]:
    return [
        Send("map_metadata_batch", {"filename": state["filename"], "batch": batch})
        for batch in state["batches"]
    ]


def _map_metadata_batch(state: MetadataWorkflowState) -> dict[str, Any]:
    return {
        "mapped_evidence": _extract_scoped_metadata(state["filename"], state["batch"])
    }


def _model_column_identifiers(segment: MetadataSourceSegment) -> list[str]:
    lines = [line.strip() for line in segment.text.splitlines() if line.strip()]
    if not lines or not re.match(r"^model(?:\s+name)?\s*\|", lines[0], re.I):
        return []
    # Wide specification tables put models in the header; catalog tables put
    # them in column one. Compatible-model columns must not become primary IDs.
    header_models = [m.group(0) for m in IDENTIFIER_CANDIDATE_PATTERN.finditer(lines[0])]
    if header_models and "compatible" not in lines[0].lower():
        return _dedupe_preserve_order(header_models)
    return _dedupe_preserve_order([
        identifier for line in lines[1:]
        if (identifier := _canonical_routing_identifier(line.split("|", 1)[0].strip(), repeated_lines=set()))
    ])


def _compatible_model_column_claims(
    segments: list[MetadataSourceSegment],
) -> list[dict[str, Any]]:
    """Recover explicit catalog compatibility rows without promoting them to primary IDs."""
    evidence: list[dict[str, Any]] = []
    for segment in segments:
        lines = [line.strip() for line in segment.text.splitlines() if line.strip()]
        if not lines or "|" not in lines[0]:
            continue
        headers = [cell.strip().casefold() for cell in lines[0].split("|")]
        compatible_indexes = [
            index
            for index, header in enumerate(headers)
            if "compatible" in header and "model" in header
        ]
        if not compatible_indexes:
            continue
        compatible_index = compatible_indexes[0]
        active_subject: str | None = None
        table_rows = [lines[0]]
        for line in lines[1:]:
            table_rows.append(line)
            cells = [cell.strip() for cell in line.split("|")]
            if cells and cells[0]:
                active_subject = _canonical_routing_identifier(
                    cells[0], repeated_lines=set()
                )
            if not active_subject or compatible_index >= len(cells):
                continue
            compatible_values = _expand_routing_identifiers(
                cells[compatible_index], repeated_lines=set()
            )
            if not compatible_values:
                continue
            # Continuation rows inherit the nearest preceding subject. Preserve
            # the complete contiguous table prefix instead of synthesizing a
            # quote from the header, subject row, and later row while skipping
            # intervening source text.
            quote = "\n".join(table_rows)
            for value in compatible_values:
                evidence.append(
                    {
                        "value": value,
                        "kind": "product_model",
                        "relation": "compatible_with",
                        "subject": active_subject,
                        "source_quote": quote,
                        "page_from": segment.page_from,
                        "page_to": segment.page_to,
                        "section_path": list(segment.section_path),
                        "source_method": "deterministic_compatible_model_column",
                        "confidence": 0.8,
                        "grounded": True,
                        "support_pages": [segment.page_from]
                        if segment.page_from is not None
                        else [],
                    }
                )
    return _dedupe_evidence(evidence)


def _verified_model_identifiers(claims: list[dict[str, Any]]) -> set[str]:
    """Return every source model identifier retained by verified relations.

    Catalog compatibility rows encode the catalog model as the relationship
    subject and the compatible model as its value.  Both sides are explicit
    source identifiers, even though only the value is eligible for the usual
    value-based completeness scan.
    """
    found: set[str] = set()
    for item in claims:
        if item.get("verification_status") != "confirmed" or item.get("grounded") is not True:
            continue
        if item.get("kind") in {"product_model", "part_number"}:
            found.add(_compact_identifier(str(item.get("value") or "")))
        if item.get("relation") == "compatible_with" and item.get("subject"):
            found.add(_compact_identifier(str(item["subject"])))
    return {value for value in found if value}


def _verified_upload_identity_identifiers(claims: list[dict[str, Any]]) -> set[str]:
    """Return verified upload identities or stronger equivalent relationships."""
    return {
        _compact_identifier(str(item.get("value") or ""))
        for item in claims
        if item.get("verification_status") == "confirmed"
        and item.get("grounded") is True
        and item.get("kind") in {"product_model", "part_number"}
        and (
            (item.get("source_method") or item.get("source")) in {
                "upload_identity_page_grounded", "opening_title_candidate",
            }
            or item.get("relation") in {"primary_product", "applies_to"}
        )
        and _compact_identifier(str(item.get("value") or ""))
    }


def _focused_model_column_claims(filename, segments):
    evidence = []
    for segment in segments:
        targets = _model_column_identifiers(segment)
        for start in range(0, len(targets), MAX_SCOPED_ENTITIES):
            group = targets[start:start + MAX_SCOPED_ENTITIES]
            messages = _scoped_prompt_messages(filename, _segment_text(segment))
            messages[1]["content"] += (
                "\nFocus only on these model-column identifiers: " + json.dumps(group)
                + ". Return one entity per target with its source-supported relationship. "
                "Do not extract compatible-model-column entries in this pass."
            )
            extraction = _call_scoped_model(filename, messages, purpose="metadata_extraction.scoped_entities")
            evidence.extend(_ground_scoped_candidates(extraction, [segment]))
    return evidence


def _reduce_metadata_claims(state: MetadataWorkflowState) -> dict[str, Any]:
    mapped = sorted(
        state.get("mapped_evidence", []),
        key=lambda item: (
            int(item.get("page_from") or 10**9),
            str(item.get("kind") or ""),
            _compact_identifier(str(item.get("value") or "")),
            str(item.get("relation") or ""),
        ),
    )
    upload_identity = _filename_grounded_identifier_evidence(state["filename"], state["segments"])
    evidence = _dedupe_evidence(
        mapped
        + _focused_model_column_claims(state["filename"], state["segments"])
        + _compatible_model_column_claims(state["segments"])
        + _opening_title_identifier_evidence(state["selected_title"], state["segments"])
        + upload_identity
        + _deterministic_protocol_evidence(state["segments"])
    )
    evidence = _demote_competing_cover_callouts(evidence, upload_identity)
    return {"claims": reconcile_metadata_claims(evidence)}


def _verify_metadata_workflow_claims(state: MetadataWorkflowState) -> dict[str, Any]:
    verified_claims = verify_metadata_claims(
        state["filename"], state["claims"], state["segments"]
    )
    expected_version_kinds = _expected_version_kinds(state["segments"])
    confirmed_version_kinds = {
        str(item.get("kind") or "")
        for item in verified_claims
        if item.get("verification_status") == "confirmed"
        and item.get("grounded") is True
    }
    missing_version_kinds = expected_version_kinds - confirmed_version_kinds
    if missing_version_kinds:
        raise MetadataExtractionIncomplete(
            f"Independent verification for {state['filename']} rejected or could not resolve "
            f"all grounded {sorted(missing_version_kinds)} claims"
        )
    missing_values = _missing_explicit_software_versions(state["segments"], [
        item for item in verified_claims if item.get("verification_status") == "confirmed"
    ])
    if missing_values:
        raise MetadataExtractionIncomplete(f"Verification lost explicit software versions: {sorted(missing_values)}")
    expected_models = {
        _compact_identifier(value)
        for segment in state["segments"]
        for value in _model_column_identifiers(segment)
    }
    expected_models.update(
        _compact_identifier(str(item["value"]))
        for item in _compatible_model_column_claims(state["segments"])
    )
    found_models = _verified_model_identifiers(verified_claims)
    if expected_models - found_models:
        raise MetadataExtractionIncomplete(f"Verification lost model-column identifiers: {sorted(expected_models - found_models)}")
    expected_upload_identity = {
        _compact_identifier(str(item.get("value") or ""))
        for item in _filename_grounded_identifier_evidence(state["filename"], state["segments"])
    }
    verified_upload_identity = _verified_upload_identity_identifiers(verified_claims)
    if expected_upload_identity - verified_upload_identity:
        raise MetadataExtractionIncomplete(
            "Verification lost grounded upload-identity identifiers: "
            f"{sorted(expected_upload_identity - verified_upload_identity)}"
        )
    return {"verified_claims": verified_claims}


def _publish_metadata_workflow(state: MetadataWorkflowState) -> dict[str, Any]:
    return {
        "result": _materialize_verified_metadata(
            state["base"],
            state["selected_title"],
            state.get("printed_title_evidence"),
            state["verified_claims"],
            state["segments"],
        )
    }


def build_metadata_extraction_graph() -> Any:
    builder = StateGraph(MetadataWorkflowState)
    builder.add_node("prepare", _prepare_metadata_workflow)
    builder.add_node("map_metadata_batch", _map_metadata_batch)
    builder.add_node("reduce_claims", _reduce_metadata_claims)
    builder.add_node("verify_claims", _verify_metadata_workflow_claims)
    builder.add_node("publish", _publish_metadata_workflow)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _dispatch_metadata_batches, ["map_metadata_batch"])
    builder.add_edge("map_metadata_batch", "reduce_claims")
    builder.add_edge("reduce_claims", "verify_claims")
    builder.add_edge("verify_claims", "publish")
    builder.add_edge("publish", END)
    return builder.compile()


_METADATA_EXTRACTION_GRAPH: Any | None = None


def infer_document_metadata_from_segments(
    filename: str,
    segments: list[MetadataSourceSegment],
    *,
    max_segment_chars: int = DEFAULT_METADATA_SEGMENT_CHARS,
) -> DocumentMetadata:
    """Run evidence-first Map–Reduce–Verify extraction over a whole document."""
    nonempty = [segment for segment in segments if segment.text.strip()]
    if not nonempty:
        raise MetadataExtractionIncomplete("No source text is available; reparse/OCR before metadata extraction")
    readable_words = sum(len(re.findall(r"[^\W\d_]{2,}", segment.text)) for segment in nonempty)
    if readable_words == 0 or (len(nonempty) >= 10 and readable_words < 5):
        raise MetadataExtractionIncomplete("Source text contains no readable words; reparse/OCR before metadata extraction")
    global _METADATA_EXTRACTION_GRAPH
    if _METADATA_EXTRACTION_GRAPH is None:
        _METADATA_EXTRACTION_GRAPH = build_metadata_extraction_graph()
    output = _METADATA_EXTRACTION_GRAPH.invoke(
        {
            "filename": filename,
            "segments": nonempty,
            "max_segment_chars": max_segment_chars,
            "mapped_evidence": [],
        },
        config={"max_concurrency": METADATA_MAP_MAX_CONCURRENCY},
    )
    return output["result"]
