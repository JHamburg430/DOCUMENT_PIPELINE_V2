import json

import pytest

from manuals_rag_parsers.metadata import (
    LIST_FIELD_INSTRUCTIONS,
    METADATA_NUM_CTX,
    MetadataExtractionIncomplete,
    MetadataExtraction,
    MetadataSourceSegment,
    ScopedMetadataExtraction,
    _canonical_protocol,
    _canonical_routing_identifier,
    _bisect_metadata_segments,
    _call_scoped_model,
    _deterministic_version_evidence,
    _expand_routing_identifiers,
    _expected_version_kinds,
    _family_identifiers_for_routing,
    _ground_scoped_candidates,
    _plausible_company_name,
    _values_for_routing,
    infer_document_metadata,
    infer_document_metadata_from_segments,
    pack_metadata_source_segments,
)
from manuals_rag_common.config import settings


def test_infer_document_metadata_from_model_response(monkeypatch):
    def fake_chat_json(**kwargs):
        assert kwargs["model"] == settings.ollama_metadata_model
        assert kwargs["purpose"].startswith("metadata_extraction")
        assert "properties" in kwargs["json_schema"]
        assert kwargs["num_ctx"] == METADATA_NUM_CTX
        return (
            {
                "manufacturer": "Keyence",
                "companies": ["KEYENCE AMERICA"],
                "product_model": "CA-EN100U",
                "product_models": ["CA-EN100U"],
                "product_family": "CA",
                "product_families": ["CA", "CA-EN"],
                "devices": ["Encoder relay unit"],
                "document_kind": "datasheet",
                "title": "CA-EN100U Datasheet",
                "revision_date": "2026-01-12",
                "effective_date": "2026-01-12",
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)

    metadata = infer_document_metadata(
        "CA-EN100U_Datasheet.pdf",
        "KEYENCE AMERICA\nCA-EN100U\nEncoder relay unit\n2026/01/12",
    )

    assert metadata.manufacturer == "KEYENCE AMERICA"
    assert metadata.companies == ["KEYENCE AMERICA"]
    assert metadata.product_model == "CA-EN100U"
    assert metadata.product_family == "CA"
    assert "CA-EN100U" in metadata.product_models
    assert "CA-EN" in metadata.product_families
    assert metadata.devices == ["Encoder relay unit"]
    assert metadata.document_kind.value == "datasheet"
    assert metadata.revision_date is not None


def test_infer_document_metadata_extracts_filter_terms(monkeypatch):
    def fake_chat_json(**kwargs):
        return (
            {
                "manufacturer": "Keyence",
                "companies": ["Keyence"],
                "product_model": "LJ-X8000",
                "product_models": ["LJ-X8000", "LJ-X8080", "LJ-X8060"],
                "product_families": ["LJ", "LJ-X"],
                "devices": ["laser profiler"],
                "part_numbers": ["OP-88310"],
                "protocol_terms": ["EtherNet/IP"],
                "settings": ["Sensor setup"],
                "parameters": ["communication parameter 1"],
                "menu_labels": ["[Sensor setup]"],
                "document_topics": ["configuration", "installation", "safety", "communications"],
                "document_kind": "manual",
                "title": "LJ-X8000 Manual",
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)

    metadata = infer_document_metadata(
        "AS_151019_LJ-X8000_C_689092_KA_US_2055_2.pdf",
        "KEYENCE LJ-X8080 LJ-X8060 setup over EtherNet/IP communications. Select [Sensor setup]. OP-88310 wiring caution safety. Configure communication parameter 1.",
    )

    assert metadata.product_model == "LJ-X8000"
    assert {"LJ-X8000", "LJ-X8080", "LJ-X8060"}.issubset(set(metadata.product_models))
    assert "LJ-X" in metadata.product_families
    assert "ethernet/ip" in metadata.protocol_terms
    assert "Sensor setup" in metadata.settings
    assert "communication parameter 1" in metadata.parameters
    assert "[Sensor setup]" in metadata.menu_labels
    assert "OP-88310" in metadata.part_numbers
    assert {"safety", "communications"}.issubset(set(metadata.document_topics))
    assert "configuration" not in metadata.document_topics


def test_infer_document_metadata_falls_back_on_invalid_model_response(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"document_kind": "not_a_kind"}, "{}"),
    )

    metadata = infer_document_metadata("bad.pdf", "text")
    assert metadata.document_kind.value == "manual"
    assert metadata.title == "bad"


def test_metadata_prompt_examples_are_vendor_neutral():
    instruction_text = "\n".join(LIST_FIELD_INSTRUCTIONS.values()).lower()
    assert "keyence" not in instruction_text
    assert "lj-x" not in instruction_text
    assert "ca-en" not in instruction_text
    assert "op-88310" not in instruction_text


def test_page_aware_metadata_preserves_scope_aliases_and_late_evidence(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(
            manufacturer="Intel",
            companies=["Intel", "KEYENCE"],
            product_model="CV-X482",
            product_models=["CV-X482"],
            document_kind="manual",
            title="CV-X Manual",
        ),
    )

    def fake_chat_json(**kwargs):
        assert kwargs["purpose"] == "metadata_extraction.scoped_entities"
        source = kwargs["messages"][1]["content"]
        entities = []
        if "CV-X482 vision controller" in source:
            entities.extend(
                [
                    {
                        "value": "KEYENCE",
                        "kind": "company",
                        "relation": "primary_manufacturer",
                        "source_quote": "KEYENCE CV-X482 vision controller",
                        "confidence": 0.99,
                    },
                    {
                        "value": "CV-X482",
                        "kind": "product_model",
                        "relation": "primary_product",
                        "source_quote": "KEYENCE CV-X482 vision controller",
                        "confidence": 0.99,
                    },
                ]
            )
        if "OP-42284" in source:
            entities.append(
                {
                    "value": "OP-42284",
                    "kind": "part_number",
                    "relation": "accessory_for",
                    "subject": "CV-X482",
                    "source_quote": "Use cable OP-42284 with CV-X482.",
                    "confidence": 0.95,
                }
            )
        if "firmware 6.0" in source:
            entities.append(
                {
                    "value": "6.0",
                    "kind": "firmware_version",
                    "relation": "applies_to",
                    "subject": "CV-X482",
                    "source_quote": "CV-X482 firmware 6.0 or later is required.",
                    "confidence": 0.98,
                }
            )
        if "KV-7500 firmware 2.1" in source:
            entities.extend(
                [
                    {
                        "value": "KV-7500",
                        "kind": "device",
                        "relation": "external_reference",
                        "source_quote": "The external KV-7500 firmware 2.1 example uses the PLC.",
                        "confidence": 0.98,
                    },
                    {
                        "value": "2.1",
                        "kind": "firmware_version",
                        "relation": "applies_to",
                        "subject": "KV-7500",
                        "source_quote": "The external KV-7500 firmware 2.1 example uses the PLC.",
                        "confidence": 0.98,
                    },
                ]
            )
        return ({"entities": entities}, "{}")

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "cvx_manual.pdf",
        [
            MetadataSourceSegment("Intel Ethernet adapter example. KEYENCE CV-X482 vision controller", 1, 1, ("Introduction",)),
            MetadataSourceSegment("Use cable OP-42284 with CV-X482.", 80, 80, ("Accessories",)),
            MetadataSourceSegment("CV-X482 firmware 6.0 or later is required.", 120, 120, ("Compatibility",)),
            MetadataSourceSegment("The external KV-7500 firmware 2.1 example uses the PLC.", 121, 121, ("PLC example",)),
        ],
        max_segment_chars=100,
    )

    assert metadata.metadata_schema_version == 2
    assert metadata.manufacturer == "KEYENCE"
    assert metadata.product_model == "CV-X482"
    assert metadata.routing_product_models == ["CV-X482"]
    assert metadata.routing_part_numbers == ["OP-42284"]
    assert "CVX482" in metadata.normalized_identifier_aliases
    assert "OP42284" in metadata.normalized_identifier_aliases
    assert metadata.firmware_applicability[0]["subject"] == "CV-X482"
    assert all(item["version"] != "2.1" for item in metadata.firmware_applicability)
    assert any(item["page_from"] == 80 for item in metadata.metadata_evidence)


def test_metadata_segment_packing_covers_the_full_document():
    segments = [MetadataSourceSegment(f"page {page} " + "x" * 30, page, page) for page in range(1, 8)]
    batches = pack_metadata_source_segments(segments, max_chars=80)
    pages = [segment.page_from for batch in batches for segment in batch]
    assert pages == list(range(1, 8))


def test_metadata_bisection_splits_dense_single_segment_below_legacy_threshold():
    segment = MetadataSourceSegment("field | value\n" * 80, 7, 7, ("Specifications",))
    split = _bisect_metadata_segments([segment])
    assert split is not None
    left, right = split
    assert left[0].text + right[0].text == segment.text
    assert left[0].page_from == right[0].page_from == 7


def test_routing_canonicalization_rejects_document_codes_and_cleans_table_values():
    assert _canonical_routing_identifier("Model | CA-DZW10X", repeated_lines=set()) == "CA-DZW10X"
    assert _canonical_routing_identifier("CA-CF5E (5 m 16.4')", repeated_lines=set()) == "CA-CF5E"
    assert _canonical_routing_identifier("KV-X", repeated_lines=set()) == "KV-X"
    assert _canonical_routing_identifier("CV-X", repeated_lines=set()) == "CV-X"
    assert _canonical_routing_identifier("VS", repeated_lines=set()) is None
    assert _canonical_routing_identifier("LRT-KA-C2-US 2074-3 611C20", repeated_lines=set()) is None
    assert _canonical_routing_identifier("2074-3", repeated_lines=set()) is None
    assert _canonical_protocol("EtherNet/IP™") == "ethernet/ip"
    assert _canonical_protocol("945 nm") is None
    assert _expand_routing_identifiers("SR-2000/1000", repeated_lines=set()) == ["SR-2000", "SR-1000"]
    assert _expand_routing_identifiers("CV-X302/X322/X352", repeated_lines=set()) == [
        "CV-X302", "CV-X322", "CV-X352"
    ]
    assert _plausible_company_name("KEYENCE CORPORATION")
    assert not _plausible_company_name("CA-S20D")
    assert not _plausible_company_name("Based on KEYENCE's unique algorithm")
    assert not _plausible_company_name("C A L L T O L L F R E E")


def test_late_primary_product_claim_is_not_a_hard_routing_key():
    evidence = [{
        "value": "CA-S20D",
        "kind": "product_model",
        "relation": "primary_product",
        "page_from": 19,
        "confidence": 0.99,
        "grounded": True,
        "source": "page_aware_model_extraction",
    }]
    assert _values_for_routing(evidence, "product_model", repeated_lines=set()) == []


def test_identifier_like_primary_family_can_be_an_exact_routing_key():
    evidence = [{
        "value": "SV2 Series", "kind": "product_family", "relation": "primary_product",
        "page_from": 1, "confidence": 0.95, "grounded": True,
    }]
    assert _family_identifiers_for_routing(evidence, repeated_lines=set()) == ["SV2"]


def test_version_signal_does_not_cross_line_boundaries():
    segments = [MetadataSourceSegment("STUDIO mode, Modbus master/slave\nCC-Link Ver. 2.0", 1, 1)]
    assert _expected_version_kinds(segments) == set()


def test_explicit_software_version_fallback_is_grounded_but_not_applicability():
    segments = [MetadataSourceSegment(
        "Graphic software | Downloadable | VT STUDIO Ver.8 (US edition)", 66, 66, ("Software",)
    )]
    evidence = _deterministic_version_evidence(segments, {"software_version"})
    assert [(item["subject"], item["value"], item["relation"]) for item in evidence] == [
        ("VT STUDIO", "8", "mentioned")
    ]
    assert evidence[0]["page_from"] == 66


def test_scoped_grounding_rejects_ungrounded_relationship_and_dimension_as_firmware():
    extraction = ScopedMetadataExtraction.model_validate({"entities": [
        {
            "value": "CA-S20D", "kind": "product_model", "relation": "compatible_with",
            "subject": "product", "source_quote": "CA-S20D", "confidence": 0.99,
        },
        {
            "value": "2.95", "kind": "firmware_version", "relation": "compatible_with",
            "subject": "CA-EN100U", "source_quote": '2.95"', "confidence": 0.99,
        },
    ]})
    grounded = _ground_scoped_candidates(
        extraction,
        [MetadataSourceSegment('CA-S20D\nCA-EN100U dimension: 2.95"', 7, 7)],
    )
    assert grounded == []


def test_filename_identifier_requires_matching_front_page_evidence(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="Command Manual"),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    metadata = infer_document_metadata_from_segments(
        "AS_124150_LJ-X8000_CM_C10GB_WW_GB_2101_1.pdf",
        [MetadataSourceSegment("LJ-X8000 Communication Command Manual", 1, 1)],
    )

    assert metadata.routing_product_models == ["LJ-X8000"]
    assert metadata.product_model == "LJ-X8000"
    assert any(item["source"] == "upload_identity_page_grounded" for item in metadata.metadata_evidence)


def test_scalar_metadata_normalizes_array_shape_dates_and_kind_synonyms(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ([{"title": "Release Notes", "document_kind": "release_notes", "revision_date": "2025/04/22", "effective_date": "null"}], "[]"),
    )

    metadata = infer_document_metadata("release.pdf", "Release Notes\n2025/04/22")

    assert metadata.document_kind.value == "release_note"
    assert metadata.revision_date.isoformat() == "2025-04-22"
    assert metadata.effective_date == metadata.revision_date


@pytest.mark.parametrize(
    ("raw_kind", "expected_kind"),
    [
        ("product_specification", "spec_sheet"),
        ("product_brochure", "brochure"),
        ("technical installation instructions", "installation_guide"),
        ("catalog", "parts_catalog"),
    ],
)
def test_scalar_metadata_normalizes_product_kind_variants(monkeypatch, raw_kind, expected_kind):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"title": "Product Document", "document_kind": raw_kind}, "{}"),
    )

    metadata = infer_document_metadata("specifications.pdf", "Product Specifications")

    assert metadata.document_kind.value == expected_kind


def test_scoped_metadata_retries_and_accepts_top_level_array_and_entity_alias(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="CV-X Manual"),
    )
    calls = 0

    def fake_chat_json(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ("truncated", "truncated")
        return (
            [
                {
                    "entity_label": "CV-X482",
                    "entity_type": "product_model",
                    "relationship": "primary_product",
                    "verbatim_evidence_text": "CV-X482 vision controller",
                    "score": 97,
                }
            ],
            "[]",
        )

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "cvx.pdf",
        [MetadataSourceSegment("CV-X482 vision controller", 1, 1)],
    )

    assert calls == 2
    assert metadata.routing_product_models == ["CV-X482"]
    assert metadata.product_model == "CV-X482"
    assert metadata.metadata_evidence[0]["confidence"] == pytest.approx(0.97)


def test_scoped_metadata_splits_immediately_after_deterministic_json_error(monkeypatch):
    calls = 0

    def malformed_chat_json(**kwargs):
        nonlocal calls
        calls += 1
        raise json.JSONDecodeError("unterminated", '"', 0)

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", malformed_chat_json)
    with pytest.raises(MetadataExtractionIncomplete):
        _call_scoped_model("manual.pdf", [{"role": "user", "content": "text"}], purpose="test")
    assert calls == 1


def test_version_bearing_batch_gets_focused_completeness_pass(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="Compatibility"),
    )
    purposes = []

    def fake_chat_json(**kwargs):
        purposes.append(kwargs["purpose"])
        if kwargs["purpose"] == "metadata_extraction.scoped_entities":
            return ({"entities": []}, "{}")
        return (
            {
                "entities": [
                    {
                        "value": "3.14",
                        "kind": "firmware_version",
                        "relation": "applies_to",
                        "subject": "LJ-V7001",
                        "source_quote": "LJ-V7001 firmware version 3.14 or later",
                        "confidence": 0.96,
                    },
                    {
                        "value": "3.1",
                        "kind": "software_version",
                        "relation": "external_reference",
                        "subject": "TwinCAT",
                        "source_quote": "TwinCAT software version 3.1 is required",
                        "confidence": 0.94,
                    },
                ]
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "compatibility.pdf",
        [
            MetadataSourceSegment(
                "LJ-V7001 firmware version 3.14 or later. TwinCAT software version 3.1 is required.",
                44,
                44,
            )
        ],
    )

    assert "metadata_extraction.version_applicability" in purposes
    assert metadata.firmware_applicability[0]["subject"] == "LJ-V7001"
    assert metadata.software_applicability == []
    assert any(
        item["kind"] == "software_version" and item["relation"] == "external_reference"
        for item in metadata.metadata_evidence
    )


def test_unresolved_critical_version_batch_fails_instead_of_silently_degrading(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="Compatibility"),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    with pytest.raises(MetadataExtractionIncomplete, match="missing grounded"):
        infer_document_metadata_from_segments(
            "compatibility.pdf",
            [MetadataSourceSegment("Controller firmware version 5.0 or later is required.", 8, 8)],
        )


def test_repeated_footer_document_code_cannot_become_routing_product(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(
            product_model="CV-X UM_A",
            product_models=["CV-X UM_A"],
            document_kind="manual",
            title="CV-X Manual",
        ),
    )

    def fake_chat_json(**kwargs):
        return (
            {
                "entities": [
                    {
                        "value": "CV-X UM_A",
                        "kind": "product_model",
                        "relation": "primary_product",
                        "source_quote": "CV-X UM_A",
                        "confidence": 0.99,
                    },
                    {
                        "value": "CV-X482",
                        "kind": "product_model",
                        "relation": "applies_to",
                        "source_quote": "CV-X482 vision controller",
                        "confidence": 0.95,
                    },
                ]
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "cvx.pdf",
        [
            MetadataSourceSegment(f"CV-X UM_A\nPage {page}\n" + ("CV-X482 vision controller" if page == 1 else ""), page, page)
            for page in range(1, 5)
        ],
    )

    assert metadata.routing_product_models == ["CV-X482"]
    assert metadata.product_model != "CV-X UM_A"
    assert "CVXUMA" not in metadata.normalized_identifier_aliases


def test_exhausted_malformed_scoped_batch_is_quarantined(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="Manual"),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ("truncated", "truncated"),
    )

    with pytest.raises(MetadataExtractionIncomplete, match="exhausted retries"):
        infer_document_metadata_from_segments("manual.pdf", [MetadataSourceSegment("ordinary setup text", 1, 1)])


def test_truncated_large_batch_is_bisected_and_recovered(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="Manual"),
    )

    def fake_chat_json(**kwargs):
        source = kwargs["messages"][1]["content"]
        has_a = "A-100 controller" in source
        has_b = "B-200 controller" in source
        if has_a and has_b:
            return ("truncated", "truncated")
        entities = []
        if has_a:
            entities.append(
                {
                    "value": "A-100",
                    "kind": "product_model",
                    "relation": "applies_to",
                    "source_quote": "A-100 controller",
                    "confidence": 0.95,
                }
            )
        if has_b:
            entities.append(
                {
                    "value": "B-200",
                    "kind": "product_model",
                    "relation": "applies_to",
                    "source_quote": "B-200 controller",
                    "confidence": 0.95,
                }
            )
        return ({"entities": entities}, "{}")

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "manual.pdf",
        [MetadataSourceSegment("A-100 controller\n" + "x" * 2200 + "\nB-200 controller", 1, 1)],
    )

    assert metadata.routing_product_models == ["A-100", "B-200"]
