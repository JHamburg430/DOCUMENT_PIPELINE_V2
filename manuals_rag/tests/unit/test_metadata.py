import json

import pytest

from manuals_rag_parsers.metadata import (
    LIST_FIELD_INSTRUCTIONS,
    METADATA_PIPELINE_VERSION,
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
    build_metadata_extraction_graph,
    harvest_metadata_candidates,
    infer_document_metadata,
    infer_document_metadata_from_segments,
    pack_metadata_source_segments,
    reconcile_metadata_claims,
    verify_metadata_claims,
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


def test_alphanumeric_family_is_promoted_when_model_field_is_empty(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: (
            {
                "manufacturer": "Acme Controls",
                "product_family": "AX-1200U",
                "product_families": ["AX-1200U"],
                "product_models": [],
                "document_kind": "datasheet",
                "title": "AX-1200U Datasheet",
            },
            "{}",
        ),
    )

    metadata = infer_document_metadata(
        "datasheet.pdf",
        "Acme Controls AX-1200U Datasheet",
    )

    assert metadata.product_family == "AX-1200U"
    assert metadata.product_model == "AX-1200U"
    assert metadata.product_models == ["AX-1200U"]


def test_synthetic_page_and_section_markers_are_not_menu_labels(monkeypatch):
    def fake_chat_json(**kwargs):
        return (
            {
                "document_kind": "manual",
                "title": "Setup Manual",
                "menu_labels": ["[PAGE 1]", "[SECTION Setup]", "[Run]"],
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)

    metadata = infer_document_metadata(
        "setup.pdf",
        "[PAGE 1] [SECTION Setup] Setup Manual. Select [Run].",
    )

    assert metadata.menu_labels == ["[Run]"]


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
        if kwargs["purpose"] == "metadata_extraction.document_title":
            return ({"title": "KEYENCE CV-X482 vision controller"}, "{}")
        if kwargs["purpose"] == "metadata_extraction.claim_verification":
            content = kwargs["messages"][1]["content"]
            claims = json.loads(content.split("CLAIMS TO VERIFY:\n", 1)[1].split("\n\n", 1)[0])
            return ({"entities": claims}, "{}")
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


def test_scoped_metadata_accepts_type_and_entity_aliases():
    extraction = ScopedMetadataExtraction.model_validate({"entities": [{
        "entity": "CV-X482",
        "type": "product_model",
        "relation": "primary_product",
        "source_quote": "CV-X482 vision controller",
        "confidence": 0.95,
    }]})
    assert extraction.entities[0].value == "CV-X482"


def test_scoped_metadata_accepts_entity_kind_alias():
    extraction = ScopedMetadataExtraction.model_validate({"entities": [{
        "value": "AX-420",
        "entity_kind": "product_model",
        "relation": "primary_product",
        "source_quote": "AX-420 controller",
        "confidence": 0.95,
    }]})
    assert extraction.entities[0].kind == "product_model"
    assert extraction.entities[0].kind == "product_model"


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


def test_filename_identifier_routes_when_matching_descriptive_front_title(monkeypatch):
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

    assert metadata.title == "LJ-X8000 Communication Command Manual"
    assert metadata.routing_product_models == ["LJ-X8000"]
    assert metadata.product_model == "LJ-X8000"
    assert any(
        item.get("source_method") == "opening_title_candidate"
        and item["relation"] == "mentioned"
        for item in metadata.metadata_evidence
    )


def test_noisy_filename_title_is_replaced_by_grounded_opening_page_title(monkeypatch):
    noisy_filename = "AS_103012_LineScan_C_611J12_KA_US_2073_3.pdf"
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="spec_sheet", title=noisy_filename),
    )

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "metadata_extraction.document_title":
            assert "FILENAME" not in kwargs["messages"][1]["content"]
            assert "Overview" in kwargs["messages"][1]["content"]
            assert "Later appendix" not in kwargs["messages"][1]["content"]
            return ({"title": "Automatic Inspection of Defects, Uneven Surfaces, and Contamination"}, "{}")
        return ({"entities": []}, "{}")

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        noisy_filename,
        [
            MetadataSourceSegment(
                "Automatic Inspection of Defects, Uneven Surfaces, and Contamination\nXG-X Series",
                1,
                1,
            ),
            MetadataSourceSegment("Later appendix", 6, 6),
            MetadataSourceSegment("Overview", 2, 2),
            MetadataSourceSegment("Unrelated Section Heading", 3, 3),
        ],
    )

    assert metadata.title == "Automatic Inspection of Defects, Uneven Surfaces, and Contamination"
    title_evidence = [item for item in metadata.metadata_evidence if item["kind"] == "document_title"]
    assert len(title_evidence) == 1
    assert title_evidence[0]["page_from"] == 1
    assert title_evidence[0]["source"] == "opening_page_title"


def test_title_selection_does_not_use_a_later_section_heading(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="Advanced Configuration"),
    )

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "metadata_extraction.document_title":
            return ({"publication_title": "Vision System User's Manual"}, "{}")
        return ({"entities": []}, "{}")

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "opaque_4815162342.pdf",
        [
            MetadataSourceSegment("Vision System User's Manual", 1, 1),
            MetadataSourceSegment("Contents", 2, 2),
            MetadataSourceSegment("Advanced Configuration", 3, 3),
        ],
    )

    assert metadata.title == "Vision System User's Manual"


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
        lambda filename, text: MetadataExtraction(document_kind="manual", title="CV-X482 vision controller"),
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

    assert calls == 3
    assert metadata.routing_product_models == ["CV-X482"]
    assert metadata.product_model == "CV-X482"
    assert metadata.metadata_evidence[0]["confidence"] == pytest.approx(0.85)


def test_scoped_metadata_retries_json_errors_before_splitting(monkeypatch):
    calls = 0

    def malformed_chat_json(**kwargs):
        nonlocal calls
        calls += 1
        raise json.JSONDecodeError("unterminated", '"', 0)

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", malformed_chat_json)
    with pytest.raises(MetadataExtractionIncomplete):
        _call_scoped_model("manual.pdf", [{"role": "user", "content": "text"}], purpose="test")
    assert calls == 3


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
        if kwargs["purpose"] == "metadata_extraction.claim_verification":
            content = kwargs["messages"][1]["content"]
            claims = json.loads(content.split("CLAIMS TO VERIFY:\n", 1)[1].split("\n\n", 1)[0])
            return ({"entities": claims}, "{}")
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


def test_candidate_harvester_preserves_page_section_and_quote():
    candidates = harvest_metadata_candidates(
        [
            MetadataSourceSegment(
                "Controller AX-420 requires firmware version 3.14 and supports EtherCAT.",
                42,
                42,
                ("Compatibility", "Controller"),
            )
        ]
    )

    assert any(item["value"] == "AX-420" for item in candidates)
    assert any(item["candidate_kind"] == "version_statement" for item in candidates)
    assert any(item["value"].casefold() == "ethercat" for item in candidates)
    assert all(item["page_from"] == 42 for item in candidates)
    assert all(item["section_path"] == ["Compatibility", "Controller"] for item in candidates)
    assert all("AX-420" in item["source_quote"] for item in candidates)


def test_reducer_merges_alias_equivalent_claims_across_pages():
    claims = reconcile_metadata_claims(
        [
            {
                "value": "CV-X482",
                "kind": "product_model",
                "relation": "primary_product",
                "source_quote": "CV-X482 controller",
                "page_from": 1,
                "grounded": True,
            },
            {
                "value": "CVX482",
                "kind": "product_model",
                "relation": "primary_product",
                "source_quote": "CVX482 settings",
                "page_from": 8,
                "grounded": True,
            },
        ]
    )

    assert len(claims) == 1
    assert claims[0]["support_pages"] == [1, 8]
    assert claims[0]["normalized_value"] == "CVX482"


def test_reducer_marks_contradictory_scope_as_conflicting():
    evidence = [
        {
            "value": "PLC-900",
            "kind": "product_model",
            "relation": relation,
            "subject": None,
            "source_quote": "PLC-900 controller",
            "page_from": page,
            "grounded": True,
        }
        for relation, page in (("primary_product", 1), ("external_reference", 9))
    ]

    claims = reconcile_metadata_claims(evidence)

    assert {claim["verification_status"] for claim in claims} == {"conflicting"}


def test_reducer_rejects_prose_shaped_model_and_sanitizes_company_footer():
    claims = reconcile_metadata_claims(
        [
            {
                "value": "ZX Series obtains detailed measurements and displays the resulting image on screen",
                "kind": "product_model",
                "relation": "mentioned",
                "source_quote": "ZX Series obtains detailed measurements and displays the resulting image on screen.",
                "page_from": 8,
                "grounded": True,
            },
            {
                "value": "ACME CORPORATION. All rights reserved: DOC-42 Printed in Japan",
                "kind": "company",
                "relation": "mentioned",
                "source_quote": "Copyright © 2026 ACME CORPORATION. All rights reserved: DOC-42 Printed in Japan",
                "page_from": 9,
                "grounded": True,
            },
        ]
    )

    assert [(claim["kind"], claim["value"]) for claim in claims] == [
        ("company", "ACME CORPORATION")
    ]


def test_opening_pages_are_ordered_by_physical_page_before_title_extraction(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="ZX-900 Easy Configuration Manual"),
    )

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "metadata_extraction.scoped_entities":
            return ({"entities": []}, "{}")
        raise AssertionError(kwargs["purpose"])

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "opaque.pdf",
        [
            MetadataSourceSegment("Chapter 1 Installation | Contents", 2, 2),
            MetadataSourceSegment("ZX-900 Easy Configuration Manual", 1, 1),
        ],
    )

    assert metadata.title == "ZX-900 Easy Configuration Manual"


def test_descriptive_opening_title_beats_isolated_document_code(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="D47AB"),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    metadata = infer_document_metadata_from_segments(
        "opaque.pdf",
        [
            MetadataSourceSegment("D47AB", 1, 1, ("D47AB",)),
            MetadataSourceSegment("ZX: 8000 Series Easy Configuration Manual", 1, 1),
        ],
    )

    assert metadata.title == "ZX: 8000 Series Easy Configuration Manual"
    assert "D47AB" not in metadata.routing_product_models


def test_independent_verifier_controls_routing_and_derives_confidence(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="ZX-900 Manual"),
    )

    def fake_chat_json(**kwargs):
        purpose = kwargs["purpose"]
        if purpose == "metadata_extraction.scoped_entities":
            return (
                {
                    "entities": [
                        {
                            "value": "ZX-900",
                            "kind": "product_model",
                            "relation": "primary_product",
                            "source_quote": "ZX-900 Manual",
                            "confidence": 0.01,
                        }
                    ]
                },
                "{}",
            )
        if purpose == "metadata_extraction.claim_verification":
            content = kwargs["messages"][1]["content"]
            claims = json.loads(content.split("CLAIMS TO VERIFY:\n", 1)[1].split("\n\n", 1)[0])
            return ({"entities": claims}, "{}")
        raise AssertionError(purpose)

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "opaque_upload.pdf",
        [MetadataSourceSegment("ZX-900 Manual", 1, 1, ("Cover",))],
    )

    assert metadata.metadata_pipeline_version == METADATA_PIPELINE_VERSION
    assert metadata.routing_product_models == ["ZX-900"]
    claim = next(item for item in metadata.metadata_claims if item["value"] == "ZX-900")
    assert claim["verification_status"] == "confirmed"
    assert claim["confidence"] == pytest.approx(0.85)


def test_confirmed_opening_title_model_can_route_when_relation_is_only_mentioned(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="datasheet", title="ZX-900 Datasheet"),
    )

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "metadata_extraction.scoped_entities":
            return ({"entities": []}, "{}")
        if kwargs["purpose"] == "metadata_extraction.claim_verification":
            content = kwargs["messages"][1]["content"]
            claims = json.loads(content.split("CLAIMS TO VERIFY:\n", 1)[1].split("\n\n", 1)[0])
            return ({"entities": claims}, "{}")
        raise AssertionError(kwargs["purpose"])

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "opaque_upload.pdf",
        [MetadataSourceSegment("ZX-900 Datasheet", 1, 1, ("Cover",))],
    )

    assert metadata.product_model == "ZX-900"
    assert metadata.routing_product_models == ["ZX-900"]


def test_late_table_primary_label_does_not_replace_document_identity(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(
            document_kind="brochure",
            title="System Overview",
            product_model="AX-100",
            product_models=["AX-100"],
            product_family="Accessory Series",
            product_families=["Accessory Series"],
        ),
    )

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "metadata_extraction.scoped_entities":
            source = kwargs["messages"][1]["content"]
            if "Model | BX-200" in source:
                return (
                    {
                        "entities": [
                            {
                                "value": "BX-200",
                                "kind": "product_model",
                                "relation": "primary_product",
                                "source_quote": "Model | BX-200",
                                "confidence": 0.9,
                            }
                        ]
                    },
                    "{}",
                )
            return ({"entities": []}, "{}")
        if kwargs["purpose"] == "metadata_extraction.claim_verification":
            content = kwargs["messages"][1]["content"]
            claims = json.loads(content.split("CLAIMS TO VERIFY:\n", 1)[1].split("\n\n", 1)[0])
            return ({"entities": claims}, "{}")
        raise AssertionError(kwargs["purpose"])

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "opaque_upload.pdf",
        [
            MetadataSourceSegment("System Overview", 1, 1, ("Cover",)),
            MetadataSourceSegment("Model | BX-200", 10, 10, ("Accessories",)),
        ],
    )

    assert metadata.product_model is None
    assert metadata.product_family is None
    assert "AX-100" in metadata.product_models
    assert "Accessory Series" not in metadata.product_families
    assert "BX-200" not in metadata.routing_product_models


def test_literal_opening_title_model_routes_when_llm_verifier_rejects_it(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="ZX: 900 Easy Configuration Manual"),
    )

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] in {
            "metadata_extraction.scoped_entities",
            "metadata_extraction.claim_verification",
        }:
            return ({"entities": []}, "{}")
        raise AssertionError(kwargs["purpose"])

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    metadata = infer_document_metadata_from_segments(
        "opaque_upload.pdf",
        [MetadataSourceSegment("ZX: 900 Easy Configuration Manual", 1, 1, ("Cover",))],
    )

    assert metadata.product_model == "ZX:900"
    assert metadata.routing_product_models == ["ZX:900"]
    claim = next(item for item in metadata.metadata_claims if item["value"] == "ZX:900")
    assert claim["verification_status"] == "confirmed"
    assert claim["confidence"] >= 0.8


def test_successful_verifier_rejection_is_distinct_from_unresolved_failure(monkeypatch):
    claims = reconcile_metadata_claims(
        [
            {
                "value": "ZX-900",
                "kind": "product_model",
                "relation": "primary_product",
                "source_quote": "ZX-900 is an external example controller.",
                "page_from": 8,
                "grounded": True,
            }
        ]
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    verified = verify_metadata_claims(
        "manual.pdf",
        claims,
        [MetadataSourceSegment("ZX-900 is an external example controller.", 8, 8)],
    )

    assert verified[0]["verification_status"] == "rejected"
    assert verified[0]["confidence"] == 0.0


def test_literal_deterministic_version_claim_survives_model_omission(monkeypatch):
    claims = reconcile_metadata_claims(
        [
            {
                "value": "12",
                "kind": "software_version",
                "relation": "mentioned",
                "subject": "CONTROL STUDIO",
                "source_quote": "CONTROL STUDIO Ver.12",
                "page_from": 4,
                "grounded": True,
                "source": "deterministic_explicit_version",
            }
        ]
    )
    calls = 0

    def fake_chat_json(**kwargs):
        nonlocal calls
        calls += 1
        return ({"entities": []}, "{}")

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    verified = verify_metadata_claims(
        "manual.pdf",
        claims,
        [MetadataSourceSegment("CONTROL STUDIO Ver.12", 4, 4)],
    )

    assert calls == 1
    assert verified[0]["verification_status"] == "confirmed"
    assert verified[0]["confidence"] >= 0.8


def test_filename_prefix_collision_cannot_override_grounded_opening_title_identity(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="MOD-500 Manual"),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    metadata = infer_document_metadata_from_segments(
        "MOD-5_manual.pdf",
        [MetadataSourceSegment("MOD-500 Manual", 1, 1)],
    )

    assert metadata.product_model == "MOD-500"
    assert metadata.routing_product_models == ["MOD-500"]
    assert "MOD-5" not in metadata.routing_product_models
    assert not any(item.get("value") == "MOD-5" for item in metadata.metadata_claims)


def test_langgraph_workflow_compiles():
    assert build_metadata_extraction_graph() is not None


def test_workflow_fails_when_verifier_rejects_all_version_claims(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="manual", title="Controller Manual"),
    )

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "metadata_extraction.scoped_entities":
            return (
                {
                    "entities": [
                        {
                            "value": "3.14",
                            "kind": "firmware_version",
                            "relation": "applies_to",
                            "subject": "ZX-900",
                            "source_quote": "ZX-900 firmware version 3.14 or later is required.",
                            "confidence": 0.9,
                        }
                    ]
                },
                "{}",
            )
        if kwargs["purpose"] == "metadata_extraction.claim_verification":
            return ({"entities": []}, "{}")
        raise AssertionError(kwargs["purpose"])

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)

    with pytest.raises(MetadataExtractionIncomplete, match="Independent verification"):
        infer_document_metadata_from_segments(
            "opaque_upload.pdf",
            [
                MetadataSourceSegment("Controller Manual", 1, 1, ("Cover",)),
                MetadataSourceSegment(
                    "ZX-900 firmware version 3.14 or later is required.",
                    10,
                    10,
                    ("Compatibility",),
                ),
            ],
        )
