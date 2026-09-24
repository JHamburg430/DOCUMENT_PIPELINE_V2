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
    _ground_values,
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
        assert kwargs["num_batch"] == settings.ollama_metadata_num_batch
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
        "KEYENCE AMERICA\nCA-EN100U\nEncoder relay unit\nRevision 2026/01/12\nEffective 2026/01/12",
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


def test_scoped_metadata_defaults_missing_advisory_confidence():
    extraction = ScopedMetadataExtraction.model_validate({"entities": [{
        "value": "EtherNet/IP",
        "kind": "protocol",
        "relation": "applies_to",
        "source_quote": "Protocol | EtherNet/IP",
    }]})

    assert extraction.entities[0].confidence == 0.5


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
    assert _canonical_routing_identifier("KV: X", repeated_lines=set()) == "KV-X"
    assert _canonical_routing_identifier("LJ: X8000", repeated_lines=set()) == "LJ-X8000"
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


def test_version_completeness_detects_named_editor_and_runtime_statements():
    for text in (
        "Using the ExampleEditor(Ver.5.1.0020, Ver.4.2.0020 or later), upload settings.",
        "The ExampleLanguage version used in this system is Ver. 5.1.4.",
    ):
        assert _expected_version_kinds([MetadataSourceSegment(text, 2, 2)]) == {"software_version"}


def test_plain_ethernet_is_harvested_without_collapsing_ethernet_ip():
    from manuals_rag_parsers.metadata import _deterministic_protocol_evidence
    claims = _deterministic_protocol_evidence([MetadataSourceSegment("Ethernet and EtherNet/IP", 1, 1)])
    assert {c["value"] for c in claims} == {"ethernet", "ethernet/ip"}
    assert all(c["relation"] == "mentioned" for c in claims)


def test_deterministic_protocol_quote_is_a_literal_contiguous_source_span():
    from manuals_rag_parsers.metadata import _deterministic_protocol_evidence

    source = "  IO-Link   specification V.1.1/COM2 is supported.  "
    claims = _deterministic_protocol_evidence([MetadataSourceSegment(source, 7, 7)])

    assert claims[0]["source_quote"] == source.strip()
    assert claims[0]["source_quote"] in source


def test_software_version_gate_requires_each_explicit_value():
    from manuals_rag_parsers.metadata import _missing_explicit_software_versions
    source = [MetadataSourceSegment("ExampleEditor(Ver.5.1.0020, Ver.4.2.0020 or later)", 3, 3)]
    assert _missing_explicit_software_versions(source, [{"kind": "software_version", "value": "5.1.0020"}]) == {"4.2.0020"}
    assert _deterministic_version_evidence([MetadataSourceSegment("The runtime version used in this system is Ver. 5.1.4.", 2, 2)], {"software_version"}) == []


def test_grounding_preserves_late_identifier_in_full_table_quote():
    text = "Model | Specification\n" + "Earlier row | details\n" * 30 + "ZX-2400 | 2400 mm"
    extracted = ScopedMetadataExtraction.model_validate({"entities": [{
        "value": "ZX-2400", "kind": "product_model", "relation": "primary_product",
        "source_quote": text, "confidence": 0.95,
    }]})
    claims = _ground_scoped_candidates(extracted, [MetadataSourceSegment(text, 2, 2)])
    assert len(claims) == 1
    assert claims[0]["source_quote"] == text
    assert "ZX-2400" in claims[0]["source_quote"]


def test_code_only_footer_cannot_become_primary_product_family():
    extracted = ScopedMetadataExtraction.model_validate({"entities": [
        {
            "value": "KA-US", "kind": "product_family", "relation": "primary_product",
            "source_quote": "KA-US 2114-1 689034", "confidence": 0.95,
        },
        {
            "value": "SZ-V", "kind": "product_family", "relation": "primary_product",
            "source_quote": "SZ-V Series laser safety scanner", "confidence": 0.95,
        },
    ]})
    segment = MetadataSourceSegment(
        "SZ-V Series laser safety scanner\nKA-US 2114-1 689034", 2, 2
    )

    claims = _ground_scoped_candidates(extracted, [segment])

    assert next(item for item in claims if item["value"] == "KA-US")["relation"] == "mentioned"
    assert next(item for item in claims if item["value"] == "SZ-V")["relation"] == "primary_product"


def test_option_combination_cannot_become_product_family():
    text = "OP-87772 + OP-87775 + LR-TB2000/TB2000C/TB2000CL"
    extracted = ScopedMetadataExtraction.model_validate({"entities": [{
        "value": text,
        "kind": "product_family",
        "relation": "mentioned",
        "source_quote": text,
        "confidence": 0.9,
    }]})

    assert _ground_scoped_candidates(
        extracted,
        [MetadataSourceSegment(text, 21, 21)],
    ) == []


def test_slash_compressed_models_cannot_become_scoped_model_claims():
    quote = "M12 connector type models: LR-TB2000C/TB2000CL"
    extracted = ScopedMetadataExtraction.model_validate({"entities": [
        {
            "value": "LR-TB2000C/TB2000CL",
            "kind": "product_model",
            "relation": "applies_to",
            "source_quote": quote,
            "confidence": 0.9,
        },
        {
            "value": "TB2000CL",
            "kind": "product_model",
            "relation": "compatible_with",
            "source_quote": quote,
            "confidence": 0.9,
        },
    ]})

    assert _ground_scoped_candidates(
        extracted,
        [MetadataSourceSegment(quote, 19, 19)],
    ) == []


def test_accessory_part_number_is_not_a_device():
    text = "Use the OP-26751 mounting accessory."
    assert _ground_values("devices", ["OP-26751"], "manual.pdf", text) == []
    extracted = ScopedMetadataExtraction.model_validate({"entities": [{
        "value": "OP-26751",
        "kind": "device",
        "relation": "mentioned",
        "source_quote": text,
        "confidence": 0.9,
    }]})
    assert _ground_scoped_candidates(
        extracted,
        [MetadataSourceSegment(text, 1, 1)],
    ) == []


@pytest.mark.parametrize("value", ["COM2", "M12", "IP67", "SUS304"])
def test_specification_tokens_are_not_devices(value):
    assert _ground_values("devices", [value], "manual.pdf", value) == []


def test_model_column_coverage_excludes_compatible_products():
    from manuals_rag_parsers.metadata import (
        _compatible_model_column_claims,
        _literal_compatible_model_column_claim_is_confirmed,
        _model_column_identifiers,
    )
    segment = MetadataSourceSegment(
        "Model | Length | Recommended compatible models\n"
        "ZX-1000 | 1000 | AB-20 / AB-40\n | | AB-60\nZX-2400 | 2400 | AB-80", 2, 2)
    assert _model_column_identifiers(segment) == ["ZX-1000", "ZX-2400"]
    claims = _compatible_model_column_claims([segment])
    assert [(item["value"], item["subject"]) for item in claims] == [
        ("AB-20", "ZX-1000"),
        ("AB-40", "ZX-1000"),
        ("AB-60", "ZX-1000"),
        ("AB-80", "ZX-2400"),
    ]
    assert all(item["relation"] == "compatible_with" for item in claims)
    assert "ZX-1000" in claims[2]["source_quote"]
    assert "Recommended compatible models" in claims[2]["source_quote"]
    assert "ZX-1000 | 1000 | AB-20 / AB-40\n| | AB-60" in claims[2]["source_quote"]
    assert " ".join(claims[3]["source_quote"].split()) == " ".join(segment.text.split())
    assert all(_literal_compatible_model_column_claim_is_confirmed(item) for item in claims)
    assert _model_column_identifiers(MetadataSourceSegment("Model name | ZX-15 | ZX-25\nRange | 5 | 10", 2, 2)) == ["ZX-15", "ZX-25"]


def test_verified_compatibility_subject_counts_for_model_column_coverage():
    from manuals_rag_parsers.metadata import _verified_model_identifiers

    claims = [{
        "value": "GL-R191F",
        "kind": "product_model",
        "relation": "compatible_with",
        "subject": "GL-FB2400",
        "grounded": True,
        "verification_status": "confirmed",
    }]

    assert _verified_model_identifiers(claims) == {"GLR191F", "GLFB2400"}


def test_dedupe_preserves_grounded_upload_identity_provenance_for_routing():
    from manuals_rag_parsers.metadata import _dedupe_evidence

    model_claim = {
        "value": "CA-EN100U", "kind": "product_model", "relation": "mentioned",
        "subject": None, "page_from": 1, "source_method": "page_aware_model_extraction",
    }
    upload_claim = {
        **model_claim, "source_method": "upload_identity_page_grounded", "confidence": 0.45,
    }

    assert _dedupe_evidence([model_claim, upload_claim])[0]["source_method"] == "upload_identity_page_grounded"


def test_verified_upload_identity_requires_deterministic_provenance():
    from manuals_rag_parsers.metadata import _verified_upload_identity_identifiers

    common = {
        "value": "CA-EN100U", "kind": "product_model", "relation": "mentioned",
        "grounded": True, "verification_status": "confirmed",
    }
    assert _verified_upload_identity_identifiers([{**common, "source": "upload_identity_page_grounded"}]) == {"CAEN100U"}
    assert _verified_upload_identity_identifiers([{**common, "source_method": "page_aware_model_extraction"}]) == set()
    assert _verified_upload_identity_identifiers([{
        **common, "relation": "primary_product", "source_method": "page_aware_model_extraction",
    }]) == {"CAEN100U"}


def test_compatibility_bullet_cannot_replace_catalog_title():
    from manuals_rag_parsers.metadata import _select_document_title
    title, _ = _select_document_title("catalog.pdf", "■ ZX-900 series integrated model", [
        MetadataSourceSegment("ZX-900 Series mounting bracket : ZX-FB31\n■ ZX-900 series integrated model", 1, 1)
    ])
    assert title == "ZX-900 Series mounting bracket : ZX-FB31"


def test_metadata_schema_workaround_preserves_other_model_budgets(monkeypatch):
    from types import SimpleNamespace
    from manuals_rag_parsers import metadata as module
    from manuals_rag_parsers.metadata import _metadata_thinking, _metadata_token_budget
    monkeypatch.setattr(module, "settings", SimpleNamespace(ollama_metadata_model="qwen3.5:9b"))
    assert not _metadata_thinking() and _metadata_token_budget(320) == 320
    monkeypatch.setattr(module, "settings", SimpleNamespace(ollama_metadata_model="different-model:8b"))
    assert not _metadata_thinking() and _metadata_token_budget(320) == 320


def test_dense_identifier_batch_is_split_before_schema_cap_loses_rows(monkeypatch):
    from manuals_rag_parsers import metadata as module
    calls = []
    def extract(filename, messages, *, purpose):
        content = messages[1]["content"]
        candidates = json.loads(content.split("classified):\n", 1)[1].split("\n\n", 1)[0])
        calls.append(len(candidates))
        return ScopedMetadataExtraction.model_validate({"entities": [
            {"value": c["value"], "kind": "product_model", "relation": "mentioned",
             "source_quote": c["source_quote"], "page_from": c["page_from"], "confidence": 0.9}
            for c in candidates
        ]})
    monkeypatch.setattr(module, "_call_scoped_model", extract)
    segments = [MetadataSourceSegment(f"Model ZX-{1000+i}", i+1, i+1) for i in range(12)]
    claims = module._extract_scoped_metadata("catalog.pdf", segments)
    assert len(claims) == 12
    assert len(calls) > 1 and max(calls) <= module.MAX_SCOPED_ENTITIES


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


def test_title_selection_rejects_download_call_to_action(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(
            document_kind="datasheet",
            title="Download CAD file or product manual for larger image/text and more detail.",
        ),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: (
            {"title": "Download CAD file or product manual for larger image/text and more detail."}
            if kwargs["purpose"] == "metadata_extraction.document_title"
            else {"entities": []},
            "{}",
        ),
    )

    metadata = infer_document_metadata_from_segments(
        "VJ-H500CX_Datasheet.pdf",
        [
            MetadataSourceSegment("Model | VJ-H500CX", 1, 1),
            MetadataSourceSegment(
                "Download CAD file or product manual for larger image/text and more detail.",
                2,
                2,
            ),
        ],
    )

    assert metadata.title == "VJ-H500CX Datasheet"


def test_title_selection_prefers_page_one_identifier_heading_over_preface(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(
            document_kind="manual",
            title="Please read the instruction manual carefully in advance.",
        ),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    metadata = infer_document_metadata_from_segments(
        "OP-88310_OP-88381.pdf",
        [
            MetadataSourceSegment(
                "Smart Bracket OP-88310 / OP-88381\n"
                "Please read the instruction manual carefully in advance.",
                1,
                1,
            )
        ],
    )

    assert metadata.title == "Smart Bracket OP-88310 / OP-88381"
    assert metadata.product_model is None
    assert metadata.routing_product_models == []
    assert metadata.routing_part_numbers == ["OP-88310", "OP-88381"]


def test_title_selection_rejects_competing_accessory_callout(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(
            document_kind="brochure", title="All-Purpose Laser Sensor",
        ),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    metadata = infer_document_metadata_from_segments(
        "AS_79692_LR-T_C_611C20_KA_US_2074_3_unlocked.pdf",
        [MetadataSourceSegment(
            "New Standard! All-Purpose Laser Sensor\n"
            "Multi-Sensor Controller MU-N Series\nLR-T\nSERIES",
            1,
            1,
        )],
    )

    assert metadata.title == "All-Purpose Laser Sensor"
    assert metadata.product_model == "LR-T"
    assert metadata.routing_product_models == ["LR-T"]


def test_title_selection_recovers_split_cover_product_heading(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(document_kind="brochure"),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    metadata = infer_document_metadata_from_segments(
        "AS_79692_LR-T_C_611C20_KA_US_2074_3_unlocked.pdf",
        [
            MetadataSourceSegment("New Standard!  All-Purpose Laser Sensor", 1, 1),
            MetadataSourceSegment("Multi-Sensor Controller MU-N Series", 1, 1),
            MetadataSourceSegment("LR-T", 1, 1),
            MetadataSourceSegment("SERIES", 1, 1),
        ],
    )

    assert metadata.title == "All-Purpose Laser Sensor"
    title_evidence = [
        item for item in metadata.metadata_evidence
        if item["kind"] == "document_title"
    ]
    assert title_evidence == [{
        "value": "All-Purpose Laser Sensor",
        "kind": "document_title",
        "relation": "printed_title",
        "subject": None,
        "source_quote": "All-Purpose Laser Sensor",
        "page_from": 1,
        "page_to": 1,
        "section_path": [],
        "confidence": 0.95,
        "grounded": True,
        "source": "opening_page_title",
    }]


def test_title_selection_rejects_competing_model_list_from_title_fallback(monkeypatch):
    from manuals_rag_parsers.metadata import _select_document_title

    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_printed_title",
        lambda _text: "CA-HL02MX/04MX/08MX, CA-H048CX/MX, CA-H200CX/MX, CA-H500CX/MX",
    )
    title, evidence = _select_document_title(
        "CA-S20D_Datasheet.pdf",
        "CA-HL02MX/04MX/08MX, CA-H048CX/MX, CA-H200CX/MX, CA-H500CX/MX",
        [MetadataSourceSegment(
            "CA-S20D\nSupported camera models:\n"
            "CA-HL02MX/04MX/08MX, CA-H048CX/MX, CA-H200CX/MX, CA-H500CX/MX",
            1,
            1,
        )],
    )

    assert title == "CA-S20D Datasheet"
    assert evidence is None


def test_competing_short_series_cover_callout_is_not_primary():
    from manuals_rag_parsers.metadata import _demote_competing_cover_callouts

    evidence = [{
        "value": "MU-N", "kind": "product_model", "relation": "primary_product",
        "page_from": 1,
    }]
    identity = [{"value": "LR-T", "kind": "product_model", "relation": "mentioned"}]

    assert _demote_competing_cover_callouts(evidence, identity)[0]["relation"] == "mentioned"


def test_scalar_metadata_normalizes_array_shape_dates_and_kind_synonyms(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ([{"title": "Release Notes", "document_kind": "release_notes", "revision_date": "2025/04/22", "effective_date": "null"}], "[]"),
    )

    metadata = infer_document_metadata("release.pdf", "Release Notes\nRevision 2025/04/22")

    assert metadata.document_kind.value == "release_note"
    assert metadata.revision_date.isoformat() == "2025-04-22"
    assert metadata.effective_date == metadata.revision_date


def test_scalar_metadata_rejects_unlabeled_footer_date_as_revision(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: (
            {
                "title": "VJ-3302 Image processing unit",
                "document_kind": "datasheet",
                "revision_date": "2025/04/22",
                "effective_date": "2025/04/22",
            },
            "{}",
        ),
    )

    metadata = infer_document_metadata(
        "VJ-3302_Datasheet.pdf",
        "VJ-3302 Image processing unit\n2025/04/22\nPage 1 of 3",
    )

    assert metadata.revision_date is None
    assert metadata.effective_date is None


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
        if kwargs["purpose"] in {
            "metadata_extraction.scoped_entities",
            "metadata_extraction.claim_verification",
        }:
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


def test_late_compatible_model_is_not_a_hard_routing_key():
    evidence = [
        {
            "value": "KV-X",
            "kind": "product_model",
            "relation": "primary_product",
            "page_from": 1,
            "grounded": True,
            "verification_status": "confirmed",
            "confidence": 0.95,
        },
        {
            "value": "CV500",
            "kind": "product_model",
            "relation": "compatible_with",
            "page_from": 43,
            "grounded": True,
            "verification_status": "confirmed",
            "confidence": 0.95,
        },
    ]

    assert _values_for_routing(evidence, "product_model") == ["KV-X"]


def test_unscoped_table_part_number_is_not_a_hard_routing_key():
    evidence = [
        {
            "value": "SV2-040L2",
            "kind": "part_number",
            "relation": "applies_to",
            "subject": None,
            "page_from": 60,
            "grounded": True,
            "verification_status": "confirmed",
            "confidence": 0.95,
        },
        {
            "value": "OP-42284",
            "kind": "part_number",
            "relation": "accessory_for",
            "subject": "CV-X482",
            "page_from": 80,
            "grounded": True,
            "verification_status": "confirmed",
            "confidence": 0.95,
        },
        {
            "value": "OP-99999",
            "kind": "part_number",
            "relation": "accessory_for",
            "subject": "External-PLC",
            "page_from": 81,
            "grounded": True,
            "verification_status": "confirmed",
            "confidence": 0.95,
        },
    ]

    assert _values_for_routing(
        evidence,
        "part_number",
        routing_subjects=["CV-X482"],
    ) == ["OP-42284"]


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

    assert metadata.product_model == "ZX-900"
    assert metadata.routing_product_models == ["ZX-900"]
    claim = next(item for item in metadata.metadata_claims if item["value"] == "ZX-900")
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


def test_verifier_transport_failure_quarantines_entire_document(monkeypatch):
    claims = reconcile_metadata_claims(
        [
            {
                "value": "ZX-900",
                "kind": "product_model",
                "relation": "primary_product",
                "source_quote": "ZX-900 Controller Manual",
                "page_from": 1,
                "grounded": True,
            }
        ]
    )

    def fail_verifier(**kwargs):
        raise RuntimeError("empty model response")

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fail_verifier)

    with pytest.raises(MetadataExtractionIncomplete, match="Independent claim verification did not complete"):
        verify_metadata_claims(
            "manual.pdf",
            claims,
            [MetadataSourceSegment("ZX-900 Controller Manual", 1, 1)],
        )


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


def test_integration_guide_routes_grounded_upload_products_not_external_cover_title(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._extract_metadata_with_model",
        lambda filename, text: MetadataExtraction(
            document_kind="manual",
            title="SIEMENS S7-1500/1200/300 SERIES",
        ),
    )
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata.chat_json",
        lambda **kwargs: ({"entities": []}, "{}"),
    )

    metadata = infer_document_metadata_from_segments(
        "AS_SR-1000_SR-2000_SR-PN1_guide.pdf",
        [
                MetadataSourceSegment(
                    "SIEMENS S7-1500/1200/300 SERIES\n"
                    "Connection Guide: PROFINET Communication\n"
                    "SR-X300/X100/2000/1000\nSR-PN1",
                1,
                1,
            )
        ],
    )

    assert metadata.product_model == "SR-1000"
    assert metadata.routing_product_models == ["SR-1000", "SR-2000", "SR-PN1"]
    assert metadata.routing_protocol_terms == ["profinet"]
    assert "S7-1500" not in metadata.routing_product_models


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


def test_protocol_support_and_external_example_are_not_scope_contradictions():
    claims = reconcile_metadata_claims([
        dict(value='RS-232C', kind='protocol', relation=relation, subject=None,
             source_quote=quote, page_from=page, grounded=True)
        for relation, quote, page in [
            ('compatible_with', 'Controller supports RS-232C communication.', 2),
            ('external_reference', 'Example external device using RS-232C.', 8),
            ('mentioned', 'RS-232C communication.', 9),
        ]
    ])
    assert len(claims) == 3
    assert all(item['verification_status'] == 'unresolved' for item in claims)
    # Independent verification is still required before any routing promotion.


def test_metadata_rejects_garbage_source_before_model(monkeypatch):
    def no_model(**kwargs):
        raise AssertionError('Bad source must not invoke a model')
    monkeypatch.setattr('manuals_rag_parsers.metadata.chat_json', no_model)
    with pytest.raises(MetadataExtractionIncomplete, match='reparse/OCR'):
        infer_document_metadata_from_segments('manual.pdf', [MetadataSourceSegment('q', n, n) for n in range(30)])


def test_metadata_retry_seeds_are_stable_and_attempt_specific():
    from manuals_rag_parsers.metadata import _metadata_seed

    first = _metadata_seed("metadata_extraction.scoped_entities", 1)
    assert first == _metadata_seed("metadata_extraction.scoped_entities", 1)
    assert first != _metadata_seed("metadata_extraction.scoped_entities", 2)
    assert first != _metadata_seed("metadata_extraction.claim_verification", 1)
    assert 0 <= first <= 0x7FFFFFFF


def test_source_native_identifier_ledger_is_literal_ordered_and_repeatable():
    from manuals_rag_parsers.metadata import (
        _source_native_identifier_evidence,
        reconcile_metadata_claims,
    )

    segments = [
        MetadataSourceSegment(
            "  LR-TB2000 + OP-87772  \nM12 IP67 SUS304 COM2",
            12,
            12,
            ("Accessories",),
        ),
        MetadataSourceSegment("OP-87770 for LR-TB2000", 11, 11, ("Accessories",)),
    ]
    first = reconcile_metadata_claims(_source_native_identifier_evidence(segments))
    second = reconcile_metadata_claims(_source_native_identifier_evidence(segments))

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert [(item["page_from"], item["value"]) for item in first] == [
        (11, "OP-87770"),
        (11, "LR-TB2000"),
        (12, "OP-87772"),
    ]
    assert next(item for item in first if item["value"] == "LR-TB2000")[
        "support_pages"
    ] == [11, 12]
    source_lines = {line.strip() for segment in segments for line in segment.text.splitlines()}
    assert all(item["source_quote"] in source_lines for item in first)
    assert {item["value"] for item in first}.isdisjoint({"M12", "IP67", "SUS304", "COM2"})


def test_source_native_identifier_claims_bypass_model_verifier(monkeypatch):
    from manuals_rag_parsers.metadata import _canonical_source_native_claims

    segment = MetadataSourceSegment("LR-TB2000 uses OP-87772", 7, 7, ("Accessories",))
    monkeypatch.setattr(
        "manuals_rag_parsers.metadata._call_scoped_model",
        lambda *args, **kwargs: pytest.fail("literal source-native claims must not call the model"),
    )

    verified = _canonical_source_native_claims("LR-T.pdf", "LR-T", [segment])

    assert len(verified) == 2
    assert {item["verification_status"] for item in verified} == {"confirmed"}
    assert {item["source_method"] for item in verified} == {"source_native_identifier"}


def test_scoped_metadata_uses_deterministic_sampling_controls(monkeypatch):
    from manuals_rag_parsers.metadata import _metadata_seed

    observed = []

    def fake_chat_json(**kwargs):
        observed.append(kwargs)
        return ({"entities": []}, "{}")

    monkeypatch.setattr("manuals_rag_parsers.metadata.chat_json", fake_chat_json)
    _call_scoped_model(
        "manual.pdf",
        [{"role": "user", "content": "Extract entities"}],
        purpose="metadata_extraction.scoped_entities",
    )

    assert observed[0]["temperature"] == 0.0
    assert observed[0]["seed"] == _metadata_seed("metadata_extraction.scoped_entities", 1)


def test_claim_id_verdict_preserves_original_grounded_claim(monkeypatch):
    from manuals_rag_parsers.metadata import _verification_prompt_messages
    claim = dict(value='AB-200', kind='product_model', relation='applies_to', subject=None,
                 source_quote='This function supports AB-200 controllers.', confidence=.8)
    messages = _verification_prompt_messages('manual.pdf', [claim], [MetadataSourceSegment(claim['source_quote'],1,1)])
    monkeypatch.setattr('manuals_rag_parsers.metadata.chat_json', lambda **kwargs: (
        {'decisions': [{'claim_id':'claim_1','supported':True,'reason':'The source explicitly names the supported controller.'}]}, '{}'))
    result = _call_scoped_model('manual.pdf', messages, purpose='metadata_extraction.claim_verification')
    assert result.entities[0].value == 'AB-200'
    assert result.entities[0].source_quote == claim['source_quote']


def test_claim_id_verdict_rejects_duplicate_decisions(monkeypatch):
    from manuals_rag_parsers.metadata import _verification_prompt_messages
    claim = dict(value='AB-200',kind='product_model',relation='applies_to',subject=None,
                 source_quote='AB-200 controllers.',confidence=.8)
    messages = _verification_prompt_messages('manual.pdf',[claim],[MetadataSourceSegment(claim['source_quote'],1,1)])
    verdict = {'claim_id':'claim_1','supported':True,'reason':'supported'}
    monkeypatch.setattr('manuals_rag_parsers.metadata.chat_json',lambda **kwargs: ({'decisions':[verdict,verdict]},'{}'))
    with pytest.raises(MetadataExtractionIncomplete):
        _call_scoped_model('manual.pdf',messages,purpose='metadata_extraction.claim_verification')


def test_scoped_model_rejects_missing_collection(monkeypatch):
    monkeypatch.setattr('manuals_rag_parsers.metadata.chat_json', lambda **kwargs: ({}, '{}'))
    with pytest.raises(MetadataExtractionIncomplete, match='omitted its entities'):
        _call_scoped_model('manual.pdf', [{'role':'user','content':'Extract entities'}],
                           purpose='metadata_extraction.scoped_entities')


def test_runtime_version_does_not_become_firmware_without_firmware_evidence():
    text = "The Lua version used in this system is Ver. 5.1.4."
    extracted = ScopedMetadataExtraction.model_validate({"entities": [{
        "value": "5.1.4", "kind": "firmware_version", "relation": "applies_to",
        "subject": "Lua", "source_quote": text, "confidence": 0.95,
    }]})
    assert _ground_scoped_candidates(extracted, [MetadataSourceSegment(text, 2, 2)]) == []


def test_version_coverage_accepts_printed_version_prefix_without_losing_values():
    from manuals_rag_parsers.metadata import _missing_explicit_software_versions
    source = [MetadataSourceSegment("ExampleEditor(Ver.5.1.0020, Ver.4.2.0020 or later)",3,3)]
    claims = [{"kind":"software_version","value":"Ver. 5.1.0020"},
              {"kind":"software_version","value":"Version 4.2.0020"}]
    assert _missing_explicit_software_versions(source, claims) == set()
    assert _missing_explicit_software_versions(source, claims[:1]) == {"4.2.0020"}


def test_explicit_parenthesized_version_list_is_mentions_only():
    from manuals_rag_parsers.metadata import (
        _deterministic_version_evidence, _literal_deterministic_version_claim_is_confirmed,
        _parenthesized_version_mentions,
    )
    source = [MetadataSourceSegment("Using the ExampleEditor(Ver.5.1.0020, Ver.4.2.0020 or later), upload settings.", 3, 3)]
    claims = _deterministic_version_evidence(source, {"software_version"})
    assert {(c["subject"], c["value"]) for c in claims} == {
        ("ExampleEditor", "5.1.0020"), ("ExampleEditor", "4.2.0020")}
    for claim in claims:
        claim["source_method"] = claim["source"]
        assert claim["relation"] == "mentioned"
        assert _literal_deterministic_version_claim_is_confirmed(claim)
        assert not _literal_deterministic_version_claim_is_confirmed({**claim, "relation": "applies_to"})
    assert not _parenthesized_version_mentions("ExampleEditor(Ver.5.1, OtherEditor Ver.4.2)")


def test_grounding_rejects_unquoted_subject_even_for_mentions():
    from manuals_rag_parsers.metadata import _ground_scoped_candidates, ScopedMetadataExtraction
    extraction = ScopedMetadataExtraction.model_validate({"entities": [{
        "kind": "protocol", "value": "Ethernet", "relation": "mentioned",
        "subject": "invented_manual.pdf", "source_quote": "Ethernet communication", "confidence": 0.9}]})
    assert not _ground_scoped_candidates(extraction, [MetadataSourceSegment("Ethernet communication", 1, 1)])
