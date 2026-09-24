from scripts.benchmark.audit_visual_document_understanding import (
    answer_matches,
    answer_matches_in_context,
    adjudication_can_override,
    anchored_unit_completion_passes,
    dual_visual_consensus_passes,
    _numbers,
    claim_support_passes,
    _merge_report,
    _claims,
    _vision_json,
    quote_is_anchored,
    reanchor_answer,
    remap_batch_results,
    select_pages,
)


def test_answer_match_requires_number_unit_and_identifier():
    passed, checks = answer_matches("CV-X452: 26.4 V", "The CV-X452 rating is 26.4 V")
    assert passed
    assert checks["number_match"]
    assert not answer_matches("26.4 V", "24 V")[0]
    assert not answer_matches("26.4 V", "26.4 A")[0]
    assert not answer_matches("CV-X452", "CV-X482")[0]


def test_quote_anchor_normalizes_punctuation_and_spacing():
    assert quote_is_anchored("Maximum input: 26.4 V", "Maximum input — 26.4 V")
    assert not quote_is_anchored("Maximum input 24 V", "Maximum input — 26.4 V")


def test_quote_anchor_normalizes_version_prefixes_and_multi_dot_versions():
    assert quote_is_anchored(
        "4.2.0020",
        "Using XG VisionEditor (Ver.5.1.0020, Ver.4.2.0020 or later)",
    )


def test_answer_reanchoring_preserves_numbers_units_and_identifiers():
    page = "Model | Rating\nCV-X452 | Maximum input voltage | 26.4 V\nCV-X482 | 24 V"
    assert reanchor_answer("CV-X452 26.4 V", page) == "CV-X452 | Maximum input voltage | 26.4 V"
    assert reanchor_answer("CV-X452 24 V", page) is None


def test_answer_reanchoring_can_bind_a_table_row_to_a_distant_header():
    page = "\n".join([
        "Model | Length [mm inch]", "GL-FB1000 | 1029 40.51", "row 2", "row 3", "row 4",
        "GL-FB1400 | 1429 56.26",
    ])
    anchor = reanchor_answer("1429 mm", page)
    assert anchor is not None
    assert "Length [mm inch]" in anchor
    assert "GL-FB1400 | 1429" in anchor


def test_claim_support_requires_contiguous_value_and_subject_scope():
    claim = {
        "value": "GL-R", "subject": "GL-FB1000", "relation": "compatible_with",
        "source_quote": "GL-FB1000 Compatible model GL-R",
    }
    visual = {
        "supported": True, "conflicting": False, "visible_value": "GL-R",
        "visible_subject": "GL-FB1000", "evidence_text": "GL-FB1000 Compatible model GL-R",
    }
    assert claim_support_passes(claim, visual, "GL-FB1000 Compatible model GL-R")[0]
    visual["visible_subject"] = ""
    visual["evidence_text"] = "Compatible model GL-R"
    assert not claim_support_passes(claim, visual, "Compatible model GL-R")[0]


def test_claim_support_reconciles_self_contradictory_flags_with_exact_binding():
    claim = {
        "value": "GL-R95F", "subject": "GL-FB1400", "relation": "compatible_with",
        "source_quote": "GL-FB1400 | GL-R95F / GL-R48H / GL-R24L",
    }
    visual = {
        "supported": False, "conflicting": True, "visible_value": "GL-R95F",
        "visible_subject": "GL-FB1400",
        "evidence_text": "GL-FB1400 | GL-R95F / GL-R48H / GL-R24L",
        "explanation": "The row lists GL-R95F, therefore the claim is supported.",
    }
    assert claim_support_passes(claim, visual, claim["source_quote"])[0]


def test_claim_support_does_not_ignore_conflict_without_exact_binding():
    claim = {
        "value": "GL-R95F", "subject": "GL-FB1400", "relation": "compatible_with",
        "source_quote": "GL-FB1400 | GL-R95F",
    }
    visual = {
        "supported": False, "conflicting": True, "visible_value": "GL-R95F",
        "visible_subject": "GL-FB1400", "evidence_text": "GL-R95F",
        "explanation": "The claim is supported.",
    }
    assert not claim_support_passes(claim, visual, claim["source_quote"])[0]


def test_adjudication_can_resolve_omission_but_not_explicit_conflict():
    assert adjudication_can_override({"supported": False, "conflicting": False}, True)
    assert not adjudication_can_override({"supported": False, "conflicting": True}, True)
    assert not adjudication_can_override({"supported": False, "conflicting": False}, False)


def test_unit_completion_requires_matching_facts_and_both_source_anchors():
    comparison = {"number_match": True, "identifier_match_in_context": True}
    assert anchored_unit_completion_passes("124", "124 mm", comparison, "124", "124\nUnit: mm")
    assert not anchored_unit_completion_passes("124", "125 mm", {**comparison, "number_match": False}, "124", "125")
    assert not anchored_unit_completion_passes("124", "124 mm", comparison, "124", None)


def test_dual_visual_consensus_accepts_unit_completion_without_ocr_text():
    passed, comparison = answer_matches_in_context(
        "145.7", "145.7 mm", "What is the overall width dimension?"
    )
    assert not passed
    assert dual_visual_consensus_passes(
        "145.7", "145.7 mm", comparison,
        "The overall width is labeled 145.7.",
        "The dimension line is labeled 145.7 mm.",
    )


def test_dual_visual_consensus_rejects_numeric_disagreement():
    _passed, comparison = answer_matches_in_context(
        "16.5", "46 mm", "What is the height dimension?"
    )
    assert not dual_visual_consensus_passes(
        "16.5", "46 mm", comparison, "16.5", "46 mm"
    )


def test_dual_visual_consensus_requires_independent_evidence():
    _passed, comparison = answer_matches_in_context(
        "43", "43 mm", "What is the labeled dimension?"
    )
    assert not dual_visual_consensus_passes("43", "43 mm", comparison, "43", "")


def test_page_selection_never_drops_claim_pages():
    claims = [{"page_from": 2}, {"page_from": 8}, {"page_from": 8}]
    metrics = [
        {"page": 1, "table_count": 0, "text_chars": 100},
        {"page": 5, "table_count": 3, "text_chars": 1000},
        {"page": 10, "table_count": 0, "text_chars": 10},
    ]
    claim_pages, selected = select_pages(page_count=10, claims=claims, page_metrics=metrics, max_question_pages=2)
    assert claim_pages == [2, 8]
    assert set(claim_pages) <= set(selected)


def test_answer_match_rejects_missing_displayed_unit():
    assert not answer_matches("145.7", "145.7 mm")[0]


def test_answer_match_can_inherit_unit_explicitly_stated_in_question():
    passed, checks = answer_matches_in_context("1029 mm", "1029", "What is the length in mm?")
    assert passed
    assert checks["units_inherited_from_question"]
    assert not answer_matches_in_context("26.4 V", "24", "What is the voltage in V?")[0]
    assert not answer_matches_in_context('4 mm 0.16 inches', "4 mm", "What material is used?")[0]


def test_model_identifier_digits_are_not_extracted_as_answer_numbers():
    assert _numbers("GL-FB1000 length is 1029 mm") == ["1029"]
    assert answer_matches_in_context(
        "1029 mm",
        "The length in mm for the GL-FB1000 model is 1029.",
        "What is the length in mm for the GL-FB1000 model?",
    )[0]


def test_batch_local_claim_ids_are_remapped_to_global_ids():
    batch = [{"claim_id": "c9"}, {"claim_id": "c10"}]
    raw = [{"claim_id": "c1", "supported": True}, {"claim_id": "c2", "supported": False}]
    results, mapping = remap_batch_results(batch, raw)
    assert mapping == {"c1": "c9", "c2": "c10"}
    assert [item["claim_id"] for item in results] == ["c9", "c10"]


def test_report_merge_allows_strict_selected_subset():
    documents = [{"document_id": "a", "version_id": "v1"}]
    report = [
        {"document_id": "a", "version_id": "v1", "status": "planned", "metadata": {"x": 1}},
        {"document_id": "b", "version_id": "v2", "status": "failed"},
    ]
    assert _merge_report(documents, report)[0]["metadata_json"] == {"x": 1}


def test_visual_json_disables_thinking_for_every_model():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"ok": true}'}}

    class Client:
        def post(self, _path, *, json, timeout):
            self.payload = json
            self.timeout = timeout
            return Response()

    client = Client()
    body, _usage = _vision_json(
        client=client, model="gemma4:latest", prompt="inspect", image=b"png",
        schema={"type": "object"}, timeout_seconds=10, num_predict=100,
    )
    assert body == {"ok": True}
    assert client.payload["think"] is False


def test_visual_audit_includes_grounded_legacy_metadata_evidence_once():
    metadata = {
        "metadata_claims": [{
            "kind": "product_model", "value": "AX-100", "relation": "primary_product",
            "subject": None, "page_from": 1, "grounded": True, "verification_status": "confirmed",
        }],
        "metadata_evidence": [
            {
                "kind": "product_model", "value": "AX-100", "relation": "primary_product",
                "subject": None, "page_from": 1, "grounded": True, "verification_status": "confirmed",
            },
            {
                "kind": "document_title", "value": "AX-100 Manual", "relation": "printed_title",
                "subject": None, "page_from": 1, "grounded": True,
            },
        ],
    }
    assert [(item["kind"], item["value"]) for item in _claims(metadata)] == [
        ("product_model", "AX-100"), ("document_title", "AX-100 Manual")
    ]
