from pathlib import Path

import fitz

from manuals_rag_answering.generator import (
    _comparison_answer_covers_retrieved_model_sides,
    _fallback_evidence_results,
    _parse_relevance_response,
    generate_answer,
    generate_answer_with_trace,
    judge_retrieval_relevance,
    prioritize_results_for_answer,
    summarize_results_for_answer,
    validate_answer,
)
from manuals_rag_parsers.docling_parser import (
    _classify_block,
    _docling_page_batches,
    _docling_pipeline_options,
    _docling_table_blocks,
    _docling_table_child_refs,
    _docling_text_blocks,
    _is_ignorable_block,
    _looks_like_heading,
    parse_document,
    _resolved_page_no,
)
from manuals_rag_schemas.documents import AnswerResponse, SearchResult
from manuals_rag_schemas.enums import NodeType, ParseProfile
from tests.helpers import tmp_eval_small_pdf_path


def test_classify_block_detects_spec_and_table():
    spec_type, spec_meta = _classify_block("Voltage: 24 V")
    assert spec_type == NodeType.spec
    assert spec_meta["spec_name"] == "Voltage"

    table_type, table_meta = _classify_block("Item | Value\nCurrent | 1 A")
    assert table_type == NodeType.table
    assert table_meta["table_json"]["headers"] == ["Item", "Value"]


def test_parser_ignores_docling_image_placeholders_and_asset_names():
    assert _is_ignorable_block("<!-- image -->")
    assert _is_ignorable_block("ca_en100u_dimension_01.gif")

    node_type, metadata = _classify_block("ca: en100u_dimension_01.gif")
    assert node_type == NodeType.paragraph
    assert "spec_name" not in metadata


def test_parser_does_not_treat_urls_or_page_markers_as_specs():
    node_type, metadata = _classify_block("https: //www.keyence.com")
    assert node_type == NodeType.paragraph
    assert "spec_name" not in metadata

    node_type, metadata = _classify_block("Page 1 of 8")
    assert node_type == NodeType.paragraph
    assert "spec_name" not in metadata


def test_parser_ignores_low_signal_headings():
    assert _looks_like_heading("Y") is False
    assert _looks_like_heading(":") is False
    assert _looks_like_heading("5") is False


def test_validate_answer_adds_conflict_warning_and_citation():
    answer = AnswerResponse(
        answer="Use the latest revision.",
        confidence="medium",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Setup"],
            content="First revision content",
            metadata={"chunk_type": "procedure_record"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v2",
            source_document_id="d1",
            pages=[2],
            section_path=["Setup"],
            content="Second revision content",
            metadata={"chunk_type": "procedure_record"},
        ),
    ]
    validated = validate_answer(answer, results)
    assert validated.citations
    assert any("multiple document versions" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_when_generated_answer_is_not_supported():
    answer = AnswerResponse(
        answer="Step 1: Images captured when Track Object is enabled.",
        confidence="medium",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Specify other conditions for the Defect tool as required.",
            metadata={"chunk_type": "atomic_text"},
        ),
    ]
    validated = validate_answer(answer, results)
    assert validated.answer == "Defect Tool"
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_direct_configuration_fallback_prefers_phrase_bound_evidence():
    answer = AnswerResponse(
        answer="Configure OPC-UA security certificates.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="broad-preparation",
            score=0.95,
            title="XG-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[970],
            section_path=["Preparing a Line Scan Camera"],
            content=(
                "Preparing a Line Scan Camera. Preparation 1: Changing the Camera, Trigger, "
                "and Light Settings. Configure the following settings when using fixed capture."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="camera-trigger-light-settings",
            score=0.9,
            title="XG-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[979],
            section_path=["Capture Using Line Scan Cameras"],
            content=(
                "Camera: Trigger - Light Configuration Settings. The connected cameras and "
                "illumination expansion units, trigger input for each camera, and illumination "
                "control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="For XG-X line-scan camera setup, what camera configuration area is used for trigger and light settings?",
    )

    assert "Camera: Trigger - Light Configuration Settings" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == [
        "camera-trigger-light-settings"
    ]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_fails_closed_when_plausible_answer_has_only_wrong_mode_evidence():
    answer = AnswerResponse(
        answer="Use Camera: Trigger - Light Configuration Settings for trigger and light settings.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="wrong-mode-config",
            score=0.92,
            title="XG-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[985],
            section_path=["Capture Using Line Scan Cameras", "LumiTrax Specular Reflection Mode"],
            content=(
                "Capture Using Line Scan Cameras (LumiTrax Specular Reflection Mode). "
                "Camera: Trigger - Light Configuration Settings. Configure trigger input "
                "and illumination control targets for LumiTrax line-scan capture."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    validated = validate_answer(
        answer,
        results,
        query=(
            "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
            "configuration area is used for trigger and light settings?"
        ),
    )

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "Camera: Trigger - Light Configuration Settings" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_multi_part_procedure_fallback_keeps_clause_bound_evidence():
    answer = AnswerResponse(
        answer="Configure OPC-UA security certificates.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="async-trigger",
            score=0.91,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[113],
            section_path=["Timing chart"],
            content=(
                "Control/data output via I/O terminals Timing chart. "
                "Typical operations at trigger input, Capture Type: Asynchronous Trigger."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="multi-capture-operation",
            score=0.89,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[114],
            section_path=["Multi-Capture"],
            content=(
                "Typical operations at trigger input, Capture Type: Multi-Capture. "
                "Performs multiple image captures at the same location and processes them as a single measurement."
            ),
            metadata={"chunk_type": "procedure_record"},
        ),
        SearchResult(
            chunk_id="timing-chart",
            score=0.88,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[114],
            section_path=["Timing chart"],
            content="Timing chart Control/data output via I/O terminals.",
            metadata={"chunk_type": "procedure_record"},
        ),
        SearchResult(
            chunk_id="ljv-continuous-timing",
            score=0.87,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[125],
            section_path=["LJ-V continuous timing"],
            content=(
                "Control/data output via I/O terminals Timing chart. "
                "Typical operations at trigger input when the LJ-V series head is used, "
                "[Continuous] is set, and [Total Number of Lines] is enabled."
            ),
            metadata={"chunk_type": "procedure_record"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="For CV-X Multi-Capture trigger input timing, what operation does the section describe and which control/data I/O timing chart should I use?",
    )

    assert [citation["chunk_id"] for citation in validated.citations] == [
        "multi-capture-operation",
        "timing-chart",
    ]
    assert "LJ-V series head" not in validated.answer
    assert "Asynchronous Trigger" not in validated.answer
    assert "Multi-Capture" in validated.answer
    assert "Control/data output via I/O terminals" in validated.answer


def test_multi_part_procedure_fallback_expands_heading_to_operation_context():
    answer = AnswerResponse(
        answer="Configure OPC-UA security certificates.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="multi-capture-heading",
            score=0.91,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[114],
            section_path=["Multi-Capture"],
            content="Procedure step 2: 2. Typical operations at trigger input (Capture Type: Multi-Capture)",
            metadata={
                "chunk_type": "procedure_record",
                "local_rerank_context": (
                    "Timing chart Control/data output via I/O terminals. "
                    "2. Typical operations at trigger input (Capture Type: Multi-Capture). "
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
        ),
        SearchResult(
            chunk_id="timing-chart",
            score=0.88,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[114],
            section_path=["Timing chart"],
            content="Timing chart Control/data output via I/O terminals.",
            metadata={"chunk_type": "procedure_record"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="For CV-X Multi-Capture trigger input timing, what operation does the section describe and which control/data I/O timing chart should I use?",
    )

    assert [citation["chunk_id"] for citation in validated.citations] == ["multi-capture-heading"]
    assert "Performs multiple image captures" in validated.answer
    assert "single measurement" in validated.answer


def test_procedure_fallback_prefers_persisted_citation_context():
    answer = AnswerResponse(
        answer="Configure OPC-UA security certificates.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="multi-capture-heading",
            score=0.91,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[114],
            section_path=["Multi-Capture"],
            content="Procedure step 2: 2. Typical operations at trigger input (Capture Type: Multi-Capture)",
            metadata={
                "chunk_type": "procedure_record",
                "content": (
                    "Procedure step 2: 2. Typical operations at trigger input (Capture Type: Multi-Capture)\n\n"
                    "Timing chart Control/data output via I/O terminals.\n\n"
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="For CV-X Multi-Capture trigger input timing, what operation does the section describe and which control/data I/O timing chart should I use?",
    )

    assert [citation["chunk_id"] for citation in validated.citations] == ["multi-capture-heading"]
    assert "Control/data output via I/O terminals" in validated.answer
    assert "Performs multiple image captures" in validated.answer


def test_direct_procedure_fallback_still_uses_top_evidence_only():
    results = [
        SearchResult(
            chunk_id="top-procedure",
            score=0.91,
            title="Manual",
            document_version_id="v1",
            source_document_id="doc-1",
            pages=[5],
            section_path=["Setup"],
            content="Set the controller mode to Run before starting inspection.",
            metadata={"chunk_type": "procedure_record"},
        ),
        SearchResult(
            chunk_id="nearby-procedure",
            score=0.89,
            title="Manual",
            document_version_id="v1",
            source_document_id="doc-1",
            pages=[6],
            section_path=["Setup"],
            content="Set the controller mode to Program before editing tools.",
            metadata={"chunk_type": "procedure_record"},
        ),
    ]

    selected = _fallback_evidence_results(
        "Which controller mode is required before starting inspection?",
        results,
    )

    assert [result.chunk_id for result in selected] == ["top-procedure"]


def test_image_capture_buffer_fallback_prefers_disabled_trigger_rule():
    results = [
        SearchResult(
            chunk_id="branching-section",
            score=0.94,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[600, 601],
            section_path=["Image Capture Buffer"],
            content=(
                "When capture priorities are different after branching, trigger input can be allowed "
                "or prohibited depending on the capture unit reached after branching."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="disabled-trigger-rule",
            score=0.88,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[596, 597],
            section_path=["Image Capture Buffer"],
            content=(
                "Examples of How the Image Capture Buffer is Used. 1. Image Capture Buffer: Disabled. "
                "If the image capture buffer is disabled, trigger input is only permitted while the flow "
                "is stopped at the capture unit and trigger input is prohibited at any other time. "
                "Receiving trigger input Prohibited Prohibited Prohibited Prohibited Allowed."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    selected = _fallback_evidence_results(
        "With the image capture buffer disabled, are trigger inputs allowed while capture is in progress?",
        results,
    )

    assert [result.chunk_id for result in selected] == ["disabled-trigger-rule"]


def test_image_capture_buffer_fallback_binds_same_priority_condition():
    results = [
        SearchResult(
            chunk_id="multi-capture-branch",
            score=0.94,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[599, 600],
            section_path=["Image Capture Buffer"],
            content=(
                "When multiple capture units are used, the branch destination can receive trigger "
                "inputs for another camera if the first capture unit passes."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="disabled-buffer-context",
            score=0.91,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[596, 597],
            section_path=["Image Capture Buffer"],
            content=(
                "Examples of How the Image Capture Buffer is Used. 1. Image Capture Buffer: Disabled. "
                "If the image capture buffer is disabled, trigger input is only permitted while the flow "
                "is stopped at the capture unit."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="same-priority-condition",
            score=0.82,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[597],
            section_path=["Image Capture Buffer"],
            content="Using only one camera or multiple cameras that all use the same capture priority condition.",
            metadata={"chunk_type": "atomic_text"},
        ),
    ]

    selected = _fallback_evidence_results(
        "With image capture buffer disabled, can it be used with one camera or multiple cameras sharing the same capture-priority condition?",
        results,
    )

    assert [result.chunk_id for result in selected] == [
        "same-priority-condition",
        "disabled-buffer-context",
    ]


def test_image_capture_buffer_fallback_uses_source_parent_context_for_atomic_condition():
    results = [
        SearchResult(
            chunk_id="same-priority-condition",
            score=0.95,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[597],
            section_path=["Image Capture Buffer"],
            content="Using only one camera or multiple cameras that all use the same capture priority condition.",
            metadata={
                "chunk_type": "atomic_text",
                "parent_context": (
                    "Examples of How the Image Capture Buffer is Used. "
                    "Using only one camera or multiple cameras that all use the same capture priority condition. "
                    "1. Image Capture Buffer: Disabled. "
                    "If the image capture buffer is disabled, trigger input is prohibited while capture is in progress."
                ),
            },
        )
    ]

    selected = _fallback_evidence_results(
        "With image capture buffer disabled, can it be used with one camera or multiple cameras sharing the same capture-priority condition?",
        results,
    )

    assert [result.chunk_id for result in selected] == ["same-priority-condition"]


def test_image_capture_buffer_fallback_rejects_metadata_only_disabled_state():
    results = [
        SearchResult(
            chunk_id="same-priority-condition",
            score=0.95,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[597],
            section_path=["Image Capture Buffer"],
            content="Using only one camera or multiple cameras that all use the same capture priority condition.",
            metadata={
                "chunk_type": "atomic_text",
                "content": (
                    "Using only one camera or multiple cameras that all use the same capture priority condition\n\n"
                    "Step 1: 1. Image Capture Buffer: Disabled"
                ),
            },
        )
    ]

    selected = _fallback_evidence_results(
        "With image capture buffer disabled, can it be used with one camera or multiple cameras sharing the same capture-priority condition?",
        results,
    )

    assert selected == []


def test_validate_answer_falls_back_when_image_buffer_state_is_omitted():
    query = (
        "With image capture buffer disabled, can it be used with one camera or multiple "
        "cameras sharing the same capture-priority condition?"
    )
    answer = AnswerResponse(
        answer="Using only one camera or multiple cameras that all use the same capture priority condition.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "same-priority-condition",
                "document_id": "doc-x",
                "pages": [597],
                "quote_span": None,
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="same-priority-condition",
            score=0.95,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="doc-x",
            pages=[597],
            section_path=["Image Capture Buffer"],
            content="Using only one camera or multiple cameras that all use the same capture priority condition.",
            metadata={
                "chunk_type": "atomic_text",
                "content": (
                    "Using only one camera or multiple cameras that all use the same capture priority condition\n\n"
                    "Step 1: 1. Image Capture Buffer: Disabled"
                ),
            },
        )
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "disabled" not in validated.answer.lower()
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_when_citation_quote_is_not_in_cited_chunk():
    answer = AnswerResponse(
        answer="Change the trigger signal assignment to one that can be used.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "settings-page",
                "document_id": "d1",
                "pages": [10],
                "quote_span": "Change the trigger signal assignment to one that can be used.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="settings-page",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Trigger settings"],
            content="If capture on trigger input is disabled, this setting cannot be changed.",
            metadata={"chunk_type": "atomic_text"},
        )
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "If capture on trigger input is disabled, this setting cannot be changed."
    assert validated.citations[0]["quote_span"] is None
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_does_not_repair_unsupported_citation_from_answer_overlap():
    answer = AnswerResponse(
        answer="Change to a trigger signal that can be used.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "settings-page",
                "document_id": "d1",
                "pages": [10],
                "quote_span": "Change to a trigger signal that can be used.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="settings-page",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Trigger settings"],
            content="If capture on trigger input is disabled, this setting cannot be changed.",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="action-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Corrective Action: Change to a trigger signal that can be used.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "If capture on trigger input is disabled, this setting cannot be changed."
    assert validated.citations[0]["chunk_id"] == "settings-page"
    assert validated.citations[0]["quote_span"] is None
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_keeps_supported_citation_and_drops_unsupported_sibling_quote():
    answer = AnswerResponse(
        answer=(
            "Use the row for multiple trigger lines: set the line scan cameras or LJ-X/LJ-V heads "
            "assigned to the same trigger to the same [No. Lines], or assign the other camera to a different trigger."
        ),
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "line-count-row",
                "document_id": "d1",
                "pages": [12],
                "quote_span": (
                    "be sure that [No. Lines] of the line scan cameras or LJ-X/LJ-V series head "
                    "assigned to the same trigger are the same."
                ),
            },
            {
                "chunk_id": "sibling-row",
                "document_id": "d1",
                "pages": [12],
                "quote_span": (
                    "be sure the line scan camera or LJ-X/LJ-V series head capture methods "
                    "assigned to the same trigger are the same."
                ),
            },
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="line-count-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Error Message: Image capture stopped, invalid camera setting. A trigger has multiple lines.; "
                "Cause: Multiple line Nos. are included in the same trigger.; "
                "Corrective Action: In the Capture unit camera settings, be sure that [No. Lines] "
                "of the line scan cameras or LJ-X/LJ-V series head assigned to the same trigger are the same. "
                "Or, assign the other camera which is assigned to the same trigger to a different trigger."
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="sibling-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content="Corrective Action: Change capture methods to sheet-fed.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert [citation["chunk_id"] for citation in validated.citations] == ["line-count-row"]
    assert not any("Unsupported citation quote spans were removed" in warning for warning in validated.warnings)
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_instead_of_pruning_unsupported_citation():
    answer = AnswerResponse(
        answer="Set voltage to 5 volts.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "voltage-row",
                "document_id": "d1",
                "pages": [7],
                "quote_span": "Set voltage to 5 volts.",
            },
            {
                "chunk_id": "nearby-row",
                "document_id": "d1",
                "pages": [7],
                "quote_span": "Disable encryption.",
            },
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="voltage-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["Settings"],
            content="Corrective Action: Set voltage to 5 volts.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="nearby-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["Settings"],
            content="Corrective Action: Check the network cable.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "Corrective Action: Set voltage to 5 volts."
    assert [citation["chunk_id"] for citation in validated.citations] == ["voltage-row"]
    assert not any("Unsupported citation quote spans were removed" in warning for warning in validated.warnings)
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_when_pruned_citations_do_not_support_all_claims():
    answer = AnswerResponse(
        answer="Set voltage to 5 volts. Disable encryption.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "voltage-row",
                "document_id": "d1",
                "pages": [7],
                "quote_span": "Set voltage to 5 volts.",
            },
            {
                "chunk_id": "nearby-row",
                "document_id": "d1",
                "pages": [7],
                "quote_span": "Disable encryption.",
            },
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="voltage-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["Settings"],
            content="Corrective Action: Set voltage to 5 volts.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="nearby-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["Settings"],
            content="Corrective Action: Check the network cable.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "Corrective Action: Set voltage to 5 volts."
    assert "Disable encryption" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["voltage-row"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_swapped_numeric_bindings_after_bad_citation():
    answer = AnswerResponse(
        answer="Set voltage to 5 volts and current to 10 amps.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "swapped-row",
                "document_id": "d1",
                "pages": [7],
                "quote_span": "Set voltage to 10 volts and current to 5 amps.",
            },
            {
                "chunk_id": "nearby-row",
                "document_id": "d1",
                "pages": [7],
                "quote_span": "Set voltage to 5 volts and current to 10 amps.",
            },
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="swapped-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["Settings"],
            content="Corrective Action: Set voltage to 10 volts and current to 5 amps.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="nearby-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["Settings"],
            content="Corrective Action: Check the network cable.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "Corrective Action: Set voltage to 10 volts and current to 5 amps."
    assert "5 volts and current to 10 amps" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["swapped-row"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_polarity_inversion_after_bad_citation():
    answer = AnswerResponse(
        answer="Enable remote start before maintenance.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "safety-row",
                "document_id": "d1",
                "pages": [9],
                "quote_span": "Disable remote start before maintenance.",
            },
            {
                "chunk_id": "nearby-row",
                "document_id": "d1",
                "pages": [9],
                "quote_span": "Enable remote start before maintenance.",
            },
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="safety-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[9],
            section_path=["Safety"],
            content="Corrective Action: Disable remote start before maintenance.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="nearby-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[9],
            section_path=["Safety"],
            content="Corrective Action: Inspect the indicator.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "Corrective Action: Disable remote start before maintenance."
    assert "Enable remote start" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["safety-row"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_cross_chunk_role_mixing_after_bad_citation():
    answer = AnswerResponse(
        answer="Set voltage to 5 volts and current to 10 amps.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "voltage-row",
                "document_id": "d1",
                "pages": [11],
                "quote_span": "Set voltage to 5 volts.",
            },
            {
                "chunk_id": "current-row",
                "document_id": "d1",
                "pages": [12],
                "quote_span": "Set current to 10 amps.",
            },
            {
                "chunk_id": "nearby-row",
                "document_id": "d1",
                "pages": [12],
                "quote_span": "Set voltage to 5 volts and current to 10 amps.",
            },
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="voltage-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[11],
            section_path=["Voltage setup"],
            content="Corrective Action: Set voltage to 5 volts.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="current-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Current setup"],
            content="Corrective Action: Set current to 10 amps.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="nearby-row",
            score=0.7,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Current setup"],
            content="Corrective Action: Check the output terminal.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "Corrective Action: Set voltage to 5 volts."
    assert "current to 10 amps" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["voltage-row"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_comparison_fallback_uses_multiple_structured_rows():
    answer = AnswerResponse(
        answer="The first controller is IP67 and the second controller is rated for shock.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "wrong-neighbor",
                "document_id": "d2",
                "pages": [12],
                "quote_span": "500 m/s2, 6 directions",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="wrong-neighbor",
            score=0.95,
            title="Controller B Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[12],
            section_path=["Specifications"],
            content="Column headers: Controller; Row headers: Enclosure rating; Cell value: IP67",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="enclosure-row",
            score=0.9,
            title="Controller A Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Specifications"],
            content="Column headers: Controller; Row headers: Enclosure rating; Cell value: IP67",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="shock-row",
            score=0.8,
            title="Controller B Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[12],
            section_path=["Specifications"],
            content="Column headers: Controller; Row headers: Shock resistance; Cell value: 500 m/s2, 6 directions",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="Compare Controller A enclosure rating and Controller B shock resistance.",
    )

    assert "Retrieved evidence:" in validated.answer
    assert "Enclosure rating" in validated.answer
    assert "Shock resistance" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == [
        "wrong-neighbor",
        "enclosure-row",
        "shock-row",
    ]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_comparison_troubleshooting_fallback_prefers_side_specific_symptom_rows():
    query = (
        "Compare the corrective action for unstable gray-binary inspection on CV-X482 "
        "with the XG-X guidance for an unsupported SD card access failure."
    )
    results = [
        SearchResult(
            chunk_id="cvx-sibling",
            score=0.95,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[507],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective action; Row headers: Inspection is not stable in color extraction.; "
                "Cell value: Select Color to Grayscale in Extract Colors."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "CV-X482",
                "product_family": "CV-X Series",
            },
        ),
        SearchResult(
            chunk_id="cvx-gray-binary",
            score=0.8,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[507],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective action; Row headers: Inspection is not stable in gray binary.; "
                "Cell value: Select Color to Binary in Extract Colors and extract the desired colors."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "CV-X482",
                "product_family": "CV-X Series",
            },
        ),
        SearchResult(
            chunk_id="xgx-unsupported-card",
            score=0.7,
            title="XG-X Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[1262],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective Action; Row headers: Failed to access SD Card 1. "
                "> An unsupported SD card is being used.; Cell value: KEYENCE does not guarantee operation "
                "with commercially available SD cards."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "XG-X Series",
            },
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert [result.chunk_id for result in selected[:2]] == ["cvx-gray-binary", "xgx-unsupported-card"]
    assert "Color to Grayscale" not in "\n".join(result.content for result in selected[:2])


def test_configuration_fallback_keeps_same_mode_configuration_evidence():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    results = [
        SearchResult(
            chunk_id="broad-standard-mode",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[973],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Change the settings in accordance with the capture environment."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="standard-config",
            score=0.8,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[981],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert selected[0].chunk_id == "standard-config"


def test_configuration_fallback_keeps_same_scope_setup_context_with_direct_evidence():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    results = [
        SearchResult(
            chunk_id="standard-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[981],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="standard-setup-step",
            score=0.8,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[973],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Change the settings in accordance with the capture environment on the "
                "Line Camera Setting Navigation screen."
            ),
            metadata={"chunk_type": "procedure_record"},
        ),
        SearchResult(
            chunk_id="wrong-mode-setup",
            score=0.85,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1001],
            section_path=["LumiTrax Specular Reflection Mode"],
            content=(
                "Capture Using Line Scan Cameras (LumiTrax Specular Reflection Mode). "
                "Change the settings in accordance with the capture environment."
            ),
            metadata={"chunk_type": "procedure_record"},
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert [result.chunk_id for result in selected] == ["standard-config", "standard-setup-step"]


def test_configuration_fallback_prefers_direct_label_over_broad_mode_context():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    results = [
        SearchResult(
            chunk_id="broad-standard-mode",
            score=0.95,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[973],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Preparation 1: Changing the Camera, Trigger, and Light Settings. "
                "Configure the following settings when using fixed capture."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="standard-config",
            score=0.8,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[981],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert selected[0].chunk_id == "standard-config"


def test_configuration_fallback_does_not_promote_wrong_mode_configuration_evidence():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    results = [
        SearchResult(
            chunk_id="wrong-mode-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1001],
            section_path=["LumiTrax Specular Reflection Mode"],
            content=(
                "Capture Using Line Scan Cameras (LumiTrax Specular Reflection Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="standard-mode-context",
            score=0.8,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[973],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Change the settings in accordance with the capture environment on the Capture Environment screen."
            ),
            metadata={"chunk_type": "procedure_record"},
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert selected == []


def test_configuration_fallback_returns_no_evidence_for_only_wrong_mode_candidate():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    results = [
        SearchResult(
            chunk_id="wrong-mode-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1001],
            section_path=["LumiTrax Specular Reflection Mode"],
            content=(
                "Capture Using Line Scan Cameras (LumiTrax Specular Reflection Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    selected = _fallback_evidence_results(query, results)

    assert selected == []


def test_configuration_fallback_rejects_direct_label_without_trigger_light_details():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    results = [
        SearchResult(
            chunk_id="label-only",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[981],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    selected = _fallback_evidence_results(query, results)

    assert selected == []


def test_configuration_fallback_rejects_same_mode_wrong_capture_type():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    results = [
        SearchResult(
            chunk_id="standard-area-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[944],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Area Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="standard-line-config",
            score=0.8,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[981],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert [result.chunk_id for result in selected] == ["standard-line-config"]


def test_validate_answer_fails_closed_for_direct_label_without_trigger_light_details():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    answer = AnswerResponse(
        answer=(
            "Use Camera: Trigger - Light Configuration Settings. The trigger input for each camera "
            "and illumination control targets can be configured there."
        ),
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="label-only",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[981],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "trigger input for each camera" not in validated.answer
    assert "illumination control targets" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_fails_closed_for_same_mode_wrong_capture_type():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    answer = AnswerResponse(
        answer=(
            "Use Camera: Trigger - Light Configuration Settings. The trigger input for each camera "
            "and illumination control targets can be configured there."
        ),
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="standard-area-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[944],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Area Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "trigger input for each camera" not in validated.answer
    assert "illumination control targets" not in validated.answer


def test_validate_answer_fallback_is_insufficient_for_only_wrong_mode_candidate():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    answer = AnswerResponse(
        answer="Use OPC-UA security certificate settings.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="wrong-mode-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1001],
            section_path=["LumiTrax Specular Reflection Mode"],
            content=(
                "Capture Using Line Scan Cameras (LumiTrax Specular Reflection Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "LumiTrax" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_keeps_direct_configuration_with_trigger_light_details():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    answer = AnswerResponse(
        answer=(
            "Use Camera: Trigger - Light Configuration Settings. The trigger input for each camera "
            "and illumination control targets can be configured there."
        ),
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="standard-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[981],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Camera: Trigger - Light Configuration Settings. "
                "The trigger input for each camera and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is False
    assert [citation["chunk_id"] for citation in validated.citations] == ["standard-config"]
    assert "trigger input for each camera" in validated.answer
    assert "illumination control targets" in validated.answer


def test_validate_answer_fails_closed_for_broad_same_mode_configuration_context():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    answer = AnswerResponse(
        answer="Use Camera: Trigger - Light Configuration Settings for trigger and light settings.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="broad-standard-mode",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[973],
            section_path=["Standard Lighting Mode"],
            content=(
                "Capture Using Line Scan Cameras (Standard Lighting Mode). "
                "Preparation 1: Changing the Camera, Trigger, and Light Settings. "
                "Configure the following settings when using fixed capture."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "Camera: Trigger - Light Configuration Settings" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_fails_closed_for_conflicting_mode_configuration_context():
    query = (
        "For XG-X Standard Lighting Mode line-scan setup, which Camera-Trigger-Light "
        "configuration area is used for trigger and light settings?"
    )
    answer = AnswerResponse(
        answer="Use Camera: Trigger - Light Configuration Settings for trigger and light settings.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="lumitrax-config",
            score=0.9,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1001],
            section_path=["LumiTrax Specular Reflection Mode"],
            content=(
                "Capture Using Line Scan Cameras (LumiTrax Specular Reflection Mode). "
                "Camera: Trigger - Light Configuration Settings. The trigger input for each camera "
                "and illumination control targets can be configured together."
            ),
            metadata={
                "chunk_type": "section_window",
                "local_rerank_context": "Capture Using Line Scan Cameras (Standard Lighting Mode).",
            },
        )
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "Camera: Trigger - Light Configuration Settings" not in validated.answer


def test_comparison_troubleshooting_fallback_rejects_wrong_sibling_answer():
    query = (
        "Compare the corrective action for unstable gray-binary inspection on CV-X482 "
        "with the XG-X guidance for an unsupported SD card access failure."
    )
    answer = AnswerResponse(
        answer=(
            "For CV-X482, select Color to Grayscale in Extract Colors. "
            "For XG-X, KEYENCE does not guarantee operation with commercially available SD cards."
        ),
        confidence="high",
        used_documents=[],
        citations=[
            {"chunk_id": "cvx-sibling", "document_id": "d1", "pages": [507], "quote_span": None},
            {"chunk_id": "xgx-unsupported-card", "document_id": "d2", "pages": [1262], "quote_span": None},
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="cvx-sibling",
            score=0.95,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[507],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective action; Row headers: Inspection is not stable in color extraction.; "
                "Cell value: Select Color to Grayscale in Extract Colors."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "CV-X482",
                "product_family": "CV-X Series",
            },
        ),
        SearchResult(
            chunk_id="cvx-gray-binary",
            score=0.8,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[507],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective action; Row headers: Inspection is not stable in gray binary.; "
                "Cell value: Select Color to Binary in Extract Colors and extract the desired colors."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "CV-X482",
                "product_family": "CV-X Series",
            },
        ),
        SearchResult(
            chunk_id="xgx-unsupported-card",
            score=0.7,
            title="XG-X Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[1262],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective Action; Row headers: Failed to access SD Card 1. "
                "> An unsupported SD card is being used.; Cell value: KEYENCE does not guarantee operation "
                "with commercially available SD cards."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "XG-X Series",
            },
        ),
    ]

    validated = validate_answer(answer, results, query=query)

    assert "Color to Binary" in validated.answer
    assert "Color to Grayscale" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations[:2]] == [
        "cvx-gray-binary",
        "xgx-unsupported-card",
    ]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_comparison_troubleshooting_fallback_rejects_partial_side_evidence_answer():
    query = (
        "Compare the corrective action for unstable gray-binary inspection on CV-X482 "
        "with the XG-X guidance for an unsupported SD card access failure."
    )
    answer = AnswerResponse(
        answer=(
            "For CV-X482, select Color to Binary in Extract Colors. "
            "For XG-X, KEYENCE does not guarantee operation with commercially available SD cards."
        ),
        confidence="high",
        used_documents=[],
        citations=[
            {"chunk_id": "cvx-gray-binary", "document_id": "d1", "pages": [507], "quote_span": None},
            {"chunk_id": "xgx-unrelated", "document_id": "d2", "pages": [1260], "quote_span": None},
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="cvx-gray-binary",
            score=0.9,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[507],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective action; Row headers: Inspection is not stable in gray binary.; "
                "Cell value: Select Color to Binary in Extract Colors and extract the desired colors."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "CV-X482",
                "product_family": "CV-X Series",
            },
        ),
        SearchResult(
            chunk_id="xgx-unrelated",
            score=0.8,
            title="XG-X Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[1260],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective Action; Row headers: A global variable cannot be read.; "
                "Cell value: Check the variable name and PLC communication settings."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "XG-X Series",
            },
        ),
    ]

    validated = validate_answer(answer, results, query=query)

    assert "Color to Binary" in validated.answer
    assert "commercially available SD cards" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["cvx-gray-binary"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_comparison_fallback_replaces_overcautious_insufficient_answer():
    answer = AnswerResponse(
        answer="The documents do not distinguish the causes for the two controllers.",
        confidence="low",
        used_documents=[],
        citations=[],
        warnings=["Evidence appears incomplete."],
        followup_questions=[],
        insufficient_evidence=True,
    )
    results = [
        SearchResult(
            chunk_id="controller-a-cause",
            score=0.9,
            title="Controller A Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[20],
            section_path=["Troubleshooting"],
            content="Column headers: Cause; Row headers: Startup memory read error; Cell value: Noise or power switched OFF during writing.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="controller-b-cause",
            score=0.8,
            title="Controller B Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[30],
            section_path=["Troubleshooting"],
            content="Column headers: Cause; Row headers: Startup memory read error; Cell value: A data error occurred.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="Compare the listed causes for startup memory read errors on Controller A and Controller B.",
    )

    assert validated.insufficient_evidence is False
    assert "Noise or power switched OFF" in validated.answer
    assert "A data error occurred" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["controller-a-cause", "controller-b-cause"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_comparison_fallback_replaces_overcautious_text_without_flag():
    answer = AnswerResponse(
        answer=(
            "The evidence defines the first controller cause. The evidence does not contain "
            "information for the second controller, so a comparison cannot be made."
        ),
        confidence="medium",
        used_documents=[],
        citations=[
            {
                "chunk_id": "controller-a-cause",
                "document_id": "d1",
                "pages": [20],
                "quote_span": None,
            }
        ],
        warnings=["Evidence does not contain information for the second controller."],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="controller-a-cause",
            score=0.9,
            title="Controller A Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[20],
            section_path=["Troubleshooting"],
            content="Column headers: Cause; Row headers: Startup error; Cell value: Program data is invalid.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="controller-b-cause",
            score=0.8,
            title="Controller B Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[30],
            section_path=["Troubleshooting"],
            content="Column headers: Cause; Row headers: Startup error; Cell value: Power was interrupted during writing.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="Compare the startup error causes for Controller A and Controller B.",
    )

    assert validated.insufficient_evidence is False
    assert "Program data is invalid" in validated.answer
    assert "Power was interrupted" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["controller-a-cause", "controller-b-cause"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_comparison_fallback_when_generated_answer_cites_only_one_model_side():
    answer = AnswerResponse(
        answer="The IV4-G600CA cause is a memory read error when the sensor starts.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "iv4-cause",
                "document_id": "d-iv4",
                "pages": [532],
                "quote_span": None,
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="iv4-cause",
            score=0.9,
            title="IV4-G600CA Manual",
            document_version_id="v1",
            source_document_id="d-iv4",
            pages=[532],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Cause; Row headers: Failed to read nonvolatile memory at sensor startup; "
                "Cell value: A memory read error occurred when the sensor started."
            ),
            metadata={"chunk_type": "table_record", "product_model": "IV4-G600CA"},
        ),
        SearchResult(
            chunk_id="ivh-cause",
            score=0.8,
            title="IV-HG500CA Manual",
            document_version_id="v1",
            source_document_id="d-ivh",
            pages=[406],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Cause; Row headers: Sensor program damaged. Initialization necessary.; "
                "Cell value: A memory read error occurred when the sensor started."
            ),
            metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query=(
            "For IV-HG500CA, what cause is listed when the sensor program is damaged and initialization "
            "is necessary, and how does that differ from the IV4-G600CA startup memory read error cause?"
        ),
    )

    assert "Retrieved evidence:" in validated.answer
    assert "IV4-G600CA" in validated.answer
    assert "IV-HG500CA" in validated.answer
    assert {citation["chunk_id"] for citation in validated.citations} == {"iv4-cause", "ivh-cause"}
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_comparison_side_coverage_rejects_unrelated_only_citations():
    results = [
        SearchResult(
            chunk_id="ivh-cause",
            score=0.9,
            title="IV-HG500CA Manual",
            document_version_id="v1",
            source_document_id="d-ivh",
            pages=[10],
            section_path=["Troubleshooting"],
            content="Model IV-HG500CA cause: Sensor program data is damaged.",
            metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
        ),
        SearchResult(
            chunk_id="iv4-cause",
            score=0.8,
            title="IV4-G600CA Manual",
            document_version_id="v1",
            source_document_id="d-iv4",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Model IV4-G600CA cause: A startup memory read error occurred.",
            metadata={"chunk_type": "table_record", "product_model": "IV4-G600CA"},
        ),
        SearchResult(
            chunk_id="xg-row",
            score=0.7,
            title="XG-X1000 Manual",
            document_version_id="v1",
            source_document_id="d-xg",
            pages=[12],
            section_path=["Troubleshooting"],
            content="Model XG-X1000 cause: The image capture setting is invalid.",
            metadata={"chunk_type": "table_record", "product_model": "XG-X1000"},
        ),
    ]

    assert (
        _comparison_answer_covers_retrieved_model_sides(
            "Compare IV-HG500CA and IV4-G600CA startup error causes.",
            [{"chunk_id": "xg-row", "document_id": "d-xg", "pages": [12], "quote_span": None}],
            results,
        )
        is False
    )


def test_comparison_side_coverage_rejects_one_available_side_only():
    results = [
        SearchResult(
            chunk_id="ivh-cause",
            score=0.9,
            title="IV-HG500CA Manual",
            document_version_id="v1",
            source_document_id="d-ivh",
            pages=[10],
            section_path=["Troubleshooting"],
            content="Model IV-HG500CA cause: Sensor program data is damaged.",
            metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
        ),
        SearchResult(
            chunk_id="iv4-cause",
            score=0.8,
            title="IV4-G600CA Manual",
            document_version_id="v1",
            source_document_id="d-iv4",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Model IV4-G600CA cause: A startup memory read error occurred.",
            metadata={"chunk_type": "table_record", "product_model": "IV4-G600CA"},
        ),
    ]

    assert (
        _comparison_answer_covers_retrieved_model_sides(
            "Compare IV-HG500CA and IV4-G600CA startup error causes.",
            [{"chunk_id": "ivh-cause", "document_id": "d-ivh", "pages": [10], "quote_span": None}],
            results,
        )
        is False
    )


def test_comparison_side_coverage_accepts_both_available_sides():
    results = [
        SearchResult(
            chunk_id="ivh-cause",
            score=0.9,
            title="IV-HG500CA Manual",
            document_version_id="v1",
            source_document_id="d-ivh",
            pages=[10],
            section_path=["Troubleshooting"],
            content="Model IV-HG500CA cause: Sensor program data is damaged.",
            metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
        ),
        SearchResult(
            chunk_id="iv4-cause",
            score=0.8,
            title="IV4-G600CA Manual",
            document_version_id="v1",
            source_document_id="d-iv4",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Model IV4-G600CA cause: A startup memory read error occurred.",
            metadata={"chunk_type": "table_record", "product_model": "IV4-G600CA"},
        ),
    ]

    assert (
        _comparison_answer_covers_retrieved_model_sides(
            "Compare IV-HG500CA and IV4-G600CA startup error causes.",
            [
                {"chunk_id": "ivh-cause", "document_id": "d-ivh", "pages": [10], "quote_span": None},
                {"chunk_id": "iv4-cause", "document_id": "d-iv4", "pages": [11], "quote_span": None},
            ],
            results,
        )
        is True
    )


def test_comparison_side_coverage_rejects_unavailable_or_unrecognized_side():
    results = [
        SearchResult(
            chunk_id="ivh-cause",
            score=0.9,
            title="IV-HG500CA Manual",
            document_version_id="v1",
            source_document_id="d-ivh",
            pages=[10],
            section_path=["Troubleshooting"],
            content="Model IV-HG500CA cause: Sensor program data is damaged.",
            metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
        )
    ]

    assert (
        _comparison_answer_covers_retrieved_model_sides(
            "Compare IV-HG500CA and IV4-G600CA startup error causes.",
            [{"chunk_id": "ivh-cause", "document_id": "d-ivh", "pages": [10], "quote_span": None}],
            results,
        )
        is False
    )


def test_comparison_side_coverage_rejects_prefix_and_code_false_matches():
    prefix_results = [
        SearchResult(
            chunk_id="iv4-prefix",
            score=0.9,
            title="IV4 Series Manual",
            document_version_id="v1",
            source_document_id="d-iv4",
            pages=[10],
            section_path=["Troubleshooting"],
            content="Model IV4-G120 cause: A startup memory read error occurred.",
            metadata={"chunk_type": "table_record", "product_model": "IV4-G120"},
        ),
        SearchResult(
            chunk_id="ivh-cause",
            score=0.8,
            title="IV-HG500CA Manual",
            document_version_id="v1",
            source_document_id="d-ivh",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Model IV-HG500CA cause: Sensor program data is damaged.",
            metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
        ),
    ]

    assert (
        _comparison_answer_covers_retrieved_model_sides(
            "Compare IV4-G600CA and IV-HG500CA startup error causes.",
            [
                {"chunk_id": "iv4-prefix", "document_id": "d-iv4", "pages": [10], "quote_span": None},
                {"chunk_id": "ivh-cause", "document_id": "d-ivh", "pages": [11], "quote_span": None},
            ],
            prefix_results,
        )
        is False
    )

    code_results = [
        SearchResult(
            chunk_id="error-code",
            score=0.9,
            title="Troubleshooting Codes",
            document_version_id="v1",
            source_document_id="d-code",
            pages=[20],
            section_path=["Troubleshooting"],
            content="Error IV4-G600CA-20503: invalid pattern data.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="ivh-cause",
            score=0.8,
            title="IV-HG500CA Manual",
            document_version_id="v1",
            source_document_id="d-ivh",
            pages=[21],
            section_path=["Troubleshooting"],
            content="Model IV-HG500CA cause: Sensor program data is damaged.",
            metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
        ),
    ]

    assert (
        _comparison_answer_covers_retrieved_model_sides(
            "Compare IV4-G600CA and IV-HG500CA startup error causes.",
            [
                {"chunk_id": "error-code", "document_id": "d-code", "pages": [20], "quote_span": None},
                {"chunk_id": "ivh-cause", "document_id": "d-ivh", "pages": [21], "quote_span": None},
            ],
            code_results,
        )
        is False
    )


def test_summarize_results_keeps_small_structured_evidence_set_separate(monkeypatch):
    results = [
        SearchResult(
            chunk_id=f"row-{index}",
            score=1.0 - index / 10,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[index],
            section_path=["Specs"],
            content=f"Column headers: Value; Row headers: Setting {index}; Cell value: {index}",
            metadata={"chunk_type": "table_record"},
        )
        for index in range(5)
    ]

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "recursive_summary":
            raise AssertionError("small direct structured evidence should not be recursively merged")
        raise AssertionError("direct structured evidence should not call the model")

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    summaries = summarize_results_for_answer("Compare Setting 1 and Setting 4.", results)

    assert [summary["chunk_id"] for summary in summaries] == [f"row-{index}" for index in range(5)]
    assert all(summary["summary_source"] == "direct_evidence" for summary in summaries)


def test_prioritize_results_preserves_comparison_evidence_before_model_pruning(monkeypatch):
    results = [
        SearchResult(
            chunk_id="controller-a-cause",
            score=0.9,
            title="Controller A Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[20],
            section_path=["Troubleshooting"],
            content="Column headers: Cause; Row headers: Startup error; Cell value: Program data is invalid.",
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="controller-b-cause",
            score=0.8,
            title="Controller B Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[30],
            section_path=["Troubleshooting"],
            content="Column headers: Cause; Row headers: Startup error; Cell value: Power was interrupted during writing.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    monkeypatch.setattr(
        "manuals_rag_answering.generator.judge_retrieval_relevance",
        lambda _query, _results: [
            {"chunk_id": "controller-a-cause", "verdict": "relevant", "reason": "Relevant."},
            {"chunk_id": "controller-b-cause", "verdict": "not_relevant", "reason": "Incorrectly pruned."},
        ],
    )

    prioritized = prioritize_results_for_answer(
        "Compare the startup error causes for Controller A and Controller B.",
        results,
    )

    assert [result.chunk_id for result in prioritized["prioritized_results"][:2]] == [
        "controller-a-cause",
        "controller-b-cause",
    ]


def test_prioritize_results_preserves_procedure_rule_evidence_before_model_pruning(monkeypatch):
    results = [
        SearchResult(
            chunk_id="flowchart-rule",
            score=0.9,
            title="XG-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[600],
            section_path=["Image capture buffer"],
            content=(
                "When multiple capture units are used, the passing status of the capture unit "
                "executed before the branch unit must be specified as the branch condition."
            ),
            metadata={"chunk_type": "parent_section"},
        ),
        SearchResult(
            chunk_id="adjacent-precaution",
            score=0.8,
            title="XG-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[601],
            section_path=["Image capture buffer"],
            content=(
                "In older controller series, multiple images captured with the same capture unit "
                "parameters may be processed by a different capture unit."
            ),
            metadata={"chunk_type": "atomic_text"},
        ),
    ]

    monkeypatch.setattr(
        "manuals_rag_answering.generator.judge_retrieval_relevance",
        lambda _query, _results: [
            {"chunk_id": "flowchart-rule", "verdict": "not_relevant", "reason": "Incorrectly pruned."},
            {"chunk_id": "adjacent-precaution", "verdict": "relevant", "reason": "Adjacent topic."},
        ],
    )

    prioritized = prioritize_results_for_answer(
        "For asynchronous capture with multiple capture units, what flowchart branching rule should be followed?",
        results,
    )

    assert prioritized["prioritized_results"][0].chunk_id == "flowchart-rule"


def test_validate_answer_fallback_uses_matching_troubleshooting_row_from_parent_context():
    answer = AnswerResponse(
        answer=(
            "If capture on trigger input is disabled in the trigger settings, this setting cannot be changed."
        ),
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "trigger-settings-section",
                "document_id": "d1",
                "pages": [985],
                "quote_span": None,
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="trigger-settings-section",
            score=0.9,
            title="XG-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1259],
            section_path=["Troubleshooting"],
            content=(
                "Trigger Parameters Capture on trigger input. Choose whether or not the capture unit "
                "will wait for a trigger signal to capture an image."
            ),
            metadata={
                "chunk_type": "section_window",
                "parent_context": (
                    "Image capture stopped, invalid camera setting. A trigger signal that cannot be used is assigned. | "
                    "Trigger signals on the even-number side of the camera input unit to which the LJ-X/LJ-V Series head "
                    "is connected cannot be used. | Change to a trigger signal that can be used. |"
                ),
            },
        ),
        SearchResult(
            chunk_id="generic-trigger-settings",
            score=0.8,
            title="XG-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[985],
            section_path=["Trigger settings"],
            content="If capture on trigger input is disabled in the trigger settings, this setting cannot be changed.",
            metadata={"chunk_type": "section_window"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query=(
            "On an XG-X Series controller, if Allow Trigger Input During Line Capture and End Capture By EXT Signal "
            "are enabled together and the invalid camera setting error says a trigger signal that cannot be used is "
            "assigned, what should I change?"
        ),
    )

    assert "Change to a trigger signal that can be used" in validated.answer
    assert "even-number side" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["trigger-settings-section"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_fallback_selects_returned_quantity_evidence():
    answer = AnswerResponse(
        answer="The timing section says to use the external trigger as the trigger input.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "wrong-line-scan-settings",
                "document_id": "d1",
                "pages": [1063],
                "quote_span": None,
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="wrong-line-scan-settings",
            score=0.9,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1063],
            section_path=["Line Scan Settings"],
            content=(
                "Changing the Interval for Obtaining the Profiles with the LJ-V Series Head. "
                "Trigger uses the external trigger as trigger input."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="continuous-mode-example",
            score=0.8,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[863],
            section_path=["Timing chart"],
            content=(
                "Typical operations at trigger input when the LJ-V series head is used, "
                "[Continuous] is set, and [Total Number of Lines] is enabled. "
                "For this description, the number of lines is 10 and the number of overlap lines is two."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query=(
            "For CV-X with an LJ-V head in continuous mode, what example line count and overlap count "
            "does the timing description use?"
        ),
    )

    assert "number of lines is 10" in validated.answer
    assert "overlap lines is two" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["continuous-mode-example"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def _quantity_answer(answer_text: str, cited_chunk_id: str = "continuous-mode-example") -> AnswerResponse:
    return AnswerResponse(
        answer=answer_text,
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": cited_chunk_id,
                "document_id": "d1",
                "pages": [863],
                "quote_span": None,
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )


def _quantity_result(
    content: str,
    chunk_id: str = "continuous-mode-example",
    score: float = 0.8,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        title="CV-X manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[863],
        section_path=["Timing chart"],
        content=content,
        metadata={"chunk_type": "section_window"},
    )


def _validate_quantity_answer(answer_text: str, results: list[SearchResult]) -> AnswerResponse:
    return validate_answer(
        _quantity_answer(answer_text),
        results,
        query=(
            "For CV-X with an LJ-V head in continuous mode, what example line count and overlap count "
            "does the timing description use?"
        ),
    )


def test_validate_answer_falls_back_for_wrong_quantity_role_values():
    results = [
        _quantity_result(
            content=(
                "Typical operations at trigger input when the LJ-V series head is used, "
                "[Continuous] is set, and [Total Number of Lines] is enabled. "
                "For this description, the number of lines is 10 and the number of overlap lines is two."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses 23 total lines and one overlap line.", results)

    assert "number of lines is 10" in validated.answer
    assert "overlap lines is two" in validated.answer
    assert "23 total lines" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["continuous-mode-example"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_when_quantity_answer_omits_requested_role():
    results = [
        _quantity_result(
            content=(
                "Typical operations at trigger input when the LJ-V series head is used, "
                "[Continuous] is set, and [Total Number of Lines] is enabled. "
                "For this description, the number of lines is 10 and the number of overlap lines is two."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses 10 lines.", results)

    assert "number of lines is 10" in validated.answer
    assert "overlap lines is two" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["continuous-mode-example"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_accepts_relation_preserving_quantity_binding():
    results = [
        _quantity_result(
            content=(
                "For this description, the number of lines is 10 and the number of overlap lines is two."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses 10 lines and two overlap lines.", results)

    assert validated.answer == "The example uses 10 lines and two overlap lines."
    assert not any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_swapped_quantity_role_values():
    results = [
        _quantity_result(
            content=(
                "For this description, the number of lines is 10 and the number of overlap lines is two."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses two lines and 10 overlap lines.", results)

    assert "number of lines is 10" in validated.answer
    assert "overlap lines is two" in validated.answer
    assert "two lines and 10 overlap" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_quantity_cross_clause_role_mixing():
    results = [
        _quantity_result(
            content=(
                "For continuous mode, the number of lines is 10. "
                "For the separate total-lines setting, the number of overlap lines is two."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses 10 lines and two overlap lines.", results)

    assert "For continuous mode" in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_quantity_cross_chunk_role_mixing():
    results = [
        _quantity_result(
            "For continuous mode, the number of lines is 10.",
            chunk_id="line-count-row",
            score=0.9,
        ),
        _quantity_result(
            "For a different timing mode, the number of overlap lines is two.",
            chunk_id="overlap-count-row",
            score=0.8,
        ),
    ]

    validated = _validate_quantity_answer("The example uses 10 lines and two overlap lines.", results)

    assert "For continuous mode" in validated.answer or "For a different timing mode" in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_sibling_quantity_values():
    results = [
        _quantity_result(
            "Requested setup: the number of lines is 10 and the number of overlap lines is two.",
            chunk_id="requested-row",
            score=0.9,
        ),
        _quantity_result(
            "Sibling setup: the number of lines is 8 and the number of overlap lines is one.",
            chunk_id="sibling-row",
            score=0.8,
        ),
    ]

    validated = _validate_quantity_answer("The example uses 8 lines and one overlap line.", results)

    assert "number of lines is 10" in validated.answer
    assert "overlap lines is two" in validated.answer
    assert "8 lines" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_handles_quantity_units_and_ranges_without_role_swapping():
    results = [
        _quantity_result(
            content=(
                "For this description, the number of lines is 10 lines and the number of overlap lines is two lines. "
                "The unrelated index range is 1 to 23."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses 10 lines and two overlap lines.", results)

    assert validated.answer == "The example uses 10 lines and two overlap lines."
    assert not any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_ignores_irrelevant_quantity_numbers():
    results = [
        _quantity_result(
            content=(
                "For this description, the number of lines is 10 and the number of overlap lines is two. "
                "The figure number is 23 and the page number is 863."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses 23 lines and 863 overlap lines.", results)

    assert "number of lines is 10" in validated.answer
    assert "overlap lines is two" in validated.answer
    assert "863 overlap" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_does_not_treat_total_as_requested_line_count():
    results = [
        _quantity_result(
            content=(
                "For this description, the number of lines is 10 and the number of overlap lines is two. "
                "Total Number of Lines is enabled and the total becomes 23."
            ),
        ),
    ]

    validated = _validate_quantity_answer("The example uses 23 total lines and two overlap lines.", results)

    assert "number of lines is 10" in validated.answer
    assert "23 total lines" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_fallback_prefers_full_troubleshooting_row_for_cause_remedy_query():
    answer = AnswerResponse(
        answer="Connect only one LJ-S head to each CA-E300LJ unit.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "wrong-sibling",
                "document_id": "d1",
                "pages": [12],
                "quote_span": "Connect only one LJ-S head to each CA-E300LJ unit.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="wrong-sibling",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective Action; Row headers: The following error - There is no LJ connected. "
                "> The following error - Three or more are connected.; Cell value: Connect only one LJ-S head to each unit."
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="full-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Error Message: The following error occurred. - There is no LJ head connected.; "
                "Cause: The LJ-S/LJ-X/LJ-V Series head is not connected to the camera input unit.; "
                "Corrective Action: Connect the LJ-S/LJ-X/LJ-V Series head to the camera input unit."
            ),
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="What causes The following error occurred. - There is no LJ head connected. and how should it be corrected?",
    )

    assert validated.answer.startswith("Error Message: The following error occurred. - There is no LJ head connected.")
    assert validated.citations[0]["chunk_id"] == "full-row"
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_rejects_quote_from_context_window_under_cited_chunk():
    answer = AnswerResponse(
        answer="The light-controller communication error is corrected by setting FLASH output time to 0.1 msec.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "nearby-row",
                "document_id": "d1",
                "pages": [10],
                "quote_span": "Set the FLASH output time to 0.1 msec.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="nearby-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Troubleshooting"],
            content="Error Number: 13001; Error Messages: Failed in the communication with the PC Program.",
            metadata={
                "chunk_type": "table_record",
                "context_window": "Error Number: 10109; Remedy: Set the FLASH output time to 0.1 msec.",
            },
        )
    ]

    validated = validate_answer(answer, results)

    assert validated.answer.startswith("Error Number: 13001; Error Messages: Failed in the communication with the PC Program.")
    assert validated.citations[0]["chunk_id"] == "nearby-row"
    assert validated.citations[0]["quote_span"] is None
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_fallback_when_troubleshooting_answer_uses_wrong_error_anchor():
    answer = AnswerResponse(
        answer=(
            "Failed Ethernet communication is corrected by checking whether the PC/PLC is ready "
            "and whether the Ethernet software is running."
        ),
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "ethernet-row",
                "document_id": "d1",
                "pages": [12],
                "quote_span": None,
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="light-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Troubleshooting"],
            content=(
                "Error Messages: An error occurred in the communication with the light controller. "
                "Cause: The next FLASH was input while the light was being emitted. "
                "Remedy: Set the FLASH output time to 0.1 msec."
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="ethernet-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Error Messages: Failed in the Ethernet communication. "
                "Cause: An error occurred with Ethernet communication. "
                "Remedy: Check whether the PC/PLC is ready."
            ),
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query=(
            "What causes An error occurred in the communication with the light controller, "
            "and how should it be corrected?"
        ),
    )

    assert validated.answer.startswith("Error Messages: An error occurred in the communication with the light controller.")
    assert validated.citations[0]["chunk_id"] == "light-row"
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_prioritize_results_keeps_exact_troubleshooting_anchor_before_model_judgments(monkeypatch):
    light_row = SearchResult(
        chunk_id="light-row",
        score=0.9,
        title="Doc",
        document_version_id="v1",
        source_document_id="d1",
        pages=[10],
        section_path=["Troubleshooting"],
        content=(
            "Error Messages: An error occurred in the communication with the light controller. "
            "Cause: The next FLASH was input while the light was being emitted. "
            "Remedy: Set the FLASH output time to 0.1 msec."
        ),
        metadata={"chunk_type": "table_record"},
    )
    ethernet_row = SearchResult(
        chunk_id="ethernet-row",
        score=0.8,
        title="Doc",
        document_version_id="v1",
        source_document_id="d1",
        pages=[12],
        section_path=["Troubleshooting"],
        content=(
            "Error Messages: Failed in the Ethernet communication. "
            "Cause: An error occurred with Ethernet communication. "
            "Remedy: Check whether the PC/PLC is ready."
        ),
        metadata={"chunk_type": "table_record"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.judge_retrieval_relevance",
        lambda _query, _results: [
            {"chunk_id": "light-row", "verdict": "not_relevant", "reason": "model miss"},
            {"chunk_id": "ethernet-row", "verdict": "relevant", "reason": "model selected a sibling row"},
        ],
    )

    prioritized = prioritize_results_for_answer(
        "What causes An error occurred in the communication with the light controller, and how should it be corrected?",
        [light_row, ethernet_row],
    )

    assert [result.chunk_id for result in prioritized["prioritized_results"]] == ["light-row"]


def test_validate_answer_keeps_troubleshooting_answer_with_matching_error_anchor():
    answer = AnswerResponse(
        answer=(
            "Error Messages: An error occurred in the communication with the light controller. "
            "Cause: The next FLASH was input while the light was being emitted. "
            "Remedy: Set the FLASH output time to 0.1 msec."
        ),
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "light-row",
                "document_id": "d1",
                "pages": [10],
                "quote_span": "Set the FLASH output time to 0.1 msec.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="light-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Troubleshooting"],
            content=(
                "Error Messages: An error occurred in the communication with the light controller. "
                "Cause: The next FLASH was input while the light was being emitted. "
                "Remedy: Set the FLASH output time to 0.1 msec."
            ),
            metadata={"chunk_type": "table_record"},
        )
    ]

    validated = validate_answer(
        answer,
        results,
        query=(
            "What causes An error occurred in the communication with the light controller, "
            "and how should it be corrected?"
        ),
    )

    assert validated.answer == answer.answer
    assert validated.citations[0]["chunk_id"] == "light-row"


def test_validate_answer_accepts_citation_quote_from_cited_chunk():
    answer = AnswerResponse(
        answer="Change to a trigger signal that can be used.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "action-row",
                "document_id": "d1",
                "pages": [11],
                "quote_span": "Change to a trigger signal that can be used.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="action-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Corrective Action: Change to a trigger signal that can be used.",
            metadata={"chunk_type": "table_record"},
        )
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "Change to a trigger signal that can be used."
    assert not any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_accepts_short_identifier_answer_when_supported():
    answer = AnswerResponse(
        answer="CA-EN100U",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="CA-EN100U Data Sheet",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["CA-EN100U"],
            content="CA-EN100U Encoder relay unit",
            metadata={"chunk_type": "atomic_text"},
        )
    ]
    validated = validate_answer(answer, results)
    assert validated.answer == "CA-EN100U"
    assert not any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_expands_terse_structured_table_answer():
    answer = AnswerResponse(
        answer="OP-42284",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Ring light options",
            document_version_id="v1",
            source_document_id="d1",
            pages=[13],
            section_path=["Options"],
            content='Part number: 19.69" OP-42284; Applicable light: CA-DRx9',
            metadata={"chunk_type": "table_record"},
        )
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == 'Part number: 19.69" OP-42284; Applicable light: CA-DRx9'
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_parse_relevance_response_detects_missing_chunk_ids_and_normalizes_null_fields():
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect grouping function",
            metadata={"chunk_type": "section_window"},
        ),
    ]
    parsed, diagnostics = _parse_relevance_response(
        '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":null,"reason":null}]}',
        "Defect Tool",
        results,
    )
    assert parsed[0]["verdict"] == "relevant"
    assert parsed[1]["verdict"] == "relevant"
    assert parsed[1]["reason"]
    assert diagnostics["invalid_items"]
    assert diagnostics["missing_chunk_ids"] == []


def test_judge_retrieval_relevance_retries_when_chunk_coverage_is_incomplete(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect grouping function",
            metadata={"chunk_type": "section_window"},
        ),
    ]
    prompts: list[str] = []

    class FakeResponse:
        def __init__(self, body: str):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": self._body}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, _path, json):
            prompts.append(json["prompt"])
            self._calls += 1
            if self._calls == 1:
                return FakeResponse('{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."}]}')
            return FakeResponse('{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related but broader."}]}')

    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **kwargs: (
            {"items": []},
            '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."}]}' if len(prompts) == 0 else '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related but broader."}]}',
        ),
    )
    original_prompt = None

    def fake_chat_json(**kwargs):
        nonlocal original_prompt
        prompts.append(kwargs["messages"][-1]["content"])
        if len(prompts) == 1:
            return {"items": []}, '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."}]}'
        return {"items": []}, '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related but broader."}]}'

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    judgments = judge_retrieval_relevance("Defect Tool", results)

    assert [item["chunk_id"] for item in judgments] == ["c1", "c2"]
    assert judgments[1]["verdict"] == "potentially_relevant"
    assert len(prompts) == 2
    assert "Required chunk_ids in order" in prompts[1]


def test_answer_prioritization_excludes_wrong_model_family_table_rows(monkeypatch):
    results = [
        SearchResult(
            chunk_id="lj-x-warning",
            score=0.9,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[36],
            section_path=["LJ-X8200/LJ-X8300/LJ-X8400/LJ-X8900"],
            content="LASER RADIATION CLASS 2M LASER PRODUCT Wavelength : 405nm Output : 10mW",
            metadata={"chunk_type": "spec_record"},
        ),
        SearchResult(
            chunk_id="lj-x-table",
            score=0.8,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[36],
            section_path=["HALCON"],
            content=(
                "Column headers: LJ-X8020 > LJ-X8060 > LJ-X8080 > LJ-X8200; "
                "Row headers: Light source > Laser class; Cell value: Class 2M laser product"
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="lj-v-table",
            score=0.7,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[45],
            section_path=["135°"],
            content=(
                "Column headers: LJ-V7080/ LJ-V7080B > LJ-V7200/ LJ-V7200B; "
                "Row headers: Light source > Laser class; Cell value: Class 2"
            ),
            metadata={"chunk_type": "table_record", "product_model": "New LJ-X8000 Series"},
        ),
        SearchResult(
            chunk_id="lj-s-table",
            score=0.6,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[47],
            section_path=["135°"],
            content="Column headers: LJ-S015 > LJ-S025 > LJ-S040; Row headers: Laser light source; Cell value: 405",
            metadata={"chunk_type": "table_record", "product_model": "New LJ-X8000 Series"},
        ),
        SearchResult(
            chunk_id="lj-s-grouped-table",
            score=0.5,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[47],
            section_path=["135°"],
            content=(
                "Laser light source | nm(visible light) wavelength blue semiconductor laser | 405\n"
                "Laser class | 2Mlaser product | Class\n"
                "Output | 10mW"
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "New LJ-X8000 Series",
                "table_row_group": True,
                "identifier_tokens": ["LJ-S015", "LJ-S025", "LJ-S040"],
            },
        ),
    ]

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    prioritized = prioritize_results_for_answer("New LJ-X8000 Series laser radiation", results)

    judgments = {item["chunk_id"]: item["verdict"] for item in prioritized["judgments"]}
    assert judgments["lj-x-table"] != "not_relevant"
    assert judgments["lj-v-table"] == "not_relevant"
    assert judgments["lj-s-table"] == "not_relevant"
    assert judgments["lj-s-grouped-table"] == "not_relevant"
    assert [result.chunk_id for result in prioritized["prioritized_results"]] == ["lj-x-warning", "lj-x-table"]


def test_generate_answer_uses_fast_model_for_relevance_and_summaries_then_answer_model(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool setup instructions",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[2],
            section_path=["Tools"],
            content="Defect grouping function details",
            metadata={"chunk_type": "section_window"},
        ),
    ]
    calls: list[dict[str, str]] = []

    def fake_chat_json(**kwargs):
        calls.append(
            {
                "model": kwargs["model"],
                "prompt": kwargs["messages"][-1]["content"],
                "num_predict": kwargs.get("num_predict"),
                "think": kwargs.get("think"),
            }
        )
        prompt = kwargs["messages"][0]["content"]
        if "You are judging whether each evidence item is relevant" in prompt:
            return {"items": []}, '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Directly answers."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related context."}]}'
        if "You summarize retrieved evidence" in prompt:
            if '"chunk_id": "c1"' in kwargs["messages"][-1]["content"]:
                return {"summary": "Defect Tool setup instructions."}, '{"summary":"Defect Tool setup instructions."}'
            return {"summary": "Defect grouping function details."}, '{"summary":"Defect grouping function details."}'
        if "You compress multiple evidence summaries" in prompt:
            return {"summary": "Defect Tool setup and grouping details."}, '{"summary":"Defect Tool setup and grouping details."}'
        return {
            "answer": "Use the Defect Tool setup and grouping details.",
            "confidence": "medium",
            "used_documents": [],
            "citations": [],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        }, '{"answer":"Use the Defect Tool setup and grouping details.","confidence":"medium","used_documents":[],"citations":[],"warnings":[],"followup_questions":[],"insufficient_evidence":false}'

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer = generate_answer("How do I use defect detection?", results)

    assert answer.answer
    assert [call["model"] for call in calls[:-1]] == ["qwen3.5:4b", "qwen3.5:4b", "qwen3.5:4b"]
    assert calls[-1]["model"] == "qwen3.5:9b"
    assert calls[-1]["num_predict"] == -1
    assert calls[-1]["think"] is False


def test_generate_answer_reuses_precomputed_prioritized_results_and_summaries(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool setup instructions",
            metadata={"chunk_type": "atomic_text"},
        )
    ]
    summarized_evidence = [
        {
            "chunk_id": "c1",
            "title": "Doc",
            "pages": [1],
            "section_path": ["Tools"],
            "summary": "Defect Tool setup instructions.",
            "source_document_id": "d1",
            "document_version_id": "v1",
        }
    ]
    calls: list[str] = []

    def fail_prioritize(*args, **kwargs):
        raise AssertionError("prioritize_results_for_answer should not be called")

    def fail_summarize(*args, **kwargs):
        raise AssertionError("summarize_results_for_answer should not be called")

    def fake_chat_json(**kwargs):
        calls.append(kwargs["model"])
        return {
            "answer": "Defect Tool setup instructions.",
            "confidence": "medium",
            "used_documents": [],
            "citations": [],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        }, '{"answer":"Defect Tool setup instructions.","confidence":"medium","used_documents":[],"citations":[],"warnings":[],"followup_questions":[],"insufficient_evidence":false}'

    monkeypatch.setattr("manuals_rag_answering.generator.prioritize_results_for_answer", fail_prioritize)
    monkeypatch.setattr("manuals_rag_answering.generator.summarize_results_for_answer", fail_summarize)
    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer = generate_answer(
        "Defect Tool",
        results,
        prioritized_results=results,
        summarized_evidence=summarized_evidence,
    )

    assert answer.answer == "Defect Tool setup instructions."
    assert calls == ["qwen3.5:9b"]


def test_generate_answer_with_trace_exposes_summary_input_and_fallback_state(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool setup instructions",
            metadata={"chunk_type": "atomic_text"},
        )
    ]
    summarized_evidence = [
        {
            "chunk_id": "c1",
            "title": "Doc",
            "pages": [1],
            "section_path": ["Tools"],
            "summary": "Use the Defect Tool for setup.",
            "source_document_id": "d1",
            "document_version_id": "v1",
        }
    ]

    def fake_chat_json(**kwargs):
        return (
            {
                "answer": "Use the Defect Tool for setup.",
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            '{"answer":"Use the Defect Tool for setup.","confidence":"high","used_documents":[],"citations":[],"warnings":[],"followup_questions":[],"insufficient_evidence":false}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "Defect Tool",
        results,
        prioritized_results=results,
        summarized_evidence=summarized_evidence,
    )

    assert answer.answer == "Use the Defect Tool for setup."
    assert trace["final_answer"]["model"] == "qwen3.5:9b"
    assert trace["final_answer"]["num_predict"] == -1
    assert trace["final_answer"]["used_fallback"] is False
    assert trace["final_answer"]["summarized_evidence"][0]["summary"] == "Use the Defect Tool for setup."


def test_structured_evidence_uses_direct_summary_without_chunk_summary_model(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[3],
            section_path=["Settings"],
            content="Column headers: setting; Row headers: Width; Cell value: measure",
            metadata={"chunk_type": "table_record", "context_window": "Width setting selects measure."},
        )
    ]
    calls: list[str] = []

    def fake_chat_json(**kwargs):
        calls.append(kwargs["purpose"])
        if kwargs["purpose"] == "chunk_summary":
            raise AssertionError("focused table evidence should not need model summarization")
        if kwargs["purpose"] == "relevance_review":
            return (
                {"items": [{"chunk_id": "c-table", "verdict": "relevant", "reason": "Direct table match."}]},
                '{"items":[{"chunk_id":"c-table","verdict":"relevant","reason":"Direct table match."}]}',
            )
        return (
            {
                "answer": "The Width setting selects measure.",
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            '{"answer":"The Width setting selects measure.","confidence":"high",'
            '"used_documents":[],"citations":[],"warnings":[],'
            '"followup_questions":[],"insufficient_evidence":false}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace("What setting selects measure?", results)

    assert answer.answer == "The Width setting selects measure."
    assert calls == ["relevance_review", "final_answer"]
    summary = trace["final_answer"]["summarized_evidence"][0]
    assert summary["summary_source"] == "direct_evidence"
    assert summary["summary"].startswith("Column headers: setting")
    assert "Context: Width setting selects measure." in summary["summary"]


def test_validation_fallback_preserves_focused_table_cell_before_context(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["PLC"],
            content=(
                "Column headers: 6bit > 5bit; Row headers: 0028 65.0 > "
                "Command output area; Cell value: Command Result"
            ),
            metadata={
                "chunk_type": "table_record",
                "context_window": "status Bit area | 0000 | Result Ready | Cmd Ready",
            },
        )
    ]

    def fake_chat_json(**kwargs):
        return (
            {
                "answer": "Unrelated generated answer",
                "confidence": "medium",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            (
                '{"answer":"Unrelated generated answer","confidence":"medium",'
                '"used_documents":[],"citations":[],"warnings":[],'
                '"followup_questions":[],"insufficient_evidence":false}'
            ),
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "What command 0028 65.0 Command output area 6bit value applies?",
        results,
        prioritized_results=results,
        summarized_evidence=[
            {
                "chunk_id": "c-table",
                "title": "Doc",
                "pages": [7],
                "section_path": ["PLC"],
                "summary": "The table row says Command Result.",
                "source_document_id": "d1",
                "document_version_id": "v1",
            }
        ],
    )

    assert answer.answer.startswith("Column headers: 6bit > 5bit")
    assert "Cell value: Command Result" in answer.answer
    assert "Context: status Bit area" in answer.answer
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


def test_validation_fallback_omits_unrequested_neighbor_setting_context(monkeypatch):
    results = [
        SearchResult(
            chunk_id="settings-row",
            score=0.9,
            title="LJ-S8000 Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[172],
            section_path=["5-81"],
            content=(
                'Setting item: Call Text at Read Error; Settings: If enabled, the character data specified in "Text Called" '
                "will be output when reading fails.\n"
                "Setting item: Output Symbol Identifier; Settings: When enabled, a symbol identifier (3 bytes) defined by "
                "the ISO / IEC 15424 / JIS X 0530 data carrier identifier (including the symbology identifier) is added "
                "to the beginning of the read data.\n"
                "Setting item: Expansion Channel Interpretation (ECI); Settings: When enabled, ECI is output as the result "
                "of reading code that contains ECI."
            ),
            metadata={
                "chunk_type": "table_record",
                "context_window": (
                    'Setting item: Conditions; Settings: To set multiple criteria for matching, select "Multiple".\n'
                    "Setting item: Condition List; Settings: Up to 16 collation conditions can be set.\n"
                    "Setting item: Data Range; Settings: Choose the range to match against the matching pattern.\n"
                    "Setting item: Reference Pattern; Settings: Enter a pattern to match the code reading results."
                ),
            },
        )
    ]

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "final_answer":
            return (
                {
                    "answer": "Unsupported generated answer",
                    "confidence": "medium",
                    "used_documents": [],
                    "citations": [],
                    "warnings": [],
                    "followup_questions": [],
                    "insufficient_evidence": False,
                },
                '{"answer":"Unsupported generated answer","confidence":"medium",'
                '"used_documents":[],"citations":[],"warnings":[],'
                '"followup_questions":[],"insufficient_evidence":false}',
            )
        return (
            {"items": [{"chunk_id": "settings-row", "verdict": "relevant", "reason": "Direct setting row."}]},
            '{"items":[{"chunk_id":"settings-row","verdict":"relevant","reason":"Direct setting row."}]}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "What does the LJ-S8000 Output Symbol Identifier setting add when it is enabled?",
        results,
    )

    assert "Output Symbol Identifier" in answer.answer
    assert "symbol identifier (3 bytes)" in answer.answer
    assert "Call Text at Read Error" not in answer.answer
    assert "Expansion Channel Interpretation" not in answer.answer
    assert "Conditions" not in answer.answer
    assert "Data Range" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


def test_validation_support_checks_table_cell_before_context(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[13],
            section_path=["Ring lights"],
            content='Column headers: Part number; Cell value: 19.69" OP-42284; Row: 4; Column: 0',
            metadata={"chunk_type": "table_record", "context_window": "CA-DRR3 | 1.5W | 12VDC"},
        )
    ]

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "relevance_review":
            return (
                {"items": [{"chunk_id": "c-table", "verdict": "relevant", "reason": "Direct table match."}]},
                '{"items":[{"chunk_id":"c-table","verdict":"relevant","reason":"Direct table match."}]}',
            )
        return (
            {
                "answer": 'The part number is 19.69" OP-42284.',
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            '{"answer":"The part number is 19.69\\" OP-42284.","confidence":"high",'
            '"used_documents":[],"citations":[],"warnings":[],'
            '"followup_questions":[],"insufficient_evidence":false}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace("What 19.69 OP-42284 applies?", results)

    assert answer.answer == 'The part number is 19.69" OP-42284.'
    assert trace["final_answer"]["used_fallback"] is False


def test_docling_page_batches_cover_full_document():
    assert _docling_page_batches(10, 4) == [(1, 4), (5, 8), (9, 10)]


def test_resolved_page_no_preserves_original_range():
    assert _resolved_page_no(7, batch_start=5, batch_end=8) == 7
    assert _resolved_page_no(3, batch_start=5, batch_end=8) == 7


def test_docling_text_blocks_map_page_numbers_from_batched_export():
    exported = {
        "texts": [
            {"text": "Section A", "prov": [{"page_no": 1}]},
            {"text": "Step 1. Connect cable", "prov": [{"page_no": 2}]},
        ]
    }
    blocks = _docling_text_blocks(exported, batch_start=5, batch_end=6)
    assert blocks == [(5, "Section A"), (6, "Step 1. Connect cable")]


def test_docling_text_blocks_correct_collapsed_page_one_provenance_from_pdf_text():
    exported = {
        "texts": [
            {"text": "First page setup instructions", "prov": [{"page_no": 1}]},
            {"text": "Second page calibration steps", "prov": [{"page_no": 1}]},
            {"text": "Third page troubleshooting table", "prov": [{"page_no": 1}]},
        ]
    }
    blocks = _docling_text_blocks(
        exported,
        batch_start=7,
        batch_end=9,
        page_texts={
            7: "First page setup instructions",
            8: "Second page calibration steps",
            9: "Third page troubleshooting table",
        },
    )

    assert blocks == [
        (7, "First page setup instructions"),
        (8, "Second page calibration steps"),
        (9, "Third page troubleshooting table"),
    ]


def test_docling_table_blocks_extract_structured_cells_and_skip_child_text_refs():
    exported = {
        "texts": [
            {"text": "Heading", "prov": [{"page_no": 1}]},
            {"text": "Model", "prov": [{"page_no": 1}]},
            {"text": "LJ-X8000", "prov": [{"page_no": 1}]},
        ],
        "tables": [
            {
                "children": [{"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}],
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4, "coord_origin": "BOTTOMLEFT"}}],
                "data": {
                    "num_rows": 2,
                    "num_cols": 2,
                    "table_cells": [
                        {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Model", "column_header": True},
                        {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "Repeatability", "column_header": True},
                        {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "LJ-X8000", "row_header": True},
                        {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "0.3 um"},
                    ],
                },
            }
        ],
    }

    refs = _docling_table_child_refs(exported)
    assert refs == {"#/texts/1", "#/texts/2"}
    assert _docling_text_blocks(exported, batch_start=5, batch_end=5, excluded_refs=refs) == [(5, "Heading")]

    table_blocks = _docling_table_blocks(exported, batch_start=5, batch_end=5)
    assert table_blocks[0][0] == 5
    table_json = table_blocks[0][1]
    assert table_json["headers"] == ["Model", "Repeatability"]
    assert table_json["rows"] == [["LJ-X8000", "0.3 um"]]
    assert table_json["row_count"] == 1
    assert table_json["column_count"] == 2
    assert table_json["cells"][0]["column_header"] is True
    assert table_json["bbox"]["coord_origin"] == "BOTTOMLEFT"


def test_docling_pipeline_enables_tableformer_for_standard_manuals():
    options = _docling_pipeline_options(ParseProfile.standard_manual, device="cuda")

    assert options.do_table_structure is True
    assert options.table_structure_options.do_cell_matching is True
    assert options.accelerator_options.device == "cuda"


def _read_pdf(pdf_path: Path) -> bytes:
    return pdf_path.read_bytes()


def test_parse_document_preserves_page_provenance_for_multi_page_fixture():
    pdf_path = tmp_eval_small_pdf_path("AS_151019_LJ-X8000_C_689092_KA_US_2055_2.pdf")
    result = parse_document("page-provenance-version", pdf_path.name, _read_pdf(pdf_path))

    distinct_pages = {node.page_from for node in result.logical_nodes}
    expected_page_count = fitz.open(pdf_path).page_count

    assert result.page_count == expected_page_count
    assert result.page_count > 1
    assert len(distinct_pages) > 1
    assert min(distinct_pages) == 1
    assert max(distinct_pages) == result.page_count
    assert any(node.page_from == 1 for node in result.logical_nodes)
    assert any(node.page_from == result.page_count for node in result.logical_nodes)
    assert all(1 <= node.page_from <= result.page_count for node in result.logical_nodes)


def test_parse_document_docling_artifact_batches_cover_full_page_range():
    pdf_path = tmp_eval_small_pdf_path("AS_151019_LJ-X8000_C_689092_KA_US_2055_2.pdf")
    result = parse_document("artifact-batch-version", pdf_path.name, _read_pdf(pdf_path))

    artifact = result.docling_artifact
    if artifact.get("parser") == "pymupdf_fallback":
        return
    batches = artifact["batches"]
    covered_pages: list[int] = []
    for batch in batches:
        start, end = batch["page_range"]
        covered_pages.extend(range(start, end + 1))

    assert artifact["original_page_count"] == result.page_count
    assert artifact["batch_count"] == len(batches)
    assert covered_pages[0] == 1
    assert covered_pages[-1] == result.page_count
    assert covered_pages == list(range(1, result.page_count + 1))
