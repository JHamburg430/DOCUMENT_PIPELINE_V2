from pathlib import Path

import fitz

from manuals_rag_answering.generator import (
    _comparison_answer_covers_retrieved_model_sides,
    _fallback_answer,
    _fallback_evidence_results,
    _is_comparison_query,
    _parse_relevance_response,
    _troubleshooting_citations_match_query_anchor,
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


def test_insufficient_diagnostic_table_answer_falls_back_to_source_row():
    answer = AnswerResponse(
        answer="The provided evidence does not contain the specific definition of error A-14.",
        confidence="low",
        used_documents=[],
        citations=[],
        warnings=[
            "The retrieved text only confirms an A-14 section title.",
            "The provided evidence does not contain field-network instructions.",
        ],
        followup_questions=[],
        insufficient_evidence=True,
    )
    results = [
        SearchResult(
            chunk_id="a14-row",
            score=0.92,
            title="IV4 manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[527],
            section_path=["12-38"],
            content=(
                "PWR/ERR indicator light status: A buffer overrun has occurred.; "
                "Cause: Execute result acquisition completion notification from an external "
                "device such as a PLC. Disable handshake control for field networks "
                "(EtherNet/IP, Profinet)."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "MOD-600",
                "identifier_tokens": ["A-14"],
            },
        )
    ]

    validated = validate_answer(
        answer,
        results,
        query="On MOD-600, what does error A-14 indicate, and what field-network settings should I check?",
    )

    assert "buffer overrun" in validated.answer.lower()
    assert "handshake control" in validated.answer.lower()
    assert "system error" not in validated.answer.lower()
    assert validated.insufficient_evidence is False
    assert [citation["chunk_id"] for citation in validated.citations] == ["a14-row"]
    assert any("not sufficiently supported" in warning for warning in validated.warnings)
    assert not any("section title" in warning for warning in validated.warnings)
    assert not any("does not contain field-network" in warning for warning in validated.warnings)


def test_insufficient_diagnostic_table_answer_requires_requested_remedy_facet():
    answer = AnswerResponse(
        answer="The provided evidence does not contain the specific definition of error A-14.",
        confidence="low",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=True,
    )
    results = [
        SearchResult(
            chunk_id="partial-a14-row",
            score=0.92,
            title="IV4 manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[527],
            section_path=["12-38"],
            content=(
                "PWR/ERR indicator light status: A buffer overrun has occurred.; "
                "Cause: The result output buffer is full."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "MOD-600",
                "identifier_tokens": ["A-14"],
            },
        )
    ]

    validated = validate_answer(
        answer,
        results,
        query="On MOD-600, what does error A-14 indicate, and what field-network settings should I check?",
    )

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "field-network" not in validated.answer.lower()


def test_insufficient_diagnostic_table_answer_keeps_wrong_scope_closed():
    answer = AnswerResponse(
        answer="The provided evidence does not contain the specific definition of error A-14.",
        confidence="low",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=True,
    )
    results = [
        SearchResult(
            chunk_id="a15-other-model",
            score=0.92,
            title="IV4 manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[528],
            section_path=["12-39"],
            content=(
                "PWR/ERR indicator light status: A buffer overrun has occurred.; "
                "Cause: Disable handshake control for field networks (EtherNet/IP, Profinet)."
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "MOD-120",
                "identifier_tokens": ["A-15"],
            },
        )
    ]

    validated = validate_answer(
        answer,
        results,
        query="On MOD-600, what does error A-14 indicate, and what field-network settings should I check?",
    )

    assert validated.insufficient_evidence is True
    assert validated.citations == []


def test_status_output_fallback_prefers_exact_row_cell_binding():
    answer = AnswerResponse(
        answer="The provided evidence does not contain the requested output status.",
        confidence="low",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=True,
    )
    query = (
        "For MOD-120 status output settings, when ON equals Set value, Count value is 9, "
        "previous count is 7, quantity counted at one time is 3, and current count is 0, "
        "what output status is listed?"
    )
    results = [
        SearchResult(
            chunk_id="shifted-key-values",
            score=0.95,
            title="Controller manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[168],
            section_path=["Status output settings"],
            content=(
                "Status output settings: ON when = Set value; Previous count value "
                "(Display value): Count value= 9; Quantity counted at one time: 7; "
                "Current count value (Display value): 3; Output status: 0"
            ),
            metadata={
                "chunk_type": "table_record",
                "table_key_value": True,
                "context_window": (
                    "ON when = Set value | Count value= 10 | 7 | 3 | 0 | One-Shot output\n"
                    "ON when = Set value | Count value= 11 | 7 | 3 | 0 | Does not output"
                ),
            },
        ),
        SearchResult(
            chunk_id="status-output-cell",
            score=0.88,
            title="Controller manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[168],
            section_path=["Status output settings"],
            content=(
                "Column headers: Output status; Row headers: ON when = Set value > "
                "Count value= 9; Cell value: One-Shot output; Row: 4; Column: 5"
            ),
            metadata={
                "chunk_type": "table_record",
                "table_cell": True,
                "table_column_headers": ["Output status"],
                "table_row_headers": ["ON when = Set value", "Count value= 9"],
                "context_window": (
                    "ON when = Set value | Count value= 9 | 7 | 2 | 9 | Latching output\n"
                    "ON when = Set value | Count value= 9 | 7 | 3 | 0 | One-Shot output"
                ),
            },
        ),
    ]

    validated = validate_answer(answer, results, query=query)

    assert validated.insufficient_evidence is False
    assert "One-Shot output" in validated.answer
    assert "Current count value (Display value): 3" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["status-output-cell"]


def test_status_output_fallback_fails_closed_without_exact_row_cell_binding():
    answer = AnswerResponse(
        answer="The provided evidence does not contain the requested output status.",
        confidence="low",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=True,
    )
    results = [
        SearchResult(
            chunk_id="shifted-key-values",
            score=0.95,
            title="Controller manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[168],
            section_path=["Status output settings"],
            content=(
                "Status output settings: ON when = Set value; Previous count value "
                "(Display value): Count value= 9; Quantity counted at one time: 7; "
                "Current count value (Display value): 3; Output status: 0"
            ),
            metadata={
                "chunk_type": "table_record",
                "table_key_value": True,
                "context_window": (
                    "ON when = Set value | Count value= 10 | 7 | 3 | 0 | One-Shot output\n"
                    "ON when = Set value | Count value= 11 | 7 | 3 | 0 | Does not output"
                ),
            },
        )
    ]

    validated = validate_answer(
        answer,
        results,
        query=(
            "For MOD-120 status output settings, when ON equals Set value, Count value is 9, "
            "previous count is 7, quantity counted at one time is 3, and current count is 0, "
            "what output status is listed?"
        ),
    )

    assert validated.insufficient_evidence is True
    assert validated.citations == []
    assert "Current count value (Display value): 3" not in validated.answer


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


def test_comparison_table_fallback_keeps_requested_model_sides():
    query = "For LJ-S8000 and LJ-X8000, compare the measured-data format for ERRC with T1 Angle 1 MS/AB."
    results = [
        SearchResult(
            chunk_id="lj-s-errc",
            score=0.9,
            title="LJ-S8000 manual",
            document_version_id="v1",
            source_document_id="doc-ljs",
            pages=[462],
            section_path=["Measured data"],
            content=(
                "Column headers: Form of measured data; Row headers: ERRC > Error Code; "
                "Cell value: Integer 7 digits; Row: 4; Column: 3"
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "LJ-S8000 Series",
                "table_row_headers": ["ERRC", "Error Code"],
                "table_column_headers": ["Form of measured data"],
            },
        ),
        SearchResult(
            chunk_id="lj-s-t1-sibling",
            score=0.89,
            title="LJ-S8000 manual",
            document_version_id="v1",
            source_document_id="doc-ljs",
            pages=[463],
            section_path=["Measured data"],
            content=(
                "Column headers: Form of measured data; Row headers: T1 > Angle 1 > MS,AB; "
                "Cell value: Sign, Integer 3 digits, 3 digits after the decimal point"
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "LJ-S8000 Series",
                "table_row_headers": ["T1", "Angle 1", "MS,AB"],
                "table_column_headers": ["Form of measured data"],
            },
        ),
        SearchResult(
            chunk_id="lj-x-t1",
            score=0.88,
            title="LJ-X8000 manual",
            document_version_id="v1",
            source_document_id="doc-ljx",
            pages=[698],
            section_path=["Measured data"],
            content=(
                "Column headers: Form of measured data; Row headers: T1 > Angle 1 > MS,AB; "
                "Cell value: Sign, Integer 3 digits, 3 digits after the decimal point; Row: 26; Column: 4"
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "LJ-X8000 Series",
                "table_row_headers": ["T1", "Angle 1", "MS,AB"],
                "table_column_headers": ["Form of measured data"],
            },
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert [result.chunk_id for result in selected] == ["lj-s-errc", "lj-x-t1"]


def test_comparison_table_fallback_fails_closed_for_wrong_side_sibling_row():
    query = "For LJ-S8000 and LJ-X8000, compare the measured-data format for ERRC with T1 Angle 1 MS/AB."
    results = [
        SearchResult(
            chunk_id="lj-s-errc",
            score=0.9,
            title="LJ-S8000 manual",
            document_version_id="v1",
            source_document_id="doc-ljs",
            pages=[462],
            section_path=["Measured data"],
            content=(
                "Column headers: Form of measured data; Row headers: ERRC > Error Code; "
                "Cell value: Integer 7 digits"
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "LJ-S8000 Series",
                "table_row_headers": ["ERRC", "Error Code"],
                "table_column_headers": ["Form of measured data"],
            },
        ),
        SearchResult(
            chunk_id="lj-x-t1hi",
            score=0.89,
            title="LJ-X8000 manual",
            document_version_id="v1",
            source_document_id="doc-ljx",
            pages=[698],
            section_path=["Measured data"],
            content=(
                "Column headers: Form of measured data; Row headers: T1HI > Angle 1 (max) > MS,AB; "
                "Cell value: Sign, Integer 3 digits, 3 digits after the decimal point"
            ),
            metadata={
                "chunk_type": "table_record",
                "product_family": "LJ-X8000 Series",
                "table_row_headers": ["T1HI", "Angle 1 (max)", "MS,AB"],
                "table_column_headers": ["Form of measured data"],
            },
        ),
    ]

    selected = _fallback_evidence_results(query, results)

    assert selected == []


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


def test_insufficient_quantity_answer_falls_back_when_mode_state_is_set():
    answer = AnswerResponse(
        answer="I could not answer from the available evidence.",
        confidence="low",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=True,
    )
    results = [
        SearchResult(
            chunk_id="sheet-fed-settings",
            score=0.91,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="doc-cvx",
            pages=[123],
            section_path=["Timing chart"],
            content=(
                "Control/data output via I/O terminals Timing chart. "
                "Typical operations at trigger input when the LJ-V series head is used and [Sheet-fed] is set. "
                "Camera settings Number of Lines 10 Line Scan Interval Specify Encoder 1 pulse/line Sampling mode x1."
            ),
            metadata={"chunk_type": "section_window"},
        )
    ]

    validated = validate_answer(
        answer,
        results,
        query="For CV-X with an LJ-V head in sheet-fed mode, what camera line settings are shown for line count and line scan interval?",
    )

    assert validated.insufficient_evidence is False
    assert [citation["chunk_id"] for citation in validated.citations] == ["sheet-fed-settings"]
    assert "Number of Lines 10" in validated.answer
    assert "Line Scan Interval Specify Encoder 1 pulse/line" in validated.answer


def test_screen_request_fallback_prefers_actual_trigger_settings_screen():
    answer = AnswerResponse(
        answer="Use the general camera-trigger-light menu.",
        confidence="medium",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="camera-trigger-light-menu",
            score=0.95,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="doc-xgx",
            pages=[228],
            section_path=["Camera - Trigger - Light Configuration Settings"],
            content=(
                "Camera - Trigger - Light Configuration Settings: Line Camera Setting Navigation. "
                "The connected cameras, trigger input for each camera, and illumination control targets can be configured together."
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="trigger-settings-screen",
            score=0.89,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="doc-xgx",
            pages=[199],
            section_path=["Preparing a Line Scan Camera"],
            content=(
                "Select Next. The Step 2/3 Trigger Settings screen of the Line Camera Setting Navigation appears. "
                "Change the settings in accordance with the triggers that will be input into the controller."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    validated = validate_answer(
        answer,
        results,
        query="For XG-X line camera setup, which screen is used to change trigger settings?",
    )

    assert [citation["chunk_id"] for citation in validated.citations] == ["trigger-settings-screen"]
    assert "Step 2/3 Trigger Settings screen" in validated.answer


def test_procedure_membership_fallback_prefers_step_bound_preparation():
    results = [
        SearchResult(
            chunk_id="broad-line-camera-navigation",
            score=0.95,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="doc-xgx",
            pages=[205],
            section_path=["Line Camera Settings"],
            content=(
                "Line Camera Setting Navigation. Change the capture options in accordance with "
                "the onscreen instructions so that the workpiece can be correctly captured. "
                "For more details, refer to Preparation 2: Changing the Settings to Capture "
                "the Workpiece Correctly."
            ),
            metadata={"chunk_type": "section_window"},
        ),
        SearchResult(
            chunk_id="preparation-image-ratio-step",
            score=0.89,
            title="XG-X Manual",
            document_version_id="v1",
            source_document_id="doc-xgx",
            pages=[198, 201],
            section_path=["Preparing a Line Scan Camera"],
            content=(
                "Preparation 2: Changing the Settings to Capture the Workpiece Correctly "
                "(Line Camera Setting Navigation). Use the Line Camera Setting Navigation to "
                "change the capture settings such that the workpiece can be captured correctly. "
                "4. Adjust the image ratio: Change the settings so that the image aspect ratio is 1:1."
            ),
            metadata={"chunk_type": "section_window"},
        ),
    ]

    selected = _fallback_evidence_results(
        "For XG-X line camera setup, what procedure is the image-ratio adjustment part of?",
        results,
    )

    assert [result.chunk_id for result in selected] == ["preparation-image-ratio-step"]


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


def _repeated_side_troubleshooting_results_fixture() -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id="controller-error",
            score=0.9,
            title="Controller Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[40],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Remedy; Row headers: 43101 communication timeout; "
                "Cell value: Set flow control to None and inspect the serial cable."
            ),
            metadata={"chunk_type": "table_record", "product_model": "CTRL-900"},
        ),
        SearchResult(
            chunk_id="family-card-full",
            score=0.8,
            title="Vision Family Manual",
            document_version_id="v2",
            source_document_id="d2",
            pages=[85],
            section_path=["Troubleshooting"],
            content=(
                "Error Message: Storage Card 2 is full.; Cause: There is not enough free space.; "
                "Corrective Action: Make space by deleting or moving unnecessary files.; Error Code: 85\n"
                "Error Message: Storage Card 2 is write-protected.; Cause: The switch is enabled.; "
                "Corrective Action: Disable the write-protection switch.; Error Code: 86"
            ),
            metadata={"chunk_type": "table_record", "product_family": "VSN Series"},
        ),
    ]


def test_repeated_product_side_troubleshooting_fallback_binds_each_clause():
    query = (
        "On a CTRL-900, what should a technician check for Error 43101, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )
    results = _repeated_side_troubleshooting_results_fixture()

    selected = _fallback_evidence_results(query, results)

    assert _is_comparison_query(query)
    assert [result.chunk_id for result in selected] == ["controller-error", "family-card-full"]


def test_repeated_product_side_troubleshooting_fallback_rejects_missing_side():
    query = (
        "On a CTRL-900, what should a technician check for Error 43101, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )

    assert _fallback_evidence_results(query, _repeated_side_troubleshooting_results_fixture()[:1]) == []


def test_repeated_product_side_troubleshooting_fallback_rejects_sibling_symptom():
    query = (
        "On a CTRL-900, what should a technician check for Error 43101, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )
    results = _repeated_side_troubleshooting_results_fixture()
    results[1] = results[1].model_copy(
        update={
            "chunk_id": "family-card-protected",
            "content": (
                "Error Message: Storage Card 2 is write-protected.; Cause: The switch is enabled.; "
                "Corrective Action: Disable the write-protection switch.; Error Code: 86"
            ),
        }
    )

    assert _fallback_evidence_results(query, results) == []


def test_repeated_product_side_troubleshooting_rejects_cause_only_for_requested_action():
    query = (
        "On a CTRL-900, what should a technician check for Error 43101, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )
    results = _repeated_side_troubleshooting_results_fixture()
    results[0] = results[0].model_copy(
        update={"content": "Error Code: 43101; Message: Communication timeout; Cause: Cable disconnected."}
    )

    assert _fallback_evidence_results(query, results) == []
    assert _fallback_answer(query, results).insufficient_evidence is True


def test_repeated_product_side_troubleshooting_rejects_action_only_for_requested_cause():
    query = (
        "On a CTRL-900, what causes Error 43101, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )
    results = _repeated_side_troubleshooting_results_fixture()
    results[0] = results[0].model_copy(
        update={"content": "Error Code: 43101; Remedy: Set flow control to None and inspect the serial cable."}
    )

    assert _fallback_evidence_results(query, results) == []


def test_repeated_product_side_troubleshooting_does_not_mix_roles_across_chunks():
    query = (
        "On a CTRL-900, what causes Error 43101 and what should a technician check, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )
    results = _repeated_side_troubleshooting_results_fixture()
    results[0] = results[0].model_copy(
        update={"chunk_id": "controller-cause", "content": "Error Code: 43101; Cause: Cable disconnected."}
    )
    results.insert(
        1,
        results[0].model_copy(
            update={
                "chunk_id": "controller-remedy",
                "content": "Error Code: 43101; Remedy: Set flow control to None and inspect the serial cable.",
            }
        ),
    )

    assert _fallback_evidence_results(query, results) == []


def test_repeated_product_side_troubleshooting_preserves_clause_order_when_results_are_reversed():
    query = (
        "On a CTRL-900, what should a technician check for Error 43101, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )

    selected = _fallback_evidence_results(query, list(reversed(_repeated_side_troubleshooting_results_fixture())))

    assert [result.chunk_id for result in selected] == ["controller-error", "family-card-full"]


def test_repeated_product_side_troubleshooting_validation_uses_grounded_fallback():
    query = (
        "On a CTRL-900, what should a technician check for Error 43101, "
        "and on the VSN Series, what corrective action applies when Storage Card 2 is full?"
    )
    generated = AnswerResponse(
        answer="Restart both devices.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )

    validated = validate_answer(generated, _repeated_side_troubleshooting_results_fixture(), query=query)

    assert validated.insufficient_evidence is False
    assert [citation["chunk_id"] for citation in validated.citations] == [
        "controller-error",
        "family-card-full",
    ]
    assert "flow control" in validated.answer
    assert "deleting or moving" in validated.answer
    assert "write-protected" not in validated.answer
    assert "Restart both" not in validated.answer


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


def _validate_quantity_answer_for_query(answer_text: str, results: list[SearchResult], query: str) -> AnswerResponse:
    return validate_answer(_quantity_answer(answer_text), results, query=query)


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


def test_validate_answer_binds_table_style_line_and_overlap_values():
    results = [
        _quantity_result(
            content=(
                "Camera settings Number of Lines 10 "
                "Number of Overlapping Lines Two lines "
                "Total Number of Lines 23 lines."
            ),
        ),
    ]

    correct = _validate_quantity_answer("The example uses 10 lines and two overlap lines.", results)
    swapped = _validate_quantity_answer("The example uses two lines and 10 overlap lines.", results)
    wrong_overlap = _validate_quantity_answer("The example uses 10 lines and 10 overlap lines.", results)

    assert correct.answer == "The example uses 10 lines and two overlap lines."
    assert not any("not sufficiently supported" in warning for warning in correct.warnings)
    assert "Number of Lines 10" in swapped.answer
    assert "Number of Overlapping Lines Two lines" in swapped.answer
    assert "two lines and 10 overlap" not in swapped.answer
    assert any("not sufficiently supported" in warning for warning in swapped.warnings)
    assert "Number of Lines 10" in wrong_overlap.answer
    assert "Number of Overlapping Lines Two lines" in wrong_overlap.answer
    assert "10 lines and 10 overlap" not in wrong_overlap.answer
    assert any("not sufficiently supported" in warning for warning in wrong_overlap.warnings)


def test_validate_answer_binds_quantity_role_alias_values():
    results = [
        _quantity_result(
            content=(
                "Camera settings Line Count 10 "
                "Overlap Count two "
                "Total Number of Lines 23 lines."
            ),
        ),
    ]

    correct = _validate_quantity_answer("The example uses 10 lines and two overlap lines.", results)
    swapped = _validate_quantity_answer("The example uses two lines and 10 overlap lines.", results)
    total_as_line = _validate_quantity_answer("The example uses 23 lines and two overlap lines.", results)

    assert correct.answer == "The example uses 10 lines and two overlap lines."
    assert not any("not sufficiently supported" in warning for warning in correct.warnings)
    assert "Line Count 10" in swapped.answer
    assert "Overlap Count two" in swapped.answer
    assert "two lines and 10 overlap" not in swapped.answer
    assert any("not sufficiently supported" in warning for warning in swapped.warnings)
    assert "23 lines and two overlap" not in total_as_line.answer
    assert any("not sufficiently supported" in warning for warning in total_as_line.warnings)


def test_validate_answer_selects_query_scoped_quantity_group_when_wrong_sibling_first():
    results = [
        _quantity_result(
            content=(
                "Other model settings Line Count 12 Overlap Count four. "
                "Current model uses Line Count 10 Overlap Count two."
            ),
        ),
    ]

    correct = _validate_quantity_answer_for_query(
        "The current model uses 10 lines and two overlap lines.",
        results,
        "For the current model, what line count and overlap count does the timing description use?",
    )
    wrong_sibling = _validate_quantity_answer_for_query(
        "The current model uses 12 lines and four overlap lines.",
        results,
        "For the current model, what line count and overlap count does the timing description use?",
    )

    assert correct.answer == "The current model uses 10 lines and two overlap lines."
    assert not any("not sufficiently supported" in warning for warning in correct.warnings)
    assert "Current model uses Line Count 10" in wrong_sibling.answer
    assert "12 lines and four overlap" not in wrong_sibling.answer
    assert any("not sufficiently supported" in warning for warning in wrong_sibling.warnings)


def test_validate_answer_rejects_wrong_sibling_quantity_group_when_wrong_sibling_second():
    results = [
        _quantity_result(
            content=(
                "Current model uses Line Count 10 Overlap Count two. "
                "Other model settings Line Count 12 Overlap Count four."
            ),
        ),
    ]

    validated = _validate_quantity_answer_for_query(
        "The current model uses 12 lines and four overlap lines.",
        results,
        "For the current model, what line count and overlap count does the timing description use?",
    )

    assert "Current model uses Line Count 10" in validated.answer
    assert "12 lines and four overlap" not in validated.answer
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_for_ambiguous_sibling_quantity_groups():
    results = [
        _quantity_result(
            content=(
                "Other model settings Line Count 12 Overlap Count four. "
                "Current model uses Line Count 10 Overlap Count two."
            ),
        ),
    ]

    validated = _validate_quantity_answer_for_query(
        "The example uses 12 lines and four overlap lines.",
        results,
        "What line count and overlap count does the timing description use?",
    )

    assert validated.insufficient_evidence is False
    assert "12 lines and four overlap" not in validated.answer
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


def test_validate_answer_prefers_quantity_citation_with_visible_content_over_metadata_only_context():
    answer = AnswerResponse(
        answer="The example uses 9 lines and one overlap line.",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="metadata-expanded-procedure",
            score=0.95,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[864],
            section_path=["Timing chart"],
            content=(
                "Procedure step 2: Typical operations at trigger input when the LJ-V series head is used, "
                "[Continuous] is set, and [Total Number of Lines]"
            ),
            metadata={
                "chunk_type": "procedure_record",
                "content": (
                    "Procedure step 2: Typical operations at trigger input when the LJ-V series head is used, "
                    "[Continuous] is set, and [Total Number of Lines] is enabled. "
                    "For this description, the number of lines is 10 and the number of overlap lines is two."
                ),
            },
        ),
        SearchResult(
            chunk_id="visible-section-window",
            score=0.8,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[864],
            section_path=["Timing chart"],
            content=(
                "Procedure step 2: Typical operations at trigger input when the LJ-V series head is used, "
                "[Continuous] is set, and [Total Number of Lines] is enabled. "
                "Camera settings Number of Lines 10 Number of Overlapping Lines Two lines. "
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

    assert "number of lines is 10" in validated.answer or "Number of Lines 10" in validated.answer
    assert "overlap lines is two" in validated.answer or "Number of Overlapping Lines Two lines" in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["visible-section-window"]
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


def test_validate_answer_rejects_quote_from_hidden_metadata_content():
    answer = AnswerResponse(
        answer="The 10109 light-controller error is corrected by setting FLASH output time to 0.1 msec.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "sibling-cell",
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
            chunk_id="sibling-cell",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Troubleshooting"],
            content="Error Number: 10101; Error Messages: Light controller is disconnected.",
            metadata={
                "chunk_type": "table_record",
                "content": (
                    "Error Number: 10109; Error Messages: An error occurred in the communication with the light controller. "
                    "Cause: The next FLASH was input while the light was being emitted. "
                    "Remedy: Set the FLASH output time to 0.1 msec."
                ),
            },
        ),
        SearchResult(
            chunk_id="visible-row-group",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Troubleshooting"],
            content=(
                "Error Number: 10109; Error Messages: An error occurred in the communication with the light controller. "
                "Cause: The next FLASH was input while the light was being emitted. "
                "Remedy: Set the FLASH output time to 0.1 msec."
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

    assert validated.answer.startswith("Error Number: 10109; Error Messages:")
    assert [citation["chunk_id"] for citation in validated.citations] == ["visible-row-group"]
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


def _troubleshooting_row(chunk_id: str, message: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.9,
        title="Troubleshooting manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[10],
        section_path=["Troubleshooting"],
        content=f"Error Messages: {message}; Cause: A source-backed cause. Remedy: Take the documented action.",
        metadata={"chunk_type": "table_record"},
    )


def _citation(chunk_id: str) -> dict[str, object]:
    return {"chunk_id": chunk_id, "document_id": "d1", "pages": [10], "quote_span": None}


def test_troubleshooting_citations_accept_two_explicit_error_rows():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    encoder = _troubleshooting_row("encoder", "Encoder communication lost")
    query = (
        "Compare the causes and remedies for the errors Servo overload detected "
        "and Encoder communication lost."
    )

    assert _troubleshooting_citations_match_query_anchor(
        query, [_citation("servo"), _citation("encoder")], [servo, encoder]
    )


def test_troubleshooting_citations_accept_reversed_requested_and_citation_order():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    encoder = _troubleshooting_row("encoder", "Encoder communication lost")
    query = (
        "Compare the remedies for Encoder communication lost versus Servo overload detected."
    )

    assert _troubleshooting_citations_match_query_anchor(
        query, [_citation("servo"), _citation("encoder")], [servo, encoder]
    )


def test_troubleshooting_citations_accept_second_requested_side_alone():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    encoder = _troubleshooting_row("encoder", "Encoder communication lost")

    assert _troubleshooting_citations_match_query_anchor(
        "What is the remedy for Encoder communication lost?",
        [_citation("encoder")],
        [servo, encoder],
    )


def test_troubleshooting_citations_accept_one_source_equivalent_duplicate():
    primary = _troubleshooting_row("servo-primary", "Servo overload detected")
    duplicate = _troubleshooting_row("servo-duplicate", "Servo overload detected")

    assert _troubleshooting_citations_match_query_anchor(
        "What causes Servo overload detected, and how should it be corrected?",
        [_citation("servo-primary")],
        [primary, duplicate],
    )


def test_troubleshooting_citations_ignore_shorter_error_code_prefix_collision():
    requested = _troubleshooting_row("error-10109", "10109")
    prefix = _troubleshooting_row("error-1010", "1010")

    assert _troubleshooting_citations_match_query_anchor(
        "What causes Error 10109, and how should it be corrected?",
        [_citation("error-10109")],
        [requested, prefix],
    )


def test_troubleshooting_citations_bind_explicit_codes_without_surrounding_context():
    first = _troubleshooting_row("error-10109", "10109")
    second = _troubleshooting_row("error-10110", "10110")
    query = (
        "For CV-X482, compare Error 10109, the light-controller communication error, "
        "with Error 10110, where the light controller does not support LumiTrax."
    )

    assert _troubleshooting_citations_match_query_anchor(
        query,
        [_citation("error-10109"), _citation("error-10110")],
        [first, second],
    )
    assert not _troubleshooting_citations_match_query_anchor(
        query,
        [_citation("error-10109")],
        [first],
    )

    grouped = SearchResult(
        chunk_id="grouped",
        score=0.9,
        title="Doc",
        document_version_id="v1",
        source_document_id="d1",
        pages=[10],
        section_path=["Troubleshooting"],
        content=(
            "Error Number: 10109; Error Messages: Communication failed; Cause: First cause; Remedy: First remedy.\n"
            "Error Number: 10110; Error Messages: LumiTrax unsupported; Cause: Second cause; Remedy: Second remedy.\n"
            "Error Number: 10111; Error Messages: HDR unsupported; Cause: Unrequested cause; Remedy: Unrequested remedy."
        ),
        metadata={"chunk_type": "table_record", "product_model": "CV-X482"},
    )
    unsupported = AnswerResponse(
        answer="The answer is unrelated.",
        confidence="high",
        used_documents=[],
        citations=[_citation("grouped")],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )

    fallback = validate_answer(unsupported, [grouped], query=query)
    assert "10109" in fallback.answer and "10110" in fallback.answer
    assert "10111" not in fallback.answer


def test_troubleshooting_citations_accept_same_error_across_two_models():
    first = _troubleshooting_row("cvx482-error-10109", "10109")
    second = _troubleshooting_row("cvx400-error-10109", "10109")

    assert _troubleshooting_citations_match_query_anchor(
        "Compare the cause and remedy for Error 10109 on CV-X482 versus CV-X400.",
        [_citation("cvx482-error-10109"), _citation("cvx400-error-10109")],
        [first, second],
    )


def test_validate_answer_fallback_keeps_exact_same_error_rows_for_both_model_sides():
    def row(chunk_id: str, model: str, meaning: str) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id, score=1.0, title=f"{model} Manual",
            document_version_id=f"v-{model}", source_document_id=f"d-{model}",
            pages=[10], section_path=["Errors"],
            content=f"Error Code: 30109; Message: {meaning}",
            metadata={"chunk_type": "table_record", "product_model": model},
        )

    cv = row("cv", "CV-X482", "SD Card 1 is full.")
    vs = row("vs", "VS", "Cannot execute when multiple subtasks are selected.")
    generated = AnswerResponse(
        answer="Unsupported generated comparison.", confidence="high", used_documents=[], citations=[],
        warnings=[], followup_questions=[], insufficient_evidence=False,
    )
    validated = validate_answer(
        generated, [cv, vs], query="How does Error 30109 differ between CV-X482 and VS Series?"
    )

    assert not validated.insufficient_evidence
    assert {citation["chunk_id"] for citation in validated.citations} == {"cv", "vs"}
    assert "SD Card 1 is full" in validated.answer
    assert "multiple subtasks" in validated.answer


def test_fallback_evidence_keeps_distinct_chunks_for_each_requested_error():
    first = _troubleshooting_row("error-10101", "10101")
    first.metadata["product_model"] = "CV-X482"
    second = _troubleshooting_row("error-10115", "10115")
    second.metadata["product_model"] = "CV-X482"

    selected = _fallback_evidence_results(
        "For CV-X482, compare Error 10101 with Error 10115: what causes each and what is the remedy?",
        [second, first],
    )

    assert {result.chunk_id for result in selected} == {"error-10101", "error-10115"}


def test_fallback_evidence_fails_closed_when_a_requested_error_is_missing():
    first = _troubleshooting_row("error-10101", "10101")
    first.metadata["product_model"] = "CV-X482"
    unrequested = _troubleshooting_row("error-10116", "10116")
    unrequested.metadata["product_model"] = "CV-X482"

    selected = _fallback_evidence_results(
        "For CV-X482, compare Error 10101 with Error 10115: what causes each and what is the remedy?",
        [first, unrequested],
    )

    assert selected == []


def test_fallback_evidence_does_not_fill_requested_error_from_wrong_product():
    first = _troubleshooting_row("error-10101", "10101")
    first.metadata["product_model"] = "CV-X482"
    wrong_product = _troubleshooting_row("other-error-10115", "10115")
    wrong_product.metadata["product_model"] = "OTHER-700"

    selected = _fallback_evidence_results(
        "For CV-X482, compare Error 10101 with Error 10115: what causes each and what is the remedy?",
        [first, wrong_product],
    )

    assert selected == []


def test_fallback_evidence_binds_each_requested_error_to_its_product_side():
    cv = _troubleshooting_row("cv-error-10101", "10101")
    cv.metadata["product_model"] = "CV-X482"
    vs = _troubleshooting_row("vs-error-10115", "10115")
    vs.metadata["product_family"] = "VS"

    selected = _fallback_evidence_results(
        "For CV-X482 Error 10101 and VS Series Error 10115, compare causes and remedies.",
        [vs, cv],
    )

    assert {result.chunk_id for result in selected} == {"cv-error-10101", "vs-error-10115"}


def test_fallback_evidence_rejects_swapped_error_product_bindings():
    cv_wrong = _troubleshooting_row("cv-error-10115", "10115")
    cv_wrong.metadata["product_model"] = "CV-X482"
    vs_wrong = _troubleshooting_row("vs-error-10101", "10101")
    vs_wrong.metadata["product_family"] = "VS"

    selected = _fallback_evidence_results(
        "For CV-X482 Error 10101 and VS Series Error 10115, compare causes and remedies.",
        [cv_wrong, vs_wrong],
    )

    assert selected == []


def test_fallback_evidence_rejects_two_codes_from_only_one_requested_product():
    first = _troubleshooting_row("cv-error-10101", "10101")
    first.metadata["product_model"] = "CV-X482"
    second = _troubleshooting_row("cv-error-10115", "10115")
    second.metadata["product_model"] = "CV-X482"

    selected = _fallback_evidence_results(
        "For CV-X482 Error 10101 and VS Series Error 10115, compare causes and remedies.",
        [first, second],
    )

    assert selected == []


def test_fallback_evidence_preserves_reversed_error_product_order():
    cv = _troubleshooting_row("cv-error-10101", "10101")
    cv.metadata["product_model"] = "CV-X482"
    vs = _troubleshooting_row("vs-error-10115", "10115")
    vs.metadata["product_family"] = "VS"

    selected = _fallback_evidence_results(
        "Compare Error 10115 on VS Series with Error 10101 on CV-X482.",
        [cv, vs],
    )

    assert {result.chunk_id for result in selected} == {"cv-error-10101", "vs-error-10115"}


def test_fallback_evidence_requires_all_sides_for_grouped_codes_and_products():
    cv_first = _troubleshooting_row("cv-error-10101", "10101")
    cv_first.metadata["product_model"] = "CV-X482"
    vs_first = _troubleshooting_row("vs-error-10101", "10101")
    vs_first.metadata["product_family"] = "VS"
    cv_second = _troubleshooting_row("cv-error-10115", "10115")
    cv_second.metadata["product_model"] = "CV-X482"

    selected = _fallback_evidence_results(
        "For CV-X482 and VS Series, compare Error 10101 and Error 10115.",
        [cv_first, vs_first, cv_second],
    )

    assert selected == []


def test_validate_answer_fallback_fails_closed_for_explicit_code_superstring():
    sibling = SearchResult(
        chunk_id="sibling", score=1.0, title="CV-X482 Manual", document_version_id="v1",
        source_document_id="d1", pages=[10], section_path=["Errors"],
        content="Error Number: 30109; Error Messages: SD Card 1 is full.",
        metadata={"chunk_type": "table_record", "product_model": "CV-X482"},
    )
    generated = AnswerResponse(
        answer="Error 301090 means the SD card is full.", confidence="high", used_documents=[],
        citations=[_citation("sibling")], warnings=[], followup_questions=[], insufficient_evidence=False,
    )
    validated = validate_answer(generated, [sibling], query="What does Error 301090 mean on CV-X482?")

    assert validated.insufficient_evidence
    assert validated.citations == []


def test_troubleshooting_citations_accept_same_error_across_two_versions():
    first = _troubleshooting_row("v2-error-10109", "10109")
    second = _troubleshooting_row("v3-error-10109", "10109")

    assert _troubleshooting_citations_match_query_anchor(
        "Compare Error 10109 in version 2.0 versus version 3.0.",
        [_citation("v2-error-10109"), _citation("v3-error-10109")],
        [first, second],
    )


def test_troubleshooting_citations_reject_longer_error_code_superstring_collision():
    requested = _troubleshooting_row("error-1010", "1010")
    superstring = _troubleshooting_row("error-10109", "10109")

    assert not _troubleshooting_citations_match_query_anchor(
        "What causes Error 1010, and how should it be corrected?",
        [_citation("error-10109")],
        [requested, superstring],
    )


def test_troubleshooting_citations_leave_short_no_anchor_query_unconstrained():
    servo = _troubleshooting_row("servo", "Servo overload detected")

    assert _troubleshooting_citations_match_query_anchor(
        "How should this fault be corrected?", [_citation("servo")], [servo]
    )


def test_troubleshooting_citations_allow_mixed_structured_and_unstructured_support():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    note = SearchResult(
        chunk_id="note",
        score=0.8,
        title="Troubleshooting manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[11],
        section_path=["Troubleshooting"],
        content="After correcting the fault, cycle power and confirm normal operation.",
        metadata={"chunk_type": "atomic_text"},
    )

    assert _troubleshooting_citations_match_query_anchor(
        "What causes Servo overload detected, and how should it be corrected?",
        [_citation("servo"), _citation("note")],
        [servo, note],
    )


def test_troubleshooting_citations_reject_single_error_unrelated_sibling():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    encoder = _troubleshooting_row("encoder", "Encoder communication lost")

    assert not _troubleshooting_citations_match_query_anchor(
        "What causes Servo overload detected, and how should it be corrected?",
        [_citation("encoder")],
        [servo, encoder],
    )


def test_troubleshooting_citations_reject_missing_requested_side():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    encoder = _troubleshooting_row("encoder", "Encoder communication lost")

    assert not _troubleshooting_citations_match_query_anchor(
        "Compare the causes of Servo overload detected and Encoder communication lost.",
        [_citation("servo")],
        [servo, encoder],
    )


def test_troubleshooting_citations_reject_requested_side_absent_from_results():
    servo = _troubleshooting_row("servo", "Servo overload detected")

    assert not _troubleshooting_citations_match_query_anchor(
        "Compare the causes of Servo overload detected versus Encoder communication lost.",
        [_citation("servo")],
        [servo],
    )


def test_troubleshooting_citations_reject_and_joined_requested_side_absent_from_results():
    servo = _troubleshooting_row("servo", "Servo overload detected")

    assert not _troubleshooting_citations_match_query_anchor(
        "Compare the causes and remedies for the errors Servo overload detected and Encoder communication lost.",
        [_citation("servo")],
        [servo],
    )


def test_validate_answer_fails_closed_when_requested_troubleshooting_side_is_absent():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    answer = AnswerResponse(
        answer="Servo overload is caused by an excessive load; take the documented action.",
        confidence="high",
        used_documents=[],
        citations=[_citation("servo")],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )

    validated = validate_answer(
        answer,
        [servo],
        query="Compare the causes and remedies for the errors Servo overload detected and Encoder communication lost.",
    )

    assert validated.insufficient_evidence
    assert not validated.citations
    assert "Encoder communication lost" not in validated.answer
    assert any("every requested troubleshooting side" in warning for warning in validated.warnings)


def test_troubleshooting_citations_bind_alarm_sides_without_comparison_suffix():
    first = _troubleshooting_row("alarm-a14", "A-14")
    second = _troubleshooting_row("alarm-e42", "E-42")
    query = "Compare Alarm A-14 with Alarm E-42 causes and remedies."

    assert _troubleshooting_citations_match_query_anchor(
        query,
        [_citation("alarm-a14"), _citation("alarm-e42")],
        [first, second],
    )
    assert not _troubleshooting_citations_match_query_anchor(
        query,
        [_citation("alarm-a14")],
        [first],
    )


def test_troubleshooting_citations_reject_extra_unrequested_sibling():
    servo = _troubleshooting_row("servo", "Servo overload detected")
    encoder = _troubleshooting_row("encoder", "Encoder communication lost")

    assert not _troubleshooting_citations_match_query_anchor(
        "What is the remedy for Servo overload detected?",
        [_citation("servo"), _citation("encoder")],
        [servo, encoder],
    )


def test_validate_answer_rejects_sibling_troubleshooting_citations_for_specific_anchor():
    answer = AnswerResponse(
        answer=(
            "The light-controller communication error is caused by the next FLASH being input while the light is being emitted. "
            "Set the FLASH output time to 0.1 msec. Also check the light-controller connection and power source for related errors."
        ),
        confidence="high",
        used_documents=[],
        citations=[
            {"chunk_id": "light-row", "document_id": "d1", "pages": [10], "quote_span": None},
            {"chunk_id": "disconnected-row", "document_id": "d1", "pages": [10], "quote_span": None},
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
                "Error Number: 10109; Error Messages: An error occurred in the communication with the light controller. "
                "Cause: The next FLASH was input while the light was being emitted. "
                "Remedy: Set the FLASH output time to 0.1 msec."
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="disconnected-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Troubleshooting"],
            content=(
                "Error Number: 10101; Error Messages: Light controller is disconnected. "
                "Cause: The illumination expansion unit has been disconnected or the power is not turned on. "
                "Remedy: Make sure the illumination expansion unit is attached correctly."
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

    assert validated.answer.startswith("Error Number: 10109; Error Messages:")
    assert [citation["chunk_id"] for citation in validated.citations] == ["light-row"]
    assert "disconnected" not in validated.answer.lower()
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


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
                    "table_cell": True,
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
    assert "Context: status Bit area" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


def test_validation_fallback_rejects_aggregate_signal_description_inference(monkeypatch):
    results = [
        SearchResult(
            chunk_id="aggregate-signal-table",
            score=0.9,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="cvx",
            pages=[819],
            section_path=["6-78"],
            content=(
                "Column headers: Function; Row headers: OUT_DATA0 > OUT_DATA1 > OUT_DATA2 > "
                "OUT_DATA3 > OUT_DATA4 > OUT_DATA5 > OUT_DATA6 > OUT_DATA7 > OUT_DATA8 > "
                "OUT_DATA9 > OUT_DATA10 > OUT_DATA11 > OUT_DATA12 > OUT_DATA13 > OUT_DATA14 > "
                "OUT_DATA15 > Data output bit 0 (LSB) > Data output bit 1 > Data output bit 2 > "
                "Data output bit 3 > Data output bit 4 > Data output bit 5 > Data output bit 6 > "
                "Data output bit 7 > Data output bit 8 > Data output bit 9 > Data output bit 10 > "
                "Data output bit 11 > Data output bit 12 > Data output bit 13 > Data output bit 14 > "
                "Data output bit 15 (MSB); Cell value: Any of tool judgment, partial judgment, "
                "CAM judgment, or group judgment of measurement results is output according to the "
                "output settings."
            ),
            metadata={"chunk_type": "table_record"},
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
            {"items": [{"chunk_id": "aggregate-signal-table", "verdict": "relevant", "reason": "Direct table match."}]},
            '{"items":[{"chunk_id":"aggregate-signal-table","verdict":"relevant","reason":"Direct table match."}]}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "Which controller output line corresponds to data output bit 10?",
        results,
    )

    assert answer.answer != "OUT_DATA10 corresponds to Data output bit 10."
    assert "OUT_DATA10 corresponds" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "fallback_validation"
    assert any("not sufficiently supported" in warning for warning in answer.warnings)


def test_validation_fallback_accepts_single_signal_description_cell(monkeypatch):
    results = [
        SearchResult(
            chunk_id="signal-description-cell",
            score=0.9,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="cvx",
            pages=[819],
            section_path=["6-78"],
            content=(
                "Column headers: Signal Description; Row headers: OUT_DATA10; "
                "Cell value: Data output bit 10; Row: 11; Column: 1"
            ),
            metadata={
                "chunk_type": "table_record",
                "table_cell": True,
                "table_column_headers": ["Signal Description"],
                "table_row_headers": ["OUT_DATA10"],
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
            {"items": [{"chunk_id": "signal-description-cell", "verdict": "relevant", "reason": "Direct table match."}]},
            '{"items":[{"chunk_id":"signal-description-cell","verdict":"relevant","reason":"Direct table match."}]}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "Which controller output line corresponds to data output bit 10?",
        results,
    )

    assert answer.answer == "OUT_DATA10 corresponds to Data output bit 10."
    assert [citation["chunk_id"] for citation in answer.citations] == ["signal-description-cell"]
    assert trace["final_answer"]["answer_source"] == "fallback_validation"
    assert any("not sufficiently supported" in warning for warning in answer.warnings)


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


def test_fallback_fails_closed_for_sibling_table_allocation_rows(monkeypatch):
    results = [
        SearchResult(
            chunk_id="section-window",
            score=0.9,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="cvx",
            pages=[920],
            section_path=["6-180"],
            content=(
                "Setting | Address (byte) | 7bit | 6bit | 5bit | 4bit | 3bit | 2bit | 1bit | 0bit\n"
                "status Bit area | 0000 | I0.7 | Reserved | I0.6 | Reserved | I0.5 | Reserved\n"
                " | 0004 | PIB256 bit No. 7 | Allocation possible | PIB256 bit No. 6 | Allocation possible\n"
                " | 0010 | PIB262 bit No. 7 | Allocation possible | PIB262 bit No. 6 | Allocation possible\n"
                "Mea- surement count area | 0016 0017 0018 | PID 428 | Total count\n"
                "Command output area | 0019 0020 0021 0022 | PID 432 | Command Result\n"
                + "\n".join(f"filler table row {index} | Reserved | Reserved" for index in range(80))
            ),
            metadata={"chunk_type": "section_window", "product_model": "CV-X482"},
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
            {"items": [{"chunk_id": "section-window", "verdict": "relevant", "reason": "Direct table match."}]},
            '{"items":[{"chunk_id":"section-window","verdict":"relevant","reason":"Direct table match."}]}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "Can CV-X482 allocate command 0016 PID 428 in the status bit area?",
        results,
    )

    assert answer.answer == "I could not answer from the available evidence."
    assert answer.insufficient_evidence is True
    assert answer.citations == []
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


def test_fallback_focuses_table_like_section_window_when_allocation_binding_is_same_row(monkeypatch):
    results = [
        SearchResult(
            chunk_id="section-window",
            score=0.9,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="cvx",
            pages=[920],
            section_path=["6-180"],
            content=(
                "Setting | Address (byte) | Item | Allocation\n"
                "status Bit area | 0004 | PIB256 bit No. 7 | Allocation possible\n"
                "status Bit area | 0016 | PID 428 command | Allocation possible\n"
                "Mea- surement count area | 0016 0017 0018 | PID 428 | Total count\n"
                + "\n".join(f"filler table row {index} | Reserved | Reserved" for index in range(80))
            ),
            metadata={"chunk_type": "section_window", "product_model": "CV-X482"},
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
            {"items": [{"chunk_id": "section-window", "verdict": "relevant", "reason": "Direct table match."}]},
            '{"items":[{"chunk_id":"section-window","verdict":"relevant","reason":"Direct table match."}]}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "Can CV-X482 allocate command 0016 PID 428 in the status bit area?",
        results,
    )

    assert answer.answer.startswith("Relevant retrieved table row:")
    assert "status Bit area | 0016 | PID 428 command | Allocation possible" in answer.answer
    assert "PIB256" not in answer.answer
    assert "Total count" not in answer.answer
    assert "filler table row 79" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


def test_fallback_fails_closed_for_negated_table_allocation_cell(monkeypatch):
    results = [
        SearchResult(
            chunk_id="section-window",
            score=0.9,
            title="CV-X manual",
            document_version_id="v1",
            source_document_id="cvx",
            pages=[920],
            section_path=["6-180"],
            content=(
                "Setting | Address (byte) | Item | Allocation\n"
                "status Bit area | 0016 | PID 428 command | Allocation not possible\n"
                "status Bit area | 0017 | PID 429 command | Allocation possible\n"
                + "\n".join(f"filler table row {index} | Reserved | Reserved" for index in range(80))
            ),
            metadata={"chunk_type": "section_window", "product_model": "CV-X482"},
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
            {"items": [{"chunk_id": "section-window", "verdict": "relevant", "reason": "Direct table match."}]},
            '{"items":[{"chunk_id":"section-window","verdict":"relevant","reason":"Direct table match."}]}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "Can CV-X482 allocate command 0016 PID 428 in the status bit area?",
        results,
    )

    assert answer.answer == "I could not answer from the available evidence."
    assert answer.insufficient_evidence is True
    assert answer.citations == []
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


def test_vs_product_question_does_not_trigger_comparison_fallback_or_version_warning(monkeypatch):
    assert _is_comparison_query("How do I limit a saved VS program setting so only selected cameras can use it?") is False
    assert _is_comparison_query("Compare model A vs model B program settings.") is True

    results = [
        SearchResult(
            chunk_id="vs-settings-overview",
            score=0.95,
            title="VS manual",
            document_version_id="vs-v1",
            source_document_id="vs-doc",
            pages=[218],
            section_path=["4-94"],
            content=(
                "Settings Protection: Set to restrict edits to be made for the tools and the Vision Dashboard, "
                "and allow only a restricted set of VS cameras to use the program setting."
            ),
            metadata={"chunk_type": "section_window", "product_family": "VS Series"},
        ),
        SearchResult(
            chunk_id="vs-program-protection",
            score=0.9,
            title="VS manual",
            document_version_id="vs-v1",
            source_document_id="vs-doc",
            pages=[220],
            section_path=["4-104"],
            content=(
                "Program Setting Protection: Restrict the VS cameras that can use the program setting by setting a "
                "password and the MAC addresses of the cameras to allow."
            ),
            metadata={"chunk_type": "section_window", "product_family": "VS Series"},
        ),
        SearchResult(
            chunk_id="xg-save",
            score=0.7,
            title="XG-X manual",
            document_version_id="xg-v1",
            source_document_id="xg-doc",
            pages=[88],
            section_path=["2-41"],
            content="Save the current settings to the program file in SD Card 1 or SD Card 2.",
            metadata={"chunk_type": "section_window"},
        ),
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
            {
                "items": [
                    {"chunk_id": "vs-settings-overview", "verdict": "relevant", "reason": "Overview match."},
                    {"chunk_id": "vs-program-protection", "verdict": "relevant", "reason": "Direct scope match."},
                ]
            },
            '{"items":[{"chunk_id":"vs-settings-overview","verdict":"relevant","reason":"Overview match."},'
            '{"chunk_id":"vs-program-protection","verdict":"relevant","reason":"Direct scope match."}]}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "How do I limit a saved VS program setting so only selected cameras can use it?",
        results,
    )

    assert "password" in answer.answer
    assert "MAC addresses" in answer.answer
    assert "SD Card" not in answer.answer
    assert [citation["chunk_id"] for citation in answer.citations] == ["vs-program-protection"]
    assert not any("multiple document versions" in warning for warning in answer.warnings)
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


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
