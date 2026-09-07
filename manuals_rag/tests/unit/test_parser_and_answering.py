from pathlib import Path
from dataclasses import replace

import fitz
import manuals_rag_answering.generator as generator_module
import pytest

from manuals_rag_answering.generator import (
    _comparison_answer_covers_retrieved_model_sides,
    _concise_configuration_location_answer,
    _concise_general_fallback_answer,
    _concise_structured_table_answer,
    _concise_structured_fact_answer,
    _concise_troubleshooting_answer,
    _configuration_path_labels,
    _configuration_location_subject,
    _fallback_evidence_results,
    _is_comparison_query,
    _parse_relevance_response,
    _query_troubleshooting_anchor,
    _troubleshooting_field_records,
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


def test_concise_answer_keeps_bullets_required_by_following_precautions_reference():
    result = SearchResult(
        chunk_id="anti-noise",
        score=0.9,
        title="IV4 Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[5],
        section_path=["Installation"],
        content=(
            "Table summary: • To improve the anti-noise feature, install the unit following the "
            "precautions below. Otherwise, a malfunction may occur. • Ground the FG cable of the "
            "sensor amplifier (IV4-G120). • Do not mount the unit in a cabinet where high-voltage "
            "equipment is already installed. • Mount the unit as far from power lines as possible. "
            "• Separate the unit from devices that emit strong electric or magnetic fields."
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer = _concise_general_fallback_answer(
        "How do I improve anti-noise performance for the IV4-G120?",
        result,
    )

    assert "Ground the FG cable" in answer
    assert "high-voltage equipment" in answer
    assert "power lines" in answer
    assert "magnetic fields" in answer


def test_validate_answer_removes_terminal_table_coordinates():
    result = SearchResult(
        chunk_id="ac-warning",
        score=0.9,
        title="LR-ZH500C3P Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[1],
        section_path=["Safety"],
        content=(
            "Do not apply alternating current. Doing so may cause rupture or burnout.; "
            "Row: 1; Column: 1"
        ),
        metadata={"chunk_type": "table_record"},
    )
    answer = AnswerResponse(
        answer=result.content,
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )

    validated = validate_answer(
        answer,
        [result],
        query="What happens if I apply alternating current to the LR-ZH500C3P?",
    )

    assert validated.answer == (
        "Do not apply alternating current. Doing so may cause rupture or burnout."
    )
    assert "Row:" not in validated.answer
    assert "Column:" not in validated.answer


def test_structured_table_answer_binds_value_to_terminal_model_row_header():
    wrong = SearchResult(
        chunk_id="dzw30-power",
        score=1.0,
        title="Lighting catalog",
        document_version_id="v1",
        source_document_id="d1",
        pages=[10],
        section_path=["Specifications"],
        content=(
            "Column headers: Power consumption; Row headers: Model CA-DZW10X "
            "CA-DZW30X CA-DZW50X > CA-DZW30X; Cell value: 62 W; Row: 2; Column: 2"
        ),
        metadata={"chunk_type": "table_record"},
    )
    correct = SearchResult(
        chunk_id="dzw10-power",
        score=0.9,
        title="Lighting catalog",
        document_version_id="v1",
        source_document_id="d1",
        pages=[10],
        section_path=["Specifications"],
        content=(
            "Column headers: Power consumption; Row headers: Model CA-DZW10X "
            "CA-DZW30X CA-DZW50X > CA-DZW10X; Cell value: 27 W; Row: 1; Column: 2"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, support = _concise_structured_table_answer(
        "What power does the CA-DZW10X consume?",
        [wrong, correct],
    )

    assert answer.endswith("CA-DZW10X — Power consumption: 27 W")
    assert [result.chunk_id for result in support] == ["dzw10-power"]


def test_structured_table_answer_directly_answers_sold_separately_question():
    result = SearchResult(
        chunk_id="lr-tb2000-cable",
        score=1.0,
        title="Sensors for Laser Sensor",
        document_version_id="v1",
        source_document_id="d1",
        pages=[10],
        section_path=["CMOS"],
        content=(
            "Column headers: Model; Row headers: Cable (2 m6.56') M12 connector "
            "(Cable sold separately); Cell value: LR-TB2000 LR-TB2000C/LR-TB2000CL"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, support = _concise_structured_table_answer(
        "Does the LR-TB2000 come with a cable included?",
        [result],
    )

    assert answer == "No. The M12 cable is sold separately."
    assert [item.chunk_id for item in support] == ["lr-tb2000-cable"]


def test_structured_fact_answer_binds_connector_type_to_requested_model():
    unrelated = SearchResult(
        chunk_id="l-shaped-note",
        score=1.0,
        title="Sensors for Laser Sensor",
        document_version_id="v1",
        source_document_id="d1",
        pages=[9],
        section_path=["CAD DATA DOWNLOAD"],
        content=(
            "When the L-shaped connector is in use, the cable is fixed in the direction shown. "
            "The connector is not rotatable."
        ),
        metadata={"chunk_type": "text"},
    )
    matching = SearchResult(
        chunk_id="lr-tb2000-cable",
        score=0.9,
        title="Sensors for Laser Sensor",
        document_version_id="v1",
        source_document_id="d1",
        pages=[10],
        section_path=["CMOS"],
        content=(
            "Column headers: Model; Row headers: Cable (2 m6.56') M12 connector "
            "(Cable sold separately); Cell value: LR-TB2000 LR-TB2000C/LR-TB2000CL"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, support = _concise_structured_fact_answer(
        "What connector type is used on the LR-TB2000 cable?",
        [unrelated, matching],
    )

    assert answer == "The LR-TB2000 cable uses an M12 connector."
    assert [item.chunk_id for item in support] == ["lr-tb2000-cable"]


def test_structured_fact_answer_extracts_initial_output_polarity():
    result = SearchResult(
        chunk_id="w500-initial-polarity",
        score=1.0,
        title="LR-W500 instruction manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[4],
        section_path=["Initial values"],
        content=(
            "Column headers: Initial value; Row headers: Item > NPN/PNP selection; "
            "Cell value: NPN; Row: 1; Column: 1"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, support = _concise_structured_fact_answer(
        "Which output polarity does the W500 use out of the box?",
        [result],
    )

    assert answer == "The initial output polarity is NPN."
    assert [item.chunk_id for item in support] == ["w500-initial-polarity"]


def test_structured_fact_answer_extracts_complete_voltage_option_set():
    warning = SearchResult(
        chunk_id="ca-dc40e-warning",
        score=1.0,
        title="CV-X user manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[719],
        section_path=["Light controller"],
        content=(
            "Connecting a 12 V DC illumination unit when the Setting Voltage is set at "
            "24 V DC may damage the CA-DC40E."
        ),
        metadata={"chunk_type": "warning_record"},
    )
    options = SearchResult(
        chunk_id="ca-dc40e-options",
        score=0.9,
        title="CV-X user manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[719],
        section_path=["Light controller"],
        content=(
            "Select either 12V (Default) or 24V for the voltage to be supplied to the "
            "illumination unit connected to the CADC40E light controller."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, support = _concise_structured_fact_answer(
        "Which voltage options can be selected for an illumination unit connected to the CA-DC40E light controller?",
        [warning, options],
    )

    assert answer == "Select either 12 V (default) or 24 V."
    assert [item.chunk_id for item in support] == ["ca-dc40e-options"]


def test_answer_model_scope_treats_punctuation_variants_as_same_identifier():
    exact_without_hyphen = SearchResult(
        chunk_id="ca-dc40e-options",
        score=0.9,
        title="CV-X user manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[719],
        section_path=["Light controller"],
        content=(
            "Select either 12V (Default) or 24V for the voltage supplied to the "
            "CADC40E light controller."
        ),
        metadata={"chunk_type": "atomic_text"},
    )
    warning_with_hyphen = SearchResult(
        chunk_id="ca-dc40e-warning",
        score=1.0,
        title="CV-X user manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[719],
        section_path=["Light controller"],
        content="The CA-DC40E setting voltage must match the illumination unit.",
        metadata={"chunk_type": "warning_record"},
    )

    scoped = generator_module._scope_answer_results_to_query_models(
        "Which voltage options apply to the CA-DC40E light controller?",
        [exact_without_hyphen, warning_with_hyphen],
    )

    assert [item.chunk_id for item in scoped] == ["ca-dc40e-options", "ca-dc40e-warning"]


def test_structured_fact_answer_scopes_trigger_delay_to_capture_context():
    generic = SearchResult(
        chunk_id="generic-trigger-delay",
        score=1.0,
        title="XG-X user manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[985],
        section_path=["Trigger Parameters"],
        content=(
            "The trigger delay can be set in the range between 0 and 999.999 ms "
            "for each camera."
        ),
        metadata={"chunk_type": "atomic_text"},
    )
    lj_s = SearchResult(
        chunk_id="lj-s-trigger-delay",
        score=0.9,
        title="XG-X user manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[924, 925],
        section_path=["Capture Using LJ-S Series Heads"],
        content=(
            "Capture Using LJ-S Series Heads. To delay image capture, specify the Trigger "
            "Delay time. The trigger delay can be set in a range of 0 to 999 ms for each camera."
        ),
        metadata={"chunk_type": "section_window"},
    )

    range_answer, range_support = _concise_structured_fact_answer(
        "For XG-X capture using an LJ-S Series head, what trigger-delay range can be set for each camera?",
        [generic, lj_s],
    )
    procedure_answer, procedure_support = _concise_structured_fact_answer(
        "For XG-X capture using an LJ-S Series head, how do I delay image capture after the selected trigger?",
        [generic, lj_s],
    )

    assert range_answer == "The trigger delay range is 0 to 999 ms for each camera."
    assert procedure_answer == (
        "Specify the Trigger Delay time; for each camera, it can be set from 0 to 999 ms."
    )
    assert [item.chunk_id for item in range_support] == ["lj-s-trigger-delay"]
    assert [item.chunk_id for item in procedure_support] == ["lj-s-trigger-delay"]


def test_structured_fact_answer_separates_response_light_cause_and_action():
    result = SearchResult(
        chunk_id="response-light-condition",
        score=1.0,
        title="LR-W70(C) manual",
        document_version_id="v1",
        source_document_id="doc-lrw70",
        pages=[7],
        section_path=["SET"],
        content=(
            "When using the product with the [500 μ s] or [2.1 ms] response time, the "
            "indicators may be displayed if the light intensity is saturated or insufficient, "
            "respectively. It is beneficial to recalibrate the sensor, since the light intensity "
            "is automatically adjusted during calibration."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    cause, cause_support = _concise_structured_fact_answer(
        "At the 500 µs or 2.1 ms response-time settings, what conditions can make the "
        "LR-W70(C) display the saturation or insufficient-light indicators?",
        [result],
    )
    action, action_support = _concise_structured_fact_answer(
        "What corrective action does the LR-W70(C) manual recommend when those "
        "saturation/insufficient-light indicators appear at 500 µs or 2.1 ms response time?",
        [result],
    )

    assert cause == (
        "At 500 µs or 2.1 ms response time, the indicators may appear when the light "
        "intensity is saturated or insufficient."
    )
    assert action == (
        "Recalibrate the LR-W70(C); calibration automatically adjusts the light intensity."
    )
    assert [item.chunk_id for item in cause_support] == ["response-light-condition"]
    assert [item.chunk_id for item in action_support] == ["response-light-condition"]

    generated, trace = generate_answer_with_trace(
        "What corrective action does the LR-W70(C) manual recommend when the "
        "saturation/insufficient-light indicators appear at 500 µs or 2.1 ms response time?",
        [result],
    )

    assert generated.answer == action
    assert generated.citations[0]["chunk_id"] == "response-light-condition"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"

    neighboring_result = result.model_copy(
        update={
            "chunk_id": "neighboring-light-adjustment",
            "pages": [6],
            "content": (
                "In this situation, it may be possible to increase stability by adjusting "
                "the light intensity to the optimal value using the steps below."
            ),
        }
    )
    generated, trace = generate_answer_with_trace(
        "What corrective action does the LR-W70(C) manual recommend when the "
        "saturation/insufficient-light indicators appear at 500 µs or 2.1 ms response time?",
        [neighboring_result],
        prioritized_results=[result, neighboring_result],
        summarized_evidence=[],
    )

    assert generated.answer == action
    assert generated.citations[0]["chunk_id"] == "response-light-condition"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_conditional_install_answer_stops_before_neighboring_manual_labels():
    result = SearchResult(
        chunk_id="ambient-light-action",
        score=1.0,
        title="LR-ZH500C3P manual",
        document_version_id="v1",
        source_document_id="doc-lrzh",
        pages=[2],
        section_path=["OUT"],
        content=(
            "If the sensor is affected by ambient light, install a light blocking plate, "
            "or change the installation location. 3-1 Part names and functions SET: Calibration"
        ),
        metadata={"chunk_type": "section_window", "product_model": "LR-ZH500C3P"},
    )

    answer, trace = generate_answer_with_trace(
        "What should I install if ambient light affects the LR-ZH500C3P?",
        [result],
    )

    assert answer.answer == "Install a light blocking plate."
    assert answer.citations[0]["chunk_id"] == "ambient-light-action"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_event_action_prefers_condition_leading_instruction_over_neighboring_release():
    release = SearchResult(
        chunk_id="release-too-early",
        score=1.0,
        title="W500 manual",
        document_version_id="v1",
        source_document_id="doc-w500",
        pages=[2],
        section_path=["SET"],
        content="Release the button when ' SEt ' flashes.",
        metadata={"chunk_type": "atomic_text", "product_model": "W500"},
    )
    exact = release.model_copy(
        update={
            "chunk_id": "continue-holding",
            "content": (
                "When ' SEt ' flashes, continue holding the [SET] button and pass the "
                "workpiece in front of the sensor."
            ),
        }
    )

    answer, trace = generate_answer_with_trace(
        "What should I do when the SET indicator flashes on the W500?",
        [release],
        prioritized_results=[release, exact],
        summarized_evidence=[],
    )

    assert answer.answer == (
        "Continue holding the [SET] button and pass the workpiece in front of the sensor."
    )
    assert answer.citations[0]["chunk_id"] == "continue-holding"
    assert trace["final_answer"]["answer_source"] == "deterministic_event_action"


def test_calibration_instruction_binds_requested_pass_workpiece_action():
    unrelated = SearchResult(
        chunk_id="master-addition",
        score=1.0,
        title="W500 manual",
        document_version_id="v1",
        source_document_id="doc-w500",
        pages=[2],
        section_path=["Calibration"],
        content=(
            "To add an allowable range, lower the setting value, and perform the master "
            "addition calibration again."
        ),
        metadata={"chunk_type": "section_window", "product_model": "W500"},
    )
    exact = unrelated.model_copy(
        update={
            "chunk_id": "pass-workpiece",
            "section_path": ["SET"],
            "content": (
                "When ' SEt ' flashes, continue holding the [SET] button and pass the "
                "workpiece in front of the sensor."
            ),
            "metadata": {"chunk_type": "atomic_text", "product_model": "W500"},
        }
    )

    answer, trace = generate_answer_with_trace(
        "How do I pass the workpiece during W500 calibration?",
        [unrelated],
        prioritized_results=[unrelated, exact],
        summarized_evidence=[],
    )

    assert answer.answer == (
        "Continue holding the [SET] button and pass the workpiece in front of the sensor."
    )
    assert answer.citations[0]["chunk_id"] == "pass-workpiece"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_single_open_instruction_stops_before_following_programming_step():
    result = SearchResult(
        chunk_id="open-main-ob1",
        score=1.0,
        title="PLC-Link guide",
        document_version_id="v1",
        source_document_id="doc-plc",
        pages=[6],
        section_path=["Program blocks"],
        content=(
            "Under 'Program blocks,' double-click 'Main [OB1]' to open the programme block, "
            "and then enter the following ladder programme."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I open the Main OB1 program block?",
        [result],
    )

    assert answer.answer == (
        "Under 'Program blocks,' double-click 'Main [OB1]' to open the programme block."
    )
    assert answer.citations[0]["chunk_id"] == "open-main-ob1"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_entry_location_extracts_opened_program_block():
    unrelated = SearchResult(
        chunk_id="download-programme",
        score=1.0,
        title="PROFINET guide",
        document_version_id="v1",
        source_document_id="doc-profinet",
        pages=[7],
        section_path=["Transmit"],
        content="Click Finish and start the ladder programme.",
        metadata={"chunk_type": "section_window", "product_family": "PROFINET"},
    )
    exact = unrelated.model_copy(
        update={
            "chunk_id": "main-ob1-entry",
            "pages": [6],
            "section_path": ["Program blocks"],
            "content": (
                "Under 'Program blocks,' double-click 'Main [OB1]' to open the programme "
                "block, and then enter the following ladder programme."
            ),
            "metadata": {"chunk_type": "atomic_text", "product_family": "PROFINET"},
        }
    )

    answer, trace = generate_answer_with_trace(
        "Where should I enter the ladder programme for PROFINET communication?",
        [unrelated],
        prioritized_results=[unrelated, exact],
        summarized_evidence=[],
    )

    assert answer.answer == (
        "Enter the ladder programme for PROFINET communication in 'Main [OB1]' under "
        "'Program blocks'."
    )
    assert answer.citations[0]["chunk_id"] == "main-ob1-entry"
    assert trace["final_answer"]["answer_source"] == "deterministic_physical_location"


def test_concise_structured_fact_collapses_pdf_spacing_inside_sentence():
    result = SearchResult(
        chunk_id="floating-point-format",
        score=1.0,
        title="LJ-X8000 manual",
        document_version_id="v1",
        source_document_id="doc-ljx",
        pages=[438],
        section_path=["Decimal Point"],
        content=(
            "Column headers: Description; Row headers: Floating decimal point; Cell value: "
            "uses the measured result in  the data memory as a single-precision "
            "floating-point data (32 bits).; Row: 1; Column: 2"
        ),
        metadata={"chunk_type": "table_record", "product_model": "LJ-X8000"},
    )

    answer, trace = generate_answer_with_trace(
        "What output format does the LJ-X8000 use when Decimal Point is set to Floating-point?",
        [result],
    )

    assert "in  the" not in answer.answer
    assert "single-precision floating-point" in answer.answer
    assert answer.citations[0]["chunk_id"] == "floating-point-format"


def test_answer_cleanup_repairs_queried_hyphenated_model_split_by_source_colon():
    result = SearchResult(
        chunk_id="display-cable-limit",
        score=1.0,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[7],
        section_path=["General precautions"],
        content=(
            "When using the display expansion unit (IV4: DU10), make sure the length of "
            "the cable used for connecting to the external device is less than 30 meters long."
        ),
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )

    answer, trace = generate_answer_with_trace(
        "How long can the connection cable be when using the IV4-DU10?",
        [result],
    )

    assert "IV4-DU10" in answer.answer
    assert "IV4: DU10" not in answer.answer
    assert "less than 30 meters" in answer.answer
    assert answer.citations[0]["chunk_id"] == "display-cable-limit"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "Which Windows 10 editions are supported by XG-H1XA?",
            "Microsoft Windows 10 Home, Pro, and Enterprise are supported; only the 64-bit versions are supported.",
        ),
        (
            "How much additional free disk space does XG-H1XA require if Microsoft .NET Framework must be installed?",
            "4.5 GB or more of additional free disk space is required.",
        ),
    ],
)
def test_structured_system_requirement_answers_use_exact_answer_bearing_row(query, expected_answer):
    heading = SearchResult(
        chunk_id="xg-h1xa-heading",
        score=1.0,
        title="3D Vision",
        document_version_id="v1",
        source_document_id="doc-3d",
        pages=[22],
        section_path=["Specifications"],
        content="Supported OS and recommended running environment for XG: H1XA",
        metadata={"chunk_type": "spec_record"},
    )
    requirements = SearchResult(
        chunk_id="xg-h1xa-requirements",
        score=0.9,
        title="3D Vision",
        document_version_id="v1",
        source_document_id="doc-3d",
        pages=[22],
        section_path=["Specifications"],
        content=(
            "Supported OS: Microsoft Windows 10 Home, Pro, Enterprise (64 bit version). "
            "If installation of Microsoft .NET Framework is necessary, 4.5 GB or more of free "
            "space is required in addition to the above."
        ),
        metadata={"chunk_type": "table_record", "identifier_page_sibling": True},
    )

    answer, trace = generate_answer_with_trace(query, [heading, requirements])

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "xg-h1xa-requirements"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "What scan speed does the WM-P6200 support?",
            "Scan speed — WM-P6200: Up to 2.4 million points per second",
        ),
        (
            "Which laser classification applies to the WM-P6200?",
            "Laser classification — WM-P6200: 2M",
        ),
        (
            "How many LEDs does the CA-DQP12X LumiTrax light source contain?",
            "Number of LEDs — CA-DQP12X: 72",
        ),
        (
            "What is the color output of the CA-DQP25X LumiTrax light source?",
            "Pattern — CA-DQP25X: White",
        ),
        (
            "How many images can the IV4-G120 store in its history?",
            "Numbers — IV4-G120: 100 images",
        ),
    ],
)
def test_structured_model_field_answer_stops_before_neighboring_field(query, expected_answer):
    result = SearchResult(
        chunk_id="wm-specifications",
        score=1.0,
        title="WM brochure",
        document_version_id="v1",
        source_document_id="doc-wm",
        pages=[42],
        section_path=["Specifications"],
        content=(
            "Model: Pattern; CA-DQP12X: Color; CA-DQP25X: White "
            "Model: Number of LEDs; CA-DQP12X: 72; CA-DQP25X: 168 "
            "Model: Numbers; IV4-G120: 100 images "
            "Model: Scan speed; WM-P6200: Up to 2.4 million points per second "
            "Model: Laser classification; WM-P6200: 2M Display | Display method | "
            "3.5-inch transmissive TFT color LCD Model: Resolution; WM-P6200: 640 × 480 pixels"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, trace = generate_answer_with_trace(query, [result])

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "wm-specifications"
    assert trace["final_answer"]["answer_source"] in {
        "deterministic_structured_fact",
        "structured_evidence",
    }


def test_count_fact_prefers_numeric_capacity_over_neighboring_condition_row():
    condition = SearchResult(
        chunk_id="condition",
        score=1.1,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[446],
        section_path=["Image history"],
        content=(
            "Column headers: IV4-G120; Row headers: Image history > Condition; "
            "Cell value: Logging Settings 1: OK only, NG only, None."
        ),
        metadata={"chunk_type": "table_record"},
    )
    capacity = SearchResult(
        chunk_id="capacity",
        score=1.0,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[484],
        section_path=["Specifications"],
        content="Model: Numbers; IV4-G120: 100 images",
        metadata={"chunk_type": "table_record"},
    )

    answer, trace = generate_answer_with_trace(
        "How many images can the IV4-G120 store in its history?",
        [condition, capacity],
    )

    assert answer.answer == "Numbers — IV4-G120: 100 images"
    assert answer.citations[0]["chunk_id"] == "capacity"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_named_field_fact_prefers_transfer_destination_over_transfer_condition():
    condition = SearchResult(
        chunk_id="condition",
        score=1.1,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[484],
        section_path=["Image transfer"],
        content=(
            "Model: Transfer Condition; IV4-G120: OK/NG/near threshold OK/All are selectable"
        ),
        metadata={"chunk_type": "table_record"},
    )
    destination = SearchResult(
        chunk_id="destination",
        score=1.0,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[484],
        section_path=["Image transfer"],
        content=(
            "Column headers: IV4-G120; Row headers: Image data transfer > Transfer Destination; "
            "Cell value: microSD card/FTP server/SFTP server is selectable"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, trace = generate_answer_with_trace(
        "Which transfer destinations are selectable for the IV4-G120?",
        [condition, destination],
    )

    assert "microSD card/FTP server/SFTP server" in answer.answer
    assert answer.citations[0]["chunk_id"] == "destination"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "What is the valid STO pulse duration range for the CV-X482?",
            "The valid STO pulse duration range is 1 to 999 ms.",
        ),
        (
            "What is the default STO output duration on the CV-X482?",
            "The default STO output duration is 10 ms.",
        ),
    ],
)
def test_signal_duration_answers_bind_range_and_default(query, expected_answer):
    result = SearchResult(
        chunk_id="sto-duration",
        score=1.0,
        title="CV-X manual",
        document_version_id="v1",
        source_document_id="doc-cvx",
        pages=[865],
        section_path=["STO/ACK/NACK Output Duration"],
        content=(
            "Set the length of time from when the STO rises to when the STO falls within the "
            "range of 1 to 999 (ms). (Default: 10 ms)"
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "CV-X482"},
    )

    answer, trace = generate_answer_with_trace(query, [result])

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "sto-duration"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "Does selecting 300 µs or 1.1 ms response time reduce stability on the LR-W70(C)?",
            "Yes. With the 300 μs or 1.1 ms response time selected, stable operation may be reduced.",
        ),
        (
            "How can I improve stability if the LR-W70(C) response time is set to 300 µs?",
            "Adjust the light intensity to the optimal value to increase stability.",
        ),
    ],
)
def test_response_time_stability_answers_preserve_polarity_and_remedy(query, expected_answer):
    result = SearchResult(
        chunk_id="response-stability",
        score=1.0,
        title="LR-W70(C) manual",
        document_version_id="v1",
        source_document_id="doc-lrw",
        pages=[6],
        section_path=["RUN"],
        content=(
            "When using the product with the [300 μ s *1 ] or [1.1 ms *2 ] response time selected, "
            "stable operation may be reduced. In this situation, it may be possible to increase "
            "stability by adjusting the light intensity to the optimal value using the steps below."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "LR-W70(C) Edition"},
    )

    answer, trace = generate_answer_with_trace(query, [result])

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "response-stability"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "What is the maximum input voltage for the IV-HG500CA with PNP polarity?",
            "With PNP polarity selected, the maximum input voltage is 26.4 V.",
        ),
        (
            "Does selecting PNP polarity make the IV-HG500CA inputs voltage-based?",
            "Yes. Selecting PNP makes the circuit a voltage input circuit.",
        ),
    ],
)
def test_polarity_answers_bind_electrical_rating_and_circuit_type(query, expected_answer):
    troubleshooting = SearchResult(
        chunk_id="pnp-troubleshooting",
        score=1.1,
        title="IV-HG manual",
        document_version_id="v1",
        source_document_id="doc-ivhg",
        pages=[403],
        section_path=["Troubleshooting"],
        content=(
            "When the PNP is selected in the Polarity, the circuit becomes a voltage input circuit. "
            "Check the cables."
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
    )
    result = SearchResult(
        chunk_id="pnp-input",
        score=1.0,
        title="IV-HG manual",
        document_version_id="v1",
        source_document_id="doc-ivhg",
        pages=[58],
        section_path=["IN1 - IN6"],
        content=(
            "When PNP is selected in the Polarity, the circuit becomes voltage input circuit. "
            "Input maximum rating : 26.4 V"
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV-HG500CA"},
    )

    evidence = [troubleshooting, result] if "maximum" in query.lower() else [result]
    answer, trace = generate_answer_with_trace(query, evidence)

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "pnp-input"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "How do I set the IP address for the SR-PN1?",
            "Double-click the SR-PN1 image, open General > PROFINET interface > Ethernet addresses, "
            "and set the SR-PN1 IP address there.",
        ),
        (
            "Which menu path leads to the Ethernet addresses for the SR-PN1?",
            "Location: SR-PN1 image > General tab > PROFINET interface > Ethernet addresses.",
        ),
    ],
)
def test_ip_address_action_chain_preserves_complete_menu_path(query, expected_answer):
    result = SearchResult(
        chunk_id="sr-pn1-ip",
        score=1.0,
        title="SR-PN1 manual",
        document_version_id="v1",
        source_document_id="doc-sr",
        pages=[4],
        section_path=["Configuring Siemens PLC settings"],
        content=(
            "Double-click the SR-PN1 image, click the 'General' tab, click 'PROFINET interface,' "
            "click 'Ethernet addresses,' and then set the IP address of the SR-PN1."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(query, [result])

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "sr-pn1-ip"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "When must I enable reference point monitoring for SZ-V access protection?",
            "Enable reference-point monitoring when SZ-V is used for access protection under "
            "IEC61496-3: 2008 Annex A.12 and A.13, where the approach angle exceeds ±30° to the detection plane.",
        ),
        (
            "What response time limit applies to SZ-V when using reference point monitoring?",
            "With reference-point monitoring, the response time must be 90 ms or less.",
        ),
    ],
)
def test_reference_point_monitoring_answers_bind_condition_and_response_limit(query, expected_answer):
    result = SearchResult(
        chunk_id="reference-monitoring",
        score=1.0,
        title="SZ-V manual",
        document_version_id="v1",
        source_document_id="doc-szv",
        pages=[29],
        section_path=["SZ-V"],
        content=(
            "Reference points monitoring function must be applied when the SZ-V is used for the access "
            "protection specified in IEC61496-3: 2008 Annex A.12 and A.13 (the application where the angle "
            "of the approach exceeds ±30° to the detection plane). In this case, the tolerance for reference "
            "points must be ±100 mm or less and the response time must be 90 ms or less."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(query, [result])

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "reference-monitoring"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "How does the Rough Search Characteristic Reduction Rate affect feature compression?",
            "Rough Search Characteristic Reduction Rate: Specify the degree of compression of features "
            "extracted from the reference image and the input image in coarse search. "
            "0 (shrinkage: small) to 10 (shrinkage: large)",
        ),
        (
            "What range is available for the Rough Search Characteristic Reduction Rate?",
            "Rough Search Characteristic Reduction Rate ranges from 0 (shrinkage: small) "
            "to 10 (shrinkage: large).",
        ),
    ],
)
def test_explicit_multiword_setting_answers_preserve_behavior_and_range(query, expected_answer):
    result = SearchResult(
        chunk_id="rough-reduction",
        score=1.0,
        title="LJ-X8000 manual",
        document_version_id="v1",
        source_document_id="doc-ljx",
        pages=[250],
        section_path=["Settings"],
        content=(
            "Setting item: Rough Search Characteristic Reduction Rate; Settings: Specify the degree "
            "of compression of features extracted from the reference image and the input image in "
            "coarse search. 0 (shrinkage: small) to 10 (shrinkage: large) Context"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, trace = generate_answer_with_trace(query, [result])

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "rough-reduction"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_private_use_pdf_glyphs_are_removed_from_final_answer():
    result = SearchResult(
        chunk_id="small-defects",
        score=1.0,
        title="CV-X manual",
        document_version_id="v1",
        source_document_id="doc-cvx",
        pages=[162],
        section_path=["Intensity Threshold Level"],
        content=(
            "Small defects can no longer be detected if the Intensity Threshold Level is increased. "
            "\uf071 Decrease the Segment Size. Set the \uf020 Segment Size to the equivalent size of the "
            "smallest defect that you want to detect."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, _trace = generate_answer_with_trace(
        "How do I restore detection of small defects after raising the Intensity Threshold Level?",
        [result],
    )

    assert "\uf071" not in answer.answer
    assert "\uf020" not in answer.answer
    assert "Decrease the Segment Size" in answer.answer


def test_structured_model_field_prefers_exact_labeled_cell_and_citation():
    exact = SearchResult(
        chunk_id="wm-laser-classification",
        score=1.0,
        title="WM brochure",
        document_version_id="v1",
        source_document_id="doc-wm",
        pages=[42],
        section_path=["Specifications"],
        content=(
            "Column headers: WM-P6200; Row headers: Laser classification; "
            "Cell value: 2M; Row: 6; Column: 2"
        ),
        metadata={"chunk_type": "table_record"},
    )
    heading = SearchResult(
        chunk_id="wm-heading",
        score=0.9,
        title="WM brochure",
        document_version_id="v1",
        source_document_id="doc-wm",
        pages=[43],
        section_path=["Specifications"],
        content="Laser: scanning probe WM-P6200",
        metadata={"chunk_type": "spec_record"},
    )

    answer, trace = generate_answer_with_trace(
        "Which laser classification applies to the WM-P6200?",
        [exact, heading],
    )

    assert answer.answer == "Laser classification — WM-P6200: 2M"
    assert answer.citations[0]["chunk_id"] == "wm-laser-classification"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "Which UL safety standards apply to the CA-U5 power supply?",
            "The applicable UL safety standards are UL508 and UL60950-1.",
        ),
        (
            "What EN safety standards does the CA-U5 compact switch-mode power supply meet?",
            "The applicable EN safety standards are EN62368-1 and EN50178.",
        ),
    ],
)
def test_named_safety_standards_do_not_match_ultra_compact_description(
    query,
    expected_answer,
):
    description = SearchResult(
        chunk_id="product-description",
        score=1.0,
        title="CA-U5 datasheet",
        document_version_id="v1",
        source_document_id="doc-cau5",
        pages=[1],
        section_path=["CA-U5"],
        content="Ultra: compact Switch-mode power supply",
        metadata={"chunk_type": "datasheet_record", "product_model": "CA-U5"},
    )
    standards = description.model_copy(
        update={
            "chunk_id": "safety-standards",
            "content": (
                    "Column headers: CA-U5; Row headers: Applicable standards > Safety standards; "
                    "Cell value: UL: UL508, UL60950-1 C-UL: CSA C22.2 No.14, "
                    "CSA C22.2 No.60950-1 EN: EN62368-1, EN50178 "
                    "IEC: IEC62368-1; Row: 8; Column: 2"
            ),
            "metadata": {"chunk_type": "table_record", "product_model": "CA-U5"},
        }
    )

    answer, trace = generate_answer_with_trace(
        query,
        [description],
        prioritized_results=[standards, description],
        summarized_evidence=[],
    )

    assert answer.answer == expected_answer
    assert answer.citations[0]["chunk_id"] == "safety-standards"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_selection_impact_uses_the_exact_named_option_row():
    wrong = SearchResult(
        chunk_id="min-match",
        score=1.0,
        title="LJ-X8000 manual",
        document_version_id="v1",
        source_document_id="doc-ljx",
        pages=[254],
        section_path=["Pattern Search"],
        content=(
            "Setting item: Min. Match%; Settings: Patterns below the lower limit are "
            "excluded from measurement candidates."
        ),
        metadata={"chunk_type": "section_window", "product_model": "LJ-X8000"},
    )
    overlap = wrong.model_copy(
        update={
            "chunk_id": "overlap-removal",
            "pages": [248],
            "content": (
                "Setting item: Overlap Removal; Settings: When multiple search results "
                "overlap, removes candidates according to the removal target. "
                "- Non-Max. Match Candidate: Of those multiple candidates which overlap, "
                "leaves the one with the highest match %. - All: Removes all of them."
            ),
            "metadata": {"chunk_type": "table_record", "product_model": "LJ-X8000"},
        }
    )

    answer, trace = generate_answer_with_trace(
        "What happens to overlapping candidates if I select Non-Max. Match Candidate on the LJ-X8000?",
        [wrong, overlap],
    )

    assert "leaves the one with the highest match %" in answer.answer
    assert "Min. Match%" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "overlap-removal"
    assert trace["final_answer"]["answer_source"] == "deterministic_capability"


def test_enclosure_and_material_questions_extract_only_requested_fields():
    result = SearchResult(
        chunk_id="iv-enclosure-materials",
        score=1.0,
        title="IV-500C manual",
        document_version_id="v1",
        source_document_id="doc-iv",
        pages=[9],
        section_path=["Specifications"],
        content=(
            "Column headers: IV-500C; Row headers: Enclosure rating Material Weight; "
            "Cell value: IP67 Main unit case : Aluminum die-casting, Packing : NBR, "
            "Front Cover : Acrylic, Mounting adapter : POM Approx. 270 g; Row: 32; Column: 2"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV-500C"},
    )

    rating, rating_trace = generate_answer_with_trace(
        "What is the enclosure rating for the IV-500C main unit?",
        [result],
    )
    materials, materials_trace = generate_answer_with_trace(
        "Which materials compose the IV-500C main unit case and front cover?",
        [result],
    )

    assert rating.answer == "The enclosure rating is IP67."
    assert materials.answer == (
        "The main unit case is Aluminum die-casting, and the front cover is Acrylic."
    )
    assert rating.citations[0]["chunk_id"] == "iv-enclosure-materials"
    assert materials.citations[0]["chunk_id"] == "iv-enclosure-materials"
    assert rating_trace["final_answer"]["answer_source"] == "deterministic_structured_fact"
    assert materials_trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_wire_requirements_extract_temperature_and_cross_section_separately():
    result = SearchResult(
        chunk_id="wire-requirements",
        score=1.0,
        title="Power wiring manual",
        document_version_id="v1",
        source_document_id="doc-power",
        pages=[55],
        section_path=["Wiring"],
        content=(
            "Use a wire with 60°C or higher temperature rating as the electrical wire.: "
            "Follow the instructions below. • The nominal cross-sectional area of the wire "
            "for connecting the push type terminal block should be 0.2mm 2 to 1.5mm 2 "
            "(AWG16 to 26). • The stripped part should not be soldered."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    temperature, _temperature_trace = generate_answer_with_trace(
        "What temperature rating is required for the electrical wire?",
        [result],
    )
    area, _area_trace = generate_answer_with_trace(
        "What wire cross-sectional area is needed for the push type terminal block?",
        [result],
    )

    assert temperature.answer == "Use electrical wire rated for 60°C or higher."
    assert area.answer == (
        "Use wire with a nominal cross-sectional area of 0.2mm² to 1.5mm² (AWG 16 to 26)."
    )
    assert temperature.citations[0]["chunk_id"] == "wire-requirements"
    assert area.citations[0]["chunk_id"] == "wire-requirements"


def test_model_specific_cable_length_and_weight_ignore_neighboring_rows():
    exact_record = SearchResult(
        chunk_id="sz-vp10pw-record",
        score=0.9,
        title="SZ-V manual",
        document_version_id="v1",
        source_document_id="doc-szv",
        pages=[18],
        section_path=["Cable specifications"],
        content=(
            "Type: Power cable when using PROFIsafe or CIP Safety; Length: 10 m32.81'; "
            "Model: SZ-VP10PW; Weight: Approx. 650 g\n"
            "Type: M12 quick disconnect; Length: 0.3 m0.98'; Model: SZ-VPC03; "
            "Weight: Approx. 80 g"
        ),
        metadata={"chunk_type": "table_record", "product_family": "SZ-V"},
    )
    table_window = exact_record.model_copy(
        update={
            "chunk_id": "sz-v-cable-table",
            "content": (
                "Type | Length | Model | Weight\n"
                "Standard | 10 m32.81' | SZ-VP10 | Approx. 800 g\n"
                "Power cable when using PROFIsafe or CIP Safety | 10 m32.81' | "
                "SZ-VP10PW | Approx. 650 g"
            ),
            "metadata": {"chunk_type": "section_window", "product_family": "SZ-V"},
        }
    )

    length, _length_trace = generate_answer_with_trace(
        "What is the length of the SZ-VP10PW power cable for PROFIsafe?",
        [exact_record],
    )
    weight, _weight_trace = generate_answer_with_trace(
        "How much does the SZ-VP10PW safety power cable weigh?",
        [table_window],
    )

    assert length.answer == "The SZ-VP10PW cable length is 10 m (32.81 ft)."
    assert weight.answer == "The SZ-VP10PW cable weighs Approx. 650 g."
    assert length.citations[0]["chunk_id"] == "sz-vp10pw-record"
    assert weight.citations[0]["chunk_id"] == "sz-v-cable-table"


def test_named_setting_range_binds_value_after_label_not_previous_field():
    preceding_field = SearchResult(
        chunk_id="number-of-characters",
        score=1.0,
        title="VS manual",
        document_version_id="v1",
        source_document_id="doc-vs",
        pages=[740],
        section_path=["AI OCR"],
        content=(
            "Number of Characters. Lower Limit. Unit: Number of Characters. Range: 0 to 128. "
            "Match Degree Set the lower limit for the match degree."
        ),
        metadata={"chunk_type": "section_window", "product_family": "VS Series"},
    )
    exact = preceding_field.model_copy(
        update={
            "chunk_id": "match-degree-range",
            "pages": [741],
            "content": (
                "Match Degree. Lower Limit Set the lower limit for the match degree of detected "
                "characters. Unit: Number. Range: 0 to 99. Display/Hide Judgment Conditions."
            ),
        }
    )

    answer, trace = generate_answer_with_trace(
        "What is the valid range for the Match Degree setting in VS Series AI OCR?",
        [preceding_field, exact],
    )

    assert answer.answer == "The valid range for the Match Degree setting is 0 to 99."
    assert answer.citations[0]["chunk_id"] == "match-degree-range"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_model_field_record_treats_mass_as_weight_not_unit_cell_size():
    cell_size = SearchResult(
        chunk_id="unit-cell-size",
        score=1.0,
        title="VJ-H500CX specifications",
        document_version_id="v1",
        source_document_id="doc-vj",
        pages=[1],
        section_path=["Specifications"],
        content="Model: Unit cell size; VJ-H500CX: 3.45 µm × 3.45 µm",
        metadata={"chunk_type": "table_record", "product_model": "VJ-H500CX"},
    )
    weight = cell_size.model_copy(
        update={
            "chunk_id": "weight",
            "content": (
                "VJ-H500CX Data Sheet\nModel | | VJ-H500CX\nUnit cell size | | "
                "3.45 µm × 3.45 µm\nWeight | | Approx. 280 g (not including lens)"
            ),
            "metadata": {"chunk_type": "section_window", "product_model": "VJ-H500CX"},
        }
    )

    answer, trace = generate_answer_with_trace(
        "What is the approximate mass of the VJ-H500CX unit?",
        [cell_size, weight],
    )

    assert answer.answer == "The VJ-H500CX weighs approximately 280 g (not including lens)."
    assert "3.45" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "weight"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_structured_table_answer_binds_current_to_requested_model_column():
    result = SearchResult(
        chunk_id="dzw-model-table",
        score=1.0,
        title="LineScan C-611J12 installation guide",
        document_version_id="v1",
        source_document_id="d1",
        pages=[17],
        section_path=["CA-EN100U"],
        content=(
            "Model | CA-DZW10X | CA-DZW30X | CA-DZW50X\n"
            "Power consumption | 30 W(1.3 A) | 69 W(2.9 A) | 116 W(4.9 A)"
        ),
        metadata={"chunk_type": "section_window"},
    )

    answer, support = _concise_structured_table_answer(
        "How much current does the CA-DZW30X draw?",
        [result],
    )

    assert answer == "The CA-DZW30X draws 2.9 A."
    assert [item.chunk_id for item in support] == ["dzw-model-table"]


def test_structured_table_answer_binds_physical_terminal_to_its_function():
    result = SearchResult(
        chunk_id="a1-record",
        score=1.0,
        title="IV Series Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[26],
        section_path=["Specifications"],
        content=(
            "Terminal No.: A1; Wiring color: Brown; Name: IN1; Assigning default value: "
            "External trigger; Description: Set external trigger. Rising timing or falling timing "
            "can be set. Terminal No.: A2; Wiring color: Red; Name: IN2; Assigning default value: OFF"
        ),
        metadata={"chunk_type": "table_record", "product_family": "IV Series"},
    )

    answer, support = _concise_structured_table_answer(
        "What function does the brown A1 terminal perform on the IV Series?",
        [result],
    )

    assert "Terminal: A1" in answer
    assert "Wiring color: Brown" in answer
    assert "Name: IN1" in answer
    assert "External trigger" in answer
    assert "Terminal: A2" not in answer
    assert [item.chunk_id for item in support] == ["a1-record"]


def test_troubleshooting_descriptive_anchor_matches_cause_when_display_is_symbol_font():
    result = SearchResult(
        chunk_id="overcurrent-cause",
        score=1.0,
        title="LR-W70(C) Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[12],
        section_path=["Troubleshooting"],
        content=(
            "Column headers: Cause; Row headers: \uf045\uf072\uf043; Cell value: "
            "Excessive current (ov ercurrent) is f lowing through the output wire.; "
            "Row: 2; Column: 1"
        ),
        metadata={"chunk_type": "table_record", "table_row": 2},
    )

    answer, support = _concise_troubleshooting_answer(
        'What causes the "Excessive current" error on the LR-W70(C) Edition?',
        [result],
    )

    validated = validate_answer(
        AnswerResponse(
            answer=answer,
            confidence="high",
            used_documents=[],
            citations=[],
            warnings=[],
            followup_questions=[],
            insufficient_evidence=False,
        ),
        support,
        query='What causes the "Excessive current" error on the LR-W70(C) Edition?',
    )

    assert validated.answer == "Cause: Excessive current (overcurrent) is flowing through the output wire."
    assert [item.chunk_id for item in support] == ["overcurrent-cause"]


def test_structured_measurement_ignores_header_whose_context_has_neighbor_value():
    header = SearchResult(
        chunk_id="temperature-header",
        score=1.0,
        title="LJ-S8000 Catalog",
        document_version_id="v1",
        source_document_id="d1",
        pages=[33],
        section_path=["Specifications"],
        content="Table header: Environmental Operating ambient temperature* 12;",
        metadata={
            "chunk_type": "table_record",
            "product_family": "LJ-S8000 Series",
            "context_window": "Operating ambient temperature: 0 to +45°C",
        },
    )
    value = SearchResult(
        chunk_id="temperature-value",
        score=0.9,
        title="LJ-S8000 Catalog",
        document_version_id="v1",
        source_document_id="d1",
        pages=[33],
        section_path=["Specifications"],
        content="Environmental Operating ambient temperature: 0 to +45°C (32°F to 113°F)",
        metadata={"chunk_type": "table_record", "product_family": "LJ-S8000 Series"},
    )

    answer, support = _concise_structured_fact_answer(
        "What ambient operating temperature range is specified for the IP65 LJ-S8000 sensor head?",
        [header, value],
    )

    assert "0 to +45°C" in answer
    assert [item.chunk_id for item in support] == ["temperature-value"]


def test_structured_measurement_uses_direct_component_disambiguator():
    controller = SearchResult(
        chunk_id="controller-temperature",
        score=1.0,
        title="LJ-S8000 Catalog",
        document_version_id="v1",
        source_document_id="d1",
        pages=[32],
        section_path=["Controller"],
        content=(
            "Environmental — Operating ambient temperature: 0 to +45°C (DIN rail mount) / "
            "0 to +40°C (base surface mount)"
        ),
        metadata={"chunk_type": "table_record", "product_family": "LJ-S8000 Series"},
    )
    sensor_head = SearchResult(
        chunk_id="head-temperature",
        score=0.9,
        title="LJ-S8000 Catalog",
        document_version_id="v1",
        source_document_id="d1",
        pages=[33],
        section_path=["Sensor head"],
        content=(
            "Sensor head enclosure rating: IP65 (IEC60529). Environmental operating ambient "
            "temperature: 0 to +45°C (32°F to 113°F)."
        ),
        metadata={"chunk_type": "table_record", "product_family": "LJ-S8000 Series"},
    )

    answer, support = _concise_structured_fact_answer(
        "What ambient operating temperature range is specified for the IP65 LJ-S8000 sensor head?",
        [controller, sensor_head],
    )

    assert "0 to +45°C" in answer
    assert "DIN rail" not in answer
    assert [item.chunk_id for item in support] == ["head-temperature"]


def test_structured_table_answer_focuses_requested_labeled_cell_without_neighbor_context():
    result = SearchResult(
        chunk_id="cable-length",
        score=0.9,
        title="Cable Catalog",
        document_version_id="v1",
        source_document_id="d1",
        pages=[33],
        section_path=["Cables"],
        content=(
            "Column headers: Cable length (m feet); Row headers: CA-D3P; "
            "Cell value: 3 9.8'; Row: 1; Column: 1"
        ),
        metadata={
            "chunk_type": "table_record",
            "context_window": "Part number: CA-D3X; Cable length: 3 m; Part number: CA-D5X; Cable length: 5 m",
        },
    )

    answer, _trace = generate_answer_with_trace("How long is the cable on the CA-D3P model?", [result])

    assert answer.answer == "CA-D3P — Cable length (m feet): 3 9.8'"


def test_named_selection_answer_returns_explicit_mode_from_matching_sentence():
    result = SearchResult(
        chunk_id="width-mode",
        score=0.9,
        title="Width tool",
        document_version_id="v1",
        source_document_id="d1",
        pages=[167],
        section_path=["Width Mode"],
        content=(
            "Select the width extraction from the [Width Mode] pull-down menu. "
            "When extracting the width which is close to the one registered in the master image, "
            "select [Master Width]. When extracting from outside, select [Outside]."
        ),
        metadata={"chunk_type": "parent_section"},
    )

    answer, trace = generate_answer_with_trace(
        "Which width extraction mode compares against the master image?",
        [result],
    )

    assert answer.answer == "The width extraction mode is Master Width."
    assert answer.citations[0]["chunk_id"] == "width-mode"
    assert trace["final_answer"]["answer_source"] == "deterministic_named_selection"


def test_named_option_behavior_answer_returns_labeled_definition_not_heading():
    result = SearchResult(
        chunk_id="width-options",
        score=0.9,
        title="Width tool",
        document_version_id="v1",
        source_document_id="d1",
        pages=[166],
        section_path=["Width Extraction"],
        content=(
            "Items | Description | Setting range | Default value\n"
            "Width Extraction | Select a method (Width Mode) to extract the target width. | "
            "y Master Width Detects the width which is close to the one registered in the master image. "
            "y Outside Extracts width from outside in tool window. "
            "y Inside Extracts width from center in tool window. | Master Width\n"
            "Select [Outside] when searching from the outside of the tool window."
        ),
        metadata={"chunk_type": "section_window"},
    )

    answer, trace = generate_answer_with_trace(
        "How does the Outside width extraction method work?",
        [result],
    )

    assert answer.answer == (
        "The Outside width extraction method extracts width from outside in tool window."
    )
    assert trace["final_answer"]["answer_source"] == "deterministic_named_option_behavior"


def test_named_alternative_answer_preserves_explicit_options_and_selection_condition():
    result = SearchResult(
        chunk_id="auto-mode",
        score=0.9,
        title="LR-W500",
        document_version_id="v1",
        source_document_id="d1",
        pages=[1],
        section_path=["Detection mode"],
        content=(
            "Detection mode: Auto (default); Explanation: When adjusting the sensitivity, "
            "the optimal mode is automatically selected between C+I or C."
        ),
        metadata={"chunk_type": "table_record"},
    )

    for query in (
        "How does the W500 Auto detection mode select the optimal setting?",
        "Which detection modes does the W500 Auto setting choose between?",
    ):
        answer, trace = generate_answer_with_trace(query, [result])

        assert "When adjusting the sensitivity" in answer.answer
        assert "selected between C+I or C" in answer.answer
        assert answer.citations[0]["chunk_id"] == "auto-mode"
        assert trace["final_answer"]["answer_source"] == "deterministic_named_alternative"


def test_required_setting_answer_returns_alignment_instruction_without_neighbor_warning():
    generic_result = SearchResult(
        chunk_id="generic-voltage-caution",
        score=0.95,
        title="CA-DC40E",
        document_version_id="v0",
        source_document_id="d0",
        pages=[5],
        section_path=["CAUTION"],
        content=(
            "Table summary: CAUTION | Make sure to set the setting voltage for the illumination unit "
            "of the CA-DC40E light controller correctly. Connecting a 12 V DC illumination unit when "
            "the setting voltage is 24 V DC may cause damage."
        ),
        metadata={"chunk_type": "table_record"},
    )
    result = SearchResult(
        chunk_id="voltage-caution",
        score=0.9,
        title="CA-DC40E",
        document_version_id="v1",
        source_document_id="d1",
        pages=[5],
        section_path=["CAUTION"],
        content=(
            "Make sure to set the Voltage Output for the illumination unit of the CA-DC40E light "
            "controller correctly. Connecting a 12 V DC illumination unit when the Voltage Output "
            "is set at 24 V DC may cause fire, electric shock, or damage."
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, trace = generate_answer_with_trace(
        "Which voltage setting must match the illumination unit for the CA-DC40E?",
        [generic_result, result],
    )

    assert answer.answer == (
        "Make sure to set the Voltage Output for the illumination unit of the CA-DC40E light "
        "controller correctly."
    )
    assert "12 V" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "deterministic_required_setting"


def test_troubleshooting_display_code_binds_cause_to_exact_key_value_row():
    wrong = SearchResult(
        chunk_id="pulsing-bar",
        score=0.95,
        title="LR-W500",
        document_version_id="v1",
        source_document_id="d1",
        pages=[4],
        section_path=[],
        content=(
            "Display: - (The bar pulses across the display.); "
            "Cause: The display selection is set to OFF.; Solution: Set the display selection to ON."
        ),
        metadata={"chunk_type": "table_record"},
    )
    correct = SearchResult(
        chunk_id="uuu",
        score=0.9,
        title="LR-W500",
        document_version_id="v1",
        source_document_id="d1",
        pages=[4],
        section_path=[],
        content=(
            "Display: uuu; Cause: Displayed when excessive light is received by the sensor "
            "(Auto/C+I/C modes); Solution: Adjust the sensor's installation angle so that "
            "specular reflections do not enter the receiver."
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, _trace = generate_answer_with_trace(
        "What causes the W500 to display uuu?",
        [wrong, correct],
    )

    assert "excessive light" in answer.answer
    assert "display selection is set to OFF" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "uuu"
    assert "CA-D3X" not in answer.answer


def test_troubleshooting_display_code_selects_exact_row_from_multirow_record():
    result = SearchResult(
        chunk_id="w500-table",
        score=0.9,
        title="LR-W500",
        document_version_id="v1",
        source_document_id="d1",
        pages=[4],
        section_path=["Troubleshooting"],
        content=(
            "Display: ErE; Cause: The memory has reached its end of life.; "
            "Solution: Perform initialization. "
            "Display: uuu; Cause: Displayed when excessive light is received by the sensor; "
            "Solution: Adjust the sensor's installation angle so that specular reflections do not enter the receiver. "
            "Display: nnn; Cause: Displayed when insufficient light is received by the sensor; "
            "Solution: Check whether the detection distance is within the specified range."
        ),
        metadata={"chunk_type": "table_record", "product_model": "W500"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I fix the uuu error on the W500?",
        [result],
    )

    assert answer.answer == (
        "Corrective action: Adjust the sensor's installation angle so that specular reflections "
        "do not enter the receiver."
    )
    assert answer.citations[0]["chunk_id"] == "w500-table"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_troubleshooting_display_code_uses_remedy_from_cell_row_hierarchy():
    result = SearchResult(
        chunk_id="uuu-control-output-cell",
        score=0.9,
        title="LR-ZH500C3P",
        document_version_id="v1",
        source_document_id="d1",
        pages=[4],
        section_path=["Troubleshooting"],
        content=(
            "Column headers: Control Output; Row headers: uuu > Excessive reflected light > "
            "Adjust the installation angle of the sensor.; Cell value: Inconsistent; "
            "Row: 5; Column: 3"
        ),
        metadata={"chunk_type": "table_record", "product_model": "LR-ZH500C3P"},
    )

    wrong = result.model_copy(
        update={
            "chunk_id": "no-display-control-output-cell",
            "content": (
                "Column headers: Control Output; Row headers: No display or indicators > "
                "The sensor is not turned on > • Check the power voltage and power capacity. "
                "• Check the sensor power cable.; Cell value: Inconsistent; Row: 12; Column: 3"
            ),
        }
    )

    answer, trace = generate_answer_with_trace(
        "How do I fix the uuu display on the LR-ZH500C3P sensor?",
        [wrong, result],
    )

    assert answer.answer == "Corrective action: Adjust the installation angle of the sensor."
    assert "Inconsistent" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "uuu-control-output-cell"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"

    far = result.model_copy(
        update={
            "chunk_id": "far-display-section-window",
            "content": "Item | Default\nShift function | ON\nDisplay | ON",
            "metadata": {
                "chunk_type": "section_window",
                "product_model": "LR-ZH500C3P",
                "parent_context": (
                    "Display | Description | Checks and Remedies | Control Output\n"
                    "uuu | Excessive reflected light | Adjust the installation angle of the sensor. | Inconsistent\n"
                    "-FF | The detected object is too far from the display range | "
                    "• Move the target closer. • Turn OFF the shift function. | Normal"
                ),
            },
        }
    )
    far_answer, _far_trace = generate_answer_with_trace(
        "How should I resolve the -FF display on the LR-ZH500C3P sensor?",
        [wrong, result, far],
    )

    assert far_answer.answer == (
        "Corrective action: • Move the target closer. • Turn OFF the shift function."
    )
    assert far_answer.citations[0]["chunk_id"] == "far-display-section-window"


def test_answer_uses_clean_summary_for_dependent_list_instead_of_table_cell_metadata():
    summary = (
        "• To improve the anti-noise feature, install the unit following the precautions below. "
        "Otherwise, a malfunction may occur. • Ground the FG cable of the sensor amplifier "
        "(IV4-G120). • Do not mount the unit near high-voltage equipment."
    )
    noisy_cell = SearchResult(
        chunk_id="cell",
        score=0.95,
        title="IV4 Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[5],
        section_path=[],
        content=f"Column headers: {summary}; Row headers: Devices; Cell value: Separate the I/O line.; Row: 1; Column: 0",
        metadata={"chunk_type": "table_record"},
    )
    clean_summary = SearchResult(
        chunk_id="summary",
        score=0.9,
        title="IV4 Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[5],
        section_path=[],
        content=f"Table summary: {summary}",
        metadata={"chunk_type": "table_record"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I improve anti-noise performance for the IV4-G120?",
        [noisy_cell, clean_summary],
    )

    assert "Ground the FG cable" in answer.answer
    assert "Column headers" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "summary"
    assert trace["final_answer"]["prompt_kind"] == "dependent_list"


def test_troubleshooting_anchor_supports_generated_cause_and_action_questions():
    symptom = "The RS-232C communication setting cannot be changed because PLC-Link is enabled."

    expected = symptom.rstrip(".")

    assert _query_troubleshooting_anchor(f"What causes {symptom} for XG-X Series?") == expected
    assert (
        _query_troubleshooting_anchor(f"How should {symptom} for XG-X Series be corrected?")
        == expected
    )


def test_location_question_prioritizes_complete_setting_hierarchy_over_related_context(monkeypatch):
    direct = SearchResult(
        chunk_id="direct",
        score=0.7,
        title="Controller Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[20],
        section_path=["Capture"],
        content=(
            "Overlapping lines: Specify the number of lines overlapped from the previous capture "
            "when storing in the image variable."
        ),
        metadata={
            "chunk_type": "section_window",
            "parent_context": (
                "Line Camera Settings\n\nImage Area (Common for All Capture Units)\n\n"
                "Fixed Capture Settings\n\nContinuous Capture Settings\n\n"
                "Overlapping lines (for Continuous only): Specify the number of lines overlapped "
                "from the previous capture when storing in the image variable."
            ),
        },
    )
    related = SearchResult(
        chunk_id="related",
        score=0.8,
        title="Controller Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[8],
        section_path=["Classification"],
        content="Duplicate suppression is enabled when overlapping lines is one or higher.",
        metadata={"chunk_type": "section_window"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.judge_retrieval_relevance",
        lambda _query, _results: [
            {"chunk_id": "related", "verdict": "relevant", "reason": "related"},
            {"chunk_id": "direct", "verdict": "potentially_relevant", "reason": "setting details"},
        ],
    )

    prioritized = prioritize_results_for_answer(
        "Where do I set overlapping lines for a line scan camera?",
        [related, direct],
    )

    assert prioritized["prioritized_results"][0].chunk_id == "direct"


def test_location_subject_supports_natural_where_and_menu_paraphrases():
    assert _configuration_location_subject("Where is the overlap distance setting for the scanner?") == "overlap distance"
    assert _configuration_location_subject("Which menu contains overlapping lines for the camera?") == "overlapping lines"
    assert (
        _configuration_location_subject("Where can I configure overlap between continuous line-scan captures?")
        == "overlap"
    )


def test_location_question_replaces_raw_context_with_concise_path_and_purpose():
    result = SearchResult(
        chunk_id="direct",
        score=0.9,
        title="Controller Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[20],
        section_path=["Capture"],
        content=(
            "Controller Manual.pdf | Capture\n\nOverlapping lines: Specify the number of lines "
            "overlapped from the previous capture when storing in the image variable."
        ),
        metadata={
            "chunk_type": "section_window",
            "parent_context": (
                "Line Camera Settings\n\nImage Area (Common for All Capture Units)\n\n"
                "Fixed Capture Settings\n\nContinuous Capture Settings\n\n"
                "Overlapping lines (for Continuous only): Specify the number of lines overlapped "
                "from the previous capture when storing in the image variable."
            ),
        },
    )
    raw_answer = AnswerResponse(
        answer=result.content,
        confidence="medium",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )

    validated = validate_answer(
        raw_answer,
        [result],
        query="Where do I set overlapping lines for a line scan camera?",
    )

    assert validated.answer.startswith("Location: In the Capture Unit, open Line Camera Settings")
    assert "Line Camera Settings > Image Area >" in validated.answer
    assert "Continuous Capture Settings > Overlapping lines" in validated.answer
    assert "Purpose:" in validated.answer
    assert ".pdf" not in validated.answer


def test_location_answer_merges_parent_screen_and_rejects_sibling_device_path():
    definition = SearchResult(
        chunk_id="definition",
        score=0.9,
        title="Controller Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[207],
        section_path=["Capture Using Line Scan Cameras"],
        content=(
            "Continuous Capture Settings\n\nOverlapping lines: Specify the number of lines "
            "overlapped from the previous capture when storing in the image variable.\n\n"
            "These settings are common for all capture units."
        ),
        metadata={"chunk_type": "section_window"},
    )
    parent_screen = SearchResult(
        chunk_id="parent",
        score=0.7,
        title="Controller Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[205],
        section_path=["Capture Using Line Scan Cameras"],
        content=(
            "Camera 1 to Camera 4: Select the tab for the camera being configured.\n\n"
            "Line Camera Settings: Specify the conditions for image capture."
        ),
        metadata={"chunk_type": "section_window"},
    )
    sibling_device = SearchResult(
        chunk_id="sibling",
        score=0.95,
        title="Controller Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[164],
        section_path=["Capture When Using a Profile Head"],
        content=(
            "Capture Area Settings\n\nOverlapping lines: Specify the number of lines "
            "to overlap the image."
        ),
        metadata={"chunk_type": "section_window"},
    )
    raw_answer = AnswerResponse(
        answer=sibling_device.content,
        confidence="medium",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    filler_results = [
        SearchResult(
            chunk_id=f"filler-{index}",
            score=0.8 - index / 100,
            title="Controller Manual.pdf",
            document_version_id="v1",
            source_document_id="d1",
            pages=[300 + index],
            section_path=["Other settings"],
            content="Unrelated camera timing information.",
            metadata={"chunk_type": "section_window"},
        )
        for index in range(7)
    ]
    retrieval_results = [definition, sibling_device, *filler_results, parent_screen]
    concise, _support = _concise_configuration_location_answer(
        "Where can I configure overlap between continuous line-scan captures?",
        retrieval_results,
    )
    assert "parent" in [result.chunk_id for result in _support], [
        result.chunk_id for result in _support
    ]
    assert "Line Camera Settings" in _configuration_path_labels(parent_screen.content, ""), (
        _configuration_path_labels(parent_screen.content, "")
    )
    assert concise.startswith(
        "Location: In the Capture Unit, open Line Camera Settings > Continuous Capture Settings"
    )

    validated = validate_answer(
        raw_answer,
        retrieval_results,
        query="Where can I configure overlap between continuous line-scan captures?",
    )

    assert validated.answer.startswith(
        "Location: In the Capture Unit, open Line Camera Settings > Continuous Capture Settings"
    )
    assert "Capture Area Settings" not in validated.answer
    assert "Purpose:" in validated.answer

    partial = raw_answer.model_copy(
        update={
            "answer": (
                "Location: In the Capture Unit, open Continuous Capture Settings > Overlapping lines. "
                "Purpose: Specify how many lines from the previous capture are retained."
            )
        }
    )
    completed = validate_answer(
        partial,
        retrieval_results,
        query="Where can I configure overlap between continuous line-scan captures?",
    )
    assert "Line Camera Settings > Continuous Capture Settings" in completed.answer


def test_location_answer_prefers_richer_supported_hierarchy_over_earlier_partial_definition():
    partial = SearchResult(
        chunk_id="partial",
        score=1.0,
        title="Controller Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[207],
        section_path=["Capture"],
        content=(
            "Continuous Capture Settings\n\nOverlapping lines: Specify the number of lines "
            "overlapped from the previous capture."
        ),
        metadata={"chunk_type": "section_window"},
    )
    complete = SearchResult(
        chunk_id="complete",
        score=0.8,
        title="Controller Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[228],
        section_path=["Capture"],
        content=(
            "Line Camera Settings\n\nImage Area\n\nContinuous Capture Settings\n\n"
            "Overlapping lines: Specify the number of lines overlapped from the previous capture "
            "when storing in the image variable."
        ),
        metadata={"chunk_type": "section_window"},
    )

    answer, support = _concise_configuration_location_answer(
        "Where do I set overlapping lines for a line scan camera?",
        [partial, complete],
    )

    assert "Line Camera Settings > Image Area > Continuous Capture Settings > Overlapping lines" in answer
    assert {result.chunk_id for result in support} == {"partial", "complete"}


def test_generate_answer_uses_complete_location_evidence_without_model(monkeypatch):
    result = SearchResult(
        chunk_id="complete",
        score=0.9,
        title="Controller Manual.pdf",
        document_version_id="v1",
        source_document_id="d1",
        pages=[228],
        section_path=["Capture Unit"],
        content=(
            "Line Camera Settings\n\nImage Area\n\nContinuous Capture Settings\n\n"
            "Overlapping lines: Specify the number of lines overlapped from the previous capture "
            "when storing in the image variable. Select the tab for the camera being configured."
        ),
        metadata={"chunk_type": "section_window"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("answer model should not run")),
    )

    answer, trace = generate_answer_with_trace(
        "Where do I set overlapping lines for a linescan camera, and what is it for?",
        [result],
    )

    assert "Line Camera Settings > Image Area > Continuous Capture Settings > Overlapping lines" in answer.answer
    assert "Purpose:" in answer.answer
    assert trace["final_answer"]["prompt_kind"] == "configuration_location"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_troubleshooting_symptom_with_different_or_while_is_not_a_comparison():
    assert not _is_comparison_query(
        'What causes calibration data generated under different conditions from the "Axes Configuration" error?'
    )
    assert not _is_comparison_query("What causes the error while HDR capture is enabled?")
    assert not _is_comparison_query('The controller shows "Fan error". What caused it, and what should I do?')
    assert _is_comparison_query("Compare the Controller A and Controller B corrective actions.")


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
    assert "Asynchronous Trigger" not in validated.answer
    assert "Multi-Capture" in validated.answer
    assert "Control/data output via I/O terminals" in validated.answer


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


def test_prioritize_results_promotes_distinctive_query_evidence_over_generic_model_relevance(monkeypatch):
    generic = SearchResult(
        chunk_id="generic",
        score=0.9,
        title="XG-X Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[1],
        section_path=["Overview"],
        content="The XG-X Series sends data logs to a connected PC.",
        metadata={"chunk_type": "spec_record"},
    )
    cable = SearchResult(
        chunk_id="cable",
        score=0.8,
        title="XG-X Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[2],
        section_path=["VisionDataStorage"],
        content="Connect VisionDataStorage with the dedicated USB cable OP-88263.",
        metadata={"chunk_type": "spec_record"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.judge_retrieval_relevance",
        lambda _query, _results: [
            {"chunk_id": "generic", "verdict": "relevant", "reason": "model false positive"},
            {"chunk_id": "cable", "verdict": "potentially_relevant", "reason": "contains the answer"},
        ],
    )

    prioritized = prioritize_results_for_answer(
        "Which cable connects the XG-X Series to VisionDataStorage?",
        [generic, cable],
    )

    assert prioritized["prioritized_results"][0].chunk_id == "cable"


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
    assert validated.answer.startswith("Corrective action:")
    assert "Retrieved evidence:" not in validated.answer
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


def test_quantity_fallback_binds_the_count_to_the_asked_component_and_stays_concise():
    query = "How many communication expansion units can I connect to the controller?"
    wrong = SearchResult(
        chunk_id="installation",
        score=0.91,
        title="User Manual",
        document_version_id="v1",
        source_document_id="doc-user",
        pages=[30],
        section_path=["Installing the Communication Expansion Unit"],
        content=(
            "Install the communication expansion unit on the controller. "
            "The monitor output is 1024 x 768 pixels. Up to two heads can be connected."
        ),
        metadata={"chunk_type": "section_window"},
    )
    correct = SearchResult(
        chunk_id="expansion-limit",
        score=0.82,
        title="Installation Guide",
        document_version_id="v1",
        source_document_id="doc-guide",
        pages=[30],
        section_path=["Specifications"],
        content="Only one communication expansion unit can be connected.",
        metadata={"chunk_type": "spec_record"},
    )
    unsupported = AnswerResponse(
        answer="Install the unit on the right side of the controller.",
        confidence="high",
        used_documents=[],
        citations=[{"chunk_id": "installation", "document_id": "doc-user", "pages": [30], "quote_span": None}],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )

    selected = _fallback_evidence_results(query, [wrong, correct])
    validated = validate_answer(unsupported, [wrong, correct], query=query)

    assert [result.chunk_id for result in selected] == ["expansion-limit"]
    assert validated.answer == "Only one communication expansion unit can be connected."
    assert "Retrieved evidence:" not in validated.answer
    assert [citation["chunk_id"] for citation in validated.citations] == ["expansion-limit"]


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

    assert validated.answer == (
        "Cause: The LJ-S/LJ-X/LJ-V Series head is not connected to the camera input unit.\n"
        "Corrective action: Connect the LJ-S/LJ-X/LJ-V Series head to the camera input unit."
    )
    assert validated.citations[0]["chunk_id"] == "full-row"
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_troubleshooting_answer_combines_cause_and_action_without_dumping_table_context():
    results = [
        SearchResult(
            chunk_id="cause-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Cause; Row headers: Fan error; "
                "Cell value: The fan unit has failed."
            ),
            metadata={
                "chunk_type": "table_record",
                "context_window": "Fan error | The fan unit has failed. | Replace the fan unit with CA-F100.",
            },
        ),
        SearchResult(
            chunk_id="action-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective Action; Row headers: Fan error; "
                "Cell value: Replace the fan unit with CA-F100."
            ),
            metadata={"chunk_type": "table_record"},
        ),
    ]

    answer, trace = generate_answer_with_trace(
        "What causes the Fan error, and how should it be corrected?",
        results,
    )

    assert answer.answer == (
        "Cause: The fan unit has failed.\n"
        "Corrective action: Replace the fan unit with CA-F100."
    )
    assert {citation["chunk_id"] for citation in answer.citations} == {"cause-row", "action-row"}
    assert "Column headers" not in answer.answer
    assert "Context:" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_troubleshooting_answer_does_not_append_the_next_error_number_to_a_remedy():
    results = [
        SearchResult(
            chunk_id="firmware-errors",
            score=0.9,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[791],
            section_path=["Troubleshooting"],
            content=(
                "Error Number: 14301; Error Messages: The controller was booted using unsupported firmware.; "
                "Cause: The firmware version is not supported by the controller.; "
                "Remedy: Update the firmware to one supported by the controller. "
                "Error Number: 14302; Error Messages: The update file is not supported.; "
                "Cause: An unsupported update file is on the SD card.; "
                "Remedy: Replace it with a supported update file."
            ),
            metadata={"chunk_type": "table_record"},
        )
    ]

    answer, _trace = generate_answer_with_trace(
        "What should I do if the LJ-X8000 controller boots with unsupported firmware?",
        results,
    )

    assert answer.answer == "Corrective action: Update the firmware to one supported by the controller."
    assert "14302" not in answer.answer


def test_troubleshooting_answer_respects_explicit_product_model_for_duplicate_rows():
    common_row = (
        "Error Number: 14301; Error Messages: The controller was booted using unsupported firmware.; "
        "Cause: The firmware version is not supported by the controller.; "
        "Remedy: Update the firmware to one supported by the controller."
    )
    results = [
        SearchResult(
            chunk_id="lj-s-duplicate",
            score=1.0,
            title="LJ-S8000 User's Manual",
            document_version_id="v-ljs",
            source_document_id="d-ljs",
            pages=[490],
            section_path=["Troubleshooting"],
            content=common_row,
            metadata={"chunk_type": "table_record", "product_model": "LJ-S8000"},
        ),
        SearchResult(
            chunk_id="lj-x-expected",
            score=0.9,
            title="LJ-X8000 User's Manual",
            document_version_id="v-ljx",
            source_document_id="d-ljx",
            pages=[791],
            section_path=["Troubleshooting"],
            content=common_row,
            metadata={"chunk_type": "table_record", "product_model": "LJ-X8000"},
        ),
    ]

    answer, _trace = generate_answer_with_trace(
        "What should I do if the LJ-X8000 controller boots with unsupported firmware?",
        results,
    )

    assert answer.answer == "Corrective action: Update the firmware to one supported by the controller."
    assert {citation["document_id"] for citation in answer.citations} == {"d-ljx"}


def test_troubleshooting_answer_binds_requested_error_number_to_its_own_row():
    results = [
        SearchResult(
            chunk_id="firmware-error-group",
            score=1.0,
            title="LJ-X8000 User's Manual",
            document_version_id="v-ljx",
            source_document_id="d-ljx",
            pages=[791],
            section_path=["Troubleshooting"],
            content=(
                "Error Number: 14301; Error Messages: The controller booted using unsupported firmware.; "
                "Cause: The controller firmware is unsupported.; "
                "Remedy: Update the controller firmware.\n"
                "Error Number: 14302; Error Messages: The update file is unsupported.; "
                "Cause: An unsupported update file was added to SD card 2.; "
                "Remedy: Replace the file with a supported update file, and then start the controller."
            ),
            metadata={"chunk_type": "table_record", "product_model": "LJ-X8000"},
        )
    ]

    answer, _trace = generate_answer_with_trace(
        "How do I fix error 14302 on the LJ-X8000 regarding unsupported update files?",
        results,
    )

    assert answer.answer == (
        "Corrective action: Replace the file with a supported update file, and then start the controller."
    )
    assert "Update the controller firmware" not in answer.answer


def test_troubleshooting_uses_prioritized_exact_error_code_evidence():
    wrong = SearchResult(
        chunk_id="generic-unit-error",
        score=1.0,
        title="XG-X manual",
        document_version_id="v1",
        source_document_id="doc-xgx",
        pages=[1276],
        section_path=["Error Messages"],
        content=(
            "Error Message: A unit setting error exists.; Cause: Program data is invalid.; "
            "Corrective Action: Use XG-X VisionEditor to check the program data.; Error Code: -"
        ),
        metadata={"chunk_type": "table_record", "product_family": "XG-X Series"},
    )
    exact = wrong.model_copy(
        update={
            "chunk_id": "illumination-215",
            "pages": [1264],
            "content": (
                "Error Message: An error occurred in communicating with the illumination expansion unit.; "
                "Cause: The next FLASH was input while light emission is active.; "
                "Corrective Action: • Set the FLASH output time to 0.1 msec. "
                "• If FLASH on-delay is positive, make it close to 0. "
                "• Increase the trigger interval if the error persists.; Error Code: 215"
            ),
        }
    )

    answer, _trace = generate_answer_with_trace(
        "How do I fix error 215 on the XG-X Series illumination unit?",
        [wrong],
        prioritized_results=[exact, wrong],
        summarized_evidence=[],
    )

    assert "FLASH output time to 0.1 msec" in answer.answer
    assert "FLASH on-delay" in answer.answer
    assert "Increase the trigger interval" in answer.answer
    assert "program data" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "illumination-215"

    records = _troubleshooting_field_records(exact.content)
    assert len(records) == 1
    assert records[0]["message"] == (
        "An error occurred in communicating with the illumination expansion unit."
    )
    assert records[0]["corrective action"].startswith(
        "• Set the FLASH output time to 0.1 msec."
    )
    assert records[0]["error code"] == "215"


def test_troubleshooting_answer_associates_cells_from_the_exact_same_table_row():
    common_metadata = {
        "chunk_type": "table_record",
        "table_row": 6,
    }
    results = [
        SearchResult(
            chunk_id="message-cell",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Error Message; Row headers: Controller: X.X > A head not supported by the controller is connected.; "
                "Cell value: The camera firmware is not the latest version."
            ),
            metadata=common_metadata,
        ),
        SearchResult(
            chunk_id="cause-cell",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Cause; Row headers: Controller: X.X > A head not supported by the controller is connected.; "
                "Cell value: The connected head requires newer camera firmware."
            ),
            metadata=common_metadata,
        ),
        SearchResult(
            chunk_id="action-cell",
            score=0.7,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective Action; Row headers: Controller: X.X > A head not supported by the controller is connected.; "
                "Cell value: Turn off the controller and upgrade the firmware."
            ),
            metadata=common_metadata,
        ),
    ]

    answer, _trace = generate_answer_with_trace(
        "What causes The camera firmware is not the latest version, and how should it be corrected?",
        results,
    )

    assert answer.answer == (
        "Cause: The connected head requires newer camera firmware.\n"
        "Corrective action: Turn off the controller and upgrade the firmware."
    )
    assert {citation["chunk_id"] for citation in answer.citations} == {"cause-cell", "action-cell"}


def test_uppercase_vs_product_family_is_not_treated_as_a_comparison():
    results = [
        SearchResult(
            chunk_id="vs-error-row",
            score=0.9,
            title="VS Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Error Message: Failed to connect to [Target IP Address].; "
                "Cause: Connection to the camera of the specified IP address failed.; "
                "Corrective Action: Check the destination IP address and the connection settings."
            ),
            metadata={"chunk_type": "table_record"},
        )
    ]

    answer, _trace = generate_answer_with_trace(
        "What causes Failed to connect to [Target IP Address] for VS Series, and how should it be corrected?",
        results,
    )

    assert answer.answer == (
        "Cause: Connection to the camera of the specified IP address failed.\n"
        "Corrective action: Check the destination IP address and the connection settings."
    )


def test_troubleshooting_answer_retains_action_at_tenth_retrieval_position():
    distractors = [
        SearchResult(
            chunk_id=f"distractor-{index}",
            score=1.0 - index / 100,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=f"General image stitching guidance {index}.",
            metadata={"chunk_type": "atomic_text"},
        )
        for index in range(8)
    ]
    results = [
        *distractors,
        SearchResult(
            chunk_id="cause-row",
            score=0.2,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Cause; Row headers: Image Stitching unit is not available when using the Height Image.; "
                "Cell value: A multi-camera image variable was selected."
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="action-row",
            score=0.1,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Corrective Action; Row headers: Image Stitching unit is not available when using the Height Image.; "
                "Cell value: Change to an image variable that is not the multi-camera type."
            ),
            metadata={"chunk_type": "table_record"},
        ),
    ]

    answer, _trace = generate_answer_with_trace(
        "What causes Image Stitching unit is not available when using the Height Image, and how should it be corrected?",
        results,
    )

    assert "Cause: A multi-camera image variable was selected." in answer.answer
    assert "Corrective action: Change to an image variable that is not the multi-camera type." in answer.answer
    assert {citation["chunk_id"] for citation in answer.citations} == {"cause-row", "action-row"}


def test_error_cause_number_column_is_not_rendered_as_the_cause():
    results = [
        SearchResult(
            chunk_id="error-number-cell",
            score=0.9,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Column headers: Error cause No.; Row headers: Unable to output to the PLC-Link due to a full output buffer.; "
                "Cell value: 13302."
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="full-error-row",
            score=0.8,
            title="CV-X Manual",
            document_version_id="v1",
            source_document_id="d1",
            pages=[12],
            section_path=["Troubleshooting"],
            content=(
                "Error Number: 13302; Error Messages: Unable to output to the PLC-Link due to a full output buffer.; "
                "Cause: The controller output buffer is full.; Remedy: Reduce the amount of output data."
            ),
            metadata={"chunk_type": "table_record"},
        ),
    ]

    answer, _trace = generate_answer_with_trace(
        "For error 13302, what causes Unable to output to the PLC-Link due to a full output buffer?",
        results,
    )

    assert answer.answer == "Cause: The controller output buffer is full."
    assert answer.citations[0]["chunk_id"] == "full-error-row"


def test_generate_answer_uses_deterministic_troubleshooting_path_for_section_evidence(monkeypatch):
    result = SearchResult(
        chunk_id="encoder-row",
        score=0.9,
        title="XG-X Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[12],
        section_path=["Troubleshooting"],
        content=(
            "Error Message: Encoder timeout error.; Cause: Encoder input stopped.; "
            "Corrective Action: Check the encoder connection."
        ),
        metadata={"chunk_type": "section_window"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("answer model should not run")),
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.judge_retrieval_relevance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("relevance model should not run")),
    )

    answer, trace = generate_answer_with_trace(
        "What causes the encoder timeout error, and how should it be corrected?",
        [result],
    )

    assert "Check the encoder connection" in answer.answer
    assert trace["final_answer"]["answer_source"] == "structured_evidence"
    assert trace["final_answer"]["used_fallback"] is False


def test_rendered_table_answer_returns_matching_row_with_column_labels(monkeypatch):
    result = SearchResult(
        chunk_id="output-settings",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[174],
        section_path=["Output Settings"],
        content=(
            "IV4_Manual.pdf | Output Settings\n\n"
            "Setting range | Description\n"
            "OFF | Do not output.\n"
            "Error | The output turns ON for system error, startup memory readout error, "
            "program switching error, external master registration error, and SD card access error.\n"
            "Insufficient SD capacity | Output turns ON when the SD card does not have sufficient capacity."
        ),
        metadata={"chunk_type": "section_window"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("answer model should not run")),
    )

    answer, trace = generate_answer_with_trace(
        "Which errors cause the output to turn ON when the setting is Error?",
        [result],
    )

    assert answer.answer.startswith("Setting range: Error; Description:")
    assert "program switching error" in answer.answer
    assert "Insufficient SD capacity" not in answer.answer
    assert trace["final_answer"]["prompt_kind"] == "structured_table"


def test_rendered_table_answer_selects_requested_effect_without_raw_context(monkeypatch):
    result = SearchResult(
        chunk_id="output-settings",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[174],
        section_path=["Output Settings"],
        content=(
            "Setting range | Description\n"
            "Error | The output turns ON for system errors.\n"
            "Insufficient SD capacity | Output turns ON when the SD card does not have sufficient capacity."
        ),
        metadata={"chunk_type": "section_window"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("answer model should not run")),
    )

    answer, trace = generate_answer_with_trace(
        "What happens to the output when the SD card capacity is insufficient?",
        [result],
    )

    assert answer.answer == (
        "Setting range: Insufficient SD capacity; "
        "Description: Output turns ON when the SD card does not have sufficient capacity."
    )
    assert "Retrieved evidence" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


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

    assert validated.answer == (
        "Cause: The next FLASH was input while the light was being emitted.\n"
        "Corrective action: Set the FLASH output time to 0.1 msec."
    )
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


def test_validate_answer_cleans_matching_troubleshooting_answer_to_requested_fields():
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

    assert validated.answer == (
        "Cause: The next FLASH was input while the light was being emitted.\n"
        "Corrective action: Set the FLASH output time to 0.1 msec."
    )
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


def test_answer_prioritization_prefers_imperative_checklist_for_verification_question(monkeypatch):
    results = [
        SearchResult(
            chunk_id="status-screen",
            score=1.0,
            title="CV-X482",
            document_version_id="v1",
            source_document_id="d1",
            pages=[645],
            section_path=["Utility"],
            content=(
                "Column headers: Checks communication status of RS-232C; "
                "Cell value: Checks communication status of PLC-Link during run mode and setup mode."
            ),
            metadata={"chunk_type": "table_record", "product_model": "CV-X482"},
        ),
        SearchResult(
            chunk_id="verification-checklist",
            score=0.9,
            title="CV-X482",
            document_version_id="v1",
            source_document_id="d1",
            pages=[798],
            section_path=["PLC-Link"],
            content=(
                "When connected by RS-232C: Check the PLC-Link communication settings, "
                "connection cable, and status at the device the cable is connected to."
            ),
            metadata={"chunk_type": "spec_record", "product_model": "CV-X482"},
        ),
    ]
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: ({"items": []}, "{}"),
    )

    prioritized = prioritize_results_for_answer(
        "Which items require verification for CV-X482 RS-232C PLC-Link communication?",
        results,
    )

    assert prioritized["prioritized_results"][0].chunk_id == "verification-checklist"


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
    assert calls[-1]["num_predict"] == 1024
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


def test_generate_answer_rescopes_precomputed_evidence_and_discards_conflicting_summaries(monkeypatch):
    conflicting = SearchResult(
        chunk_id="xgx-port",
        score=0.95,
        title="XG-X Manual",
        document_version_id="vx",
        source_document_id="dx",
        pages=[738],
        section_path=["FTP"],
        content="Enter the FTP server port number (default 21; SFTP 22).",
        metadata={"chunk_type": "section_window", "product_family": "XG-X Series"},
    )
    matching = SearchResult(
        chunk_id="iv4-user",
        score=0.9,
        title="IV4 Manual",
        document_version_id="vi",
        source_document_id="di",
        pages=[283],
        section_path=["SFTP"],
        content=(
            "Input the user name (max: 48 characters) to log in to the FTP/SFTP server. "
            "(Default: Not set (blank))"
        ),
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )
    stale_summaries = [
        {
            "chunk_id": "xgx-port,iv4-user",
            "title": "XG-X Manual",
            "pages": [283, 738],
            "section_path": ["FTP"],
            "summary": "Mixed FTP settings.",
            "source_document_id": "dx",
            "document_version_id": "vx",
        }
    ]

    def fail_final_answer(**kwargs):
        if kwargs["purpose"] == "final_answer":
            raise ValueError("force grounded fallback")
        raise AssertionError(f"unexpected model call: {kwargs['purpose']}")

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fail_final_answer)

    answer, trace = generate_answer_with_trace(
        "Is a username required by default for the IV4-G600CA SFTP server?",
        [conflicting, matching],
        prioritized_results=[conflicting, matching],
        summarized_evidence=stale_summaries,
    )

    assert "Not set (blank)" in answer.answer
    assert answer.citations[0]["chunk_id"] == "iv4-user"
    assert all(item["source_document_id"] == "di" for item in trace["final_answer"]["summarized_evidence"])


def test_prioritize_results_protects_direct_spec_fact_over_same_document_toc_noise(monkeypatch):
    toc = SearchResult(
        chunk_id="toc",
        score=0.95,
        title="IV4 Manual",
        document_version_id="vi",
        source_document_id="di",
        pages=[15],
        section_path=["Contents"],
        content="Required environment ..............................................................",
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-G600CA"},
    )
    username = SearchResult(
        chunk_id="username",
        score=0.9,
        title="IV4 Manual",
        document_version_id="vi",
        source_document_id="di",
        pages=[283],
        section_path=["SFTP"],
        content=(
            "Input the user name (max: 48 characters) to log in to the FTP/SFTP server. "
            "(Default: Not set (blank))"
        ),
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.judge_retrieval_relevance",
        lambda _query, results: [
            {"chunk_id": result.chunk_id, "verdict": "relevant", "reason": "test"}
            for result in results
        ],
    )

    prioritized = prioritize_results_for_answer(
        "Is a username required by default for the IV4-G600CA SFTP server?",
        [toc, username],
    )

    assert prioritized["prioritized_results"][0].chunk_id == "username"


def test_answer_chooses_spec_revision_covering_default_over_partial_older_revision():
    older = SearchResult(
        chunk_id="older-user",
        score=0.95,
        title="Older IV4 Manual",
        document_version_id="old",
        source_document_id="old-doc",
        pages=[275],
        section_path=["SFTP"],
        content="Input the user name (max: 48 characters) to log in to the FTP/SFTP server.",
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )
    complete = SearchResult(
        chunk_id="complete-user",
        score=0.9,
        title="Current IV4 Manual",
        document_version_id="current",
        source_document_id="current-doc",
        pages=[283],
        section_path=["SFTP"],
        content=(
            "Input the user name (max: 48 characters) to log in to the FTP/SFTP server. "
            "(Default: Not set (blank))"
        ),
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )

    answer, trace = generate_answer_with_trace(
        "Is a username required by default for the IV4-G600CA SFTP server?",
        [older, complete],
    )

    assert "Default: Not set (blank)" in answer.answer
    assert answer.citations[0]["chunk_id"] == "complete-user"
    assert trace["final_answer"]["prompt_kind"] == "structured_fact"


def test_direct_measurement_prefers_query_aligned_atomic_evidence_over_generic_spec():
    generic_spec = SearchResult(
        chunk_id="generic-m4-depth",
        score=0.99,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="doc-1",
        pages=[39],
        section_path=["Sensor head"],
        content="Screw hole on the sensor head: M4 (screw depth: 3.5 mm)",
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )
    exact_mounting_hole = SearchResult(
        chunk_id="front-m4-depth",
        score=0.90,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="doc-1",
        pages=[44],
        section_path=["Mounting the Sensor"],
        content=(
            "In addition to the back M3 mounting hole, the sensor can be mounted on the "
            "back M2.5 (depth 3.4 mm) and the front M4 (depth 4.1 mm) hole."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-G600CA"},
    )

    answer, trace = generate_answer_with_trace(
        "How deep is the front M4 mounting hole on the IV4-G600CA sensor?",
        [generic_spec, exact_mounting_hole],
    )

    assert "front M4 (depth 4.1 mm)" in answer.answer
    assert "3.5 mm" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "front-m4-depth"
    assert trace["final_answer"]["prompt_kind"] == "structured_fact"


def test_direct_measurement_prefers_requested_model_section_over_other_lens_section():
    other_lens = SearchResult(
        chunk_id="other-lens-wd",
        score=1.0,
        title="Lens Manual",
        document_version_id="v1",
        source_document_id="doc-1",
        pages=[30, 31],
        section_path=["CA-LH50"],
        content="CA-LH50. *1 WD indicates a working distance at reference magnification.",
        metadata={"chunk_type": "section_window", "product_model": "CA-DRM10X"},
    )
    requested_lens = SearchResult(
        chunk_id="ca-lm0510-wd",
        score=0.9,
        title="Lens Manual",
        document_version_id="v1",
        source_document_id="doc-1",
        pages=[52],
        section_path=["CA-LM0510"],
        content=(
            "CA-LM0510. These lenses enable a long WD of 110 mm 4.33 inches and a high "
            "resolution supporting up to 2/3 inch image sensors."
        ),
        metadata={"chunk_type": "section_window", "product_model": "CA-DRM10X"},
    )

    answer, trace = generate_answer_with_trace(
        "What working distance do CA-LM0510 lenses provide?",
        [other_lens, requested_lens],
    )

    assert "110 mm" in answer.answer
    assert answer.citations[0]["chunk_id"] == "ca-lm0510-wd"
    assert trace["final_answer"]["prompt_kind"] == "structured_fact"


def test_summary_fallback_prefers_requested_model_section_over_generic_table():
    exact = SearchResult(
        chunk_id="ca-lm0510",
        score=0.9,
        title="Lens Manual",
        document_version_id="v1",
        source_document_id="doc-1",
        pages=[52],
        section_path=["CA-LM0510"],
        content="CA-LM0510 lenses support up to 2/3-inch image sensors.",
        metadata={"chunk_type": "section_window"},
    )
    generic = SearchResult(
        chunk_id="generic-ca-lm",
        score=0.95,
        title="Lens Manual",
        document_version_id="v1",
        source_document_id="doc-1",
        pages=[50],
        section_path=["CA-LM"],
        content="Compatible image sensor size: 4/3 inch for CA-LMHE0510.",
        metadata={"chunk_type": "section_window"},
    )
    summaries = [
        {
            "chunk_id": "ca-lm0510",
            "summary": "The CA-LM0510 lenses support image sensors up to 2/3 inch.",
            "source_documents": [{"chunk_id": "ca-lm0510"}],
        },
        {
            "chunk_id": "generic-ca-lm",
            "summary": "The CA-LM0510 lenses support a 4/3-inch image sensor.",
            "source_documents": [{"chunk_id": "generic-ca-lm"}],
        },
    ]

    answer = generator_module._fallback_answer_from_summaries(
        "Which image sensor sizes are supported by CA-LM0510 lenses?",
        summaries,
        [exact, generic],
    )

    assert answer is not None
    assert "2/3 inch" in answer.answer
    assert "4/3" not in answer.answer
    assert [citation["chunk_id"] for citation in answer.citations] == ["ca-lm0510"]


def test_calculation_question_skips_observed_spec_value_for_definition_row():
    observed_value = SearchResult(
        chunk_id="observed-match-degree",
        score=0.99,
        title="CV-X Manual",
        document_version_id="v1",
        source_document_id="doc-cvx",
        pages=[1100],
        section_path=["Results"],
        content="Match Degree (%): 87.445",
        metadata={"chunk_type": "spec_record", "product_model": "CV-X482"},
    )
    definition = SearchResult(
        chunk_id="match-degree-definition",
        score=0.9,
        title="CV-X Manual",
        document_version_id="v1",
        source_document_id="doc-cvx",
        pages=[1116],
        section_path=["Match Degree"],
        content=(
            "Column headers: Setting; Row headers: Match Degree (%); Cell value: Match Degree is "
            "the proportion of parts that match the set outlines. It is calculated from the number "
            "of detected outlines and the number of set outlines."
        ),
        metadata={"chunk_type": "table_record", "product_model": "CV-X482"},
    )

    answer, trace = generate_answer_with_trace(
        "How is the match degree percentage calculated for the CV-X482?",
        [observed_value, definition],
    )

    assert "number of detected outlines" in answer.answer
    assert "87.445" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "match-degree-definition"
    assert trace["final_answer"]["prompt_kind"] == "structured_table"


def test_capture_time_measurement_binds_to_cap_time_instead_of_utility_speed_text():
    utility = SearchResult(
        chunk_id="utility-speed",
        score=1.0,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[446],
        section_path=["Utility"],
        content=(
            "Operation Information: Cap. Time (Latest, MAX, MIN, AVE). "
            + "unrelated details " * 30
            + "Utility: High Speed program switching, Simulator *7, and automatic backup."
        ),
        metadata={"chunk_type": "section_window", "product_model": "IV4-G120"},
    )
    capture_time = SearchResult(
        chunk_id="high-speed-cap-time",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[94],
        section_path=["Operation Mode"],
        content=(
            "Operation Mode: High Speed; Cap. Time: 12 ms; Max. No. of Detections: 10"
        ),
        metadata={"chunk_type": "section_window", "product_model": "IV4-G120"},
    )

    answer, trace = generate_answer_with_trace(
        "What capture time does the High Speed mode use on the IV4-G120?",
        [utility, capture_time],
    )

    assert "12 ms" in answer.answer
    assert "Utility" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "high-speed-cap-time"
    assert trace["final_answer"]["prompt_kind"] == "structured_fact"


def test_direct_torque_measurement_does_not_include_adjacent_step_or_heading():
    result = SearchResult(
        chunk_id="bracket-torque",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[39],
        section_path=["Mounting"],
        content=(
            "1 Attach the bracket using the screws included with the 2-axis adjustment bracket "
            "(OP-88912).\n\nTightening torque: 0.5 to 0.7 N·m Mounting examples Set the "
            "concave part on the bracket on the connector\n\n"
            "2 Loosen the screws and adjust the angle.\n\nTightening torque: 1.0 to 1.5 N·m"
        ),
        metadata={"chunk_type": "section_window"},
    )

    for query in (
        "What tightening torque is required for the OP-88912 bracket?",
        "How much torque should I apply when installing the 2-axis adjustment bracket?",
    ):
        answer, trace = generate_answer_with_trace(query, [result])

        assert answer.answer == "Tightening torque: 0.5 to 0.7 N·m"
        assert "Attach" not in answer.answer
        assert "Mounting examples" not in answer.answer
        assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_direct_torque_measurement_stops_at_unit_before_ocr_adjacent_instruction():
    result = SearchResult(
        chunk_id="sensor-cable-torque",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[20],
        section_path=["Sensor head cable"],
        content=(
            "Tightening torque: 0.6 to 0.8 N·m Align the pins and pin connection "
            "Sensor head Rotating connector on the attachment"
        ),
        metadata={"chunk_type": "section_window", "product_model": "IV4-G600CA"},
    )

    answer, _trace = generate_answer_with_trace(
        "What tightening torque is required for the IV4-G600CA sensor head cable?",
        [result],
    )

    assert answer.answer == "Tightening torque: 0.6 to 0.8 N·m"
    assert "Align" not in answer.answer


def test_direct_torque_prefers_promoted_atomic_measurement_over_broad_mounting_context():
    broad = SearchResult(
        chunk_id="broad-mounting-context",
        score=1.0,
        title="IV-HG manual",
        document_version_id="v1",
        source_document_id="doc-ivhg",
        pages=[45, 46],
        section_path=["Fix the dome attachment for IV-HG series with mounting screws"],
        content=(
            "If secured from the sheet metal side, use M4 screws and a tightening "
            "torque of 0.7 to 1.5 N·m."
        ),
        metadata={
            "chunk_type": "section_window",
            "retrieval_stage": "identifier_contextual_promoted",
            "product_model": "IV-HG500CA",
        },
    )
    exact = broad.model_copy(
        update={
            "chunk_id": "dome-dedicated-screw-torque",
            "pages": [49],
            "section_path": ["Fix the dome attachment with attached dedicated screws"],
            "content": "Tightening torque: 0.25 to 0.35 N·m",
            "metadata": {
                "chunk_type": "atomic_text",
                "retrieval_stage": "measurement_promoted",
                "product_model": "IV-HG500CA",
            },
        }
    )

    answer, trace = generate_answer_with_trace(
        "What tightening torque is required for the IV-HG500CA dome attachment?",
        [broad, exact],
    )

    assert answer.answer == "Tightening torque: 0.25 to 0.35 N·m"
    assert answer.citations[0]["chunk_id"] == "dome-dedicated-screw-torque"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_enumerated_options_answer_preserves_all_close_up_ring_thicknesses():
    unrelated = SearchResult(
        chunk_id="unrelated-fov",
        score=0.95,
        title="Lens guide",
        document_version_id="v0",
        source_document_id="d0",
        pages=[40],
        section_path=["FOV"],
        content="A 50 mm lens provides a 3 mm field of view at 90 mm working distance.",
        metadata={"chunk_type": "section_window"},
    )
    exact = SearchResult(
        chunk_id="ring-options",
        score=0.9,
        title="Camera manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[36],
        section_path=["OP-51612"],
        content=(
            'Close-up rings are available in a set of five different sizes, 0.5 mm 0.02", '
            '1.0 mm 0.04", 5 mm 0.20", 10 mm 0.39", and 22 mm 0.87". '
            "Use a locking adhesive when combining thin rings."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(
        "What thickness options are included in the OP-51612 close-up ring set?",
        [unrelated, exact],
    )

    for thickness in ("0.5 mm", "1.0 mm", "5 mm", "10 mm", "22 mm"):
        assert thickness in answer.answer
    assert "working distance" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "ring-options"
    assert trace["final_answer"]["answer_source"] == "deterministic_enumerated_options"


def test_physical_location_answer_returns_explicit_between_relation():
    wrong = SearchResult(
        chunk_id="camera-heading",
        score=0.95,
        title="Camera manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[36],
        section_path=["Field of view"],
        content="Camera field of view: KV-CA1H, KV-CA1W",
        metadata={"chunk_type": "section_window"},
    )
    exact = SearchResult(
        chunk_id="ring-location",
        score=0.9,
        title="Camera manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[36],
        section_path=["Close-up ring"],
        content=(
            "The close-up ring is installed between the camera and the lens. Close-up rings are "
            "available in five sizes."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(
        "Where is the close-up ring installed on a KV-CA1H or KV-CA1W camera setup?",
        [wrong, exact],
    )

    assert answer.answer == "The close-up ring is installed between the camera and the lens."
    assert answer.citations[0]["chunk_id"] == "ring-location"
    assert trace["final_answer"]["answer_source"] == "deterministic_physical_location"


def test_alignment_components_answer_excludes_neighboring_torque():
    result = SearchResult(
        chunk_id="lighting-cable-alignment",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[43],
        section_path=["AI Lighting unit cable"],
        content=(
            "Connect the cable to the rotating connector. Tightening torque: 0.6 to 0.8 N·m "
            "Align the pins and pin connection\nNext section"
        ),
        metadata={"chunk_type": "section_window", "product_model": "IV4-G600CA"},
    )

    answer, trace = generate_answer_with_trace(
        "What components should be aligned when attaching the IV4-G600CA lighting cable?",
        [result],
    )

    assert answer.answer == "Align the pins and the pin connection."
    assert "torque" not in answer.answer.lower()
    assert trace["final_answer"]["answer_source"] == "deterministic_alignment_components"


def test_capture_time_segment_outranks_quantity_in_same_broad_section():
    broad_section = SearchResult(
        chunk_id="broad-spec-section",
        score=1.0,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[94, 446],
        section_path=["Specifications"],
        content=(
            "Utility | High Speed program switching, Simulator *7, automatic backup.\n"
            "Cap. Time: High speed = 12 ms; Standard = 60 ms; High Accuracy = 200 ms."
        ),
        metadata={"chunk_type": "section_window", "product_model": "IV4-G120"},
    )

    answer = _concise_general_fallback_answer(
        "What capture time does the High Speed mode use on the IV4-G120?",
        broad_section,
    )

    assert "12 ms" in answer
    assert not answer.startswith("Utility")


def test_temperature_measurement_handles_concatenated_table_header():
    result = SearchResult(
        chunk_id="xt060-environment",
        score=0.9,
        title="XT Manual",
        document_version_id="v1",
        source_document_id="doc-xt",
        pages=[22],
        section_path=["Environmental"],
        content=(
            "Environmental | Operating AmbientTemperature | 0°C to 40°C 32 to 104°F\n"
            "Resistance | Operating Ambient Humidity | 20 to 85% RH (no condensation)\n"
            "XT-024: Dimensions; XT-060: 286 × 286 × 286 mm\n"
            "XT-024: Weight; XT-060: 10.1 kg"
        ),
        metadata={"chunk_type": "table_record", "product_model": "XT-060"},
    )

    answer, trace = generate_answer_with_trace(
        "What ambient temperature range is required for the XT-060 inspection system?",
        [result],
    )

    assert "0°C to 40°C" in answer.answer
    assert "10.1 kg" not in answer.answer
    assert trace["final_answer"]["prompt_kind"] == "structured_fact"


def test_temperature_threshold_binds_to_plural_temperature_sentence():
    result = SearchResult(
        chunk_id="dzw-temperature",
        score=0.9,
        title="Lighting guide",
        document_version_id="v1",
        source_document_id="doc-light",
        pages=[10],
        section_path=["Environmental resistance"],
        content=(
            "Environmental resistance of the light components: ambient temperature "
            "0 to 40°C (32°F to 104°F). For CA-DZW50X, light volume is limited in "
            "ambient environment temperatures above 35°C (95°F)."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "CA-DZW50X"},
    )

    answer, trace = generate_answer_with_trace(
        "At what temperature does the CA-DZW50X light volume become limited?",
        [result],
    )

    assert "above 35°C" in answer.answer
    assert "0 to 40°C" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "dzw-temperature"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_trigger_interval_range_outranks_trigger_indicator_specification():
    indicator = SearchResult(
        chunk_id="trigger-indicator",
        score=1.0,
        title="IV-H Manual",
        document_version_id="v1",
        source_document_id="doc-ivh",
        pages=[36],
        section_path=["TRIG"],
        content="Green light lights up (one-shot) according to input of the internal or external trigger.",
        metadata={"chunk_type": "spec_record", "product_model": "IV-HG500CA"},
    )
    interval = SearchResult(
        chunk_id="trigger-interval",
        score=0.9,
        title="IV-H Manual",
        document_version_id="v1",
        source_document_id="doc-ivh",
        pages=[95],
        section_path=["Set the trigger interval"],
        content=(
            "When the [Internal Trigger] is selected, set the trigger interval within "
            "the range of 1 to 10,000 ms."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV-HG500CA"},
    )

    answer, trace = generate_answer_with_trace(
        "What is the valid trigger interval range for the IV-HG500CA Internal Trigger?",
        [indicator, interval],
    )

    assert "1 to 10,000 ms" in answer.answer
    assert "Green light" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "trigger-interval"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_profile_capture_rate_outranks_unrelated_product_specification():
    certification = SearchResult(
        chunk_id="certification",
        score=1.0,
        title="LJ-X8000 Manual",
        document_version_id="v1",
        source_document_id="wrong-doc",
        pages=[6],
        section_path=["Certification"],
        content=(
            "The obtained CSA authentication of the LJ-X8000 Series head applies only "
            "when used with the LJ-X8000 Series controller."
        ),
        metadata={"chunk_type": "spec_record", "product_model": "LJ-X8000"},
    )
    capture_rate = SearchResult(
        chunk_id="profile-rate",
        score=0.9,
        title="LJ-X8000 Guide",
        document_version_id="v1",
        source_document_id="right-doc",
        pages=[10],
        section_path=["High-speed profile measurement"],
        content=(
            "With the ability to capture profiles at 64,000Hz, targets transported at "
            "high speeds can be measured without missing data."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "New LJ-X8000 Series"},
    )

    answer, trace = generate_answer_with_trace(
        "What profile capture rate does the LJ-X8000 Series support?",
        [certification, capture_rate],
    )

    assert "64,000Hz" in answer.answer
    assert "CSA" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "profile-rate"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_capability_answer_uses_explicit_ability_sentence_instead_of_heading():
    result = SearchResult(
        chunk_id="profile-capability",
        score=0.9,
        title="LJ-X8000 Guide",
        document_version_id="v1",
        source_document_id="doc-ljx",
        pages=[41, 42],
        section_path=["High-speed profile measurement"],
        content=(
            "High-speed 2D Laser Profiler\n"
            "With the ability to capture profiles at 64,000Hz, the shape of targets being "
            "transported at high speeds can be measured without missing any data."
        ),
        metadata={"chunk_type": "section_window", "product_model": "New LJ-X8000 Series"},
    )

    answer, trace = generate_answer_with_trace(
        "Can the LJ-X8000 Series measure high-speed targets without missing data?",
        [result],
    )

    assert answer.answer.startswith("Yes.")
    assert "64,000Hz" in answer.answer
    assert "without missing any data" in answer.answer
    assert answer.citations[0]["chunk_id"] == "profile-capability"
    assert trace["final_answer"]["answer_source"] == "deterministic_capability"


def test_capability_answer_recognizes_run_and_preserves_coexistence_claim():
    result = SearchResult(
        chunk_id="kv-xle02-network-capability",
        score=0.9,
        title="KV-XLE02 User's Manual",
        document_version_id="v1",
        source_document_id="doc-kv-xle02",
        pages=[41],
        section_path=["Network compatibility"],
        content=(
            "EtherNet/IP and PROFINET can be used together with a general-purpose "
            "Ethernet network."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "KV-XLE02"},
    )

    answer, trace = generate_answer_with_trace(
        "Can KV-XLE02 run EtherNet/IP and PROFINET with general Ethernet?",
        [result],
    )

    assert answer.answer == (
        "Yes. EtherNet/IP and PROFINET can be used together with a general-purpose "
        "Ethernet network."
    )
    assert answer.citations[0]["chunk_id"] == "kv-xle02-network-capability"
    assert trace["final_answer"]["answer_source"] == "deterministic_capability"


def test_capability_answer_extracts_selected_protocol_impact():
    result = SearchResult(
        chunk_id="kv-xle02-network-capability",
        score=0.9,
        title="KV-XLE02 User's Manual",
        document_version_id="v1",
        source_document_id="doc-kv-xle02",
        pages=[41],
        section_path=["Network compatibility"],
        content=(
            "*1 EtherNet/IP and PROFINET can be used together with a general-purpose "
            "Ethernet network* 3. *2 When CC-Link IE Field or EtherCAT is selected, "
            "a general-purpose Ethernet network* 3 cannot be used at the same time."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "KV-XLE02"},
    )

    answer, trace = generate_answer_with_trace(
        "What happens to general Ethernet functions if I select CC-Link IE Field on KV-XLE02?",
        [result],
    )

    assert answer.answer == (
        "When CC-Link IE Field or EtherCAT is selected, a general-purpose Ethernet "
        "network cannot be used at the same time."
    )
    assert answer.citations[0]["chunk_id"] == "kv-xle02-network-capability"
    assert trace["final_answer"]["answer_source"] == "deterministic_capability"


def test_capability_answer_treats_eliminated_need_as_no_requirement():
    exact = SearchResult(
        chunk_id="auto-image-selector-skills",
        score=0.9,
        title="VS Series",
        document_version_id="v1",
        source_document_id="vs-doc",
        pages=[11],
        section_path=["KEYENCE AI"],
        content=(
            "With AI Auto Image Selector, the software automatically selects the images "
            "for learning, eliminating the need for specialized skills and significantly "
            "reducing the time needed for learning."
        ),
        metadata={"chunk_type": "atomic_text", "product_family": "VS Series"},
    )
    distractor = SearchResult(
        chunk_id="selector-execution-conditions",
        score=1.0,
        title="VS Series",
        document_version_id="v1",
        source_document_id="vs-manual",
        pages=[734],
        section_path=["Start Training"],
        content="The Auto Image Selector can be executed if the following conditions are met.",
        metadata={"chunk_type": "section_window", "product_family": "VS Series"},
    )

    answer, trace = generate_answer_with_trace(
        "Does the VS Series AI Auto Image Selector require specialized skills to use?",
        [distractor, exact],
    )

    assert answer.answer.startswith("No.")
    assert "eliminating the need for specialized skills" in answer.answer
    assert answer.citations[0]["chunk_id"] == "auto-image-selector-skills"
    assert trace["final_answer"]["answer_source"] == "deterministic_capability"


def test_troubleshooting_answer_extracts_prose_remedy_after_symptom():
    result = SearchResult(
        chunk_id="one-spot-remedy",
        score=0.9,
        title="LR-ZH500C3P Manual",
        document_version_id="v1",
        source_document_id="doc-lr-zh",
        pages=[4],
        section_path=["Universal Change Detection"],
        content=(
            "If the [1spot] indicator is off after calibration, detection is unstable. "
            "The possible causes are shown below. Check the installation condition, "
            "and perform calibration again."
        ),
        metadata={
            "chunk_type": "atomic_text",
            "product_model": "LR-ZH500C3P",
            "parent_context": (
                "An unrelated setting cannot be used in another mode. "
                "Release the buttons when SET flashes."
            ),
        },
    )

    answer, trace = generate_answer_with_trace(
        "How should I fix unstable detection if the 1spot light is off on the LR-ZH500C3P?",
        [result],
    )

    assert _query_troubleshooting_anchor(
        "How should I fix unstable detection if the 1spot light is off on the LR-ZH500C3P?"
    ) == "unstable detection if the 1spot light is off"
    assert answer.answer == (
        "Corrective action: Check the installation condition, and perform calibration again."
    )
    assert answer.citations[0]["chunk_id"] == "one-spot-remedy"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_instruction_answer_keeps_character_limit_setup_sequence_and_exact_product():
    distractor = SearchResult(
        chunk_id="other-product-ocr",
        score=1.0,
        title="Other Product Manual",
        document_version_id="v1",
        source_document_id="other-doc",
        pages=[307],
        section_path=["OCR"],
        content=(
            "Set the upper limit and lower limit of [Character Count] to default to "
            "maximum string length using zero padding."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "LJ-X8000"},
    )
    exact = SearchResult(
        chunk_id="vs-character-count-limits",
        score=0.9,
        title="VS Series User's Manual",
        document_version_id="v1",
        source_document_id="vs-doc",
        pages=[556],
        section_path=["OCR2", "Number of Characters"],
        content=(
            "Set the upper and lower limits for the number of all recognized characters. "
            "The results of the recognition are displayed in real time for [Current Value]. "
            "Select the check boxes for [Upper Limit] or [Lower Limit], and then enter "
            "the number of characters."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "VS Series"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I set character count limits for VS Series OCR?",
        [distractor, exact],
    )

    assert "Select the check boxes for [Upper Limit] or [Lower Limit]" in answer.answer
    assert "enter the number of characters" in answer.answer
    assert "zero padding" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "vs-character-count-limits"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_display_location_fact_returns_named_ui_field():
    exact = SearchResult(
        chunk_id="vs-current-value",
        score=0.9,
        title="VS Series User's Manual",
        document_version_id="v1",
        source_document_id="vs-doc",
        pages=[556],
        section_path=["OCR2", "Number of Characters"],
        content=(
            "Set the upper and lower limits for the number of all recognized characters. "
            "The results of the recognition are displayed in real time for [Current Value]."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "VS Series"},
    )
    distractor = SearchResult(
        chunk_id="ocr-correlation",
        score=1.0,
        title="VS Series User's Manual",
        document_version_id="v1",
        source_document_id="vs-doc",
        pages=[535],
        section_path=["OCR"],
        content="Recognition correlation results range from 0 to 99.",
        metadata={"chunk_type": "table_record", "product_model": "VS Series"},
    )

    answer, trace = generate_answer_with_trace(
        "Where does the VS Series display real-time character recognition results?",
        [distractor, exact],
    )

    assert answer.answer == (
        "The results of the recognition are displayed in real time for [Current Value]."
    )
    assert answer.citations[0]["chunk_id"] == "vs-current-value"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_instruction_answer_respects_disable_polarity():
    result = SearchResult(
        chunk_id="w500-password",
        score=0.9,
        title="LR-W500 Instruction Manual",
        document_version_id="v1",
        source_document_id="w500-doc",
        pages=[3],
        section_path=["Password"],
        content=(
            "An optional password can be set to prohibit unauthorized releasing of the "
            "Key Lock. Select a value from 1 to 999 for this setting. If '0' is selected, "
            "the password will not be required."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "W500"},
    )
    distractor = SearchResult(
        chunk_id="w500-require-password",
        score=1.0,
        title="LR-W500 Instruction Manual",
        document_version_id="v1",
        source_document_id="w500-doc",
        pages=[4],
        section_path=["Key Lock"],
        content="To require a password to release the key lock, set a password in advance.",
        metadata={"chunk_type": "section_window", "product_model": "W500"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I disable the password for the W500 Key Lock?",
        [distractor, result],
    )

    assert answer.answer == "If '0' is selected, the password will not be required."
    assert answer.citations[0]["chunk_id"] == "w500-password"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_structured_table_answer_binds_repeated_wiring_record():
    result = SearchResult(
        chunk_id="iv-out3-records",
        score=0.9,
        title="IV-500C Manual",
        document_version_id="v1",
        source_document_id="iv-doc",
        pages=[6],
        section_path=["Wiring"],
        content=(
            "Wiring color: Gray; Name: OUT3; Assigning default value: Error (N.O.); "
            "Description: y Judge result of each tool (Tool 1 to Tool 16) "
            "Wiring color: Orange; Name: OUT4; Assigning default value: OFF; "
            "Description: Logic 1 to Logic 4"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV-500C"},
    )

    answer, trace = generate_answer_with_trace(
        "What does the gray OUT3 terminal indicate on the IV-500C?",
        [result],
    )

    assert answer.answer == (
        "Wiring color: Gray; Name: OUT3; Default assignment: Error (N.O.); "
        "Description: Judge result of each tool (Tool 1 to Tool 16)"
    )
    assert "OUT4" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "iv-out3-records"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"

    semantic_answer, _semantic_trace = generate_answer_with_trace(
        "Which terminal on the IV-500C signals an error condition by default?",
        [result],
    )

    assert semantic_answer.answer == answer.answer
    assert "OUT4" not in semantic_answer.answer


def test_structured_table_answer_prefers_range_over_measurement_test_point():
    test_point = SearchResult(
        chunk_id="response-test-point",
        score=0.95,
        title="LR-T Manual",
        document_version_id="v1",
        source_document_id="lr-doc",
        pages=[18],
        section_path=["Accuracy"],
        content=(
            "Column headers: LR-TB2000/TB2000C (Class 2 laser) > White Paper "
            "(Reflectivity: 90%) > Response Time [ms] > 1; Row headers: Detecting "
            'distance [mm inch] > 1500 59.06"; Cell value: ±13 ±0.51"; Row: 8; Column: 2'
        ),
        metadata={"chunk_type": "table_record"},
    )
    range_record = SearchResult(
        chunk_id="detectable-range",
        score=0.9,
        title="LR-T Manual",
        document_version_id="v1",
        source_document_id="lr-doc",
        pages=[16],
        section_path=["Specifications"],
        content=(
            "Column headers: LR-TB2000 > - > LR-TB2000C > LR-TB2000CL; "
            'Row headers: Detectable distance; Cell value: 60 to 2000 mm 2.36" to 78.74" *2; '
            "Row: 2; Column: 4"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, _trace = generate_answer_with_trace(
        "What is the detection range for the LR-TB2000 laser sensor?",
        [test_point, range_record],
    )

    assert "60 to 2000 mm" in answer.answer
    assert "Response Time" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "detectable-range"


def test_structured_table_answer_binds_model_to_requested_connector_field():
    result = SearchResult(
        chunk_id="iv2-cp50-spec",
        score=0.9,
        title="IV Series Manual",
        document_version_id="v1",
        source_document_id="iv-doc",
        pages=[25],
        section_path=["Specifications"],
        content=(
            "Model: Connector; IV2-CP50: M12 4-pin connector "
            "Model: Languages* 2; IV2-CP50: English / Japanese / German "
            "Model: Expanded memory; IV2-CP50: USB memory"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, _trace = generate_answer_with_trace(
        "What connector type does the IV2-CP50 use?",
        [result],
    )

    assert answer.answer == "Connector — IV2-CP50: M12 4-pin connector"
    assert "Languages" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "iv2-cp50-spec"


def test_default_setting_answer_binds_value_to_matching_repeated_item():
    result = SearchResult(
        chunk_id="rotation-range-setting",
        score=0.9,
        title="IV-HG Manual",
        document_version_id="v1",
        source_document_id="iv-hg-doc",
        pages=[157],
        section_path=["Tool Settings"],
        content=(
            "Items: Rotation Range; Description: Sets a range to adjust the position; "
            "Setting range: 0 to ±180° (Unit: ±1°); Default value: ±20° "
            "Items: Margin; Description: Allows a wider search; Setting range: ON or OFF; Default value: ON"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV-HG500CA"},
    )

    answer, trace = generate_answer_with_trace(
        "What is the default rotation range for the IV-HG500CA?",
        [result],
    )

    assert answer.answer == "The default Rotation Range is ±20°."
    assert answer.citations[0]["chunk_id"] == "rotation-range-setting"
    assert trace["final_answer"]["answer_source"] == "deterministic_default_setting_value"


def test_matching_model_answer_requires_all_requested_row_features():
    result = SearchResult(
        chunk_id="lr-adjustable-analog-model",
        score=0.9,
        title="LR-T Manual",
        document_version_id="v1",
        source_document_id="lr-doc",
        pages=[13],
        section_path=["Models"],
        content=(
            "Column headers: Model; Row headers: Cable (2 m) > Adjustable > "
            "[Control Output + Analog Output]; Cell value: LR-TB5000; Row: 1; Column: 4"
        ),
        metadata={"chunk_type": "table_record"},
    )

    answer, trace = generate_answer_with_trace(
        "Which laser sensor model offers an adjustable spot and analog output?",
        [result],
    )

    assert answer.answer == (
        "The matching model is LR-TB5000; it is adjustable and supports analog output."
    )
    assert answer.citations[0]["chunk_id"] == "lr-adjustable-analog-model"
    assert trace["final_answer"]["answer_source"] == "deterministic_matching_model"


def test_instruction_answer_keeps_complete_insulation_action_single():
    result = SearchResult(
        chunk_id="unused-io-cables",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="iv4-doc",
        pages=[2],
        section_path=["Wiring"],
        content=(
            "y Individually insulate the unused input-output cables. "
            "y For input cables of this sensor, connect with non-contact output "
            "(transistor output/SSR output)."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-400CA"},
    )

    answer, trace = generate_answer_with_trace(
        "How should I handle unused input-output cables on the IV4-400CA?",
        [result],
    )

    assert answer.answer == "Individually insulate the unused input-output cables."
    assert "non-contact output" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_contamination_action_binds_light_axis_remedy():
    distractor = SearchResult(
        chunk_id="blob-tool",
        score=1.0,
        title="LJ-S8000 Manual",
        document_version_id="v1",
        source_document_id="lj-doc",
        pages=[154],
        section_path=["Blob Tool"],
        content="The Blob Tool can judge damage or dirt using the blob darkness.",
        metadata={"chunk_type": "atomic_text"},
    )
    remedy = SearchResult(
        chunk_id="light-axis-contamination",
        score=0.9,
        title="LJ-S8000 Manual",
        document_version_id="v1",
        source_document_id="lj-doc",
        pages=[5],
        section_path=["Precautions"],
        content=(
            "Intrusion of floating or sprinkled dust or dirt into the light-axis range: "
            "In this case, take corrective action with a protective cover or air purge."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(
        "What should I do if dirt enters the LJ-S8000 light axis?",
        [distractor, remedy],
    )

    assert answer.answer == "Use a protective cover or air purge."
    assert answer.citations[0]["chunk_id"] == "light-axis-contamination"
    assert trace["final_answer"]["answer_source"] == "deterministic_contamination_action"


def test_named_selection_extracts_quoted_option_that_performs_requested_action():
    result = SearchResult(
        chunk_id="plc-download-option",
        score=0.9,
        title="PROFINET Communication Manual",
        document_version_id="v1",
        source_document_id="profinet-doc",
        pages=[4],
        section_path=["Configuring Siemens PLC Settings"],
        content=(
            "After the compilation is finished, right-click the PLC, point to 'Download to device,' "
            "and then click 'Hardware and software (only changes)' to transmit the compiled programme to the PLC."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(
        "What option should I select to transmit the compiled program to the PLC?",
        [result],
    )

    assert answer.answer == "The option is Hardware and software (only changes)."
    assert answer.citations[0]["chunk_id"] == "plc-download-option"
    assert trace["final_answer"]["answer_source"] == "deterministic_named_selection"


def test_instruction_answer_selects_specific_lens_securing_action():
    distractor = SearchResult(
        chunk_id="telecentric-overview",
        score=1.0,
        title="Lens Manual",
        document_version_id="v1",
        source_document_id="lens-doc",
        pages=[51],
        section_path=["Overview"],
        content="These are the optimal lenses to use when applying telecentric lenses to line scan cameras.",
        metadata={"chunk_type": "atomic_text"},
    )
    instruction = SearchResult(
        chunk_id="ca-lmxx-mounting",
        score=0.9,
        title="Lens Manual",
        document_version_id="v1",
        source_document_id="lens-doc",
        pages=[46],
        section_path=["Installation"],
        content=(
            "When installing the telecentric lens (CA-LMxx) to the line scan camera, make sure "
            "to secure the lens unit with the dedicated mounting stand (OP-87337, sold separately) "
            "or an equivalent mount."
        ),
        metadata={"chunk_type": "atomic_text"},
    )

    answer, trace = generate_answer_with_trace(
        "How should I secure the CA-LMxx telecentric lens to the camera?",
        [distractor, instruction],
    )

    assert "dedicated mounting stand (OP-87337" in answer.answer
    assert "equivalent mount" in answer.answer
    assert answer.citations[0]["chunk_id"] == "ca-lmxx-mounting"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_part_number_answer_selects_code_nearest_requested_component():
    result = SearchResult(
        chunk_id="lens-accessory-table",
        score=0.9,
        title="Lens Manual",
        document_version_id="v1",
        source_document_id="lens-doc",
        pages=[46],
        section_path=["Accessories"],
        content="Part number: Lens mount; OP-87896: C-mount",
        metadata={
            "chunk_type": "table_record",
            "context_window": "Part number: Lens mount; OP-87896: C-mount",
            "parent_context": (
                "Secure the CA-LMxx lens with the dedicated mounting stand for the macro lens | "
                + " | ".join(f"specification {index}" for index in range(18))
                + " | Part | OP-87337. "
                "F-mount conversion adapter | Part number | OP-87319."
            ),
            "product_model": "CA-DRM10X",
        },
    )

    answer, trace = generate_answer_with_trace(
        "Which mounting stand part number is required for the CA-LMxx lens?",
        [result],
    )

    assert answer.answer == "The required mounting stand part number is OP-87337."
    assert "OP-87319" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "lens-accessory-table"
    assert trace["final_answer"]["answer_source"] == "deterministic_part_number"


def test_calibration_instruction_prefers_concrete_sensor_steps_over_controller_capability():
    controller = SearchResult(
        chunk_id="controller",
        score=0.95,
        title="LR-T Manual",
        document_version_id="v1",
        source_document_id="lr-t",
        pages=[12],
        section_path=["Controller"],
        content=(
            "The MU-N Series controller provides a remote display that can be used to quickly "
            "calibrate and monitor attached LR-T Series sensors. This controller pairs with the "
            "LR-T Series. Connect up to 4 controllers."
        ),
        metadata={"chunk_type": "section_window", "product_model": "LR-T"},
    )
    procedure = SearchResult(
        chunk_id="calibration",
        score=0.8,
        title="LR-T Manual",
        document_version_id="v1",
        source_document_id="lr-t",
        pages=[7],
        section_path=["Simplified setup"],
        content=(
            "Calibrate your sensor in seconds by simply pressing the SET button while the target "
            "you would like to detect is present, and then again when it is absent. The sensor "
            "will automatically set the optimum ON/OFF set point for your output."
        ),
        metadata={"chunk_type": "section_window", "product_model": "LR-T"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I calibrate the LR-T Series laser sensor?",
        [controller, procedure],
    )

    assert "pressing the SET button" in answer.answer
    assert "target" in answer.answer
    assert "present" in answer.answer and "absent" in answer.answer
    assert answer.citations[0]["chunk_id"] == "calibration"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_button_effect_answer_includes_calibration_sequence_and_result():
    distractor = SearchResult(
        chunk_id="applications",
        score=0.95,
        title="LR-T Manual",
        document_version_id="v1",
        source_document_id="lr-t",
        pages=[10],
        section_path=["Applications"],
        content=(
            "The LR-T Series laser sensor detects targets on conveyance systems, machining "
            "centers, web tension controls, welding cells, transfer presses, and hoppers."
        ),
        metadata={"chunk_type": "section_window", "product_model": "LR-T"},
    )
    procedure = SearchResult(
        chunk_id="calibration",
        score=0.8,
        title="LR-T Manual",
        document_version_id="v1",
        source_document_id="lr-t",
        pages=[7],
        section_path=["Simplified setup"],
        content=(
            "Calibrate your sensor in seconds by simply pressing the SET button while the target "
            "you would like to detect is present, and then again when it is absent. The sensor "
            "will automatically set the optimum ON/OFF set point for your output."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "LR-T"},
    )

    answer, trace = generate_answer_with_trace(
        "What happens when I press the SET button on the LR-T Series sensor?",
        [distractor, procedure],
    )

    assert "target" in answer.answer
    assert "present" in answer.answer and "absent" in answer.answer
    assert "automatically set the optimum ON/OFF set point" in answer.answer
    assert answer.citations[0]["chunk_id"] == "calibration"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_movable_range_answer_binds_to_requested_component():
    stage = SearchResult(
        chunk_id="stage-range",
        score=0.95,
        title="LJ-S8000 Manual",
        document_version_id="v1",
        source_document_id="lj-s",
        pages=[39],
        section_path=["Dedicated stand"],
        content='Stage movable range: 52 mm 2.05" (±26 mm ±1.02")',
        metadata={"chunk_type": "spec_record", "product_model": "LJ-S8000"},
    )
    cap_bolt = SearchResult(
        chunk_id="cap-bolt-range",
        score=0.8,
        title="LJ-S8000 Manual",
        document_version_id="v1",
        source_document_id="lj-s",
        pages=[39],
        section_path=["Dedicated stand"],
        content='Cap bolt Movable range: 10 mm 0.39" (±5 mm ±0.20")',
        metadata={"chunk_type": "section_window", "product_model": "LJ-S8000"},
    )

    answer, trace = generate_answer_with_trace(
        "What is the movable range for the cap bolt on the NEW LJ: S8000 Series?",
        [stage, cap_bolt],
    )

    assert '10 mm 0.39"' in answer.answer
    assert '±5 mm ±0.20"' in answer.answer
    assert "52 mm" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "cap-bolt-range"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"

    paraphrased, paraphrased_trace = generate_answer_with_trace(
        "How far can the cap bolt move on the S8000 Series laser sensor?",
        [stage, cap_bolt],
    )

    assert '10 mm 0.39"' in paraphrased.answer
    assert "52 mm" not in paraphrased.answer
    assert paraphrased.citations[0]["chunk_id"] == "cap-bolt-range"
    assert paraphrased_trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_unit_only_question_does_not_append_neighboring_output_format():
    neighboring = SearchResult(
        chunk_id="program-time",
        score=0.9,
        title="LJ-X8000 Manual",
        document_version_id="v1",
        source_document_id="lj-x",
        pages=[375],
        section_path=["Output"],
        content="Program Time: Program time is output in 2 words (Unit: ms).",
        metadata={"chunk_type": "section_window", "product_model": "LJ-X8000"},
    )
    measurement_time = SearchResult(
        chunk_id="measurement-time",
        score=0.8,
        title="LJ-X8000 Manual",
        document_version_id="v1",
        source_document_id="lj-x",
        pages=[369],
        section_path=["Output"],
        content=(
            "Measurement Time: The measurement time is output as integer 7 digits plus "
            "1 decimal digit (Unit: ms)."
        ),
        metadata={"chunk_type": "spec_record", "product_model": "LJ-X8000"},
    )

    answer, trace = generate_answer_with_trace(
        "In what unit is the measurement time reported by the LJ-X8000?",
        [neighboring, measurement_time],
    )

    assert answer.answer == "The measurement time is reported in milliseconds (ms)."
    assert "2 words" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "measurement-time"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_reset_count_command_answer_uses_reset_behavior_from_local_context():
    distractor = SearchResult(
        chunk_id="change-count",
        score=0.95,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="iv4",
        pages=[350],
        section_path=["Command control"],
        content="Command Control Address 6 to 11: Change Count Value in AI Through Count Mode",
        metadata={"chunk_type": "section_window", "product_model": "IV4-G600CA"},
    )
    reset = SearchResult(
        chunk_id="reset-count",
        score=0.8,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="iv4",
        pages=[415],
        section_path=["Reset Count"],
        content="RCR command",
        metadata={
            "chunk_type": "section_window",
            "product_model": "IV4-G600CA",
            "context_window": (
                "Resets the serial number for the next AI OCR judgment. "
                "Resets the count value of the AI Through Count tool in AI Through Count mode. "
                "All tools are reset at a time."
            ),
        },
    )

    answer, trace = generate_answer_with_trace(
        "What does the Reset Count command clear on the IV4-G600CA?",
        [distractor, reset],
    )

    assert "serial number" in answer.answer
    assert "count value" in answer.answer
    assert "All tools are reset" in answer.answer
    assert answer.citations[0]["chunk_id"] == "reset-count"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"

    simultaneous, _trace = generate_answer_with_trace(
        "Does the IV4-G600CA reset all tools simultaneously with Reset Count?",
        [distractor, reset],
    )

    assert simultaneous.answer.startswith("Yes.")
    assert "All tools are reset" in simultaneous.answer


def test_named_setting_behavior_binds_to_its_labeled_table_row():
    result = SearchResult(
        chunk_id="pattern-settings",
        score=0.9,
        title="LJ-X8000 Manual",
        document_version_id="v1",
        source_document_id="doc-ljx",
        pages=[247],
        section_path=["Pattern settings"],
        content=(
            "Setting item: Light/Shade Inversion; Settings: Enable to search for a target "
            "whose black and white gradation is inverted.\n"
            "Setting item: Distortion Tolerance Range; Settings: When edge characteristics "
            "are skewed due to the angle of view or workpiece tilt, a fine search detects "
            "them if the skew is within the specified pixel value.\n"
            "Setting item: Min. Match%; Settings: Excludes candidates below the lower limit."
        ),
        metadata={"chunk_type": "table_record", "product_model": "LJ-X8000"},
    )

    answer, trace = generate_answer_with_trace(
        "What does the Distortion Tolerance Range setting control?",
        [result],
    )

    assert "fine search" in answer.answer
    assert "specified pixel value" in answer.answer
    assert "black and white" not in answer.answer
    assert "lower limit" not in answer.answer
    assert "Context:" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "pattern-settings"
    assert trace["final_answer"]["answer_source"] == "deterministic_named_option_behavior"


def test_labeled_indicator_list_excludes_neighboring_specifications():
    wrong = SearchResult(
        chunk_id="screw-spec",
        score=1.0,
        title="IV installation manual",
        document_version_id="v1",
        source_document_id="doc-iv",
        pages=[5],
        section_path=["Mounting"],
        content="Screw: M3 x 4. Use screws with a head thickness of 3 mm or lower.",
        metadata={"chunk_type": "spec_record", "product_model": "IV-500C"},
    )
    exact = SearchResult(
        chunk_id="indicator-row",
        score=0.8,
        title="IV installation manual",
        document_version_id="v1",
        source_document_id="doc-iv",
        pages=[9],
        section_path=["Indicators"],
        content=(
            "Indicators | PWR/ERR, OUT, TRIG, STATUS, LINK/ACT\n"
            "Input | No-voltage input/voltage input is switchable"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV-500C"},
    )

    answer, trace = generate_answer_with_trace(
        "Which LED indicators are available on the IV-500C?",
        [wrong, exact],
    )

    assert answer.answer == "The available indicators are PWR/ERR, OUT, TRIG, STATUS, LINK/ACT."
    assert "M3" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "indicator-row"
    assert trace["final_answer"]["answer_source"] == "deterministic_labeled_list"


def test_conditioned_table_cell_returns_only_requested_on_voltage():
    result = SearchResult(
        chunk_id="input-thresholds",
        score=0.9,
        title="IV installation manual",
        document_version_id="v1",
        source_document_id="doc-iv",
        pages=[9],
        section_path=["Input"],
        content=(
            "Column headers: IV-500C; Row headers: Input; Cell value: "
            "No-voltage input/voltage input is switchable For no-voltage input : "
            "ONvoltage 2V or lower, OFF current 0.1mA or lower, ONcurrent 2mA "
            "(short circuit) For voltage input : Maximum input rating 26.4V, "
            "ON voltage 15V or higher; Row: 18; Column: 2"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV-500C"},
    )

    answer, trace = generate_answer_with_trace(
        "What voltage triggers an ON state for the IV-500C no-voltage input?",
        [result],
    )

    assert answer.answer == "For the no-voltage input, ON voltage is 2V or lower."
    assert "15V" not in answer.answer
    assert "0.1mA" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "input-thresholds"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_input_terminal_count_uses_input_row_not_connector_pin_count():
    result = SearchResult(
        chunk_id="input-count",
        score=0.9,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[446],
        section_path=["Specifications"],
        content=(
            "Column headers: IV4-G120 > Standard Mode, Sorting Mode, AI Through Count Mode; "
            "Row headers: Number of inputs; Cell value: 8 (IN1 to IN8); Row: 17; Column: 2"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV4-G120"},
    )

    answer, trace = generate_answer_with_trace(
        "How many input terminals does the IV4-G120 provide?",
        [result],
    )

    assert answer.answer == "The IV4-G120 provides 8 input terminals (IN1 to IN8)."
    assert answer.citations[0]["chunk_id"] == "input-count"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_first_input_function_uses_exact_function_table_cell():
    result = SearchResult(
        chunk_id="input-function",
        score=0.9,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[446],
        section_path=["Specifications"],
        content=(
            "Column headers: IV4-G120; Row headers: Function; Cell value: "
            "IN1: External trigger, IN2 to IN8: Enable by assigning the optional functions; "
            "Row: 18; Column: 2"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV4-G120"},
    )

    answer, trace = generate_answer_with_trace(
        "What function is assigned to the first input on the IV4-G120?",
        [result],
    )

    assert answer.answer == "The first input (IN1) is assigned to External trigger."
    assert answer.citations[0]["chunk_id"] == "input-function"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_instruction_does_not_cross_ocr_section_marker():
    result = SearchResult(
        chunk_id="high-speed",
        score=0.9,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[158, 159],
        section_path=["Search Algorithm"],
        content=(
            "Select [High Speed] to shorten the processing time. "
            "\u0084 Position Adjustment Setting You can select the Position Adjustment/"
            "High-Speed Position Adjustment tool to be applied."
        ),
        metadata={"chunk_type": "section_window", "product_model": "IV4-G600CA"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I shorten processing time on the IV4-G600CA?",
        [result],
    )

    assert answer.answer == "Select [High Speed] to shorten the processing time."
    assert "Position Adjustment" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_instruction_resolves_select_this_to_named_option():
    result = SearchResult(
        chunk_id="high-speed-mode",
        score=0.9,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[130],
        section_path=["High speed mode"],
        content=(
            "Enables or disables high-speed mode. When [Enable] is selected, priority is given "
            "to reading speed. Select this to shorten the processing time."
        ),
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I shorten processing time on the IV4-G600CA?",
        [result],
    )

    assert answer.answer == "Select [Enable] to shorten the processing time."
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_mode_priority_question_returns_priority_target():
    result = SearchResult(
        chunk_id="high-speed-mode",
        score=0.9,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[130],
        section_path=["High speed mode"],
        content=(
            "Enables or disables high-speed mode. When [Enable] is selected, priority is given "
            "to reading speed. Select this to shorten the processing time."
        ),
        metadata={"chunk_type": "spec_record", "product_model": "IV4-G600CA"},
    )

    answer, trace = generate_answer_with_trace(
        "What does enabling high speed mode prioritize on the IV4-G600CA?",
        [result],
    )

    assert answer.answer == "Enabling high speed mode prioritizes reading speed."
    assert trace["final_answer"]["answer_source"] == "deterministic_named_option_behavior"


def test_numbered_mode_requirement_returns_only_its_threshold():
    result = SearchResult(
        chunk_id="large-area-search",
        score=0.9,
        title="LJ-S8000 manual",
        document_version_id="v1",
        source_document_id="doc-lj",
        pages=[132],
        section_path=["Pattern settings"],
        content=(
            "Setting item: Large Area Search Mode; Settings: Select this option when the pattern "
            "region is set to wide. • Disabled: Wide Search Mode not used. • Mode 1: If the region "
            "size exceeds a width of 2,432 pixels and/or a height of 2,050 pixels, this mode must "
            "be selected. • Mode 2: If the region size exceeds a width of 4096 pixels and/or a "
            "height of 4096 pixels, this mode must be selected. Setting item: Detection Order; "
            "Settings: Selects how numbers are assigned."
        ),
        metadata={"chunk_type": "table_record", "product_model": "LJ-S8000 Series"},
    )

    answer, trace = generate_answer_with_trace(
        "When must I enable Large Area Search Mode 1 on the LJ-S8000 Series?",
        [result],
    )

    assert answer.answer == (
        "Select Large Area Search Mode 1 if the region size exceeds a width of 2,432 pixels "
        "and/or a height of 2,050 pixels."
    )
    assert "4096" not in answer.answer
    assert trace["final_answer"]["answer_source"] == "deterministic_named_mode_requirement"


def test_why_troubleshooting_question_returns_cause_before_table_display_cell():
    display_cell = SearchResult(
        chunk_id="display-cell",
        score=1.0,
        title="W500 manual",
        document_version_id="v1",
        source_document_id="doc-w500",
        pages=[4],
        section_path=["Troubleshooting"],
        content=(
            "Column headers: Display; Row headers: Loc - (The bar pulses across the display.); "
            "Cell value: - (The bar pulses across the display.); Row: 7; Column: 0"
        ),
        metadata={"chunk_type": "table_record", "product_model": "W500"},
    )
    cause_row = SearchResult(
        chunk_id="pulsing-bar",
        score=0.9,
        title="W500 manual",
        document_version_id="v1",
        source_document_id="doc-w500",
        pages=[4],
        section_path=["Troubleshooting"],
        content=(
            "Display: - (The bar pulses across the display.); "
            "Cause: The display selection is set to OFF.; "
            "Solution: Set the display selection to ON."
        ),
        metadata={"chunk_type": "table_record", "product_model": "W500"},
    )

    answer, trace = generate_answer_with_trace(
        "Why does the W500 display a pulsing bar?",
        [display_cell, cause_row],
    )

    assert answer.answer == "Cause: The display selection is set to OFF."
    assert answer.citations[0]["chunk_id"] == "pulsing-bar"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_stop_symptom_question_returns_corrective_action():
    wrong = SearchResult(
        chunk_id="normal-operation",
        score=1.0,
        title="W500 manual",
        document_version_id="v1",
        source_document_id="doc-w500",
        pages=[4],
        section_path=["Troubleshooting"],
        content=(
            "Display: - (The bar pulses across the display.); "
            "Output Condition: Normal operation; Indicator Condition: Normal operation"
        ),
        metadata={"chunk_type": "table_record", "product_model": "W500"},
    )
    correct = SearchResult(
        chunk_id="pulsing-bar",
        score=0.9,
        title="W500 manual",
        document_version_id="v1",
        source_document_id="doc-w500",
        pages=[4],
        section_path=["Troubleshooting"],
        content=(
            "Display: - (The bar pulses across the display.); "
            "Cause: The display selection is set to OFF.; "
            "Solution: Set the display selection to ON."
        ),
        metadata={"chunk_type": "table_record", "product_model": "W500"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I stop the W500 display from pulsing?",
        [wrong, correct],
    )

    assert answer.answer == "Corrective action: Set the display selection to ON."
    assert answer.citations[0]["chunk_id"] == "pulsing-bar"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_solution_table_cell_is_normalized_to_corrective_action():
    result = SearchResult(
        chunk_id="pulsing-bar-solution",
        score=0.9,
        title="W500 manual",
        document_version_id="v1",
        source_document_id="doc-w500",
        pages=[4],
        section_path=["Troubleshooting"],
        content=(
            "Column headers: Solution; Row headers: - (The bar pulses across the display.) > "
            "The display selection is set to OFF.; Cell value: Set the display selection to ON. "
            "( page 5); Row: 6; Column: 2"
        ),
        metadata={"chunk_type": "table_record", "product_model": "W500"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I stop the W500 display from pulsing?",
        [result],
    )

    assert answer.answer == "Corrective action: Set the display selection to ON. ( page 5)."
    assert answer.citations[0]["chunk_id"] == "pulsing-bar-solution"
    assert trace["final_answer"]["answer_source"] == "structured_evidence"


def test_analog_output_option_uses_explicit_bracketed_selection():
    wrong = SearchResult(
        chunk_id="wire-state",
        score=1.0,
        title="LR-W70 manual",
        document_version_id="v1",
        source_document_id="doc-lrw",
        pages=[11],
        section_path=["Analog output selection"],
        content=(
            "Column headers: Violet; Row headers: Analog output selection > Display value *1 > "
            "OFF; Cell value: OFF; Row: 3; Column: 5"
        ),
        metadata={"chunk_type": "table_record", "product_model": "LR-W70(C)"},
    )
    exact = SearchResult(
        chunk_id="display-value-option",
        score=0.9,
        title="LR-W70 manual",
        document_version_id="v1",
        source_document_id="doc-lrw",
        pages=[4],
        section_path=["Analog output"],
        content=(
            "Select the data to output in analog format from the following: Display value "
            "[Disp. Value] *1: Outputs the value (0-999) displayed on the unit."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "LR-W70(C)"},
    )

    answer, trace = generate_answer_with_trace(
        "Which analog output option sends the value shown on the LR-W70(C) display?",
        [exact, wrong],
    )

    assert answer.answer == "The analog output option is Disp. Value."
    assert answer.citations[0]["chunk_id"] == "display-value-option"
    assert trace["final_answer"]["answer_source"] == "deterministic_named_selection"


def test_analog_display_value_range_returns_only_requested_range():
    result = SearchResult(
        chunk_id="display-value-range",
        score=0.9,
        title="LR-W70 manual",
        document_version_id="v1",
        source_document_id="doc-lrw",
        pages=[4],
        section_path=["Analog output"],
        content=(
            "Select the data to output in analog format from the following: Display value "
            "[Disp. Value] *1: Outputs the value (0-999) displayed on the unit."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "LR-W70(C)"},
    )

    answer, trace = generate_answer_with_trace(
        "What numeric range does the LR-W70(C) analog output use for the display value?",
        [result],
    )

    assert answer.answer == "The analog-output display-value range is 0 to 999."
    assert answer.citations[0]["chunk_id"] == "display-value-range"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_mode_detection_count_returns_requested_mode_value_only():
    wrong = SearchResult(
        chunk_id="tool-counts",
        score=1.0,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[445],
        section_path=["Tool count"],
        content="High Accuracy mode supports six AI Count tools and two Total tools.",
        metadata={"chunk_type": "section_window", "product_model": "IV4-G120"},
    )
    exact = SearchResult(
        chunk_id="mode-detections",
        score=0.8,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[94],
        section_path=["Operation mode"],
        content=(
            "Operation Mode: High Accuracy; Cap. Time: 200ms; Max. No. of Detections: 160\n"
            "Operation Mode: Standard; Cap. Time: 60ms; Max. No. of Detections: 40\n"
            "Operation Mode: High Speed; Cap. Time: 12ms; Max. No. of Detections: 10"
        ),
        metadata={"chunk_type": "table_record", "product_model": "IV4-G120"},
    )

    answer, trace = generate_answer_with_trace(
        "What is the maximum detection count for High Accuracy mode on the IV4-G120?",
        [wrong, exact],
    )

    assert answer.answer == "Max. No. of Detections: 160"
    assert "200ms" not in answer.answer
    assert "six AI Count" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "mode-detections"
    assert trace["final_answer"]["answer_source"] == "deterministic_structured_fact"


def test_mode_detection_count_trims_pipe_row_to_requested_metric():
    result = SearchResult(
        chunk_id="mode-detections-window",
        score=0.9,
        title="IV4 manual",
        document_version_id="v1",
        source_document_id="doc-iv4",
        pages=[90, 91, 92, 93, 94],
        section_path=["Operation mode"],
        content="High Accuracy | 200ms | 160\nStandard | 60ms | 40\nHigh Speed | 12ms | 10",
        metadata={"chunk_type": "section_window", "product_model": "IV4-G120"},
    )

    answer, _trace = generate_answer_with_trace(
        "What is the maximum detection count for High Accuracy mode on the IV4-G120?",
        [result],
    )

    assert answer.answer == "Max. No. of Detections: 160"


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
    assert trace["final_answer"]["num_predict"] == 1024
    assert trace["final_answer"]["used_fallback"] is False
    assert trace["final_answer"]["summarized_evidence"][0]["summary"] == "Use the Defect Tool for setup."


def test_final_answer_retry_uses_larger_output_budget(monkeypatch):
    result = SearchResult(
        chunk_id="temperature-limit",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[447],
        section_path=["Specifications"],
        content="If ambient exceeds 40°C, the case temperature must not exceed 65°C.",
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-G120"},
    )
    summary = [{
        "chunk_id": result.chunk_id,
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "summary": result.content,
        "source_document_id": result.source_document_id,
        "document_version_id": result.document_version_id,
    }]
    budgets: list[int] = []

    def fake_chat_json(**kwargs):
        budgets.append(kwargs["num_predict"])
        if len(budgets) == 1:
            raise ValueError("truncated JSON")
        return ({
            "answer": "The case temperature must not exceed 65°C when ambient exceeds 40°C.",
            "confidence": "high",
            "used_documents": [],
            "citations": [],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        }, "{}")

    monkeypatch.setattr(
        generator_module,
        "settings",
        replace(generator_module.settings, ollama_answer_num_predict=256),
    )
    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "What is the IV4-G120 case temperature rating?",
        [result],
        prioritized_results=[result],
        summarized_evidence=summary,
    )

    assert "65°C" in answer.answer
    assert budgets == [256, 1024]
    assert trace["final_answer"]["num_predict"] == 1024
    assert trace["final_answer"]["attempts"] == 2


def test_malformed_final_answer_falls_back_to_grounded_summary(monkeypatch):
    result = SearchResult(
        chunk_id="temperature-limit",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[447],
        section_path=["Specifications"],
        content=(
            "For the IV4-G120, if ambient exceeds 40°C, the case temperature must not exceed "
            "65°C. Measure the case temperature at the metal on the back of the sensor amplifier."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-G120"},
    )
    summary = [{
        "chunk_id": result.chunk_id,
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "summary": result.content,
        "source_document_id": result.source_document_id,
        "document_version_id": result.document_version_id,
    }]
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("malformed JSON")),
    )

    answer, trace = generate_answer_with_trace(
        "What is the IV4-G120 case temperature rating?",
        [result],
        prioritized_results=[result],
        summarized_evidence=summary,
    )

    assert "IV4-G120" in answer.answer
    assert "40°C" in answer.answer
    assert "65°C" in answer.answer
    assert answer.citations[0]["chunk_id"] == "temperature-limit"
    assert trace["final_answer"]["answer_source"] == "fallback_summary"


def test_unsupported_measurement_answer_recovers_from_grounded_summary(monkeypatch):
    result = SearchResult(
        chunk_id="temperature-limit",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[447],
        section_path=["Specifications"],
        content="For the IV4-G120, if ambient exceeds 40°C, case temperature must not exceed 65°C.",
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-G120"},
    )
    summary = [{
        "chunk_id": result.chunk_id,
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "summary": result.content,
        "source_document_id": result.source_document_id,
        "document_version_id": result.document_version_id,
    }]
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: ({
            "answer": "Set the network port to 21.",
            "confidence": "medium",
            "used_documents": [],
            "citations": [],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        }, "{}"),
    )

    answer, trace = generate_answer_with_trace(
        "What is the IV4-G120 case temperature rating?",
        [result],
        prioritized_results=[result],
        summarized_evidence=summary,
    )

    assert "65°C" in answer.answer
    assert "40°C" in answer.answer
    assert trace["final_answer"]["answer_source"] in {"fallback_summary_validation", "fallback_validation"}


def test_conditioned_measurement_limit_uses_condition_and_rated_value():
    result = SearchResult(
        chunk_id="temperature-limit",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[447],
        section_path=["Specifications"],
        content=(
            "Operating ambient temperature: 0 to +50°C. If the operating ambient temperature "
            "exceeds 40°C, confirm that the case temperature does not exceed the rated 65°C. "
            "Measure the case temperature at the metal on the back of the sensor amplifier."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-G120"},
    )

    answer, trace = generate_answer_with_trace(
        "What case temperature limit applies to the IV4-G120 if ambient exceeds 40°C?",
        [result],
    )

    assert answer.answer == (
        "For IV4-G120, if the operating ambient temperature exceeds 40°C, confirm that the case "
        "temperature does not exceed the rated 65°C."
    )
    assert trace["final_answer"]["answer_source"] == "deterministic_conditioned_measurement"


def test_physical_measurement_location_uses_location_sentence():
    result = SearchResult(
        chunk_id="temperature-location",
        score=0.9,
        title="IV4 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[447],
        section_path=["Specifications"],
        content=(
            "If ambient exceeds 40°C, the case temperature must not exceed 65°C. "
            "Measure the case temperature at the metal on the back of the sensor amplifier."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV4-G120"},
    )

    answer, trace = generate_answer_with_trace(
        "Where should I measure the case temperature on the IV4-G120 sensor amplifier?",
        [result],
    )

    assert answer.answer == (
        "For IV4-G120, measure the case temperature at the metal on the back of the sensor amplifier."
    )
    assert trace["final_answer"]["answer_source"] == "deterministic_conditioned_measurement"


def test_instruction_prefers_exact_light_fixture_shielding_phrase():
    exact = SearchResult(
        chunk_id="light-shielding-board",
        score=0.9,
        title="LJ-S8000 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[4],
        section_path=["Operating precautions"],
        content=(
            "Do not operate this device near lighting fixtures. If the unit must be used in such "
            "a location, install a light shielding board or similar device so that the light will "
            "not affect the measurement."
        ),
        metadata={"chunk_type": "section_window", "product_model": "LJ-S8000"},
    )
    laser_safety = SearchResult(
        chunk_id="laser-enclosure",
        score=0.95,
        title="LJ-S8000 Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[7],
        section_path=["Laser safety"],
        content=(
            "If the operator may be exposed to specular or diffuse reflections, block the beam "
            "by installing a protective enclosure."
        ),
        metadata={"chunk_type": "section_window", "product_model": "LJ-S8000"},
    )

    answer, trace = generate_answer_with_trace(
        "How should I shield the LJ-S8000 from nearby lighting fixtures?",
        [exact, laser_safety],
    )

    assert "install a light shielding board or similar device" in answer.answer
    assert "protective enclosure" not in answer.answer
    assert answer.citations[0]["chunk_id"] == "light-shielding-board"
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_instruction_keeps_adjacent_confirmation_and_save_step():
    result = SearchResult(
        chunk_id="convert-program-version",
        score=0.9,
        title="VS Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[162],
        section_path=["Convert Program Setting"],
        content=(
            "On the menu bar, select [File] - [Convert Program Setting to Latest Version] "
            "to open the [Confirm] dialog. Click the [OK] button to convert and save the "
            "currently displayed program setting as the latest file version."
        ),
        metadata={"chunk_type": "section_window", "product_model": "VS Series"},
    )

    answer, trace = generate_answer_with_trace(
        "How do I update a VS Series program setting to the latest version?",
        [result],
    )

    assert "select [File] - [Convert Program Setting to Latest Version]" in answer.answer
    assert "Click the [OK] button" in answer.answer
    assert "convert and save" in answer.answer
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_why_answer_keeps_the_causal_sentence_with_the_recommendation():
    result = SearchResult(
        chunk_id="relay-contact-bounce",
        score=0.9,
        title="IV-500C Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[6],
        section_path=["Input cables"],
        content=(
            "For input cables of this sensor, connect with non-contact output (transistor output/SSR output). "
            "For contact output (relay output), incorrect input may be operated due to the contact bouncing "
            "in the system."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "IV-500C"},
    )

    answer, _trace = generate_answer_with_trace(
        "Why should I avoid using relay output with the IV-500C sensor input?",
        [result],
    )

    assert "non-contact output" in answer.answer
    assert "contact bouncing" in answer.answer


def test_how_to_query_does_not_use_unrelated_structured_fact(monkeypatch):
    unrelated = SearchResult(
        chunk_id="other-settings",
        score=0.95,
        title="CA-EN100U",
        document_version_id="v1",
        source_document_id="d1",
        pages=[3],
        section_path=["Settings"],
        content="You can configure CA-EN100U settings from a PC.",
        metadata={"chunk_type": "spec_record", "product_model": "CA-EN100U"},
    )
    procedure = SearchResult(
        chunk_id="vision-system-switch",
        score=0.9,
        title="CA-EN100U",
        document_version_id="v1",
        source_document_id="d1",
        pages=[3],
        section_path=["Settings"],
        content='Set the switch on the RS-232C connector to "VISION SYSTEM".',
        metadata={"chunk_type": "atomic_text", "product_model": "CA-EN100U"},
    )
    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **_kwargs: ({
            "answer": 'Set the switch on the RS-232C connector to "VISION SYSTEM".',
            "confidence": "high",
            "used_documents": [],
            "citations": [],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        }, "{}"),
    )

    answer, trace = generate_answer_with_trace(
        "How do I set the CA-EN100U switch for the image processing system?",
        [unrelated, procedure],
        prioritized_results=[procedure, unrelated],
        summarized_evidence=[{
            "chunk_id": procedure.chunk_id,
            "title": procedure.title,
            "pages": procedure.pages,
            "section_path": procedure.section_path,
            "summary": procedure.content,
            "source_document_id": procedure.source_document_id,
            "document_version_id": procedure.document_version_id,
        }],
    )

    assert "VISION SYSTEM" in answer.answer
    assert "from a PC" not in answer.answer
    assert trace["final_answer"]["answer_source"] != "deterministic_structured_fact"


def test_factory_default_switch_question_uses_exact_instruction():
    wrong = SearchResult(
        chunk_id="pc-settings",
        score=0.95,
        title="CA-EN100U",
        document_version_id="v1",
        source_document_id="d1",
        pages=[3],
        section_path=["Settings"],
        content='Set the switch to "PC" when configuring from a device other than an image processing system.',
        metadata={"chunk_type": "spec_record", "product_model": "CA-EN100U"},
    )
    correct = SearchResult(
        chunk_id="vision-system-switch",
        score=0.9,
        title="CA-EN100U",
        document_version_id="v1",
        source_document_id="d1",
        pages=[3],
        section_path=["Settings"],
        content=(
            '1 Set the switch on the RS-232C connector to "VISION SYSTEM" '
            '(this is the factory default setting), and use the included cable.'
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "CA-EN100U"},
    )

    answer, trace = generate_answer_with_trace(
        "What is the factory default setting for the CA-EN100U RS-232C switch?",
        [wrong, correct],
    )

    assert answer.answer.startswith('Set the switch on the RS-232C connector to "VISION SYSTEM"')
    assert trace["final_answer"]["answer_source"] == "deterministic_instruction"


def test_temporal_effect_question_uses_subsequent_calibration_sentence():
    result = SearchResult(
        chunk_id="calibration-effect",
        score=0.9,
        title="LR-W Manual",
        document_version_id="v1",
        source_document_id="d1",
        pages=[5],
        section_path=["Calibration"],
        content=(
            "Changing the master calibration set value after a master calibration has been performed "
            "does not affect the current setting value, only subsequent calibrations."
        ),
        metadata={"chunk_type": "atomic_text", "product_model": "LR-W70"},
    )

    answer, trace = generate_answer_with_trace(
        "When does a new master calibration set value take effect on the LR-W70(C)?",
        [result],
    )

    assert "only subsequent calibrations" in answer.answer
    assert answer.answer.startswith("For LR-W70")
    assert trace["final_answer"]["answer_source"] == "deterministic_temporal_effect"


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
