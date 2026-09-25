import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "freeze_heldout_eval_bank.py"
_SPEC = importlib.util.spec_from_file_location("freeze_heldout_eval_bank", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _case():
    return {
        "case_id": "case-1",
        "source_document_id": "heldout-doc",
        "document_version_id": "version-1",
        "source_chunk_id": "chunk-1",
        "expected_snippet": "Install the sensor 20 mm away.",
        "expected_terms": ["sensor", "20 mm"],
    }


def _chunk():
    return {
        "id": "chunk-1",
        "source_document_id": "heldout-doc",
        "document_version_id": "version-1",
        "content": "Procedure:  Install the sensor 20 mm away. Then tighten it.",
        "is_active": True,
    }


def test_freezes_only_source_verified_document_disjoint_cases():
    frozen = _MODULE.verify_and_freeze_cases(
        [_case()],
        {"chunk-1": _chunk()},
        tuning_document_ids={"different-doc"},
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert frozen[0]["evaluation_split"] == "held_out"
    assert frozen[0]["adjudication"]["status"] == "source_verified"
    assert frozen[0]["adjudication"]["human_reviewed"] is False
    assert len(frozen[0]["adjudication"]["source_chunk_sha256"]) == 64


def test_optional_source_reanchor_replaces_answerless_generated_snippet():
    case = {
        **_case(),
        "query": "What distance applies to MODEL-7?",
        "expected_snippet": "Model: Distance",
        "expected_terms": ["model", "distance"],
    }
    chunk = {
        **_chunk(),
        "content": "Model: Distance; MODEL-7: 20 mm",
    }

    frozen = _MODULE.verify_and_freeze_cases(
        [case],
        {"chunk-1": chunk},
        tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
        reanchor_source_snippets=True,
    )

    assert frozen[0]["expected_snippet"] == "MODEL-7: 20 mm"
    assert frozen[0]["expected_terms"] == ["model-7", "20"]


def test_drops_generic_model_header_when_question_asks_for_another_value():
    case = {
        **_case(),
        "query": "Which EMC standards does the CA-U5 power supply meet?",
        "expected_snippet": (
            "Model: EMC standard; CA-U5: FCC Part15B ClassA, "
            "EN55011 ClassA, EN61000-6-2"
        ),
        "expected_terms": ["model", "standard", "ca-u5", "part15b"],
        "anchor_terms": ["model", "standard", "ca-u5", "part15b"],
    }
    chunk = {
        **_chunk(),
        "content": case["expected_snippet"],
    }

    frozen = _MODULE.verify_and_freeze_cases(
        [case],
        {"chunk-1": chunk},
        tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert frozen[0]["expected_terms"] == ["standard", "ca-u5", "part15b"]
    assert frozen[0]["anchor_terms"] == ["standard", "ca-u5", "part15b"]


def test_keeps_model_term_when_question_explicitly_asks_for_model():
    assert _MODULE.answer_relevant_expected_terms(
        "Which model meets this EMC standard?",
        ["model", "standard", "ca-u5"],
    ) == ["model", "standard", "ca-u5"]


def test_how_many_contract_drops_sibling_model_counts():
    assert _MODULE.answer_relevant_expected_terms(
        "How many protection zones does the SZ-V04 safety laser scanner support?",
        ["protection zones", "2", "1"],
    ) == ["protection zones", "2"]


def test_display_code_meaning_contract_drops_layout_label():
    assert _MODULE.answer_relevant_expected_terms(
        "What does the ErC display code indicate on the LR-W500?",
        ["display", "cause", "excessive", "current"],
    ) == ["cause", "excessive", "current"]


def test_rejects_question_that_drops_axis_qualifier():
    case = {
        **_case(),
        "query": "What reference distance applies to the LJ-S015 sensor?",
        "expected_snippet": "LJ-S015: 15 mm",
        "expected_terms": ["lj-s015", "15 mm"],
    }
    chunk = {
        **_chunk(),
        "content": "Model name: X Reference distance; LJ-S015: 15 mm; LJ-S025: 23 mm",
    }

    with pytest.raises(ValueError, match="drops source qualifier.*x-axis"):
        _MODULE.verify_and_freeze_cases(
            [case],
            {"chunk-1": chunk},
            tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
            reanchor_source_snippets=True,
        )


def test_accepts_question_that_preserves_axis_qualifier():
    case = {
        **_case(),
        "query": "What X-axis reference distance applies to the LJ-S015 sensor?",
        "expected_snippet": "LJ-S015: 15 mm",
        "expected_terms": ["lj-s015", "15 mm"],
    }
    chunk = {
        **_chunk(),
        "content": "Model name: X Reference distance; LJ-S015: 15 mm; LJ-S025: 23 mm",
    }

    frozen = _MODULE.verify_and_freeze_cases(
        [case],
        {"chunk-1": chunk},
        tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
        reanchor_source_snippets=True,
    )

    assert frozen[0]["query"].startswith("What X-axis")


def test_rejects_generic_monitor_model_when_source_names_specific_model():
    assert _MODULE.missing_query_qualifiers(
        "What resolution and color depth does the Monitor model support?",
        "IV2-H1: Resolution: 1024 × 768 pixels or higher, Display color: High Color (16 bit)",
        "IV2-H1: Resolution: 1024 × 768 pixels or higher, Display color: High Color (16 bit)",
    ) == ["monitor model identifier"]


def test_rejects_family_wide_question_when_context_names_model_variant():
    assert _MODULE.missing_query_qualifiers(
        "What is the exposure time range for the VS Series camera?",
        "Exposure time | 0.037 msec to 1000 msec",
        "Exposure time | 0.037 msec to 1000 msec",
        "Next chunk: Table header: VS-LxxxCX; Header role: column",
    ) == ["model variant"]


def test_accepts_model_family_prefix_from_structural_context():
    assert _MODULE.missing_query_qualifiers(
        "What is the exposure time range for the VS-L camera family?",
        "Exposure time | 0.037 msec to 1000 msec",
        "Exposure time | 0.037 msec to 1000 msec",
        "Next chunk: Table header: VS-LxxxCX; Header role: column",
    ) == []


@pytest.mark.parametrize(
    "query",
    [
        "What shutter speed range can I set on this camera?",
        "What DC voltage range powers this laser sensor?",
        "How is that distance based laser sensor wired?",
        "What colors appear on these laser sensors?",
        "Can this filter smooth images along the X axis?",
    ],
)
def test_rejects_standalone_question_with_deictic_product_subject(query):
    assert _MODULE.missing_query_qualifiers(
        query,
        "Electronic shutter | Can be set to 0.022 to 1000 msec",
        "Electronic shutter | Can be set to 0.022 to 1000 msec",
    ) == ["explicit subject"]


def test_accepts_that_as_relative_pronoun_not_deictic_subject():
    assert _MODULE.missing_query_qualifiers(
        "What warning appears for an IP address that already belongs to a PLC?",
        "Warning: The IP address is already in use by another device.",
        "Warning: The IP address is already in use by another device.",
    ) == []


def test_rejects_generic_device_spec_without_product_model_scope():
    assert _MODULE.missing_query_qualifiers(
        "What shock resistance rating applies to the laser sensor in X, Y, and Z axes?",
        "Shock resistance | 1000 m/s 2 in X, Y, Z axis directions respectively 6 times",
        "Shock resistance | 1000 m/s 2 in X, Y, Z axis directions respectively 6 times",
    ) == ["explicit product/model"]


def test_accepts_device_spec_with_product_model_scope():
    assert _MODULE.missing_query_qualifiers(
        "What shock resistance rating applies to the LR-ZH500N sensor in X, Y, and Z axes?",
        "Shock resistance | 1000 m/s 2 in X, Y, Z axis directions respectively 6 times",
        "Shock resistance | 1000 m/s 2 in X, Y, Z axis directions respectively 6 times",
    ) == []


@pytest.mark.parametrize(
    ("query", "content"),
    [
        (
            "What does the one shot input do to the output status of current results?",
            "The one shot input resets outputs.",
        ),
        (
            "What minimum detectable object size must be selected if the detection plane height exceeds 1000 mm?",
            "Minimum detectable object size is 70 mm.",
        ),
    ],
)
def test_rejects_product_specific_control_without_product_scope(query, content):
    assert _MODULE.missing_query_qualifiers(
        query,
        content,
        content,
    ) == ["explicit product/model"]


def test_rejects_torque_question_that_drops_waterproof_cap_scope():
    assert _MODULE.missing_query_qualifiers(
        "What tightening torque is required for the IV4-400CA connector?",
        "When the cable is not connected, attach the waterproof cap. Tightening torque: 0.45 to 0.55 N m",
        "Tightening torque: 0.45 to 0.55 N m",
    ) == ["waterproof cap"]


def test_accepts_torque_question_with_waterproof_cap_scope():
    assert _MODULE.missing_query_qualifiers(
        "What tightening torque is required for the IV4-400CA waterproof cap?",
        "When the cable is not connected, attach the waterproof cap. Tightening torque: 0.45 to 0.55 N m",
        "Tightening torque: 0.45 to 0.55 N m",
    ) == []


def test_rejects_temperature_contract_with_leading_unrelated_quantity():
    assert _MODULE.missing_expected_answer_contract(
        "What is the operating ambient temperature range of the WM-6025?",
        "1.25 A: Operating ambient temperature 0 to 40 C",
        ["0", "40", "operating ambient temperature"],
    ) == ["leading unrelated quantity"]


def test_accepts_clean_operating_temperature_contract():
    assert _MODULE.missing_expected_answer_contract(
        "What is the operating ambient temperature range of the WM-6025?",
        "Operating ambient temperature 0 to 40 C",
        ["0", "40", "operating ambient temperature"],
    ) == []


def test_rejects_corpus_wide_numeric_spec_without_product_scope():
    assert _MODULE.missing_query_qualifiers(
        "What ambient temperature range is allowed for operation without freezing?",
        "Operating ambient temperature: 0 to +50 C (no freezing)",
        "Operating ambient temperature: 0 to +50 C (no freezing)",
    ) == ["explicit product/model"]


def test_rejects_named_product_absent_from_source_scope():
    assert _MODULE.missing_query_qualifiers(
        "Does the SZ-FB31 bracket protect the light curtain from impacts?",
        "Ultra-robust structure protects the light curtain from strong impacts.",
        "Ultra-robust structure protects the light curtain from strong impacts.",
    ) == ["source scope sz-fb31"]


def test_accepts_named_product_present_in_source_scope_context():
    assert _MODULE.missing_query_qualifiers(
        "Which illumination methods are supported by the CA-F100 series?",
        "Illumination method: block lighting and pattern projection",
        "Illumination method: block lighting and pattern projection",
        "Table header: CA-F100 series",
    ) == []


def test_rejects_selection_recommendation_without_selection_criterion():
    assert _MODULE.missing_query_qualifiers(
        "Which analog output type should I select?",
        "Analog output: Select current output or voltage output.",
        "Analog output: Select current output or voltage output.",
    ) == ["selection criterion"]

    assert _MODULE.missing_query_qualifiers(
        "Should I use a zoom camera when selecting the resolution for my application?",
        "Select the camera resolution. Selecting a zoom camera. Select the resolution according to your application.",
        "Select the camera resolution. Selecting a zoom camera.",
    ) == ["selection criterion"]


def test_rejects_effect_question_without_with_and_without_output_load_evidence():
    assert _MODULE.missing_answer_requirements(
        "How does including an output load of 120 mA affect current consumption?",
        "3.4 A or less, including an output load of 120 mA",
    ) == ["with/without output-load comparison"]


def test_rejects_benefit_question_without_benefit_statement():
    assert _MODULE.missing_answer_requirements(
        "Why is the zoom function beneficial for optical adjustments?",
        "Select the camera type. Zoom smart camera. This unit supports many applications.",
    ) == ["benefit statement"]


def test_rejects_angle_question_anchored_to_linear_resolution():
    assert _MODULE.missing_answer_requirements(
        "What is the display resolution for the CA-S20D when measuring angles?",
        'CA-S20D: Display resolution 1 mm 0.04" (0.1 mm with vernier scale)',
    ) == ["angular measurement"]


def test_rejects_output_protection_question_anchored_to_input_timing():
    assert _MODULE.missing_answer_requirements(
        "What protection features are included in the laser sensor's control output circuit?",
        "External input: 35 ms or more ON; laser emission stop: 2 ms or more ON",
    ) == ["output protection feature"]


def test_rejects_checklist_question_anchored_only_to_fault_description():
    assert _MODULE.missing_answer_requirements(
        "What checks should I perform if the LR-ZH500C3P shows ErC?",
        "Display: ErC; Description: Current of 100 mA or more flows through the control output",
    ) == ["diagnostic check action"]


def test_rejects_rated_voltage_question_anchored_to_current_value():
    assert _MODULE.missing_answer_requirements(
        "What is the rated voltage for the WM-P6000 battery operation?",
        "WM-P6000: 1.25 A",
    ) == ["rated voltage value"]


@pytest.mark.parametrize(
    ("query", "snippet", "requirement"),
    [
        (
            "What is the approximate spot size at a working distance of 240 mm?",
            "Distance based laser sensor: 1.5 ms / 10 ms / 50 ms selectable",
            "spot size value",
        ),
        (
            "What pixel dimensions does the VJ-H048CX/H048MX model support?",
            "0.47 megapixel mode: (H) × 596 (V), approx",
            "complete pixel dimensions",
        ),
        (
            "What load resistance limits apply to the 4-20 mA current output?",
            "Current output: 4 to 20 mA with a max",
            "load resistance value",
        ),
    ],
)
def test_rejects_question_when_requested_measurement_is_absent(query, snippet, requirement):
    assert _MODULE.missing_answer_requirements(query, snippet) == [requirement]


def test_rejects_object_size_limit_contract_that_only_names_forbidden_size():
    assert _MODULE.missing_answer_requirements(
        "What object size limit applies when detection plane height is 1000 mm or less?",
        "You cannot select the object size of 150 mm when height is 1000 mm or less.",
    ) == ["applicable object size"]


def test_rejects_ethernet_speed_contract_with_truncated_second_standard():
    assert _MODULE.missing_answer_requirements(
        "Which Ethernet speeds does the XG-X2902LJ support?",
        "Supports BOOTP functions; 1000BASE-T/100",
    ) == ["complete Ethernet speeds"]


def test_rejects_input_type_question_anchored_to_power_voltage_field():
    assert _MODULE.missing_answer_requirements(
        "What input type does the VJ-3302 model support?",
        "VJ-3302: Power voltage",
    ) == ["input type"]


def test_rejects_password_range_question_without_numeric_range():
    assert _MODULE.missing_answer_requirements(
        "What password range disables the Key Lock on the W500?",
        "An optional password can be set to prohibit unauthorized releasing of the Key Lock.",
    ) == ["password range"]


def test_rejects_plc_link_ports_question_anchored_to_incompatible_interface():
    assert _MODULE.missing_answer_requirements(
        "Which ports support PLC Link communication on the LJ-S8002?",
        "Ethernet port or optional EtherNet/IP unit (Cannot be used with PLC Link).",
    ) == ["PLC Link port mapping"]


def test_rejects_devid_format_question_bound_to_neighboring_str_field():
    assert _MODULE.missing_answer_requirements(
        "What character string format does the devId parameter expect?",
        "devId: device ID, 2 for RS-232C and 3 for Ethernet str: character string",
    ) == ["devId character-string binding"]
    assert _MODULE.missing_answer_requirements(
        "What character string format does the devId parameter expect?",
        "devId: character string containing the device ID",
    ) == []


def test_rejects_accuracy_question_anchored_to_measurement_range_field():
    assert _MODULE.missing_query_qualifiers(
        "What is the scanning system accuracy specification for the WM-P6200 model?",
        "Measurement range: WM-P6200: ±100 mm",
        "WM-P6200: ±100 mm",
    ) == ["accuracy field"]


def test_enriches_numeric_mapping_with_requested_protocol_value():
    query = "What numeric value represents RS-232C communication for the OutputFilter devId parameter?"
    snippet = "devId: the device ID. 2 for RS-232C, and 3 for Ethernet"

    terms = _MODULE.enrich_expected_answer_terms(
        query,
        snippet,
        ["devid", "device", "rs-232c", "ethernet"],
    )

    assert "2" in terms
    assert "3" not in terms
    assert _MODULE.missing_expected_answer_contract(query, snippet, terms) == []


def test_rejects_default_ip_question_without_ip_value_and_enriches_valid_value():
    query = "What is the default IP address for the LJ-X8000 before configuration?"
    assert _MODULE.missing_expected_answer_contract(
        query,
        "Set the IP address of the LJ-X8000.",
        ["set", "address", "lj-x8000"],
    ) == ["IP address value"]

    snippet = "IP address initial value: 192.168.10.10"
    terms = _MODULE.enrich_expected_answer_terms(query, snippet, ["ip address", "initial value"])
    assert "192.168.10.10" in terms
    assert _MODULE.missing_expected_answer_contract(query, snippet, terms) == []


def test_reanchors_hdd_capacity_to_capacity_value_instead_of_neighboring_details():
    query = "What is the maximum HDD capacity supported on the LJ-S8002 USB port?"
    snippet = (
        "LJ-S8002: Images and other data can be output by connecting an HDD (2TB max.) "
        "to the USB port, rated output 900 mA."
    )
    terms = _MODULE.enrich_expected_answer_terms(
        query,
        snippet,
        ["lj-s8002", "images", "other", "connecting", "900"],
    )

    assert terms == ["lj-s8002", "HDD", "2TB"]
    assert _MODULE.missing_expected_answer_contract(query, snippet, terms) == []


def test_focuses_mounting_hole_contract_on_requested_torque_clause():
    query = "What tightening torque is required for the back M2.5 mounting hole?"
    snippet = (
        "In addition to the back M3 mounting hole, the sensor can be mounted by the "
        "back M2.5 (depth 3.4 mm, tightening torque: 0.2 to 0.3 N·m) and the front "
        "M4 (depth 4.1 mm, tightening torque: 0.8 to 1.2 N·m) hole."
    )

    focused = _MODULE.focus_expected_snippet(query, snippet)

    assert focused == "tightening torque: 0.2 to 0.3 N·m"
    assert _MODULE._answer_quantity_values(query, focused) == ["0.2", "0.3"]


def test_focuses_numerical_input_contract_on_the_requested_enumeration():
    query = "What numerical inputs can be specified for the electronic shutter setting?"
    snippet = (
        "Electronic shutter | Can be set to 0.05 to 9000 msec by specifying the following "
        "numerical inputs: 1/15, 1/30, 1/60, 1/120, 1/240, 1/500, 1/1000, 1/2000, "
        "1/5000, 1/10000, 1/20000"
    )

    focused = _MODULE.focus_expected_snippet(query, snippet)

    assert focused == (
        "Numerical inputs: 1/15, 1/30, 1/60, 1/120, 1/240, 1/500, 1/1000, "
        "1/2000, 1/5000, 1/10000, 1/20000"
    )


def test_focuses_ultra_narrow_field_of_view_on_requested_distance_column():
    query = "What is the field of view for the ultra-narrow model at 23 mm to 40 mm?"
    snippet = (
        'Field of view | Installation distance of 23 mm0.91": 9.8 (H) × 7.3 (V)mm '
        'to Installation distance of 40 mm1.57": 15 (H) × 11.2 (V)mm | '
        'Installation distance of 400 mm15.75": 58 (H) × 44 (V)mm'
    )

    focused = _MODULE.focus_expected_snippet(query, snippet)

    assert "9.8 (H) × 7.3 (V)" in focused
    assert "15 (H) × 11.2 (V)" in focused
    assert "400 mm" not in focused


def test_focuses_analog_option_across_period_inside_bracketed_label():
    focused = _MODULE.focus_expected_snippet(
        "Which analog output option sends the unit's displayed value?",
        "Select the data to output in analog format: Display value [Disp",
        "Select the data to output in analog format: Display value [Disp. Value] *1.",
    )

    assert focused == "Display value [Disp. Value]"


def test_focuses_download_option_on_exact_named_choice():
    focused = _MODULE.focus_expected_snippet(
        "Which download option transmits only the changed hardware and software?",
        "After compilation, right-click the PLC and choose the download command.",
        (
            "After compilation, right-click the PLC, point to Download to device, and click "
            "Hardware and software (only changes) to transmit the compiled program."
        ),
    )

    assert focused == "Hardware and software (only changes)"


def test_focuses_object_size_contract_on_forbidden_and_applicable_limits():
    focused = _MODULE.focus_expected_snippet(
        "What object size limit applies when detection plane height is 1000 mm or less?",
        "You cannot select the object size of 150 mm when height is 1000 mm or less.",
        (
            "You cannot select the object size of 150 mm when height is 1000 mm or less. "
            "You must select the object size of 70 mm or smaller."
        ),
    )

    assert "150 mm" in focused
    assert "70 mm or smaller" in focused


def test_focuses_ethernet_speed_contract_on_complete_standards_from_source():
    focused = _MODULE.focus_expected_snippet(
        "Which Ethernet speeds does the XG-X2902LJ support?",
        "Supports BOOTP functions; 1000BASE-T/100",
        "Supports BOOTP functions; 1000BASE-T/100BASE-TX",
    )

    assert focused == "Ethernet speeds: 1000BASE-T, 100BASE-TX"


def test_reanchors_power_cable_contract_to_answer_bearing_facts():
    query = "How do I power the LJ-S8000 head using the power I/O cable?"
    snippet = (
        "Supply 24 V DC to the power I/O connector using the power I/O cable for head, "
        "and connect the head Ethernet cable to the Ethernet connector."
    )

    terms = _MODULE.enrich_expected_answer_terms(query, snippet, ["supply", "power", "connector"])

    assert terms == ["24 V DC", "power I/O connector", "power I/O cable", "Ethernet connector"]


def test_reanchors_profinet_cyclic_optional_unit_to_ca_npn_identifier():
    terms = _MODULE.enrich_expected_answer_terms(
        "Which optional unit is required for PROFINET cyclic communication on the XG-X2902LJ?",
        (
            "XG-X2902LJ: Complies with Conformance Class A (Ethernet port) / C(CA-NPN20E). "
            "Supports cyclic communication."
        ),
        ["xg-x2902lj", "complies", "conformance", "class"],
    )

    assert terms == ["CA-NPN20E", "PROFINET", "cyclic communication"]


def test_focuses_profinet_cyclic_optional_unit_on_profinet_row():
    source = (
        "Model: PROFINET; XG-X2902LJ: • Can output numerical values and perform control I/O "
        "using the Ethernet port or the optional PROFINET unit CA-NPN20E. "
        "• Supports cyclic communication (max. 1408 bytes (Ethernet port) / 1248 bytes "
        "(CA-NPN20E)) • Supports acyclic communication (recorded data) "
        "Model: EtherCAT; XG-X2902LJ: • Uses CA-NEC20E."
    )

    focused = _MODULE.focus_expected_snippet(
        "Which optional unit is required for PROFINET cyclic communication on the XG-X2902LJ?",
        "XG-X2902LJ: CA-NPN20E supports cyclic communication.",
        source,
    )

    assert focused.startswith("Model: PROFINET; XG-X2902LJ:")
    assert "optional PROFINET unit CA-NPN20E" in focused
    assert "Supports cyclic communication" in focused
    assert "CA-NEC20E" not in focused


def test_removes_ca_dex10x_ocr_footnote_from_frozen_query():
    assert _MODULE.normalize_frozen_query(
        "How much power does the VS Series consume if CA-DEx10X 4 is connected?"
    ) == "How much power does the VS Series consume if CA-DEx10X is connected?"


def test_repairs_lj_s8000_ocr_model_separator_in_frozen_query():
    assert _MODULE.normalize_frozen_query(
        "What is the movable range for the NEW LJ: S8000 Series sensor?"
    ) == "What is the movable range for the LJ-S8000 Series sensor?"


def test_qualifies_laser_on_activation_by_source_manual():
    expected = (
        "In the AS_124150 LJ-X8000 communication manual, how do I activate "
        "the Laser ON input?"
    )

    assert _MODULE.normalize_frozen_query(
        "How do I activate the Laser ON input on this device?"
    ) == expected
    assert _MODULE.normalize_frozen_query(
        "How do I activate the Laser ON input on the LJ-X8000 controller?"
    ) == expected


def test_qualifies_ca_e100_camera_count_by_source_manual():
    assert _MODULE.normalize_frozen_query(
        "How many cameras connect to one CA-E100 area camera input unit?"
    ) == (
        "In the AS_160148 XG-X manual, how many color/monochrome cameras connect "
        "to one CA-E100 area camera input unit?"
    )


def test_qualifies_iv_500c_screw_size_by_mounting_orientation():
    assert _MODULE.normalize_frozen_query(
        "Which screw size is specified for the IV-500C sensor mounting?"
    ) == "Which screw size is specified for wall-mounting the IV-500C sensor?"


def test_repairs_saved_settings_query_with_exact_activation_context():
    assert _MODULE.normalize_frozen_query(
        "In the VS Series KUKA robot connection manual, what action must be taken "
        "after saving settings to enable them?"
    ) == (
        "In the VS Series KUKA robot connection manual, after pressing Save and "
        "selecting Yes, what must be done to enable the changed settings?"
    )


def test_repairs_pc_to_plc_query_with_exact_protocol_scope():
    assert _MODULE.normalize_frozen_query(
        "Which menu path transfers data from the PC to the PLC?"
    ) == (
        "In the LJ-X8000 EtherNet/IP setup for CompactLogix or ControlLogix, which "
        "menu path transfers data from the PC to the PLC?"
    )


def test_qualifies_conflicting_vs_c160m_frame_rate_by_source_manual():
    assert _MODULE.normalize_frozen_query(
        "What frame rate does the VS-C160M/CX model support?"
    ) == (
        "In the AS_145861 VS-C specification manual, what frame rate is listed "
        "for VS-C160M/CX?"
    )
    assert _MODULE.normalize_frozen_query(
        "In the AS_145861 VS-C specification manual, what frame rate is listed "
        "for the VS-C160M/CX model?"
    ) == (
        "In the AS_145861 VS-C specification manual, what frame rate is listed "
        "for VS-C160M/CX?"
    )


def test_repairs_w500_password_query_to_match_the_source_contract():
    assert _MODULE.normalize_frozen_query(
        "What password range disables the Key Lock on the W500?"
    ) == "What password values can be set for the W500 Key Lock, and what does selecting 0 do?"


def test_repairs_ca_f100_section_label_to_pattern_light_model_scope():
    assert _MODULE.normalize_frozen_query(
        "Which illumination methods are supported by the CA-F100 series?"
    ) == (
        "Which illumination methods are listed for the CA-DQP12X and CA-DQP25X "
        "pattern-projection lights?"
    )


def test_focuses_w500_password_contract_on_range_and_zero_meaning():
    query = "What password values can be set for the W500 Key Lock, and what does selecting 0 do?"
    source = (
        "An optional password can be set to further prohibit unauthorized releasing of the "
        "'6-1 Key Lock' (page 4). Select a value from 1 to 999 for this setting. "
        "If '0' is selected, the password will not be required."
    )

    focused = _MODULE.focus_expected_snippet(query, source, source)
    terms = _MODULE.enrich_expected_answer_terms(query, focused, ["optional", "password"])

    assert "1 to 999" in focused
    assert "password will not be required" in focused
    assert terms == ["1", "999", "0", "password", "required"]


def test_focuses_laser_on_activation_on_voltage_type_and_shorting_action():
    query = (
        "In the AS_124150 LJ-X8000 communication manual, how do I activate "
        "the Laser ON input?"
    )
    source = (
        "The Laser ON input is a non: voltage input; "
        "(Turns ON by simply short circuiting it) Other terminal details follow."
    )

    focused = _MODULE.focus_expected_snippet(query, source, source)
    terms = _MODULE.enrich_expected_answer_terms(
        query,
        focused,
        ["laser", "voltage", "turns", "short"],
    )

    assert focused == (
        "The Laser ON input is a non: voltage input; "
        "(Turns ON by simply short circuiting it)"
    )
    assert terms == ["laser", "short"]


def test_focuses_ca_e100_camera_count_on_one_unit_not_capture_capacity():
    query = (
        "In the AS_160148 XG-X manual, how many color/monochrome cameras connect "
        "to one CA-E100 area camera input unit?"
    )
    source = (
        "Model: With area camera input unit CA-E100 connected: "
        "2 color/monochrome cameras per CA-E100, up to 4 cameras via a maximum "
        "of 2 units can be connected. Simultaneous/individual capture with up "
        "to 4 cameras/heads can be selected."
    )

    focused = _MODULE.focus_expected_snippet(query, source, source)
    terms = _MODULE.enrich_expected_answer_terms(
        query,
        focused,
        ["simultaneous/individual", "capture", "cameras/heads"],
    )

    assert focused == (
        "With area camera input unit CA-E100 connected: "
        "2 color/monochrome cameras per CA-E100"
    )
    assert terms == ["CA-E100", "2", "color/monochrome cameras"]


def test_focuses_iv_500c_wall_mounting_on_m3_x_4_not_neighboring_screws():
    query = "Which screw size is specified for wall-mounting the IV-500C sensor?"
    source = (
        "Mounting on the wall Screw: M3 x 4 Use the commercially available screws. "
        "Mounting from the jig side Screw: M4 x 4 Use the commercially available screws. "
        "Fix the mounting adapter and sensor using the attached screws. Screw: M3 x 1."
    )

    focused = _MODULE.focus_expected_snippet(query, source, source)
    terms = _MODULE.enrich_expected_answer_terms(
        query,
        focused,
        ["screw", "commercially", "available", "3"],
    )

    assert focused == "Mounting on the wall — Screw: M3 x 4"
    assert terms == ["M3 x 4"]


def test_focuses_wm_p6200_scanning_accuracy_on_requested_row():
    query = "What is the scanning system accuracy specification for the WM-P6200 model?"
    source = (
        "model: Scanning system accuracy; WM-P6200: ±(50 + 5 L/1000) µm* 1\n"
        "model: Repeatability; WM-P6200: 0.025 mm0.001\" * 2\n"
        "model: Depth of field; WM-P6200: ±100 mm±3.94\""
    )

    focused = _MODULE.focus_expected_snippet(query, source, source)
    terms = _MODULE.enrich_expected_answer_terms(query, focused, ["wm-p6200", "l/1000"])

    assert focused == "WM-P6200: ±(50 + 5 L/1000) µm* 1"
    assert terms == ["wm-p6200", "l/1000", "50", "5"]


def test_focuses_iv_h500ca_installed_distance_on_range_not_view_dimensions():
    query = "What installed distance range applies to the IV-H500CA model?"
    source = (
        'Type | Standard distance | Short range | Long range Installed distance | '
        '50 to 500 mm 1.97" to 19.69" | 50 to 150 mm 1.97" to 5.91" | '
        'Model: View; IV-H500CA: Installed distance 50 mm 1.97": 25 (H) × 18 (V) '
        'to installed distance 500 mm 19.69"'
    )

    focused = _MODULE.focus_expected_snippet(query, source, source)

    assert focused == 'Installed distance | 50 to 500 mm 1.97" to 19.69"'


def test_repairs_generic_xg_x_shutter_query_with_document_and_model_scope():
    expected = (
        "In the AS_160148 XG-X camera specification table, what electronic shutter range "
        "is listed for the CA-H048CX or CA-H048MX?"
    )

    assert _MODULE.normalize_frozen_query(
        "What shutter speed range can I set on this camera?"
    ) == expected
    assert _MODULE.normalize_frozen_query(
        "What electronic shutter speed range can I set on an XG-X Series camera?"
    ) == expected
    assert _MODULE.normalize_frozen_query(
        "What electronic shutter speed range can I set on a CA-200C or CA-200M camera "
        "in the XG-X Series?"
    ) == expected
    assert _MODULE.normalize_frozen_query(
        "In the AS_160148 XG-X camera specification table, what electronic shutter range "
        "is listed immediately before the C-mount lens-mount row?"
    ) == expected
    assert _MODULE.normalize_frozen_query(
        "In the AS_160148 XG-X camera specification table, what electronic shutter range "
        "is listed for the CA-200C or CA-200M?"
    ) == expected


def test_repairs_existing_xg_x_dent_query_with_series_scope():
    assert _MODULE.normalize_frozen_query(
        "For the XG-X inline 3D inspection system, which dent-depth conditions can be "
        "inspected by freely setting the reference plane?"
    ) == (
        "For the XG-X Series inline 3D inspection system, which dent-depth conditions "
        "can be inspected by freely setting the reference plane?"
    )


def test_dent_range_contract_scores_answer_bearing_terms_not_grammatical_subject():
    query = (
        "For the XG-X Series inline 3D inspection system, which dent-depth conditions "
        "can be inspected by freely setting the reference plane?"
    )

    assert _MODULE.answer_relevant_expected_terms(
        query,
        ["users", "freely", "set", "reference"],
    ) == ["sharp", "shallow", "dents", "reference"]


def test_analog_display_contract_accepts_the_manuals_abbreviated_option_label():
    assert _MODULE.answer_relevant_expected_terms(
        "Which analog output option sends the unit's displayed value?",
        ["display"],
    ) == ["disp"]


def test_image_capacity_comparison_scores_both_answer_values_not_marketing_copy():
    assert _MODULE.answer_relevant_expected_terms(
        "How many images can the controller store with VGA color cameras versus 21 megapixel cameras?",
        ["furthermore", "largest-in-class", "image", "memory", "28"],
    ) == ["28,300", "290"]


def test_iv4_output_question_requests_the_full_scored_configuration():
    assert _MODULE.normalize_frozen_query(
        "Can the N.O./N.C. configuration be switched on the IV4-400MA output?"
    ) == (
        "What output type and switchable configurations are specified for the IV4-400MA?"
    )


def test_qualifies_iv_output_rating_by_manual_and_full_scored_contract():
    assert _MODULE.normalize_frozen_query(
        "What are the maximum voltage and current ratings for the IV Series open collector NPN output?"
    ) == (
        "In the AS_145624 IV Series specification table, what output type, NPN/PNP and "
        "N.O./N.C. switchable configurations, maximum NPN rating, and remaining voltage "
        "are specified?"
    )


def test_qualifies_iv_in1_edge_timing_by_source_manual():
    assert _MODULE.normalize_frozen_query(
        "Which edge timings can be set for the IV Series IN1 input when it is assigned as an external trigger?"
    ) == (
        "In the AS_145624 IV Series specification table, which edge timings can be set "
        "for the IN1 input when it is assigned as an external trigger?"
    )


def test_qualifies_lj_s8000_color_range_by_manual_and_source_control():
    assert _MODULE.normalize_frozen_query(
        "How do I adjust the color range for height data on the LJ-S8000?"
    ) == (
        "In the LJ-S8000 Easy Configuration Manual, which icon should I click to adjust "
        "the color range depending on the specification method?"
    )


def test_qualifies_vs_camera_only_power_contract_by_source_and_both_voltages():
    assert _MODULE.normalize_frozen_query(
        "What is the power consumption of the camera when only the sensor is active at 19.2 V?"
    ) == (
        "In the AS_151195 VS camera specification table, what current and power consumption "
        "are listed for camera-only operation at 19.2 V and 24 V?"
    )


def test_qualifies_lr_z_shock_rating_by_source_manual():
    expected = (
        "In the AS_86111 LR-Z specification table, what shock resistance rating applies "
        "in the X, Y, and Z axes?"
    )
    assert _MODULE.normalize_frozen_query(
        "What shock resistance rating applies to the laser sensor in X, Y, and Z axes?"
    ) == expected
    assert _MODULE.normalize_frozen_query(
        "What shock resistance rating applies to the LR-Z laser sensor in the X, Y, and Z axes?"
    ) == expected


@pytest.mark.parametrize(
    ("query", "required_scope"),
    [
        ("What shutter speed range can I set on this camera?", "AS_160148"),
        ("What ambient temperature range is allowed for operation without freezing?", "IV4 Series"),
        ("What does the one shot input do to the output status of current results?", "LJ-X8000"),
        ("How do I activate the Laser ON input on this device?", "AS_124150"),
        (
            "Which controllers support the high-resolution camera CA-HFxM/C in System configuration diagram XG?",
            "XG-X controllers",
        ),
        ("What shock resistance rating applies to the laser sensor in X, Y, and Z axes?", "LR-Z"),
        (
            "What is the recommended installation distance range for this megapixel resolution smart camera?",
            "IV4",
        ),
        ("What resolution and color depth does the Monitor model support?", "IV2-H1"),
        (
            "What minimum detectable object size must be selected if the detection plane height exceeds 1000 mm for area protection?",
            "SZ safety scanner",
        ),
        (
            "What display colors are assigned to the indicator, output, DATUM, and spot indicators on these laser sensors?",
            "LR-Z laser sensors",
        ),
        (
            "Which system configuration diagram applies when connecting to an XT controller?",
            "XG-X controllers",
        ),
        (
            "What part number applies to the infrared polarized filter for IV Series sensors?",
            "IV2-H1",
        ),
        (
            "What numerical inputs can be specified for the electronic shutter setting?",
            "CV-X camera specifications",
        ),
        (
            "What action must be taken after saving settings to enable them on the VS Series device?",
            "VS Series KUKA robot connection manual",
        ),
        (
            "In the VS Series KUKA robot connection manual, what action must be taken after saving settings to enable them?",
            "pressing Save and selecting Yes",
        ),
        (
            "Which menu path transfers data from the PC to the PLC?",
            "LJ-X8000 EtherNet/IP setup",
        ),
        (
            "What installation precaution applies when adjusting a manual-focus sensor after installation?",
            "IV-500C",
        ),
        ("Which amplifier models support the Intelligent Monitor feature?", "IV Series amplifier types"),
        (
            "Which dent-depth conditions can be inspected by freely setting the reference plane?",
            "XG-X Series inline 3D inspection system",
        ),
        (
            "Which numeric value should I use for devId if my XG controller connects via Ethernet?",
            "XG-7000 or XG-8000",
        ),
        ("What functions can OUT3 control when its default is set to Error?", "IV Series IN1 input"),
        (
            "What conditions allow the ShapeTrax TM 3A Search tool to maintain stable target search?",
            "CV-X catalog",
        ),
        (
            "What is the maximum image count for an XR 15 mm lens with binning enabled?",
            "AS_103012 LineScan system specification table",
        ),
        (
            "For the XR 15 mm lens in the LJ-X8000 line-scan system, what is the maximum image count with binning enabled?",
            "AS_103012 LineScan system specification table",
        ),
        (
            "How do I add a new EtherNet/IP module to the controller configuration?",
            "LJ-X8000 EtherNet/IP setup",
        ),
        (
            "Which parameters can be adjusted to set the optimal evaluation tolerance for a given application?",
            "ceramic protective-sheet lifting example",
        ),
        (
            "What additional distance applies to horizontal sensing without vertical sensing?",
            "SZ-V safety scanner",
        ),
        (
            "What power source supplies the WM-C6010 laser-scanning probe relay unit?",
            "How is the WM-C6010",
        ),
    ],
)
def test_repairs_generated_query_with_explicit_source_scope(query, required_scope):
    assert required_scope in _MODULE.normalize_frozen_query(query)


def test_focuses_safety_step_on_action_sentence_only():
    snippet = (
        "Check that power (24 VDC) is not being supplied to the CA-EN100U, and then "
        "connect the encoder head CA-EN100H to the encoder connector of the CA-EN100U. "
        "The cable length between the encoder head and the CA-EN100U can be extended up "
        "to 30 m."
    )

    focused = _MODULE.focus_expected_snippet(
        "What safety step must be taken before connecting the encoder head CA-EN100H to the CA-EN100U?",
        snippet,
    )

    assert focused.startswith("Check that power (24 VDC) is not being supplied")
    assert "30 m" not in focused


def test_focuses_powered_question_on_power_source_instead_of_rated_voltage():
    source = (
        "USBCommunication | USB 3.0 Infrared Communication | 945nm model: Power supply; "
        "WM-C6010: Supplied from dedicated AC; WM-C6025: adapter model: Ratings; "
        "WM-C6010: Rated voltage; WM-C6025: 24VDC"
    )

    focused = _MODULE.focus_expected_snippet(
        "How is the WM-C6010 laser-scanning probe relay unit powered?",
        "WM-C6010: Rated voltage",
        source,
    )

    assert focused == "WM-C6010: Supplied from dedicated AC"


def test_focuses_vj_3302_field_of_view_without_unasked_performance_values():
    source = (
        '60 mm 2.36" field of view, 1 µm 0.000039" precision repeatability, '
        "0.6-second inspection intervals."
    )

    focused = _MODULE.focus_expected_snippet(
        "What is the field of view size for the VJ-3302 inspection system?",
        source,
        source,
    )

    assert focused == '60 mm 2.36" field of view'


def test_rejects_powered_question_contract_that_only_names_rated_voltage():
    assert "power-source answer" in _MODULE.missing_expected_answer_contract(
        "How is the WM-C6010 laser-scanning probe relay unit powered?",
        "WM-C6010: Rated voltage",
        ["wm-c6010", "rated", "voltage"],
    )


def test_enriches_multivalue_measurement_contract_with_milliwatts():
    terms = _MODULE.enrich_expected_answer_terms(
        "What wavelength and output power are specified for the LJ-X8000 laser radiation?",
        "Laser radiation Class 2M; Wavelength: 405 nm; Output: 10 mW",
        ["laser radiation", "class 2m", "405"],
    )

    assert "405" in terms
    assert "10" in terms


def test_rejects_display_range_question_that_drops_displayed_quantity():
    assert _MODULE.missing_query_qualifiers(
        "What is the display range for the W500 sensor?",
        "Display range: 0 to 999 (The more the workpiece conform to reference workpiece, the higher the value.)",
        "Display range: 0 to 999 (The more the workpiece conform to reference workpiece, the higher the value.)",
    ) == ["workpiece conformity"]


def test_accepts_display_range_question_with_displayed_quantity():
    assert _MODULE.missing_query_qualifiers(
        "What is the display range for received light intensity on the W500?",
        "Display range: 0 to 999 (The greater the received light intensity, the higher the value.)",
        "Display range: 0 to 999 (The greater the received light intensity, the higher the value.)",
    ) == []


def test_rejects_response_time_question_that_drops_mu_n_controller_scope():
    assert _MODULE.missing_query_qualifiers(
        "What response times are selectable for the LR-TB5000(C) laser sensor?",
        (
            "Response time | LR-TB5000(C), LR-TB2000(C): "
            "7 ms/15 ms/30 ms/105 ms/1000 ms selectable"
        ),
        (
            "Response time | LR-TB5000(C), LR-TB2000(C): "
            "7 ms/15 ms/30 ms/105 ms/1000 ms selectable"
        ),
        (
            "Column headers: MU-N11 > MU-N12; "
            "Model: Main unit/expansion unit; MU-N11: Main unit; MU-N12: Expansion unit"
        ),
    ) == ["MU-N controller"]


def test_accepts_response_time_question_with_mu_n_controller_scope():
    assert _MODULE.missing_query_qualifiers(
        "For an MU-N controller connected to LR-TB5000(C), what response times are selectable?",
        (
            "Response time | LR-TB5000(C), LR-TB2000(C): "
            "7 ms/15 ms/30 ms/105 ms/1000 ms selectable"
        ),
        (
            "Response time | LR-TB5000(C), LR-TB2000(C): "
            "7 ms/15 ms/30 ms/105 ms/1000 ms selectable"
        ),
        (
            "Column headers: MU-N11 > MU-N12; "
            "Model: Main unit/expansion unit; MU-N11: Main unit; MU-N12: Expansion unit"
        ),
    ) == []


def test_rejects_duration_question_anchored_to_neighboring_current_row():
    case = {
        **_case(),
        "query": "How long does the WM-P6000 take to charge?",
        "expected_snippet": "WM-P6000: 1 A",
        "expected_terms": ["wm-p6000"],
    }
    chunk = {
        **_chunk(),
        "content": "Charging time; WM-P6000: 6.5 hours Current consumption; WM-P6000: 1 A",
    }

    with pytest.raises(ValueError, match="does not answer.*duration value"):
        _MODULE.verify_and_freeze_cases(
            [case],
            {"chunk-1": chunk},
            tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_rejects_io_range_question_when_evidence_omits_terminals():
    case = {
        **_case(),
        "query": "Which wires correspond to OUT1-4, IN1-2, and IN3-6?",
        "expected_snippet": "Black (OUT1) White (OUT2) Gray (OUT3) Orange (OUT4) Pink (IN1) Yellow (IN2)",
        "expected_terms": ["black", "out1", "in1"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match="does not answer.*IN3, IN4, IN5, IN6"):
        _MODULE.verify_and_freeze_cases(
            [case],
            {"chunk-1": chunk},
            tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_ignores_unrelated_output_label_after_input_answer():
    case = {
        **_case(),
        "query": "What input voltage range does the supply accept?",
        "expected_snippet": "Input conditions | Rated input voltage | 85 to 264 VAC",
        "expected_terms": ["input", "voltage", "85", "264"],
    }
    chunk = {
        **_chunk(),
        "content": (
            "Input conditions | Rated input voltage | 85 to 264 VAC "
            "Output conditions | Rated output voltage | 24 VDC"
        ),
    }

    frozen = _MODULE.verify_and_freeze_cases(
        [case],
        {"chunk-1": chunk},
        tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert frozen[0]["case_id"] == "case-1"


def test_rejects_yes_no_question_anchored_to_unrelated_setting():
    case = {
        **_case(),
        "query": "Does turning off the laser diode affect the ability to perform Universal Change Detection?",
        "expected_snippet": "OFF (oFF): Sets the DSC function to OFF.",
        "expected_terms": ["function"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match="question subject/outcome alignment"):
        _MODULE.verify_and_freeze_cases(
            [case], {"chunk-1": chunk}, tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_rejects_range_contract_without_numeric_answer_terms():
    case = {
        **_case(),
        "query": "What length range do GL-FB models cover for robust floor mounting columns?",
        "expected_snippet": "GL-FB models approximately 1000 to 2400 mm",
        "expected_terms": ["robust", "floor", "mounting"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match=r"expected answer value term.*1000, 2400"):
        _MODULE.verify_and_freeze_cases(
            [case], {"chunk-1": chunk}, tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_rejects_connector_contract_that_omits_connector_identifier():
    case = {
        **_case(),
        "query": "What connector type is used for the sensor-to-controller cable?",
        "expected_snippet": "Sensor-to-controller cable (4-pin M12 connector type)",
        "expected_terms": ["sensor", "cable", "4-pin"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match="expected connector term.*m12"):
        _MODULE.verify_and_freeze_cases(
            [case], {"chunk-1": chunk}, tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_rejects_extension_cable_question_anchored_to_header_only_chunk():
    case = {
        **_case(),
        "query": "What extension cable should be used with the CA-CF3 camera cable?",
        "expected_snippet": (
            "Cable type | Connector shape | Camera cable length | "
            "Extension cable | Repeater cable"
        ),
        "expected_terms": ["cable", "connector", "shape", "camera"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match="cable identifier"):
        _MODULE.verify_and_freeze_cases(
            [case], {"chunk-1": chunk}, tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_accepts_extension_cable_contract_with_answer_models():
    case = {
        **_case(),
        "query": "What extension cable should be used with the CA-CF3 camera cable?",
        "expected_snippet": "CA-CF3: Use CA-CF5E or CA-CF10E as the extension cable.",
        "expected_terms": ["ca-cf5e", "ca-cf10e"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    frozen = _MODULE.verify_and_freeze_cases(
        [case], {"chunk-1": chunk}, tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert frozen[0]["expected_terms"] == ["ca-cf5e", "ca-cf10e"]


def test_rejects_command_question_without_command_identifier():
    case = {
        **_case(),
        "query": "Which command should I use to save the current program settings?",
        "expected_snippet": "Command details: the current program",
        "expected_terms": ["command", "current", "program"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match="command identifier"):
        _MODULE.verify_and_freeze_cases(
            [case], {"chunk-1": chunk}, tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_rejects_interface_question_without_answer_interface_term():
    case = {
        **_case(),
        "query": "Which interfaces connect directly to SZ-V Series scanners?",
        "expected_snippet": "Connect through either USB or Ethernet.",
        "expected_terms": ["connect", "series"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match="expected interface term"):
        _MODULE.verify_and_freeze_cases(
            [case], {"chunk-1": chunk}, tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_rejects_named_filter_question_anchored_to_effect_only():
    case = {
        **_case(),
        "query": "Which filter removes abnormal spike-like noise height values?",
        "expected_snippet": "Eliminates abnormal, spike-like noise height values.",
        "expected_terms": ["eliminates", "abnormal", "spike", "noise"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    with pytest.raises(ValueError, match="filter identifier"):
        _MODULE.verify_and_freeze_cases(
            [case], {"chunk-1": chunk}, tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_accepts_named_filter_question_with_filter_identifier():
    case = {
        **_case(),
        "query": "Which filter removes abnormal spike-like noise height values?",
        "expected_snippet": "Median filter: eliminates abnormal spike-like noise height values.",
        "expected_terms": ["median filter", "spike", "noise"],
    }
    chunk = {**_chunk(), "content": case["expected_snippet"]}

    frozen = _MODULE.verify_and_freeze_cases(
        [case], {"chunk-1": chunk}, tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert frozen[0]["case_id"] == "case-1"


def test_accepts_complete_quantitative_and_connector_contracts():
    quantitative = {
        **_case(),
        "query": "What is the Z range tolerance for model XT-024?",
        "expected_snippet": 'XT-024: ±2 mm ±0.08"',
        "expected_terms": ["xt-024", "2 mm", "0.08"],
    }
    connector = {
        **_case(),
        "case_id": "case-2",
        "source_chunk_id": "chunk-2",
        "query": "What connector type is used for the sensor-to-controller cable?",
        "expected_snippet": "Sensor-to-controller cable (4-pin M12 connector type)",
        "expected_terms": ["4-pin", "m12"],
    }
    chunks = {
        "chunk-1": {**_chunk(), "content": quantitative["expected_snippet"]},
        "chunk-2": {**_chunk(), "id": "chunk-2", "content": connector["expected_snippet"]},
    }

    frozen = _MODULE.verify_and_freeze_cases(
        [quantitative, connector], chunks, tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert len(frozen) == 2


def test_normalizes_and_requires_part_number_answer_contract():
    query = "What part number applies to the infrared polarized filter attachment for the IV2-H1?"
    snippet = "Infrared polarized filter attachment OP: 87437"

    assert _MODULE._answer_part_numbers(snippet) == ["OP: 87437"]
    assert _MODULE.enrich_expected_answer_terms(query, snippet, ["infrared polarized filter"]) == [
        "infrared polarized filter",
        "OP: 87437",
    ]
    assert _MODULE.missing_expected_answer_contract(query, snippet, ["OP: 87437"]) == []
    assert _MODULE.missing_expected_answer_contract(query, snippet, ["infrared polarized filter"]) == [
        "answer-specific expected term",
        "expected part number term(s) OP: 87437",
    ]


def test_does_not_treat_voltage_in_safety_risk_condition_as_requested_value():
    assert _MODULE.missing_expected_answer_contract(
        "What safety risks occur if I use a voltage other than 24 VDC?",
        "This may cause fire, electric shock, or equipment failure.",
        ["fire", "electric shock", "equipment failure"],
    ) == []


def test_rejects_contract_whose_expected_terms_only_repeat_the_question():
    missing = _MODULE.missing_expected_answer_contract(
        "Which wire color corresponds to connector pin B1?",
        "Connector pin number: B1; Wire color: Orange",
        ["connector", "pin", "b1", "wire", "color"],
    )

    assert missing == ["answer-specific expected term"]


def test_extracts_unitless_range_and_noun_bound_how_many_values():
    assert _MODULE._answer_quantity_values(
        "What is the display range for received light intensity?",
        "Display range: 0 to 999",
    ) == ["0", "999"]
    assert _MODULE._answer_quantity_values(
        "How many area cameras can be connected?",
        "A maximum of 4 cameras can be connected across 2 units.",
    ) == ["4"]
    assert _MODULE._answer_quantity_values(
        "How many protection zones does the SZ-V04 multi-function model support?",
        "Protection zone | 2 zones | 1 zone | 1 zone | 2 zones",
    ) == ["2"]


def test_how_many_rejects_unrelated_speed_values():
    assert _MODULE._answer_quantity_values(
        "How many head input units are compatible?",
        "Maximum speed is 16 kHz (63 us).",
    ) == []


def test_verifies_every_multi_step_evidence_chunk():
    case = {
        **_case(),
        "query": "What caused error E101 and how should I correct it?",
        "expected_snippet": "Cause: cable disconnected | Corrective action: reconnect cable",
        "expected_terms": ["cable", "reconnect"],
        "expected_evidence": [
            {
                "chunk_id": "chunk-cause",
                "source_document_id": "heldout-doc",
                "snippet": "Cause: cable disconnected",
                "expected_terms": ["cable", "disconnected"],
            },
            {
                "chunk_id": "chunk-action",
                "source_document_id": "heldout-doc",
                "snippet": "Corrective action: reconnect cable",
                "expected_terms": ["reconnect", "cable"],
            },
        ],
    }
    chunks = {
        "chunk-1": _chunk(),
        "chunk-cause": {
            **_chunk(),
            "id": "chunk-cause",
            "content": "Cause: cable disconnected",
        },
        "chunk-action": {
            **_chunk(),
            "id": "chunk-action",
            "content": "Corrective action: reconnect cable",
        },
    }

    frozen = _MODULE.verify_and_freeze_cases(
        [case],
        chunks,
        tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert set(frozen[0]["adjudication"]["evidence_chunk_sha256"]) == {
        "chunk-cause",
        "chunk-action",
    }


def test_rejects_unpersisted_multi_step_evidence():
    case = {
        **_case(),
        "query": "What caused error E101?",
        "expected_evidence": [
            {
                "chunk_id": "missing-chunk",
                "source_document_id": "heldout-doc",
                "snippet": "Cause: cable disconnected",
                "expected_terms": ["cable"],
            }
        ],
    }

    with pytest.raises(ValueError, match="expected evidence chunk is missing"):
        _MODULE.verify_and_freeze_cases(
            [case],
            {"chunk-1": _chunk()},
            tuning_document_ids=set(),
            verified_at="2026-09-23T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_document_id": "tuning-doc"}, "overlap"),
        ({"expected_snippet": "not in source"}, "not present"),
    ],
)
def test_rejects_overlap_or_unverifiable_snippet(mutation, message):
    case = {**_case(), **mutation}
    with pytest.raises(ValueError, match=message):
        _MODULE.verify_and_freeze_cases(
            [case],
            {"chunk-1": _chunk()},
            tuning_document_ids={"tuning-doc"},
            verified_at="2026-09-23T00:00:00+00:00",
        )


def test_partition_verified_cases_keeps_valid_cases_and_records_rejections():
    valid = {
        **_case(),
        "query": "What X-axis reference distance applies to model-7?",
        "expected_snippet": "model-7: 15 mm",
        "expected_terms": ["model-7", "15 mm"],
    }
    ambiguous = {
        **_case(),
        "case_id": "case-2",
        "query": "What reference distance applies to model-7?",
        "expected_snippet": "model-7: 15 mm",
        "expected_terms": ["model-7", "15 mm"],
    }
    chunk = {
        **_chunk(),
        "content": "Model name: X Reference distance; model-7: 15 mm",
    }

    frozen, rejected = _MODULE.partition_verified_cases(
        [valid, ambiguous],
        {"chunk-1": chunk},
        tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert [case["case_id"] for case in frozen] == ["case-1"]
    assert rejected == [
        {
            "case_id": "case-2",
            "query": "What reference distance applies to model-7?",
            "reason": "case-2: query drops source qualifier(s): x-axis",
        }
    ]


def test_partition_verified_cases_rejects_duplicate_case_ids_without_hiding_valid_case():
    duplicate = {**_case(), "query": "Which voltage is required?"}

    frozen, rejected = _MODULE.partition_verified_cases(
        [_case(), duplicate],
        {"chunk-1": _chunk()},
        tuning_document_ids=set(),
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert [case["case_id"] for case in frozen] == ["case-1"]
    assert rejected[0]["reason"] == "missing or duplicate case_id: 'case-1'"
