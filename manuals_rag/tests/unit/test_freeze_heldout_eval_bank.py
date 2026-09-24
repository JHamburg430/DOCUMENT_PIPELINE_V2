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
    assert frozen[0]["expected_terms"] == ["model-7"]


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


def test_rejects_standalone_question_with_deictic_product_subject():
    assert _MODULE.missing_query_qualifiers(
        "What shutter speed range can I set on this camera?",
        "Electronic shutter | Can be set to 0.022 to 1000 msec",
        "Electronic shutter | Can be set to 0.022 to 1000 msec",
    ) == ["explicit subject"]


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
        "expected_terms": ["input", "voltage"],
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
