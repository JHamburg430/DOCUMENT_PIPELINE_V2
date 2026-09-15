from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "build_agent_eval_bank.py"
_SPEC = importlib.util.spec_from_file_location("build_agent_eval_bank", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _case(*, sibling_header: str) -> dict:
    return {
        "generation_method": "table_sibling_error_cause_action",
        "expected_evidence": [
            {
                "field": "symptom",
                "snippet": (
                    "Column headers: Symptom; Cell value: Image update is slow. "
                    "(during operation/settings); Row: 2; Column: 0"
                ),
            },
            {
                "field": "cause",
                "snippet": (
                    f"Column headers: Cause; Row headers: {sibling_header}; "
                    "Cell value: example; Row: 2; Column: 1"
                ),
            },
        ],
    }


def test_bank_builder_accepts_sibling_bound_to_anchor_row():
    assert _MODULE._has_bound_sibling_evidence(
        _case(sibling_header="Image update is slow. (during operation/settings)")
    )


def test_bank_builder_rejects_same_row_number_from_unrelated_table():
    assert not _MODULE._has_bound_sibling_evidence(
        _case(sibling_header="External master image registration failed")
    )


def test_bank_builder_reclassifies_contextual_direct_question_as_single_hop():
    case = _MODULE._categorized(
        {
            "generation_method": "contextual_procedure_plus_section_evidence",
            "retrieval_task": "multi_step_retrieval",
            "expected_evidence": [
                {"chunk_id": "heading", "expected_terms": ["procedure"]},
                {"chunk_id": "rule", "expected_terms": ["flowchart"]},
            ],
        },
        "dependent_multi_hop",
        "legacy.jsonl",
    )

    assert case["retrieval_task"] == "retrieval"
    assert case["expected_evidence_graph"]["category"] == "single_hop_control"
    assert case["expected_evidence_graph"]["mode"] == "single"


def test_bank_builder_preserves_contextual_capture_mode_in_query():
    case = _MODULE._categorized(
        {
            "query": "For XG-X line-scan camera setup, which settings are used?",
            "generation_method": "contextual_procedure_plus_section_evidence",
            "source_metadata": {
                "local_rerank_context": "Capture Using Line Scan Cameras (Standard Lighting Mode)"
            },
            "expected_evidence": [
                {"chunk_id": "heading", "expected_terms": ["standard", "lighting"]},
                {"chunk_id": "settings", "expected_terms": ["camera", "trigger"]},
            ],
        },
        "dependent_multi_hop",
        "legacy.jsonl",
    )

    assert case["query"] == (
        "In Standard Lighting Mode, for XG-X line-scan camera setup, which settings are used?"
    )


def test_bank_builder_rejects_contextual_table_from_adjacent_mode():
    case = {
        "generation_method": "contextual_procedure_plus_section_evidence",
        "expected_evidence": [
            {
                "snippet": (
                    "Procedure step 2: Typical operations at trigger input "
                    "(Capture Type: Multi-Capture)"
                )
            },
            {
                "snippet": (
                    "Column headers: Image buffer (once); Row headers: Capture options; "
                    "Cell value: Image buffer (once); Row: 0; Column: 1"
                )
            },
        ],
    }

    assert not _MODULE._has_bound_sibling_evidence(case)


def test_bank_builder_accepts_contextual_table_bound_to_named_mode():
    case = {
        "generation_method": "contextual_procedure_plus_section_evidence",
        "expected_evidence": [
            {
                "snippet": (
                    "Procedure step 2: Typical operations at trigger input "
                    "(Capture Type: Multi-Capture)"
                )
            },
            {
                "snippet": (
                    "Column headers: Multi-Capture; Row headers: Trigger Mode; "
                    "Cell value: External trigger; Row: 3; Column: 1"
                )
            },
        ],
    }

    assert _MODULE._has_bound_sibling_evidence(case)
