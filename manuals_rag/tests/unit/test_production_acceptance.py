import importlib.util
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "check_production_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("check_production_acceptance", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _case():
    return {
        "case_id": "case-1",
        "source_document_id": "held-out-document",
        "evaluation_split": "held_out",
        "adjudication": {"status": "source_verified"},
        "expected_evidence_graph": {"expected_outcome": "answerable"},
    }


def _record():
    cells = {name: {"status": "pass", "metrics": {}} for name in _MODULE.REQUIRED_CELLS}
    cells["grounded_answer"]["metrics"] = {
        "claim_grounding": {"claim-1": {"terms": True, "citation": True}},
        "invalid_citation_chunks": [],
        "insufficient_evidence": False,
    }
    return {"elapsed_ms": 100.0, "agent_evaluation": {"cells": cells}}


def test_accepts_complete_disjoint_source_verified_case():
    case = _case()
    item = {**case, "agent_case_category": "single_hop_control", "langgraph": _record(), "llamaindex": _record()}
    artifact = {
        "complete": True,
        "process_exit_status": 0,
        "dataset_sha256": "hash",
        "items": [item],
        "provenance": {"provenance_complete": True, "source": {"dirty": False}},
    }

    report = _MODULE.evaluate(
        artifact,
        [case],
        dataset_sha256="hash",
        tuning_document_ids={"different-document"},
        min_cases=1,
        max_p95_latency_ms=1000,
    )

    assert report["accepted"] is True


def test_rejects_tuned_unadjudicated_and_failed_cases():
    case = _case()
    case.pop("evaluation_split")
    case.pop("adjudication")
    failed = _record()
    failed["agent_evaluation"]["cells"]["grounded_answer"]["status"] = "fail"
    item = {**case, "agent_case_category": "single_hop_control", "langgraph": failed, "llamaindex": failed}
    artifact = {
        "complete": True,
        "process_exit_status": 0,
        "dataset_sha256": "hash",
        "items": [item],
        "provenance": {"provenance_complete": True, "source": {"dirty": False}},
    }

    report = _MODULE.evaluate(
        artifact,
        [case],
        dataset_sha256="hash",
        tuning_document_ids={"held-out-document"},
        min_cases=1,
        max_p95_latency_ms=1000,
    )

    assert report["accepted"] is False
    assert any("evaluation_split=held_out" in blocker for blocker in report["blockers"])
    assert any("document overlap" in blocker for blocker in report["blockers"])
    assert any("required cells" in blocker for blocker in report["blockers"])
