import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "validate_agentic_retrieval_mrv.py"
_SPEC = importlib.util.spec_from_file_location("validate_agentic_retrieval_mrv", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_case_checks = _MODULE._case_checks


def _payload(*, sufficient: bool = True, supporting_document: str = "doc-a") -> dict:
    return {
        "answer": "The documented value is 280 g.",
        "insufficient_evidence": not sufficient,
        "retrieval_trace": {
            "pipeline": "evidence_map_reduce_verify_v1",
            "stages": [
                "map_claims",
                "retrieve_scoped_branches",
                "verify_claims",
                "reduce_confirmed_evidence",
            ],
            "sufficient": sufficient,
            "evidence_ledger": {
                "claim-a": {
                    "assessment": {
                        "trust_state": "confirmed" if sufficient else "unresolved",
                        "supporting_document_ids": [supporting_document] if sufficient else [],
                        "verification": {
                            "trust_state": "confirmed" if sufficient else "unresolved",
                            "supporting_chunk_ids": ["chunk-a"] if sufficient else [],
                            "invalid_citation_ids": [],
                            "out_of_scope_chunk_ids": [],
                        },
                    }
                }
            },
            "context_assembly": {"all_required_claims_retained": sufficient},
        }
    }


def test_case_checks_accept_confirmed_scoped_evidence():
    failures, summary = _case_checks(
        {
            "expect_sufficient": True,
            "expected_document_ids": ["doc-a"],
            "expected_answer_terms": ["280 g"],
            "min_mapped_claims": 1,
        },
        _payload(),
    )

    assert failures == []
    assert summary["confirmed_chunk_ids"] == ["chunk-a"]
    assert summary["supporting_document_ids"] == ["doc-a"]


def test_case_checks_reject_invalid_and_out_of_scope_citations():
    payload = _payload()
    verification = payload["retrieval_trace"]["evidence_ledger"]["claim-a"]["assessment"]["verification"]
    verification["invalid_citation_ids"] = ["invented"]
    verification["out_of_scope_chunk_ids"] = ["wrong-scope"]

    failures, _summary = _case_checks(
        {"expect_sufficient": True, "expected_document_ids": ["doc-a"]},
        payload,
    )

    assert "verifier_emitted_invalid_citations" in failures
    assert "verifier_emitted_out_of_scope_citations" in failures


def test_case_checks_accept_expected_abstention():
    failures, summary = _case_checks(
        {"expect_sufficient": False, "min_mapped_claims": 1},
        _payload(sufficient=False),
    )

    assert failures == []
    assert summary["trust_states"] == {"claim-a": "unresolved"}


def test_case_checks_reports_missing_answer_terms():
    failures, summary = _case_checks(
        {"expect_sufficient": True, "expected_answer_terms": ["not including lens"]},
        _payload(),
    )

    assert "expected_answer_terms_missing" in failures
    assert summary["missing_answer_terms"] == ["not including lens"]
