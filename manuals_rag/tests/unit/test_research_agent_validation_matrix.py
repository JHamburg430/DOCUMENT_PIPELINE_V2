import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/benchmark/validate_research_agent_matrix.py")
DATASET = Path("tests/fixtures/research_agent_validation_matrix_v1.jsonl")
MANIFEST = Path("tests/fixtures/research_agent_validation_matrix_v1.manifest.json")


def _module():
    spec = importlib.util.spec_from_file_location("validate_research_agent_matrix", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_research_agent_matrix_is_valid_and_complete():
    report = _module().validate_matrix(DATASET, MANIFEST)

    assert report["status"] == "valid"
    assert report["case_count"] == 21
    assert report["layers"] == {"controller_contract": 18, "live_corpus": 3}
    assert report["mandatory_case_count"] == 6


def test_manifest_hash_matches_frozen_dataset_bytes():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == manifest["dataset_sha256"]


def test_duplicate_query_cannot_be_a_terminal_condition():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    duplicate_cases = [row for row in rows if row["category"] == "duplicate_recovery"]

    assert len(duplicate_cases) == 2
    for case in duplicate_cases:
        assert case["expected"]["terminal_state"] == "supported"
        assert "duplicate_query_terminal" in case["expected"]["prohibited_events"]
        assert case["expected"]["min_distinct_rag_queries"] >= 2


def test_live_cases_require_both_backends_and_no_scripted_rag_results():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    live_cases = [row for row in rows if row["layer"] == "live_corpus"]

    assert {case["case_id"] for case in live_cases} == {"ra-v1-019", "ra-v1-020", "ra-v1-021"}
    for case in live_cases:
        assert case["rag_observations"] == []
        assert set(case["live_contract"]["run_backends"]) == {
            "langgraph_agent",
            "llamaindex_agent",
        }


def test_validator_rejects_success_with_no_citations():
    module = _module()
    row = json.loads(DATASET.read_text(encoding="utf-8").splitlines()[0])
    row["expected"]["citation_policy"] = "none"

    with pytest.raises(module.MatrixValidationError, match="supported answer cannot forbid citations"):
        module.validate_case(row)
