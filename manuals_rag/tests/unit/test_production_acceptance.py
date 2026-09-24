import importlib.util
import json
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


def _configuration():
    return {
        "corpus_ids": ["corpus"],
        "backends": ["baseline", "langgraph", "llamaindex"],
        "max_hops": 4,
        "planner": "ollama",
        "ollama_url": "http://ollama",
        "ollama_embed_url": "http://embed",
        "ollama_embed_model": "embed",
        "ollama_fast_model": "fast",
        "ollama_retrieval_verifier_model": "verify",
        "ollama_answer_model": "answer",
        "qdrant_url": "http://qdrant",
        "rerank_model": "reranker",
        "rerank_device": "cpu",
        "result_limit": 10,
        "retrieval_branch_max_workers": 2,
        "retrieval_qdrant_max_concurrency": 2,
        "case_concurrency": 1,
    }


def _artifact(case, *, dataset_hash="hash"):
    item = {
        **case,
        "agent_case_category": "single_hop_control",
        "langgraph": _record(),
        "llamaindex": _record(),
    }
    return {
        "artifact_schema_version": _MODULE.ARTIFACT_SCHEMA_VERSION,
        "run_id": "run-1",
        "complete": True,
        "process_exit_status": 0,
        "dataset_sha256": dataset_hash,
        "case_count": 1,
        "items": [item],
        "provenance": {
            "run_id": "run-1",
            "provenance_complete": True,
            "source": {"revision": "release-commit", "dirty": False},
            "dataset": {
                "sha256": dataset_hash,
                "ordered_case_keys": [f"{dataset_hash}:{case['case_id']}"],
            },
            "configuration": _configuration(),
        },
    }


def test_accepts_complete_disjoint_source_verified_case():
    case = _case()
    artifact = _artifact(case)

    report = _MODULE.evaluate(
        artifact,
        [case],
        dataset_sha256="hash",
        tuning_document_ids={"different-document"},
        min_cases=1,
        max_p95_latency_ms=1000,
        expected_source_revision="release-commit",
    )

    assert report["accepted"] is True


def test_rejects_tuned_unadjudicated_and_failed_cases():
    case = _case()
    case.pop("evaluation_split")
    case.pop("adjudication")
    failed = _record()
    failed["agent_evaluation"]["cells"]["grounded_answer"]["status"] = "fail"
    artifact = _artifact(case)
    artifact["items"][0]["langgraph"] = failed
    artifact["items"][0]["llamaindex"] = failed

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


def test_rejects_malformed_or_unverifiable_launch_contract():
    case = _case()
    artifact = _artifact(case)
    artifact["artifact_schema_version"] = "unknown"
    artifact["run_id"] = "different-run"
    artifact["provenance"]["dataset"]["ordered_case_keys"] = ["synthetic-pass"]
    artifact["provenance"]["configuration"].pop("ollama_answer_model")

    report = _MODULE.evaluate(
        artifact,
        [case],
        dataset_sha256="hash",
        tuning_document_ids={"different-document"},
        min_cases=1,
        max_p95_latency_ms=1000,
        expected_source_revision="release-commit",
    )

    assert report["accepted"] is False
    assert any("artifact_schema_version" in blocker for blocker in report["blockers"])
    assert any("run_id" in blocker for blocker in report["blockers"])
    assert any("ordered case keys" in blocker for blocker in report["blockers"])
    assert any("configuration" in blocker for blocker in report["blockers"])


def test_rejects_nested_tuning_document_overlap_and_stale_revision():
    case = _case()
    case["expected_evidence_graph"]["nodes"] = [
        {"expected_document_ids": ["nested-tuning-document"]}
    ]
    artifact = _artifact(case)

    report = _MODULE.evaluate(
        artifact,
        [case],
        dataset_sha256="hash",
        tuning_document_ids={"nested-tuning-document"},
        min_cases=1,
        max_p95_latency_ms=1000,
        expected_source_revision="newer-release",
    )

    assert report["accepted"] is False
    assert any("document overlap" in blocker for blocker in report["blockers"])
    assert any("release revision" in blocker for blocker in report["blockers"])


def test_rejects_duplicate_dataset_case_ids():
    case = _case()
    artifact = _artifact(case)
    artifact["case_count"] = 2
    artifact["items"] = [artifact["items"][0], artifact["items"][0]]
    artifact["provenance"]["dataset"]["ordered_case_keys"] *= 2

    report = _MODULE.evaluate(
        artifact,
        [case, case],
        dataset_sha256="hash",
        tuning_document_ids={"different-document"},
        min_cases=1,
        max_p95_latency_ms=1000,
        expected_source_revision="release-commit",
    )

    assert report["accepted"] is False
    assert any("duplicate case_id" in blocker for blocker in report["blockers"])


def test_cli_writes_blocked_report_for_malformed_real_schema_input(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "acceptance.json"
    artifact.write_text("{malformed", encoding="utf-8")
    dataset.write_text(json.dumps(_case()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_production_acceptance.py",
            "--artifact",
            str(artifact),
            "--dataset",
            str(dataset),
            "--expected-source-revision",
            "release-commit",
            "--output",
            str(output),
        ],
    )

    assert _MODULE.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["accepted"] is False
    assert report["blockers"][0].startswith("unverifiable input: JSONDecodeError:")
