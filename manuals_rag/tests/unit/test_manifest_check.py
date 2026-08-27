import copy
import importlib.util
from pathlib import Path


def _load_manifest_check_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "maintenance"
        / "check_retrieval_accuracy_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("check_retrieval_accuracy_manifest", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_manifest():
    duplicate = {"run": "aa4d2bdc", "classification": "diagnostic"}
    failure_categories = {"expected_terms_missing": 1}
    retrieval_failures = {"candidate_miss": 1}
    paired = {
        "latest_false_negative_repair": duplicate,
        "next_target": "continue source-first evidence selection",
        "answer_grounding_status": {"rows": "diagnostic"},
        "answer_grounding_rotation": {"next": "rows 15-17"},
        "run_exclusions": {"excluded": ["retrieval_eval_20260826_170130"]},
        "unresolved_guardrail_findings": ["source-first citation fidelity"],
        "current_retrieval_failure_source": "retrieval_eval_20260826_005723",
        "latest_cross_document_validation": {"run": "retrieval_eval_20260827_042612"},
        "latest_contextual_row14_repair": {"run": "retrieval_eval_20260827_053144"},
        "partial_claim_citation_pruning_containment": {"status": "addressed_conservative_fallback"},
        "llm_answer_judge_policy": {"status": "diagnostic_only"},
        "failure_categories": retrieval_failures,
        "current_failure_categories": retrieval_failures,
        "current_retrieval_failure_categories": retrieval_failures,
        "answer_current_failure_categories": failure_categories,
        "current_answer_failure_categories": failure_categories,
    }
    return {**copy.deepcopy(paired), "question_bank": copy.deepcopy(paired)}


def test_manifest_checker_accepts_equal_required_pairs():
    module = _load_manifest_check_module()

    errors = module.check_manifest(_minimal_manifest())

    assert errors == []


def test_manifest_checker_rejects_missing_root_false_negative_repair():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["latest_false_negative_repair"]

    errors = module.check_manifest(manifest)

    assert any("latest_false_negative_repair missing required duplicate" in error for error in errors)


def test_manifest_checker_rejects_missing_question_bank_false_negative_repair():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["question_bank"]["latest_false_negative_repair"]

    errors = module.check_manifest(manifest)

    assert any(
        "question_bank.latest_false_negative_repair" in error and "missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_unequal_false_negative_repair():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["latest_false_negative_repair"] = {
        "run": "retrieval_eval_20260826_222558",
        "classification": "clean",
    }

    errors = module.check_manifest(manifest)

    assert any("latest_false_negative_repair mismatch" in error for error in errors)
