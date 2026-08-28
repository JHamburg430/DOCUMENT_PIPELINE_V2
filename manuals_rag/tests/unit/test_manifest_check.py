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
        "latest_contextual_quantity_answer_repair": {"run": "retrieval_eval_20260827_212947"},
        "latest_matrix_retrieval_guardrail_containment": {"status": "addressed"},
        "latest_http_fallback_telemetry_containment": {"status": "addressed"},
        "latest_composite_citation_scoring_containment": {"status": "addressed"},
        "latest_comparison_setting_side_binding_containment": {"status": "addressed"},
        "latest_eval_question_generation_context_scope_review": {"status": "recorded_scope_blocker"},
        "answer_grounding_cross_document_rows_6_7": {"status": "row_6_failed_row_7_clean"},
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


def test_manifest_checker_rejects_missing_root_matrix_retrieval_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["latest_matrix_retrieval_guardrail_containment"]

    errors = module.check_manifest(manifest)

    assert any("latest_matrix_retrieval_guardrail_containment missing required duplicate" in error for error in errors)


def test_manifest_checker_rejects_missing_question_bank_matrix_retrieval_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["question_bank"]["latest_matrix_retrieval_guardrail_containment"]

    errors = module.check_manifest(manifest)

    assert any(
        "question_bank.latest_matrix_retrieval_guardrail_containment" in error
        and "missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_unequal_matrix_retrieval_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["latest_matrix_retrieval_guardrail_containment"] = {"status": "stale"}

    errors = module.check_manifest(manifest)

    assert any("latest_matrix_retrieval_guardrail_containment mismatch" in error for error in errors)


def test_manifest_checker_rejects_missing_root_cross_document_rows_6_7():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["answer_grounding_cross_document_rows_6_7"]

    errors = module.check_manifest(manifest)

    assert any("answer_grounding_cross_document_rows_6_7 missing required duplicate" in error for error in errors)


def test_manifest_checker_rejects_missing_question_bank_cross_document_rows_6_7():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["question_bank"]["answer_grounding_cross_document_rows_6_7"]

    errors = module.check_manifest(manifest)

    assert any(
        "question_bank.answer_grounding_cross_document_rows_6_7" in error
        and "missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_unequal_cross_document_rows_6_7():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["answer_grounding_cross_document_rows_6_7"] = {"status": "stale"}

    errors = module.check_manifest(manifest)

    assert any("answer_grounding_cross_document_rows_6_7 mismatch" in error for error in errors)


def test_manifest_checker_rejects_missing_root_composite_citation_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["latest_composite_citation_scoring_containment"]

    errors = module.check_manifest(manifest)

    assert any("latest_composite_citation_scoring_containment missing required duplicate" in error for error in errors)


def test_manifest_checker_rejects_missing_question_bank_composite_citation_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["question_bank"]["latest_composite_citation_scoring_containment"]

    errors = module.check_manifest(manifest)

    assert any(
        "question_bank.latest_composite_citation_scoring_containment" in error
        and "missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_unequal_composite_citation_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["latest_composite_citation_scoring_containment"] = {"status": "stale"}

    errors = module.check_manifest(manifest)

    assert any("latest_composite_citation_scoring_containment mismatch" in error for error in errors)


def test_manifest_checker_rejects_missing_root_comparison_setting_side_binding_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["latest_comparison_setting_side_binding_containment"]

    errors = module.check_manifest(manifest)

    assert any(
        "latest_comparison_setting_side_binding_containment missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_missing_question_bank_comparison_setting_side_binding_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["question_bank"]["latest_comparison_setting_side_binding_containment"]

    errors = module.check_manifest(manifest)

    assert any(
        "question_bank.latest_comparison_setting_side_binding_containment" in error
        and "missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_unequal_comparison_setting_side_binding_containment():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["latest_comparison_setting_side_binding_containment"] = {
        "status": "stale"
    }

    errors = module.check_manifest(manifest)

    assert any(
        "latest_comparison_setting_side_binding_containment mismatch" in error
        for error in errors
    )


def test_manifest_checker_rejects_missing_root_eval_question_generation_scope_review():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["latest_eval_question_generation_context_scope_review"]

    errors = module.check_manifest(manifest)

    assert any(
        "latest_eval_question_generation_context_scope_review missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_missing_question_bank_eval_question_generation_scope_review():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    del manifest["question_bank"]["latest_eval_question_generation_context_scope_review"]

    errors = module.check_manifest(manifest)

    assert any(
        "question_bank.latest_eval_question_generation_context_scope_review" in error
        and "missing required duplicate" in error
        for error in errors
    )


def test_manifest_checker_rejects_unequal_eval_question_generation_scope_review():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["latest_eval_question_generation_context_scope_review"] = {
        "status": "stale"
    }

    errors = module.check_manifest(manifest)

    assert any(
        "latest_eval_question_generation_context_scope_review mismatch" in error
        for error in errors
    )
