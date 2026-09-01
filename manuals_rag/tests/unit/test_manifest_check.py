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
    run_exclusions = [
        {"run_id": f"retrieval_eval_20260830_{index:06d}", "reason": "diagnostic"}
        for index in range(126)
    ]
    paired = {
        "latest_false_negative_repair": duplicate,
        "updated_at": "2026-08-29T13:23:00Z",
        "next_target": "continue source-first evidence selection",
        "remaining": "continue source-first evidence selection",
        "answer_grounding_status": {
            "remaining": "continue source-first evidence selection",
            "rows": "diagnostic",
        },
        "answer_grounding_rotation": {
            "latest_run": "retrieval_eval_20260830_201603",
            "dataset": "test_reports/retrieval_eval_dataset_20260830_201603.jsonl",
            "results": "test_reports/retrieval_eval_results_20260830_201603.jsonl",
            "summary": "test_reports/retrieval_eval_summary_20260830_201603.json",
            "manifest": "test_reports/retrieval_eval_manifest_20260830_201603.json",
            "next": "rows 15-17",
        },
        "run_exclusions": run_exclusions,
        "unresolved_guardrail_findings": ["source-first citation fidelity"],
        "current_retrieval_failure_source": (
            "retrieval_eval_20260830_201603 is the current retrieval failure source."
        ),
        "latest_cross_document_validation": {"run": "retrieval_eval_20260827_042612"},
        "latest_cross_document_probe": {"status": "preserved"},
        "latest_contextual_row14_repair": {"run": "retrieval_eval_20260827_053144"},
        "latest_contextual_quantity_answer_repair": {"run": "retrieval_eval_20260827_212947"},
        "latest_matrix_retrieval_guardrail_containment": {"status": "addressed"},
        "latest_http_fallback_telemetry_containment": {"status": "addressed"},
        "latest_composite_citation_scoring_containment": {"status": "addressed"},
        "latest_comparison_setting_side_binding_containment": {"status": "addressed"},
        "latest_eval_question_generation_context_scope_review": {"status": "recorded_scope_blocker"},
        "latest_cross_document_answer_rotation": {"status": "row_10_diagnostic"},
        "latest_cross_document_row8_answer_repair": {"status": "addressed"},
        "latest_cross_document_row8_answer_partial_side_containment": {"status": "addressed"},
        "latest_contextual_procedure_rows_3_4_source_review": {
            "status": "diagnostic_source_equivalence"
        },
        "latest_contextual_procedure_rows_5_6_answer_evidence_failure": {
            "status": "current_expected_evidence_not_cited"
        },
        "latest_multi_step_expected_context_scoring_containment": {
            "status": "role_mixing_fail_closed"
        },
        "answer_grounding_cross_document_rows_6_7": {"status": "row_6_failed_row_7_clean"},
        "answer_grounding_contextual_rows_15_16": {"status": "diagnostic"},
        "answer_grounding_contextual_rows_17_18": {"status": "diagnostic"},
        "answer_grounding_contextual_rows_19_20": {"status": "diagnostic"},
        "answer_grounding_sibling_rows_15_16": {"status": "diagnostic"},
        "answer_grounding_single_step_v2_rows_5_6": {"status": "diagnostic"},
        "answer_grounding_single_step_v2_rows_7_8": {"status": "diagnostic"},
        "answer_grounding_single_step_v2_rows_9_10": {"status": "diagnostic"},
        "latest_status_output_row2_diagnostic_experiment": {
            "status": "not_clean_no_production_change_accepted"
        },
        "latest_temp_worktree_hygiene_guardrail": {
            "status": "needs_fix_recorded",
            "worktree_policy": "reuse_one_clean_job_owned_worktree",
        },
        "partial_claim_citation_pruning_containment": {"status": "addressed_conservative_fallback"},
        "llm_answer_judge_policy": {"status": "diagnostic_only"},
        "manifest_integrity_repairs": [{"status": "current"}],
        "failure_categories": retrieval_failures,
        "current_failure_categories": retrieval_failures,
        "current_retrieval_failure_categories": retrieval_failures,
        "retrieval_current_failure_categories": retrieval_failures,
        "answer_current_failure_categories": failure_categories,
        "current_answer_failure_categories": failure_categories,
    }
    return {
        **copy.deepcopy(paired),
        "quality_policy": {"active_counts_must_not_shrink": True},
        "question_bank": {
            **copy.deepcopy(paired),
            "total_questions": 208,
            "single_step_questions": 101,
            "multi_step_questions": 107,
            "exploratory_questions": 208,
            "locked_regression_questions": 0,
        },
    }


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


def test_manifest_checker_rejects_missing_or_unequal_remaining():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["remaining"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["remaining"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["remaining"] = "stale target"

    assert any(
        "remaining missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.remaining" in error and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any("remaining mismatch" in error for error in module.check_manifest(unequal))


def test_manifest_checker_rejects_question_bank_count_reset():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["total_questions"] = 0
    manifest["question_bank"]["single_step_questions"] = 0
    manifest["question_bank"]["multi_step_questions"] = 0
    manifest["question_bank"]["exploratory_questions"] = 0

    errors = module.check_manifest(manifest)

    assert any("question_bank.total_questions dropped below monotonic floor" in error for error in errors)
    assert any("question_bank.single_step_questions dropped below monotonic floor" in error for error in errors)
    assert any("question_bank.multi_step_questions dropped below monotonic floor" in error for error in errors)
    assert any("question_bank.exploratory_questions dropped below monotonic floor" in error for error in errors)


def test_manifest_checker_rejects_run_exclusion_reset():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["run_exclusions"] = []
    manifest["question_bank"]["run_exclusions"] = []

    errors = module.check_manifest(manifest)

    assert any("run_exclusions dropped below monotonic floor" in error for error in errors)
    assert any(
        "question_bank.run_exclusions dropped below monotonic floor" in error
        for error in errors
    )


def test_manifest_checker_rejects_answer_grounding_rotation_artifact_run_mismatch():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["answer_grounding_rotation"][
        "dataset"
    ] = "test_reports/retrieval_eval_dataset_20260830_194345.jsonl"
    manifest["question_bank"]["answer_grounding_rotation"][
        "dataset"
    ] = "test_reports/retrieval_eval_dataset_20260830_194345.jsonl"

    errors = module.check_manifest(manifest)

    assert any(
        "answer_grounding_rotation.dataset run id mismatch" in error for error in errors
    )
    assert any(
        "question_bank.answer_grounding_rotation.dataset run id mismatch" in error
        for error in errors
    )


def test_manifest_checker_rejects_nested_answer_grounding_rotation_artifact_run_mismatch():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["question_bank"]["answer_grounding_rotation"][
        "results"
    ] = "test_reports/retrieval_eval_results_20260830_194345.jsonl"

    errors = module.check_manifest(manifest)

    assert any("answer_grounding_rotation mismatch" in error for error in errors)
    assert any(
        "question_bank.answer_grounding_rotation.results run id mismatch" in error
        for error in errors
    )


def test_manifest_checker_rejects_missing_accepted_clean_run_artifacts(tmp_path):
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    accepted = [
        "retrieval_eval_20260830_201603 row 1",
        "retrieval_eval_20260901_131404 cross-document row 2",
    ]
    manifest["answer_grounding_status"]["accepted_clean_runs"] = accepted
    manifest["question_bank"]["answer_grounding_status"]["accepted_clean_runs"] = accepted
    reports = tmp_path / "test_reports"
    reports.mkdir()
    for suffix in ("20260830_201603", "20260901_131404"):
        (reports / f"retrieval_eval_dataset_{suffix}.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
        (reports / f"retrieval_eval_results_{suffix}.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )

    assert module.check_manifest(manifest, tmp_path) == []

    (reports / "retrieval_eval_results_20260901_131404.jsonl").unlink()
    errors = module.check_manifest(manifest, tmp_path)

    assert any(
        "answer_grounding_status.accepted_clean_runs references missing results artifact"
        in error
        and "retrieval_eval_20260901_131404" in error
        for error in errors
    )
    assert any(
        "question_bank.answer_grounding_status.accepted_clean_runs references missing results artifact"
        in error
        and "retrieval_eval_20260901_131404" in error
        for error in errors
    )


def test_manifest_checker_rejects_stale_current_retrieval_failure_source():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["current_retrieval_failure_source"] = (
        "retrieval_eval_20260827_074631 is historical cross-document evidence."
    )
    manifest["question_bank"]["current_retrieval_failure_source"] = (
        "retrieval_eval_20260827_074631 is historical cross-document evidence."
    )

    errors = module.check_manifest(manifest)

    assert any("root.current_retrieval_failure_source stale" in error for error in errors)
    assert any(
        "question_bank.current_retrieval_failure_source stale" in error for error in errors
    )


def test_manifest_checker_allows_historical_source_when_no_current_retrieval_failures():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    for key in (
        "failure_categories",
        "current_failure_categories",
        "current_retrieval_failure_categories",
        "retrieval_current_failure_categories",
    ):
        manifest[key] = {}
        manifest["question_bank"][key] = {}
    manifest["current_retrieval_failure_source"] = (
        "retrieval_eval_20260827_074631 is historical cross-document evidence."
    )
    manifest["question_bank"]["current_retrieval_failure_source"] = (
        "retrieval_eval_20260827_074631 is historical cross-document evidence."
    )

    errors = module.check_manifest(manifest)

    assert errors == []


def test_manifest_checker_rejects_current_source_when_retrieval_failures_empty():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    for key in (
        "failure_categories",
        "current_failure_categories",
        "current_retrieval_failure_categories",
        "retrieval_current_failure_categories",
    ):
        manifest[key] = {}
        manifest["question_bank"][key] = {}

    errors = module.check_manifest(manifest)

    assert any(
        "root.current_retrieval_failure_source stale" in error for error in errors
    )
    assert any(
        "question_bank.current_retrieval_failure_source stale" in error
        for error in errors
    )


def test_manifest_checker_rejects_stale_remaining_row_target():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["next_target"] = manifest["question_bank"][
        "next_target"
    ] = "Continue with contextual procedure row 7 answer evidence selection."
    manifest["remaining"] = manifest["question_bank"][
        "remaining"
    ] = "Fix contextual procedure row 5 Multi-Capture answer evidence selection."

    errors = module.check_manifest(manifest)

    assert any("current_target_alias mismatch" in error for error in errors)


def test_manifest_checker_rejects_stale_remaining_text_even_when_rows_match():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    current_target = (
        "Fix row 9 answer/citation grounding after retrieval_eval_20260831_123253 "
        "retrieved PID 428 evidence."
    )
    stale_remaining = (
        "Fix row 9 production retrieval/context selection after retrieval_eval_20260831_120437 "
        "missed PID 428 evidence."
    )
    manifest["next_target"] = manifest["question_bank"]["next_target"] = current_target
    manifest["answer_grounding_status"]["remaining"] = current_target
    manifest["question_bank"]["answer_grounding_status"]["remaining"] = current_target
    manifest["remaining"] = manifest["question_bank"]["remaining"] = stale_remaining

    errors = module.check_manifest(manifest)

    assert any("current_target_alias mismatch" in error for error in errors)


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


def test_manifest_checker_rejects_missing_or_unequal_cross_document_answer_rotation():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["latest_cross_document_answer_rotation"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["latest_cross_document_answer_rotation"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["latest_cross_document_answer_rotation"] = {"status": "stale"}

    assert any(
        "latest_cross_document_answer_rotation missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.latest_cross_document_answer_rotation" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "latest_cross_document_answer_rotation mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_missing_or_unequal_row8_answer_repair():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["latest_cross_document_row8_answer_repair"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["latest_cross_document_row8_answer_repair"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["latest_cross_document_row8_answer_repair"] = {"status": "stale"}

    assert any(
        "latest_cross_document_row8_answer_repair missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.latest_cross_document_row8_answer_repair" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "latest_cross_document_row8_answer_repair mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_missing_or_unequal_row8_partial_side_containment():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["latest_cross_document_row8_answer_partial_side_containment"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["latest_cross_document_row8_answer_partial_side_containment"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["latest_cross_document_row8_answer_partial_side_containment"] = {
        "status": "stale"
    }

    assert any(
        "latest_cross_document_row8_answer_partial_side_containment missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.latest_cross_document_row8_answer_partial_side_containment" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "latest_cross_document_row8_answer_partial_side_containment mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_missing_or_unequal_temp_worktree_hygiene_guardrail():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["latest_temp_worktree_hygiene_guardrail"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["latest_temp_worktree_hygiene_guardrail"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["latest_temp_worktree_hygiene_guardrail"] = {
        "status": "stale"
    }

    assert any(
        "latest_temp_worktree_hygiene_guardrail missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.latest_temp_worktree_hygiene_guardrail" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "latest_temp_worktree_hygiene_guardrail mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_missing_or_unequal_contextual_procedure_source_review():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["latest_contextual_procedure_rows_3_4_source_review"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["latest_contextual_procedure_rows_3_4_source_review"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["latest_contextual_procedure_rows_3_4_source_review"] = {
        "status": "stale"
    }

    assert any(
        "latest_contextual_procedure_rows_3_4_source_review missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.latest_contextual_procedure_rows_3_4_source_review" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "latest_contextual_procedure_rows_3_4_source_review mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_missing_or_unequal_contextual_rows_5_6_answer_failure():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["latest_contextual_procedure_rows_5_6_answer_evidence_failure"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["latest_contextual_procedure_rows_5_6_answer_evidence_failure"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["latest_contextual_procedure_rows_5_6_answer_evidence_failure"] = {
        "status": "stale"
    }

    assert any(
        "latest_contextual_procedure_rows_5_6_answer_evidence_failure missing required duplicate"
        in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.latest_contextual_procedure_rows_5_6_answer_evidence_failure" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "latest_contextual_procedure_rows_5_6_answer_evidence_failure mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_missing_or_unequal_multi_step_expected_context_containment():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["latest_multi_step_expected_context_scoring_containment"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["latest_multi_step_expected_context_scoring_containment"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["latest_multi_step_expected_context_scoring_containment"] = {
        "status": "stale"
    }

    assert any(
        "latest_multi_step_expected_context_scoring_containment missing required duplicate"
        in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.latest_multi_step_expected_context_scoring_containment" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "latest_multi_step_expected_context_scoring_containment mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_missing_or_unequal_late_current_state_pairs():
    module = _load_manifest_check_module()
    fields = [
        "latest_cross_document_probe",
        "answer_grounding_contextual_rows_15_16",
        "answer_grounding_contextual_rows_17_18",
        "answer_grounding_contextual_rows_19_20",
        "answer_grounding_sibling_rows_15_16",
        "answer_grounding_single_step_v2_rows_5_6",
        "answer_grounding_single_step_v2_rows_7_8",
        "answer_grounding_single_step_v2_rows_9_10",
        "latest_status_output_row2_diagnostic_experiment",
        "updated_at",
    ]

    for field in fields:
        missing_root = _minimal_manifest()
        del missing_root[field]
        missing_nested = _minimal_manifest()
        del missing_nested["question_bank"][field]
        unequal = _minimal_manifest()
        unequal["question_bank"][field] = {"status": "stale"}

        assert any(
            f"{field} missing required duplicate" in error
            for error in module.check_manifest(missing_root)
        )
        assert any(
            f"question_bank.{field}" in error and "missing required duplicate" in error
            for error in module.check_manifest(missing_nested)
        )
        assert any(f"{field} mismatch" in error for error in module.check_manifest(unequal))


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


def test_manifest_checker_rejects_retrieval_failure_semantic_alias_mismatch():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["retrieval_current_failure_categories"] = {"candidate_miss": 99}

    errors = module.check_manifest(manifest)

    assert any("retrieval_failure_categories mismatch" in error for error in errors)
