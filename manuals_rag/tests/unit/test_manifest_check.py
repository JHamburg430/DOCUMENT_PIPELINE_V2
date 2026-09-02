import copy
import importlib.util
import json
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
            "runtime_load_evidence": {
                "validated_run": "retrieval_eval_20260830_201603",
                "api_url": "http://127.0.0.1:9193",
                "container": "manuals-rag-test-api",
                "health": "ok at 2026-08-30T20:15:00Z",
                "fingerprints": {"generator.py": "abc123"},
            },
            "manual_jsonl_inspection": (
                "Inspected retrieval_eval_20260830_201603 user-visible answer and citations."
            ),
            "scope": "retrieval_eval_20260830_201603 tracking validation only",
            "next": "rows 15-17",
        },
        "run_exclusions": run_exclusions,
        "unresolved_guardrail_findings": ["source-first citation fidelity"],
        "current_retrieval_failure_source": (
            "retrieval_eval_20260830_201603 is the current retrieval failure source."
        ),
        "current_answer_failure_source": (
            "retrieval_eval_20260830_201603 is the current answer failure source: "
            "expected_terms_missing."
        ),
        "current_failure_sources": {
            "retrieval": "retrieval_eval_20260830_201603 candidate_miss",
            "answer": "retrieval_eval_20260830_201603 expected_terms_missing",
        },
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
            "datasets": [],
        },
    }


def _active_dataset_entry():
    return {
        "path": "test_reports/new_active_dataset.jsonl",
        "total_questions": 1,
        "single_step_questions": 0,
        "multi_step_questions": 1,
        "status": "exploratory_active_diagnostic_retrieval_failure",
        "generation": "manual_source_reviewed",
        "quality_review": "Source reviewed and preserved as diagnostic evidence.",
        "replacement_status": "new_active_case_no_replacement_debt",
        "results_path": "test_reports/new_active_results.jsonl",
        "summary_path": "test_reports/new_active_summary.json",
        "manifest_path": "test_reports/new_active_manifest.json",
        "active_count_delta": {
            "total_questions": 1,
            "single_step_questions": 0,
            "multi_step_questions": 1,
            "exploratory_questions": 1,
        },
    }


def _row(task="multi_step_retrieval", case_id="case-1"):
    return {"case_id": case_id, "retrieval_task": task, "query": "Source-backed question?"}


def _write_new_active_artifacts(root, row=None):
    reports = root / "test_reports"
    reports.mkdir(exist_ok=True)
    (reports / "new_active_dataset.jsonl").write_text(
        json.dumps(row or _row()) + "\n", encoding="utf-8"
    )
    (reports / "new_active_results.jsonl").write_text("{}\n", encoding="utf-8")
    (reports / "new_active_summary.json").write_text("{}\n", encoding="utf-8")
    (reports / "new_active_manifest.json").write_text("{}\n", encoding="utf-8")


def _registered_dataset_entry(total=1, single=0, multi=1):
    return {
        "path": "test_reports/registered_dataset.jsonl",
        "total_questions": total,
        "single_step_questions": single,
        "multi_step_questions": multi,
        "status": "exploratory_active",
        "generation": "manual_source_reviewed",
        "quality_review": "Source reviewed.",
        "replacement_status": "new_active_case_no_replacement_debt",
    }


def _extension_manifests():
    parent = _minimal_manifest()
    parent["question_bank"]["datasets"].append(_registered_dataset_entry())
    current = copy.deepcopy(parent)
    current["question_bank"]["total_questions"] += 1
    current["question_bank"]["multi_step_questions"] += 1
    current["question_bank"]["exploratory_questions"] += 1
    entry = current["question_bank"]["datasets"][0]
    entry["total_questions"] = 2
    entry["multi_step_questions"] = 2
    entry["active_count_delta"] = {
        "total_questions": 1,
        "single_step_questions": 0,
        "multi_step_questions": 1,
        "exploratory_questions": 1,
    }
    entry["ledger_change"] = {"kind": "extended_registered_dataset"}
    return parent, current


def test_manifest_checker_accepts_equal_required_pairs():
    module = _load_manifest_check_module()

    errors = module.check_manifest(_minimal_manifest())

    assert errors == []


def test_manifest_change_accepts_count_increase_with_registered_dataset(tmp_path):
    module = _load_manifest_check_module()
    parent = _minimal_manifest()
    current = copy.deepcopy(parent)
    current["question_bank"]["total_questions"] += 1
    current["question_bank"]["multi_step_questions"] += 1
    current["question_bank"]["exploratory_questions"] += 1
    current["question_bank"]["datasets"].append(_active_dataset_entry())
    _write_new_active_artifacts(tmp_path)

    assert module.check_manifest_change(current, parent, tmp_path) == []


def test_manifest_change_rejects_count_increase_without_registry_entry():
    module = _load_manifest_check_module()
    parent = _minimal_manifest()
    current = copy.deepcopy(parent)
    current["question_bank"]["total_questions"] += 1
    current["question_bank"]["multi_step_questions"] += 1
    current["question_bank"]["exploratory_questions"] += 1

    errors = module.check_manifest_change(current, parent)

    assert any("increased without a new dataset registry entry" in error for error in errors)


def test_manifest_change_rejects_per_type_ledger_delta_mismatch(tmp_path):
    module = _load_manifest_check_module()
    parent = _minimal_manifest()
    current = copy.deepcopy(parent)
    current["question_bank"]["total_questions"] += 1
    current["question_bank"]["single_step_questions"] += 1
    current["question_bank"]["exploratory_questions"] += 1
    current["question_bank"]["datasets"].append(_active_dataset_entry())
    _write_new_active_artifacts(tmp_path)

    errors = module.check_manifest_change(current, parent, tmp_path)

    assert any("single_step_questions" in error for error in errors)
    assert any("multi_step_questions" in error for error in errors)


def test_manifest_change_accepts_append_only_registered_dataset_extension(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (parent_root / "test_reports").mkdir(parents=True)
    (current_root / "test_reports").mkdir(parents=True)
    first = json.dumps(_row(case_id="old")) + "\n"
    (parent_root / "test_reports/registered_dataset.jsonl").write_text(first, encoding="utf-8")
    (current_root / "test_reports/registered_dataset.jsonl").write_text(
        first + json.dumps(_row(case_id="new")) + "\n", encoding="utf-8"
    )

    assert module.check_manifest_change(
        current, parent, current_root, parent_root
    ) == []


def test_manifest_change_rejects_unchanged_registered_dataset_extension(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    for root in (parent_root, current_root):
        (root / "test_reports").mkdir(parents=True)
        (root / "test_reports/registered_dataset.jsonl").write_text(
            json.dumps(_row(case_id="old")) + "\n", encoding="utf-8"
        )

    errors = module.check_manifest_change(current, parent, current_root, parent_root)

    assert any("must append a nonblank JSONL row" in error for error in errors)


def test_manifest_change_rejects_missing_extension_artifact(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (parent_root / "test_reports").mkdir(parents=True)
    (parent_root / "test_reports/registered_dataset.jsonl").write_text(
        json.dumps(_row(case_id="old")) + "\n", encoding="utf-8"
    )

    errors = module.check_manifest_change(current, parent, current_root, parent_root)

    assert any("current dataset is missing" in error for error in errors)


def test_manifest_change_rejects_missing_parent_extension_artifact(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (current_root / "test_reports").mkdir(parents=True)
    (current_root / "test_reports/registered_dataset.jsonl").write_text(
        json.dumps(_row(case_id="old")) + "\n"
        + json.dumps(_row(case_id="new")) + "\n",
        encoding="utf-8",
    )

    errors = module.check_manifest_change(current, parent, current_root, parent_root)

    assert any("parent dataset is missing" in error for error in errors)


def test_manifest_change_rejects_metadata_only_extension(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    for root in (parent_root, current_root):
        (root / "test_reports").mkdir(parents=True)
        (root / "test_reports/registered_dataset.jsonl").write_text(
            json.dumps(_row(case_id="old")) + "\n", encoding="utf-8"
        )
    current["question_bank"]["datasets"][0]["notes"] = "metadata-only edit"

    errors = module.check_manifest_change(current, parent, current_root, parent_root)

    assert any("must append a nonblank JSONL row" in error for error in errors)


def test_manifest_change_rejects_replaced_rows_masquerading_as_extension(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (parent_root / "test_reports").mkdir(parents=True)
    (current_root / "test_reports").mkdir(parents=True)
    (parent_root / "test_reports/registered_dataset.jsonl").write_text(
        json.dumps(_row(case_id="old")) + "\n", encoding="utf-8"
    )
    (current_root / "test_reports/registered_dataset.jsonl").write_text(
        json.dumps(_row(case_id="rewritten")) + "\n"
        + json.dumps(_row(case_id="new")) + "\n",
        encoding="utf-8",
    )

    errors = module.check_manifest_change(current, parent, current_root, parent_root)

    assert any("must be append-only; parent rows changed" in error for error in errors)


def test_manifest_change_rejects_extension_row_type_mismatch(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (parent_root / "test_reports").mkdir(parents=True)
    (current_root / "test_reports").mkdir(parents=True)
    first = json.dumps(_row(case_id="old")) + "\n"
    (parent_root / "test_reports/registered_dataset.jsonl").write_text(first, encoding="utf-8")
    (current_root / "test_reports/registered_dataset.jsonl").write_text(
        first + json.dumps(_row("single_step_retrieval", "new")) + "\n",
        encoding="utf-8",
    )

    errors = module.check_manifest_change(current, parent, current_root, parent_root)

    assert any("appended single_step_questions mismatch" in error for error in errors)
    assert any("appended multi_step_questions mismatch" in error for error in errors)


def test_manifest_change_rejects_new_active_dataset_missing_provenance(tmp_path):
    module = _load_manifest_check_module()
    parent = _minimal_manifest()
    current = copy.deepcopy(parent)
    current["question_bank"]["total_questions"] += 1
    current["question_bank"]["multi_step_questions"] += 1
    current["question_bank"]["exploratory_questions"] += 1
    entry = _active_dataset_entry()
    del entry["quality_review"]
    del entry["replacement_status"]
    current["question_bank"]["datasets"].append(entry)
    _write_new_active_artifacts(tmp_path)

    errors = module.check_manifest_change(current, parent, tmp_path)

    assert any("requires nonempty valid quality_review" in error for error in errors)
    assert any("requires replacement or supersession semantics" in error for error in errors)


def test_manifest_change_rejects_new_active_dataset_missing_evidence_artifacts(tmp_path):
    module = _load_manifest_check_module()
    parent = _minimal_manifest()
    current = copy.deepcopy(parent)
    current["question_bank"]["total_questions"] += 1
    current["question_bank"]["multi_step_questions"] += 1
    current["question_bank"]["exploratory_questions"] += 1
    current["question_bank"]["datasets"].append(_active_dataset_entry())
    reports = tmp_path / "test_reports"
    reports.mkdir()
    (reports / "new_active_dataset.jsonl").write_text(
        json.dumps(_row()) + "\n", encoding="utf-8"
    )

    errors = module.check_manifest_change(current, parent, tmp_path)

    assert any("references missing results_path" in error for error in errors)
    assert any("references missing summary_path" in error for error in errors)
    assert any("references missing manifest_path" in error for error in errors)


def test_manifest_change_rejects_skeletal_active_entry_masked_by_valid_extension(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (parent_root / "test_reports").mkdir(parents=True)
    (current_root / "test_reports").mkdir(parents=True)
    first = json.dumps(_row(case_id="old")) + "\n"
    (parent_root / "test_reports/registered_dataset.jsonl").write_text(
        first, encoding="utf-8"
    )
    (current_root / "test_reports/registered_dataset.jsonl").write_text(
        first + json.dumps(_row(case_id="new")) + "\n", encoding="utf-8"
    )

    current["question_bank"]["datasets"].append(
        {
            "path": "test_reports/skeletal_active_dataset.jsonl",
            "total_questions": 99,
            "single_step_questions": 99,
            "multi_step_questions": 0,
            "status": "active",
        }
    )

    errors = module.check_manifest_change(
        current, parent, current_root, parent_root
    )

    assert any("requires active_count_delta" in error for error in errors)
    assert any("requires nonempty valid generation" in error for error in errors)
    assert any("requires nonempty valid quality_review" in error for error in errors)
    assert any("requires replacement or supersession semantics" in error for error in errors)
    assert any("new active dataset is missing" in error for error in errors)


def test_manifest_change_rejects_existing_count_rewrite_masked_by_valid_extension(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (parent_root / "test_reports").mkdir(parents=True)
    (current_root / "test_reports").mkdir(parents=True)
    first = json.dumps(_row(case_id="old")) + "\n"
    (parent_root / "test_reports/registered_dataset.jsonl").write_text(
        first, encoding="utf-8"
    )
    (current_root / "test_reports/registered_dataset.jsonl").write_text(
        first + json.dumps(_row(case_id="new")) + "\n", encoding="utf-8"
    )

    unchanged_path = "test_reports/second_registered_dataset.jsonl"
    parent["question_bank"]["datasets"].append(
        {
            **_registered_dataset_entry(),
            "path": unchanged_path,
        }
    )
    current["question_bank"]["datasets"].append(
        {
            **_registered_dataset_entry(total=99, single=99, multi=0),
            "path": unchanged_path,
        }
    )

    errors = module.check_manifest_change(
        current, parent, current_root, parent_root
    )

    assert any(
        "second_registered_dataset.jsonl changed count fields" in error
        and "without active_count_delta" in error
        for error in errors
    )


def test_manifest_change_accepts_registered_dataset_reactivation(tmp_path):
    module = _load_manifest_check_module()
    parent = _minimal_manifest()
    entry = {
        **_active_dataset_entry(),
        "status": "inactive_diagnostic",
        "active_count_delta": None,
        "ledger_change": None,
    }
    parent["question_bank"]["datasets"].append(entry)
    current = copy.deepcopy(parent)
    current["question_bank"]["total_questions"] += 1
    current["question_bank"]["multi_step_questions"] += 1
    current["question_bank"]["exploratory_questions"] += 1
    activated = current["question_bank"]["datasets"][0]
    activated["status"] = "exploratory_active_diagnostic"
    activated["active_count_delta"] = {
        "total_questions": 1,
        "single_step_questions": 0,
        "multi_step_questions": 1,
        "exploratory_questions": 1,
        "locked_regression_questions": 0,
    }
    activated["ledger_change"] = {"kind": "activated_registered_dataset"}
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    parent_root.mkdir()
    current_root.mkdir()
    _write_new_active_artifacts(parent_root)
    _write_new_active_artifacts(current_root)

    assert module.check_manifest_change(current, parent, current_root, parent_root) == []


def test_manifest_change_rejects_inactive_activation_masked_by_valid_extension(tmp_path):
    module = _load_manifest_check_module()
    parent, current = _extension_manifests()
    parent_root = tmp_path / "parent"
    current_root = tmp_path / "current"
    (parent_root / "test_reports").mkdir(parents=True)
    (current_root / "test_reports").mkdir(parents=True)
    first = json.dumps(_row(case_id="old")) + "\n"
    (parent_root / "test_reports/registered_dataset.jsonl").write_text(
        first, encoding="utf-8"
    )
    (current_root / "test_reports/registered_dataset.jsonl").write_text(
        first + json.dumps(_row(case_id="new")) + "\n", encoding="utf-8"
    )

    inactive_path = "test_reports/inactive_registered_dataset.jsonl"
    inactive = {
        **_active_dataset_entry(),
        "path": inactive_path,
        "status": "inactive_diagnostic",
    }
    inactive.pop("active_count_delta")
    parent["question_bank"]["datasets"].append(inactive)
    activated = copy.deepcopy(inactive)
    activated["status"] = "exploratory_active_diagnostic"
    current["question_bank"]["datasets"].append(activated)
    (parent_root / inactive_path).write_text(first, encoding="utf-8")
    (current_root / inactive_path).write_text(first, encoding="utf-8")

    errors = module.check_manifest_change(current, parent, current_root, parent_root)

    assert any("reactivated dataset" in error and "requires active_count_delta" in error for error in errors)


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


def test_manifest_checker_rejects_stale_latest_rotation_provenance():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    for scope in (manifest, manifest["question_bank"]):
        rotation = scope["answer_grounding_rotation"]
        rotation["runtime_load_evidence"][
            "validated_run"
        ] = "retrieval_eval_20260830_194345"
        rotation["manual_jsonl_inspection"] = (
            "Inspected retrieval_eval_20260830_194345 user-visible answer and citations."
        )
        rotation["scope"] = "retrieval_eval_20260830_194345 tracking validation only"

    errors = module.check_manifest(manifest)

    assert any("runtime_load_evidence stale" in error for error in errors)
    assert any("manual_jsonl_inspection stale" in error for error in errors)
    assert any("scope stale" in error for error in errors)


def test_manifest_checker_rejects_tracking_only_scope_for_clean_production_evidence():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    misleading_scope = (
        "retrieval_eval_20260830_201603 tracking-integrity repair only: reconcile "
        "latest-run provenance; no production retrieval or answering behavior changed"
    )
    validated_changes = [
        {
            "commit": "a9d870a",
            "component": "retrieval",
            "paths": [
                "packages/retrieval/src/manuals_rag_retrieval/query_analysis.py"
            ],
        },
        {
            "commit": "9ad25b1",
            "component": "answering",
            "paths": ["packages/answering/src/manuals_rag_answering/generator.py"],
        },
    ]
    for scope in (manifest, manifest["question_bank"]):
        rotation = scope["answer_grounding_rotation"]
        rotation["classification"] = "accepted_clean_grounded_answer"
        rotation["validated_production_changes"] = copy.deepcopy(validated_changes)
        rotation["scope"] = misleading_scope

    errors = module.check_manifest(manifest)

    assert any("scope misleading" in error for error in errors)


def test_manifest_checker_accepts_structured_clean_production_scope(monkeypatch):
    module = _load_manifest_check_module()
    changed_paths = {
        "packages/retrieval/src/manuals_rag_retrieval/query_analysis.py",
        "packages/retrieval/src/manuals_rag_retrieval/retriever.py",
        "packages/answering/src/manuals_rag_answering/generator.py",
    }
    monkeypatch.setattr(
        module,
        "_git_commit_changed_paths",
        lambda commit, base: (commit, changed_paths, None),
    )
    monkeypatch.setattr(
        module, "_git_path_exists_at_commit", lambda commit, path, base: True
    )
    manifest = _minimal_manifest()
    validated_changes = [
        {
            "commit": "a9d870a",
            "component": "retrieval",
            "paths": [
                "packages/retrieval/src/manuals_rag_retrieval/query_analysis.py",
                "packages/retrieval/src/manuals_rag_retrieval/retriever.py",
            ],
        },
        {
            "commit": "9ad25b1",
            "component": "answering",
            "paths": ["packages/answering/src/manuals_rag_answering/generator.py"],
        },
    ]
    for scope in (manifest, manifest["question_bank"]):
        rotation = scope["answer_grounding_rotation"]
        rotation["classification"] = "accepted_clean_grounded_answer"
        rotation["validated_production_changes"] = copy.deepcopy(validated_changes)
        rotation["scope"] = (
            "retrieval_eval_20260830_201603 validates production retrieval and "
            "answering changes"
        )

    assert module.check_manifest(manifest) == []


def test_manifest_checker_rejects_nonexistent_validated_production_commit(monkeypatch):
    module = _load_manifest_check_module()
    monkeypatch.setattr(
        module,
        "_git_commit_changed_paths",
        lambda commit, base: (None, set(), "unknown revision"),
    )
    manifest = _minimal_manifest()
    validated_changes = [
        {
            "commit": "dddddddddddddddddddddddddddddddddddddddd",
            "component": "retrieval",
            "paths": ["packages/retrieval/src/manuals_rag_retrieval/retriever.py"],
        }
    ]
    for scope in (manifest, manifest["question_bank"]):
        rotation = scope["answer_grounding_rotation"]
        rotation["classification"] = "accepted_clean_grounded_answer"
        rotation["validated_production_changes"] = copy.deepcopy(validated_changes)
        rotation["scope"] = (
            "retrieval_eval_20260830_201603 validates production retrieval changes"
        )

    errors = module.check_manifest(manifest, Path.cwd())

    assert any("commit is not resolvable in Git" in error for error in errors)


def test_manifest_checker_rejects_existing_commit_with_unrelated_path(monkeypatch):
    module = _load_manifest_check_module()
    monkeypatch.setattr(
        module,
        "_git_commit_changed_paths",
        lambda commit, base: (
            commit,
            {"packages/retrieval/src/manuals_rag_retrieval/retriever.py"},
            None,
        ),
    )
    monkeypatch.setattr(
        module, "_git_path_exists_at_commit", lambda commit, path, base: True
    )
    manifest = _minimal_manifest()
    validated_changes = [
        {
            "commit": "a9d870a",
            "component": "retrieval",
            "paths": ["packages/answering/src/manuals_rag_answering/generator.py"],
        }
    ]
    for scope in (manifest, manifest["question_bank"]):
        rotation = scope["answer_grounding_rotation"]
        rotation["classification"] = "accepted_clean_grounded_answer"
        rotation["validated_production_changes"] = copy.deepcopy(validated_changes)
        rotation["scope"] = (
            "retrieval_eval_20260830_201603 validates production retrieval changes"
        )

    errors = module.check_manifest(manifest, Path.cwd())

    assert any("path is not changed by commit" in error for error in errors)


def test_manifest_checker_rejects_nonexistent_path_in_existing_commit(monkeypatch):
    module = _load_manifest_check_module()
    monkeypatch.setattr(
        module,
        "_git_commit_changed_paths",
        lambda commit, base: (commit, set(), None),
    )
    monkeypatch.setattr(
        module, "_git_path_exists_at_commit", lambda commit, path, base: False
    )
    manifest = _minimal_manifest()
    validated_changes = [
        {
            "commit": "9ad25b1",
            "component": "answering",
            "paths": ["packages/answering/src/does_not_exist.py"],
        }
    ]
    for scope in (manifest, manifest["question_bank"]):
        rotation = scope["answer_grounding_rotation"]
        rotation["classification"] = "accepted_clean_grounded_answer"
        rotation["validated_production_changes"] = copy.deepcopy(validated_changes)
        rotation["scope"] = (
            "retrieval_eval_20260830_201603 validates production answering changes"
        )

    errors = module.check_manifest(manifest, Path.cwd())

    assert any("path does not exist at commit" in error for error in errors)


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
    manifest["current_failure_sources"]["retrieval"] = "none"
    manifest["question_bank"]["current_failure_sources"]["retrieval"] = "none"

    errors = module.check_manifest(manifest)

    assert errors == []


def test_manifest_checker_rejects_stale_current_answer_failure_source_run():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["current_answer_failure_source"] = (
        "retrieval_eval_20260827_074631 is historical answer evidence."
    )
    manifest["question_bank"]["current_answer_failure_source"] = (
        "retrieval_eval_20260827_074631 is historical answer evidence."
    )

    errors = module.check_manifest(manifest)

    assert any("root.current_answer_failure_source stale" in error for error in errors)
    assert any(
        "question_bank.current_answer_failure_source stale" in error for error in errors
    )


def test_manifest_checker_rejects_stale_current_answer_failure_source_reason():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["current_answer_failure_source"] = (
        "retrieval_eval_20260830_201603 is the current answer failure source: "
        "expected_evidence_not_cited."
    )
    manifest["question_bank"]["current_answer_failure_source"] = (
        "retrieval_eval_20260830_201603 is the current answer failure source: "
        "expected_evidence_not_cited."
    )

    errors = module.check_manifest(manifest)

    assert any(
        "root.current_answer_failure_source stale" in error
        and "expected_terms_missing" in error
        for error in errors
    )
    assert any(
        "question_bank.current_answer_failure_source stale" in error
        and "expected_terms_missing" in error
        for error in errors
    )


def test_manifest_checker_rejects_missing_or_unequal_current_answer_failure_source():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["current_answer_failure_source"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["current_answer_failure_source"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["current_answer_failure_source"] = (
        "retrieval_eval_20260830_201603 stale answer failure source: "
        "expected_terms_missing."
    )

    assert any(
        "current_answer_failure_source missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.current_answer_failure_source" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "current_answer_failure_source mismatch" in error
        for error in module.check_manifest(unequal)
    )


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


def test_manifest_checker_rejects_current_failure_sources_when_failures_empty():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    for key in (
        "answer_current_failure_categories",
        "current_answer_failure_categories",
    ):
        manifest[key] = {}
        manifest["question_bank"][key] = {}
    manifest["current_answer_failure_source"] = (
        "No current answer failures after retrieval_eval_20260902_013157 passed row 17."
    )
    manifest["question_bank"]["current_answer_failure_source"] = manifest[
        "current_answer_failure_source"
    ]
    manifest["current_failure_sources"]["answer"] = "retrieval_eval_20260901_235625"
    manifest["question_bank"]["current_failure_sources"]["answer"] = (
        "retrieval_eval_20260901_235625"
    )

    errors = module.check_manifest(manifest)

    assert any(
        "root.current_failure_sources.answer stale" in error for error in errors
    )
    assert any(
        "question_bank.current_failure_sources.answer stale" in error
        for error in errors
    )


def test_manifest_checker_rejects_missing_or_unequal_current_failure_sources():
    module = _load_manifest_check_module()
    missing_root = _minimal_manifest()
    del missing_root["current_failure_sources"]
    missing_nested = _minimal_manifest()
    del missing_nested["question_bank"]["current_failure_sources"]
    unequal = _minimal_manifest()
    unequal["question_bank"]["current_failure_sources"] = {
        "retrieval": "retrieval_eval_20260830_201603 candidate_miss",
        "answer": "retrieval_eval_20260827_074631 expected_terms_missing",
    }

    assert any(
        "current_failure_sources missing required duplicate" in error
        for error in module.check_manifest(missing_root)
    )
    assert any(
        "question_bank.current_failure_sources" in error
        and "missing required duplicate" in error
        for error in module.check_manifest(missing_nested)
    )
    assert any(
        "current_failure_sources mismatch" in error
        for error in module.check_manifest(unequal)
    )


def test_manifest_checker_rejects_stale_current_failure_sources_reason():
    module = _load_manifest_check_module()
    manifest = _minimal_manifest()
    manifest["current_failure_sources"]["answer"] = (
        "retrieval_eval_20260830_201603 expected_evidence_not_cited"
    )
    manifest["question_bank"]["current_failure_sources"]["answer"] = (
        "retrieval_eval_20260830_201603 expected_evidence_not_cited"
    )

    errors = module.check_manifest(manifest)

    assert any(
        "root.current_failure_sources.answer stale" in error
        and "expected_terms_missing" in error
        for error in errors
    )
    assert any(
        "question_bank.current_failure_sources.answer stale" in error
        and "expected_terms_missing" in error
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
