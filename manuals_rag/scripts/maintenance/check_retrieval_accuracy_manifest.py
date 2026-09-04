#!/usr/bin/env python3
"""Deterministic consistency checks for retrieval accuracy tracking state."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_ROOT_TO_QUESTION_BANK_FIELDS = (
    "latest_false_negative_repair",
    "next_target",
    "remaining",
    "answer_grounding_status",
    "answer_grounding_rotation",
    "run_exclusions",
    "unresolved_guardrail_findings",
    "current_retrieval_failure_source",
    "current_answer_failure_source",
    "current_failure_sources",
    "latest_cross_document_validation",
    "latest_cross_document_probe",
    "latest_contextual_row14_repair",
    "latest_contextual_quantity_answer_repair",
    "latest_matrix_retrieval_guardrail_containment",
    "latest_http_fallback_telemetry_containment",
    "latest_composite_citation_scoring_containment",
    "latest_comparison_setting_side_binding_containment",
    "latest_eval_question_generation_context_scope_review",
    "latest_cross_document_answer_rotation",
    "latest_cross_document_row8_answer_repair",
    "latest_cross_document_row8_answer_partial_side_containment",
    "latest_contextual_procedure_rows_3_4_source_review",
    "latest_contextual_procedure_rows_5_6_answer_evidence_failure",
    "latest_multi_step_expected_context_scoring_containment",
    "answer_grounding_cross_document_rows_6_7",
    "answer_grounding_contextual_rows_15_16",
    "answer_grounding_contextual_rows_17_18",
    "answer_grounding_contextual_rows_19_20",
    "answer_grounding_sibling_rows_15_16",
    "answer_grounding_single_step_v2_rows_5_6",
    "answer_grounding_single_step_v2_rows_7_8",
    "answer_grounding_single_step_v2_rows_9_10",
    "latest_status_output_row2_diagnostic_experiment",
    "latest_temp_worktree_hygiene_guardrail",
    "partial_claim_citation_pruning_containment",
    "llm_answer_judge_policy",
    "manifest_integrity_repairs",
    "updated_at",
)

RETRIEVAL_FAILURE_ALIASES = (
    "failure_categories",
    "current_failure_categories",
    "current_retrieval_failure_categories",
    "retrieval_current_failure_categories",
)

ANSWER_FAILURE_ALIASES = (
    "answer_current_failure_categories",
    "current_answer_failure_categories",
)

MINIMUM_ACTIVE_COUNTS = {
    "total_questions": 208,
    "single_step_questions": 101,
    "multi_step_questions": 107,
    "exploratory_questions": 208,
}
MINIMUM_RUN_EXCLUSIONS = 126
ACTIVE_COUNT_FIELDS = (
    "total_questions",
    "single_step_questions",
    "multi_step_questions",
    "exploratory_questions",
)

_MISSING = object()
_RUN_ID_RE = re.compile(
    r"retrieval_eval_(?:dataset_|results_|summary_|manifest_)?(\d{8}_\d{6})"
)
_ACCEPTED_RUN_ID_RE = re.compile(r"\bretrieval_eval_(\d{8}_\d{6})\b")


def _target_row_refs(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    normalized = value.lower()
    active_index = normalized.rfind("continue with")
    if active_index >= 0:
        normalized = normalized[active_index:]
    normalized = normalized.split("preserve", 1)[0]
    refs: set[str] = set()
    for prefix in ("row", "rows"):
        marker = f"{prefix} "
        start = 0
        while True:
            index = normalized.find(marker, start)
            if index < 0:
                break
            cursor = index + len(marker)
            token = []
            while cursor < len(normalized) and (
                normalized[cursor].isdigit() or normalized[cursor] in "-, /and"
            ):
                token.append(normalized[cursor])
                cursor += 1
            segment = "".join(token)
            number = []
            for char in segment:
                if char.isdigit():
                    number.append(char)
                elif number:
                    refs.add("".join(number))
                    number = []
            if number:
                refs.add("".join(number))
            start = cursor
    return refs


def _get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _same(label: str, values: dict[str, Any], errors: list[str]) -> None:
    comparable = {key: value for key, value in values.items() if value is not _MISSING}
    if len(comparable) <= 1:
        return
    first_key, first_value = next(iter(comparable.items()))
    for key, value in list(comparable.items())[1:]:
        if value != first_value:
            errors.append(
                f"{label} mismatch: {first_key}={first_value!r} but {key}={value!r}"
            )


def _require_all_present(label: str, values: dict[str, Any], errors: list[str]) -> bool:
    missing = [key for key, value in values.items() if value is _MISSING]
    if not missing:
        return True
    present = [key for key, value in values.items() if value is not _MISSING]
    errors.append(
        f"{label} missing required duplicate(s): {', '.join(missing)}"
        + (f"; present: {', '.join(present)}" if present else "")
    )
    return False


def _require_same(label: str, values: dict[str, Any], errors: list[str]) -> None:
    if _require_all_present(label, values, errors):
        _same(label, values, errors)


def _run_id_from_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _RUN_ID_RE.search(value)
    if not match:
        return None
    return f"retrieval_eval_{match.group(1)}"


def _git_commit_changed_paths(
    commit: str, artifact_base_dir: Path | None
) -> tuple[str | None, set[str], str | None]:
    """Resolve a commit and return its changed paths relative to the manifest root."""
    git_dir = artifact_base_dir or Path.cwd()
    try:
        resolved = subprocess.run(
            ["git", "-C", str(git_dir), "rev-parse", "--verify", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        prefix = subprocess.run(
            ["git", "-C", str(git_dir), "rev-parse", "--show-prefix"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changed = subprocess.run(
            [
                "git",
                "-C",
                str(git_dir),
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--name-only",
                "-r",
                resolved,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, OSError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        return None, set(), detail

    relative_paths: set[str] = set()
    for path in changed:
        if not prefix:
            relative_paths.add(path)
        elif path.startswith(prefix):
            relative_paths.add(path[len(prefix) :])
    return resolved, relative_paths, None


def _git_path_exists_at_commit(
    commit: str, path: str, artifact_base_dir: Path | None
) -> bool:
    git_dir = artifact_base_dir or Path.cwd()
    try:
        prefix = subprocess.run(
            ["git", "-C", str(git_dir), "rev-parse", "--show-prefix"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", str(git_dir), "cat-file", "-e", f"{commit}:{prefix}{path}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return True


def _check_answer_grounding_rotation_artifacts(
    label: str,
    rotation: Any,
    artifact_base_dir: Path | None,
    errors: list[str],
) -> None:
    if not isinstance(rotation, dict):
        return
    latest_run = rotation.get("latest_run")
    if not isinstance(latest_run, str):
        return
    for key in ("dataset", "results", "summary", "manifest"):
        value = rotation.get(key)
        artifact_run = _run_id_from_value(value)
        if artifact_run is None:
            continue
        if artifact_run != latest_run:
            errors.append(
                f"{label}.{key} run id mismatch: latest_run={latest_run!r} "
                f"but {key}={value!r}"
            )

    runtime = rotation.get("runtime_load_evidence")
    validated_run = runtime.get("validated_run") if isinstance(runtime, dict) else None
    if validated_run != latest_run:
        errors.append(
            f"{label}.runtime_load_evidence stale: latest_run={latest_run!r} "
            f"but validated_run={validated_run!r}"
        )
    for key in ("api_url", "container", "health", "fingerprints"):
        value = runtime.get(key) if isinstance(runtime, dict) else None
        if not value:
            errors.append(f"{label}.runtime_load_evidence.{key} must be nonempty")

    manual_inspection = rotation.get("manual_jsonl_inspection")
    if not isinstance(manual_inspection, str) or (
        latest_run not in manual_inspection
        and _run_id_from_value(manual_inspection) != latest_run
    ):
        errors.append(
            f"{label}.manual_jsonl_inspection stale: "
            f"must identify latest_run={latest_run!r}"
        )
    scope = rotation.get("scope")
    if not isinstance(scope, str) or latest_run not in scope:
        errors.append(f"{label}.scope stale: must identify latest_run={latest_run!r}")

    classification = rotation.get("classification")
    if isinstance(classification, str) and classification.startswith("accepted_clean"):
        validated_changes = rotation.get("validated_production_changes")
        if not isinstance(validated_changes, list) or not validated_changes:
            errors.append(
                f"{label}.validated_production_changes must identify the production "
                "changes validated by an accepted clean run"
            )
            return

        components: set[str] = set()
        for index, change in enumerate(validated_changes):
            change_label = f"{label}.validated_production_changes[{index}]"
            if not isinstance(change, dict):
                errors.append(f"{change_label} must be an object")
                continue
            commit = change.get("commit")
            component = change.get("component")
            paths = change.get("paths")
            commit_valid = isinstance(commit, str) and bool(
                re.fullmatch(r"[0-9a-f]{7,40}", commit)
            )
            if not commit_valid:
                errors.append(f"{change_label}.commit must be a Git commit id")
            if not isinstance(component, str) or not component.strip():
                errors.append(f"{change_label}.component must be nonempty")
            else:
                components.add(component.strip().lower())
            if not isinstance(paths, list) or not paths or not all(
                isinstance(path, str) and path.strip() for path in paths
            ):
                errors.append(f"{change_label}.paths must contain nonempty paths")
            elif commit_valid:
                _resolved, changed_paths, git_error = _git_commit_changed_paths(
                    commit, artifact_base_dir
                )
                if git_error is not None:
                    errors.append(
                        f"{change_label}.commit is not resolvable in Git: {commit!r}"
                    )
                else:
                    for path in paths:
                        normalized_path = path.strip().lstrip("./")
                        if not _git_path_exists_at_commit(
                            commit, normalized_path, artifact_base_dir
                        ):
                            errors.append(
                                f"{change_label}.path does not exist at commit "
                                f"{commit!r}: {normalized_path!r}"
                            )
                        elif normalized_path not in changed_paths:
                            errors.append(
                                f"{change_label}.path is not changed by commit "
                                f"{commit!r}: {normalized_path!r}"
                            )

        scope_lower = scope.lower() if isinstance(scope, str) else ""
        if re.match(
            rf"^{re.escape(latest_run.lower())}\s+tracking[-\w ]{{0,40}}\bonly\b",
            scope_lower,
        ):
            errors.append(
                f"{label}.scope misleading: accepted clean evidence cannot be "
                "described as tracking-only"
            )
        for component in components:
            if component not in scope_lower:
                errors.append(
                    f"{label}.scope incomplete: validated production component "
                    f"{component!r} is not named"
                )


def _check_latest_answer_mode_run_freshness(
    manifest: dict[str, Any],
    artifact_base_dir: Path | None,
    errors: list[str],
) -> None:
    """Reject current-state tracking that predates a complete answer-mode run."""
    if artifact_base_dir is None:
        return
    reports_dir = artifact_base_dir / "test_reports"
    if not reports_dir.is_dir():
        return
    try:
        subprocess.run(
            ["git", "-C", str(artifact_base_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_backed = True
    except (subprocess.CalledProcessError, OSError):
        git_backed = False

    def _tracked(path: Path) -> bool:
        if not git_backed:
            return True
        result = subprocess.run(
            [
                "git",
                "-C",
                str(artifact_base_dir),
                "ls-files",
                "--error-unmatch",
                str(path.relative_to(artifact_base_dir)),
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _jsonl_rows(path: Path) -> list[dict[str, Any]] | None:
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            return None
        if not rows or not all(isinstance(row, dict) for row in rows):
            return None
        return rows

    def _declares_incomplete(payload: dict[str, Any]) -> bool:
        if payload.get("evaluation_skipped") is True:
            return True
        status = payload.get("status")
        return isinstance(status, str) and status.strip().lower() in {
            "cancelled",
            "canceled",
            "skipped",
            "partial",
            "incomplete",
        }

    complete_answer_runs: list[str] = []
    for path in reports_dir.glob("retrieval_eval_manifest_*.json"):
        if not _tracked(path):
            continue
        artifact_run = _run_id_from_value(path.name)
        if artifact_run is None:
            continue
        try:
            run_manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            run_manifest.get("response_mode") != "answer_with_citations"
            or _declares_incomplete(run_manifest)
        ):
            continue
        suffix = artifact_run.removeprefix("retrieval_eval_")
        dataset_path, results_path, summary_path = (
            reports_dir / f"retrieval_eval_dataset_{suffix}.jsonl",
            reports_dir / f"retrieval_eval_results_{suffix}.jsonl",
            reports_dir / f"retrieval_eval_summary_{suffix}.json",
        )
        required = (path, dataset_path, results_path, summary_path)
        if not all(
            candidate.is_file() and _tracked(candidate) for candidate in required
        ):
            continue
        dataset_rows = _jsonl_rows(dataset_path)
        result_rows = _jsonl_rows(results_path)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            dataset_rows is None
            or result_rows is None
            or not isinstance(summary, dict)
            or _declares_incomplete(summary)
        ):
            continue
        total_queries = summary.get("total_queries")
        answer_eval_count = summary.get("answer_eval_count")
        dataset_case_ids = [row.get("case_id") for row in dataset_rows]
        result_case_ids = [
            row.get("case", {}).get("case_id")
            if isinstance(row.get("case"), dict)
            else None
            for row in result_rows
        ]
        answer_verdicts = [
            row.get("answer_evaluation", {}).get("passed")
            if isinstance(row.get("answer_evaluation"), dict)
            else None
            for row in result_rows
        ]
        answer_passed_queries = summary.get("answer_passed_queries")
        answer_failed_queries = summary.get("answer_failed_queries")
        if (
            not isinstance(total_queries, int)
            or isinstance(total_queries, bool)
            or total_queries <= 0
            or answer_eval_count != total_queries
            or len(dataset_rows) != total_queries
            or len(result_rows) != total_queries
            or not all(
                isinstance(case_id, str) and case_id.strip()
                for case_id in dataset_case_ids + result_case_ids
            )
            or len(set(dataset_case_ids)) != total_queries
            or len(set(result_case_ids)) != total_queries
            or set(dataset_case_ids) != set(result_case_ids)
            or not all(isinstance(verdict, bool) for verdict in answer_verdicts)
            or summary.get("passed_queries", 0) + summary.get("failed_queries", 0)
            != total_queries
            or not isinstance(answer_passed_queries, int)
            or isinstance(answer_passed_queries, bool)
            or answer_passed_queries < 0
            or not isinstance(answer_failed_queries, int)
            or isinstance(answer_failed_queries, bool)
            or answer_failed_queries < 0
            or answer_passed_queries + answer_failed_queries != answer_eval_count
            or sum(verdict is True for verdict in answer_verdicts)
            != answer_passed_queries
            or sum(verdict is False for verdict in answer_verdicts)
            != answer_failed_queries
        ):
            continue
        complete_answer_runs.append(artifact_run)

    if not complete_answer_runs:
        return
    newest_run = max(complete_answer_runs)
    rotation_run = _get_path(manifest, "answer_grounding_rotation.latest_run")
    if not isinstance(rotation_run, str) or rotation_run < newest_run:
        errors.append(
            "answer_grounding current-run state stale: "
            f"answer_grounding_rotation.latest_run={rotation_run!r} but newest complete "
            f"answer-mode artifacts are {newest_run!r}"
        )


def _check_accepted_clean_run_artifacts(
    label: str,
    status: Any,
    artifact_base_dir: Path | None,
    errors: list[str],
) -> None:
    if artifact_base_dir is None or not isinstance(status, dict):
        return
    accepted_runs = status.get("accepted_clean_runs")
    if not isinstance(accepted_runs, list):
        return
    for entry in accepted_runs:
        if not isinstance(entry, str):
            continue
        match = _ACCEPTED_RUN_ID_RE.search(entry)
        if not match:
            continue
        run_suffix = match.group(1)
        for kind in ("dataset", "results"):
            rel_path = Path(f"test_reports/retrieval_eval_{kind}_{run_suffix}.jsonl")
            if not (artifact_base_dir / rel_path).is_file():
                errors.append(
                    f"{label}.accepted_clean_runs references missing {kind} artifact "
                    f"for retrieval_eval_{run_suffix}: {rel_path}"
                )


def _check_current_retrieval_failure_source(
    label: str, manifest: dict[str, Any], errors: list[str]
) -> None:
    failures = manifest.get("current_retrieval_failure_categories")
    rotation = manifest.get("answer_grounding_rotation")
    latest_run = rotation.get("latest_run") if isinstance(rotation, dict) else None
    source = manifest.get("current_retrieval_failure_source")
    if not failures:
        if (
            isinstance(source, str)
            and (
                " is the current retrieval failure source" in source.lower()
                or "current retrieval and answer failure categories" in source.lower()
            )
        ):
            errors.append(
                f"{label}.current_retrieval_failure_source stale: "
                "current retrieval failures are empty"
            )
        return
    if not isinstance(failures, dict):
        errors.append(f"{label}.current_retrieval_failure_categories must be an object")
        return
    if not isinstance(latest_run, str):
        return
    if not isinstance(source, str) or latest_run not in source:
        errors.append(
            f"{label}.current_retrieval_failure_source stale: "
            f"current retrieval failures are tracked on {latest_run!r}"
        )


def _check_current_answer_failure_source(
    label: str, manifest: dict[str, Any], errors: list[str]
) -> None:
    failures = manifest.get("current_answer_failure_categories")
    rotation = manifest.get("answer_grounding_rotation")
    latest_run = rotation.get("latest_run") if isinstance(rotation, dict) else None
    source = manifest.get("current_answer_failure_source")
    if not failures:
        if isinstance(source, str) and (
            " is the current answer failure source" in source.lower()
            or "current retrieval and answer failure categories" in source.lower()
        ):
            errors.append(
                f"{label}.current_answer_failure_source stale: "
                "current answer failures are empty"
            )
        return
    if not isinstance(failures, dict):
        errors.append(f"{label}.current_answer_failure_categories must be an object")
        return
    if not isinstance(source, str):
        errors.append(
            f"{label}.current_answer_failure_source stale: "
            "current answer failures have no source"
        )
        return
    if isinstance(latest_run, str) and latest_run not in source:
        errors.append(
            f"{label}.current_answer_failure_source stale: "
            f"current answer failures are tracked on {latest_run!r}"
        )
    source_lower = source.lower()
    for reason in failures:
        if str(reason).lower() not in source_lower:
            errors.append(
                f"{label}.current_answer_failure_source stale: "
                f"current answer failure reason {reason!r} is not named"
            )


def _check_current_failure_sources_alias(
    label: str, manifest: dict[str, Any], errors: list[str]
) -> None:
    sources = manifest.get("current_failure_sources")
    if not isinstance(sources, dict):
        errors.append(f"{label}.current_failure_sources must be an object")
        return
    rotation = manifest.get("answer_grounding_rotation")
    latest_run = rotation.get("latest_run") if isinstance(rotation, dict) else None
    for kind, failure_key in (
        ("retrieval", "current_retrieval_failure_categories"),
        ("answer", "current_answer_failure_categories"),
    ):
        failures = manifest.get(failure_key)
        source = sources.get(kind)
        source_text = source.lower() if isinstance(source, str) else ""
        if not failures:
            if source not in (None, "", "none"):
                errors.append(
                    f"{label}.current_failure_sources.{kind} stale: "
                    f"{failure_key} is empty"
                )
            continue
        if not isinstance(failures, dict):
            continue
        if not isinstance(source, str) or source in ("", "none"):
            errors.append(
                f"{label}.current_failure_sources.{kind} stale: "
                f"{failure_key} has failures but no current source"
            )
            continue
        if isinstance(latest_run, str) and latest_run not in source:
            errors.append(
                f"{label}.current_failure_sources.{kind} stale: "
                f"{failure_key} is tracked on {latest_run!r}"
            )
        for reason in failures:
            if str(reason).lower() not in source_text:
                errors.append(
                    f"{label}.current_failure_sources.{kind} stale: "
                    f"current failure reason {reason!r} is not named"
                )


def _run_exclusion_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        excluded = value.get("excluded")
        if isinstance(excluded, list):
            return len(excluded)
    return None


def _check_monotonic_tracking_floors(
    label: str, manifest: dict[str, Any], errors: list[str]
) -> None:
    policy = manifest.get("quality_policy")
    if not isinstance(policy, dict) or not policy.get("active_counts_must_not_shrink"):
        return

    question_bank = manifest.get("question_bank")
    if not isinstance(question_bank, dict):
        return
    for key, minimum in MINIMUM_ACTIVE_COUNTS.items():
        value = question_bank.get(key)
        if not isinstance(value, int):
            errors.append(f"question_bank.{key} must be an integer count")
        elif value < minimum:
            errors.append(
                f"question_bank.{key} dropped below monotonic floor: "
                f"{value!r} < {minimum!r}"
            )

    for path, value in {
        "run_exclusions": manifest.get("run_exclusions"),
        "question_bank.run_exclusions": question_bank.get("run_exclusions"),
    }.items():
        count = _run_exclusion_count(value)
        if count is None:
            errors.append(f"{path} must be a list or object with an excluded list")
        elif count < MINIMUM_RUN_EXCLUSIONS:
            errors.append(
                f"{path} dropped below monotonic floor: "
                f"{count!r} < {MINIMUM_RUN_EXCLUSIONS!r}"
            )


def check_manifest(
    data: dict[str, Any], artifact_base_dir: Path | None = None
) -> list[str]:
    errors: list[str] = []
    question_bank = data.get("question_bank")
    if not isinstance(question_bank, dict):
        return ["missing question_bank object"]

    for field in REQUIRED_ROOT_TO_QUESTION_BANK_FIELDS:
        _require_same(
            field,
            {
                field: _get_path(data, field),
                f"question_bank.{field}": _get_path(data, f"question_bank.{field}"),
            },
            errors,
        )

    _require_same(
        "retrieval_failure_categories",
        {
            field: _get_path(data, field)
            for field in RETRIEVAL_FAILURE_ALIASES
        }
        | {
            f"question_bank.{field}": _get_path(data, f"question_bank.{field}")
            for field in RETRIEVAL_FAILURE_ALIASES
        },
        errors,
    )
    _require_same(
        "answer_failure_categories",
        {
            field: _get_path(data, field)
            for field in ANSWER_FAILURE_ALIASES
        }
        | {
            f"question_bank.{field}": _get_path(data, f"question_bank.{field}")
            for field in ANSWER_FAILURE_ALIASES
        },
        errors,
    )
    _check_answer_grounding_rotation_artifacts(
        "answer_grounding_rotation",
        _get_path(data, "answer_grounding_rotation"),
        artifact_base_dir,
        errors,
    )
    _check_answer_grounding_rotation_artifacts(
        "question_bank.answer_grounding_rotation",
        _get_path(data, "question_bank.answer_grounding_rotation"),
        artifact_base_dir,
        errors,
    )
    _check_latest_answer_mode_run_freshness(data, artifact_base_dir, errors)
    _check_accepted_clean_run_artifacts(
        "answer_grounding_status",
        _get_path(data, "answer_grounding_status"),
        artifact_base_dir,
        errors,
    )
    _check_accepted_clean_run_artifacts(
        "question_bank.answer_grounding_status",
        _get_path(data, "question_bank.answer_grounding_status"),
        artifact_base_dir,
        errors,
    )
    _check_current_retrieval_failure_source("root", data, errors)
    _check_current_retrieval_failure_source("question_bank", question_bank, errors)
    _check_current_answer_failure_source("root", data, errors)
    _check_current_answer_failure_source("question_bank", question_bank, errors)
    _check_current_failure_sources_alias("root", data, errors)
    _check_current_failure_sources_alias("question_bank", question_bank, errors)
    _check_monotonic_tracking_floors("root", data, errors)

    target_rows = _target_row_refs(data.get("next_target"))
    remaining_rows = _target_row_refs(data.get("remaining"))
    if target_rows and remaining_rows and target_rows != remaining_rows:
        errors.append(
            "current_target_alias mismatch: "
            f"next_target rows={sorted(target_rows)!r} but remaining rows={sorted(remaining_rows)!r}"
        )
    status_remaining = _get_path(data, "answer_grounding_status.remaining")
    if status_remaining is not _MISSING:
        _same(
            "current_target_alias",
            {
                "next_target": _get_path(data, "next_target"),
                "remaining": _get_path(data, "remaining"),
                "answer_grounding_status.remaining": status_remaining,
            },
            errors,
        )

    root_counts = {
        "total_questions": data.get("total_questions"),
        "single_step_questions": data.get("single_step_questions"),
        "multi_step_questions": data.get("multi_step_questions"),
        "exploratory_questions": data.get("exploratory_questions"),
        "locked_regression_questions": data.get("locked_regression_questions"),
    }
    for key, root_value in root_counts.items():
        question_bank_value = question_bank.get(key)
        if root_value is not None and question_bank_value is not None:
            _same(
                f"count.{key}",
                {key: root_value, f"question_bank.{key}": question_bank_value},
                errors,
            )

    return errors


def check_manifest_change(
    current: dict[str, Any],
    parent: dict[str, Any],
    artifact_base_dir: Path | None = None,
    parent_artifact_base_dir: Path | None = None,
    parent_git_ref: str | None = None,
) -> list[str]:
    """Require active-count increases to be backed by an explicit dataset-ledger delta."""
    errors: list[str] = []
    current_bank = current.get("question_bank")
    parent_bank = parent.get("question_bank")
    if not isinstance(current_bank, dict) or not isinstance(parent_bank, dict):
        return ["current and parent manifests must contain question_bank objects"]

    count_delta: dict[str, int] = {}
    for field in ACTIVE_COUNT_FIELDS:
        current_value = current_bank.get(field)
        parent_value = parent_bank.get(field)
        if not isinstance(current_value, int) or not isinstance(parent_value, int):
            errors.append(f"question_bank.{field} must be an integer in current and parent")
            continue
        count_delta[field] = current_value - parent_value

    current_datasets = current_bank.get("datasets")
    parent_datasets = parent_bank.get("datasets")
    if not isinstance(current_datasets, list) or not isinstance(parent_datasets, list):
        errors.append("question_bank.datasets must be a list in current and parent")
        return errors

    def by_path(entries: list[Any], label: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                errors.append(f"{label} dataset entry must contain a string path")
                continue
            path = entry["path"]
            if path in indexed:
                errors.append(f"{label} dataset path is duplicated: {path}")
            indexed[path] = entry
        return indexed

    current_by_path = by_path(current_datasets, "current")
    parent_by_path = by_path(parent_datasets, "parent")
    changed_entries = [
        entry
        for path, entry in current_by_path.items()
        if path not in parent_by_path or entry != parent_by_path[path]
    ]

    ledger_delta = {field: 0 for field in ACTIVE_COUNT_FIELDS}

    def artifact_lines(
        path: str,
        *,
        base_dir: Path | None = None,
        git_ref: str | None = None,
        label: str,
    ) -> list[str] | None:
        try:
            if git_ref is not None:
                repo_root = Path(
                    subprocess.run(
                        ["git", "rev-parse", "--show-toplevel"],
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )
                candidate_path = ((artifact_base_dir or Path.cwd()) / path).resolve()
                relative_path = candidate_path.relative_to(repo_root)
                payload = subprocess.run(
                    ["git", "show", f"{git_ref}:{relative_path.as_posix()}"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            elif base_dir is not None:
                candidate_path = base_dir / path
                if not candidate_path.is_file():
                    errors.append(f"{label} dataset is missing: {path}")
                    return None
                payload = candidate_path.read_text(encoding="utf-8")
            else:
                errors.append(f"{label} dataset cannot be verified without an artifact source: {path}")
                return None
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            errors.append(f"{label} dataset is missing or unreadable: {path} ({exc})")
            return None
        return [line for line in payload.splitlines() if line.strip()]

    def classify_active_rows(path: str, lines: list[str]) -> dict[str, int] | None:
        counts = {
            "total_questions": 0,
            "single_step_questions": 0,
            "multi_step_questions": 0,
        }
        for index, line in enumerate(lines, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"dataset {path} row {index} is not valid JSON: {exc}")
                return None
            if not isinstance(row, dict):
                errors.append(f"dataset {path} row {index} must be a JSON object")
                return None
            if row.get("active") is False:
                continue
            task = row.get("retrieval_task")
            if task == "single_step_retrieval":
                counts["single_step_questions"] += 1
            elif task == "multi_step_retrieval":
                counts["multi_step_questions"] += 1
            else:
                errors.append(
                    f"dataset {path} row {index} has unclassifiable retrieval_task={task!r}"
                )
                return None
            counts["total_questions"] += 1
        return counts

    def status_identifies_active_coverage(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        normalized = value.strip().lower()
        if "inactive" in normalized:
            return False
        return bool(re.search(r"(?:^|[_\-\s])active(?:$|[_\-\s])", normalized))

    def require_active_entry_provenance(entry: dict[str, Any], label: str) -> None:
        path = entry["path"]
        for field in ("generation", "status", "quality_review"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
                "none",
                "unknown",
                "tbd",
            }:
                errors.append(f"{label} dataset {path} requires nonempty valid {field}")
        status = entry.get("status")
        if not status_identifies_active_coverage(status):
            errors.append(f"{label} dataset {path} status must identify active coverage")
        if not any(
            isinstance(entry.get(field), str) and entry[field].strip()
            for field in ("replacement_status", "supersedes", "replacement_or_supersession")
        ):
            errors.append(
                f"{label} dataset {path} requires replacement or supersession semantics"
            )
        for field in ("results_path", "summary_path", "manifest_path"):
            evidence_path = entry.get(field)
            if not isinstance(evidence_path, str) or not evidence_path.strip():
                errors.append(f"{label} dataset {path} requires nonempty {field}")
            elif artifact_base_dir is None:
                errors.append(
                    f"{label} dataset {path} {field} cannot be verified without an artifact source"
                )
            elif not (artifact_base_dir / evidence_path).is_file():
                errors.append(
                    f"{label} dataset {path} references missing {field}: {evidence_path}"
                )

    for entry in changed_entries:
        path = entry["path"]
        declared = entry.get("active_count_delta")
        is_new_entry = path not in parent_by_path
        parent_entry = parent_by_path.get(path)
        is_activation = bool(
            parent_entry
            and not status_identifies_active_coverage(parent_entry.get("status"))
            and status_identifies_active_coverage(entry.get("status"))
        )
        if not is_new_entry:
            assert parent_entry is not None
            changed_count_fields = [
                field
                for field in ("total_questions", "single_step_questions", "multi_step_questions")
                if entry.get(field) != parent_entry.get(field)
            ]
            if changed_count_fields and not isinstance(declared, dict):
                errors.append(
                    f"existing dataset {path} changed count fields "
                    f"{changed_count_fields!r} without active_count_delta and a verified "
                    "registered-dataset extension"
                )
        status = entry.get("status")
        is_new_active_entry = is_new_entry and (
            isinstance(declared, dict)
            or status_identifies_active_coverage(status)
        )

        # Validate every newly active registry entry before considering its
        # declared aggregate contribution. Otherwise one legitimate extension
        # can mask an additional skeletal active entry that omits its delta.
        if is_new_active_entry:
            require_active_entry_provenance(entry, "new active")
            current_lines = artifact_lines(
                path, base_dir=artifact_base_dir, label="new active"
            )
            if not isinstance(declared, dict):
                errors.append(f"new active dataset {path} requires active_count_delta")
            if current_lines is not None:
                active_counts = classify_active_rows(path, current_lines)
                if active_counts is not None:
                    for field, actual in active_counts.items():
                        entry_value = entry.get(field)
                        if entry_value != actual:
                            errors.append(
                                f"new dataset {path} {field} mismatch: "
                                f"file={actual} entry={entry_value}"
                            )
                        if isinstance(declared, dict):
                            declared_value = declared.get(field)
                            if declared_value != actual:
                                errors.append(
                                    f"new dataset {path} active_count_delta.{field} mismatch: "
                                    f"file={actual} declared={declared_value}"
                                )

        if is_activation:
            require_active_entry_provenance(entry, "reactivated")
            current_lines = artifact_lines(
                path, base_dir=artifact_base_dir, label="reactivated current"
            )
            parent_lines = artifact_lines(
                path,
                base_dir=parent_artifact_base_dir,
                git_ref=parent_git_ref,
                label="reactivated parent",
            )
            if not isinstance(declared, dict):
                errors.append(f"reactivated dataset {path} requires active_count_delta")
            if current_lines is not None and parent_lines is not None:
                if current_lines != parent_lines:
                    errors.append(
                        f"reactivated dataset {path} must use the unchanged registered dataset; "
                        "use a separate extension ledger change for appended rows"
                    )
                active_counts = classify_active_rows(path, current_lines)
                if active_counts is not None:
                    expected_delta = {
                        **active_counts,
                        "exploratory_questions": (
                            active_counts["total_questions"]
                            if "exploratory" in str(status).lower()
                            else 0
                        ),
                        "locked_regression_questions": (
                            active_counts["total_questions"]
                            if "locked" in str(status).lower()
                            or "regression" in str(status).lower()
                            else 0
                        ),
                    }
                    for field, actual in active_counts.items():
                        if entry.get(field) != actual:
                            errors.append(
                                f"reactivated dataset {path} {field} mismatch: "
                                f"file={actual} entry={entry.get(field)}"
                            )
                    if isinstance(declared, dict):
                        for field, actual in expected_delta.items():
                            if declared.get(field) != actual:
                                errors.append(
                                    f"reactivated dataset {path} active_count_delta.{field} "
                                    f"mismatch: file/status={actual} declared={declared.get(field)}"
                                )

        if not isinstance(declared, dict):
            continue
        for field in ACTIVE_COUNT_FIELDS:
            value = declared.get(field)
            if not isinstance(value, int) or value < 0:
                errors.append(
                    f"dataset {path} active_count_delta.{field} must be a non-negative integer"
                )
                continue
            ledger_delta[field] += value

        if path in parent_by_path and is_activation:
            change = entry.get("ledger_change")
            if not isinstance(change, dict) or change.get("kind") != "activated_registered_dataset":
                errors.append(
                    f"reactivated dataset {path} requires ledger_change.kind="
                    "'activated_registered_dataset'"
                )
        elif path in parent_by_path:
            change = entry.get("ledger_change")
            if not isinstance(change, dict) or change.get("kind") != "extended_registered_dataset":
                errors.append(
                    f"dataset {path} was modified without ledger_change.kind="
                    "'extended_registered_dataset'"
                )
            else:
                parent_entry = parent_by_path[path]
                for field in ("total_questions", "single_step_questions", "multi_step_questions"):
                    before = parent_entry.get(field)
                    after = entry.get(field)
                    declared_value = declared.get(field)
                    if not isinstance(before, int) or not isinstance(after, int):
                        errors.append(
                            f"extended dataset {path} must retain integer {field} counts"
                        )
                    elif isinstance(declared_value, int) and after - before != declared_value:
                        errors.append(
                            f"extended dataset {path} {field} delta mismatch: "
                            f"entry={after - before} declared={declared_value}"
                        )
                current_lines = artifact_lines(
                    path, base_dir=artifact_base_dir, label="current"
                )
                parent_lines = artifact_lines(
                    path,
                    base_dir=parent_artifact_base_dir,
                    git_ref=parent_git_ref,
                    label="parent",
                )
                if current_lines is not None and parent_lines is not None:
                    if len(current_lines) <= len(parent_lines):
                        errors.append(
                            f"extended dataset {path} must append a nonblank JSONL row"
                        )
                    elif current_lines[: len(parent_lines)] != parent_lines:
                        errors.append(
                            f"extended dataset {path} must be append-only; parent rows changed"
                        )
                    else:
                        appended_counts = classify_active_rows(
                            path, current_lines[len(parent_lines) :]
                        )
                        if appended_counts is not None:
                            for field, actual in appended_counts.items():
                                declared_value = declared.get(field)
                                if declared_value != actual:
                                    errors.append(
                                        f"extended dataset {path} appended {field} mismatch: "
                                        f"file={actual} declared={declared_value}"
                                    )

    if any(delta > 0 for delta in count_delta.values()) and not any(
        isinstance(entry.get("active_count_delta"), dict) for entry in changed_entries
    ):
        errors.append(
            "active question counts increased without a new dataset registry entry "
            "or verified registered-dataset extension"
        )
    for field, delta in count_delta.items():
        if delta != ledger_delta[field]:
            errors.append(
                f"active count delta mismatch for {field}: "
                f"aggregate={delta} dataset_ledger={ledger_delta[field]}"
            )
    return errors


def _manifest_from_git_ref(path: Path, git_ref: str) -> dict[str, Any]:
    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    relative_path = path.resolve().relative_to(repo_root)
    payload = subprocess.run(
        ["git", "show", f"{git_ref}:{relative_path.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        default="test_reports/retrieval_accuracy_question_bank_manifest.json",
    )
    parser.add_argument(
        "--parent-git-ref",
        help="also validate active-count and dataset-ledger deltas against this Git revision",
    )
    args = parser.parse_args()

    path = Path(args.manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = check_manifest(data, Path.cwd())
    if args.parent_git_ref:
        parent = _manifest_from_git_ref(path, args.parent_git_ref)
        errors.extend(
            check_manifest_change(
                data,
                parent,
                Path.cwd(),
                parent_git_ref=args.parent_git_ref,
            )
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{path}: manifest consistency OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
