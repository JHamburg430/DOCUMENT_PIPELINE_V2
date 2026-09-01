#!/usr/bin/env python3
"""Deterministic consistency checks for retrieval accuracy tracking state."""

from __future__ import annotations

import argparse
import json
import re
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
    "answer_grounding_cross_document_rows_6_7",
    "answer_grounding_contextual_rows_15_16",
    "answer_grounding_contextual_rows_17_18",
    "answer_grounding_contextual_rows_19_20",
    "answer_grounding_sibling_rows_15_16",
    "answer_grounding_single_step_v2_rows_5_6",
    "answer_grounding_single_step_v2_rows_7_8",
    "answer_grounding_single_step_v2_rows_9_10",
    "latest_status_output_row2_diagnostic_experiment",
    "partial_claim_citation_pruning_containment",
    "llm_answer_judge_policy",
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

_MISSING = object()
_RUN_ID_RE = re.compile(
    r"retrieval_eval_(?:dataset_|results_|summary_|manifest_)?(\d{8}_\d{6})"
)


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


def _check_answer_grounding_rotation_artifacts(
    label: str, rotation: Any, errors: list[str]
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


def _check_current_retrieval_failure_source(
    label: str, manifest: dict[str, Any], errors: list[str]
) -> None:
    failures = manifest.get("current_retrieval_failure_categories")
    if not failures:
        return
    if not isinstance(failures, dict):
        errors.append(f"{label}.current_retrieval_failure_categories must be an object")
        return
    rotation = manifest.get("answer_grounding_rotation")
    if not isinstance(rotation, dict):
        return
    latest_run = rotation.get("latest_run")
    if not isinstance(latest_run, str):
        return
    source = manifest.get("current_retrieval_failure_source")
    if not isinstance(source, str) or latest_run not in source:
        errors.append(
            f"{label}.current_retrieval_failure_source stale: "
            f"current retrieval failures are tracked on {latest_run!r}"
        )


def check_manifest(data: dict[str, Any]) -> list[str]:
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
        errors,
    )
    _check_answer_grounding_rotation_artifacts(
        "question_bank.answer_grounding_rotation",
        _get_path(data, "question_bank.answer_grounding_rotation"),
        errors,
    )
    _check_current_retrieval_failure_source("root", data, errors)
    _check_current_retrieval_failure_source("question_bank", question_bank, errors)

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        default="test_reports/retrieval_accuracy_question_bank_manifest.json",
    )
    args = parser.parse_args()

    path = Path(args.manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = check_manifest(data)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{path}: manifest consistency OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
