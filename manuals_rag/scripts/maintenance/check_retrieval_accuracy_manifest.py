#!/usr/bin/env python3
"""Deterministic consistency checks for retrieval accuracy tracking state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_ROOT_TO_QUESTION_BANK_FIELDS = (
    "latest_false_negative_repair",
    "next_target",
    "answer_grounding_status",
    "answer_grounding_rotation",
    "run_exclusions",
    "unresolved_guardrail_findings",
    "current_retrieval_failure_source",
    "latest_cross_document_validation",
    "latest_contextual_row14_repair",
    "latest_contextual_quantity_answer_repair",
    "latest_matrix_retrieval_guardrail_containment",
    "latest_http_fallback_telemetry_containment",
    "latest_composite_citation_scoring_containment",
    "latest_comparison_setting_side_binding_containment",
    "latest_eval_question_generation_context_scope_review",
    "answer_grounding_cross_document_rows_6_7",
    "partial_claim_citation_pruning_containment",
    "llm_answer_judge_policy",
)

RETRIEVAL_FAILURE_ALIASES = (
    "failure_categories",
    "current_failure_categories",
    "current_retrieval_failure_categories",
)

ANSWER_FAILURE_ALIASES = (
    "answer_current_failure_categories",
    "current_answer_failure_categories",
)

_MISSING = object()


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
