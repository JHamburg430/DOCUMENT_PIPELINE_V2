#!/usr/bin/env python3
"""Deterministic consistency checks for retrieval accuracy tracking state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_TO_QUESTION_BANK_FIELDS = (
    "latest_false_negative_repair",
    "next_target",
    "answer_grounding_status",
    "answer_grounding_rotation",
    "run_exclusions",
    "unresolved_guardrail_findings",
    "current_retrieval_failure_source",
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


def _get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _same(label: str, values: dict[str, Any], errors: list[str]) -> None:
    present = {key: value for key, value in values.items() if value is not None}
    if len(present) <= 1:
        return
    first_key, first_value = next(iter(present.items()))
    for key, value in list(present.items())[1:]:
        if value != first_value:
            errors.append(
                f"{label} mismatch: {first_key}={first_value!r} but {key}={value!r}"
            )


def check_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    question_bank = data.get("question_bank")
    if not isinstance(question_bank, dict):
        return ["missing question_bank object"]

    for field in ROOT_TO_QUESTION_BANK_FIELDS:
        root_value = data.get(field)
        question_bank_value = question_bank.get(field)
        if root_value is not None or question_bank_value is not None:
            _same(
                field,
                {
                    field: root_value,
                    f"question_bank.{field}": question_bank_value,
                },
                errors,
            )

    _same(
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
    _same(
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
