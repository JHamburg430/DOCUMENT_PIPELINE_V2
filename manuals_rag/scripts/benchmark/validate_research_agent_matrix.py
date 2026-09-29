#!/usr/bin/env python3
"""Validate the frozen evidence-aware research-agent contract matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_LAYERS = {"controller_contract", "live_corpus"}
ALLOWED_TERMINAL_STATES = {"supported", "clarification", "insufficient"}
ALLOWED_CITATION_POLICIES = {"required", "none", "used_only"}
ALLOWED_MEMORY_POLICIES = {"context_only", "disabled"}
REQUIRED_EXPECTED_KEYS = {
    "interpretation",
    "first_tool",
    "min_rag_calls",
    "max_rag_calls",
    "min_distinct_rag_queries",
    "required_strategy_tags",
    "query_requirements",
    "terminal_state",
    "stop_reason",
    "citation_policy",
    "memory_policy",
    "prohibited_events",
}


class MatrixValidationError(ValueError):
    """Raised when a frozen matrix or manifest violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixValidationError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixValidationError(f"cannot load JSON {path}: {error}") from error


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MatrixValidationError(f"cannot load JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise MatrixValidationError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        _require(isinstance(value, dict), f"{path}:{line_number} must contain an object")
        rows.append(value)
    return rows


def validate_case(case: dict[str, Any]) -> None:
    case_id = str(case.get("case_id") or "<missing>")
    _require(case.get("schema_version") == "research-agent-validation-case-v1", f"{case_id}: bad schema")
    _require(case_id != "<missing>", "case_id is required")
    _require(bool(str(case.get("title") or "").strip()), f"{case_id}: title is required")
    _require(case.get("layer") in ALLOWED_LAYERS, f"{case_id}: invalid layer")
    _require(bool(str(case.get("category") or "").strip()), f"{case_id}: category is required")
    turns = case.get("turns")
    _require(isinstance(turns, list) and turns, f"{case_id}: turns must be nonempty")
    for turn in turns:
        _require(turn.get("role") in {"user", "assistant"}, f"{case_id}: invalid turn role")
        _require(bool(str(turn.get("content") or "").strip()), f"{case_id}: empty turn content")
    claims = case.get("required_claims")
    _require(isinstance(claims, list) and claims, f"{case_id}: required_claims must be nonempty")
    _require(len(claims) == len(set(claims)), f"{case_id}: duplicate required claims")
    expected = case.get("expected")
    _require(isinstance(expected, dict), f"{case_id}: expected contract is required")
    missing = sorted(REQUIRED_EXPECTED_KEYS - set(expected))
    _require(not missing, f"{case_id}: missing expected keys {missing}")
    _require(expected["first_tool"] == "manuals_rag", f"{case_id}: first tool must be manuals_rag")
    minimum = expected["min_rag_calls"]
    maximum = expected["max_rag_calls"]
    distinct = expected["min_distinct_rag_queries"]
    _require(isinstance(minimum, int) and minimum >= 1, f"{case_id}: invalid min_rag_calls")
    _require(isinstance(maximum, int) and maximum >= minimum, f"{case_id}: invalid max_rag_calls")
    _require(isinstance(distinct, int) and 1 <= distinct <= maximum, f"{case_id}: invalid distinct-query gate")
    _require(expected["terminal_state"] in ALLOWED_TERMINAL_STATES, f"{case_id}: invalid terminal state")
    _require(expected["citation_policy"] in ALLOWED_CITATION_POLICIES, f"{case_id}: invalid citation policy")
    _require(expected["memory_policy"] in ALLOWED_MEMORY_POLICIES, f"{case_id}: invalid memory policy")
    _require(isinstance(expected["required_strategy_tags"], list), f"{case_id}: strategy tags must be a list")
    _require(isinstance(expected["query_requirements"], list), f"{case_id}: query requirements must be a list")
    _require(isinstance(expected["prohibited_events"], list), f"{case_id}: prohibited events must be a list")
    if expected["terminal_state"] == "supported":
        _require(expected["citation_policy"] != "none", f"{case_id}: supported answer cannot forbid citations")
    if expected["terminal_state"] == "clarification":
        _require(expected["citation_policy"] == "none", f"{case_id}: clarification must not cite evidence")
    if not case.get("memory_enabled"):
        _require(expected["memory_policy"] == "disabled", f"{case_id}: disabled memory must have disabled policy")
        _require(not case.get("initial_memory"), f"{case_id}: disabled memory cannot have initial state")
    if case["layer"] == "live_corpus":
        live = case.get("live_contract")
        _require(isinstance(live, dict), f"{case_id}: live_contract is required")
        _require(bool(str(live.get("corpus") or "").strip()), f"{case_id}: live corpus is required")
        backends = live.get("run_backends")
        _require(isinstance(backends, list) and set(backends) == {"langgraph_agent", "llamaindex_agent"}, f"{case_id}: both backends required")
        _require(case.get("rag_observations") == [], f"{case_id}: live cases cannot embed scripted observations")
    else:
        observations = case.get("rag_observations")
        _require(isinstance(observations, list) and observations, f"{case_id}: contract observations required")


def validate_matrix(dataset_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    rows = _load_jsonl(dataset_path)
    _require(manifest.get("schema") == "manuals-rag-research-agent-validation-manifest-v1", "bad manifest schema")
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    _require(digest == manifest.get("dataset_sha256"), "dataset SHA-256 does not match manifest")
    _require(len(rows) == manifest.get("case_count"), "case count does not match manifest")
    ids = [str(row.get("case_id") or "") for row in rows]
    _require(len(ids) == len(set(ids)), "case IDs must be unique")
    prompt_keys = [
        (
            row["layer"],
            "\n".join(turn["content"].strip().lower() for turn in row["turns"]),
        )
        for row in rows
    ]
    _require(
        len(prompt_keys) == len(set(prompt_keys)),
        "conversation prompts must be unique within each validation layer",
    )
    for row in rows:
        validate_case(row)
    categories = Counter(str(row["category"]) for row in rows)
    layers = Counter(str(row["layer"]) for row in rows)
    _require(dict(sorted(categories.items())) == manifest.get("categories"), "category counts do not match manifest")
    _require(dict(sorted(layers.items())) == manifest.get("layers"), "layer counts do not match manifest")
    mandatory = manifest.get("mandatory_case_ids")
    _require(isinstance(mandatory, list) and mandatory, "mandatory_case_ids must be nonempty")
    _require(set(mandatory).issubset(set(ids)), "mandatory case IDs must exist in dataset")
    acceptance = manifest.get("acceptance") or {}
    _require(acceptance.get("controller_contract_pass_rate") == 1.0, "contract pass gate must be 100%")
    _require(acceptance.get("mandatory_live_pass_rate") == 1.0, "mandatory live pass gate must be 100%")
    _require(acceptance.get("required_backend_coverage") == 1.0, "backend coverage gate must be 100%")
    return {
        "status": "valid",
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "dataset_sha256": digest,
        "case_count": len(rows),
        "layers": dict(sorted(layers.items())),
        "categories": dict(sorted(categories.items())),
        "mandatory_case_count": len(mandatory),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/research_agent_validation_matrix_v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/research_agent_validation_matrix_v1.manifest.json"),
    )
    args = parser.parse_args()
    try:
        report = validate_matrix(args.dataset, args.manifest)
    except MatrixValidationError as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
