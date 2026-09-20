#!/usr/bin/env python3
"""Audit identifier preservation, decomposition selectivity, and missing-claim traces."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


IDENTIFIER = re.compile(r"\b(?=[A-Z0-9./:-]*\d)[A-Z0-9]+(?:[-./:][A-Z0-9]+)+\b|\b\d+(?:\.\d+)?\b", re.I)


def audit_backend(item: dict[str, Any], backend: str) -> dict[str, Any]:
    result = item[backend]
    trace = result.get("trace") or {}
    plan = trace.get("plan") or {}
    hops = list(plan.get("hops") or [])
    original = str(item.get("query") or "")
    identifiers = {value.casefold() for value in IDENTIFIER.findall(original)}
    planned_text = " ".join(
        f"{hop.get('objective', '')} {hop.get('query', '')}" for hop in hops
    ).casefold()
    missing_identifiers = sorted(value for value in identifiers if value not in planned_text)
    category = str(item.get("agent_case_category") or "")
    mode = str(plan.get("mode") or "")
    failures: list[str] = []
    if missing_identifiers:
        failures.append("original_identifier_or_numeric_qualifier_lost")
    if mode == "single" and (
        len(hops) != 1 or original.casefold() not in str(hops[0].get("query") or "").casefold()
    ):
        failures.append("single_lookup_not_preserved_verbatim")
    if category == "dependent_multi_hop":
        if mode != "dependent" or len(hops) < 2 or not any(hop.get("depends_on") for hop in hops):
            failures.append("dependent_request_not_dependency_bound")
    if category in {"parallel_multi_part", "cross_document"}:
        if mode != "parallel" or len([hop for hop in hops if hop.get("required")]) < 2:
            failures.append("parallel_request_not_split_into_required_claims")
    max_hops = int(trace.get("max_hops") or 0)
    completed = list(trace.get("completed_hops") or [])
    if max_hops and len(completed) > max_hops:
        failures.append("hop_budget_exceeded")
    required = {str(hop.get("hop_id")) for hop in hops if hop.get("required")}
    support = set((trace.get("required_claim_support") or {}).keys())
    missing_claims = sorted(required - support)
    if missing_claims and result.get("sufficient") is True:
        failures.append("missing_required_claim_silently_sufficient")
    return {
        "case_id": item.get("case_id"), "category": category, "mode": mode,
        "identifier_count": len(identifiers), "missing_identifiers": missing_identifiers,
        "required_claims": sorted(required), "missing_claims": missing_claims,
        "sufficient": result.get("sufficient"), "stop_reason": result.get("stop_reason"),
        "checks_passed": not failures, "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite artifact: {args.output}")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    backends = {}
    for backend in ("langgraph", "llamaindex"):
        rows = [audit_backend(item, backend) for item in matrix["items"]]
        backends[backend] = {
            "passed": sum(row["checks_passed"] for row in rows),
            "failed": sum(not row["checks_passed"] for row in rows),
            "rows": rows,
        }
    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "matrix": str(args.matrix), "dataset_sha256": matrix.get("dataset_sha256"),
        "checks_passed": all(value["failed"] == 0 for value in backends.values()),
        "backends": backends,
        "category_outcomes": matrix.get("category_summary"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "checks_passed": artifact["checks_passed"],
        "backends": {name: {"passed": value["passed"], "failed": value["failed"]} for name, value in backends.items()},
    }, indent=2))
    raise SystemExit(0 if artifact["checks_passed"] else 1)


if __name__ == "__main__":
    main()
