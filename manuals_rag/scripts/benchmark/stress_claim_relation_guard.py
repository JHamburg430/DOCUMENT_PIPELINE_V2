#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manuals_rag_common.claim_relations import answer_relations_supported


def _case(index: int, rng: random.Random) -> tuple[str, str, list[str], bool]:
    low = rng.randint(1, 12)
    high = low + rng.randint(1, 12)
    current = rng.randint(1, 20)
    category = index % 12
    if category == 0:
        return (
            "safe_unit_paraphrase",
            f"Set voltage to {low} V and current to {current} A.",
            [f"Set voltage to {low} volts and current to {current} amps."],
            True,
        )
    if category == 1:
        return (
            "swapped_roles",
            f"Set voltage to {current} A and current to {low} V.",
            [f"Set voltage to {low} V and current to {current} A."],
            False,
        )
    if category == 2:
        return (
            "negation_inversion",
            f"Do not set voltage to {low} V or current to {current} A.",
            [f"Set voltage to {low} V and current to {current} A."],
            False,
        )
    if category == 3:
        return (
            "relocated_range_bound",
            f"Voltage is {low} V and current is {current} A; the menu count is {high}.",
            [f"Voltage is {low} V to {high} V and current is {current} A."],
            False,
        )
    if category == 4:
        return (
            "cross_citation_range",
            f"Voltage is {low} V to {high} V.",
            [f"Voltage is {low} V.", f"Voltage is {high} V for a different model."],
            False,
        )
    if category == 5:
        return (
            "unit_mismatch",
            f"Voltage is {low} A.",
            [f"Voltage is {low} V."],
            False,
        )
    if category == 6:
        return (
            "borrowed_unrelated_negation",
            f"Do not set voltage to {low} V.",
            [f"Set voltage to {low} V.", f"Do not increase line count above {high}."],
            False,
        )
    if category == 7:
        return (
            "instruction_without_action_evidence",
            f"Set voltage to {low} V.",
            [f"The nominal voltage is {low} V."],
            False,
        )
    if category == 8:
        return (
            "compound_role_positive",
            f"Use {high} lines with {low} overlap lines.",
            [f"Use {high} lines with {low} overlap lines."],
            True,
        )
    if category == 9:
        return (
            "compound_role_swap",
            f"Use {low} lines with {high} overlap lines.",
            [f"Use {high} lines with {low} overlap lines."],
            False,
        )
    if category == 10:
        return (
            "invented_action_target",
            "Disable encryption.",
            [f"Set voltage to {low} V."],
            False,
        )
    return (
        "cross_evidence_action_mix",
        f"Disable encryption and set voltage to {low} V.",
        [f"Set voltage to {low} V.", "Disable logging."],
        False,
    )


def run(*, cases: int = 5000, seed: int = 20260911, alpha: float = 0.05) -> dict[str, Any]:
    if cases < 1:
        raise ValueError("cases must be positive")
    rng = random.Random(seed)
    failures: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for index in range(cases):
        category, answer, evidence, expected = _case(index, rng)
        categories[category] += 1
        actual, details = answer_relations_supported(answer, evidence)
        if actual != expected:
            failures.append(
                {
                    "index": index,
                    "category": category,
                    "answer": answer,
                    "evidence": evidence,
                    "expected_supported": expected,
                    "actual_supported": actual,
                    "details": details,
                }
            )
    failure_count = len(failures)
    zero_failure_upper_bound = 1.0 - alpha ** (1.0 / cases) if failure_count == 0 else None
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "cases": cases,
        "categories": dict(categories),
        "failures": failure_count,
        "checks_passed": failure_count == 0,
        "one_sided_confidence": 1.0 - alpha,
        "zero_failure_upper_bound": zero_failure_upper_bound,
        "sample_failures": failures[:20],
        "scope_note": (
            "Deterministic adversarial relation-guard stress test; this is not a substitute "
            "for independent human-adjudicated end-to-end questions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress the claim/evidence relation safety guard.")
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(cases=args.cases, seed=args.seed, alpha=args.alpha)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["checks_passed"] else 1)


if __name__ == "__main__":
    main()
