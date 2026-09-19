from scripts.benchmark.stress_claim_relation_guard import run


def test_relation_guard_stress_harness_covers_safe_and_unsafe_cases():
    report = run(cases=70, seed=7)

    assert report["checks_passed"] is True
    assert report["failures"] == 0
    assert set(report["categories"]) == {
        "safe_unit_paraphrase",
        "swapped_roles",
        "negation_inversion",
        "relocated_range_bound",
        "cross_citation_range",
        "unit_mismatch",
        "borrowed_unrelated_negation",
        "instruction_without_action_evidence",
        "compound_role_positive",
        "compound_role_swap",
        "invented_action_target",
        "cross_evidence_action_mix",
    }
