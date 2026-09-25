from manuals_rag_common.claim_relations import answer_relations_supported, relation_profile


def test_relation_profile_normalizes_common_manual_units():
    profile = relation_profile(
        "Set voltage to 24 millivolts, current to 5 milliamps, power to 2 watts, "
        "and resolution to 10 micrometers."
    )

    assert profile.role_values == {
        "voltage": frozenset({"24 mv"}),
        "current": frozenset({"5 ma"}),
        "power": frozenset({"2 w"}),
        "resolution": frozenset({"10 um"}),
    }
    assert profile.role_action_polarities == {
        "voltage": frozenset({"affirmative"}),
        "current": frozenset({"affirmative"}),
        "power": frozenset({"affirmative"}),
        "resolution": frozenset({"affirmative"}),
    }


def test_relation_profile_preserves_decimal_inch_distance_range():
    profile = relation_profile(
        'Installed distance 50 mm 1.97" to installed distance 500 mm 19.69".'
    )

    assert profile.role_values == {
        "distance": frozenset({"50 mm", "1.97 in", "500 mm", "19.69 in"})
    }


def test_role_bound_polarity_cannot_be_borrowed_from_another_citation():
    supported, details = answer_relations_supported(
        "Do not set voltage to 5 V.",
        [
            "Set voltage to 5 V.",
            "Do not increase the unrelated line count above 20.",
        ],
    )

    assert supported is False
    assert details["missing"] == {"voltage": ["5 v"]}


def test_complete_range_must_be_supported_in_one_evidence_item():
    supported, details = answer_relations_supported(
        "The voltage range is 5 V to 10 V.",
        ["The lower voltage limit is 5 V.", "An unrelated upper voltage limit is 10 V."],
    )

    assert supported is False
    assert sorted(next(iter(details["missing"].values()))) == ["10 v", "5 v"]


def test_affirmative_instruction_requires_action_evidence():
    supported, details = answer_relations_supported(
        "Set voltage to 24 V.",
        ["The nominal voltage is 24 V."],
    )

    assert supported is False
    assert details["polarity_supported"] is False


def test_compound_line_roles_remain_distinct_and_preserve_values():
    profile = relation_profile("Use 10 lines with 2 overlap lines.")

    assert profile.role_values == {
        "line": frozenset({"10 lines"}),
        "overlap_line": frozenset({"2 lines"}),
    }

    supported, _ = answer_relations_supported(
        "Use 10 lines with 2 overlap lines.",
        ["Use 10 lines with 2 overlap lines."],
    )
    swapped, _ = answer_relations_supported(
        "Use 2 lines with 10 overlap lines.",
        ["Use 10 lines with 2 overlap lines."],
    )
    assert supported is True
    assert swapped is False


def test_invented_action_identity_and_target_are_rejected():
    for answer in ("Remove encryption.", "Disable encryption."):
        supported, _ = answer_relations_supported(answer, ["Set voltage to 5 volts."])
        assert supported is False


def test_every_retained_action_target_requires_support():
    for answer, evidence in (
        ("Disable logging and encryption.", "Disable logging."),
        ("Disable encryption and logging.", "Disable logging."),
        ("Replace the damaged cable and sensor.", "Replace the damaged cable."),
        ("Replace the damaged cable.", "Replace the cable."),
    ):
        assert answer_relations_supported(answer, [evidence])[0] is False


def test_supported_multiple_targets_and_subset_remain_valid():
    evidence = "Disable logging and encryption."
    for answer in (evidence, "Disable encryption and logging.", "Disable logging."):
        assert answer_relations_supported(answer, [evidence])[0] is True


def test_targets_cannot_be_borrowed_from_another_action_or_chunk():
    answer = "Disable logging and encryption."
    for evidence in (
        ["Disable logging. Enable encryption."],
        ["Disable logging.", "Disable encryption."],
    ):
        assert answer_relations_supported(answer, evidence)[0] is False


def test_multi_action_claim_cannot_mix_separate_evidence_items():
    supported, details = answer_relations_supported(
        "Disable encryption and set voltage to 5 volts.",
        ["Set voltage to 5 volts.", "Disable logging."],
    )

    assert supported is False
    assert details["complete_profile_supported"] is False


def test_action_aliases_preserve_safe_semantics():
    supported, _ = answer_relations_supported(
        "Turn off logging.",
        ["Disable diagnostic logging."],
    )

    assert supported is True


def test_structured_table_binds_row_count_and_output_quantity_separately():
    profile = relation_profile(
        "Column headers: Quantity counted at one time; "
        "Row headers: ON when = Set value > Count value= 9; "
        "Cell value: 3; Row: 4; Column: 3"
    )

    assert profile.role_values["count"] == frozenset({"9"})
    assert profile.role_values["quantity"] == frozenset({"3"})
    assert profile.actions == frozenset()


def test_structured_table_rejects_wrong_quantity_for_matching_count():
    supported, details = answer_relations_supported(
        "When the count value is 9, the quantity counted at one time is 7.",
        [
            "Column headers: Quantity counted at one time; "
            "Row headers: ON when = Set value > Count value= 9; "
            "Cell value: 3; Row: 4; Column: 3"
        ],
    )

    assert supported is False
    assert details["missing"] == {"quantity": ["7"]}


def test_negative_condition_before_comma_does_not_negate_following_action():
    evidence = (
        "Status: Detection is performed with Contrast, but NG judgment is not given.; "
        "Corrective action: Increase the lower limit of Quality Match (%)."
    )
    answer = (
        "When Detection is performed with Contrast, but NG judgment is not given, "
        "increase the lower limit of Quality Match (%)."
    )

    supported, details = answer_relations_supported(answer, [evidence])

    assert supported is True
    assert details["answer_action_polarities"] == ["affirmative"]
