from manuals_rag_evals.agent_eval import (
    AGENT_EVALUATION_LAYERS,
    _result_preserves_expected_evidence,
    score_agent_run,
)


def test_structured_equivalence_accepts_cross_page_duplicate_cell():
    expected = (
        "Column headers: Scaling Target; Row headers: Position X Minimum.Absolute Measured Value; "
        "Cell value: -; Row: 16; Column: 6"
    )
    result = {
        "source_document_id": "doc-a",
        "pages": [15],
        "content": expected.replace("Row: 16", "Row: 15"),
        "metadata": {"chunk_type": "table_record"},
    }

    assert _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={16},
        snippet=expected,
    )


def test_structured_equivalence_accepts_cross_page_exact_property_reference():
    expected = (
        'Image Enhance | Image Enhance | Input.ImageEnhancement | See "Image Enhance" '
        "Category: unrelated following row Input.Limit.Average.Max.Enable"
    )
    result = {
        "source_document_id": "doc-a",
        "pages": [1703],
        "content": (
            "Column headers: Options > Label; Row headers: Image Enhance > "
            'Input.ImageEnhancement; Cell value: Refer to "Image Enhance".; Row: 5; Column: 3'
        ),
        "metadata": {"chunk_type": "table_record"},
    }

    assert _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={1548},
        snippet=expected,
    )


def test_structured_equivalence_rejects_cross_page_different_cell_value():
    result = {
        "source_document_id": "doc-a",
        "pages": [22],
        "content": (
            "Column headers: Lower Limit Value; Row headers: Display Settings > Green > "
            "Input.Graphic.Region.Mask.ColorFail.Green; Cell value: 1; Row: 5; Column: 5"
        ),
        "metadata": {"chunk_type": "table_record"},
    }

    assert not _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={11},
        snippet=(
            "Column headers: Lower Limit Value; Row headers: Display Settings > Green > "
            "Input.Graphic.Region.Mask.ColorFail.Green; Cell value: 0; Row: 5; Column: 5"
        ),
    )


def _case():
    return {
        "case_id": "dependent",
        "query": "Which cable, then what orientation?",
        "retrieval_task": "multi_step_retrieval",
        "source_document_id": "doc-a",
        "expected_source_chunk_ids": ["identify", "orientation"],
        "expected_terms": ["op-26487", "straight"],
        "expected_evidence": [
            {"chunk_id": "identify", "source_document_id": "doc-a"},
            {"chunk_id": "orientation", "source_document_id": "doc-a"},
        ],
    }


def test_agent_evaluation_scores_each_requested_layer():
    trace = {
        "sufficient": True,
        "duration_ms": 1200,
        "plan": {
            "mode": "dependent",
            "hops": [
                {"hop_id": "one", "depends_on": []},
                {"hop_id": "two", "depends_on": ["one"]},
            ],
        },
        "evidence_ledger": {
            "one": {"required": True, "sufficient": True, "strategy": "sparse", "chunk_ids": ["identify"]},
            "two": {"required": True, "sufficient": True, "strategy": "structural", "chunk_ids": ["orientation"]},
        },
        "cost": {"retrieval_calls": 2, "llm_token_estimate": 300},
    }
    evaluation = score_agent_run(
        _case(),
        trace=trace,
        results=[{"chunk_id": "orientation", "source_document_id": "doc-a"}],
        answer={
            "answer": "The cable is OP-26487 and it is straight.",
            "citations": [{"chunk_id": "identify"}, {"chunk_id": "orientation"}],
        },
    )

    assert tuple(evaluation["cells"]) == AGENT_EVALUATION_LAYERS
    assert evaluation["passed"] is True


def test_agent_evaluation_separates_candidate_recall_from_final_context_retention():
    trace = {
        "sufficient": True,
        "plan": {"mode": "dependent", "hops": [{"hop_id": "one", "depends_on": []}, {"hop_id": "two", "depends_on": ["one"]}]},
        "evidence_ledger": {
            "one": {"required": True, "sufficient": True, "strategy": "sparse", "chunk_ids": ["identify"]},
            "two": {"required": True, "sufficient": True, "strategy": "structural", "chunk_ids": ["orientation"]},
        },
        "cost": {},
    }
    evaluation = score_agent_run(
        _case(),
        trace=trace,
        results=[{"chunk_id": "other", "source_document_id": "doc-b"}],
        answer={"answer": "OP-26487 is straight.", "citations": [{"chunk_id": "orientation"}]},
    )

    assert evaluation["cells"]["candidate_recall"]["status"] == "pass"
    assert evaluation["cells"]["document_retention"]["status"] == "fail"


def _quantity_case():
    return {
        "case_id": "quantity-bindings",
        "query": "What voltage and current should I set?",
        "retrieval_task": "single_step_retrieval",
        "source_document_id": "doc-controller",
        "source_chunk_id": "setup-values",
        "expected_terms": ["voltage", "5", "current", "10"],
        "expected_snippet": "Set voltage to 5 volts and current to 10 amps.",
        "expected_evidence": [
            {
                "chunk_id": "setup-values",
                "source_document_id": "doc-controller",
                "snippet": "Set voltage to 5 volts and current to 10 amps.",
                "expected_terms": ["voltage", "5", "current", "10"],
            }
        ],
    }


def _quantity_trace():
    return {
        "sufficient": True,
        "plan": {"mode": "single", "hops": [{"hop_id": "one", "depends_on": []}]},
        "evidence_ledger": {
            "one": {
                "required": True,
                "sufficient": True,
                "strategy": "structural",
                "chunk_ids": ["setup-values"],
            }
        },
        "cost": {},
    }


def test_agent_evaluation_rejects_swapped_quantity_role_bindings():
    evaluation = score_agent_run(
        _quantity_case(),
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "Set voltage to 10 amps and current to 5 volts.",
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    grounded = evaluation["cells"]["grounded_answer"]
    assert grounded["status"] == "fail"
    assert grounded["metrics"]["relation_grounding"]["checked"] is True
    assert grounded["metrics"]["relation_grounding"]["passed"] is False


def test_agent_evaluation_rejects_missing_required_quantity_role():
    evaluation = score_agent_run(
        _quantity_case(),
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={"answer": "Set voltage to 5 volts.", "citations": [{"chunk_id": "setup-values"}]},
    )

    assert evaluation["cells"]["grounded_answer"]["status"] == "fail"


def test_agent_evaluation_accepts_relation_preserving_quantity_bindings():
    evaluation = score_agent_run(
        _quantity_case(),
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "Set voltage to 5 volts and current to 10 amps.",
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"


def test_agent_evaluation_accepts_equivalent_quantity_unit_spelling():
    evaluation = score_agent_run(
        _quantity_case(),
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "Set voltage to 5 V and current to 10 A.",
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"


def test_agent_evaluation_rejects_unretrieved_irrelevant_citation():
    evaluation = score_agent_run(
        _quantity_case(),
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "Set voltage to 5 volts and current to 10 amps.",
            "citations": [{"chunk_id": "setup-values"}, {"chunk_id": "invented-neighbor"}],
        },
    )

    grounded = evaluation["cells"]["grounded_answer"]
    assert grounded["status"] == "fail"
    assert grounded["metrics"]["invalid_citation_chunks"] == ["invented-neighbor"]


def test_agent_evaluation_rejects_negated_action_against_affirmative_source():
    evaluation = score_agent_run(
        _quantity_case(),
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "Do not set voltage to 5 volts or current to 10 amps.",
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    relation = evaluation["cells"]["grounded_answer"]["metrics"]["relation_grounding"]
    assert evaluation["cells"]["grounded_answer"]["status"] == "fail"
    assert relation["polarity_mismatch"] is True


def test_agent_evaluation_ignores_relations_from_truncated_structured_cell():
    case = _quantity_case()
    case["expected_snippet"] = (
        "Column headers: Corrective Action; Row headers: Buffer full; "
        "Cell value: Change the RP"
    )
    case["expected_terms"] = ["change", "faster", "builds", "extend"]
    case["expected_evidence"] = [
        {
            "chunk_id": "setup-values",
            "source_document_id": "doc-controller",
            "field": "corrective action",
            "snippet": case["expected_snippet"],
            "expected_terms": case["expected_terms"],
        }
    ]
    evaluation = score_agent_run(
        case,
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": (
                "Change the RPI settings so output is faster than it builds, "
                "or extend the time between triggers."
            ),
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    grounded = evaluation["cells"]["grounded_answer"]
    assert grounded["status"] == "pass"
    assert grounded["metrics"]["relation_grounding"]["checked"] is False


def test_agent_evaluation_ignores_truncated_structured_row_header_without_cell():
    case = _quantity_case()
    case["expected_terms"] = ["reduce", "camera", "image", "set"]
    case["expected_evidence"] = [
        {
            "chunk_id": "setup-values",
            "source_document_id": "doc-controller",
            "field": "corrective action",
            "snippet": (
                "Column headers: Corrective Action; Row headers: The inspection "
                "region must be less than 2432 pixels in width and 2050 pixels "
            ),
            "expected_terms": case["expected_terms"],
        }
    ]
    evaluation = score_agent_run(
        case,
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "Reduce the camera image area or set a smaller inspection region.",
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    grounded = evaluation["cells"]["grounded_answer"]
    assert grounded["status"] == "pass"
    assert grounded["metrics"]["relation_grounding"]["checked"] is False


def test_agent_evaluation_requires_both_bounds_of_a_role_bound_range():
    case = _quantity_case()
    case["expected_snippet"] = "Voltage is 5 V to 10 V and current is 2 amps."
    case["expected_terms"] = ["voltage", "5", "10", "current", "2"]
    case["expected_evidence"][0]["snippet"] = case["expected_snippet"]
    case["expected_evidence"][0]["expected_terms"] = case["expected_terms"]
    evaluation = score_agent_run(
        case,
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "Voltage is 5 V and current is 2 amps; the menu has 10 items.",
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    relation = evaluation["cells"]["grounded_answer"]["metrics"]["relation_grounding"]
    assert evaluation["cells"]["grounded_answer"]["status"] == "fail"
    assert relation["missing_or_mismatched"]["voltage"]["expected"] == ["10 v", "5 v"]


def _parent_equivalence_case():
    return {
        "case_id": "parent-equivalence",
        "query": "Is OP-42284 used with CA-DRx9?",
        "retrieval_task": "single_step_retrieval",
        "source_document_id": "doc-light",
        "source_chunk_id": "exact-row",
        "page_from": 13,
        "page_to": 13,
        "expected_terms": ["op-42284", "ca-drx9"],
        "expected_snippet": 'Part number: 19.69" OP-42284; Applicable light: CA-DRx9',
    }


def _parent_equivalence_trace():
    return {
        "sufficient": True,
        "plan": {"mode": "single", "hops": [{"hop_id": "one", "depends_on": []}]},
        "evidence_ledger": {
            "one": {
                "required": True,
                "sufficient": True,
                "strategy": "structural",
                "chunk_ids": ["parent-table"],
            }
        },
        "cost": {},
    }


def test_agent_evaluation_accepts_same_page_parent_with_values_bound_on_one_row():
    evaluation = score_agent_run(
        _parent_equivalence_case(),
        trace=_parent_equivalence_trace(),
        results=[{
            "chunk_id": "parent-table",
            "source_document_id": "doc-light",
            "pages": [13],
            "content": 'Part number | Applicable light\n19.69" OP-42284 | CA-DRx9',
        }],
        answer={
            "answer": "OP-42284 is used with CA-DRx9.",
            "citations": [{"chunk_id": "parent-table"}],
        },
    )

    assert evaluation["cells"]["candidate_recall"]["status"] == "pass"
    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"


def test_agent_evaluation_rejects_parent_with_expected_values_scattered_across_rows():
    evaluation = score_agent_run(
        _parent_equivalence_case(),
        trace=_parent_equivalence_trace(),
        results=[{
            "chunk_id": "parent-table",
            "source_document_id": "doc-light",
            "pages": [13],
            "content": '19.69" OP-42284 | CA-DRx7\nOP-99999 | CA-DRx9',
        }],
        answer={
            "answer": "OP-42284 is used with CA-DRx9.",
            "citations": [{"chunk_id": "parent-table"}],
        },
    )

    assert evaluation["cells"]["candidate_recall"]["status"] == "fail"
    assert evaluation["cells"]["grounded_answer"]["status"] == "fail"


def test_agent_evaluation_rejects_exact_parent_row_on_wrong_page():
    evaluation = score_agent_run(
        _parent_equivalence_case(),
        trace=_parent_equivalence_trace(),
        results=[{
            "chunk_id": "parent-table",
            "source_document_id": "doc-light",
            "pages": [14],
            "content": '19.69" OP-42284 | CA-DRx9',
        }],
        answer={
            "answer": "OP-42284 is used with CA-DRx9.",
            "citations": [{"chunk_id": "parent-table"}],
        },
    )

    assert evaluation["cells"]["candidate_recall"]["status"] == "fail"


def test_agent_evaluation_accepts_exact_atomic_duplicate_on_another_page():
    case = _parent_equivalence_case()
    exact = case["expected_snippet"]
    trace = _parent_equivalence_trace()
    trace["evidence_ledger"]["one"]["chunk_ids"] = ["duplicate-atomic"]
    evaluation = score_agent_run(
        case,
        trace=trace,
        results=[{
            "chunk_id": "duplicate-atomic",
            "source_document_id": "doc-light",
            "pages": [14],
            "content": exact,
            "metadata": {"chunk_type": "atomic_text"},
        }],
        answer={
            "answer": "OP-42284 is used with CA-DRx9.",
            "citations": [{"chunk_id": "duplicate-atomic"}],
        },
    )

    assert evaluation["cells"]["candidate_recall"]["status"] == "pass"
    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"
