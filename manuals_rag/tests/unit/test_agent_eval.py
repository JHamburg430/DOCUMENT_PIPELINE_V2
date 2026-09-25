from manuals_rag_evals.agent_eval import (
    AGENT_EVALUATION_LAYERS,
    _equivalent_chunk_ids,
    _result_preserves_expected_evidence,
    score_agent_run,
)


def test_equivalence_accepts_long_verbatim_passage_from_another_manual_edition():
    snippet = (
        "Install this product so that the path of the laser beam is not at the same "
        "height as that of human eye."
    )
    result = {
        "source_document_id": "user-manual-edition",
        "pages": [8],
        "content": f"Safety precautions. {snippet} Follow all local requirements.",
        "metadata": {"chunk_type": "atomic_text"},
    }

    assert _result_preserves_expected_evidence(
        result,
        source_document_id="instruction-manual-edition",
        expected_pages={2},
        snippet=snippet,
    )


def test_equivalence_rejects_short_cross_document_numeric_match():
    assert not _result_preserves_expected_evidence(
        {
            "source_document_id": "different-model-manual",
            "pages": [4],
            "content": "Operating ambient temperature: 0 to 40 degrees C",
            "metadata": {"chunk_type": "table_record"},
        },
        source_document_id="expected-model-manual",
        expected_pages={4},
        snippet="Operating ambient temperature: 0 to 40 degrees C",
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


def test_structured_equivalence_accepts_compact_row_group_and_normalized_cell():
    result = {
        "source_document_id": "doc-a",
        "pages": [3],
        "content": (
            "Column headers: LJ-S015; Row headers: Measurement range (Z); "
            "Cell value: ±4 mm (F.S. = 8 mm); Row: 2; Column: 2"
        ),
        "metadata": {"chunk_type": "table_record"},
    }

    assert _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={3},
        snippet="LJ-S015: ±4 mm (F.S. = 8 mm)",
    )


def test_structured_equivalence_rejects_compact_row_group_with_different_value():
    result = {
        "source_document_id": "doc-a",
        "pages": [3],
        "content": (
            "Column headers: LJ-S015; Row headers: Measurement range (Z); "
            "Cell value: ±9 mm (F.S. = 18 mm); Row: 2; Column: 2"
        ),
        "metadata": {"chunk_type": "table_record"},
    }

    assert not _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={3},
        snippet="LJ-S015: ±4 mm (F.S. = 8 mm)",
    )


def test_structured_equivalence_accepts_exact_value_for_nested_column_path():
    result = {
        "source_document_id": "doc-a",
        "pages": [45],
        "content": (
            "Column headers: CV-X472 > CV-X452; "
            "Row headers: Control input > External trigger input; "
            "Cell value: 4 points (2 of which support special function assignment) "
            "Input rating: 26.4 V max., 3 mAmin, can select from simultaneous/individual "
            "capture with up to 4 cameras.; Row: 1; Column: 3"
        ),
        "metadata": {"chunk_type": "table_record"},
    }

    assert _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={45},
        snippet=(
            "CV-X452: 4 points (2 of which support special function assignment) "
            "Input rating: 26.4 V max., 3 mAmin, can select from simultaneous/individual "
            "capture with up to 4 cameras."
        ),
    )


def test_structured_equivalence_accepts_pipe_row_group_and_normalized_cell():
    result = {
        "source_document_id": "doc-a",
        "pages": [1],
        "content": (
            "Column headers: CA-U5; Row headers: Input conditions > Rated input voltage; "
            "Cell value: 85 to 264 VAC 47 to 63 Hz, 110 to 370 VDC *1; Row: 1; Column: 2"
        ),
        "metadata": {"chunk_type": "table_record"},
    }

    assert _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={1},
        snippet=(
            "Input conditions | Rated input voltage | 85 to 264 VAC 47 to 63 Hz, "
            "110 to 370 VDC *1\n"
            "Input conditions | Rated input current | 3.9 A max."
        ),
    )


def test_structured_equivalence_rejects_pipe_row_group_neighboring_cell():
    result = {
        "source_document_id": "doc-a",
        "pages": [1],
        "content": (
            "Column headers: CA-U5; Row headers: Input conditions > Rated input current; "
            "Cell value: 3.9 A max.; Row: 3; Column: 2"
        ),
        "metadata": {"chunk_type": "table_record"},
    }

    assert not _result_preserves_expected_evidence(
        result,
        source_document_id="doc-a",
        expected_pages={1},
        snippet=(
            "Input conditions | Rated input voltage | 85 to 264 VAC 47 to 63 Hz, "
            "110 to 370 VDC *1"
        ),
    )


def test_structured_equivalence_accepts_query_qualified_matrix_cell():
    case = {
        "query": "How many protection zones does the SZ-V04 multi-function model support?",
        "source_document_id": "doc-a",
        "source_chunk_id": "matrix-row",
        "page_from": 20,
        "page_to": 20,
        "expected_snippet": (
            "Protection zone | ✓ 2 zones | ✓ 1 zone | ✓ 1 zone | "
            "✓ 2 zones | ✓ 2 zones"
        ),
    }
    results = [
        {
            "chunk_id": "sz-v04-cell",
            "source_document_id": "doc-a",
            "pages": [20],
            "content": (
                "Column headers: SZ-V04 (X) > Multi-function; "
                "Row headers: Protection zone; Cell value: ✓ 2 zones; "
                "Row: 2; Column: 2"
            ),
            "metadata": {"chunk_type": "table_record"},
        }
    ]

    assert _equivalent_chunk_ids(case, results)["matrix-row"] == {"sz-v04-cell"}


def test_structured_equivalence_rejects_neighboring_matrix_model():
    case = {
        "query": "How many protection zones does the SZ-V04 multi-function model support?",
        "source_document_id": "doc-a",
        "source_chunk_id": "matrix-row",
        "page_from": 20,
        "page_to": 20,
        "expected_snippet": (
            "Protection zone | ✓ 2 zones | ✓ 1 zone | ✓ 1 zone | "
            "✓ 2 zones | ✓ 2 zones"
        ),
    }
    results = [
        {
            "chunk_id": "sz-v32-cell",
            "source_document_id": "doc-a",
            "pages": [20],
            "content": (
                "Column headers: SZ-V32 (X) > Multi-bank; "
                "Row headers: Protection zone; Cell value: ✓ 1 zone; "
                "Row: 2; Column: 3"
            ),
            "metadata": {"chunk_type": "table_record"},
        }
    ]

    assert _equivalent_chunk_ids(case, results)["matrix-row"] == set()


def test_cross_document_equivalence_does_not_inherit_first_document_pages():
    case = {
        "source_document_id": "first-doc",
        "page_from": 1963,
        "page_to": 1963,
        "expected_evidence": [
            {
                "chunk_id": "expected-cause",
                "source_document_id": "second-doc",
                "field": "cause",
                "expected_terms": ["memory", "error", "occurred", "sensor"],
                "snippet": "Column headers: Cause; Row headers: Sensor program damaged.",
            }
        ],
    }
    results = [
        {
            "chunk_id": "equivalent-cause",
            "source_document_id": "second-doc",
            "pages": [406],
            "content": (
                "Column headers: Cause; Row headers: Sensor internal memory reading has failed; "
                "Cell value: A memory read error occurred when the sensor started."
            ),
            "metadata": {"chunk_type": "table_record"},
        }
    ]

    assert _equivalent_chunk_ids(case, results)["expected-cause"] == {"equivalent-cause"}


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


def test_agent_evaluation_accepts_retained_long_verbatim_duplicate_edition():
    snippet = "Short: range zoom type for short-range or space-saving installation needs"
    case = {
        "case_id": "duplicate-edition",
        "query": "Which zoom type is intended for short-range or space-saving installation?",
        "retrieval_task": "single_step_retrieval",
        "source_document_id": "instruction-manual",
        "source_chunk_id": "expected-warning",
        "page_from": 2,
        "page_to": 2,
        "expected_snippet": snippet,
        "expected_terms": ["short", "range", "zoom", "space", "saving", "installation"],
    }
    trace = {
        "sufficient": True,
        "plan": {"mode": "single", "hops": [{"hop_id": "one", "depends_on": []}]},
        "evidence_ledger": {
            "one": {
                "required": True,
                "sufficient": True,
                "strategy": "hybrid",
                "chunk_ids": ["duplicate-warning"],
            }
        },
        "cost": {},
    }
    evaluation = score_agent_run(
        case,
        trace=trace,
        results=[
            {
                "chunk_id": "duplicate-warning",
                "source_document_id": "user-manual",
                "pages": [8],
                "content": snippet,
                "metadata": {"chunk_type": "atomic_text"},
            }
        ],
        answer={
            "answer": "Use the Short range zoom type for short-range or space-saving installation.",
            "citations": [{"chunk_id": "duplicate-warning"}],
        },
    )

    assert evaluation["passed"] is True
    assert evaluation["cells"]["document_retention"]["status"] == "pass"
    assert evaluation["cells"]["document_retention"]["metrics"][
        "verbatim_equivalent_documents"
    ] == ["instruction-manual"]


def test_agent_evaluation_accepts_baseline_proven_cross_document_semantic_evidence():
    snippet = "Environmental resistance Operating ambient temperature 0 to +50°C (No freezing)"
    case = {
        "case_id": "iv4-temperature-duplicate-manual",
        "query": "What operating ambient temperature range is allowed for the IV4 Series without freezing?",
        "retrieval_task": "single_step_retrieval",
        "source_document_id": "iv4-source-manual",
        "source_chunk_id": "expected-temperature",
        "document_version_id": "iv4-source-version",
        "source_title": "IV4 Source Manual",
        "source_filename": "iv4-source.pdf",
        "chunk_type": "table_record",
        "section_path": "Specifications",
        "page_from": 2,
        "page_to": 2,
        "expected_terms": ["environmental", "resistance", "operating", "ambient", "0", "50"],
        "expected_snippet": snippet,
        "generation_method": "unit",
        "source_metadata": {"product_family": "IV4 Series"},
    }
    trace = {
        "sufficient": True,
        "plan": {"mode": "single", "hops": [{"hop_id": "one", "depends_on": []}]},
        "evidence_ledger": {
            "one": {
                "required": True,
                "sufficient": True,
                "strategy": "hybrid",
                "chunk_ids": ["duplicate-temperature"],
            }
        },
        "cost": {},
    }
    evaluation = score_agent_run(
        case,
        trace=trace,
        results=[
            {
                "chunk_id": "duplicate-temperature",
                "source_document_id": "iv4-duplicate-manual",
                "section_path": ["Specifications"],
                "content": snippet,
                "metadata": {"chunk_type": "table_record", "product_family": "IV4 Series"},
            }
        ],
        answer={
            "answer": (
                "For environmental resistance, the operating ambient temperature is 0 to "
                "+50°C, with no freezing."
            ),
            "citations": [{"chunk_id": "duplicate-temperature"}],
        },
    )

    assert evaluation["passed"] is True
    assert evaluation["cells"]["candidate_recall"]["status"] == "pass"
    assert evaluation["cells"]["document_retention"]["status"] == "pass"
    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"


def test_agent_evaluation_accepts_baseline_proven_applicable_table_evidence():
    snippet = (
        "Protection circuit | Protection against reverse power connection, power "
        "supply surge, output overcurrent, output surge, and reverse output connection"
    )
    case = {
        "case_id": "lr-t-protection-duplicate",
        "query": "Which protection circuits safeguard the LR-T laser sensor against reverse power connection and output surges?",
        "retrieval_task": "single_step_retrieval",
        "source_document_id": "lr-t-source",
        "source_chunk_id": "expected-protection",
        "document_version_id": "lr-t-version",
        "source_title": "LR-T Laser Sensor",
        "source_filename": "lr-t.pdf",
        "chunk_type": "table_record",
        "section_path": "Specifications",
        "page_from": 17,
        "page_to": 17,
        "expected_terms": ["protection", "circuit", "against", "reverse"],
        "expected_snippet": snippet,
        "generation_method": "unit",
        "source_metadata": {
            "product_family": "Laser Sensor",
            "identifier_tokens": ["LR-T", "LR-TB2000", "LR-TB5000"],
        },
    }
    duplicate_chunk = "duplicate-protection"
    trace = {
        "sufficient": True,
        "plan": {"mode": "single", "hops": [{"hop_id": "one", "depends_on": []}]},
        "evidence_ledger": {
            "one": {
                "required": True,
                "sufficient": True,
                "strategy": "hybrid",
                "chunk_ids": [duplicate_chunk],
            }
        },
        "cost": {},
    }
    evaluation = score_agent_run(
        case,
        trace=trace,
        results=[
            {
                "chunk_id": duplicate_chunk,
                "source_document_id": "lr-t-duplicate",
                "section_path": ["Specifications"],
                "content": (
                    "Column headers: Model > LR-TB5000 > LR-TB2000; Cell value: "
                    "Protection circuit Protection against reverse power connection, "
                    "power supply surges, output overcurrent, reverse output connection, "
                    "and output surge; Row: 14; Column: 0"
                ),
                "metadata": {
                    "chunk_type": "table_record",
                    "product_family": "Laser Sensor",
                    "identifier_tokens": ["LR-T", "LR-TB2000", "LR-TB5000"],
                },
            }
        ],
        answer={
            "answer": "The protection circuit guards against reverse power connection and output surge.",
            "citations": [{"chunk_id": duplicate_chunk}],
        },
    )

    assert evaluation["passed"] is True
    assert evaluation["cells"]["candidate_recall"]["status"] == "pass"
    assert evaluation["cells"]["document_retention"]["status"] == "pass"
    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"


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


def test_agent_evaluation_ignores_incidental_imperative_for_factual_value_lookup():
    case = _quantity_case()
    case["query"] = "What screw torque specification applies to the RS-422 terminals?"
    case["expected_snippet"] = (
        "Connect the included RS-422 cable to the terminal block "
        "(terminal block screw torque: 0.25 Nm or less)."
    )
    case["expected_terms"] = ["rs-422", "torque", "0.25"]
    case["expected_evidence"][0]["snippet"] = case["expected_snippet"]
    case["expected_evidence"][0]["expected_terms"] = case["expected_terms"]

    evaluation = score_agent_run(
        case,
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": "The RS-422 terminal-block screw torque is 0.25 Nm or less.",
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    relation = evaluation["cells"]["grounded_answer"]["metrics"]["relation_grounding"]
    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"
    assert relation["passed"] is True


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


def test_agent_evaluation_accepts_answer_with_expected_action_and_extra_targets():
    case = _quantity_case()
    case["expected_snippet"] = "Do not use this product in a hazardous location."
    case["expected_terms"] = ["not", "use", "hazardous", "location"]
    case["expected_evidence"][0]["snippet"] = case["expected_snippet"]
    case["expected_evidence"][0]["expected_terms"] = case["expected_terms"]

    evaluation = score_agent_run(
        case,
        trace=_quantity_trace(),
        results=[{"chunk_id": "setup-values", "source_document_id": "doc-controller"}],
        answer={
            "answer": (
                "Do not use this product as explosion-proof equipment in a hazardous "
                "location or potentially explosive atmosphere."
            ),
            "citations": [{"chunk_id": "setup-values"}],
        },
    )

    relation = evaluation["cells"]["grounded_answer"]["metrics"]["relation_grounding"]
    assert relation["passed"] is True
    assert relation["action_target_superset_accepted"] is True


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


def test_agent_evaluation_accepts_same_document_equivalent_with_case_terms():
    case = {
        **_parent_equivalence_case(),
        "expected_terms": ["200", "including", "cable", "connector"],
        "expected_snippet": "Approx. 200 g including cable with connector.",
    }
    trace = _parent_equivalence_trace()
    trace["evidence_ledger"]["one"]["chunk_ids"] = ["equivalent-spec"]
    evaluation = score_agent_run(
        case,
        trace=trace,
        results=[{
            "chunk_id": "equivalent-spec",
            "source_document_id": "doc-light",
            "pages": [15],
            "content": "Weight: 200 g, including the cable and connector.",
            "metadata": {"chunk_type": "spec_record"},
        }],
        answer={
            "answer": "The weight is 200 g including the cable and connector.",
            "citations": [{"chunk_id": "equivalent-spec"}],
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


def test_agent_evaluation_accepts_section_window_with_exact_model_and_value_row():
    case = {
        "case_id": "section-window-equivalence",
        "query": "How long does image transfer take for VJ-H500CX in 5 megapixel mode?",
        "retrieval_task": "single_step_retrieval",
        "source_document_id": "doc-vj",
        "source_chunk_id": "atomic-transfer-time",
        "page_from": 1,
        "page_to": 1,
        "expected_terms": ["vj-h500cx", "megapixel", "29.2", "11.7"],
        "expected_snippet": (
            "VJ-H500CX: 5 megapixel mode: 29.2 ms "
            "2 megapixel mode: 11.7 ms"
        ),
    }
    trace = _parent_equivalence_trace()
    trace["evidence_ledger"]["one"]["chunk_ids"] = ["section-window"]
    evaluation = score_agent_run(
        case,
        trace=trace,
        results=[{
            "chunk_id": "section-window",
            "source_document_id": "doc-vj",
            "pages": [1],
            "content": (
                "Model | | VJ-H500CX\n"
                "Transfer time | | 5 megapixel mode: 29.2 ms "
                "2 megapixel mode: 11.7 ms\n"
                "Electronic shutter | | 0.017 msec to 100 msec"
            ),
            "metadata": {"chunk_type": "section_window"},
        }],
        answer={
            "answer": (
                "For VJ-H500CX, transfer time is 29.2 ms in 5 megapixel mode "
                "and 11.7 ms in 2 megapixel mode."
            ),
            "citations": [{"chunk_id": "section-window"}],
        },
    )

    assert evaluation["cells"]["candidate_recall"]["status"] == "pass"
    assert evaluation["cells"]["grounded_answer"]["status"] == "pass"
