from manuals_rag_evals.agent_eval import AGENT_EVALUATION_LAYERS, score_agent_run


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
