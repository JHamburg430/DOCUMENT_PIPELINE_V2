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
            "citations": [{"chunk_id": "orientation"}],
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
