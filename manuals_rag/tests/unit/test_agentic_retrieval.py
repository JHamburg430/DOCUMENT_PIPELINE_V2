from manuals_rag_answering.agentic_retrieval import (
    AgenticRetrievalController,
    LlamaIndexAgenticController,
    RetrievalHop,
    RetrievalPlan,
    build_langgraph_agentic_retriever,
    build_llamaindex_agentic_retriever,
    plan_retrieval,
    plan_llamaindex_retrieval,
    refine_dependent_query,
    insufficient_agent_answer,
    _assess_hop_evidence,
    verify_retrieval_claim,
)
from manuals_rag_schemas.documents import SearchResult


def _result(chunk_id: str, document_id: str, content: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=1.0,
        title=f"Manual {document_id}",
        document_version_id=f"version-{document_id}",
        source_document_id=document_id,
        pages=[1],
        section_path=["Troubleshooting", "Corrective action"],
        content=content,
        metadata={"chunk_type": "table_record"},
    )


def _invoke(factory, controller, *, max_hops: int = 4):
    return factory(controller=controller).invoke(
        {
            "query": "Compare the corrective actions for ALPHA-1 and BETA-2.",
            "corpus_ids": ["manuals"],
            "filters": {},
            "max_hops": max_hops,
        }
    )


def test_heuristic_planner_decomposes_named_scopes_and_pairs_requested_details():
    plan = plan_retrieval(
        "For Laser Sensor and LJ: X8000 Series, what weight entries are listed for "
        "pvc (polyvinyl chloride) m12 4-pin straight connector and cb-b5e?",
        use_llm=False,
    )

    assert plan.mode == "parallel"
    assert [hop.query for hop in plan.hops] == [
        "For Laser Sensor, what weight entries are listed for pvc (polyvinyl chloride) m12 4-pin straight connector?",
        "For LJ: X8000 Series, what weight entries are listed for cb-b5e?",
    ]


def test_heuristic_planner_decomposes_troubleshooting_facets():
    plan = plan_retrieval(
        "What causes alarm E17 for ZX-9, and how should it be corrected?",
        use_llm=False,
    )

    assert plan.mode == "parallel"
    assert [hop.hop_id for hop in plan.hops] == ["cause", "corrective_action"]
    assert plan.hops[0].query == "What causes alarm E17 for ZX-9?"
    assert plan.hops[1].query == "How should alarm E17 for ZX-9 be corrected?"


def test_model_planners_enforce_parallel_branches_for_colon_delimited_product_comparison(monkeypatch):
    query = "Compare the VJ-H500CX weight with grayscale settings for LJ:S8000."

    def fail_if_called(**_kwargs):
        raise AssertionError("deterministic scope safety must run before model planning")

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", fail_if_called)

    langgraph = plan_retrieval(query, use_llm=True)
    llamaindex = plan_llamaindex_retrieval(query, use_llm=True)

    assert langgraph.mode == "parallel"
    assert [hop.hop_id for hop in langgraph.hops] == ["side_1", "side_2"]
    assert all(identifier in langgraph.hops[index].query for index, identifier in enumerate(["VJ-H500CX", "LJ:S8000"]))
    assert llamaindex.mode == "parallel"
    assert [hop.hop_id for hop in llamaindex.hops] == ["subquestion_1", "subquestion_2"]
    assert langgraph.hops[0].query == "What is the VJ-H500CX weight?"
    assert langgraph.hops[1].query == "What is grayscale settings for LJ:S8000?"


def test_model_planners_split_independent_interrogative_facets(monkeypatch):
    query = "For KV-X Series, which software is listed and what upgrade benefit is stated?"

    def fail_if_called(**_kwargs):
        raise AssertionError("claim-facet safety must run before model planning")

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", fail_if_called)

    langgraph = plan_retrieval(query, use_llm=True)
    llamaindex = plan_llamaindex_retrieval(query, use_llm=True)

    assert [hop.query for hop in langgraph.hops] == [
        "For KV-X Series, which software is listed?",
        "For KV-X Series, what upgrade benefit is stated for the software?",
    ]
    assert [hop.hop_id for hop in llamaindex.hops] == ["subquestion_1", "subquestion_2"]
    assert all(hop.strategy == "hybrid" for hop in llamaindex.hops)


def test_backends_have_independent_default_planning_policies():
    query = "Which cable model connects the port, then what is that cable's orientation?"

    langgraph_plan = plan_retrieval(query, use_llm=False)
    llamaindex_plan = plan_llamaindex_retrieval(query, use_llm=False)

    assert [hop.hop_id for hop in langgraph_plan.hops] == ["hop_1", "hop_2"]
    assert [hop.hop_id for hop in llamaindex_plan.hops] == ["subquestion_1", "subquestion_2"]
    assert llamaindex_plan.rationale.startswith("LlamaIndex subquestion decomposition")
    assert AgenticRetrievalController(use_llm=False).__class__ is not LlamaIndexAgenticController


def test_both_orchestrators_preserve_parallel_document_coverage():
    plan = RetrievalPlan(
        mode="parallel",
        hops=[
            RetrievalHop(
                hop_id="alpha",
                objective="Find ALPHA-1 corrective action",
                query="ALPHA-1 corrective action",
                strategy="structural",
            ),
            RetrievalHop(
                hop_id="beta",
                objective="Find BETA-2 corrective action",
                query="BETA-2 corrective action",
                strategy="sparse",
            ),
        ],
    )

    def retrieve(query, _corpus_ids, _filters, strategy, _limit):
        if "ALPHA" in query:
            assert strategy == "structural"
            return [_result("alpha-chunk", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")]
        assert strategy == "sparse"
        return [_result("beta-chunk", "beta-doc", "Corrective action: recalibrate the BETA-2 sensor.")]

    outputs = []
    for factory in (build_langgraph_agentic_retriever, build_llamaindex_agentic_retriever):
        controller = AgenticRetrievalController(
            use_llm=False,
            planner=lambda _query: plan,
            retriever=retrieve,
        )
        outputs.append(_invoke(factory, controller))

    for output in outputs:
        assert output["sufficient"] is True
        assert output["stop_reason"] == "sufficient"
        assert {item["source_document_id"] for item in output["retrieval_results"]} == {
            "alpha-doc",
            "beta-doc",
        }
        assert output["retrieval_trace"]["completed_hops"] == ["alpha", "beta"]
    assert [item["chunk_id"] for item in outputs[0]["retrieval_results"]] == [
        item["chunk_id"] for item in outputs[1]["retrieval_results"]
    ]


def test_controller_emits_live_plan_hop_and_completion_events():
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="lookup",
                objective="Find ALPHA-1 corrective action",
                query="ALPHA-1 corrective action",
                strategy="structural",
            )
        ]
    )
    events: list[dict] = []
    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=lambda *_args: [
            _result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")
        ],
        event_callback=events.append,
    )

    output = _invoke(build_langgraph_agentic_retriever, controller)

    assert output["sufficient"] is True
    assert [event["event"] for event in events] == [
        "plan_completed",
        "hop_started",
        "claim_verified",
        "hop_completed",
        "retrieval_completed",
    ]
    assert events[1]["executed_query"] == "ALPHA-1 corrective action"
    assert events[2]["trust_state"] == "confirmed"
    assert events[3]["results"][0]["chunk_id"] == "alpha"
    assert events[4]["trace"]["stop_reason"] == "sufficient"
    assert events[4]["trace"]["pipeline"] == "evidence_map_reduce_verify_v1"


def test_agent_stops_safely_when_runtime_budget_is_exhausted(monkeypatch):
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="lookup",
                objective="Find ALPHA-1 corrective action",
                query="ALPHA-1 corrective action",
            )
        ]
    )
    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=lambda *_args: (_ for _ in ()).throw(AssertionError("retrieval must not run")),
    )
    ticks = iter([100.0, 106.0])
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.perf_counter",
        lambda: next(ticks, 106.0),
    )

    output = build_langgraph_agentic_retriever(controller=controller).invoke(
        {
            "query": "ALPHA-1 corrective action",
            "corpus_ids": ["manuals"],
            "filters": {},
            "max_hops": 4,
            "max_seconds": 5,
        }
    )

    assert output["sufficient"] is False
    assert output["stop_reason"] == "runtime_budget_exhausted"
    assert output["retrieval_trace"]["max_seconds"] == 5.0
    assert output["retrieval_results"] == []


def test_retrieval_failure_is_contained_in_claim_ledger():
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="lookup",
                objective="Find ALPHA-1 corrective action",
                query="ALPHA-1 corrective action",
            )
        ]
    )
    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=lambda *_args: (_ for _ in ()).throw(RuntimeError("qdrant unavailable")),
    )

    output = build_langgraph_agentic_retriever(controller=controller).invoke(
        {
            "query": "ALPHA-1 corrective action",
            "corpus_ids": ["manuals"],
            "filters": {},
            "max_hops": 1,
        }
    )

    assessment = output["retrieval_trace"]["evidence_ledger"]["lookup"]["assessment"]
    assert output["sufficient"] is False
    assert output["stop_reason"] == "hop_budget_exhausted"
    assert assessment["gap_reason"] == "verification_unresolved"
    assert assessment["retrieval_error"] == "RuntimeError: qdrant unavailable"


def test_verifier_rejects_model_citations_that_were_not_retrieved(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )

    def fake_chat_json(**_kwargs):
        return (
            {
                "trust_state": "confirmed",
                "claim_supported": True,
                "supporting_chunk_ids": ["invented-chunk"],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": "ALPHA-1",
                "rationale": "Claimed support.",
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", fake_chat_json)
    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("real-chunk", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        {"claim_supported": True, "supporting_chunk_ids": ["real-chunk"]},
    )

    assert result["claim_supported"] is False
    assert result["trust_state"] == "unresolved"
    assert result["invalid_citation_ids"] == ["invented-chunk"]


def test_verifier_normalizes_compact_supported_response(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "reasoning": "The cited chunk directly states the action.",
            },
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert result["trust_state"] == "confirmed"
    assert result["claim_supported"] is True
    assert result["supporting_chunk_ids"] == ["alpha"]
    assert result["rationale"] == "The cited chunk directly states the action."


def test_verifier_reconciles_internally_inconsistent_affirmative_response(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "trust_state": "unresolved",
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": "ALPHA-1",
                "rationale": "The cited chunk directly supports the claim.",
            },
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert result["trust_state"] == "confirmed"
    assert result["claim_supported"] is True


def test_verifier_normalizes_verdict_alias_response(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "verdict": "confirmed",
                "supporting_chunk_ids": ["alpha"],
                "reasoning": "The cited chunk directly supports the claim.",
            },
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert result["trust_state"] == "confirmed"
    assert result["claim_supported"] is True
    assert result["supporting_chunk_ids"] == ["alpha"]


def test_verifier_retries_once_after_malformed_model_response(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    calls = 0

    def flaky_chat_json(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Invalid JSON escape")
        return (
            {
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "reasoning": "The cited chunk directly supports the claim.",
            },
            "{}",
        )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        flaky_chat_json,
    )
    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert calls == 2
    assert result["trust_state"] == "confirmed"
    assert result["claim_supported"] is True


def test_coordinate_plan_preserves_first_branch_subject_in_second_claim():
    query = (
        "For the controller, which integrated software is listed "
        "and what upgrade benefit is stated?"
    )

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query, use_llm=False)
        assert len(plan.hops) == 2
        assert plan.hops[0].query == "For the controller, which integrated software is listed?"
        assert plan.hops[1].query == (
            "For the controller, what upgrade benefit is stated for the integrated software?"
        )


def test_verifier_treats_attributed_support_list_as_compact_affirmative_verdict(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "supporting_chunk_ids": ["alpha"],
                "reasoning": "The cited chunk directly supports the claim.",
            },
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert result["trust_state"] == "confirmed"
    assert result["claim_supported"] is True


def test_verifier_does_not_promote_support_list_when_deterministic_gate_disagrees(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {"supporting_chunk_ids": ["alpha"], "reasoning": "Suggestive evidence."},
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "ALPHA-1 overview only.")],
        {"claim_supported": False, "supporting_chunk_ids": []},
    )

    assert result["trust_state"] == "unresolved"
    assert result["claim_supported"] is False


def test_deterministic_verifier_keeps_evidence_bound_to_branch_scope():
    hop = RetrievalHop(
        hop_id="alpha",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    alpha = _result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")
    beta = _result("beta", "beta-doc", "Corrective action: recalibrate the BETA-2 sensor.")

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [alpha, beta],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha", "beta"]},
        use_llm=False,
    )

    assert result["trust_state"] == "confirmed"
    assert result["supporting_chunk_ids"] == ["alpha"]


def test_probable_verification_cannot_unlock_required_claim():
    plan = RetrievalPlan(
        hops=[RetrievalHop(hop_id="lookup", objective="Find ALPHA-1 corrective action", query="ALPHA-1 corrective action")]
    )
    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=lambda *_args: [_result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        verifier=lambda *_args: {
            "trust_state": "probable",
            "claim_supported": False,
            "supporting_chunk_ids": ["alpha"],
            "conflicting_chunk_ids": [],
            "applicability": "unknown",
            "scope_entity": "ALPHA-1",
            "rationale": "Scope is not independently established.",
        },
    )

    output = _invoke(build_langgraph_agentic_retriever, controller, max_hops=1)

    assert output["sufficient"] is False
    assert output["evidence_ledger"]["lookup"]["assessment"]["trust_state"] == "probable"
    assert output["retrieval_trace"]["required_claim_support"]["lookup"] == []


def test_dependent_hop_is_refined_from_prior_evidence():
    plan = RetrievalPlan(
        mode="dependent",
        hops=[
            RetrievalHop(
                hop_id="identify_component",
                objective="Identify the component associated with alarm E17",
                query="Which component is associated with alarm E17?",
                strategy="sparse",
            ),
            RetrievalHop(
                hop_id="find_tolerance",
                objective="Find that component's calibration tolerance",
                query="What is the calibration tolerance for the identified component?",
                strategy="structural",
                depends_on=["identify_component"],
            ),
        ],
    )
    executed_queries: list[str] = []

    def refine(hop, results):
        assert hop.hop_id == "find_tolerance"
        assert "ZX-9 pressure transducer" in results[0].content
        return "ZX-9 pressure transducer calibration tolerance"

    def retrieve(query, _corpus_ids, _filters, _strategy, _limit):
        executed_queries.append(query)
        if "E17" in query:
            return [_result("component", "alarm-doc", "Alarm E17 identifies the ZX-9 pressure transducer.")]
        return [_result("tolerance", "calibration-doc", "ZX-9 calibration tolerance is 0.2 percent.")]

    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        refiner=refine,
        retriever=retrieve,
    )
    output = _invoke(build_langgraph_agentic_retriever, controller)

    assert executed_queries == [
        "Which component is associated with alarm E17?",
        "ZX-9 pressure transducer calibration tolerance",
    ]
    assert output["evidence_ledger"]["find_tolerance"]["executed_query"] == executed_queries[1]
    assert output["sufficient"] is True


def test_deterministic_dependency_refinement_keeps_concrete_identifiers_only():
    prior = _result(
        "component",
        "alarm-doc",
        "The bus-powered RS-232C port uses cable OP-26487; see the and/or selection note.",
    ).model_copy(update={"metadata": {"identifier_tokens": ["bus-powered", "and/or", "RS-232C", "OP26487"]}})
    hop = RetrievalHop(
        hop_id="orientation",
        objective="Find that cable's connector orientation",
        query="What is that cable's connector orientation?",
        depends_on=["identify_cable"],
    )

    refined = refine_dependent_query(hop, [prior], use_llm=False)

    assert refined == (
        "Find that cable's connector orientation. Relevant prior-hop identifiers: RS-232C, OP26487"
    )
    assert "bus-powered" not in refined
    assert "and/or" not in refined


def test_insufficient_hop_gets_one_broad_recovery_within_budget():
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="primary",
                objective="Find the corrective action",
                query="What corrective action resolves alarm E17?",
                strategy="sparse",
            )
        ]
    )
    strategies: list[str] = []

    def retrieve(_query, _corpus_ids, _filters, strategy, _limit):
        strategies.append(strategy)
        if strategy == "sparse":
            return []
        return [_result("recovered", "alarm-doc", "Corrective action: replace the failed pressure transducer.")]

    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=retrieve,
    )
    output = _invoke(build_llamaindex_agentic_retriever, controller, max_hops=2)

    assert strategies == ["sparse", "broad"]
    assert output["sufficient"] is True
    assert output["stop_reason"] == "sufficient"
    assert output["retrieval_trace"]["completed_hops"] == ["primary", "primary_recovery"]


def test_hop_budget_stops_non_improving_recovery():
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="primary",
                objective="Find missing evidence",
                query="Find missing evidence",
                strategy="dense",
            )
        ]
    )
    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=lambda *_args: [],
    )

    output = _invoke(build_langgraph_agentic_retriever, controller, max_hops=2)

    assert output["sufficient"] is False
    assert output["stop_reason"] == "hop_budget_exhausted"
    assert output["retrieval_results"] == []


def test_dependent_hop_uses_successful_recovery_evidence():
    plan = RetrievalPlan(
        mode="dependent",
        hops=[
            RetrievalHop(
                hop_id="identify_component",
                objective="Identify the component associated with alarm E17",
                query="Which component is associated with alarm E17?",
                strategy="sparse",
            ),
            RetrievalHop(
                hop_id="find_tolerance",
                objective="Find that component's calibration tolerance",
                query="What is the calibration tolerance for the identified component?",
                strategy="structural",
                depends_on=["identify_component"],
            ),
        ],
    )
    refined_from: list[str] = []

    def retrieve(query, _corpus_ids, _filters, strategy, _limit):
        if "associated with alarm" in query and strategy == "sparse":
            return []
        if "associated with alarm" in query and strategy == "broad":
            return [_result("recovered-component", "alarm-doc", "Alarm E17 identifies the ZX-9 pressure transducer.")]
        return [_result("tolerance", "calibration-doc", "ZX-9 calibration tolerance is 0.2 percent.")]

    def refine(_hop, results):
        refined_from.extend(result.chunk_id for result in results)
        assert results[0].chunk_id == "recovered-component"
        return "ZX-9 pressure transducer calibration tolerance"

    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        refiner=refine,
        retriever=retrieve,
    )

    output = _invoke(build_langgraph_agentic_retriever, controller, max_hops=3)

    assert refined_from == ["recovered-component"]
    assert output["sufficient"] is True
    assert output["retrieval_trace"]["completed_hops"] == [
        "identify_component",
        "identify_component_recovery",
        "find_tolerance",
    ]


def test_dependent_sufficiency_requires_answer_signal_with_anchor():
    plan = RetrievalPlan(
        mode="dependent",
        hops=[
            RetrievalHop(
                hop_id="identify_cable",
                objective="Identify the cable model",
                query="Which cable model connects the RS-232C port?",
                strategy="sparse",
            ),
            RetrievalHop(
                hop_id="find_orientation",
                objective="Find that cable's connector orientation",
                query="What is that cable's connector orientation?",
                strategy="hybrid",
                depends_on=["identify_cable"],
            ),
        ],
    )

    def retrieve(query, _corpus_ids, _filters, strategy, _limit):
        if "Which cable model" in query:
            return [_result("cable", "port-doc", "The RS-232C port uses cable OP-26487.")]
        if strategy == "sparse":
            return [
                _result("anchor-only", "port-doc", "Cable OP-26487 connects the serial port."),
                _result("orientation-only", "other-doc", "The unrelated CB-12 connector is straight."),
            ]
        return [_result("anchored-answer", "catalog-doc", "Cable OP-26487 has a straight connector.")]

    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        refiner=lambda _hop, _results: "OP-26487 connector orientation",
        retriever=retrieve,
    )

    output = _invoke(build_llamaindex_agentic_retriever, controller, max_hops=3)

    assert output["sufficient"] is True
    assert output["retrieval_trace"]["completed_hops"] == [
        "identify_cable",
        "find_orientation",
        "find_orientation_recovery",
    ]
    assert output["evidence_ledger"]["find_orientation"]["sufficient"] is False
    assert output["evidence_ledger"]["find_orientation"]["assessment"]["dependency_anchors"] == ["OP-26487"]
    assert output["evidence_ledger"]["find_orientation_recovery"]["sufficient"] is True


def test_claim_sufficiency_rejects_cross_chunk_keyword_collage():
    sufficient, assessment = _assess_hop_evidence(
        "Find OP-26487 connector orientation",
        [
            _result("anchor", "doc-a", "OP-26487 is the serial cable model."),
            _result("facet", "doc-b", "An unrelated connector is straight."),
        ],
        dependency_anchors=["OP-26487"],
    )

    assert sufficient is False
    assert assessment["supporting_chunk_ids"] == []
    assert assessment["gap_reason"] == "no_single_chunk_supports_claim"


def test_context_reserves_attributed_support_instead_of_first_result():
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="lookup",
                objective="Find ALPHA-1 corrective action",
                query="ALPHA-1 corrective action",
                strategy="structural",
            )
        ]
    )
    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=lambda *_args: [
            _result("distractor", "doc-a", "ALPHA-1 alarm overview."),
            _result("support", "doc-a", "Corrective action: replace the ALPHA-1 fuse."),
        ],
    )

    output = _invoke(build_langgraph_agentic_retriever, controller)

    assert output["sufficient"] is True
    assert output["retrieval_results"][0]["chunk_id"] == "support"
    assert output["retrieval_results"][0]["metadata"]["agent_context_reasons"] == [
        "required_claim:lookup"
    ]
    assert output["retrieval_trace"]["context_assembly"]["all_required_claims_retained"] is True


def test_insufficient_ledger_blocks_synthesis_with_explicit_abstention():
    answer = insufficient_agent_answer(
        "What is the unsupported value?",
        {
            "stop_reason": "hop_budget_exhausted",
            "evidence_ledger": {
                "value": {
                    "required": True,
                    "sufficient": False,
                    "assessment": {"gap_reason": "missing_claim_facets"},
                }
            },
        },
    )

    assert answer.insufficient_evidence is True
    assert answer.citations == []
    assert "blocked" in answer.warnings[0].lower()


def test_llamaindex_policy_uses_its_own_alternate_query_engine_recovery():
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="subquestion_1",
                objective="Find ALPHA-1 corrective action",
                query="ALPHA-1 corrective action",
                strategy="sparse",
            )
        ]
    )
    tools: list[str] = []

    def retrieve(_query, _corpus_ids, _filters, strategy, _limit):
        tools.append(strategy)
        return [] if strategy == "sparse" else [
            _result("support", "doc-a", "Corrective action: replace the ALPHA-1 fuse.")
        ]

    controller = LlamaIndexAgenticController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=retrieve,
    )
    output = _invoke(build_llamaindex_agentic_retriever, controller, max_hops=2)

    assert tools == ["sparse", "dense"]
    assert output["sufficient"] is True
    assert output["retrieval_trace"]["completed_hops"] == [
        "subquestion_1",
        "subquestion_1_query_engine_retry_1",
    ]
