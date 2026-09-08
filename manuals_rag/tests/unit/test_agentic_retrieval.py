from manuals_rag_answering.agentic_retrieval import (
    AgenticRetrievalController,
    RetrievalHop,
    RetrievalPlan,
    build_langgraph_agentic_retriever,
    build_llamaindex_agentic_retriever,
    plan_retrieval,
    refine_dependent_query,
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
