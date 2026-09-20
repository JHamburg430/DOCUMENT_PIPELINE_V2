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
    refine_llamaindex_subquestion,
    insufficient_agent_answer,
    query_requires_visual_evidence,
    visual_evidence_unavailable_answer,
    _assess_hop_evidence,
    verify_retrieval_claim,
)
from manuals_rag_common.config import settings
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


def test_visual_dependency_router_is_conservative_for_spatial_manual_questions():
    assert query_requires_visual_evidence("Which pin in the wiring diagram carries output 4?") is True
    assert query_requires_visual_evidence("Which wire goes to pin 3 on the connector face?") is True
    assert query_requires_visual_evidence("Where on the screen is the calibration icon?") is True
    assert query_requires_visual_evidence("What is the rated input voltage?") is False


def test_visual_dependency_abstention_emits_no_citations():
    answer = visual_evidence_unavailable_answer("Show the connector pinout diagram")

    assert answer.insufficient_evidence is True
    assert answer.confidence == "low"
    assert answer.citations == []
    assert "visual" in answer.answer.lower()


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


def test_llamaindex_planner_decomposes_troubleshooting_before_labelled_lookup(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    query = (
        "What causes Failed to back up settings to VisionDatabase. for XG-X Series, "
        "and how should it be corrected?"
    )

    plan = plan_llamaindex_retrieval(query)

    assert plan.mode == "parallel"
    assert [hop.hop_id for hop in plan.hops] == ["subquestion_1", "subquestion_2"]
    assert [hop.objective for hop in plan.hops] == [
        "Find the documented cause of Failed to back up settings to VisionDatabase. for XG-X Series",
        "Find the documented corrective action for Failed to back up settings to VisionDatabase. for XG-X Series",
    ]


def test_planners_build_warning_context_dependency_without_model(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    queries = [
        (
            "When The controller should be installed in the direction of the circled figure below. "
            "for User's Manual (3D mode), what warning or caution about Caution on direction "
            "of controller mounting should be followed?"
        ),
        (
            "What warning or caution about Caution on direction of controller mounting for "
            "LJ: S8000 Series applies when the controller is designed to be mounted on a DIN rail?"
        ),
    ]

    for query in queries:
        for planner in (plan_retrieval, plan_llamaindex_retrieval):
            plan = planner(query)
            assert plan.mode == "dependent"
            assert len(plan.hops) == 2
            assert plan.hops[0].depends_on == []
            assert plan.hops[1].depends_on == [plan.hops[0].hop_id]
            assert [hop.strategy for hop in plan.hops] == ["structural", "sparse"]
            assert "Caution on direction of controller mounting" in plan.hops[1].query
            assert "50 mm" not in plan.hops[1].query
            assert "DIN rail" not in plan.hops[1].query


def test_warning_dependency_hops_verify_from_exact_scoped_atomic_records(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("verifier model must not run")),
    )
    query = (
        "When For proper ventilation, allow a space of 50 mm or more on top of the controller and a spac "
        "for User's Manual (3D mode), what warning or caution about Caution on direction of controller "
        "mounting should be followed?"
    )
    plan = plan_retrieval(query)
    context_result = _result(
        "context",
        "lj-x8000",
        "For proper ventilation, allow a space of 50 mm or more on top of the controller and a space "
        "of 50 mm or more on both sides.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "User's Manual (3D mode)",
                "product_family": "LJ: X8000 Series",
            }
        }
    )
    warning_result = _result(
        "warning",
        "lj-x8000",
        "Caution: Caution on direction of controller mounting",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "warning_record",
                "safety_flag": True,
                "product_model": "User's Manual (3D mode)",
                "product_family": "LJ: X8000 Series",
            }
        }
    )
    warning_action_result = _result(
        "warning-action",
        "lj-x8000",
        "The controller should be installed in the direction of the circled figure below. "
        "Do not install the controller in any other direction.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "User's Manual (3D mode)",
                "product_family": "LJ: X8000 Series",
            }
        }
    )

    for hop, results in zip(
        plan.hops,
        ([context_result], [warning_action_result, warning_result]),
        strict=True,
    ):
        _sufficient, assessment = _assess_hop_evidence(hop.objective, results)
        verdict = verify_retrieval_claim(hop, hop.query, results, assessment, use_llm=True)
        assert verdict["trust_state"] == "confirmed"
        assert verdict["claim_supported"] is True
        expected_chunk_id = "context" if hop.hop_id == "establish_context" else "warning"
        assert verdict["supporting_chunk_ids"] == [expected_chunk_id]


def test_warning_dependency_refiners_preserve_exact_title_query(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("refiner model must not run")),
    )
    query = (
        "When Mount the controller in a stable location for User's Manual (3D mode), "
        "what warning or caution about Caution on direction of controller mounting should be followed?"
    )
    warning_hop = plan_retrieval(query).hops[1]
    dependency_result = _result(
        "context",
        "lj-x8000",
        "Mount the controller in a stable location that is free from vibration.",
    )

    assert refine_dependent_query(warning_hop, [dependency_result], use_llm=True) == warning_hop.query
    assert refine_llamaindex_subquestion(warning_hop, [dependency_result], use_llm=True) == warning_hop.query


def test_planners_keep_unknown_identifier_value_lookup_sparse_and_single_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    query = "What is the quantum flux calibration value for the ZX-9999 controller?"

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query)
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "sparse"


def test_planners_keep_value_applies_to_lookup_structural_and_single_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    query = "What Display Settings Green Lower Limit Value value applies to VS Series Vision System with Built: in AI?"

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query)
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "structural"


def test_model_planners_reject_unsafe_hop_ids_and_fall_back(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "mode": "single",
                "rationale": "Malformed identifier.",
                "hops": [
                    {
                        "hop_id": "{0}",
                        "objective": "controller communication behavior",
                        "query": "controller communication behavior",
                        "strategy": "dense",
                        "depends_on": [],
                        "required": True,
                    }
                ],
            },
            "{}",
        ),
    )

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner("Explain controller communication behavior", use_llm=True)
        assert len(plan.hops) == 1
        assert plan.hops[0].hop_id in {"hop_1", "subquestion_1"}
        assert plan.hops[0].query == "Explain controller communication behavior"


def test_planner_decomposes_scoped_troubleshooting_what_should_i_do():
    plan = plan_retrieval(
        "On an XG-X Series controller, what causes the error that says to turn off "
        "power temporarily and check the expansion units, and what should I do?",
        use_llm=False,
    )

    assert plan.mode == "parallel"
    assert [hop.hop_id for hop in plan.hops] == ["cause", "corrective_action"]


def test_planner_routes_direct_labelled_lookup_to_structural_without_model(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )

    query = "For CV-X multi-capture mode, what trigger mode uses external triggers 1 and 2?"
    plans = [plan_retrieval(query), plan_llamaindex_retrieval(query)]

    for plan in plans:
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].strategy == "structural"


def test_model_planners_cannot_mark_primary_claims_optional(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "mode": "single",
                "rationale": "One lookup.",
                "hops": [
                    {
                        "hop_id": "lookup",
                        "objective": "Explain controller communication behavior",
                        "query": "controller communication behavior",
                        "strategy": "dense",
                        "depends_on": [],
                        "required": False,
                    }
                ],
            },
            "{}",
        ),
    )

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner("Explain controller communication behavior", use_llm=True)
        assert plan.hops[0].required is True


def test_model_planners_preserve_original_single_lookup_qualifiers(monkeypatch):
    original = (
        "On CV-X482, what does command 0028 / 65.0 map to in the 6-bit command output area?"
    )
    for mode in ("single", "parallel"):
        monkeypatch.setattr(
            "manuals_rag_answering.agentic_retrieval.chat_json",
            lambda **_kwargs: (
                {
                    "mode": mode,
                    "rationale": "One lookup.",
                    "hops": [
                        {
                            "hop_id": "lookup",
                            "objective": "Find the output mapping for command 0028 on CV-X482",
                            "query": "CV-X482 command 0028 output mapping",
                            "strategy": "sparse",
                            "depends_on": [],
                            "required": True,
                        }
                    ],
                },
                "{}",
            ),
        )

        for planner in (plan_retrieval, plan_llamaindex_retrieval):
            plan = planner(original, use_llm=True)
            assert plan.hops[0].objective == original
            assert plan.hops[0].query == original
            assert plan.hops[0].strategy == "hybrid"


def test_model_planners_route_exact_count_and_accessory_lookups_to_hybrid(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model planner must not run")),
    )
    queries = {
        (
            "On IV4-G120, how many objects are counted at one time when the count "
            "value is 9 and ON equals the set value?"
        ): "hybrid",
        "For CA-DRM10X, is OP-42284 the accessory code for the CA-DRx9 light?": "hybrid",
        (
            "On CV-X482, what adjustment is recommended when Contrast detection "
            "runs but no NG judgment is given?"
        ): "structural",
    }

    for query, expected_strategy in queries.items():
        for planner in (plan_retrieval, plan_llamaindex_retrieval):
            plan = planner(query, use_llm=True)
            assert plan.mode == "single"
            assert len(plan.hops) == 1
            assert plan.hops[0].query == query
            assert plan.hops[0].strategy == expected_strategy


def test_planners_add_canonical_camera_trigger_light_menu_label(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    query = (
        "In Standard Lighting Mode, for XG-X line-scan camera setup, which camera, "
        "trigger, and lighting settings are tied to simulation image capture?"
    )

    plans = [plan_retrieval(query), plan_llamaindex_retrieval(query)]

    for plan in plans:
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].strategy == "structural"
        assert "Camera Trigger Light Configuration Settings" in plan.hops[0].query


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


def test_model_planners_pair_scoped_named_settings_before_labelled_lookup(monkeypatch):
    query = (
        "For CV-X482 and LJ-X8000, compare what the Condition list and Standard Angle "
        "settings control."
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("deterministic scoped comparison planning must run before model planning")

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", fail_if_called)

    langgraph = plan_retrieval(query, use_llm=True)
    llamaindex = plan_llamaindex_retrieval(query, use_llm=True)

    assert langgraph.mode == "parallel"
    assert [hop.query for hop in langgraph.hops] == [
        "For CV-X482, what does the Condition list setting control?",
        "For LJ-X8000, what does the Standard Angle setting control?",
    ]
    assert llamaindex.mode == "parallel"
    assert [hop.query for hop in llamaindex.hops] == [hop.query for hop in langgraph.hops]


def test_comparison_planner_assigns_trailing_details_to_matching_branches():
    plan = plan_retrieval(
        "Compare the VJ-H500CX weight qualification with the LJ:S8000 grayscale adjustment: "
        "give the exact weight and lens caveat, then the exact adjustment direction and what that changes.",
        use_llm=False,
    )

    assert plan.mode == "parallel"
    assert plan.hops[0].query == (
        "What is the VJ-H500CX weight qualification; give the exact weight and lens caveat?"
    )
    assert plan.hops[1].query == (
        "What is the LJ:S8000 grayscale adjustment; the exact adjustment direction and what that changes?"
    )


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


def test_model_planners_make_demonstrative_coordinate_question_dependent(monkeypatch):
    query = (
        "Which encoder head model is compatible with the CA-EN100U, and how is that "
        "encoder head powered?"
    )

    planner_calls = []

    def fail_if_called(**_kwargs):
        planner_calls.append(_kwargs)
        raise AssertionError("deterministic dependency safety must run before model planning")

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", fail_if_called)

    langgraph = plan_retrieval(query, use_llm=True)
    llamaindex = plan_llamaindex_retrieval(query, use_llm=True)

    assert langgraph.mode == "dependent"
    assert langgraph.hops[1].depends_on == ["facet_1"]
    assert llamaindex.mode == "dependent"
    assert llamaindex.hops[1].depends_on == ["subquestion_1"]

    cable_query = (
        "Which cable model connects the LJ-X8000 RS-232C port, then what is that "
        "cable's connector orientation?"
    )
    cable_langgraph = plan_retrieval(cable_query, use_llm=True)
    cable_llamaindex = plan_llamaindex_retrieval(cable_query, use_llm=True)
    assert cable_langgraph.mode == "dependent"
    assert cable_langgraph.hops[1].depends_on == ["hop_1"]
    assert cable_llamaindex.mode == "dependent"
    assert cable_llamaindex.hops[1].depends_on == ["subquestion_1"]
    assert planner_calls == []


def test_verifier_deterministically_confirms_serial_cable_mapping_and_orientation(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )
    cable = _result(
        "cable",
        "ljx-doc",
        "The port to connect RS: 232C cable (OP-26487: 2.5 m, sold separately).",
    )
    cable.metadata["product_models"] = ["LJ-X8000"]
    cable_output = verify_retrieval_claim(
        RetrievalHop(
            hop_id="cable",
            objective="Which cable model connects the LJ-X8000 RS-232C port?",
            query="Which cable model connects the LJ-X8000 RS-232C port?",
        ),
        "Which cable model connects the LJ-X8000 RS-232C port?",
        [cable],
        {"claim_supported": True, "supporting_chunk_ids": ["cable"]},
    )
    assert cable_output["trust_state"] == "confirmed"
    assert cable_output["supporting_chunk_ids"] == ["cable"]

    orientation = _result(
        "orientation",
        "ljx-doc",
        "Column headers: Description; Row headers: OP-26487; "
        "Cell value: Serial connection cable (2.5 m, straight); Row: 14; Column: 1",
    )
    orientation_output = verify_retrieval_claim(
        RetrievalHop(
            hop_id="orientation",
            objective="What is that cable's connector orientation?",
            query="What is that cable's connector orientation?",
        ),
        "What is the Description for OP-26487, including the cable connector orientation?",
        [orientation],
        {"claim_supported": False, "supporting_chunk_ids": []},
    )
    assert orientation_output["trust_state"] == "confirmed"
    assert orientation_output["supporting_chunk_ids"] == ["orientation"]


def test_llamaindex_keeps_structural_tool_for_dependency_predicate():
    hop = RetrievalHop(
        hop_id="power",
        objective="How is that encoder head powered?",
        query="How is that encoder head powered?",
        strategy="structural",
        depends_on=["model"],
    )

    assert LlamaIndexAgenticController._route_tool(hop, ["CA-EN100H"]) == "structural"


def test_model_planners_split_multi_product_reported_clauses(monkeypatch):
    query = (
        "Prepare a commissioning note that states the VJ-H500CX weight and whether it includes the lens, "
        "explains how to increase grayscale percentage on LJ:S8000, and names the KV-X integrated software "
        "plus its upgrade benefit."
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("multi-product claim safety must run before model planning")

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", fail_if_called)

    langgraph = plan_retrieval(query, use_llm=True)
    llamaindex = plan_llamaindex_retrieval(query, use_llm=True)

    assert langgraph.mode == "parallel"
    assert [hop.hop_id for hop in langgraph.hops] == ["claim_1", "claim_2", "claim_3"]
    assert [hop.query for hop in langgraph.hops] == [
        "Find explicit manual evidence that states the VJ-H500CX weight and whether it includes the lens.",
        "Find explicit manual evidence that explains how to increase grayscale percentage on LJ:S8000.",
        "Find explicit manual evidence that names the KV-X integrated software plus its upgrade benefit.",
    ]
    assert [hop.hop_id for hop in llamaindex.hops] == ["subquestion_1", "subquestion_2", "subquestion_3"]


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

    verifier_kwargs = {}

    def fake_chat_json(**kwargs):
        verifier_kwargs.update(kwargs)
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
    assert verifier_kwargs["num_batch"] == settings.ollama_retrieval_verifier_num_batch
    assert result["judge"]["status"] == "checked"
    assert result["judge"]["attempts"][0]["raw_response"] == "{}"
    assert result["judge"]["attempts"][0]["parsed_response"]["supporting_chunk_ids"] == ["invented-chunk"]


def test_verifier_deterministically_confirms_condition_aligned_warning(monkeypatch):
    hop = RetrievalHop(
        hop_id="warning",
        objective=(
            "What is the XG-X warning when the output limiter is off and light "
            "intensity is 512 or higher?"
        ),
        query="XG-X output limiter warning 512",
    )
    result = _result(
        "exact-warning",
        "xgx-doc",
        "When the Limit Output is OFF and the intensity is set to 512 or higher, "
        "be careful not to damage the light through excessive heat generation.",
    )
    result.metadata["chunk_type"] = "atomic_text"
    result.metadata["product_model"] = "XG-X"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["exact-warning"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == ["exact-warning"]


def test_verifier_does_not_confirm_scattered_warning_terms(monkeypatch):
    hop = RetrievalHop(
        hop_id="warning",
        objective=(
            "What is the XG-X warning when the output limiter is off and light "
            "intensity is 512 or higher?"
        ),
        query="XG-X output limiter warning 512",
    )
    result = _result(
        "scattered-warning",
        "xgx-doc",
        "The output limiter permits intensity 512. Warning: avoid damage from heat elsewhere.",
    )
    result.metadata["product_model"] = "XG-X"
    calls = 0

    def unresolved_verifier(**_kwargs):
        nonlocal calls
        calls += 1
        return (
            {
                "trust_state": "unresolved",
                "claim_supported": False,
                "supporting_chunk_ids": [],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": "XG-X",
                "rationale": "The condition and warning are not bound in one sentence.",
            },
            "{}",
        )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        unresolved_verifier,
    )
    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["scattered-warning"]},
    )

    assert calls == 1
    assert output["claim_supported"] is False
    assert output["trust_state"] == "unresolved"


def test_verifier_deterministically_confirms_exact_structured_setting_row(monkeypatch):
    hop = RetrievalHop(
        hop_id="setting",
        objective=(
            "What does the LJ-S8000 Output Symbol Identifier setting add "
            "when it is enabled?"
        ),
        query="LJ-S8000 Output Symbol Identifier setting enabled",
    )
    result = _result(
        "setting-row",
        "lj-doc",
        "Column headers: Settings; Row headers: Output Symbol Identifier; "
        "Cell value: When enabled, a symbol identifier (3 bytes) defined by "
        "ISO / IEC 15424 is added to the beginning of the read data.; "
        "Row: 6; Column: 1",
    )
    result.metadata["chunk_type"] = "table_record"
    result.metadata["product_model"] = "LJ: S8000 Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["setting-row"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == ["setting-row"]


def test_verifier_does_not_confirm_neighboring_structured_setting_row(monkeypatch):
    hop = RetrievalHop(
        hop_id="setting",
        objective="What does the LJ-S8000 Output Symbol Identifier setting add?",
        query="LJ-S8000 Output Symbol Identifier setting",
    )
    result = _result(
        "neighbor-row",
        "lj-doc",
        "Column headers: Settings; Row headers: Decode Result Identifier; "
        "Cell value: When enabled, a result identifier is added to the read data.; "
        "Row: 7; Column: 1",
    )
    result.metadata["chunk_type"] = "table_record"
    result.metadata["product_model"] = "LJ: S8000 Series"
    calls = 0

    def unresolved_verifier(**_kwargs):
        nonlocal calls
        calls += 1
        return (
            {
                "trust_state": "unresolved",
                "claim_supported": False,
                "supporting_chunk_ids": [],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": "LJ-S8000",
                "rationale": "The retrieved row is for a different setting.",
            },
            "{}",
        )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        unresolved_verifier,
    )
    output = verify_retrieval_claim(
        hop,
        hop.objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["neighbor-row"]},
    )

    assert calls == 1
    assert output["claim_supported"] is False


def test_verifier_deterministically_confirms_exact_structured_lookup_cell(monkeypatch):
    objective = (
        "For CV-X multi-capture mode, what trigger mode uses external triggers 1 and 2?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "exact-trigger-mode",
        "cvx-doc",
        "Column headers: Multi-Capture; Row headers: Trigger Mode; "
        "Cell value: External trigger (using trigger 1 and trigger 2); "
        "Row: 3; Column: 1",
    )
    result.metadata["product_model"] = "CV-X482"
    result.metadata["product_family"] = "CV-X Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": False, "supporting_chunk_ids": []},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_confirms_numeric_row_and_bit_column_mapping(monkeypatch):
    objective = (
        "Find the mapping for command code 0028 with value 65.0 in the 6-bit "
        "command output area for device CV-X482."
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "command-result-cell",
        "cvx-doc",
        "Column headers: 6bit > 5bit > 4bit > 3bit > 2bit > 1bit > 0bit; "
        "Row headers: 0028 65.0 > Command output area; Cell value: Command Result; "
        "Row: 15; Column: 3",
    )
    result.metadata["product_model"] = "CV-X482"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_confirms_exact_count_relation_cell(monkeypatch):
    objective = (
        "On IV4-G120 in latching output mode, how many objects are counted at one time when the count "
        "value is 9 and ON equals the set value?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "exact-count-cell",
        "iv4-doc",
        "Column headers: Quantity counted at one time; Row headers: ON when = "
        "Set value > Count value= 9; Cell value: 3; Row: 4; Column: 3",
    )
    result.metadata.update({"chunk_type": "table_record", "product_model": "IV4-G120"})
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_rejects_neighboring_count_relation_cell():
    objective = (
        "On IV4-G120, how many objects are counted at one time when the count "
        "value is 9 and ON equals the set value?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    neighbor = _result(
        "neighbor-count-cell",
        "iv4-doc",
        "Column headers: Quantity counted at one time; Row headers: ON when >= "
        "Set value > Count value >= 9; Cell value: 2; Row: 3; Column: 3",
    )
    neighbor.metadata.update({"chunk_type": "table_record", "product_model": "IV4-G120"})

    output = verify_retrieval_claim(
        hop,
        objective,
        [neighbor],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert output["claim_supported"] is False
    assert output["trust_state"] == "unresolved"


def test_verifier_marks_duplicate_count_coordinates_with_different_values_conflicting():
    objective = (
        "On IV4-G120, how many objects are counted at one time when the count "
        "value is 9 and ON equals the set value?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    results = []
    for row, value in ((3, 2), (4, 3)):
        result = _result(
            f"count-row-{row}",
            "iv4-doc",
            "Column headers: Quantity counted at one time; Row headers: ON when = "
            f"Set value > Count value= 9; Cell value: {value}; Row: {row}; Column: 3",
        )
        result.metadata.update({"chunk_type": "table_record", "product_model": "IV4-G120"})
        results.append(result)

    output = verify_retrieval_claim(
        hop,
        objective,
        results,
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id for result in results]},
        use_llm=False,
    )

    assert output["trust_state"] == "conflicting"
    assert output["claim_supported"] is False
    assert set(output["conflicting_chunk_ids"]) == {"count-row-3", "count-row-4"}

    sampled_output = verify_retrieval_claim(
        hop,
        objective,
        results[:1],
        {"claim_supported": True, "supporting_chunk_ids": [results[0].chunk_id]},
        use_llm=False,
    )
    assert sampled_output["trust_state"] == "conflicting"
    assert sampled_output["claim_supported"] is False


def test_verifier_deterministically_confirms_exact_menu_mapping(monkeypatch):
    objective = (
        "In Standard Lighting Mode, for XG-X line-scan camera setup, which camera, "
        "trigger, and lighting settings are tied to simulation image capture?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "standard-lighting-menu",
        "xgx-doc",
        "Camera - Trigger - Light Configuration Settings (Page 7-205): "
        "Simulation Image Capture (Page 7-213); Camera Settings (Page 7-206)",
    )
    result.metadata.update(
        {
            "chunk_type": "table_record",
            "product_family": "XG-X Series",
            "page_context": "Standard Lighting Mode (Line Scan Camera)",
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_rejects_menu_mapping_from_adjacent_lighting_mode(monkeypatch):
    objective = (
        "In Standard Lighting Mode, for XG-X line-scan camera setup, which camera, "
        "trigger, and lighting settings are tied to simulation image capture?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "specular-lighting-menu",
        "xgx-doc",
        "Camera - Trigger - Light Configuration Settings (Page 7-205): "
        "Simulation Image Capture (Page 7-213); Camera Settings (Page 7-206)",
    )
    result.metadata.update(
        {
            "chunk_type": "table_record",
            "product_family": "XG-X Series",
            "page_context": "LumiTrax Specular Reflection Mode (Line Scan Camera)",
        }
    )
    calls = 0

    def unresolved_verifier(**_kwargs):
        nonlocal calls
        calls += 1
        return (
            {
                "trust_state": "unresolved",
                "claim_supported": False,
                "supporting_chunk_ids": [],
                "conflicting_chunk_ids": [result.chunk_id],
                "applicability": "not_requested",
                "scope_entity": "Standard Lighting Mode",
                "rationale": "The retrieved menu is for a different lighting mode.",
            },
            "{}",
        )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        unresolved_verifier,
    )
    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert calls == 1
    assert output["claim_supported"] is False
    assert output["trust_state"] == "unresolved"


def test_verifier_does_not_bind_structured_lookup_to_adjacent_capture_mode(monkeypatch):
    objective = (
        "For CV-X multi-capture mode, what trigger mode uses external triggers 1 and 2?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "wrong-capture-mode",
        "cvx-doc",
        "Column headers: 3D Capture; Row headers: Trigger Mode; "
        "Cell value: External trigger (using trigger 1 and trigger 2); "
        "Row: 3; Column: 1",
    )
    result.metadata["product_model"] = "CV-X482"
    result.metadata["product_family"] = "CV-X Series"

    def unresolved_verifier(**_kwargs):
        return (
            {
                "trust_state": "unresolved",
                "claim_supported": False,
                "supporting_chunk_ids": [],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": "CV-X",
                "rationale": "The retrieved row is for a different capture mode.",
            },
            "{}",
        )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        unresolved_verifier,
    )
    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["claim_supported"] is False
    assert output["trust_state"] == "unresolved"


def test_verifier_confirms_exact_structured_troubleshooting_cause(monkeypatch):
    hop = RetrievalHop(
        hop_id="cause",
        objective=(
            "Find the documented cause of The number of characters that can be "
            "registered for 1 character was exceeded. for XG-X Series"
        ),
        query="XG-X character registration cause",
    )
    result = _result(
        "cause-row",
        "xgx-doc",
        "Column headers: Cause; Row headers: The number of characters that can be "
        "registered for 1 character was exceeded. Cannot register.; Cell value: "
        "You are trying to register more than 200 character patterns for one type "
        "of character in the OCR2 unit.; Row: 11; Column: 1",
    )
    result.metadata["chunk_type"] = "table_record"
    result.metadata["product_family"] = "XG-X Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {
            "claim_supported": False,
            "supporting_chunk_ids": ["cause-row"],
            "contradictions": ["conflicting_orientation_values"],
        },
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["cause-row"]


def test_verifier_treats_firmware_error_text_as_troubleshooting_not_applicability(monkeypatch):
    hop = RetrievalHop(
        hop_id="cause",
        objective=(
            "Find the documented cause of The controller was booted using an "
            "unsupported firmware. for XG-X Series"
        ),
        query=(
            "What causes The controller was booted using an unsupported firmware. "
            "for XG-X Series?"
        ),
    )
    result = _result(
        "firmware-cause-row",
        "xgx-doc",
        "Column headers: Cause; Row headers: and check the file > The controller "
        "was booted using an unsupported firmware. Please turn off the power once "
        "and then update the firmware.; Cell value: The controller was started "
        "using a non-supported firmware version.; Row: 8; Column: 1",
    )
    result.metadata["product_family"] = "XG-X Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["firmware-cause-row"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["firmware-cause-row"]
    assert output["applicability"] == "not_requested"


def test_verifier_confirms_scoped_flowchart_branch_rule_without_llm(monkeypatch):
    hop = RetrievalHop(
        hop_id="rule",
        objective=(
            "For XG-X asynchronous capture with multiple capture units, what "
            "flowchart branching rule should be followed?"
        ),
        query="XG-X multiple capture unit flowchart branching rule",
    )
    result = _result(
        "flowchart-rule",
        "xgx-doc",
        "When multiple capture units are used, the flowchart is branched by the "
        "passing status of the first capture unit. The passing status of the "
        "capture unit executed before the branch unit must be specified as the "
        "branch condition.",
    )
    result.metadata["chunk_type"] = "section_window"
    result.metadata["product_family"] = "XG-X Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["flowchart-rule"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["flowchart-rule"]


def test_verifier_confirms_scoped_yes_no_sentence_without_llm(monkeypatch):
    hop = RetrievalHop(
        hop_id="fact",
        objective=(
            "For XG-X asynchronous capture with multiple cameras, can multiple "
            "capture units be placed in the flow?"
        ),
        query="XG-X multiple cameras asynchronous capture units in flow",
    )
    result = _result(
        "capture-units",
        "xgx-doc",
        "When you use multiple cameras asynchronously, you can also place multiple capture units.",
    )
    result.metadata["product_family"] = "XG-X Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["capture-units"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["capture-units"]


def test_verifier_confirms_named_timing_chart_without_llm(monkeypatch):
    objective = (
        "For CV-X multi-capture trigger timing, which control/data I/O timing chart "
        "should I check?"
    )
    hop = RetrievalHop(hop_id="reference", objective=objective, query=objective)
    result = _result(
        "timing-chart",
        "cvx-doc",
        "Timing chart Control/data output via I/O terminals",
    )
    result.metadata["product_family"] = "CV-X Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_rejects_neighboring_structured_troubleshooting_row(monkeypatch):
    hop = RetrievalHop(
        hop_id="cause",
        objective="Find the documented cause of Error 100 for XG-X Series",
        query="XG-X Error 100 cause",
    )
    result = _result(
        "neighbor-cause",
        "xgx-doc",
        "Column headers: Cause; Row headers: Error 101 occurred.; Cell value: "
        "The output buffer is full.; Row: 4; Column: 1",
    )
    result.metadata["product_family"] = "XG-X Series"
    calls = 0

    def unresolved_verifier(**_kwargs):
        nonlocal calls
        calls += 1
        return ({
            "trust_state": "unresolved",
            "claim_supported": False,
            "supporting_chunk_ids": [],
            "conflicting_chunk_ids": [],
            "applicability": "not_requested",
            "rationale": "Wrong fault row.",
        }, "{}")

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", unresolved_verifier)
    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["neighbor-cause"]},
    )

    assert calls == 1
    assert output["claim_supported"] is False


def test_verifier_confirms_atomic_status_to_corrective_action(monkeypatch):
    objective = (
        "On CV-X482 in Presence/Absence Quality Learning, what adjustment is recommended when Contrast detection "
        "runs but no NG judgment is given?"
    )
    hop = RetrievalHop(hop_id="adjustment", objective=objective, query=objective)
    result = _result(
        "contrast-action",
        "cvx-doc",
        "Status: Detection is performed with Contrast, but NG judgment is not given.; "
        "Corrective action: Increase the lower limit of Quality Match (%).",
    )
    result.metadata.update({"chunk_type": "table_record", "product_model": "CV-X482"})
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_rejects_different_atomic_troubleshooting_status():
    objective = (
        "On CV-X482, what adjustment is recommended when Contrast detection "
        "runs but no NG judgment is given?"
    )
    hop = RetrievalHop(hop_id="adjustment", objective=objective, query=objective)
    neighbor = _result(
        "focus-action",
        "cvx-doc",
        "Status: Detection is performed with Focus, but NG judgment is not given.; "
        "Corrective action: Increase the edge strength limit.",
    )
    neighbor.metadata.update({"chunk_type": "table_record", "product_model": "CV-X482"})

    output = verify_retrieval_claim(
        hop,
        objective,
        [neighbor],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert output["claim_supported"] is False


def test_verifier_marks_same_troubleshooting_status_with_different_actions_conflicting():
    objective = (
        "On CV-X482, what adjustment is recommended when Contrast detection "
        "runs but no NG judgment is given?"
    )
    hop = RetrievalHop(hop_id="adjustment", objective=objective, query=objective)
    results = []
    for chunk_id, action in (
        ("presence-action", "Increase the lower limit of Quality Match (%)."),
        ("flaw-action", "Reduce the upper limit value of defect size."),
    ):
        result = _result(
            chunk_id,
            "cvx-doc",
            "Status: Detection is performed with Contrast, but NG judgment is not given.; "
            f"Corrective action: {action}",
        )
        result.metadata.update({"chunk_type": "table_record", "product_model": "CV-X482"})
        results.append(result)

    output = verify_retrieval_claim(
        hop,
        objective,
        results,
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id for result in results]},
        use_llm=False,
    )

    assert output["trust_state"] == "conflicting"
    assert output["claim_supported"] is False
    assert set(output["conflicting_chunk_ids"]) == {"presence-action", "flaw-action"}

    sampled_output = verify_retrieval_claim(
        hop,
        objective,
        results[:1],
        {"claim_supported": True, "supporting_chunk_ids": [results[0].chunk_id]},
        use_llm=False,
    )
    assert sampled_output["trust_state"] == "conflicting"
    assert sampled_output["claim_supported"] is False


def test_scope_gate_accepts_series_suffix_alias_without_prefix_matching_models():
    matching = _result(
        "lj-series",
        "lj-doc",
        "When enabled, a symbol identifier is added to the beginning of the read data.",
    )
    matching.metadata["product_model"] = "LJ: S8000 Series"
    supported, assessment = _assess_hop_evidence(
        "What does the LJ-S8000 Output Symbol Identifier setting add when it is enabled?",
        [matching],
    )

    assert supported is True
    assert assessment["supporting_chunk_ids"] == ["lj-series"]

    different = _result("other", "other-doc", "ALPHA-10 setting details.")
    different.metadata["product_model"] = "ALPHA-10 Series"
    supported, _assessment = _assess_hop_evidence(
        "What does the ALPHA-1 setting add?",
        [different],
    )
    assert supported is False


def test_verifier_canonicalizes_vendor_prefixed_scope_for_exact_warning_title(monkeypatch):
    hop = RetrievalHop(
        hop_id="resolve_warning",
        objective=(
            "Resolve the warning or caution about Caution on direction of controller "
            "mounting for LJ: S8000 Series"
        ),
        query=(
            "For LJ: S8000 Series, retrieve the warning or caution titled: "
            "Caution on direction of controller mounting."
        ),
    )
    matching = _result(
        "s8000-warning",
        "s8000-doc",
        "Caution: Caution on direction of controller mounting",
    )
    matching.metadata.update(
        {
            "chunk_type": "warning_record",
            "safety_flag": True,
            "product_model": "LJ: S8000 Series",
            "product_family": "LJ-S8000 Series",
        }
    )
    neighboring = _result(
        "x8000-warning",
        "x8000-doc",
        "Caution: Caution on direction of controller mounting",
    )
    neighboring.metadata.update(
        {
            "chunk_type": "warning_record",
            "safety_flag": True,
            "product_model": "LJ: X8000 Series",
            "product_family": "LJ-X8000 Series",
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [neighboring, matching],
        {"claim_supported": True, "supporting_chunk_ids": [matching.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["s8000-warning"]
    assert "x8000-warning" not in output["scope_candidate_chunk_ids"]


def test_scope_gate_uses_legacy_family_when_product_model_is_document_title():
    matching = _result(
        "lj-x-series",
        "lj-x-doc",
        "Standard Angle specifies the start angle for blob numbering.",
    )
    matching.metadata.update(
        {
            "product_model": "User's Manual (3D mode)",
            "product_family": "LJ: X8000 Series",
            "product_models": ["LJ-X8000"],
        }
    )

    supported, assessment = _assess_hop_evidence(
        "For LJ-X8000, what does the Standard Angle setting control?",
        [matching],
    )

    assert supported is True
    assert assessment["supporting_chunk_ids"] == ["lj-x-series"]

    conflicting = _result("lj-s", "lj-s-doc", "Standard Angle details.")
    conflicting.metadata.update(
        {
            "product_model": "LJ-S8000",
            "product_family": "LJ: X8000 Series",
            "product_models": ["LJ-X8000"],
        }
    )
    supported, _assessment = _assess_hop_evidence(
        "For LJ-X8000, what does the Standard Angle setting control?",
        [conflicting],
    )
    assert supported is False


def test_named_setting_control_verification_confirms_one_definition_and_rejects_ambiguity():
    query = "For MOD-600, what does the Standard Angle setting control?"
    hop = RetrievalHop(hop_id="setting", objective=query, query=query, strategy="structural")
    numbering = _result(
        "numbering",
        "manual",
        "Column headers: Settings; Row headers: Standard Angle; Cell value: "
        "Specifies the start angle for blob numbering.; Row: 1; Column: 1",
    )
    numbering.metadata["product_model"] = "MOD-600"
    preliminary = {
        "claim_supported": True,
        "supporting_chunk_ids": ["numbering"],
    }

    verified = verify_retrieval_claim(
        hop,
        query,
        [numbering],
        preliminary,
        use_llm=False,
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["numbering"]

    exclusion = _result(
        "exclusion",
        "manual",
        "Column headers: Settings; Row headers: Standard Angle; Cell value: "
        "Select the reference angle of the proximity exclusion angle.; Row: 2; Column: 1",
    )
    exclusion.metadata["product_model"] = "MOD-600"
    ambiguous = verify_retrieval_claim(
        hop,
        query,
        [numbering, exclusion],
        {
            "claim_supported": True,
            "supporting_chunk_ids": ["numbering", "exclusion"],
        },
        use_llm=False,
    )

    assert ambiguous["trust_state"] == "conflicting"
    assert set(ambiguous["conflicting_chunk_ids"]) == {"numbering", "exclusion"}


def test_structured_compatibility_mapping_confirms_applicability_without_model():
    query = "Which encoder head model is compatible with the CA-EN100U?"
    hop = RetrievalHop(hop_id="compatibility", objective=query, query=query, strategy="structural")
    mapping = _result(
        "compatibility-row",
        "encoder-manual",
        "Model: Supported encoder head; CA-EN100U: CA-EN100H",
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [mapping],
        {"claim_supported": True, "supporting_chunk_ids": ["compatibility-row"]},
        use_llm=False,
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["applicability"] == "applicable"
    assert verified["supporting_chunk_ids"] == ["compatibility-row"]


def test_structured_accessory_mapping_confirms_exact_part_and_light(monkeypatch):
    query = "For CA-DRM10X, is OP-42284 the accessory code for the CA-DRx9 light?"
    hop = RetrievalHop(hop_id="accessory", objective=query, query=query)
    mapping = _result(
        "accessory-row",
        "light-manual",
        'Part number: 19.69" OP-42284; Applicable light: CA-DRx9',
    )
    mapping.metadata.update({"chunk_type": "table_record", "product_model": "CA-DRM10X"})
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [mapping],
        {"claim_supported": False, "supporting_chunk_ids": []},
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["accessory-row"]


def test_structured_accessory_mapping_rejects_neighboring_light():
    query = "For CA-DRM10X, is OP-42284 the accessory code for the CA-DRx9 light?"
    hop = RetrievalHop(hop_id="accessory", objective=query, query=query)
    neighbor = _result(
        "accessory-neighbor",
        "light-manual",
        'Part number: 19.69" OP-42284; Applicable light: CA-DRx8',
    )
    neighbor.metadata.update({"chunk_type": "table_record", "product_model": "CA-DRM10X"})

    verified = verify_retrieval_claim(
        hop,
        query,
        [neighbor],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert verified["claim_supported"] is False


def test_structured_power_source_mapping_confirms_powered_by_without_model():
    query = "How is that encoder head powered; constrain the lookup to CA-EN100H, CA-EN100U?"
    hop = RetrievalHop(hop_id="power", objective=query, query=query, strategy="structural")
    mapping = _result(
        "power-row",
        "encoder-manual",
        "Column headers: CA-EN100H; Row headers: Power-supply; "
        "Cell value: Supply from CA-EN100U; Row: 8; Column: 1",
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [mapping],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["power-row"]


def test_verifier_normalizes_single_list_wrapped_object(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            [{
                "trust_state": "confirmed",
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": "ALPHA-1",
                "rationale": "The cited chunk directly supports the claim.",
            }],
            "[]",
        ),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "Corrective action: replace the ALPHA-1 fuse.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["alpha"]


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


def test_verifier_normalizes_verified_and_chunk_ids_aliases(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "verified": True,
                "chunk_ids": ["alpha"],
                "evidence_support": "The cited chunk directly states the action.",
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


def test_verifier_normalizes_claim_verified_and_supporting_evidence_aliases(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "claim_verified": True,
                "supporting_evidence": ["alpha"],
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


def test_verifier_normalizes_verified_trust_and_confirmed_applicability(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Find ALPHA-1 corrective action",
        query="ALPHA-1 corrective action",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "trust_state": "verified",
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "conflicting_chunk_ids": [],
                "applicability": "confirmed",
                "scope_entity": "ALPHA-1",
                "rationale": "The cited chunk directly states the action.",
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
    assert result["applicability"] == "applicable"


def test_verifier_conservatively_normalizes_domain_in_applicability_field(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Name the ALPHA-1 integrated software",
        query="ALPHA-1 integrated software",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "trust_state": "confirmed",
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "conflicting_chunk_ids": [],
                "applicability": "software",
                "scope_entity": "ALPHA-1",
                "rationale": "The cited chunk names the integrated software.",
            },
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "ALPHA-1 integrated software: Control Suite.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert result["trust_state"] == "confirmed"
    assert result["claim_supported"] is True
    assert result["applicability"] == "unknown"


def test_verifier_blocks_unknown_firmware_applicability(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Determine whether firmware 6.0 applies to ALPHA-1",
        query="ALPHA-1 firmware 6.0 compatibility",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "trust_state": "confirmed",
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "conflicting_chunk_ids": [],
                "applicability": "unknown",
                "scope_entity": "ALPHA-1",
                "rationale": "The version is mentioned but its subject is unclear.",
            },
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "Firmware 6.0 is listed near ALPHA-1.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert result["claim_supported"] is False
    assert result["trust_state"] == "unresolved"
    assert result["applicability"] == "unknown"


def test_verifier_accepts_explicit_firmware_applicability(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Determine whether firmware 6.0 applies to ALPHA-1",
        query="ALPHA-1 firmware 6.0 compatibility",
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "trust_state": "confirmed",
                "claim_supported": True,
                "supporting_chunk_ids": ["alpha"],
                "conflicting_chunk_ids": [],
                "applicability": "applicable",
                "scope_entity": "ALPHA-1",
                "rationale": "The cited requirement explicitly binds firmware 6.0 to ALPHA-1.",
            },
            "{}",
        ),
    )

    result = verify_retrieval_claim(
        hop,
        hop.query,
        [_result("alpha", "alpha-doc", "ALPHA-1 requires firmware 6.0 or later.")],
        {"claim_supported": True, "supporting_chunk_ids": ["alpha"]},
    )

    assert result["claim_supported"] is True
    assert result["trust_state"] == "confirmed"
    assert result["applicability"] == "applicable"


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
    assert result["judge"]["attempts"][0]["error"] == "ValueError: Invalid JSON escape"
    assert result["judge"]["attempts"][1]["raw_response"] == "{}"


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


def test_source_audited_encoder_head_dependency_binds_discovered_model():
    plan = RetrievalPlan(
        mode="dependent",
        hops=[
            RetrievalHop(
                hop_id="identify_encoder_head",
                objective="Identify the encoder head supported by CA-EN100U",
                query="Which encoder head model is compatible with the CA-EN100U?",
                strategy="sparse",
            ),
            RetrievalHop(
                hop_id="find_power_source",
                objective="Find how the compatible encoder head is powered",
                query="How is the compatible encoder head powered?",
                strategy="hybrid",
                depends_on=["identify_encoder_head"],
            ),
        ],
    )
    executed_queries: list[str] = []

    def retrieve(query, _corpus_ids, _filters, _strategy, _limit):
        executed_queries.append(query)
        if len(executed_queries) == 1:
            return [
                _result(
                    "5441e6e3-1a1c-5f15-b8b9-aa0bee45b4c3",
                    "1a6783bf-01ae-4dde-8876-687064704e8c",
                    "Model: Supported encoder head; CA-EN100U: CA-EN100H",
                )
            ]
        assert "CA-EN100H" in query
        return [
            _result(
                "e050241a-00cd-539b-b306-576a34d1ccd6",
                "1a6783bf-01ae-4dde-8876-687064704e8c",
                "Model: Power-supply; CA-EN100H: Supply from CA-EN100U",
            )
        ]

    def verify(_hop, _query, results, _assessment):
        return {
            "trust_state": "confirmed",
            "claim_supported": True,
            "supporting_chunk_ids": [results[0].chunk_id],
            "conflicting_chunk_ids": [],
            "applicability": "applicable",
            "scope_entity": None,
            "rationale": "The source-audited table row directly supports this hop.",
        }

    for factory, controller_type in (
        (build_langgraph_agentic_retriever, AgenticRetrievalController),
        (build_llamaindex_agentic_retriever, LlamaIndexAgenticController),
    ):
        executed_queries.clear()
        controller = controller_type(
            use_llm=False,
            planner=lambda _query: plan,
            retriever=retrieve,
            verifier=verify,
        )
        output = _invoke(factory, controller, max_hops=2)

        assert len(executed_queries) == 2
        assert "CA-EN100H" in executed_queries[1]
        assert output["sufficient"] is True
        assert output["retrieval_trace"]["context_assembly"]["all_required_claims_retained"] is True


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

    assert refined == "What is the Description for OP26487, including the cable connector orientation?"
    assert refine_llamaindex_subquestion(hop, [prior], use_llm=True) == refined
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


def test_conflicting_structured_evidence_stops_without_recovery():
    query = (
        "On IV4-G120, how many objects are counted at one time when the count "
        "value is 9 and ON equals the set value?"
    )
    plan = RetrievalPlan(
        hops=[RetrievalHop(hop_id="lookup", objective=query, query=query, strategy="hybrid")]
    )
    results = []
    for row, value in ((3, 2), (4, 3)):
        result = _result(
            f"count-row-{row}",
            "iv4-doc",
            "Column headers: Quantity counted at one time; Row headers: ON when = "
            f"Set value > Count value= 9; Cell value: {value}; Row: {row}; Column: 3",
        )
        result.metadata.update({"chunk_type": "table_record", "product_model": "IV4-G120"})
        results.append(result)

    for factory, controller_type in (
        (build_langgraph_agentic_retriever, AgenticRetrievalController),
        (build_llamaindex_agentic_retriever, LlamaIndexAgenticController),
    ):
        retrieval_calls = 0

        def retrieve(*_args):
            nonlocal retrieval_calls
            retrieval_calls += 1
            return results

        controller = controller_type(
            use_llm=False,
            planner=lambda _query: plan,
            retriever=retrieve,
        )
        output = _invoke(factory, controller, max_hops=4)

        assert retrieval_calls == 1
        assert output["sufficient"] is False
        assert output["evidence_ledger"]["lookup"]["assessment"]["trust_state"] == "conflicting"
        assert list(output["evidence_ledger"]) == ["lookup"]


def test_verifier_only_failure_stops_without_futile_retrieval_recovery():
    plan = RetrievalPlan(
        hops=[
            RetrievalHop(
                hop_id="lookup",
                objective="Find the corrective action for alarm E17",
                query="What corrective action resolves alarm E17?",
                strategy="structural",
            )
        ]
    )
    result = _result(
        "alarm-row",
        "alarm-doc",
        "Alarm E17 corrective action: replace the failed pressure transducer.",
    )

    for factory, controller_type in (
        (build_langgraph_agentic_retriever, AgenticRetrievalController),
        (build_llamaindex_agentic_retriever, LlamaIndexAgenticController),
    ):
        retrieval_calls = 0

        def retrieve(*_args):
            nonlocal retrieval_calls
            retrieval_calls += 1
            return [result]

        controller = controller_type(
            use_llm=False,
            planner=lambda _query: plan,
            retriever=retrieve,
            verifier=lambda *_args: {
                "trust_state": "unresolved",
                "claim_supported": False,
                "supporting_chunk_ids": [],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": None,
                "rationale": "Independent verification did not resolve.",
            },
        )
        output = _invoke(factory, controller, max_hops=4)

        assert retrieval_calls == 1
        assert output["sufficient"] is False
        assert list(output["evidence_ledger"]) == ["lookup"]


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
    ]
    assert output["evidence_ledger"]["find_orientation"]["sufficient"] is True
    assert output["evidence_ledger"]["find_orientation"]["strategy"] == "structural"
    assert output["evidence_ledger"]["find_orientation"]["assessment"]["dependency_anchors"] == ["OP-26487"]
    assert "find_orientation_recovery" not in output["evidence_ledger"]


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


def test_claim_sufficiency_rejects_incidental_requested_model_in_conflicting_document_scope():
    wrong = _result(
        "wrong",
        "mod-600-doc",
        "E42 cause: a blocked inlet. Corrective action: clear the inlet. Compatible accessory for MOD-500.",
    ).model_copy(update={"metadata": {"chunk_type": "table_record", "product_model": "MOD-600"}})

    sufficient, assessment = _assess_hop_evidence(
        "Find the MOD-500 corrective action for error E42",
        [wrong],
    )

    assert sufficient is False
    assert assessment["supporting_chunk_ids"] == []
    assert assessment["result_assessments"][0]["scope_supported"] is False


def test_claim_sufficiency_keeps_only_authoritatively_scoped_row_from_mixed_results():
    wrong = _result(
        "wrong",
        "mod-600-doc",
        "E42 cause: a blocked inlet. Corrective action: clear the inlet. Compatible accessory for MOD-500.",
    ).model_copy(update={"metadata": {"chunk_type": "table_record", "product_model": "MOD-600"}})
    correct = _result(
        "correct",
        "mod-500-doc",
        "For MOD-500 error E42, corrective action: reseat the sensor cable.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "product_model": "MOD-500",
                "routing_product_models": ["MOD-500"],
            }
        }
    )

    sufficient, assessment = _assess_hop_evidence(
        "Find the MOD-500 corrective action for error E42",
        [wrong, correct],
    )

    assert sufficient is True
    assert assessment["supporting_chunk_ids"] == ["correct"]


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
    assert [item["chunk_id"] for item in output["retrieval_results"]] == ["support"]
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


def test_verifier_packet_preserves_late_condition_and_complete_row():
    from manuals_rag_answering.agentic_retrieval import _verification_evidence
    text = 'Background information. ' * 60 + '\n| Calibration | permitted only while stopped |'
    packet = _verification_evidence([_result('late', 'd1', text)], query='calibration')
    assert packet['evidence'][0]['content'] == text
    assert packet['omitted_count'] == 0


def test_verifier_packet_selects_relevant_evidence_beyond_rank_four():
    from manuals_rag_answering.agentic_retrieval import _verification_evidence
    results = [_result(str(i), 'd1', 'Unrelated background') for i in range(5)]
    results.append(_result('target', 'd2', 'Calibration requires stopped operation.'))
    packet = _verification_evidence(results, query='calibration stopped', max_bytes=1000)
    assert packet['evidence'][0]['chunk_id'] == 'target'
    assert packet['omitted_count'] > 0


def test_verifier_packet_budget_omits_whole_oversized_source():
    import json
    from manuals_rag_answering.agentic_retrieval import _verification_evidence
    results = [_result('large', 'd1', 'é' * 10000), _result('small', 'd2', 'Whole row.')]
    packet = _verification_evidence(results, max_bytes=1000)
    assert [item['chunk_id'] for item in packet['evidence']] == ['small']
    assert packet['omitted_count'] == 1
    assert len(json.dumps(packet, ensure_ascii=False).encode('utf-8')) <= 1000


def test_dependency_plan_cannot_silently_lose_links():
    import pytest
    from manuals_rag_answering.agentic_retrieval import _validate_plan
    plan = RetrievalPlan(mode='dependent', rationale='discover then resolve', hops=[
        RetrievalHop(hop_id='discover', objective='Find accessory', query='Accessory model'),
        RetrievalHop(hop_id='resolve', objective='Find rating', query='Accessory rating'),
    ])
    with pytest.raises(ValueError, match='dependency links'):
        _validate_plan(plan)
    plan.hops[1].depends_on = ['discover']
    _validate_plan(plan)


def test_comparison_planner_preserves_bare_family_and_uppercase_vs():
    plan = plan_retrieval('Compare the VS Series startup cause with the AB-200 memory error cause.', use_llm=False)
    assert plan.mode == 'parallel'
    assert len(plan.hops) == 2
    assert 'VS Series' in plan.hops[0].query
    assert 'AB-200' in plan.hops[1].query


def test_dependency_anchors_require_source_not_only_metadata():
    from manuals_rag_answering.agentic_retrieval import _dependency_anchors, _evidence_excerpt
    result = _result('source','doc','Background. ' * 80 + 'Use cable OP-100 for this port.')
    result.metadata['identifier_tokens'] = ['OP-999','OP-100']
    assert _dependency_anchors([result]) == ['OP-100']
    assert 'Use cable OP-100 for this port.' in _evidence_excerpt([result])


def test_dependent_query_keeps_requested_conditions_not_only_short_objective():
    prior = _result('component', 'discovery', 'The service part is ZX-9.')
    hop = RetrievalHop(
        hop_id='detail', objective='Find specifications',
        query='What is the calibration tolerance at 25 degrees C for that part?',
        depends_on=['discovery'],
    )
    refined = refine_dependent_query(hop, [prior], use_llm=False)
    assert 'calibration tolerance at 25 degrees C' in refined
    assert 'ZX-9' in refined


def test_dependent_hybrid_execution_preserves_query_facet_and_condition():
    plan = RetrievalPlan(mode='dependent', hops=[
        RetrievalHop(hop_id='discovery', objective='Identify service part',
                     query='Which service part is installed?', strategy='sparse'),
        RetrievalHop(hop_id='detail', objective='Find specifications',
                     query='What is the calibration tolerance at 25 degrees C for that part?',
                     strategy='hybrid', depends_on=['discovery']),
    ])
    queries = []

    def retrieve(query, *_args):
        queries.append(query)
        if len(queries) == 1:
            return [_result('component', 'discovery', 'The service part is ZX-9.')]
        return [_result('spec', 'catalog', 'ZX-9 calibration tolerance at 25 degrees C is 0.2 percent.')]

    controller = AgenticRetrievalController(
        use_llm=False, planner=lambda _query: plan, retriever=retrieve,
        refiner=lambda _hop, _results: 'ZX-9 calibration tolerance at 25 degrees C',
    )
    _invoke(build_langgraph_agentic_retriever, controller, max_hops=2)
    assert len(queries) == 2
    assert 'calibration tolerance at 25 degrees C' in queries[1]
    assert 'ZX-9' in queries[1]


def test_dependency_binding_excludes_unverified_candidates_after_recovery():
    from manuals_rag_answering.agentic_retrieval import _results_for_ids, _dependency_anchors
    plan = RetrievalPlan(mode='dependent', hops=[
        RetrievalHop(hop_id='discover', objective='Identify part', query='Which part?'),
        RetrievalHop(hop_id='recover', objective='Identify part', query='Which part?', recovery_for='discover'),
        RetrievalHop(hop_id='detail', objective='Find rating', query='Rating?', depends_on=['discover']),
    ])
    state = {
        'plan': plan.model_dump(),
        'hop_results': {
            'discover': [_result('rejected', 'wrong', 'Use part BAD-8.').model_dump()],
            'recover': [_result('confirmed', 'right', 'Use part GOOD-9.').model_dump(),
                        _result('noise', 'other', 'Other part NOISE-7.').model_dump()],
        },
        'evidence_ledger': {
            'discover': {'sufficient': False, 'assessment': {'supporting_chunk_ids': []}},
            'recover': {'sufficient': True, 'assessment': {'supporting_chunk_ids': ['confirmed']}},
        },
    }
    selected = _results_for_ids(state, ['discover'])
    assert [result.chunk_id for result in selected] == ['confirmed']
    assert _dependency_anchors(selected) == ['GOOD-9']
