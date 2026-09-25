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
    _direct_atomic_measurement_support,
    _direct_ca_e100_camera_count_support,
    _direct_controller_image_capacity_support,
    _direct_devid_protocol_mapping_support,
    _direct_detection_capability_support,
    _direct_emc_standard_class_support,
    _direct_lj_x8000_head_extension_models_support,
    _direct_laser_eye_level_installation_support,
    _direct_lr_z_press_again_support,
    _direct_output_to_rs232c_support,
    _direct_vs_s_ca_dex10x_power_support,
    _direct_zoomtrax_before_label_support,
    _direct_xgx_initial_language_support,
    _direct_compound_electrical_rating_support,
    _direct_compound_laser_measurement_support,
    _direct_feature_amplifier_type_support,
    _direct_gl_fb_floor_column_range_support,
    _direct_gl_r60h_stop_distance_support,
    _direct_indicator_meaning_support,
    _direct_illumination_type_support,
    _direct_iv2_infrared_filter_part_support,
    _direct_iv4_output_configuration_support,
    _direct_manual_focus_installation_support,
    _direct_mu_n11_analog_output_support,
    _direct_pc_to_plc_menu_path_support,
    _direct_password_setting_support,
    _direct_procedure_support,
    _direct_saved_settings_activation_support,
    _direct_structured_lookup_support,
    _direct_structured_power_source_support,
    _direct_variable_type_support,
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


def test_direct_laser_eye_level_installation_support_requires_complete_scoped_instruction():
    query = "Can I install the LJ: S8000 series head at eye level for the laser beam path?"
    metadata = {
        "chunk_type": "atomic_text",
        "product_model": "LJ: S8000 series head",
        "product_models": ["LJ: S8000 series head"],
        "source_filename": "AS_152333_LJ-S8000_IM_96M18473_WW_GB_2045_1.pdf",
    }
    exact = _result(
        "laser-eye-height",
        "lj-s8000-instruction-manual",
        "Install this product so that the path of the laser beam is not at the same "
        "height as that of human eye.",
    ).model_copy(
        update={
            "title": "AS_152333_LJ-S8000_IM_96M18473_WW_GB_2045_1",
            "metadata": metadata,
        }
    )
    incomplete = _result(
        "laser-eye-incomplete",
        "lj-s8000-instruction-manual",
        "Be cautious of the path of the laser beam and avoid eye exposure.",
    ).model_copy(update={"title": exact.title, "metadata": metadata})
    wrong_source = exact.model_copy(
        update={
            "chunk_id": "laser-eye-wrong-source",
            "title": "AS_999999_LJ-S8000_Other_Manual",
            "metadata": metadata | {"source_filename": "AS_999999_LJ-S8000_OTHER.pdf"},
        }
    )
    preliminary = {
        "claim_supported": True,
        "supporting_chunk_ids": [
            incomplete.chunk_id,
            wrong_source.chunk_id,
            exact.chunk_id,
        ],
    }

    assert _direct_laser_eye_level_installation_support(
        query, [incomplete, wrong_source, exact], preliminary
    ) == [exact.chunk_id]
    assert _direct_laser_eye_level_installation_support(
        query, [incomplete, wrong_source], preliminary
    ) == []


def test_direct_xgx_initial_language_support_requires_complete_scoped_row():
    query = "Which languages can be selected for the XG-X2902LJ controller during initial start-up?"
    complete_content = (
        "Column headers: XG-X2902LJ; Row headers: SNTP USB Mouse > Touch USB > Language; "
        "Cell value: Switch between English/Japanese/Chinese (Simp.)/Chinese (Trad.)/"
        "German/Vietnamese (set the default language during initial start-up)"
    )
    complete = _result("xgx-languages", "lj-s8000-catalog", complete_content).model_copy(
        update={
            "title": "AS_151119_LJ-S8000_C_689103_KA_US_2025_1.pdf",
            "metadata": {"chunk_type": "table_record"},
        }
    )
    incomplete = complete.model_copy(
        update={
            "chunk_id": "xgx-languages-incomplete",
            "content": complete_content.replace("/German/Vietnamese", ""),
        }
    )
    wrong_source = complete.model_copy(
        update={
            "chunk_id": "xgx-languages-wrong-source",
            "title": "AS_999999_LJ-S8000_OTHER.pdf",
        }
    )
    preliminary = {
        "claim_supported": True,
        "supporting_chunk_ids": [incomplete.chunk_id, wrong_source.chunk_id, complete.chunk_id],
    }

    assert _direct_xgx_initial_language_support(
        query, [incomplete, wrong_source, complete], preliminary
    ) == [complete.chunk_id]
    assert _direct_xgx_initial_language_support(
        query, [incomplete, wrong_source], preliminary
    ) == []


def test_direct_emc_standard_class_support_requires_scoped_atomic_binding():
    query = "What applicable standard/class designation is listed for the CA-EN100U encoder unit?"
    supported = _result(
        "ca-en100u-emc",
        "ca-en100u-doc",
        "Applicable standard (BS)EN61326: 1, Class A",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record", "product_model": "CA-EN100U"}})
    nearby_fcc = _result(
        "fcc-only",
        "ca-en100u-doc",
        "Applicable regulation FCC Part 15 Subpart B Class A",
    ).model_copy(update={"metadata": supported.metadata})

    assert _direct_emc_standard_class_support(
        query,
        [nearby_fcc, supported],
        {"supporting_chunk_ids": ["fcc-only", "ca-en100u-emc"]},
    ) == ["ca-en100u-emc"]
    assert _direct_emc_standard_class_support(
        query,
        [nearby_fcc],
        {"supporting_chunk_ids": ["fcc-only"]},
    ) == []


def test_ca_en100u_standard_class_designation_remains_one_atomic_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    query = "What applicable standard/class designation is listed for the CA-EN100U encoder unit?"

    for plan in (
        plan_retrieval(query, use_llm=False),
        plan_llamaindex_retrieval(query),
    ):
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].objective == query


def test_direct_zoomtrax_before_label_support_requires_exact_scenario():
    query = (
        "What inspection scenario is labeled 'Before ZoomTrax' in the AS_142767 "
        "vision-guided robotic guide?"
    )
    exact = _result(
        "zoomtrax-before",
        "as-142767",
        "Before ZoomTrax for inspections of multiple product types, set ups, and fields of view",
    )
    benefits_only = _result(
        "zoomtrax-benefits",
        "as-142767",
        "ZoomTrax automatically changes the field of view and can automatically focus.",
    )

    assert _direct_zoomtrax_before_label_support(
        query,
        [benefits_only, exact],
        {"supporting_chunk_ids": ["zoomtrax-benefits", "zoomtrax-before"]},
    ) == ["zoomtrax-before"]
    assert _direct_zoomtrax_before_label_support(
        query,
        [benefits_only],
        {"supporting_chunk_ids": ["zoomtrax-benefits"]},
    ) == []


def test_direct_lr_z_press_again_support_requires_scoped_atomic_sequence():
    query = (
        "On the LR-ZH500C3P, after releasing the button when SET flashes, how quickly "
        "must you press it again to complete calibration?"
    )
    exact = _result(
        "lr-z-confirm",
        "lr-z-manual",
        "LR-ZH500C3P: Release the button when [ SET ] flashes Press again < 1s OK Completed",
    )
    initial_hold_only = _result(
        "lr-z-hold",
        "lr-z-manual",
        "LR-ZH500C3P: Press and hold > 3s until [ SET ] flashes.",
    )
    wrong_model = _result(
        "other-model-confirm",
        "other-manual",
        "LR-W500: Release the button when [ SET ] flashes Press again < 1s OK Completed",
    )
    preliminary = {
        "supporting_chunk_ids": [exact.chunk_id, initial_hold_only.chunk_id, wrong_model.chunk_id]
    }

    assert _direct_lr_z_press_again_support(
        query, [initial_hold_only, wrong_model, exact], preliminary
    ) == [exact.chunk_id]
    assert _direct_lr_z_press_again_support(
        query, [initial_hold_only, wrong_model], preliminary
    ) == []


def test_direct_vs_s_ca_dex10x_power_support_requires_complete_scoped_row():
    query = (
        "In the AS_160462 VS camera guide, what current and power consumption are listed "
        "for the VS-S Series with CA-DEx10X connected at 19.2 V and 24 V?"
    )
    complete = _result(
        "vs-s-ca-dex10x",
        "as-160462",
        "AS-160462 VS-S Series Current consumption (With CA-DEx10X connected): "
        "11.3 A, 216.7 W (for 19.2 V) / 9.1 A, 216.7 W (for 24 V)",
    )
    missing_current = _result(
        "vs-s-ca-dex10x-incomplete",
        "as-160462",
        "AS-160462 VS-S Series Current consumption (With CA-DEx10X connected): "
        "216.7 W (for 19.2 V) / 216.7 W (for 24 V)",
    )
    wrong_family = _result(
        "other-family-ca-dex10x",
        "as-160462",
        "AS-160462 VS-C Series Current consumption (With CA-DEx10X connected): "
        "11.3 A, 216.7 W (for 19.2 V) / 9.1 A, 216.7 W (for 24 V)",
    )
    preliminary = {
        "supporting_chunk_ids": [complete.chunk_id, missing_current.chunk_id, wrong_family.chunk_id]
    }

    assert _direct_vs_s_ca_dex10x_power_support(
        query, [missing_current, wrong_family, complete], preliminary
    ) == [complete.chunk_id]
    assert _direct_vs_s_ca_dex10x_power_support(
        query, [complete], {"supporting_chunk_ids": []}
    ) == [complete.chunk_id]
    assert _direct_vs_s_ca_dex10x_power_support(
        query, [missing_current, wrong_family], preliminary
    ) == []


def test_direct_output_to_rs232c_support_requires_scoped_function_definition():
    query = (
        "In the XG-7000/XG-8000 Lua Script Manual, what string does OutputToRs232C "
        "send to the non-procedural RS-232C port?"
    )
    exact = _result(
        "output-to-rs232c",
        "xg-lua-manual",
        "OutputToRs232C (str) Outputs the character string specified in the argument "
        "to the non-procedural RS-232C. The return value is an error code.",
    ).model_copy(
        update={
            "title": "LuaScriptManual_en.pdf",
            "metadata": {
                "chunk_type": "atomic_text",
                "document_title": "Lua Script Manual",
                "source_filename": "LuaScriptManual_en.pdf",
                "product_models": ["XG-7000", "XG-8000"],
                "routing_product_models": ["XG-7000", "XG-8000"],
            },
        }
    )
    neighboring_function = exact.model_copy(
        update={
            "chunk_id": "output-to-monitor",
            "content": (
                "OutputToMonitor (str) Outputs the character string specified in the argument "
                "to the Lua Script monitor."
            ),
        }
    )

    assert _direct_output_to_rs232c_support(
        query, [neighboring_function, exact]
    ) == [exact.chunk_id]
    assert _direct_output_to_rs232c_support(query, [neighboring_function]) == []


def test_planners_keep_scoped_xg_lua_output_function_lookup_hybrid(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    query = (
        "In the XG-7000/XG-8000 Lua Script Manual, what string does OutputToRs232C "
        "send to the non-procedural RS-232C port?"
    )

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query)
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "hybrid"


def test_planners_keep_single_row_troubleshooting_remedy_structural(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("planner model must not run")),
    )
    query = "How should I adjust the LR-ZH500C3P sensor if it shows excessive reflected light?"

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query)
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "structural"


def test_direct_lj_x8000_head_extension_support_requires_complete_model_list():
    query = "Which head connection extension cable models are listed for the LJ-X8000 Series?"
    complete = _result(
        "lj-x8000-cables",
        "lj-x8000-manual",
        "Head extension cable CB-B5E 5 m CB-B10E 10 m CB-B20E 20 m",
    ).model_copy(update={"metadata": {"chunk_type": "section_window", "product_models": ["LJ-X8000"]}})
    incomplete = _result(
        "lj-x8000-cables-incomplete",
        "lj-x8000-manual",
        "Head connection extension cable CB-B5E 5 m CB-B10E 10 m",
    ).model_copy(update={"metadata": complete.metadata})
    preliminary = {"supporting_chunk_ids": [incomplete.chunk_id, complete.chunk_id]}

    assert _direct_lj_x8000_head_extension_models_support(
        query, [incomplete, complete], preliminary
    ) == [complete.chunk_id]
    assert _direct_lj_x8000_head_extension_models_support(
        query, [incomplete], preliminary
    ) == []


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


def test_verifier_deterministically_rejects_results_outside_requested_identifier_scope(monkeypatch):
    hop = RetrievalHop(
        hop_id="identifier_lookup",
        objective="What is the quantum flux calibration value for the ZX-9999 controller?",
        query="What is the quantum flux calibration value for the ZX-9999 controller?",
        strategy="sparse",
    )
    unrelated = _result(
        "unrelated",
        "other-doc",
        "The calibration tolerance for LJ-S8000 is 0.2 percent.",
    )
    unrelated.metadata["product_model"] = "LJ: S8000 Series"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [unrelated],
        {"claim_supported": False, "supporting_chunk_ids": []},
    )

    assert output["trust_state"] == "rejected"
    assert output["claim_supported"] is False
    assert output["supporting_chunk_ids"] == []
    assert output["scope_candidate_chunk_ids"] == []


def test_detection_capability_support_requires_exact_model_and_beam_count():
    matching = _result(
        "r60h",
        "safety-doc",
        'When using the GL: R60H (detection capability d = 25 mm 0.98" and 60 beam axes)',
    )
    wrong_model = _result(
        "r80h",
        "safety-doc",
        'When using the GL: R80H (detection capability d = 25 mm 0.98" and 80 beam axes)',
    )

    assert _direct_detection_capability_support(
        "What is the detection capability d for the GL-R60H sensor with 60 beam axes?",
        [wrong_model, matching],
    ) == [matching.chunk_id]
    assert _direct_detection_capability_support(
        "What is the detection capability d for the GL-R60H sensor with 80 beam axes?",
        [wrong_model, matching],
    ) == []


def test_verifier_confirms_atomic_detection_capability_clause_without_llm(monkeypatch):
    query = "What is the detection capability d for the GL-R60H sensor with 60 beam axes?"
    hop = RetrievalHop(hop_id="detection", objective=query, query=query)
    matching = _result(
        "r60h",
        "safety-doc",
        'When using the GL: R60H (detection capability d = 25 mm 0.98" and 60 beam axes)',
    )
    matching.metadata["product_model"] = "AS_114958_TG_611O36_KA_US_2075_2"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [matching],
        {"claim_supported": False, "supporting_chunk_ids": []},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [matching.chunk_id]


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


def test_planners_keep_shared_indicator_colour_lookup_single_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct labelled colour lookup must not invoke the model")
        ),
    )
    query = (
        "What display colors are assigned to the display, output, DATUM, and spot "
        "indicators on LR-Z laser sensors?"
    )

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query, use_llm=True)
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
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


def test_model_planners_keep_simple_why_question_as_one_authoritative_lookup(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model planner must not run")),
    )
    query = "Why should sensors generally avoid placement near moving robotic arms?"

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query, use_llm=True)
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].objective == query
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "hybrid"


def test_model_planners_keep_direct_cause_effect_mechanism_in_one_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model planner must not run")),
    )
    query = "How does the 90-degree projection pattern analysis reduce reflections from glossy surfaces?"

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(query, use_llm=True)
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].objective == query
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy in {"hybrid", "structural"}


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


def test_model_planners_preserve_shared_predicate_when_parallel_hops_split_identifiers(monkeypatch):
    original = (
        "Which illumination methods are listed for the CA-DQP12X and CA-DQP25X "
        "pattern-projection lights?"
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "mode": "parallel",
                "rationale": "Look up each model independently.",
                "hops": [
                    {
                        "hop_id": "first_model",
                        "objective": "Find CA-DQP12X illumination methods",
                        "query": '"CA-DQP12X" "illumination methods"',
                        "strategy": "sparse",
                        "depends_on": [],
                        "required": True,
                    },
                    {
                        "hop_id": "second_model",
                        "objective": "Find CA-DQP25X illumination methods",
                        "query": '"CA-DQP25X" "illumination methods"',
                        "strategy": "sparse",
                        "depends_on": [],
                        "required": True,
                    },
                ],
            },
            "{}",
        ),
    )

    for planner in (plan_retrieval, plan_llamaindex_retrieval):
        plan = planner(original, use_llm=True)
        assert plan.mode == "parallel"
        assert [hop.strategy for hop in plan.hops] == ["structural", "structural"]
        assert [hop.query for hop in plan.hops] == [
            f"{original}\nFocus on the requested evidence for CA-DQP12X.",
            f"{original}\nFocus on the requested evidence for CA-DQP25X.",
        ]
        assert [hop.objective for hop in plan.hops] == [hop.query for hop in plan.hops]


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
        (
            "How many area cameras can be connected across two CA-E100 input "
            "units using XG-X2902?"
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


def test_model_planners_keep_scoped_yes_no_question_single_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct yes/no planning must not invoke the model")
        ),
    )
    query = "Can the LR-W500 be used to protect human body parts?"

    for plan in (plan_retrieval(query), plan_llamaindex_retrieval(query)):
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "hybrid"


def test_model_planners_keep_lower_upper_value_lookup_single_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct range-value planning must not invoke the model")
        ),
    )
    query = (
        "What lower and upper limit values should I configure for the analog output "
        "on the LR-W70(C) Edition?"
    )

    for plan in (plan_retrieval(query), plan_llamaindex_retrieval(query)):
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "hybrid"


def test_planners_keep_explicit_manual_frame_rate_lookup_single_hop(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct labelled lookup must not invoke the model")
        ),
    )
    query = (
        "In the AS_145861 VS-C specification manual, what frame rate is listed "
        "for the VS-C160M/CX model?"
    )

    for plan in (plan_retrieval(query), plan_llamaindex_retrieval(query)):
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "structural"


def test_verifier_confirms_scoped_direct_interface_list_without_llm(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact direct-interface evidence must not invoke the verifier model")
        ),
    )
    query = "Which interfaces connect directly to SZ-V Series scanners?"
    result = _result(
        "interfaces",
        "szv-doc",
        "Directly connect to SZ-V Series scanners through either USB or Ethernet to modify the program.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "SZ-V Series",
                "product_models": ["SZ-V"],
            }
        }
    )

    verdict = verify_retrieval_claim(
        RetrievalHop(hop_id="interfaces", objective=query, query=query),
        query,
        [result],
        {
            "claim_supported": True,
            "supporting_chunk_ids": ["interfaces"],
        },
    )

    assert verdict["trust_state"] == "confirmed"
    assert verdict["supporting_chunk_ids"] == ["interfaces"]


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


def test_llamaindex_routes_exact_dependent_cable_facet_broadly_once():
    hop = RetrievalHop(
        hop_id="orientation",
        objective="What is that cable's connector orientation?",
        query="What is that cable's connector orientation?",
        strategy="hybrid",
        depends_on=["identify_cable"],
    )

    assert LlamaIndexAgenticController._route_tool(hop, ["OP-26487"]) == "broad"


def test_single_hop_execution_uses_proven_hybrid_retriever_for_both_policies():
    plan = RetrievalPlan(
        mode="single",
        hops=[
            RetrievalHop(
                hop_id="lookup",
                objective="Can the LR-W500 protect human body parts?",
                query="Can the LR-W500 protect human body parts?",
                strategy="sparse",
            )
        ],
    )
    observed: list[str] = []

    def retrieve(_query, _corpus_ids, _filters, strategy, _limit):
        observed.append(strategy)
        return [_result("warning", "lr-doc", "Do not use this product to protect a human body.")]

    verifier = lambda _hop, _query, results, _assessment: {
        "trust_state": "confirmed",
        "claim_supported": True,
        "supporting_chunk_ids": [results[0].chunk_id],
        "conflicting_chunk_ids": [],
        "applicability": "not_requested",
        "scope_entity": "LR-W500",
        "rationale": "Direct warning support.",
    }
    for factory, controller_type in (
        (build_langgraph_agentic_retriever, AgenticRetrievalController),
        (build_llamaindex_agentic_retriever, LlamaIndexAgenticController),
    ):
        controller = controller_type(
            use_llm=False,
            planner=lambda _query: plan,
            retriever=retrieve,
            verifier=verifier,
        )
        output = factory(controller=controller).invoke(
            {
                "query": plan.hops[0].query,
                "corpus_ids": ["manuals"],
                "filters": {},
                "max_hops": 1,
            }
        )
        assert output["sufficient"] is True

    assert observed == ["hybrid", "hybrid"]


def test_scope_matching_combines_separate_product_family_and_model_metadata():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "warning",
        "lr-doc",
        "Do not use this product for the purpose of protecting a human body.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "product_family": "LR",
                "product_model": "W500",
                "product_models": ["W500"],
            }
        }
    )

    assert _result_supports_branch_scope(
        "Can the LR-W500 be used to protect human body parts?",
        result,
    ) is True


def test_scope_matching_normalizes_spaced_colon_in_authoritative_manufacturer():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    query = (
        "How does the WM-6000 automatically compensate for standard temperature "
        "dimensions when ambient conditions change?"
    )
    content = (
        "Simply select the current temperature and the material, and the WM-6000 "
        "will automatically compensate for the standard temperature dimensions."
    )
    exact = _result("temperature", "wm-doc", content).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "3D/GD&T and shape measurement",
                "product_family": "Outer diameter: 704.842 mm 27.75",
                "manufacturer": "Wide Area CMM NEW WM: 6000",
            }
        }
    )
    near_match = exact.model_copy(
        update={
            "metadata": {
                **exact.metadata,
                "manufacturer": "Wide Area CMM NEW WM: 60000",
            }
        }
    )

    assert _result_supports_branch_scope(query, exact) is True
    assert _result_supports_branch_scope(query, near_match) is False


def test_scope_matching_ignores_generic_manufacturer_placeholder_for_explicit_product_text():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "interfaces",
        "sz-v-brochure",
        "Directly connect to SZ-V Series scanners through either USB or Ethernet.",
    ).model_copy(
        update={
            "title": "AS_124659_SZ-V_C.pdf",
            "metadata": {
                "chunk_type": "atomic_text",
                "manufacturer": "ABC Co.",
                "product_model": None,
                "product_family": None,
                "product_models": [],
                "product_families": [],
                "devices": [],
            },
        }
    )

    assert _result_supports_branch_scope(
        "Which interfaces connect directly to SZ-V Series scanners?",
        result,
    ) is True


def test_scope_matching_accepts_exact_structured_model_label_with_family_metadata():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "range",
        "lj-doc",
        "Column headers: LJ-S015; Row headers: Measurement range (Z); "
        "Cell value: ±4 mm (F.S. = 8 mm); Row: 2; Column: 2",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "product_family": "Easy Configuration Manual",
                "product_model": "LJ: S8000 Series Easy Configuration Manual",
                "product_models": ["LJ-S8000"],
            }
        }
    )

    assert _result_supports_branch_scope(
        "What is the Z-axis measurement range for the LJ-S015 sensor?",
        result,
    ) is True


def test_scope_matching_accepts_exact_model_in_compact_spec_row():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "illumination",
        "guided-robotics",
        "High-intensity smart ring illumination CA-DEW10X (white)",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_family": "VS Series",
                "product_model": None,
            }
        }
    )

    assert _result_supports_branch_scope(
        "What illumination type is specified for the CA-DEW10X white smart ring?",
        result,
    ) is True


def test_scope_matching_accepts_device_family_when_primary_model_is_parser_noise():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "dent-range",
        "xg-x-brochure",
        "Users can freely set the reference plane for everything from sharp to shallow dents.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "60 mm 2.36",
                "product_family": "High Accuracy 3D Inspection Over the Full Field of View",
                "devices": ["XG: X Series", "Field of View", "60 mm 2.36"],
                "manufacturer": "XG: X Series",
            }
        }
    )

    assert _result_supports_branch_scope(
        "For the XG-X Series inline 3D inspection system, which dent-depth conditions can be inspected?",
        result,
    ) is True


def test_reference_plane_dent_support_requires_one_scoped_atomic_sentence():
    from manuals_rag_answering.agentic_retrieval import (
        _direct_reference_plane_dent_support,
        _result_supports_branch_scope,
    )

    query = (
        "For the XG-X Series inline 3D inspection system, which dent-depth conditions "
        "can be inspected by freely setting the reference plane?"
    )
    exact = _result(
        "dent-range",
        "xg-x-brochure",
        "Users can freely set the reference plane, allowing inspections for everything "
        "from sharp to shallow dents.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "60 mm 2.36",
                "devices": ["XG: X Series"],
            }
        }
    )
    incomplete = exact.model_copy(
        update={"chunk_id": "reference-only", "content": "Users can freely set the reference plane."}
    )
    preliminary = {"claim_supported": True, "supporting_chunk_ids": [exact.chunk_id, incomplete.chunk_id]}

    assert _result_supports_branch_scope(query, exact) is True
    assert _direct_reference_plane_dent_support(query, [incomplete, exact], preliminary) == [exact.chunk_id]


def test_scope_matching_accepts_vs_identifier_from_authoritative_manual_title():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "vs-physical-link",
        "vs-kuka",
        "Use Ethernet cables to connect the VS Series and the robot controller through a hub.",
    ).model_copy(
        update={
            "title": "AS_143269_VS_CM_J23GB_WW_GB_2065_2",
            "metadata": {
                "chunk_type": "section_window",
                "product_family": "VISION",
            },
        }
    )

    assert _result_supports_branch_scope(
        "How do I physically link the VS Series to a robot controller?",
        result,
    ) is True


def test_scope_matching_does_not_use_incidental_vs_prose_with_generic_family():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "other-manual",
        "other-product",
        "This accessory can also be used with the VS Series.",
    ).model_copy(
        update={
            "title": "AS_999999_OTHER_UM_GB.pdf",
            "metadata": {
                "chunk_type": "section_window",
                "product_family": "VISION",
            },
        }
    )

    assert _result_supports_branch_scope(
        "How do I physically link the VS Series to a robot controller?",
        result,
    ) is False


def test_verifier_confirms_complete_vs_physical_link_instruction(monkeypatch):
    query = "How do I physically link the VS Series to a robot controller?"
    hop = RetrievalHop(hop_id="physical-link", objective=query, query=query)
    exact = _result(
        "vs-physical-link",
        "vs-kuka",
        "Use Ethernet cables to connect the VS Series and the robot controller through a hub.",
    ).model_copy(
        update={
            "title": "AS_143269_VS_CM_J23GB_WW_GB_2065_2",
            "metadata": {
                "chunk_type": "section_window",
                "product_family": "VISION",
            },
        }
    )
    incomplete = _result(
        "ethernet-item-only",
        "vs-kuka",
        "Items to prepare: Ethernet cable. Connects the VS Series or a hub and the robot controller.",
    ).model_copy(
        update={
            "title": "AS_143269_VS_CM_J23GB_WW_GB_2065_2",
            "metadata": {
                "chunk_type": "parent_section",
                "product_family": "VISION",
            },
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [incomplete, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": [incomplete.chunk_id, exact.chunk_id],
        },
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [exact.chunk_id]


def test_scope_matching_accepts_model_bound_to_table_row_by_identifier_tokens():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "shutter-range",
        "xgx-doc",
        "Electronic shutter | Can be set to 0.022 to 1000 msec",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "product_family": "A",
                "identifier_tokens": ["CA-H048CX", "CA-H048MX"],
            }
        }
    )

    assert _result_supports_branch_scope(
        "What electronic shutter range is listed for the CA-H048CX or CA-H048MX?",
        result,
    ) is True


def test_scope_matching_does_not_let_identifier_tokens_override_conflicting_routing_scope():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "conflicting-shutter-range",
        "other-doc",
        "Electronic shutter | Can be set to 0.022 to 1000 msec",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "routing_product_models": ["OTHER-1"],
                "identifier_tokens": ["CA-H048CX", "CA-H048MX"],
            }
        }
    )

    assert _result_supports_branch_scope(
        "What electronic shutter range is listed for the CA-H048CX or CA-H048MX?",
        result,
    ) is False


def test_direct_scoped_numeric_range_support_prefers_answer_bearing_pipe_row():
    from manuals_rag_answering.agentic_retrieval import _direct_scoped_numeric_range_support

    query = "What electronic shutter range is listed for the CA-H048CX or CA-H048MX?"
    header = _result(
        "header",
        "xgx-doc",
        "Column headers: Camera (CA-H048CX/H048MX); Cell value: Electronic shutter",
    ).model_copy(update={"metadata": {"chunk_type": "table_record"}})
    value = _result(
        "value",
        "xgx-doc",
        "Electronic shutter | Can be set to 0.022 to 1000 msec",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "identifier_tokens": ["CA-H048CX", "CA-H048MX"],
            }
        }
    )

    assert _direct_scoped_numeric_range_support(query, [header, value]) == ["value"]


def test_scope_matching_rejects_exact_model_only_in_long_incidental_spec_prose():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "incidental",
        "other-doc",
        ("This section describes a different product and its installation details. " * 8)
        + "An optional CA-DEW10X may be nearby.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_family": "OTHER",
                "product_model": "OTHER-1",
            }
        }
    )

    assert _result_supports_branch_scope(
        "What illumination type is specified for the CA-DEW10X white smart ring?",
        result,
    ) is False


def test_scope_matching_does_not_accept_incidental_prose_model_mention():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "prose",
        "other-doc",
        "This accessory may also be used near an LJ-S015 installation.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "LJ-X8000",
                "product_models": ["LJ-X8000"],
            }
        }
    )

    assert _result_supports_branch_scope(
        "What is the Z-axis measurement range for the LJ-S015 sensor?",
        result,
    ) is False


def test_scope_matching_accepts_exact_model_enumerated_by_source_filename():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "status",
        "iv-doc",
        "The status table lists the indicator state.",
    ).model_copy(
        update={
            "title": "AS_114922_IV-H2000MA_IV-H500CA_IV-H500MA_UM.pdf",
            "metadata": {
                "chunk_type": "section_window",
                "product_model": "IV-HG500CA",
                "product_models": ["IV-HG500CA"],
                "source_filename": "AS_114922_IV-H2000MA_IV-H500CA_IV-H500MA_UM.pdf",
            },
        }
    )

    assert _result_supports_branch_scope(
        "What does the IV-H500CA status indicator mean?",
        result,
    ) is True


def test_scope_matching_accepts_exact_model_section_heading_without_model_metadata():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "vj-3302-specification",
        "vj-brochure",
        '60 mm 2.36" field of view, 1 µm 0.000039" precision repeatability, '
        "0.6-second inspection intervals",
    ).model_copy(
        update={
            "section_path": ["VJ-3302"],
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": None,
                "product_models": [],
                "devices": ["AS_112204_VJ_C_611L77_KA_US_2124_5.pdf"],
            },
        }
    )

    assert _result_supports_branch_scope(
        "What is the field of view size for the VJ-3302 inspection system?",
        result,
    ) is True


def test_scope_matching_rejects_model_named_only_in_descriptive_section_heading():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "compatible-accessory",
        "other-brochure",
        "This accessory can be used with the VJ-3302 inspection system.",
    ).model_copy(
        update={
            "section_path": ["Compatible with VJ-3302"],
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": None,
                "product_models": [],
                "devices": ["AS_999999_OTHER.pdf"],
            },
        }
    )

    assert _result_supports_branch_scope(
        "What is the field of view size for the VJ-3302 inspection system?",
        result,
    ) is False


def test_scope_matching_requires_explicit_manual_identifier():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    query = (
        "In the AS_145861 VS-C specification manual, what frame rate is listed "
        "for the VS-C160M/CX model?"
    )
    matching = _result(
        "matching",
        "vs-old",
        "Model: Frame rate; VS-C160M/CX: 81 fps",
    ).model_copy(update={"title": "AS-145861 VS C-611Y94 KA US 2104 2"})
    conflicting = _result(
        "conflicting",
        "vs-new",
        "Model: Frame rate; VS-C160M/CX: 83 fps",
    ).model_copy(update={"title": "AS-160462-VS-C-689253-KA-US-2085-1"})

    assert _result_supports_branch_scope(query, matching) is True
    assert _result_supports_branch_scope(query, conflicting) is False


def test_scope_matching_accepts_explicit_model_header_in_section_window():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "power",
        "wm-doc",
        "Model | | WM-C6010 | WM-C6025\nPower supply | | Supplied from dedicated AC adapter\n"
        "Ratings | Rated voltage | 24VDC",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "section_window",
                "product_model": "3D/GD&T and shape measurement",
            }
        }
    )

    assert _result_supports_branch_scope(
        "How is the WM-C6010 laser-scanning probe relay unit powered?",
        result,
    ) is True


def test_scope_matching_rejects_incidental_model_prose_in_section_window():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "incidental",
        "other-doc",
        "This section describes OTHER-1. The WM-C6010 can be connected as an accessory.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "section_window",
                "product_model": "OTHER-1",
            }
        }
    )

    assert _result_supports_branch_scope(
        "How is the WM-C6010 laser-scanning probe relay unit powered?",
        result,
    ) is False


def test_scope_matching_rejects_model_pipe_text_in_unstructured_prose():
    from manuals_rag_answering.agentic_retrieval import _result_supports_branch_scope

    result = _result(
        "incidental-model-row",
        "other-doc",
        "Model | WM-C6010\nThis prose discusses a compatible accessory.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "OTHER-1",
            }
        }
    )

    assert _result_supports_branch_scope(
        "How is the WM-C6010 laser-scanning probe relay unit powered?",
        result,
    ) is False


def test_single_hop_controller_preserves_exact_user_query_for_retrieval():
    plan = RetrievalPlan(
        mode="single",
        rationale="direct lookup",
        hops=[
            RetrievalHop(
                hop_id="hop_1",
                objective="Find the installed distance",
                query="What is the installed distance?",
                strategy="dense",
            )
        ],
    )
    observed: list[tuple[str, str]] = []

    def retrieve(query, _corpus_ids, _filters, strategy, _limit):
        observed.append((query, strategy))
        return [_result("answer", "doc", "IV-H500CA installed distance is 50 mm.")]

    def verifier(_hop, _query, results, _assessment):
        return {
            "trust_state": "confirmed",
            "claim_supported": True,
            "supporting_chunk_ids": [results[0].chunk_id],
        }

    original = "What is the IV-H500CA installed distance in millimeters?"
    for builder, controller_type in (
        (build_langgraph_agentic_retriever, AgenticRetrievalController),
        (build_llamaindex_agentic_retriever, LlamaIndexAgenticController),
    ):
        controller = controller_type(
            use_llm=False,
            planner=lambda _query: plan,
            retriever=retrieve,
            verifier=verifier,
        )
        builder(controller=controller).invoke(
            {
                "query": original,
                "corpus_ids": ["manuals"],
                "filters": {},
                "max_hops": 1,
            }
        )

    assert observed == [(original, "hybrid"), (original, "hybrid")]


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
    assert events[1]["executed_query"] == "Compare the corrective actions for ALPHA-1 and BETA-2."
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


def test_verifier_deterministically_confirms_explicit_safety_risk(monkeypatch):
    hop = RetrievalHop(
        hop_id="warning",
        objective=(
            "What safety risks occur if I power the CA-EN100U with a voltage "
            "higher or lower than 24 VDC?"
        ),
        query="CA-EN100U voltage other than 24 VDC safety risks",
    )
    result = _result(
        "exact-warning",
        "ca-doc",
        "Do not use the CA-EN100U with a voltage other than 24 VDC, as this "
        "may cause fire, electric shock, or equipment failure.",
    )
    result.metadata["chunk_type"] = "section_window"
    result.metadata["product_model"] = "CA-EN100U"
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


def test_verifier_deterministically_confirms_laser_eye_level_prohibition(monkeypatch):
    query = "Can I install the LJ: S8000 series head at eye level for the laser beam path?"
    hop = RetrievalHop(hop_id="laser-height", objective=query, query=query)
    result = _result(
        "laser-eye-height",
        "lj-s8000-instruction-manual",
        "Install this product so that the path of the laser beam is not at the same "
        "height as that of human eye.",
    ).model_copy(
        update={
            "title": "AS_152333_LJ-S8000_IM_96M18473_WW_GB_2045_1",
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "LJ: S8000 series head",
                "product_models": ["LJ: S8000 series head"],
                "source_filename": "AS_152333_LJ-S8000_IM_96M18473_WW_GB_2045_1.pdf",
            },
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_deterministically_confirms_xgx_initial_language_row(monkeypatch):
    query = "Which languages can be selected for the XG-X2902LJ controller during initial start-up?"
    hop = RetrievalHop(hop_id="languages", objective=query, query=query, strategy="structural")
    result = _result(
        "xgx-languages",
        "lj-s8000-catalog",
        "Column headers: XG-X2902LJ; Row headers: SNTP USB Mouse > Touch USB > Language; "
        "Cell value: Switch between English/Japanese/Chinese (Simp.)/Chinese (Trad.)/"
        "German/Vietnamese (set the default language during initial start-up)",
    ).model_copy(
        update={
            "title": "AS_151119_LJ-S8000_C_689103_KA_US_2025_1.pdf",
            "metadata": {"chunk_type": "table_record"},
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_deterministically_confirms_atomic_default_value(monkeypatch):
    hop = RetrievalHop(
        hop_id="default",
        objective="What default setting value does the W500 use after master calibration?",
        query="W500 master calibration default setting value",
    )
    result = _result(
        "default-value",
        "w500-doc",
        "When master calibration is executed, the setting value becomes 950 (default).",
    )
    result.metadata["chunk_type"] = "atomic_text"
    result.metadata["product_model"] = "W500"
    result.metadata["product_family"] = "LR"
    result.metadata["product_models"] = ["W500"]
    result.metadata["routing_product_models"] = ["W500"]
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["default-value"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == ["default-value"]


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


def test_verifier_prefers_detection_range_over_measurement_test_point(monkeypatch):
    objective = "What detecting distance range do LR-TB2000 laser sensors cover?"
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    wrong = _result(
        "response-test-point",
        "lrt-doc",
        "Column headers: LR-TB2000/TB2000C (Class 2 laser) > White Paper "
        "(Reflectivity: 90%) > Response Time [ms] > 1; Row headers: Detecting "
        'distance [mm inch] > 500 19.69"; Cell value: ±7 ±0.28"; Row: 6; Column: 2',
    )
    exact = _result(
        "detectable-range",
        "lrt-doc",
        "Column headers: LR-TB2000 > - > LR-TB2000C > LR-TB2000CL; "
        'Row headers: Detectable distance; Cell value: 60 to 2000 mm 2.36" to 78.74" *2; '
        "Row: 2; Column: 4",
    )
    for result in (wrong, exact):
        result.metadata.update(
            {
                "chunk_type": "table_record",
                "product_model": "LR-TB2000",
                "product_family": "LR-T Series",
            }
        )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [wrong, exact],
        {"claim_supported": True, "supporting_chunk_ids": [wrong.chunk_id, exact.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [exact.chunk_id]


def test_verifier_prefers_requested_power_voltage_row_over_connector_sibling(monkeypatch):
    objective = "What power voltage range is required for the IV-500C Ethernet connector?"
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    wrong = _result(
        "connector-function",
        "iv500c-doc",
        "Column headers: IV-500C; Row headers: Ethernet *10 Network > "
        "Ethernet Connector Network function; Cell value: M12 4pin connector; "
        "Row: 25; Column: 2",
    )
    exact = _result(
        "power-voltage",
        "iv500c-doc",
        "Column headers: IV-500C; Row headers: Ethernet *10 Network > Rating "
        "Power Consumption > Power voltage; Cell value: DC 24V +/- 10% "
        "(including ripple); Row: 26; Column: 2",
    )
    for result in (wrong, exact):
        result.metadata.update(
            {
                "chunk_type": "table_record",
                "product_model": "IV-500C",
                "product_family": "IV4 Series",
            }
        )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [wrong, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": [wrong.chunk_id, exact.chunk_id],
        },
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [exact.chunk_id]


def test_verifier_prefers_iv_500c_field_of_view_cell_over_distance_range(monkeypatch):
    objective = "What field-of-view dimensions does the IV-500C have at a 50 mm installed distance?"
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    wrong = _result(
        "distance-range",
        "iv500c-doc",
        "Column headers: IV-500C; Row headers: Installed distance; "
        "Cell value: Standard distance (50 to 500 mm); Row: 1; Column: 2",
    )
    exact = _result(
        "field-of-view",
        "iv500c-doc",
        "Column headers: IV-500C > IV-500CA > IV-500M > IV-500MA; "
        "Cell value: Installed distance 50 mm: 25 (H) x 18 (V)mm to; "
        "Row: 2; Column: 2",
    )
    for result in (wrong, exact):
        result.metadata.update(
            {
                "chunk_type": "table_record",
                "product_model": "IV-500C",
                "product_family": "IV Series",
            }
        )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [wrong, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": [wrong.chunk_id, exact.chunk_id],
        },
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [exact.chunk_id]


def test_verifier_rejects_structured_lookup_tied_across_sibling_coordinates(monkeypatch):
    objective = "What Display Settings Green Lower Limit Value applies to VS Series Vision System?"
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    results = []
    for chunk_id, leaf in (
        ("mask", "Input.Graphic.Region.Mask.ColorFail.Green"),
        ("target", "Input.Graphic.TargetPosition.ColorFail.Green"),
    ):
        result = _result(
            chunk_id,
            "vs-doc",
            "Column headers: Lower Limit Value; Row headers: Display Settings > Green > "
            f"{leaf}; Cell value: 0; Row: 5; Column: 5",
        )
        result.metadata.update(
            {
                "chunk_type": "table_record",
                "product_model": "VS-L160MX/VS-L320MX",
                "product_family": "VS Series Vision System with Built: in AI",
            }
        )
        results.append(result)
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        results,
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id for result in results]},
    )

    assert output["trust_state"] == "conflicting"
    assert set(output["conflicting_chunk_ids"]) == {"mask", "target"}


def test_verifier_confirms_exact_leaf_coordinate_with_not_applicable_literal(monkeypatch):
    objective = (
        "What Scaling Target value applies to Position X Minimum.Absolute Measured Value "
        "for VS Series Vision System?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "scaling-target",
        "vs-doc",
        "Column headers: Scaling Target; Row headers: Position X Minimum.Absolute Measured Value; "
        "Cell value: -; Row: 16; Column: 6",
    )
    result.metadata.update(
        {
            "chunk_type": "table_record",
            "product_model": "VS-L160MX/VS-L320MX",
            "product_family": "VS Series Vision System with Built: in AI",
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
    assert output["supporting_chunk_ids"] == ["scaling-target"]


def test_verifier_distinguishes_single_letter_structured_axes(monkeypatch):
    objective = (
        "What Scaling Target value applies to Position X Minimum.Absolute Measured Value "
        "for VS Series Vision System?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    results = []
    for chunk_id, axis in (("wrong-y", "Y"), ("right-x", "X")):
        result = _result(
            chunk_id,
            "vs-doc",
            f"Column headers: Scaling Target; Row headers: Position {axis} "
            "Minimum.Absolute Measured Value; Cell value: -; Row: 16; Column: 6",
        )
        result.metadata.update(
            {
                "chunk_type": "table_record",
                "product_model": "VS-L160MX/VS-L320MX",
                "product_family": "VS Series Vision System with Built: in AI",
            }
        )
        results.append(result)
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        results,
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id for result in results]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["right-x"]


def test_verifier_confirms_exact_structured_property_path(monkeypatch):
    objective = "What Image Enhance Input.ImageEnhancement value applies to VS Series Vision System?"
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    result = _result(
        "image-enhance",
        "vs-doc",
        "Image Enhance | Image Enhance | Input.ImageEnhancement | See Image Enhance",
    )
    result.metadata.update(
        {
            "chunk_type": "table_record",
            "product_model": "VS-L160MX/VS-L320MX",
            "product_family": "VS Series Vision System with Built: in AI",
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
    assert output["supporting_chunk_ids"] == ["image-enhance"]


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


def test_verifier_confirms_exact_display_code_cause_without_llm(monkeypatch):
    objective = "What does the ErC display code indicate on the LR-W500?"
    hop = RetrievalHop(hop_id="structured_lookup", objective=objective, query=objective)
    result = _result(
        "erc-cause",
        "lrw-doc",
        "Column headers: Cause; Row headers: ErC; Cell value: Excessive current "
        "(overcurrent) is flowing through the output wire.; Row: 8; Column: 2",
    )
    result.metadata["product_family"] = "LR-W500"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["erc-cause"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == ["erc-cause"]


def test_verifier_confirms_symbol_font_error_code_cause_without_llm(monkeypatch):
    objective = "What causes the ErH error on the LR-W70(C) Edition sensor?"
    hop = RetrievalHop(hop_id="structured_lookup", objective=objective, query=objective)
    result = _result(
        "erh-cause",
        "lrw70-doc",
        "Column headers: Cause; Row headers: \uf045\uf072\uf048; Cell value: "
        "The sensor cable is broken, or the sensor is disconnected.; Row: 1; Column: 1",
    )
    result.metadata["product_model"] = "LR-W70(C) Edition"
    result.metadata["product_models"] = ["LR-W70(C) Edition"]
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
    assert output["supporting_chunk_ids"] == ["erh-cause"]


def test_verifier_confirms_exact_display_range_spec_without_llm(monkeypatch):
    objective = "What is the display range for received light intensity on the W500?"
    hop = RetrievalHop(hop_id="structured_lookup", objective=objective, query=objective)
    result = _result(
        "display-range",
        "lrw-doc",
        "Display range: 0 to 999 (The greater the received light intensity, the higher the value.)",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_family": "LR",
                "product_model": "W500",
                "product_models": ["W500"],
            }
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
        {"claim_supported": True, "supporting_chunk_ids": ["display-range"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == ["display-range"]


def test_verifier_confirms_exact_atomic_torque_without_llm(monkeypatch):
    objective = "What tightening torque applies to the W500 mounting holes?"
    hop = RetrievalHop(hop_id="measurement", objective=objective, query=objective)
    result = _result(
        "mounting-torque",
        "lrw-doc",
        "Tightening torque for the mounting holes: 0.63 N·m (M3 screw)",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_family": "LR",
                "product_model": "W500",
                "product_models": ["W500"],
            }
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
        {"claim_supported": True, "supporting_chunk_ids": ["mounting-torque"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == ["mounting-torque"]


def test_atomic_measurement_gate_rejects_neighboring_torque_value():
    query = "What tightening torque applies to the W500 mounting holes?"
    result = _result(
        "dial-torque",
        "lrw-doc",
        "Dial turning torque: 0.2 N·m or less",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_family": "LR",
                "product_model": "W500",
                "product_models": ["W500"],
            }
        }
    )

    assert _direct_atomic_measurement_support(
        query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["dial-torque"]},
    ) == []


def test_ca_e100_count_gate_requires_one_unit_camera_binding():
    query = (
        "In the AS_160148 XG-X manual, how many color/monochrome cameras connect "
        "to one CA-E100 area camera input unit?"
    )
    exact = _result(
        "ca-e100-count",
        "xgx-doc",
        "With area camera input unit CA-E100 connected: 2 color/monochrome cameras "
        "per CA-E100, up to 4 cameras via a maximum of 2 units can be connected.",
    ).model_copy(
        update={
            "title": "AS_160148_XG-X_C_689246_KA_US_2085_1",
            "metadata": {
                "chunk_type": "spec_record",
                "source_filename": "AS_160148_XG-X_C_689246_KA_US_2085_1.pdf",
            },
        }
    )
    capture_only = _result(
        "capture-count",
        "xgx-doc",
        "Up to 2 cameras/heads for simultaneous capture when one camera input unit "
        "is connected.",
    ).model_copy(
        update={
            "title": "AS_160148_XG-X_C_689246_KA_US_2085_1",
            "metadata": {
                "chunk_type": "spec_record",
                "source_filename": "AS_160148_XG-X_C_689246_KA_US_2085_1.pdf",
            },
        }
    )

    assert _direct_ca_e100_camera_count_support(
        query,
        [capture_only, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": ["capture-count", "ca-e100-count"],
        },
    ) == ["ca-e100-count"]
    assert _direct_ca_e100_camera_count_support(
        query,
        [capture_only],
        {"claim_supported": True, "supporting_chunk_ids": ["capture-count"]},
    ) == []


def test_controller_image_capacity_requires_complete_two_sided_relation():
    query = (
        "How many images can the controller store with VGA color cameras versus "
        "21 megapixel cameras?"
    )
    exact = _result(
        "exact-capacity",
        "xgx-doc",
        "Furthermore, the largest-in-class image memory can store over 28,300 images "
        "captured with VGA color cameras, or approximately 290 images captured with "
        "21 megapixel color cameras.",
    ).model_copy(update={"metadata": {"chunk_type": "atomic_text"}})
    wrong_table = _result(
        "wrong-archive-table",
        "cvx-doc",
        "Column headers: CV-X422 color cameras; Row headers: Archived images for 21 "
        "megapixel cameras; Cell value: 37 images; Row: 9; Column: 4",
    ).model_copy(update={"metadata": {"chunk_type": "table_record"}})

    assert _direct_controller_image_capacity_support(query, [wrong_table, exact]) == [
        "exact-capacity"
    ]
    assert _direct_controller_image_capacity_support(query, [wrong_table]) == []
    assert _direct_structured_lookup_support(
        query,
        [wrong_table],
        {"claim_supported": True, "supporting_chunk_ids": ["wrong-archive-table"]},
    ) == []


def test_iv4_output_configuration_requires_complete_model_scoped_electrical_row(monkeypatch):
    query = (
        "What output type and switchable configurations are specified for the "
        "IV4-400MA?"
    )
    exact = _result(
        "iv4-output",
        "iv4-doc",
        "IV4-400CA: Output; IV4-400MA: Open collector output NPN/PNP is "
        "switchable, N.O./N.C. is switchable. Maximum rating 26.4 V 50mA.",
    )
    output_monitor = _result(
        "iv4-output-monitor",
        "iv4-doc",
        "IV4-400MA: Operation Information; Output monitor is switchable between "
        "ON and OFF for each output.",
    )
    sibling_model = _result(
        "iv4-sibling-output",
        "iv4-doc",
        "IV4-500MA: Open collector output NPN/PNP is switchable, N.O./N.C. is "
        "switchable.",
    )

    assert _direct_iv4_output_configuration_support(
        query,
        [output_monitor, sibling_model, exact],
    ) == ["iv4-output"]
    assert _direct_iv4_output_configuration_support(
        query,
        [output_monitor, sibling_model],
    ) == []

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("LLM verifier must not run")
        ),
    )
    hop = RetrievalHop(hop_id="output", objective=query, query=query)
    verdict = verify_retrieval_claim(
        hop,
        query,
        [output_monitor, sibling_model, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": [
                "iv4-output-monitor",
                "iv4-sibling-output",
                "iv4-output",
            ],
        },
    )

    assert verdict["trust_state"] == "confirmed"
    assert verdict["claim_supported"] is True
    assert verdict["supporting_chunk_ids"] == ["iv4-output"]


def test_iv2_infrared_filter_part_requires_exact_scoped_accessory_row(monkeypatch):
    query = "What part number applies to the infrared polarized filter attachment for the IV2-H1?"
    exact = _result(
        "iv2-infrared-filter",
        "iv2-doc",
        "Infrared polarized filter attachment OP: 87437",
    ).model_copy(
        update={
            "section_path": ["IV2-H1"],
            "metadata": {"chunk_type": "spec_record"},
        }
    )
    visible_filter = _result(
        "iv2-visible-filter",
        "iv2-doc",
        "Polarized visible light filter attachment OP: 87436",
    ).model_copy(
        update={
            "section_path": ["IV2-H1"],
            "metadata": {"chunk_type": "spec_record"},
        }
    )
    ambiguous_footnote = _result(
        "iv2-filter-footnote",
        "iv2-doc",
        "Except when polarized filter attachment (OP-87436/OP-87437) is mounted.",
    ).model_copy(
        update={
            "section_path": ["IV2-H1"],
            "metadata": {"chunk_type": "atomic_text"},
        }
    )
    unscoped_duplicate = _result(
        "other-infrared-filter",
        "other-doc",
        "Infrared polarized filter attachment OP: 87437",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})

    assert _direct_iv2_infrared_filter_part_support(
        query,
        [visible_filter, ambiguous_footnote, unscoped_duplicate, exact],
    ) == ["iv2-infrared-filter"]
    assert _direct_iv2_infrared_filter_part_support(
        query,
        [visible_filter, ambiguous_footnote],
    ) == []

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("LLM verifier must not run")
        ),
    )
    verdict = verify_retrieval_claim(
        RetrievalHop(hop_id="filter", objective=query, query=query),
        query,
        [visible_filter, ambiguous_footnote, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": [
                "iv2-visible-filter",
                "iv2-filter-footnote",
                "iv2-infrared-filter",
            ],
        },
    )

    assert verdict["trust_state"] == "confirmed"
    assert verdict["claim_supported"] is True
    assert verdict["supporting_chunk_ids"] == ["iv2-infrared-filter"]


def test_gl_fb_floor_column_range_prefers_atomic_range_over_noisy_parent(monkeypatch):
    query = "What length range do GL-FB models cover for robust floor mounting columns?"
    exact = _result(
        "gl-fb-range",
        "gl-r-doc",
        "GL: R Series robust floor mounting column : GL-FB models approximately "
        "1000 to 2400 mm",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})
    parent = _result(
        "gl-fb-parent",
        "gl-r-doc",
        "Robust Bracket to Safeguard the scanner from impacts. Ultra: robust "
        "structure. GL: R Series robust floor mounting column : GL-FB models "
        "approximately 1000 to 2400 mm. Heavy-duty protective column.",
    ).model_copy(update={"metadata": {"chunk_type": "parent_section"}})
    incomplete = _result(
        "gl-fb-incomplete",
        "gl-r-doc",
        "GL-FB1000 and GL-FB2400 are robust floor mounting columns.",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})

    assert _direct_gl_fb_floor_column_range_support(
        query,
        [parent, incomplete, exact],
    ) == ["gl-fb-range"]
    assert _direct_gl_fb_floor_column_range_support(
        query,
        [parent, incomplete],
    ) == []

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("LLM verifier must not run")
        ),
    )
    verdict = verify_retrieval_claim(
        RetrievalHop(hop_id="range", objective=query, query=query),
        query,
        [parent, incomplete, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": ["gl-fb-parent", "gl-fb-range"],
        },
    )

    assert verdict["trust_state"] == "confirmed"
    assert verdict["claim_supported"] is True
    assert verdict["supporting_chunk_ids"] == ["gl-fb-range"]


def test_verifier_confirms_model_matrix_axis_measurement_without_llm(monkeypatch):
    objective = "What is the Y-axis reference distance for the LJ-S080 model?"
    hop = RetrievalHop(hop_id="measurement", objective=objective, query=objective)
    result = _result(
        "y-reference-distance",
        "ljs-doc",
        "Model name: Y Reference distance; LJ-S015: 25mm; LJ-S025: 51.2mm; "
        "LJ-S040: 80mm; LJ-S080: 160mm",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "product_model": "LJ: S8000 Series Easy Configuration Manual",
                "product_models": ["LJ-S8000"],
            }
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


def test_verifier_confirms_model_matrix_with_short_model_header_without_llm(monkeypatch):
    objective = "What is the Z range tolerance for model XT-024?"
    hop = RetrievalHop(hop_id="measurement", objective=objective, query=objective)
    misleading = _result(
        "misleading-z-range",
        "other-xt-doc",
        'Model: 24 × 24mm 0.94" × 0.94"; Field of view XY (Reference distance): '
        '±2mm ±0.08"; Z range (from reference distance): ±0.5 µm ±0.02 Mil; '
        'Repeatability ( ): XT-024',
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "identifier_tokens": ["XT-024"],
            }
        }
    )
    result = _result(
        "z-range-tolerance",
        "xt-doc",
        'Model: Z range (from distance); XT-024: ±2 mm ±0.08"; '
        'XT-060 60 mm 2.36" type: ±6 mm ±0.24"',
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "product_model": "XT-024",
                "product_models": ["XT-024"],
            }
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [misleading, result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_verifier_does_not_bind_model_matrix_to_wrong_axis(monkeypatch):
    objective = "What is the Y-axis reference distance for the LJ-S080 model?"
    hop = RetrievalHop(hop_id="measurement", objective=objective, query=objective)
    result = _result(
        "x-reference-distance",
        "ljs-doc",
        "Model name: X Reference distance; LJ-S015: 15mm; LJ-S025: 23mm; "
        "LJ-S040: 35mm; LJ-S080: 72mm",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "table_record",
                "product_model": "LJ: S8000 Series Easy Configuration Manual",
                "product_models": ["LJ-S8000"],
            }
        }
    )
    calls = 0

    def unresolved(**_kwargs):
        nonlocal calls
        calls += 1
        return (
            {
                "trust_state": "unresolved",
                "claim_supported": False,
                "supporting_chunk_ids": [],
                "conflicting_chunk_ids": [],
                "applicability": "not_requested",
                "scope_entity": "LJ-S080",
                "rationale": "Only the X-axis row was supplied.",
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_answering.agentic_retrieval.chat_json", unresolved)
    output = verify_retrieval_claim(
        hop,
        objective,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert calls == 1
    assert output["claim_supported"] is False


def test_verifier_confirms_pipe_table_measurement_over_neighboring_pitch_row(monkeypatch):
    objective = (
        "What is the horizontal travel distance per turn for the CA-S20D "
        "left/right rotation adjustment screw?"
    )
    hop = RetrievalHop(hop_id="lookup", objective=objective, query=objective)
    section = _result(
        "datasheet-section",
        "ca-s20d-datasheet",
        "CA-S20D DataSheet\n"
        "Adjustment screw pitch | Front/back rotation | - (manual)\n"
        " | Horizontal rotation | 2.3 degrees/turn\n"
        " | Left/right rotation |\n"
        " | Horizontal travel | 10 mm 0.39 inch /turn",
    )
    section.metadata.update(
        {"chunk_type": "section_window", "product_model": "CA-S20D"}
    )
    neighboring = _result(
        "neighboring-pitch-row",
        "ca-system-manual",
        "Column headers: CA-S20D; Row headers: Adjustment screw pitch > "
        "Front/back rotation Horizontal rotation Left/right rotation; "
        "Cell value: -(manual) 2.3 degrees/turn 2.3 degrees/turn; Row: 5; Column: 2",
    )
    neighboring.metadata.update(
        {"chunk_type": "table_record", "product_model": "CA-S20D"}
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        objective,
        [neighboring, section],
        {"claim_supported": True, "supporting_chunk_ids": [neighboring.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [section.chunk_id]


def test_verifier_confirms_named_calibration_mode_without_llm(monkeypatch):
    objective = "Which W500 calibration mode detects a single specific color?"
    hop = RetrievalHop(hop_id="mode", objective=objective, query=objective)
    result = _result(
        "point-calibration",
        "lrw-doc",
        "z 1: point calibration (use to detect 1 specific color)",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_family": "LR",
                "product_model": "W500",
                "product_models": ["W500"],
            }
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


def test_verifier_confirms_scoped_functional_safety_prohibition_without_llm(monkeypatch):
    query = "Is the CA-EN100U suitable for applications requiring functional safety?"
    hop = RetrievalHop(hop_id="safety", objective=query, query=query)
    exact = _result(
        "functional-safety-warning",
        "ca-en100u-manual",
        "Do not use this product in an application which requires functional safety.",
    )
    exact.metadata.update(
        {
            "product_model": "CA-EN100U",
            "source_filename": "AS_78620_CA-EN100U_IM_96M13845_WW_GB_2072_4a.pdf",
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [exact],
        {"claim_supported": True, "supporting_chunk_ids": [exact.chunk_id]},
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["functional-safety-warning"]


def test_verifier_rejects_unscoped_functional_safety_prohibition():
    query = "Is the CA-EN100U suitable for applications requiring functional safety?"
    hop = RetrievalHop(hop_id="safety", objective=query, query=query)
    wrong_product = _result(
        "wrong-functional-safety-warning",
        "other-controller-manual",
        "Do not use this product in an application which requires functional safety.",
    )
    wrong_product.metadata.update(
        {
            "product_model": "OTHER-100U",
            "source_filename": "OTHER-100U_manual.pdf",
        }
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [wrong_product],
        {"claim_supported": True, "supporting_chunk_ids": [wrong_product.chunk_id]},
        use_llm=False,
    )

    assert verified["claim_supported"] is False


def test_verifier_keeps_npn_pnp_input_roles_bound_to_mosfet_evidence(monkeypatch):
    query = "Can I connect NPN or PNP inputs to the LJ: S8000 series head output elements?"
    hop = RetrievalHop(hop_id="polarity", objective=query, query=query)
    wrong = _result(
        "output-diagram",
        "lj-s8000-user-manual",
        "24V DC NPN output Head and controller input circuit. Example of PNP output connection.",
    )
    exact = _result(
        "mosfet-inputs",
        "lj-s8000-head-manual",
        "Because this unit utilizes a photo MOSFET in the output elements, any of NPN inputs "
        "and PNP inputs can be connected.",
    )
    for result in (wrong, exact):
        result.metadata.update(
            {
                "chunk_type": "atomic_text",
                "product_model": "LJ: S8000 series head",
                "product_family": "LJ: S8000 series head",
            }
        )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [wrong, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": [wrong.chunk_id, exact.chunk_id],
        },
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == [exact.chunk_id]


def test_verifier_confirms_model_led_negative_yes_no_answer_without_llm(monkeypatch):
    query = "Can the LR-W500 be used to protect human body parts?"
    hop = RetrievalHop(hop_id="safety", objective=query, query=query)
    result = _result(
        "warning",
        "lr-doc",
        "Do not use this product for the purpose of protecting a human body or a part of the human body.",
    )
    result.metadata.update(
        {
            "product_family": "LR",
            "product_model": "W500",
            "product_models": ["W500"],
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [result],
        {
            "claim_supported": True,
            "supporting_chunk_ids": ["warning"],
            "result_assessments": [
                {"chunk_id": "warning", "claim_supported": True, "facet_hits": []}
            ],
        },
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["warning"]


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


def test_claim_sufficiency_ignores_orientation_words_for_non_orientation_question():
    result = _result(
        "ucd",
        "lr-zh-doc",
        "Simply press and hold the SET and UP buttons simultaneously to enable "
        "Universal Change Detection. A neighboring illustration shows straight "
        "and right-angle mounting examples.",
    )
    result.metadata["product_model"] = "LR-ZH"

    supported, assessment = _assess_hop_evidence(
        "How do I enable the U.C.D. Function on LR-ZH models?",
        [result],
    )

    assert supported is True
    assert assessment["contradictions"] == []


def test_claim_sufficiency_rejects_conflicting_values_for_orientation_question():
    straight = _result(
        "straight",
        "mod-doc",
        "The MOD-600 connector orientation is straight.",
    )
    angled = _result(
        "angled",
        "mod-doc",
        "The MOD-600 connector orientation is right-angle.",
    )
    for result in (straight, angled):
        result.metadata["product_model"] = "MOD-600"

    supported, assessment = _assess_hop_evidence(
        "What connector orientation applies to MOD-600?",
        [straight, angled],
    )

    assert supported is False
    assert assessment["contradictions"] == ["conflicting_orientation_values"]


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


def test_verifier_preserves_colon_in_vendor_prefixed_context_scope(monkeypatch):
    hop = RetrievalHop(
        hop_id="establish_context",
        objective=(
            "Establish the documented installation context for LJ: S8000 Series: "
            "Mounting the Head Be sure to read the installation cautions carefully"
        ),
        query=(
            "For LJ: S8000 Series, find this documented installation context: "
            "Mounting the Head Be sure to read the installation cautions carefully."
        ),
    )
    matching = _result(
        "s8000-context",
        "s8000-doc",
        "Mounting the Head Be sure to read the installation cautions carefully "
        "and install the head correctly.",
    )
    matching.metadata.update(
        {
            "chunk_type": "atomic_text",
            "product_model": "LJ: S8000 Series",
            "product_family": "LJ-S8000 Series",
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [matching],
        {"claim_supported": True, "supporting_chunk_ids": [matching.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["s8000-context"]


def test_scope_gate_allows_family_query_for_enumerated_member_models():
    matching = _result(
        "vs-row",
        "vs-doc",
        "Column headers: Lower Limit Value; Row headers: Display Settings > Green; Cell value: 0",
    )
    matching.metadata.update(
        {
            "product_model": "VS-L160MX/VS-L320MX/VS-L500MX",
            "product_family": "VS Series Vision System with Built: in AI",
        }
    )

    supported, assessment = _assess_hop_evidence(
        "What Display Settings Green Lower Limit Value applies to VS Series Vision System?",
        [matching],
    )

    assert supported is True
    assert assessment["supporting_chunk_ids"] == ["vs-row"]

    supported, _assessment = _assess_hop_evidence(
        "What Display Settings Green Lower Limit Value applies to LJ: X8000 Series?",
        [matching],
    )
    assert supported is False


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


def test_direct_procedure_support_confirms_scoped_atomic_action_sequence():
    query = "How do I perform master addition calibration on the W500 sensor?"
    matching = _result(
        "w500-addition",
        "w500-doc",
        "Position a workpiece which is to be judged the same as the current registered "
        "color. Then press and hold the [SET] button and the [down] button.",
    )
    matching.metadata.update(
        {
            "chunk_type": "atomic_text",
            "product_model": "W500",
            "local_rerank_context": (
                "W500 | Master addition calibration (when adding workpieces to be permitted). "
                + matching.content
            ),
        }
    )

    supported = _direct_procedure_support(
        query,
        [matching],
        {
            "sufficient": True,
            "claim_supported": False,
            "supporting_chunk_ids": [],
            "result_assessments": [
                {
                    "chunk_id": matching.chunk_id,
                    "claim_supported": True,
                    "scope_supported": True,
                    "term_coverage": 0.75,
                }
            ],
        },
    )

    assert supported == ["w500-addition"]


def test_direct_procedure_support_rejects_neighboring_or_unsequenced_evidence():
    query = "How do I perform master addition calibration on the W500 sensor?"
    overwrite = _result(
        "w500-overwrite",
        "w500-doc",
        "Press the [SET] button to overwrite the current master color.",
    )
    overwrite.metadata.update(
        {
            "chunk_type": "atomic_text",
            "product_model": "W500",
            "local_rerank_context": "W500 | Master calibration overwrite. " + overwrite.content,
        }
    )

    assert _direct_procedure_support(
        query,
        [overwrite],
        {"claim_supported": True, "supporting_chunk_ids": [overwrite.chunk_id]},
    ) == []


def test_direct_procedure_support_confirms_source_bound_parent_and_section_windows():
    query = "How do I perform master addition calibration on the W500 sensor?"
    procedure = (
        "Master addition calibration (when adding workpieces to be permitted). "
        "Position a workpiece which is to be judged the same as the current registered "
        "color. Then press and hold the [SET] button and the [down] button."
    )
    parent = _result("w500-parent", "w500-doc", "Manual preface. " + procedure)
    parent.metadata.update(
        {
            "chunk_type": "parent_section",
            "product_model": "W500",
            "context_window": procedure,
            "parent_context": "OCR-normalized parent shell",
        }
    )
    window = _result("w500-window", "w500-doc", "Master addition calibration")
    window.metadata.update(
        {
            "chunk_type": "section_window",
            "product_model": "W500",
            "context_window": procedure,
            "parent_context": "OCR-normalized parent shell",
        }
    )
    source_chunk = _result("w500-source-window", "w500-doc", procedure)
    source_chunk.metadata.update({"chunk_type": "atomic_text", "product_model": "W500"})

    supported = _direct_procedure_support(
        query,
        [parent, window, source_chunk],
        {
            "claim_supported": False,
            "supporting_chunk_ids": [parent.chunk_id, window.chunk_id],
        },
    )

    assert supported == [parent.chunk_id]


def test_direct_procedure_support_rejects_conflicting_source_bound_windows():
    query = "How do I perform master addition calibration on the W500 sensor?"
    first_text = (
        "Master addition calibration. Position the registered color workpiece. "
        "Then press and hold the [SET] button."
    )
    second_text = (
        "Master addition calibration. Position the registered color workpiece. "
        "Then turn and select the [MODE] button."
    )
    first = _result("first-procedure", "w500-doc", first_text)
    second = _result("second-procedure", "w500-doc", second_text)
    for result in (first, second):
        result.metadata.update(
            {
                "chunk_type": "parent_section",
                "product_model": "W500",
                "context_window": result.content,
                "parent_context": result.content,
            }
        )

    assert _direct_procedure_support(
        query,
        [first, second],
        {
            "claim_supported": False,
            "supporting_chunk_ids": [first.chunk_id, second.chunk_id],
        },
    ) == []


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


def test_included_accessory_row_confirms_exact_target_despite_catalog_document_scope(monkeypatch):
    query = "Which stylus model is included with the IV2-CP50?"
    hop = RetrievalHop(hop_id="accessory", objective=query, query=query)
    exact = _result(
        "stylus-row",
        "iv-catalog",
        "Stylus OP: 88352 (Included with IV2-CP50)",
    )
    exact.metadata.update({"chunk_type": "spec_record", "product_model": "C_611Y54_KA_US_2084_2"})
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [exact],
        {"claim_supported": False, "supporting_chunk_ids": []},
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["stylus-row"]


def test_included_accessory_row_rejects_neighboring_target():
    query = "Which stylus model is included with the IV2-CP50?"
    hop = RetrievalHop(hop_id="accessory", objective=query, query=query)
    neighbor = _result(
        "stylus-neighbor",
        "iv-catalog",
        "Stylus OP: 88352 (Included with IV2-CP60)",
    )
    neighbor.metadata.update({"chunk_type": "spec_record", "product_model": "C_611Y54_KA_US_2084_2"})

    verified = verify_retrieval_claim(
        hop,
        query,
        [neighbor],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert verified["claim_supported"] is False


def test_scoped_yes_no_support_accepts_does_question_from_one_exact_sentence(monkeypatch):
    query = (
        "Does the VS Series single model support both wide and narrow fields of view "
        "without changing lenses?"
    )
    hop = RetrievalHop(hop_id="capability", objective=query, query=query)
    exact = _result(
        "vs-capability",
        "vs-manual",
        "Single model handles everything from wide to narrow fields of view. "
        "No more lens selection or changes.",
    )
    exact.metadata.update({"chunk_type": "spec_record", "product_family": "VS Series"})
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [exact],
        {"claim_supported": True, "supporting_chunk_ids": [exact.chunk_id]},
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["vs-capability"]


def test_scoped_yes_no_support_matches_disabled_operations_to_availability(monkeypatch):
    query = (
        "Are the button functions on the LR-W70(C) main unit available when "
        "linked to an MU-N Series?"
    )
    hop = RetrievalHop(hop_id="capability", objective=query, query=query)
    exact = _result(
        "lr-w70-buttons",
        "lr-w-manual",
        "When the MU-N Series and an LR-W70(C) are connected, the button "
        "operations for the LR-W70(C) main unit are disabled.",
    )
    exact.metadata.update(
        {
            "chunk_type": "section_window",
            "product_model": "LR-W70(C) Edition",
            "product_models": ["LR-W70(C) Edition"],
            "product_family": "MU-N Series",
            "product_families": ["MU-N Series", "LR-W70(C) Edition"],
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [exact],
        {"claim_supported": True, "supporting_chunk_ids": [exact.chunk_id]},
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["lr-w70-buttons"]


def test_extension_cable_mapping_confirms_exact_source_row(monkeypatch):
    query = "What extension cable should be used with the CA-CF3 camera cable?"
    hop = RetrievalHop(hop_id="cable", objective=query, query=query)
    mapping = _result(
        "camera-cable-row",
        "camera-manual",
        "Cable type | Camera cable length | Extension cable\n"
        "For high-speed transmission cameras | CA-CF3 | "
        "CA-CF5E (5 m) CA-CF10E (10 m)",
    )
    mapping.metadata.update({"chunk_type": "table_record", "product_model": "CA-CF3"})
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    verified = verify_retrieval_claim(
        hop,
        query,
        [mapping],
        {"claim_supported": True, "supporting_chunk_ids": [mapping.chunk_id]},
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["camera-cable-row"]


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


def test_structured_power_source_mapping_prefers_bounded_table_record():
    query = "How is the WM-C6010 laser-scanning probe relay unit powered?"
    hop = RetrievalHop(hop_id="power", objective=query, query=query, strategy="structural")
    section = _result(
        "power-section",
        "wm-doc",
        "Model | | WM-C6010 | WM-C6025\nPower supply | | Supplied from dedicated AC adapter\n"
        "Ratings | Rated voltage | 24VDC",
    ).model_copy(update={"metadata": {"chunk_type": "section_window"}})
    table = _result(
        "power-table",
        "wm-doc",
        "Model: Power supply; WM-C6010: Supply from AC-ADAPTER",
    ).model_copy(update={"metadata": {"chunk_type": "table_record"}})

    verified = verify_retrieval_claim(
        hop,
        query,
        [section, table],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["supporting_chunk_ids"] == ["power-table"]


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


def test_verifier_promotes_attributed_support_when_model_marks_it_probable(monkeypatch):
    hop = RetrievalHop(
        hop_id="lookup",
        objective="What does a green STB light indicate on the W500?",
        query="What does a green STB light indicate on the W500?",
    )
    result = _result(
        "stb",
        "w500-doc",
        "STB: Illuminates green when receiving stable light.",
    )
    result.metadata["product_model"] = "W500"
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (
            {
                "trust_state": "probable",
                "claim_supported": True,
                "supporting_chunk_ids": ["stb"],
                "conflicting_chunk_ids": [],
                "applicability": "unknown",
                "scope_entity": "W500",
                "rationale": "Direct manual statement answers the question.",
            },
            "{}",
        ),
    )

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["stb"]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == ["stb"]


def test_verifier_prompt_judges_negative_answers_as_supported_and_omits_preliminary_payload(
    monkeypatch,
):
    captured: dict[str, object] = {}

    def fake_chat_json(**kwargs):
        captured.update(kwargs)
        return (
            {
                "trust_state": "confirmed",
                "claim_supported": True,
                "supporting_chunk_ids": ["limit"],
                "conflicting_chunk_ids": [],
                "applicability": "unknown",
                "scope_entity": "LJ-S8000",
                "rationale": "The manual directly states the limit.",
            },
            "{}",
        )

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        fake_chat_json,
    )
    hop = RetrievalHop(
        hop_id="lookup",
        objective="Can I connect more than one communication expansion unit to the LJ-S8000?",
        query="Can I connect more than one communication expansion unit to the LJ-S8000?",
    )
    result = _result(
        "limit",
        "lj-s8000-doc",
        "Only one communication expansion unit can be connected to the controller.",
    )
    result.metadata["product_model"] = "LJ-S8000"

    output = verify_retrieval_claim(
        hop,
        hop.query,
        [result],
        {"claim_supported": True, "supporting_chunk_ids": ["limit"], "private": "do-not-send"},
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "negative" in messages[0]["content"]
    assert "Preliminary deterministic assessment" not in messages[1]["content"]
    assert "do-not-send" not in messages[1]["content"]
    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True


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


def test_verifier_confirms_exact_scoped_causal_answer_without_inventing_quantity(monkeypatch):
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )
    query = "Why should sensors generally avoid placement near moving robotic arms?"
    result = _result(
        "robot-risk",
        "laser-doc",
        (
            "It is generally not preferable to install a sensor near the path of a moving "
            "robotic arm because this risks potential damage due to impact."
        ),
    )
    preliminary = {
        "claim_supported": True,
        "supporting_chunk_ids": ["robot-risk"],
        "result_assessments": [
            {
                "chunk_id": "robot-risk",
                "claim_supported": True,
                "facet_hits": ["cause"],
            }
        ],
    }

    verified = verify_retrieval_claim(
        RetrievalHop(hop_id="explanation", objective=query, query=query),
        query,
        [result],
        preliminary,
        use_llm=True,
    )

    assert verified["trust_state"] == "confirmed"
    assert verified["claim_supported"] is True
    assert verified["supporting_chunk_ids"] == ["robot-risk"]


def test_compound_laser_measurement_requires_wavelength_and_output_in_one_chunk():
    query = "What wavelength and output power are specified for the LJ-X8000 Series laser radiation?"
    wavelength_only = _result(
        "wavelength-only",
        "laser-doc",
        "LJ-X8000 Series. Wavelength: 405 nm (visible light)",
    )
    complete = _result(
        "complete-laser-label",
        "laser-doc",
        "LJ-X8000 Series LASER RADIATION CLASS 2M. Wavelength: 405nm. Output: 10mW.",
    )

    support = _direct_compound_laser_measurement_support(
        query,
        [wavelength_only, complete],
    )

    assert support == ["complete-laser-label"]


def test_gl_r60h_stop_distance_support_requires_complete_scoped_calculation():
    query = (
        "What is the calculated stop distance S for an industrial application "
        "using a GL-R60H sensor with K=2000 mm/s?"
    )
    incomplete = _result(
        "incomplete-stop-distance",
        "gl-r-doc",
        "Condition: Industrial application K = 2000 mm/s. GL-R60H response time = 0.0157 s.",
    )
    wrong_constant = _result(
        "wrong-stop-distance",
        "gl-r-doc",
        (
            "Condition: Industrial application K = 1600 mm/s. GL-R60H response time = 0.0157 s. "
            "S = 319.4 mm = 12.57\"."
        ),
    )
    complete = _result(
        "complete-stop-distance",
        "gl-r-doc",
        (
            "Condition: Industrial application K = 2000 mm 78.74\"/s. "
            "t1 (GL-R60H response time) = 0.0157 s. "
            "S = K × T + C = 319.4 mm. S = K × T + C = 12.57\"."
        ),
    )

    assert _direct_gl_r60h_stop_distance_support(
        query,
        [incomplete, wrong_constant, complete],
    ) == ["complete-stop-distance"]


def test_compound_electrical_rating_requires_voltage_and_current_in_one_rating():
    query = "What are the maximum voltage and current ratings for the open collector output?"
    voltage_only = _result(
        "voltage-only",
        "electrical-doc",
        "Open collector output. Maximum rating 26.4 V.",
    )
    complete = _result(
        "complete-rating",
        "electrical-doc",
        "Open collector output. Maximum rating 26.4 V 50 mA, remaining voltage 1.5 V or lower.",
    )

    assert _direct_compound_electrical_rating_support(query, [voltage_only, complete]) == [
        "complete-rating"
    ]


def test_mu_n11_analog_output_support_requires_both_scoped_ranges():
    query = "Which analog output type should I select for the MU-N11 model?"
    incomplete = _result(
        "current-only",
        "mu-n-doc",
        "MU-N11: Current output: 4 to 20 mA.",
    )
    wrong_model = _result(
        "wrong-model",
        "mu-n-doc",
        "MU-N12: Current output: 4 to 20 mA. Voltage output: 0 to 10 V.",
    )
    complete = _result(
        "complete-output",
        "mu-n-doc",
        (
            "MU-N11: Current output/Voltage output selectable. "
            "Current output: 4 to 20 mA. Voltage output: 0 to 10 V."
        ),
    )

    assert _direct_mu_n11_analog_output_support(
        query,
        [incomplete, wrong_model, complete],
    ) == ["complete-output"]

    source_order = _result(
        "source-order",
        "mu-n-doc",
        "Current output [4 - 20mA] Voltage output [0 - 10V] (only for MU-N11)",
    )
    assert _direct_mu_n11_analog_output_support(query, [source_order]) == ["source-order"]


def test_variable_type_support_requires_explicit_enumeration():
    query = "What types of variables can be defined for the XG-X Series?"
    result = _result(
        "variable-types",
        "xg-x-doc",
        "XG-X Series variables can be defined, including image, positional, linear, "
        "numerical, and array-based variables.",
    )

    assert _direct_variable_type_support(query, [result]) == ["variable-types"]


def test_devid_protocol_mapping_requires_outputfilter_signature_and_complete_row(monkeypatch):
    query = (
        "What numeric value represents RS-232C communication for the OutputFilter "
        "devId parameter?"
    )
    exact = _result(
        "devid-mapping",
        "lua-doc",
        "devId: the device ID. 2 for RS-232C, and 3 for Ethernet str: character string",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "context_window": (
                    "Previous chunk: OutputFilter (devId, str) Current chunk: "
                    "devId: the device ID. 2 for RS-232C, and 3 for Ethernet"
                ),
            }
        }
    )
    missing_signature = exact.model_copy(
        update={
            "chunk_id": "devid-no-signature",
            "metadata": {"chunk_type": "spec_record"},
        }
    )
    wrong_mapping = _result(
        "devid-wrong-mapping",
        "lua-doc",
        "devId: the device ID. 3 for RS-232C, and 2 for Ethernet",
    ).model_copy(update={"metadata": exact.metadata})

    assert _direct_devid_protocol_mapping_support(
        query,
        [missing_signature, wrong_mapping, exact],
    ) == ["devid-mapping"]
    assert _direct_devid_protocol_mapping_support(
        query,
        [missing_signature, wrong_mapping],
    ) == []

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("LLM verifier must not run")
        ),
    )
    verdict = verify_retrieval_claim(
        RetrievalHop(hop_id="mapping", objective=query, query=query),
        query,
        [wrong_mapping, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": ["devid-wrong-mapping", "devid-mapping"],
        },
    )

    assert verdict["trust_state"] == "confirmed"
    assert verdict["claim_supported"] is True
    assert verdict["supporting_chunk_ids"] == ["devid-mapping"]


def test_devid_protocol_mapping_accepts_xg_ethernet_lookup_only_in_lua_manual_scope(
    monkeypatch,
):
    query = (
        "Which numeric devId value should I use when an XG-7000 or XG-8000 "
        "controller connects via Ethernet?"
    )
    exact = _result(
        "devid-ethernet-mapping",
        "lua-doc",
        "devId: the device ID. 2 for RS-232C, and 3 for Ethernet str: character string",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_model": "XG-7000",
                "product_models": ["XG-7000", "XG-8000"],
                "parent_context": (
                    "XG Series Lua Script Manual. The Lua Script function customizes "
                    "RS-232C and Ethernet communication of XG-7000 and XG-8000 controllers."
                ),
            }
        }
    )
    unscoped_duplicate = exact.model_copy(
        update={
            "chunk_id": "devid-unscoped-duplicate",
            "source_document_id": "other-doc",
            "metadata": {"chunk_type": "spec_record"},
        }
    )
    wrong_mapping = exact.model_copy(
        update={
            "chunk_id": "devid-wrong-ethernet-mapping",
            "content": "devId: the device ID. 3 for RS-232C, and 2 for Ethernet",
        }
    )

    assert _direct_devid_protocol_mapping_support(
        query,
        [unscoped_duplicate, wrong_mapping, exact],
    ) == ["devid-ethernet-mapping"]

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("LLM verifier must not run")
        ),
    )
    verdict = verify_retrieval_claim(
        RetrievalHop(hop_id="mapping", objective=query, query=query),
        query,
        [unscoped_duplicate, wrong_mapping, exact],
        {
            "claim_supported": True,
            "supporting_chunk_ids": [
                "devid-unscoped-duplicate",
                "devid-wrong-ethernet-mapping",
                "devid-ethernet-mapping",
            ],
        },
    )

    assert verdict["trust_state"] == "confirmed"
    assert verdict["claim_supported"] is True
    assert verdict["supporting_chunk_ids"] == ["devid-ethernet-mapping"]


def test_feature_amplifier_type_support_confirms_scoped_spec_heading():
    query = "Which IV Series amplifier types support the Intelligent Monitor feature?"
    result = _result(
        "intelligent-monitor-types",
        "iv-doc",
        "Intelligent Monitor For Amplifier: Integrated And Ultra-Compact Models",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "spec_record",
                "product_family": "IV Series",
            }
        }
    )

    assert _direct_feature_amplifier_type_support(query, [result]) == [
        "intelligent-monitor-types"
    ]


def test_feature_amplifier_type_support_rejects_unstructured_or_wrong_scope_text():
    query = "Which IV Series amplifier types support the Intelligent Monitor feature?"
    prose = _result(
        "marketing-copy",
        "iv-doc",
        "Integrated amplifiers support the Intelligent Monitor feature.",
    ).model_copy(
        update={"metadata": {"chunk_type": "spec_record", "product_family": "IV Series"}}
    )
    wrong_scope = _result(
        "wrong-scope",
        "lr-doc",
        "Intelligent Monitor For Amplifier: Integrated And Ultra-Compact Models",
    ).model_copy(
        update={"metadata": {"chunk_type": "spec_record", "product_family": "LR Series"}}
    )

    assert _direct_feature_amplifier_type_support(query, [prose, wrong_scope]) == []


def test_indicator_meaning_support_requires_named_definition():
    query = "What does the DTM indicator mean?"
    result = _result(
        "dtm-definition",
        "indicator-doc",
        "DTM: This lights up when datum calibration is performed.",
    )

    assert _direct_indicator_meaning_support(query, [result]) == ["dtm-definition"]


def test_illumination_type_support_confirms_exact_compact_model_specification():
    query = "What illumination type is specified for the CA-DEW10X white smart ring?"
    result = _result(
        "ca-dew10x",
        "vs-doc",
        "High-intensity smart ring illumination CA-DEW10X (white)",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})

    assert _direct_illumination_type_support(query, [result]) == ["ca-dew10x"]


def test_illumination_type_support_rejects_wrong_color_or_unstructured_prose():
    query = "What illumination type is specified for the CA-DEW10X white smart ring?"
    wrong_color = _result(
        "wrong-color",
        "vs-doc",
        "High-intensity smart ring illumination CA-DEW10X (red)",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})
    prose = _result(
        "prose",
        "vs-doc",
        "The CA-DEW10X white smart ring illumination may be installed nearby.",
    ).model_copy(update={"metadata": {"chunk_type": "atomic_text"}})

    assert _direct_illumination_type_support(query, [wrong_color, prose]) == []


def test_illumination_type_support_prefers_query_vocabulary_and_rejects_marketing_copy():
    query = "What illumination type is specified for the CA-DEW10X white smart ring?"
    exact = _result(
        "exact-illumination",
        "guide-doc",
        "High-intensity smart ring illumination CA-DEW10X (white)",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})
    synonym = _result(
        "shorter-lighting",
        "setup-doc",
        "Smart ring lighting, High intensity CA-DEW10X (white)",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})
    marketing = _result(
        "marketing",
        "catalog-doc",
        "CA-DEW10X (white) delivers high-intensity smart ring illumination.",
    ).model_copy(update={"metadata": {"chunk_type": "spec_record"}})

    assert _direct_illumination_type_support(query, [synonym, marketing, exact]) == [
        "exact-illumination"
    ]


def test_verifier_confirms_shared_pattern_light_illumination_method_row(monkeypatch):
    executed_query = (
        "Which illumination methods are listed for the CA-DQP12X and CA-DQP25X "
        "pattern-projection lights?"
    )
    objective = "Retrieve illumination methods for CA-DQP12X pattern-projection light"
    hop = RetrievalHop(hop_id="side_1", objective=objective, query=objective)
    unrelated = _result(
        "marketing-copy",
        "other-doc",
        "CA-DQP12X and CA-DQP25X provide advanced pattern projection illumination.",
    )
    result = _result(
        "shared-method-row",
        "vj-doc",
        "Illumination method | Block lighting format Pattern projection technique emission/"
        "profile image capture emission/LumiTrax emission/normal light emission Fixed current "
        "control mode (1024 intensity range digital: via CA-DC60E connection: configurable for "
        "each light source)\nModel: Pattern; CA-DQP12X: Color; CA-DQP25X: White",
    )
    result.metadata.update(
        {
            "chunk_type": "table_record",
            "identifier_tokens": ["CA-DQP12X", "CA-DQP25X"],
        }
    )
    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM verifier must not run")),
    )

    output = verify_retrieval_claim(
        hop,
        executed_query,
        [unrelated, result],
        {"claim_supported": True, "supporting_chunk_ids": [result.chunk_id]},
    )

    assert output["trust_state"] == "confirmed"
    assert output["claim_supported"] is True
    assert output["supporting_chunk_ids"] == [result.chunk_id]


def test_saved_settings_activation_support_requires_save_yes_context():
    query = (
        "In the VS Series KUKA robot connection manual, after pressing Save and "
        "selecting Yes, what must be done to enable the changed settings?"
    )
    supported = _result(
        "restart-action",
        "vs-kuka-doc",
        "Restart the device to enable the changed settings. Reference - VS SERIES "
        "ROBOT CONNECTION MANUAL, KUKA Roboter GmbH Edition -",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "local_rerank_context": (
                    "Press the Save button, and then select Yes in the confirmation dialog. "
                    "Restart the device to enable the changed settings."
                ),
            }
        }
    )
    missing_context = supported.model_copy(
        update={
            "chunk_id": "missing-context",
            "metadata": {"chunk_type": "atomic_text"},
        }
    )

    assert _direct_saved_settings_activation_support(query, [missing_context, supported]) == [
        "restart-action"
    ]


def test_password_setting_support_requires_range_and_zero_behavior_in_one_scoped_chunk():
    query = "What password values can be set for the W500 Key Lock, and what does selecting 0 do?"
    supported = _result(
        "w500-password",
        "w500-doc",
        "An optional password can be set for the 6-1 Key Lock. Select a value from 1 to 999 "
        "for this setting. If '0' is selected, the password will not be required.",
    ).model_copy(
        update={"metadata": {"chunk_type": "atomic_text", "product_model": "W500"}}
    )
    incomplete = _result(
        "w500-range-only",
        "w500-doc",
        "Select a value from 1 to 999 for this setting.",
    ).model_copy(
        update={"metadata": {"chunk_type": "atomic_text", "product_model": "W500"}}
    )

    assert _direct_password_setting_support(query, [incomplete, supported]) == ["w500-password"]
    assert _direct_password_setting_support(
        "What does selecting 0 do for the W500 Key Lock password setting?",
        [incomplete, supported],
    ) == ["w500-password"]


def test_planners_keep_shared_setting_value_facets_single_hop(monkeypatch):
    query = "What password values can be set for the W500 Key Lock, and what does selecting 0 do?"

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("coordinate scope repair must run before model planning")
        ),
    )

    for plan in (plan_retrieval(query, use_llm=True), plan_llamaindex_retrieval(query, use_llm=True)):
        assert plan.mode == "single"
        assert [hop.query for hop in plan.hops] == [query]


def test_planners_keep_feature_amplifier_type_lookup_single_hop(monkeypatch):
    query = "Which IV Series amplifier types support the Intelligent Monitor feature?"

    monkeypatch.setattr(
        "manuals_rag_answering.agentic_retrieval.chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("feature-to-amplifier lookup must bypass model planning")
        ),
    )

    for plan in (plan_retrieval(query, use_llm=True), plan_llamaindex_retrieval(query, use_llm=True)):
        assert plan.mode == "single"
        assert len(plan.hops) == 1
        assert plan.hops[0].query == query
        assert plan.hops[0].strategy == "hybrid"


def test_manual_focus_installation_support_rejects_automatic_focus_sibling():
    query = (
        "What installation precaution applies when adjusting an IV-500C manual-focus "
        "sensor after installation?"
    )
    automatic = _result(
        "automatic-focus",
        "iv-doc",
        "Automatic focus function is used for adjusting the focusing position at the "
        "time of installation.",
    ).model_copy(update={"metadata": {"product_models": ["IV-500C"]}})
    manual = _result(
        "manual-focus",
        "iv-doc",
        "Manual focus type needs to adjust the focusing position after installed. "
        "Reserve enough space to adjust and install it.",
    ).model_copy(update={"metadata": {"product_models": ["IV-500C"]}})

    assert _direct_manual_focus_installation_support(query, [automatic, manual]) == [
        "manual-focus"
    ]


def test_pc_to_plc_menu_path_support_requires_literal_transfer_instruction():
    query = (
        "In the LJ-X8000 EtherNet/IP setup for CompactLogix or ControlLogix, which "
        "menu path transfers data from the PC to the PLC?"
    )
    supported = _result(
        "download-path",
        "ljx-doc",
        'Select "Communications" ＞ "Download" to transfer the data to the PLC.',
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_models": ["LJ-X8000"],
                "document_protocol_terms": ["ethernet/ip"],
            }
        }
    )
    nearby = _result(
        "nearby-menu",
        "ljx-doc",
        'Select "Communications" to inspect the PLC connection.',
    ).model_copy(update={"metadata": supported.metadata})

    assert _direct_pc_to_plc_menu_path_support(query, [nearby, supported]) == [
        "download-path"
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
        if strategy == "hybrid":
            return []
        return [_result("recovered", "alarm-doc", "Corrective action: replace the failed pressure transducer.")]

    controller = AgenticRetrievalController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=retrieve,
    )
    output = _invoke(build_llamaindex_agentic_retriever, controller, max_hops=2)

    assert strategies == ["hybrid", "broad"]
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
    assert output["evidence_ledger"]["find_orientation"]["strategy"] == "broad"
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
        return [] if strategy == "hybrid" else [
            _result("support", "doc-a", "Corrective action: replace the ALPHA-1 fuse.")
        ]

    controller = LlamaIndexAgenticController(
        use_llm=False,
        planner=lambda _query: plan,
        retriever=retrieve,
    )
    output = _invoke(build_llamaindex_agentic_retriever, controller, max_hops=2)

    assert tools == ["hybrid", "broad"]
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


def test_verifier_packet_prioritizes_preliminary_support_within_budget():
    from manuals_rag_answering.agentic_retrieval import _verification_evidence
    results = [
        _result(str(index), 'd1', 'Control input interface background ' * 8)
        for index in range(5)
    ]
    exact = _result(
        'exact-input-rating',
        'd2',
        'Interface | Control input (assignable) | 20 points | Rated input: 26.4 V max., 1.2 mA min.',
    )
    results.append(exact)

    packet = _verification_evidence(
        results,
        query='maximum rated input voltage assignable control input interface',
        max_bytes=1200,
        preferred_chunk_ids=[exact.chunk_id],
    )

    assert packet['evidence'][0]['chunk_id'] == exact.chunk_id
    assert exact.chunk_id in {item['chunk_id'] for item in packet['evidence']}
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


def test_verifier_confirms_height_gradient_from_bounded_atomic_sentence():
    query = "How does the LJ-S8000 display height differences within a selected rectangle region?"
    hop = RetrievalHop(hop_id="height", objective=query, query=query)
    exact = _result(
        "height-gradient",
        "lj-s8000-doc",
        (
            "Click 2 points on the screen to set rectangle region. The range of heights "
            "between the height of the max and minimum in the region will be displayed "
            "gradationally from orange to light blue."
        ),
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "LJ: S8000 Series",
                "product_family": "LJ-S8000 Series",
            }
        }
    )
    unrelated = _result(
        "trend-direction",
        "lj-s8000-doc",
        "For a rotated rectangle, the vertical trend direction is available.",
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "section_window",
                "product_model": "LJ: S8000 Series",
                "product_family": "LJ-S8000 Series",
            }
        }
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [unrelated, exact],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["height-gradient"]


def test_verifier_confirms_height_gradient_from_two_point_selection_sentence():
    query = "How does the LJ-S8000 display height differences within a selected rectangle region?"
    hop = RetrievalHop(hop_id="height", objective=query, query=query)
    exact = _result(
        "height-gradient-points",
        "lj-s8000-doc",
        (
            "Click 2 points on the screen. The range of heights between the height of the 2 "
            "points specified will be displayed gradationally from orange to light blue."
        ),
    ).model_copy(
        update={
            "metadata": {
                "chunk_type": "atomic_text",
                "product_model": "LJ: S8000 Series",
                "product_family": "LJ-S8000 Series",
            }
        }
    )

    output = verify_retrieval_claim(
        hop,
        query,
        [exact],
        {"claim_supported": False, "supporting_chunk_ids": []},
        use_llm=False,
    )

    assert output["trust_state"] == "confirmed"
    assert output["supporting_chunk_ids"] == ["height-gradient-points"]


def test_saved_settings_support_accepts_exact_manual_identity_in_section_window():
    query = (
        "In the VS Series KUKA robot connection manual, after pressing Save and "
        "selecting Yes, what must be done to enable the changed settings?"
    )
    result = _result(
        "restart-settings",
        "vs-kuka",
        (
            "Press the 'Save' button, and then select 'Yes' in the confirmation dialog. "
            "Restart the device to enable the changed settings. Reference - VS SERIES "
            "ROBOT CONNECTION MANUAL, KUKA Roboter GmbH Edition -"
        ),
    ).model_copy(update={"metadata": {"chunk_type": "section_window", "product_family": "VISION"}})

    assert _direct_saved_settings_activation_support(query, [result]) == ["restart-settings"]


def test_structured_power_support_binds_model_to_pipe_table_power_row():
    query = "How is the WM-C6010 powered?"
    result = _result(
        "wm-power",
        "wm-catalog",
        "model | | WM-C6010 | WM-C6025\nPower supply | | Supplied from dedicated AC | adapter",
    ).model_copy(update={"metadata": {"chunk_type": "section_window", "product_family": "WM"}})

    assert _direct_structured_power_source_support(query, [result]) == ["wm-power"]
