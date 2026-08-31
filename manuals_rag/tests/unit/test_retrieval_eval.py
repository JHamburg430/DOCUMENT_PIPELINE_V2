import pytest

from manuals_rag_evals.retrieval_eval import (
    RetrievalEvalCase,
    USER_STYLE_QUERY_FEW_SHOT_EXAMPLES,
    build_eval_cases_from_chunks,
    build_multi_step_eval_cases_from_chunks,
    chunk_is_queryworthy,
    score_answer_response,
    score_document_selection,
    score_search_results,
    validate_eval_case,
    _parse_generated_queries,
)


def test_large_retrieval_eval_loads_saved_dataset(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_path = tmp_path / "cases.jsonl"
    case = RetrievalEvalCase(
        case_id="case-1",
        query="What voltage does MODEL-1 require?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="spec_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["24", "vdc"],
        expected_snippet="Power supply voltage: 24 VDC",
        generation_method="unit_test",
        source_metadata={"product_model": "MODEL-1"},
    )
    dataset_path.write_text(
        "\n".join(
            [
                "",
                __import__("json").dumps(case.to_dict()),
                __import__("json").dumps({"case": {**case.to_dict(), "case_id": "case-2"}}),
            ]
        ),
        encoding="utf-8",
    )

    cases = module.load_eval_cases_from_dataset(dataset_path, max_cases=1)

    assert len(cases) == 1
    assert cases[0]["case_id"] == "case-1"
    assert cases[0]["retrieval_task"] == "single_step_retrieval"


def test_large_retrieval_eval_offsets_saved_dataset_after_validation(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _case(case_id: str, query: str, source_chunk_id: str, *, stale: bool = False) -> RetrievalEvalCase:
        return RetrievalEvalCase(
            case_id=case_id,
            query=query,
            source_document_id="doc-1",
            document_version_id="ver-1",
            source_chunk_id=source_chunk_id,
            source_title="Manual",
            source_filename="manual.pdf",
            chunk_type="table_record",
            section_path="Specifications",
            page_from=1,
            page_to=1,
            expected_terms=["24", "vdc", "power"],
            expected_snippet=(
                "Table header: Power supply voltage; Header role: row"
                if stale
                else "Column headers: Power supply voltage; Row headers: MODEL-1; Cell value: 24 VDC"
            ),
            generation_method="unit_test",
            source_metadata={
                "product_model": "MODEL-1",
                "table_header": stale,
                "table_cell": not stale,
                "table_row_headers": [] if stale else ["MODEL-1"],
                "table_column_headers": ["Power supply voltage"],
            },
            anchor_terms=["24", "vdc", "power"],
        )

    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps(case.to_dict())
            for case in [
                _case("stale-header", "What voltage applies to MODEL-1?", "header-row", stale=True),
                _case("case-1", "What power voltage does MODEL-1 require?", "chunk-1"),
                _case("case-2", "Which supply voltage is listed for MODEL-1?", "chunk-2"),
                _case("case-3", "What voltage should MODEL-1 use?", "chunk-3"),
            ]
        ),
        encoding="utf-8",
    )

    kept, rejected = module.load_eval_cases_and_rejections_from_dataset(
        dataset_path,
        max_cases=2,
        case_offset=1,
        drop_invalid_cases=True,
    )

    assert [case["case_id"] for case in kept] == ["case-2", "case-3"]
    assert [case["case_id"] for case in rejected] == ["stale-header"]


def test_large_retrieval_eval_can_drop_invalid_saved_single_step_cases(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    valid_case = RetrievalEvalCase(
        case_id="valid-cell",
        query="What 24 VDC power supply voltage applies to MODEL-1?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="table_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["24", "vdc", "power"],
        expected_snippet="Column headers: Power supply voltage; Row headers: MODEL-1; Cell value: 24 VDC",
        generation_method="table_row_column_value",
        source_metadata={
            "product_model": "MODEL-1",
            "table_cell": True,
            "table_row_headers": ["MODEL-1"],
            "table_column_headers": ["Power supply voltage"],
        },
        anchor_terms=["24", "vdc", "power"],
    )
    header_case = RetrievalEvalCase(
        case_id="stale-header",
        query="What detection applies to IV4-G120?",
        source_document_id="doc-iv4",
        document_version_id="ver-iv4",
        source_chunk_id="header-row",
        source_title="IV4 Manual",
        source_filename="iv4.pdf",
        chunk_type="table_record",
        section_path="R1.20",
        page_from=14,
        page_to=14,
        expected_terms=["detection", "becomes", "unstable"],
        expected_snippet="Table header: If the detection becomes unstable due to the effect of; Header role: row; Row: 32; Column: 0",
        generation_method="table_primary",
        source_metadata={
            "product_model": "IV4-G120",
            "table_header": True,
            "table_header_role": "row",
        },
        anchor_terms=["detection", "becomes", "unstable"],
    )
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(case.to_dict()) for case in [header_case, valid_case]),
        encoding="utf-8",
    )

    kept, rejected = module.load_eval_cases_and_rejections_from_dataset(
        dataset_path,
        max_cases=10,
        drop_invalid_cases=True,
    )

    assert [case["case_id"] for case in kept] == ["valid-cell"]
    assert rejected == [
        {
            "case_id": "stale-header",
            "query": "What detection applies to IV4-G120?",
            "source_chunk_id": "header-row",
            "reason": "not_queryworthy_source_chunk",
        }
    ]


def test_large_retrieval_eval_scores_query_timeouts():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    evaluation = module.timeout_evaluation(
        {"expected_terms": ["24", "vdc"]},
        elapsed_seconds=12.3456,
        timeout_seconds=12,
    )

    assert evaluation["passed"] is False
    assert evaluation["rank"] is None
    assert evaluation["candidate_recall"] is False
    assert evaluation["failure_category"] == "eval_timeout"
    assert evaluation["missing_terms"] == ["24", "vdc"]
    assert evaluation["elapsed_seconds"] == 12.346


def test_large_retrieval_eval_recognizes_wrapped_query_timeouts():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.is_query_timeout_exception(module.QueryTimeoutError("Search exceeded per-query timeout of 12 seconds."))
    assert module.is_query_timeout_exception(RuntimeError("Search exceeded per-query timeout of 12 seconds."))
    assert not module.is_query_timeout_exception(RuntimeError("qdrant collection unavailable"))


def test_large_retrieval_eval_persists_cited_evidence_beyond_top_five():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    results = [{"chunk_id": f"chunk-{index}", "content": f"content {index}"} for index in range(1, 8)]
    persisted = module.results_for_persisted_evidence(
        results,
        {"citations": [{"chunk_id": "chunk-7", "quote_span": None}]},
    )

    assert [result["chunk_id"] for result in persisted] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
        "chunk-5",
        "chunk-7",
    ]


def test_large_retrieval_eval_enforces_elapsed_timeout_after_swallowed_signal(monkeypatch):
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module.time, "time", lambda: 101.0)

    try:
        module.enforce_completed_query_timeout(start_time=50.0, timeout_seconds=45)
    except module.QueryTimeoutError as exc:
        assert "45 seconds" in str(exc)
    else:
        raise AssertionError("Expected elapsed timeout to be raised.")


def test_large_retrieval_eval_runs_unscored_warmups(monkeypatch):
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    searched_queries = []

    def fake_run_case_search(query, *, corpus_id, search_mode):
        searched_queries.append((query, corpus_id, search_mode))
        return [{"chunk_id": "hit"}]

    monkeypatch.setattr(module, "run_case_search", fake_run_case_search)

    warmups = module.run_warmup_searches(
        [
            {"case_id": "case-1", "query": "first query"},
            {"case_id": "case-2", "query": "second query"},
        ],
        corpus_id="manuals",
        search_mode="direct",
        warmup_queries=1,
        warmup_timeout_seconds=30,
    )

    assert searched_queries == [("first query", "manuals", "direct")]
    assert warmups == [
        {
            "case_id": "case-1",
            "status": "completed",
            "elapsed_seconds": warmups[0]["elapsed_seconds"],
            "result_count": 1,
            "answer_generated": False,
        }
    ]


def test_large_retrieval_eval_records_warmup_timeouts(monkeypatch):
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_run_case_search(query, *, corpus_id, search_mode):
        raise module.QueryTimeoutError("Search exceeded per-query timeout of 30 seconds.")

    monkeypatch.setattr(module, "run_case_search", fake_run_case_search)

    warmups = module.run_warmup_searches(
        [{"case_id": "case-1", "query": "slow startup query"}],
        corpus_id="manuals",
        search_mode="direct",
        warmup_queries=1,
        warmup_timeout_seconds=30,
    )

    assert warmups[0]["case_id"] == "case-1"
    assert warmups[0]["status"] == "eval_timeout"
    assert warmups[0]["timeout_seconds"] == 30
    assert warmups[0]["result_count"] == 0


def test_answer_response_scoring_requires_terms_and_expected_document():
    case = RetrievalEvalCase(
        case_id="case-1",
        query="What voltage does MODEL-1 use?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="spec_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["24", "vdc"],
        expected_snippet="Power supply voltage: 24 VDC",
        generation_method="unit_test",
        source_metadata={"product_model": "MODEL-1"},
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Use a 24 VDC power supply.",
            "citations": [{"document_id": "doc-1", "chunk_id": "chunk-1", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert scored["passed"] is True
    assert scored["missing_document_ids"] == []
    assert scored["term_check"]["matched_terms"] == ["24", "vdc"]


def test_answer_response_scoring_rejects_cases_without_scorable_answer_terms():
    case = RetrievalEvalCase(
        case_id="case-command-error-flag",
        query="Which state should the command error tag display when assigned?",
        source_document_id="doc-ljx",
        document_version_id="ver-ljx",
        source_chunk_id="chunk-command-error",
        source_title="LJ-X8000",
        source_filename="ljx.pdf",
        chunk_type="procedure_record",
        section_path="Command execution",
        page_from=43,
        page_to=43,
        expected_terms=["check", "whether", "ljx3d", "i.data"],
        expected_snippet=(
            "Check whether the tag (LJX3D: I.Data[0].1) to which the Command error "
            "flag has been assigned is ON or OFF."
        ),
        generation_method="unit_test",
        source_metadata={"product_model": "LJ-X8000"},
    )

    scored = score_answer_response(
        case,
        {
            "answer": "The command error tag should display ON when the command fails or processing fails.",
            "citations": [{"document_id": "doc-ljx", "chunk_id": "chunk-command-error", "pages": [43]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert scored["passed"] is False
    assert scored["term_check"]["term_source"] == "no_scorable_case_expected_terms"
    assert scored["term_check"]["expected_terms"] == []
    assert scored["failure_reasons"] == ["expected_terms_missing"]


def test_answer_response_scoring_can_use_llm_required_information_judge(monkeypatch):
    case = RetrievalEvalCase(
        case_id="case-command-error-flag",
        query="Which state should the command error tag display when assigned?",
        source_document_id="doc-ljx",
        document_version_id="ver-ljx",
        source_chunk_id="chunk-command-error",
        source_title="LJ-X8000",
        source_filename="ljx.pdf",
        chunk_type="procedure_record",
        section_path="Command execution",
        page_from=43,
        page_to=43,
        expected_terms=["check", "whether", "ljx3d", "i.data"],
        expected_snippet=(
            "Check whether the tag (LJX3D: I.Data[0].1) to which the Command error "
            "flag has been assigned is ON or OFF."
        ),
        generation_method="unit_test",
        source_metadata={"product_model": "LJ-X8000"},
    )

    def fake_chat_json(**kwargs):
        return (
            {
                "contains_required_information": True,
                "missing_information": [],
                "reason": "The answer states the required ON/OFF state.",
            },
            "{}",
        )

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.chat_json", fake_chat_json)

    scored = score_answer_response(
        case,
        {
            "answer": "The command error tag should display ON when the command fails or processing fails.",
            "citations": [{"document_id": "doc-ljx", "chunk_id": "chunk-command-error", "pages": [43]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
        use_llm_required_info_judge=True,
    )

    assert scored["passed"] is True
    assert scored["term_check"]["llm_judged"] is True
    assert scored["llm_required_information"]["passed"] is True


def test_answer_response_scoring_requires_all_multi_step_documents():
    case = RetrievalEvalCase(
        case_id="case-1",
        query="Compare the address values for MODEL-1 and MODEL-2.",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="table_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["address", "100"],
        expected_snippet="Address: 100",
        generation_method="cross_document_same_field_evidence",
        source_metadata={"product_model": "MODEL-1"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {"chunk_id": "chunk-1", "source_document_id": "doc-1", "expected_terms": ["address", "100"]},
            {"chunk_id": "chunk-2", "source_document_id": "doc-2", "expected_terms": ["address", "200"]},
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "MODEL-1 lists address 100; MODEL-2 lists address 200.",
            "citations": [{"document_id": "doc-1", "chunk_id": "chunk-1", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert scored["passed"] is False
    assert scored["missing_document_ids"] == ["doc-2"]
    assert "expected_document_not_cited_or_used" in scored["failure_reasons"]


def test_answer_response_scoring_rejects_sibling_table_cell_bindings():
    case = RetrievalEvalCase(
        case_id="case-table-binding",
        query="Can CV-X482 allocate command 0016 PID 428 in the status bit area?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="chunk-table",
        source_title="CV-X Manual",
        source_filename="cvx.pdf",
        chunk_type="table_record",
        section_path="6-180",
        page_from=920,
        page_to=920,
        expected_terms=["allocation", "possible", "0016", "428"],
        expected_snippet="Row headers: 0016 PID 428 > status Bit area; Cell value: Allocation possible; Row: 10; Column: 14",
        generation_method="unit_test",
        source_metadata={"product_model": "CV-X482", "table_cell": True},
    )
    answer = {
        "answer": (
            "Relevant retrieved table rows:\n"
            "- Mea- surement count area | 0016 0017 0018 | PID 428 | Total count\n"
            "- 0004 | PIB256 bit No. 7 | Allocation possible\n"
            "- 0005 | PIB257 bit No. 7 | Allocation possible"
        ),
        "citations": [{"document_id": "doc-cvx", "chunk_id": "chunk-table", "pages": [920]}],
        "used_documents": [],
        "insufficient_evidence": False,
    }

    scored = score_answer_response(case, answer, {"passed": True})

    assert scored["passed"] is False
    assert "expected_table_cell_binding_missing" in scored["failure_reasons"]
    assert scored["table_cell_binding"]["missing_bindings"][0]["reason"] == "row_header_and_cell_value_not_bound_in_answer_segment"


def test_answer_response_scoring_accepts_same_row_table_cell_binding():
    case = RetrievalEvalCase(
        case_id="case-table-binding",
        query="Can CV-X482 allocate command 0016 PID 428 in the status bit area?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="chunk-table",
        source_title="CV-X Manual",
        source_filename="cvx.pdf",
        chunk_type="table_record",
        section_path="6-180",
        page_from=920,
        page_to=920,
        expected_terms=["allocation", "possible", "0016", "428"],
        expected_snippet="Row headers: 0016 PID 428 > status Bit area; Cell value: Allocation possible; Row: 10; Column: 14",
        generation_method="unit_test",
        source_metadata={"product_model": "CV-X482", "table_cell": True},
    )
    answer = {
        "answer": "status Bit area | 0016 | PID 428 command | Allocation possible",
        "citations": [{"document_id": "doc-cvx", "chunk_id": "chunk-table", "pages": [920]}],
        "used_documents": [],
        "insufficient_evidence": False,
    }

    scored = score_answer_response(case, answer, {"passed": True})

    assert scored["passed"] is True
    assert scored["table_cell_binding"]["passed"] is True


def test_answer_response_scoring_rejects_sibling_quantity_table_row():
    case = RetrievalEvalCase(
        case_id="case-quantity-binding",
        query=(
            "On IV4-G120, when ON equals Set value, Count value is 9, Output status is "
            "One-Shot output, and the current count is 0, how many objects are counted at one time?"
        ),
        source_document_id="doc-iv4",
        document_version_id="ver-iv4",
        source_chunk_id="quantity-row",
        source_title="IV4 Manual",
        source_filename="iv4.pdf",
        chunk_type="table_record",
        section_path="4-78",
        page_from=168,
        page_to=168,
        expected_terms=["set", "count", "quantity", "counted", "3", "one-shot", "0"],
        expected_snippet=(
            "Status output settings | Previous count value (Display value) | Quantity counted at one time | "
            "Current count value (Display value) | Output status; "
            "ON when = Set value | Count value= 9 | 7 | 3 | 0 | One-Shot output"
        ),
        generation_method="unit_test",
        source_metadata={"product_model": "IV4-G120", "table_cell": True},
        expected_evidence=[
            {
                "chunk_id": "quantity-row",
                "source_document_id": "doc-iv4",
                "expected_terms": [
                    "on",
                    "set",
                    "value",
                    "count",
                    "9",
                    "quantity",
                    "counted",
                    "one",
                    "time",
                    "3",
                    "current",
                    "0",
                    "one-shot",
                    "output",
                ],
                "snippet": (
                    "Status output settings | Previous count value (Display value) | "
                    "Quantity counted at one time | Current count value (Display value) | Output status; "
                    "ON when = Set value | Count value= 9 | 7 | 3 | 0 | One-Shot output"
                ),
            }
        ],
    )
    answer = {
        "answer": "ON when = Set value | Count value= 9 | 7 | 2 | 9 | Latching output",
        "citations": [{"document_id": "doc-iv4", "chunk_id": "sibling-row", "pages": [168]}],
        "used_documents": [],
        "insufficient_evidence": False,
    }
    retrieved_results = [
        {
            "chunk_id": "sibling-row",
            "source_document_id": "doc-iv4",
            "content": "ON when = Set value | Count value= 9 | 7 | 2 | 9 | Latching output",
        }
    ]

    scored = score_answer_response(case, answer, {"passed": True}, retrieved_results)

    assert scored["passed"] is False
    assert "expected_terms_missing" in scored["failure_reasons"]
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_accepts_structured_table_cell_fallback_line():
    case = RetrievalEvalCase(
        case_id="case-command-result",
        query="On CV-X482, what does command 0028 / 65.0 map to in the 6-bit command output area?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="chunk-command-result",
        source_title="CV-X Manual",
        source_filename="cvx.pdf",
        chunk_type="table_record",
        section_path="6-210",
        page_from=952,
        page_to=952,
        expected_terms=["command", "result", "0028", "65.0"],
        expected_snippet=(
            "Column headers: 6bit > 5bit > 4bit > 3bit > 2bit > 1bit > 0bit; "
            "Row headers: 0028 65.0 > Command output area; Cell value: Command Result; "
            "Row: 15; Column: 3"
        ),
        generation_method="unit_test",
        source_metadata={"product_model": "CV-X482", "table_cell": True},
    )
    answer = {
        "answer": (
            "Column headers: 6bit > 5bit > 4bit > 3bit > 2bit > 1bit > 0bit; "
            "Row headers: 0028 65.0 > Command output area; Cell value: Command Result; "
            "Row: 15; Column: 3\n\nContext: broad neighboring status output text"
        ),
        "citations": [{"document_id": "doc-cvx", "chunk_id": "chunk-command-result", "pages": [952]}],
        "used_documents": [],
        "insufficient_evidence": False,
    }

    scored = score_answer_response(case, answer, {"passed": True})

    assert scored["passed"] is True
    assert scored["table_cell_binding"]["passed"] is True


def test_answer_response_scoring_rejects_scattered_signal_description_binding():
    case = RetrievalEvalCase(
        case_id="case-signal-binding",
        query="Which CV-X482 output line corresponds to data output bit 10?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="signal-row",
        source_title="CV-X Manual",
        source_filename="cvx.pdf",
        chunk_type="table_record",
        section_path="6-78",
        page_from=819,
        page_to=819,
        expected_terms=["out_data10", "data", "output", "bit"],
        expected_snippet="Column headers: Signal Description; Row headers: OUT_DATA10; Cell value: Data output bit 10; Row: 11; Column: 1",
        generation_method="unit_test",
        source_metadata={"product_model": "CV-X482", "table_cell": True},
    )
    answer = {
        "answer": (
            "Column headers: Function; Row headers: OUT_DATA0 > OUT_DATA1 > OUT_DATA2 > OUT_DATA3 > "
            "OUT_DATA4 > OUT_DATA5 > OUT_DATA6 > OUT_DATA7 > OUT_DATA8 > OUT_DATA9 > OUT_DATA10 > "
            "OUT_DATA11 > OUT_DATA12 > OUT_DATA13 > OUT_DATA14 > OUT_DATA15 > Data output bit 0 > "
            "Data output bit 1 > Data output bit 2 > Data output bit 3 > Data output bit 4 > "
            "Data output bit 5 > Data output bit 6 > Data output bit 7 > Data output bit 8 > "
            "Data output bit 9 > Data output bit 10."
        ),
        "citations": [{"document_id": "doc-cvx", "chunk_id": "aggregate-table", "pages": [803]}],
        "used_documents": [],
        "insufficient_evidence": False,
    }

    scored = score_answer_response(case, answer, {"passed": True})

    assert scored["passed"] is False
    assert "expected_table_cell_binding_missing" in scored["failure_reasons"]


def test_answer_response_scoring_accepts_direct_signal_description_binding():
    case = RetrievalEvalCase(
        case_id="case-signal-binding",
        query="Which CV-X482 output line corresponds to data output bit 10?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="signal-row",
        source_title="CV-X Manual",
        source_filename="cvx.pdf",
        chunk_type="table_record",
        section_path="6-78",
        page_from=819,
        page_to=819,
        expected_terms=["out_data10", "data", "output", "bit"],
        expected_snippet="Column headers: Signal Description; Row headers: OUT_DATA10; Cell value: Data output bit 10; Row: 11; Column: 1",
        generation_method="unit_test",
        source_metadata={"product_model": "CV-X482", "table_cell": True},
    )
    answer = {
        "answer": "OUT_DATA10 corresponds to Data output bit 10.",
        "citations": [{"document_id": "doc-cvx", "chunk_id": "signal-row", "pages": [819]}],
        "used_documents": [],
        "insufficient_evidence": False,
    }

    scored = score_answer_response(case, answer, {"passed": True})

    assert scored["passed"] is True
    assert scored["table_cell_binding"]["passed"] is True


def test_answer_response_scoring_rejects_negated_table_cell_binding():
    case = RetrievalEvalCase(
        case_id="case-table-binding",
        query="Can CV-X482 allocate command 0016 PID 428 in the status bit area?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="chunk-table",
        source_title="CV-X Manual",
        source_filename="cvx.pdf",
        chunk_type="table_record",
        section_path="6-180",
        page_from=920,
        page_to=920,
        expected_terms=["allocation", "possible", "0016", "428"],
        expected_snippet="Row headers: 0016 PID 428 > status Bit area; Cell value: Allocation possible; Row: 10; Column: 14",
        generation_method="unit_test",
        source_metadata={"product_model": "CV-X482", "table_cell": True},
    )
    answer = {
        "answer": "status Bit area | 0016 | PID 428 command | Allocation not possible",
        "citations": [{"document_id": "doc-cvx", "chunk_id": "chunk-table", "pages": [920]}],
        "used_documents": [],
        "insufficient_evidence": False,
    }

    scored = score_answer_response(case, answer, {"passed": True})

    assert scored["passed"] is False
    assert "expected_table_cell_binding_missing" in scored["failure_reasons"]


def test_cross_document_retrieval_scoring_requires_expected_evidence_documents():
    case = RetrievalEvalCase(
        case_id="case-1",
        query="Compare the address values for MODEL-1 and MODEL-2.",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="table_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["address", "100", "200"],
        expected_snippet="Address: 100 | Address: 200",
        generation_method="cross_document_same_field_evidence",
        source_metadata={"product_model": "MODEL-1"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "chunk-1",
                "source_document_id": "doc-1",
                "field": "address",
                "product_identifiers": ["model1"],
                "expected_terms": ["address", "100"],
            },
            {
                "chunk_id": "chunk-2",
                "source_document_id": "doc-2",
                "field": "address",
                "product_identifiers": ["model2"],
                "expected_terms": ["address", "200"],
            },
        ],
    )

    scored = score_search_results(
        case,
        [
            {
                "chunk_id": "chunk-1",
                "source_document_id": "doc-1",
                "content": "Column headers: Address; Model MODEL-1: 100",
                "metadata": {
                    "chunk_type": "table_record",
                    "table_column_headers": ["Address"],
                    "product_model": "MODEL-1",
                },
            },
            {
                "chunk_id": "equivalent-wrong-doc",
                "source_document_id": "doc-1",
                "content": "Column headers: Address; Model MODEL-2: 200",
                "metadata": {
                    "chunk_type": "table_record",
                    "table_column_headers": ["Address"],
                    "product_model": "MODEL-2",
                },
            },
        ],
    )

    assert scored["passed"] is False
    assert scored["missing_evidence"] == [{"chunk_id": "chunk-2", "matched": False, "rank": None, "overlap_terms": 2}]


def test_single_step_retrieval_scoring_rejects_same_document_without_expected_evidence():
    case = RetrievalEvalCase(
        case_id="external-trigger-polarity",
        query="In MOD-600 troubleshooting, when an external trigger cannot be input, which polarity checkpoint should I verify?",
        source_document_id="doc-iv4",
        document_version_id="ver-iv4",
        source_chunk_id="polarity-cell",
        source_title="IV4 Manual",
        source_filename="iv4.pdf",
        chunk_type="table_record",
        section_path="12-38",
        page_from=526,
        page_to=526,
        expected_terms=["polarity", "correctly", "set", "external", "trigger"],
        expected_snippet=(
            "Column headers: Check point; Row headers: An external trigger cannot be input.; "
            "Cell value: Is the Polarity correctly set?; Row: 11; Column: 1"
        ),
        generation_method="unit_test",
        source_metadata={
            "product_model": "MOD-600",
            "table_cell": True,
            "table_row_headers": ["An external trigger cannot be input."],
            "table_column_headers": ["Check point"],
        },
        anchor_terms=["polarity", "correctly", "set", "external", "trigger"],
        expected_evidence=[
            {
                "chunk_id": "polarity-cell",
                "source_document_id": "doc-iv4",
                "field": "external_trigger_polarity_checkpoint",
                "expected_terms": ["polarity", "correctly", "set", "external", "trigger"],
                "snippet": "An external trigger cannot be input. Check point: Is the Polarity correctly set?",
            }
        ],
    )

    scored = score_search_results(
        case,
        [
            {
                "chunk_id": "internal-trigger-workaround",
                "source_document_id": "doc-iv4",
                "title": "IV4 Manual",
                "section_path": ["Take an OK image and an NG image"],
                "content": (
                    "In the following cases, input an external trigger: When [External Trigger] is set in "
                    "Trigger Options. If a trigger cannot be input, press [Trigger ON] to take an image "
                    "temporarily using the internal trigger."
                ),
                "metadata": {"chunk_type": "section_window", "product_model": "MOD-600"},
            }
        ],
    )

    assert scored["passed"] is False
    assert scored["failure_category"] == "ranking_or_context_loss"


def test_single_step_retrieval_scoring_accepts_same_document_supported_expected_evidence():
    case = RetrievalEvalCase(
        case_id="external-trigger-polarity",
        query="In MOD-600 troubleshooting, when an external trigger cannot be input, which polarity checkpoint should I verify?",
        source_document_id="doc-iv4",
        document_version_id="ver-iv4",
        source_chunk_id="polarity-cell",
        source_title="IV4 Manual",
        source_filename="iv4.pdf",
        chunk_type="table_record",
        section_path="12-38",
        page_from=526,
        page_to=526,
        expected_terms=["polarity", "correctly", "set", "external", "trigger"],
        expected_snippet=(
            "Column headers: Check point; Row headers: An external trigger cannot be input.; "
            "Cell value: Is the Polarity correctly set?; Row: 11; Column: 1"
        ),
        generation_method="unit_test",
        source_metadata={
            "product_model": "MOD-600",
            "table_cell": True,
            "table_row_headers": ["An external trigger cannot be input."],
            "table_column_headers": ["Check point"],
        },
        anchor_terms=["polarity", "correctly", "set", "external", "trigger"],
        expected_evidence=[
            {
                "chunk_id": "polarity-cell",
                "source_document_id": "doc-iv4",
                "field": "external_trigger_polarity_checkpoint",
                "expected_terms": ["polarity", "correctly", "set", "external", "trigger"],
                "snippet": "An external trigger cannot be input. Check point: Is the Polarity correctly set?",
            }
        ],
    )

    scored = score_search_results(
        case,
        [
            {
                "chunk_id": "same-source-section-window",
                "source_document_id": "doc-iv4",
                "title": "IV4 Manual",
                "section_path": ["12-38"],
                "content": (
                    "Troubleshooting table. An external trigger cannot be input. "
                    "Check point: Is the Polarity correctly set?"
                ),
                "metadata": {"chunk_type": "section_window", "product_model": "MOD-600"},
            }
        ],
    )

    assert scored["passed"] is True
    assert scored["match_reason"] == "same_section_term_overlap"


def test_answer_response_scoring_prefers_multi_step_evidence_terms():
    case = RetrievalEvalCase(
        case_id="case-1",
        query="For the LJ-V head, what line count and overlap count are used?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing",
        page_from=1,
        page_to=1,
        expected_terms=["procedure", "typical", "operations", "description"],
        expected_snippet="The number of lines is 10 and number of overlap lines is two.",
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "chunk-1",
                "source_document_id": "doc-1",
                "expected_terms": ["procedure", "typical"],
                "snippet": "Procedure step 2: Typical operations at trigger input.",
            },
            {
                "chunk_id": "chunk-2",
                "source_document_id": "doc-1",
                "expected_terms": ["lines", "overlap"],
                "snippet": "For the purposes of this description, the number of lines is 10 and number of overlap lines is two.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "The timing example uses 10 lines and 2 overlap lines.",
            "citations": [{"document_id": "doc-1", "chunk_id": "chunk-2", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert scored["passed"] is True
    assert scored["term_check"]["term_source"] == "expected_evidence_specific_terms"
    assert {"lines", "overlap"}.issubset(set(scored["term_check"]["matched_terms"]))
    assert {"10", "two"}.issubset(set(scored["term_check"]["material_expected_terms"]))
    assert {"10", "two"}.issubset(set(scored["term_check"]["material_matched_terms"]))


def test_answer_response_scoring_rejects_quantity_answers_without_values():
    case = RetrievalEvalCase(
        case_id="case-1",
        query="For the LJ-V head, what line count and overlap count are used?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing",
        page_from=1,
        page_to=1,
        expected_terms=["procedure", "typical", "operations", "description"],
        expected_snippet="The number of lines is 10 and number of overlap lines is two.",
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "chunk-2",
                "source_document_id": "doc-1",
                "expected_terms": ["lines", "overlap"],
                "snippet": "For the purposes of this description, the number of lines is 10 and number of overlap lines is two.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "The timing example describes lines and overlap lines.",
            "citations": [{"document_id": "doc-1", "chunk_id": "chunk-2", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert scored["passed"] is False
    assert {"10", "two"}.issubset(set(scored["term_check"]["material_expected_terms"]))
    assert scored["term_check"]["material_matched_terms"] == []
    assert "expected_terms_missing" in scored["failure_reasons"]


def test_answer_response_scoring_requires_troubleshooting_corrective_action_terms():
    case = RetrievalEvalCase(
        case_id="case-ocr2-patterns",
        query=(
            "What causes The number of characters that can be registered for 1 character "
            "was exceeded for XG-X Series, and how should it be corrected?"
        ),
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["characters", "registered", "200", "patterns"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "error-cell",
                "source_document_id": "doc-xgx",
                "field": "error message",
                "expected_terms": ["characters", "registered", "exceeded"],
            },
            {
                "chunk_id": "cause-cell",
                "source_document_id": "doc-xgx",
                "field": "cause",
                "expected_terms": ["trying", "register", "200", "character"],
            },
            {
                "chunk_id": "action-cell",
                "source_document_id": "doc-xgx",
                "field": "corrective action",
                "expected_terms": ["ocr2", "maximum", "character", "patterns"],
                "snippet": (
                    "In the OCR2 unit, the maximum number of character patterns that can be "
                    "registered for one type of character is up to 200 character patterns. "
                    "Delete any unnecessary character patterns."
                ),
            },
        ],
    )

    unrelated_action = score_answer_response(
        case,
        {
            "answer": (
                "Use an asterisk in the extracted string and check the calculation script "
                "line and character limits."
            ),
            "citations": [{"document_id": "doc-xgx", "chunk_id": "action-cell", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )
    grounded_action = score_answer_response(
        case,
        {
            "answer": (
                "The OCR2 unit is trying to register more than 200 character patterns for "
                "one type of character. Keep the OCR2 character patterns within the 200 "
                "pattern limit and delete unnecessary character patterns."
            ),
            "citations": [{"document_id": "doc-xgx", "chunk_id": "action-cell", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert unrelated_action["passed"] is False
    assert (
        unrelated_action["term_check"]["material_term_source"]
        == "troubleshooting_action_and_quantity_terms"
    )
    assert {"delete", "ocr2"}.issubset(
        set(unrelated_action["term_check"]["material_expected_terms"])
    )
    assert "expected_terms_missing" in unrelated_action["failure_reasons"]
    assert grounded_action["passed"] is True
    assert {"delete", "ocr2"}.issubset(
        set(grounded_action["term_check"]["material_matched_terms"])
    )


def test_answer_response_scoring_ignores_row_header_action_artifacts():
    case = RetrievalEvalCase(
        case_id="case-unsupported-firmware",
        query=(
            "What causes the XG-X controller to boot with unsupported firmware, "
            "and how should it be corrected?"
        ),
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["controller", "booted", "unsupported", "firmware"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "error-cell",
                "source_document_id": "doc-xgx",
                "field": "error message",
                "expected_terms": ["controller", "booted", "unsupported", "firmware"],
            },
            {
                "chunk_id": "cause-cell",
                "source_document_id": "doc-xgx",
                "field": "cause",
                "expected_terms": ["controller", "started", "supported", "firmware"],
            },
            {
                "chunk_id": "action-cell",
                "source_document_id": "doc-xgx",
                "field": "corrective action",
                "expected_terms": ["update", "firmware", "supporting", "controller"],
                "snippet": (
                    "Column headers: Corrective Action; Row headers: and check the file > "
                    "The controller was booted using an unsupported firmware.; "
                    "Cell value: Update to the firmware supporting the controller.; Row: 8; Column: 2"
                ),
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "The controller was started with an unsupported firmware version. "
                "Update the firmware to a version that supports the controller."
            ),
            "citations": [{"document_id": "doc-xgx", "chunk_id": "action-cell", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert scored["passed"] is True
    assert "check" not in scored["term_check"]["material_expected_terms"]
    assert "update" in scored["term_check"]["material_matched_terms"]


def test_answer_response_scoring_requires_corrective_action_state_terms():
    case = RetrievalEvalCase(
        case_id="case-retest-online-3d-camera",
        query=(
            "What causes Retest mode (online) cannot be used when using a 3D camera "
            "for XG-X Series, and how should it be corrected?"
        ),
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["retest", "online", "3d", "camera"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "error-cell",
                "source_document_id": "doc-xgx",
                "field": "error message",
                "expected_terms": ["retest", "online", "camera"],
            },
            {
                "chunk_id": "cause-cell",
                "source_document_id": "doc-xgx",
                "field": "cause",
                "expected_terms": ["attempt", "set", "camera", "model"],
            },
            {
                "chunk_id": "action-cell",
                "source_document_id": "doc-xgx",
                "field": "corrective action",
                "expected_terms": ["set", "system", "processing", "offline"],
                "snippet": (
                    "Column headers: Corrective Action; Cell value: "
                    "Set [System Processing] to [Offline] in the image viewer option settings."
                ),
            },
        ],
    )

    sibling_row_action = score_answer_response(
        case,
        {
            "answer": (
                "The error occurs because [System Processing] was changed to [Online] "
                "with an XT or XR camera as the camera model. Correct it by setting a "
                "camera other than a 21-megapixel or greater camera."
            ),
            "citations": [{"document_id": "doc-xgx", "chunk_id": "action-cell", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )
    grounded_state_action = score_answer_response(
        case,
        {
            "answer": (
                "The controller is trying to use Retest mode online with an unsupported "
                "camera setting. Set [System Processing] to [Offline] in the image viewer "
                "option settings."
            ),
            "citations": [{"document_id": "doc-xgx", "chunk_id": "action-cell", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert sibling_row_action["passed"] is False
    assert "offline" in sibling_row_action["term_check"]["material_expected_terms"]
    assert "offline" not in sibling_row_action["term_check"]["material_matched_terms"]
    assert "expected_terms_missing" in sibling_row_action["failure_reasons"]
    assert grounded_state_action["passed"] is True
    assert "offline" in grounded_state_action["term_check"]["material_matched_terms"]


def test_answer_response_scoring_requires_action_verb_even_when_query_mentions_change():
    case = RetrievalEvalCase(
        case_id="case-unsupported-trigger-signal",
        query=(
            "On an XG-X Series controller, if Allow Trigger Input During Line Capture and "
            "End Capture By EXT Signal are enabled together and the invalid camera setting "
            "error says a trigger signal that cannot be used is assigned, what should I change?"
        ),
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["image", "capture", "trigger", "signal"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "error-cell",
                "source_document_id": "doc-xgx",
                "field": "error message",
                "expected_terms": ["image", "capture", "invalid", "trigger"],
            },
            {
                "chunk_id": "cause-cell",
                "source_document_id": "doc-xgx",
                "field": "cause",
                "expected_terms": ["trigger", "signals", "even-number", "camera"],
            },
            {
                "chunk_id": "action-cell",
                "source_document_id": "doc-xgx",
                "field": "corrective action",
                "expected_terms": ["change", "trigger", "signal"],
                "snippet": "Cell value: Change to a trigger signal that can be used.; Row: 3; Column: 2",
            },
        ],
    )

    sibling_row_action = score_answer_response(
        case,
        {
            "answer": "To resolve the error, disable capture on trigger input in the trigger settings.",
            "citations": [{"document_id": "doc-xgx", "chunk_id": "other-cell", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )
    grounded_action = score_answer_response(
        case,
        {
            "answer": "Change to a trigger signal that can be used.",
            "citations": [{"document_id": "doc-xgx", "chunk_id": "action-cell", "pages": [1]}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert sibling_row_action["passed"] is False
    assert "change" in sibling_row_action["term_check"]["material_expected_terms"]
    assert "3" not in sibling_row_action["term_check"]["material_expected_terms"]
    assert "change" not in sibling_row_action["term_check"]["material_matched_terms"]
    assert "expected_terms_missing" in sibling_row_action["failure_reasons"]
    assert grounded_action["passed"] is True
    assert "change" in grounded_action["term_check"]["material_matched_terms"]


def test_answer_response_scoring_uses_source_only_action_terms():
    case = RetrievalEvalCase(
        case_id="case-unsupported-trigger-signal",
        query=(
            "On an XG-X Series controller, if Allow Trigger Input During Line Capture and "
            "End Capture By EXT Signal are enabled together and the invalid camera setting "
            "error says a trigger signal that cannot be used is assigned, what should I change?"
        ),
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["image", "capture", "trigger", "signal"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "action-cell",
                "source_document_id": "doc-xgx",
                "field": "corrective action",
                "expected_terms": ["change", "trigger", "signal"],
                "snippet": "Cell value: Change to a trigger signal that can be used.; Row: 3; Column: 2",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Change the trigger signal assignment to one that is valid for the settings.",
            "citations": [
                {
                    "document_id": "doc-xgx",
                    "chunk_id": "settings-page",
                    "pages": [985],
                    "quote_span": (
                        "Change the trigger signal assignment to one that is valid for the settings."
                    ),
                }
            ],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
    )

    assert scored["term_check"]["material_expected_terms"] == ["change", "change trigger signal", "trigger", "signal"]
    assert "one" not in scored["term_check"]["material_expected_terms"]


def test_answer_response_scoring_requires_troubleshooting_action_object_terms():
    case = RetrievalEvalCase(
        case_id="case-trigger-signal-action-object",
        query=(
            "On an XG-X Series controller, if Allow Trigger Input During Line Capture and "
            "End Capture By EXT Signal are enabled together and the invalid camera setting "
            "error says a trigger signal that cannot be used is assigned, what should I change?"
        ),
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["image", "capture", "trigger", "signal"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "action-cell",
                "source_document_id": "doc-xgx",
                "field": "corrective action",
                "expected_terms": ["change", "trigger", "signal"],
                "snippet": "Cell value: Change to a trigger signal that can be used.; Row: 3; Column: 2",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "If capture on trigger input is disabled, this setting cannot be changed.",
            "citations": [
                {
                    "document_id": "doc-xgx",
                    "chunk_id": "settings-page",
                    "pages": [985],
                    "quote_span": None,
                }
            ],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "settings-page",
                "content": "If capture on trigger input is disabled, this setting cannot be changed.",
            }
        ],
    )

    assert scored["passed"] is False
    assert scored["term_check"]["material_expected_terms"] == ["change", "change trigger signal", "trigger", "signal"]
    assert "change trigger signal" not in scored["term_check"]["material_matched_terms"]
    assert "signal" not in scored["term_check"]["material_matched_terms"]
    assert "expected_terms_missing" in scored["failure_reasons"]


def test_answer_response_scoring_allows_source_action_phrase_with_intervening_words():
    case = RetrievalEvalCase(
        case_id="case-flash-output",
        query="What causes light controller communication errors for CV-X482, and how should they be corrected?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="error-cell",
        source_title="CV-X",
        source_filename="cvx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1274,
        page_to=1274,
        expected_terms=["error", "communication", "light", "flash"],
        expected_snippet="Error, cause, and remedy",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_model": "CV-X482"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "remedy-cell",
                "source_document_id": "doc-cvx",
                "field": "remedy",
                "expected_terms": ["set", "flash", "0.1", "lighting"],
                "snippet": "Cell value: Set the FLASH output time to 0.1 msec.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "The light-controller communication error is caused by the next FLASH being input "
                "while the light is emitted. Set the FLASH output time to 0.1 msec."
            ),
            "citations": [
                {
                    "document_id": "doc-cvx",
                    "chunk_id": "remedy-cell",
                    "pages": [1274],
                    "quote_span": "Set the FLASH output time to 0.1 msec.",
                }
            ],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "remedy-cell",
                "content": "Remedy: Set the FLASH output time to 0.1 msec.",
            }
        ],
    )

    assert scored["passed"] is True
    assert "set flash 0.1" in scored["term_check"]["material_matched_terms"]


def test_answer_response_scoring_rejects_unsupported_citation_quote_spans():
    case = RetrievalEvalCase(
        case_id="case-unsupported-trigger-signal",
        query=(
            "On an XG-X Series controller, if Allow Trigger Input During Line Capture and "
            "End Capture By EXT Signal are enabled together and the invalid camera setting "
            "error says a trigger signal that cannot be used is assigned, what should I change?"
        ),
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["image", "capture", "trigger", "signal"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "action-cell",
                "source_document_id": "doc-xgx",
                "field": "corrective action",
                "expected_terms": ["change", "trigger", "signal"],
                "snippet": "Cell value: Change to a trigger signal that can be used.; Row: 3; Column: 2",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Change to a trigger signal that can be used.",
            "citations": [
                {
                    "document_id": "doc-xgx",
                    "chunk_id": "settings-page",
                    "pages": [985],
                    "quote_span": "Change to a trigger signal that can be used.",
                }
            ],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "settings-page",
                "content": (
                    "Trigger delay can be set when capture-on-trigger input is enabled. "
                    "Live image display is disabled for some trigger settings."
                ),
            }
        ],
    )

    assert scored["passed"] is False
    assert "unsupported_citation_quote" in scored["failure_reasons"]
    assert scored["citation_fidelity"]["unsupported_quotes"][0]["chunk_id"] == "settings-page"


def test_answer_response_scoring_rejects_null_quote_citations_missing_from_results():
    case = RetrievalEvalCase(
        case_id="case-missing-cited-context",
        query=(
            "For CV-X Multi-Capture trigger input timing, what operation does the section "
            "describe and which control/data I/O timing chart should I use?"
        ),
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="multi-capture-step",
        source_title="CV-X",
        source_filename="cvx.pdf",
        chunk_type="procedure_record",
        section_path="Timing",
        page_from=853,
        page_to=853,
        expected_terms=["multi-capture", "trigger", "timing"],
        expected_snippet="Typical operations at trigger input",
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "expected_terms": ["multi-capture", "multiple", "captures", "single", "measurement"],
                "snippet": (
                    "Performs multiple image captures at the same location and processes "
                    "them as a single measurement."
                ),
            },
            {
                "chunk_id": "timing-chart",
                "source_document_id": "doc-cvx",
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "The section describes Multi-Capture performing multiple image captures "
                "at one location as a single measurement. Use the control/data I/O "
                "terminal timing chart."
            ),
            "citations": [
                {"document_id": "doc-cvx", "chunk_id": "multi-capture-step", "quote_span": None}
            ],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "nearby-trigger-section",
                "content": "Asynchronous Trigger timing for another procedure section.",
            }
        ],
    )

    assert scored["passed"] is False
    assert "unsupported_citation_quote" in scored["failure_reasons"]
    assert "expected_evidence_not_cited" in scored["failure_reasons"]
    assert scored["citation_fidelity"]["missing_cited_chunks"] == ["multi-capture-step"]


def test_answer_response_scoring_requires_cited_chunk_to_support_expected_evidence_role():
    case = RetrievalEvalCase(
        case_id="case-cross-document-spec-role",
        query=(
            "For controller A, what enclosure rating is listed, and for controller B, "
            "what shock-resistance value is listed?"
        ),
        source_document_id="doc-a",
        document_version_id="ver-a",
        source_chunk_id="enclosure-a",
        source_title="manual-a",
        source_filename="manual-a.pdf",
        chunk_type="table_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["ip67", "500", "m/s", "directions"],
        expected_snippet="Cross-document spec comparison",
        generation_method="cross_document_same_field_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "enclosure-a",
                "source_document_id": "doc-a",
                "field": "enclosure rating",
                "expected_terms": ["ip67"],
                "snippet": "Cell value: IP67",
            },
            {
                "chunk_id": "shock-b",
                "source_document_id": "doc-b",
                "field": "shock resistance",
                "expected_terms": ["500", "m/s", "directions"],
                "snippet": "Cell value: 500 m/s 2 , 6 different directions",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Controller A is IP67, and controller B is rated for 500 m/s in 6 directions.",
            "citations": [
                {"document_id": "doc-a", "chunk_id": "enclosure-a", "pages": [1]},
                {"document_id": "doc-b", "chunk_id": "enclosure-b", "pages": [2]},
            ],
            "used_documents": [
                {"document_id": "doc-a"},
                {"document_id": "doc-b"},
            ],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "enclosure-a",
                "content": "Column headers: Controller A; Row headers: Enclosure rating; Cell value: IP67",
            },
            {
                "chunk_id": "enclosure-b",
                "content": "Column headers: Controller B; Row headers: Enclosure rating; Cell value: IP67",
            },
            {
                "chunk_id": "shock-b",
                "content": (
                    "Column headers: Controller B; Row headers: Shock resistance; "
                    "Cell value: 500 m/s 2 , 6 different directions"
                ),
            },
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]
    assert scored["evidence_citation_support"]["missing_evidence"] == [
        {
            "chunk_id": "shock-b",
            "source_document_id": "doc-b",
            "expected_terms": ["500", "m/s", "directions"],
            "reason": "expected_evidence_not_supported_by_citations",
        }
    ]

    cited_role = score_answer_response(
        case,
        {
            "answer": "Controller A is IP67, and controller B is rated for 500 m/s in 6 directions.",
            "citations": [
                {"document_id": "doc-a", "chunk_id": "enclosure-a", "pages": [1]},
                {"document_id": "doc-b", "chunk_id": "shock-b", "pages": [2]},
            ],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "enclosure-a",
                "content": "Column headers: Controller A; Row headers: Enclosure rating; Cell value: IP67",
            },
            {
                "chunk_id": "shock-b",
                "content": (
                    "Column headers: Controller B; Row headers: Shock resistance; "
                    "Cell value: 500 m/s 2 , 6 different directions"
                ),
            },
        ],
    )

    assert cited_role["passed"] is True
    assert cited_role["evidence_citation_support"]["passed"] is True


def test_answer_response_scoring_requires_exact_cited_chunk_to_be_returned():
    case = RetrievalEvalCase(
        case_id="answer-exact-chunk-returned",
        query="What should I do when the trigger signal cannot be used?",
        source_document_id="doc-a",
        document_version_id="ver-a",
        source_chunk_id="symptom-a",
        source_title="Troubleshooting",
        source_filename="Manual.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["change", "trigger", "signal", "used"],
        expected_snippet="Change to a trigger signal that can be used.",
        generation_method="table_sibling_error_cause_action",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "remedy-a",
                "source_document_id": "doc-a",
                "field": "corrective action",
                "expected_terms": ["change", "trigger", "signal", "used"],
                "snippet": "Change to a trigger signal that can be used.",
            }
        ],
    )

    missing_returned_chunk = score_answer_response(
        case,
        {
            "answer": "Change to a trigger signal that can be used.",
            "citations": [{"document_id": "doc-a", "chunk_id": "remedy-a", "quote_span": None}],
            "used_documents": [{"document_id": "doc-a"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "nearby-a",
                "content": "Column headers: Cause; Cell value: A trigger signal that cannot be used is assigned.",
            }
        ],
    )

    assert missing_returned_chunk["passed"] is False
    assert "expected_evidence_not_cited" in missing_returned_chunk["failure_reasons"]

    returned_expected_chunk = score_answer_response(
        case,
        {
            "answer": "Change to a trigger signal that can be used.",
            "citations": [{"document_id": "doc-a", "chunk_id": "remedy-a", "quote_span": None}],
            "used_documents": [{"document_id": "doc-a"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "remedy-a",
                "content": "Column headers: Corrective action; Cell value: Change to a trigger signal that can be used.",
            }
        ],
    )

    assert returned_expected_chunk["passed"] is True


def test_answer_response_scoring_rejects_null_quote_nearby_sibling_chunk():
    case = RetrievalEvalCase(
        case_id="answer-null-quote-nearby-sibling",
        query="What shock resistance value is listed for the camera?",
        source_document_id="doc-b",
        document_version_id="ver-b",
        source_chunk_id="shock-b",
        source_title="Specifications",
        source_filename="Manual.pdf",
        chunk_type="table_record",
        section_path="Specifications",
        page_from=2,
        page_to=2,
        expected_terms=["500", "m/s", "directions"],
        expected_snippet="Cell value: 500 m/s 2 , 6 different directions",
        generation_method="cross_document_same_field_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "shock-b",
                "source_document_id": "doc-b",
                "field": "shock resistance",
                "expected_terms": ["500", "m/s", "directions"],
                "snippet": "Cell value: 500 m/s 2 , 6 different directions",
            }
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "The camera is rated for 500 m/s in 6 directions.",
            "citations": [{"document_id": "doc-b", "chunk_id": "enclosure-b", "quote_span": None}],
            "used_documents": [{"document_id": "doc-b"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "enclosure-b",
                "content": "Column headers: Camera; Row headers: Enclosure rating; Cell value: IP67",
            },
            {
                "chunk_id": "shock-b",
                "content": (
                    "Column headers: Camera; Row headers: Shock resistance; "
                    "Cell value: 500 m/s 2 , 6 different directions"
                ),
            },
        ],
    )

    assert scored["passed"] is False
    assert scored["evidence_citation_support"]["missing_evidence"][0]["chunk_id"] == "shock-b"


def test_answer_response_scoring_requires_expected_chunk_not_same_document_overlap():
    case = RetrievalEvalCase(
        case_id="answer-same-document-overlap",
        query="Compare what the Condition list and Standard Angle settings control.",
        source_document_id="doc-cv",
        document_version_id="ver-cv",
        source_chunk_id="condition-list",
        source_title="Settings",
        source_filename="Manual.pdf",
        chunk_type="table_record",
        section_path="Settings",
        page_from=1,
        page_to=1,
        expected_terms=["condition", "reference", "standard", "angle"],
        expected_snippet="Condition list controls reference conditions; Standard Angle controls blob numbering.",
        generation_method="manual_curated_cross_document_v2_from_diagnostic_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "condition-list",
                "source_document_id": "doc-cv",
                "expected_terms": ["condition", "reference"],
                "snippet": "A maximum of 16 reference conditions can be set.",
            },
            {
                "chunk_id": "standard-angle",
                "source_document_id": "doc-lj",
                "expected_terms": ["standard", "angle", "numbering"],
                "snippet": "Specifies the start angle for blob numbering.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Condition and angle settings are listed in the manuals.",
            "citations": [
                {"document_id": "doc-cv", "chunk_id": "angle-range", "quote_span": None},
                {"document_id": "doc-lj", "chunk_id": "communication-toc", "quote_span": None},
            ],
            "used_documents": [{"document_id": "doc-cv"}, {"document_id": "doc-lj"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "angle-range",
                "content": "Setting item: Angle range; Settings: Specifies an angle range for tilted targets.",
            },
            {
                "chunk_id": "communication-toc",
                "content": "Changing message Condition Overview. Communication command table.",
            },
            {
                "chunk_id": "condition-list",
                "content": "Setting item: Condition list; Settings: A maximum of 16 reference conditions can be set.",
            },
            {
                "chunk_id": "standard-angle",
                "content": "Setting item: Standard Angle; Settings: Specifies the start angle for blob numbering.",
            },
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]
    assert {
        item["chunk_id"] for item in scored["evidence_citation_support"]["missing_evidence"]
    } == {"condition-list", "standard-angle"}


def test_answer_response_scoring_accepts_cited_composite_chunk_with_source_evidence():
    case = RetrievalEvalCase(
        case_id="answer-composite-source-evidence",
        query="For asynchronous trigger input, what happens when TRG1 is input for camera 1?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="procedure-step",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=113,
        page_to=113,
        expected_terms=["trigger", "camera", "trg1", "measurement"],
        expected_snippet="TRG1 for CAM 1 executes capture and measurement processing.",
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "procedure-step",
                "source_document_id": "doc-cvx",
                "allow_equivalent_citation": True,
                "expected_terms": ["procedure", "typical", "operations", "trigger"],
                "snippet": "Procedure step 2: Typical operations at trigger input (Capture Type: Asynchronous Trigger)",
            },
            {
                "chunk_id": "detail-atomic",
                "source_document_id": "doc-cvx",
                "allow_equivalent_citation": True,
                "expected_terms": ["inputting", "trigger", "signal", "having"],
                "snippet": (
                    "By inputting the trigger signal having the same number as the camera number "
                    "(TRG1 for CAM 1), capture and measurement processing is executed."
                ),
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "When TRG1 is input for CAM 1, capture and measurement processing for that camera "
                "is executed as one measurement."
            ),
            "citations": [{"document_id": "doc-cvx", "chunk_id": "section-window", "quote_span": None}],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "source_document_id": "doc-cvx",
                "content": (
                    "2. Typical operations at trigger input (Capture Type: Asynchronous Trigger). "
                    "By inputting the trigger signal having the same number as the camera number "
                    "(TRG1 for CAM 1), capture + measurement processing is executed as 1 measurement."
                ),
            }
        ],
    )

    assert scored["passed"] is True
    assert scored["evidence_citation_support"]["passed"] is True


def test_answer_response_scoring_rejects_equivalent_citation_without_source_identity():
    case = RetrievalEvalCase(
        case_id="answer-equivalent-citation-no-source",
        query="Which timing chart applies to the operation?",
        source_document_id="doc-controller",
        document_version_id="ver-controller",
        source_chunk_id="timing-heading",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing",
        page_from=12,
        page_to=12,
        expected_terms=["timing", "chart", "terminals"],
        expected_snippet="Timing chart Control/data output via I/O terminals",
        generation_method="unit_test",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "timing-heading",
                "allow_equivalent_citation": True,
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Use the control/data output via I/O terminals timing chart.",
            "citations": [{"chunk_id": "section-window", "quote_span": None}],
            "used_documents": [],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "content": "Timing chart Control/data output via I/O terminals.",
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_accepts_same_document_equivalent_citation():
    case = RetrievalEvalCase(
        case_id="answer-equivalent-citation-same-source",
        query="Which timing chart applies to the operation?",
        source_document_id="doc-controller",
        document_version_id="ver-controller",
        source_chunk_id="timing-heading",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing",
        page_from=12,
        page_to=12,
        expected_terms=["timing", "chart", "terminals"],
        expected_snippet="Timing chart Control/data output via I/O terminals",
        generation_method="unit_test",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "timing-heading",
                "source_document_id": "doc-controller",
                "allow_equivalent_citation": True,
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Use the control/data output via I/O terminals timing chart.",
            "citations": [{"document_id": "doc-controller", "chunk_id": "section-window", "quote_span": None}],
            "used_documents": [{"document_id": "doc-controller"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "source_document_id": "doc-controller",
                "content": "Timing chart Control/data output via I/O terminals.",
            }
        ],
    )

    assert scored["passed"] is True
    assert scored["evidence_citation_support"]["passed"] is True


def test_answer_response_scoring_accepts_same_document_section_window_for_atomic_roles():
    case = RetrievalEvalCase(
        case_id="answer-equivalent-section-window",
        query="For LJ-X8000 image capture timing, what does the setup chapter cover before configuration?",
        source_document_id="doc-ljx",
        document_version_id="ver-ljx",
        source_chunk_id="procedure-heading",
        source_title="LJ-X8000 EtherNet/IP",
        source_filename="ljx.pdf",
        chunk_type="procedure_record",
        section_path="1.2 Checking the Connection",
        page_from=15,
        page_to=15,
        expected_terms=["controlling", "image", "capture", "inspection"],
        expected_snippet="Controlling the Image Capture Timing. This chapter explains how to output inspection results.",
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "procedure-heading",
                "source_document_id": "doc-ljx",
                "allow_equivalent_citation": True,
                "expected_terms": ["procedure", "controlling", "image", "capture"],
                "snippet": "Procedure step 2: 2. Controlling the Image Capture Timing",
            },
            {
                "chunk_id": "chapter-detail",
                "source_document_id": "doc-ljx",
                "allow_equivalent_citation": True,
                "expected_terms": ["chapter", "explains", "inspection", "results"],
                "snippet": "This chapter explains how to output data such as inspection results.",
            },
        ],
    )

    answer = {
        "answer": (
            "The setup chapter covers controlling image capture timing and explains how to "
            "output inspection results and measured values to the PLC."
        ),
        "citations": [{"document_id": "doc-ljx", "chunk_id": "section-window", "quote_span": None}],
        "used_documents": [{"document_id": "doc-ljx"}],
        "insufficient_evidence": False,
    }

    scored = score_answer_response(
        case,
        answer,
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "source_document_id": "doc-ljx",
                "content": (
                    "1.2 Checking the Connection. 2. Controlling the Image Capture Timing. "
                    "This chapter explains how to output data such as inspection results "
                    "and measured values from the LJ-X8000 to the PLC over EtherNet/IP."
                ),
            }
        ],
    )

    assert scored["passed"] is True
    assert scored["evidence_citation_support"]["passed"] is True

    missing_role = score_answer_response(
        case,
        answer,
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "source_document_id": "doc-ljx",
                "content": "1.2 Checking the Connection. Use the EtherNet/IP memory monitor.",
            }
        ],
    )

    assert missing_role["passed"] is False
    assert "expected_evidence_not_cited" in missing_role["failure_reasons"]


def test_answer_response_scoring_rejects_cross_document_equivalent_citation():
    case = RetrievalEvalCase(
        case_id="answer-equivalent-citation-cross-source",
        query="Which timing chart applies to the operation?",
        source_document_id="doc-controller",
        document_version_id="ver-controller",
        source_chunk_id="timing-heading",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing",
        page_from=12,
        page_to=12,
        expected_terms=["timing", "chart", "terminals"],
        expected_snippet="Timing chart Control/data output via I/O terminals",
        generation_method="unit_test",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "timing-heading",
                "source_document_id": "doc-controller",
                "allow_equivalent_citation": True,
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Use the control/data output via I/O terminals timing chart.",
            "citations": [{"document_id": "doc-other", "chunk_id": "section-window", "quote_span": None}],
            "used_documents": [{"document_id": "doc-other"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "source_document_id": "doc-other",
                "content": "Timing chart Control/data output via I/O terminals.",
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_document_not_cited_or_used" in scored["failure_reasons"]
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_requires_operation_fact_terms_from_source_evidence():
    case = RetrievalEvalCase(
        case_id="answer-operation-source-fact",
        query=(
            "For CV-X Multi-Capture trigger input timing, what operation does the section describe "
            "and which control/data I/O timing chart should I use?"
        ),
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="multi-capture-step",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=114,
        page_to=114,
        expected_terms=["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
        expected_snippet=(
            "Timing chart Control/data output via I/O terminals; "
            "Typical operations at trigger input (Capture Type: Multi-Capture); "
            "Performs multiple image captures at the same location and processes them as a single measurement."
        ),
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "expected_terms": ["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
                "snippet": (
                    "Procedure step 2: Typical operations at trigger input (Capture Type: Multi-Capture). "
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
            {
                "chunk_id": "timing-chart",
                "source_document_id": "doc-cvx",
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "Use the Multi-Capture trigger input timing section and the control/data I/O timing chart."
            ),
            "citations": [
                {"document_id": "doc-cvx", "chunk_id": "multi-capture-step", "quote_span": None},
                {"document_id": "doc-cvx", "chunk_id": "timing-chart", "quote_span": None},
            ],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "content": (
                    "Typical operations at trigger input (Capture Type: Multi-Capture). "
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
            {
                "chunk_id": "timing-chart",
                "source_document_id": "doc-cvx",
                "content": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    assert scored["passed"] is False
    assert "expected_terms_missing" in scored["failure_reasons"]
    assert scored["term_check"]["material_expected_terms"] == ["performs", "multiple", "image", "captures"]


def test_answer_response_scoring_accepts_operation_fact_terms_from_source_evidence():
    case = RetrievalEvalCase(
        case_id="answer-operation-source-fact-clean",
        query=(
            "For CV-X Multi-Capture trigger input timing, what operation does the section describe "
            "and which control/data I/O timing chart should I use?"
        ),
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="multi-capture-step",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=114,
        page_to=114,
        expected_terms=["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
        expected_snippet=(
            "Timing chart Control/data output via I/O terminals; "
            "Typical operations at trigger input (Capture Type: Multi-Capture); "
            "Performs multiple image captures at the same location and processes them as a single measurement."
        ),
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "expected_terms": ["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
                "snippet": (
                    "Procedure step 2: Typical operations at trigger input (Capture Type: Multi-Capture). "
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
            {
                "chunk_id": "timing-chart",
                "source_document_id": "doc-cvx",
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "The section says Multi-Capture performs multiple image captures at the same location "
                "and processes them as a single measurement; use the control/data I/O timing chart."
            ),
            "citations": [
                {"document_id": "doc-cvx", "chunk_id": "multi-capture-step", "quote_span": None},
                {"document_id": "doc-cvx", "chunk_id": "timing-chart", "quote_span": None},
            ],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "content": (
                    "Typical operations at trigger input (Capture Type: Multi-Capture). "
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
            {
                "chunk_id": "timing-chart",
                "source_document_id": "doc-cvx",
                "content": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    assert scored["passed"] is True
    assert scored["term_check"]["material_matched_terms"] == ["performs", "multiple", "image", "captures"]


def test_answer_response_scoring_rejects_expected_role_only_from_cited_local_context():
    case = RetrievalEvalCase(
        case_id="answer-operation-source-context-clean",
        query=(
            "For CV-X Multi-Capture trigger input timing, what operation does the section describe "
            "and which control/data I/O timing chart should I use?"
        ),
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="multi-capture-step",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=114,
        page_to=114,
        expected_terms=["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
        expected_snippet=(
            "Timing chart Control/data output via I/O terminals; "
            "Typical operations at trigger input (Capture Type: Multi-Capture); "
            "Performs multiple image captures at the same location and processes them as a single measurement."
        ),
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "expected_terms": ["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
                "snippet": (
                    "Procedure step 2: Typical operations at trigger input (Capture Type: Multi-Capture). "
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
            {
                "chunk_id": "timing-chart",
                "source_document_id": "doc-cvx",
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "The section says Multi-Capture performs multiple image captures at the same location "
                "and processes them as a single measurement; use the control/data I/O timing chart."
            ),
            "citations": [{"document_id": "doc-cvx", "chunk_id": "multi-capture-step", "quote_span": None}],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "content": "Procedure step 2: Typical operations at trigger input (Capture Type: Multi-Capture).",
                "metadata": {
                    "local_rerank_context": (
                        "Timing chart Control/data output via I/O terminals. "
                        "2. Typical operations at trigger input (Capture Type: Multi-Capture). "
                        "Performs multiple image captures at the same location and processes them as a single measurement."
                    )
                },
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]
    assert scored["evidence_citation_support"]["missing_evidence"] == [
        {
            "chunk_id": "timing-chart",
            "source_document_id": "doc-cvx",
            "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
            "reason": "expected_evidence_not_supported_by_citations",
        }
    ]


def test_answer_response_scoring_rejects_scattered_persisted_citation_context_terms():
    case = RetrievalEvalCase(
        case_id="answer-consolidated-scattered-context",
        query=(
            "For CV-X Multi-Capture trigger input timing, what operation does the section describe "
            "and which control/data I/O timing chart should I use?"
        ),
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="multi-capture-step",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=114,
        page_to=114,
        expected_terms=["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
        expected_snippet=(
            "Timing chart Control/data output via I/O terminals; "
            "Typical operations at trigger input (Capture Type: Multi-Capture); "
            "Performs multiple image captures at the same location and processes them as a single measurement."
        ),
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "timing-chart",
                "source_document_id": "doc-cvx",
                "expected_terms": ["timing", "chart", "control/data", "i/o", "terminals"],
                "snippet": "Timing chart Control/data output via I/O terminals",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "The section says Multi-Capture performs multiple image captures at the same location "
                "and processes them as a single measurement; use the control/data I/O timing chart."
            ),
            "citations": [{"document_id": "doc-cvx", "chunk_id": "multi-capture-step", "quote_span": None}],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "content": "Procedure step 2: Typical operations at trigger input (Capture Type: Multi-Capture).",
                "metadata": {
                    "content": (
                        "Timing setup overview. The diagnostic chart is listed elsewhere. "
                        "Control output is configured separately. Data export uses I/O mapping. "
                        "Terminal labels are shown on the next page."
                    )
                },
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_rejects_wrong_operation_in_persisted_citation_context():
    case = RetrievalEvalCase(
        case_id="answer-consolidated-wrong-operation",
        query=(
            "For CV-X Multi-Capture trigger input timing, what operation does the section describe "
            "and which control/data I/O timing chart should I use?"
        ),
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="multi-capture-step",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=114,
        page_to=114,
        expected_terms=["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
        expected_snippet=(
            "Timing chart Control/data output via I/O terminals; "
            "Typical operations at trigger input (Capture Type: Multi-Capture); "
            "Performs multiple image captures at the same location and processes them as a single measurement."
        ),
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "multi-capture-step",
                "source_document_id": "doc-cvx",
                "expected_terms": ["multi-capture", "trigger", "multiple", "image", "captures", "single", "measurement"],
                "snippet": (
                    "Procedure step 2: Typical operations at trigger input (Capture Type: Multi-Capture). "
                    "Performs multiple image captures at the same location and processes them as a single measurement."
                ),
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": (
                "The section says Multi-Capture performs multiple image captures at the same location "
                "and processes them as a single measurement."
            ),
            "citations": [{"document_id": "doc-cvx", "chunk_id": "section-window", "quote_span": None}],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "source_document_id": "doc-cvx",
                "content": "Procedure step 2: Typical operations at trigger input (Capture Type: Multi-Capture).",
                "metadata": {
                    "content": (
                        "Timing chart Control/data output via I/O terminals. "
                        "Typical operations at trigger input (Capture Type: Multi-Capture). "
                        "For asynchronous trigger operation, one image is captured for one measurement."
                    )
                },
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_rejects_composite_chunk_without_source_role_terms():
    case = RetrievalEvalCase(
        case_id="answer-composite-source-evidence-negative",
        query="For asynchronous trigger input, what happens when TRG1 is input for camera 1?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="procedure-step",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=113,
        page_to=113,
        expected_terms=["trigger", "camera", "trg1", "measurement"],
        expected_snippet="TRG1 for CAM 1 executes capture and measurement processing.",
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "detail-atomic",
                "source_document_id": "doc-cvx",
                "expected_terms": ["inputting", "trigger", "signal", "having"],
                "snippet": (
                    "By inputting the trigger signal having the same number as the camera number "
                    "(TRG1 for CAM 1), capture and measurement processing is executed."
                ),
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "The timing chart describes trigger input for camera measurements.",
            "citations": [{"document_id": "doc-cvx", "chunk_id": "nearby-section", "quote_span": None}],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "nearby-section",
                "source_document_id": "doc-cvx",
                "content": "Timing chart overview for trigger input and camera measurement setup.",
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_rejects_composite_chunk_with_sibling_identifier_binding():
    case = RetrievalEvalCase(
        case_id="answer-composite-sibling-trigger",
        query="For asynchronous trigger input, what happens when TRG1 is input for camera 1?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="detail-atomic",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="atomic_text",
        section_path="Timing chart",
        page_from=113,
        page_to=113,
        expected_terms=["trigger", "camera", "trg1", "measurement"],
        expected_snippet="TRG1 for CAM 1 executes capture and measurement processing.",
        generation_method="contextual_procedure_plus_section_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "detail-atomic",
                "source_document_id": "doc-cvx",
                "expected_terms": ["inputting", "trigger", "signal", "having"],
                "snippet": (
                    "By inputting the trigger signal having the same number as the camera number "
                    "(TRG1 for CAM 1), capture and measurement processing is executed."
                ),
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "TRG1 for CAM 1 executes capture and measurement processing.",
            "citations": [{"document_id": "doc-cvx", "chunk_id": "sibling-section", "quote_span": None}],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "sibling-section",
                "source_document_id": "doc-cvx",
                "content": (
                    "By inputting the trigger signal having the same number as the camera number "
                    "(TRG2 for CAM 2), capture and measurement processing is executed."
                ),
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_rejects_composite_chunk_with_swapped_numeric_bindings():
    case = RetrievalEvalCase(
        case_id="answer-composite-swapped-quantity",
        query="What voltage and current should I set?",
        source_document_id="doc-controller",
        document_version_id="ver-controller",
        source_chunk_id="setup-values",
        source_title="Setup",
        source_filename="Manual.pdf",
        chunk_type="table_record",
        section_path="Setup",
        page_from=12,
        page_to=12,
        expected_terms=["voltage", "5", "current", "10"],
        expected_snippet="Set voltage to 5 volts and current to 10 amps.",
        generation_method="manual_curated_cross_document_v2_from_diagnostic_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "setup-values",
                "source_document_id": "doc-controller",
                "allow_equivalent_citation": True,
                "expected_terms": ["voltage", "5", "current", "10"],
                "snippet": "Set voltage to 5 volts and current to 10 amps.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Set voltage to 5 volts and current to 10 amps.",
            "citations": [{"document_id": "doc-controller", "chunk_id": "nearby-values", "quote_span": None}],
            "used_documents": [{"document_id": "doc-controller"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "nearby-values",
                "source_document_id": "doc-controller",
                "content": "Set voltage to 10 volts and current to 5 amps.",
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_accepts_composite_chunk_with_correct_numeric_bindings():
    case = RetrievalEvalCase(
        case_id="answer-composite-correct-quantity",
        query="What voltage and current should I set?",
        source_document_id="doc-controller",
        document_version_id="ver-controller",
        source_chunk_id="setup-values",
        source_title="Setup",
        source_filename="Manual.pdf",
        chunk_type="table_record",
        section_path="Setup",
        page_from=12,
        page_to=12,
        expected_terms=["voltage", "5", "current", "10"],
        expected_snippet="Set voltage to 5 volts and current to 10 amps.",
        generation_method="manual_curated_cross_document_v2_from_diagnostic_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "setup-values",
                "source_document_id": "doc-controller",
                "allow_equivalent_citation": True,
                "expected_terms": ["voltage", "5", "current", "10"],
                "snippet": "Set voltage to 5 volts and current to 10 amps.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Set voltage to 5 volts and current to 10 amps.",
            "citations": [{"document_id": "doc-controller", "chunk_id": "combined-values", "quote_span": None}],
            "used_documents": [{"document_id": "doc-controller"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "combined-values",
                "source_document_id": "doc-controller",
                "content": "Set voltage to 5 volts and current to 10 amps.",
            }
        ],
    )

    assert scored["passed"] is True
    assert scored["evidence_citation_support"]["passed"] is True


def test_answer_response_scoring_binds_number_of_lines_relation():
    case = RetrievalEvalCase(
        case_id="answer-composite-number-of-lines",
        query="What camera line settings are shown for line count and line scan interval?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="line-settings",
        source_title="Timing chart",
        source_filename="Manual.pdf",
        chunk_type="procedure_record",
        section_path="Timing chart",
        page_from=123,
        page_to=123,
        expected_terms=["camera", "lines", "interval", "specify"],
        expected_snippet="Camera settings Number of Lines 10 Line Scan Interval Specify Encoder 1 pulse/line Sampling mode",
        generation_method="unit_test",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "line-settings",
                "source_document_id": "doc-cvx",
                "allow_equivalent_citation": True,
                "expected_terms": ["camera", "lines", "interval", "specify"],
                "snippet": "Camera settings Number of Lines 10 Line Scan Interval Specify Encoder 1 pulse/line Sampling mode",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Number of Lines is 10, and Line Scan Interval is Specify Encoder 1 pulse/line.",
            "citations": [{"document_id": "doc-cvx", "chunk_id": "section-window", "quote_span": None}],
            "used_documents": [{"document_id": "doc-cvx"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "section-window",
                "source_document_id": "doc-cvx",
                "content": (
                    "When the LJ-V series head is used and Sheet-fed is set. "
                    "Camera settings Number of Lines 10 Line Scan Interval Specify Encoder 1 pulse/line Sampling mode."
                ),
            }
        ],
    )

    assert scored["passed"] is True
    assert scored["evidence_citation_support"]["passed"] is True


def test_answer_response_scoring_rejects_composite_chunk_with_polarity_inversion():
    case = RetrievalEvalCase(
        case_id="answer-composite-polarity",
        query="Should encryption be enabled before exporting logs?",
        source_document_id="doc-security",
        document_version_id="ver-security",
        source_chunk_id="encryption-row",
        source_title="Security",
        source_filename="Manual.pdf",
        chunk_type="table_record",
        section_path="Security",
        page_from=4,
        page_to=4,
        expected_terms=["enable", "encryption", "exporting", "logs"],
        expected_snippet="Enable encryption before exporting logs.",
        generation_method="manual_curated_cross_document_v2_from_diagnostic_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "encryption-row",
                "source_document_id": "doc-security",
                "expected_terms": ["enable", "encryption", "exporting", "logs"],
                "snippet": "Enable encryption before exporting logs.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Enable encryption before exporting logs.",
            "citations": [{"document_id": "doc-security", "chunk_id": "neighbor-security", "quote_span": None}],
            "used_documents": [{"document_id": "doc-security"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "neighbor-security",
                "source_document_id": "doc-security",
                "content": "Disable encryption before exporting logs.",
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_answer_response_scoring_rejects_composite_chunk_with_cross_role_quantity_mixing():
    case = RetrievalEvalCase(
        case_id="answer-composite-cross-role-mixing",
        query="What voltage and current are required?",
        source_document_id="doc-controller",
        document_version_id="ver-controller",
        source_chunk_id="required-values",
        source_title="Setup",
        source_filename="Manual.pdf",
        chunk_type="table_record",
        section_path="Setup",
        page_from=12,
        page_to=12,
        expected_terms=["voltage", "5", "current", "10"],
        expected_snippet="Voltage is 5 volts. Current is 10 amps.",
        generation_method="manual_curated_cross_document_v2_from_diagnostic_evidence",
        source_metadata={},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "required-values",
                "source_document_id": "doc-controller",
                "expected_terms": ["voltage", "5", "current", "10"],
                "snippet": "Voltage is 5 volts. Current is 10 amps.",
            },
        ],
    )

    scored = score_answer_response(
        case,
        {
            "answer": "Voltage is 5 volts and current is 10 amps.",
            "citations": [{"document_id": "doc-controller", "chunk_id": "mixed-values", "quote_span": None}],
            "used_documents": [{"document_id": "doc-controller"}],
            "insufficient_evidence": False,
        },
        {"passed": True},
        [
            {
                "chunk_id": "mixed-values",
                "source_document_id": "doc-controller",
                "content": "Voltage is 5 volts. Current is 5 amps. Backup current limit is 10 amps.",
            }
        ],
    )

    assert scored["passed"] is False
    assert "expected_evidence_not_cited" in scored["failure_reasons"]


def test_large_retrieval_eval_summarizes_answer_metrics():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module.summarize(
        [
            {
                "case": {
                    "chunk_type": "spec_record",
                    "retrieval_task": "single_step_retrieval",
                    "source_filename": "manual.pdf",
                    "benchmark_quality": "validated",
                },
                "evaluation": {"passed": True, "rank": 1, "candidate_recall": True},
                "answer": {
                    "answer": "24 VDC",
                    "_eval_trace": {
                        "used_fallback": True,
                        "answer_source": "fallback_validation",
                        "fallback_reason": "Generated answer was replaced by retrieval-grounded fallback during validation.",
                        "summary_count": 3,
                    },
                },
                "answer_evaluation": {
                    "passed": False,
                    "failure_reasons": ["expected_terms_missing"],
                    "elapsed_seconds": 12.5,
                },
            }
        ]
    )

    assert summary["answer_eval_count"] == 1
    assert summary["answer_pass_rate"] == 0.0
    assert summary["answer_failure_reasons"] == {"expected_terms_missing": 1}
    assert summary["answer_latency"] == {
        "min_seconds": 12.5,
        "max_seconds": 12.5,
        "mean_seconds": 12.5,
        "p95_seconds": 12.5,
    }
    assert summary["answer_fallback_count"] == 1
    assert summary["answer_fallback_rate"] == 1.0
    assert summary["answer_sources"] == {"fallback_validation": 1}
    assert summary["answer_fallback_reasons"] == {
        "Generated answer was replaced by retrieval-grounded fallback during validation.": 1
    }
    assert summary["answer_mean_summary_count"] == 3.0


def test_large_retrieval_eval_scores_api_answer_payload_for_http_answer_mode(monkeypatch):
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []
    monkeypatch.setattr(module, "run_search", lambda query, *, corpus_id, response_mode: [{"chunk_id": "hit"}])
    monkeypatch.setattr(
        module,
        "run_query_answer",
        lambda query, *, corpus_id: calls.append((query, corpus_id))
        or {"answer": "24 VDC", "citations": [], "used_documents": []},
    )
    monkeypatch.setattr(module, "generate_answer_payload", lambda query, results: pytest.fail("HTTP answer mode must not regenerate a local answer"))

    payload = module.run_case_search(
        "What voltage?",
        corpus_id="manuals",
        search_mode="http",
        response_mode="answer_with_citations",
    )

    assert payload == {
        "top_results": [{"chunk_id": "hit"}],
        "answer": {
            "answer": "24 VDC",
            "citations": [],
            "used_documents": [],
            "_eval_trace": {"answer_transport": "http_api", "answer_source": "api", "used_fallback": False},
        },
    }
    assert calls == [("What voltage?", "manuals")]


def test_large_retrieval_eval_marks_warning_payload_as_validation_fallback():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module.normalize_api_answer_payload(
        {
            "answer": "Grounded fallback text.",
            "warnings": [
                "Generated answer was not sufficiently supported by retrieved evidence; using retrieval-grounded fallback."
            ],
            "_eval_trace": {"answer_source": "api", "used_fallback": False},
        }
    )

    assert payload["_eval_trace"] == {
        "answer_source": "fallback_validation",
        "used_fallback": True,
        "answer_transport": "http_api",
        "fallback_reason": "Generated answer was replaced by retrieval-grounded fallback during validation.",
    }


def test_large_retrieval_eval_summary_counts_warning_payload_fallback():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    answer = module.normalize_api_answer_payload(
        {
            "answer": "Grounded fallback text.",
            "warnings": [
                "Generated answer was not sufficiently supported by retrieved evidence; using retrieval-grounded fallback."
            ],
            "_eval_trace": {"answer_source": "api", "used_fallback": False},
        }
    )
    summary = module.summarize(
        [
            {
                "case": {
                    "chunk_type": "spec_record",
                    "retrieval_task": "single_step_retrieval",
                    "source_filename": "manual.pdf",
                    "benchmark_quality": "validated",
                },
                "evaluation": {"passed": True, "rank": 1, "candidate_recall": True},
                "answer": answer,
                "answer_evaluation": {"passed": True, "failure_reasons": [], "elapsed_seconds": 1.0},
            }
        ]
    )

    assert summary["answer_fallback_count"] == 1
    assert summary["answer_fallback_rate"] == 1.0
    assert summary["answer_sources"] == {"fallback_validation": 1}
    assert summary["answer_fallback_reasons"] == {
        "Generated answer was replaced by retrieval-grounded fallback during validation.": 1
    }


def test_large_retrieval_eval_uses_embedded_http_answer_when_present(monkeypatch):
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "run_search",
        lambda query, *, corpus_id, response_mode: {
            "top_results": [{"chunk_id": "hit"}],
            "answer": {"answer": "Use shielded cable.", "_eval_trace": {"answer_source": "api_model"}},
        },
    )
    monkeypatch.setattr(module, "run_query_answer", lambda query, *, corpus_id: pytest.fail("Embedded API answer should be scored directly"))

    payload = module.run_case_search(
        "What cable?",
        corpus_id="manuals",
        search_mode="http",
        response_mode="answer_with_citations",
    )

    assert payload["answer"] == {
        "answer": "Use shielded cable.",
        "_eval_trace": {"answer_source": "api_model", "answer_transport": "http_api", "used_fallback": False},
    }


def test_large_retrieval_eval_scores_answer_against_current_search_results(monkeypatch):
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_large_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_large_retrieval_eval", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    case = RetrievalEvalCase(
        case_id="case-1",
        query="What should I change?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=1,
        page_to=1,
        expected_terms=["change", "trigger"],
        expected_snippet="Cell value: Change the trigger signal.",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "MODEL-1"},
        retrieval_task="multi_step_retrieval",
        expected_evidence=[
            {
                "chunk_id": "chunk-1",
                "source_document_id": "doc-1",
                "field": "corrective action",
                "expected_terms": ["change", "trigger"],
                "snippet": "Cell value: Change the trigger signal.",
            }
        ],
    )
    search_payload = {
        "top_results": [
            {
                "chunk_id": "chunk-1",
                "source_document_id": "doc-1",
                "content": "Cell value: Change the trigger signal.",
            }
        ]
    }
    answer = {
        "answer": "Change the trigger signal.",
        "citations": [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "quote_span": "Change the trigger signal.",
            }
        ],
        "used_documents": [],
        "insufficient_evidence": False,
    }
    calls = []

    def _score_answer_response(eval_case, answer_payload, evaluation, retrieved_results, **kwargs):
        calls.append(retrieved_results)
        return {"passed": True}

    monkeypatch.setattr(module, "score_answer_response", _score_answer_response)

    scored = module.score_current_answer_response(case, answer, {"passed": True}, search_payload)

    assert scored == {"passed": True}
    assert calls == [search_payload["top_results"]]


def test_build_eval_cases_from_chunks_creates_queries():
    chunks = [
        {
            "id": "chunk-1",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "datasheet_record",
            "title": "CA-EN100U Datasheet",
            "source_filename": "CA-EN100U_Datasheet.pdf",
            "section_path_text": "Specifications",
            "page_from": 1,
            "page_to": 1,
            "content": "Power supply voltage: 24 VDC for standard operation.",
            "metadata_json": {"product_model": "CA-EN100U"},
            "product_model": "CA-EN100U",
        }
    ]
    cases = build_eval_cases_from_chunks(chunks, max_cases=3, use_llm_generation=False)
    assert cases
    assert all(case.benchmark_quality == "validated" for case in cases)
    assert all("this document" not in case.query.lower() for case in cases)
    assert all(case.query.endswith("?") for case in cases)
    assert any(
        "ca-en100u" in case.query.lower()
        and ("power" in case.query.lower() or "voltage" in case.query.lower())
        for case in cases
    )
    assert all(case.retrieval_task == "single_step_retrieval" for case in cases)


def test_table_eval_queries_use_row_column_and_cell_context():
    chunks = [
        {
            "id": "chunk-table-specific",
            "source_document_id": "doc-lj",
            "document_version_id": "ver-lj",
            "chunk_type": "table_record",
            "title": "LJ-X8000",
            "source_filename": "LJ-X8000.pdf",
            "section_path_text": "Specifications",
            "page_from": 36,
            "page_to": 36,
            "content": (
                "Column headers: LJ-X8200; Row headers: Measurement range > X-axis (width) > "
                "Reference distance; Cell value: 72 mm 2.83\"; Row: 4; Column: 6"
            ),
            "metadata_json": {
                "product_model": "LJ-X8000",
                "table_column_headers": ["LJ-X8200"],
                "table_row_headers": ["Measurement range", "X-axis (width)", "Reference distance"],
            },
            "product_model": "LJ-X8000",
        }
    ]

    cases = build_eval_cases_from_chunks(chunks, max_cases=3, use_llm_generation=False)
    queries = [case.query.lower() for case in cases]

    assert cases
    assert all("measurement range" in query for query in queries)
    assert any("lj-x8200" in query for query in queries)
    assert all(query not in {"lj-x8000 column", "column headers lj-x8000"} for query in queries)
    assert any("2.83" in case.expected_terms or "2.83" in (case.anchor_terms or []) for case in cases)


def test_eval_queries_avoid_unwieldy_product_list_labels():
    chunk = {
        "id": "chunk-long-model-list",
        "source_document_id": "doc-vs",
        "document_version_id": "ver-vs",
        "chunk_type": "table_record",
        "title": "VS Manual",
        "source_filename": "VS.pdf",
        "section_path_text": "Capture Settings",
        "page_from": 1,
        "page_to": 1,
        "content": "Column headers: Value Upper Limit; Row headers: Capture Settings; Cell value: 88ms; Row: 13; Column: 6",
        "metadata_json": {
            "product_model": "VS-L160MX/VS-L160CX/VS-L320MX/VS-L320CX/VS-L500MX/VS-L500CX",
            "table_column_headers": ["Value Upper Limit"],
            "table_row_headers": ["Capture Settings"],
        },
        "product_model": "VS-L160MX/VS-L160CX/VS-L320MX/VS-L320CX/VS-L500MX/VS-L500CX",
    }

    cases = build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False)

    assert cases
    assert all("vs-l160mx/" not in case.query.lower() for case in cases)
    assert any("capture settings" in case.query.lower() for case in cases)
    assert any("88ms" in case.expected_terms or "88ms" in (case.anchor_terms or []) for case in cases)


def test_eval_queries_fall_back_to_product_family_for_long_model_lists():
    chunk = {
        "id": "chunk-vs-family",
        "source_document_id": "doc-vs",
        "document_version_id": "ver-vs",
        "chunk_type": "table_record",
        "title": "VS Manual",
        "source_filename": "VS.pdf",
        "section_path_text": "Capture Settings",
        "page_from": 1,
        "page_to": 1,
        "content": (
            "Column headers: Number Format > Decimal Digits; "
            "Row headers: LumiTrax Capture Settings > Track Moving Object: Pattern Region: Height; "
            "Cell value: 0; Row: 12; Column: 9"
        ),
        "metadata_json": {
            "product_model": "VS-L160MX/VS-L160CX/VS-L320MX/VS-L320CX/VS-L500MX/VS-L500CX",
            "product_family": "VS Series Vision System",
            "table_column_headers": ["Number Format", "Decimal Digits"],
            "table_row_headers": ["LumiTrax Capture Settings", "Track Moving Object: Pattern Region: Height"],
        },
        "product_model": "VS-L160MX/VS-L160CX/VS-L320MX/VS-L320CX/VS-L500MX/VS-L500CX",
    }

    cases = build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False)

    assert cases
    assert all("vs-l160mx/" not in case.query.lower() for case in cases)
    assert all("vs series vision system" in case.query.lower() for case in cases)


def test_table_key_value_queries_include_disambiguating_adjacent_value():
    chunk = {
        "id": "chunk-key-value-table",
        "source_document_id": "doc-iv4",
        "document_version_id": "ver-iv4",
        "chunk_type": "table_record",
        "title": "IV4 Manual",
        "source_filename": "IV4.pdf",
        "section_path_text": "Data Allocation",
        "page_from": 1,
        "page_to": 1,
        "content": "Address: 6 to 7 (WORD); Stored data: 1041",
        "metadata_json": {
            "product_model": "IV4-G120",
            "table_key_value": True,
        },
        "product_model": "IV4-G120",
    }

    cases = build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False)
    queries = [case.query.lower() for case in cases]

    assert cases
    assert all("1041" in query for query in queries)
    assert all("address" in query for query in queries)


def test_table_cell_without_row_context_is_not_single_step_queryworthy():
    chunk = {
        "id": "chunk-answer-only-cell",
        "source_document_id": "doc-xgx",
        "document_version_id": "ver-xgx",
        "chunk_type": "table_record",
        "title": "XG-X Manual",
        "source_filename": "XGX.pdf",
        "section_path_text": "Error Messages",
        "page_from": 1,
        "page_to": 1,
        "content": "Column headers: Error Message; Cell value: Library conversion was interrupted.; Row: 13; Column: 0",
        "metadata_json": {
            "product_family": "XG-X Series",
            "table_cell": True,
            "table_column_headers": ["Error Message"],
            "table_row_headers": [],
        },
    }

    assert not chunk_is_queryworthy(chunk, ["library", "conversion", "interrupted", "error"])


def test_placeholder_table_cells_are_not_single_step_queryworthy():
    chunk = {
        "id": "placeholder-cell",
        "source_document_id": "doc-ljs",
        "document_version_id": "ver-ljs",
        "chunk_type": "table_record",
        "title": "LJ-S Manual",
        "source_filename": "ljs.pdf",
        "section_path_text": "Label Specification",
        "page_from": 7,
        "page_to": 7,
        "content": "Column headers: Label Specification; Row headers: CC2D_LEN > Length of Data Read; Cell value: -; Row: 7; Column: 5",
        "metadata_json": {
            "product_family": "LJ-S8000 Series",
            "table_cell": True,
            "table_column_headers": ["Label Specification"],
            "table_row_headers": ["CC2D_LEN", "Length of Data Read"],
        },
    }

    assert not chunk_is_queryworthy(chunk, ["cc2d", "length", "data", "read"])
    assert build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False) == []


def test_icon_only_table_cells_are_not_single_step_queryworthy():
    chunk = {
        "id": "icon-cell",
        "source_document_id": "doc-ljx",
        "document_version_id": "ver-ljx",
        "chunk_type": "table_record",
        "title": "LJ-X Manual",
        "source_filename": "ljx.pdf",
        "section_path_text": "Label Specification",
        "page_from": 8,
        "page_to": 8,
        "content": "Column headers: Label Specification Item ID; Row headers: PMH[] > Peak-to-Peak Height; Cell value: \uf0a1; Row: 25; Column: 6",
        "metadata_json": {
            "product_family": "LJ-X8000 Series",
            "table_cell": True,
            "table_column_headers": ["Label Specification Item ID"],
            "table_row_headers": ["PMH[]", "Peak-to-Peak Height"],
        },
    }

    assert not chunk_is_queryworthy(chunk, ["pmh", "peak-to-peak", "height"])
    assert build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False) == []


def test_cross_reference_only_atomic_text_is_not_single_step_queryworthy():
    chunk = {
        "id": "cross-reference-only",
        "source_document_id": "doc-xgx",
        "document_version_id": "ver-xgx",
        "chunk_type": "atomic_text",
        "title": "XG-X Manual",
        "source_filename": "xgx.pdf",
        "section_path_text": "Representation Format",
        "page_from": 9,
        "page_to": 9,
        "content": "For more information about the calculated result data, refer to the XG-X Series Communications Control Manual.",
        "metadata_json": {"product_family": "XG-X Series"},
    }

    assert not chunk_is_queryworthy(chunk, ["calculated", "result", "data", "xg-x"])
    assert build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False) == []


def test_page_reference_fragments_are_not_single_step_queryworthy():
    chunk = {
        "id": "page-reference-fragment",
        "source_document_id": "doc-xgx",
        "document_version_id": "ver-xgx",
        "chunk_type": "atomic_text",
        "title": "XG-X Manual",
        "source_filename": "xgx.pdf",
        "section_path_text": "3D Blob",
        "page_from": 2,
        "page_to": 2,
        "content": "[Multi-point] (Page 2-497) region, with which regions of the same shape and size can be added to multiple points, is supported in 3D Blob only.",
        "metadata_json": {"product_family": "XG-X Series"},
    }

    assert not chunk_is_queryworthy(chunk, ["multi-point", "2-497", "region", "regions"])
    assert build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False) == []


def test_build_multi_step_eval_cases_from_sibling_error_rows():
    base = {
        "source_document_id": "doc-xgx",
        "document_version_id": "ver-xgx",
        "chunk_type": "table_record",
        "title": "XG-X",
        "source_filename": "xgx.pdf",
        "section_path_text": "Troubleshooting",
        "page_from": 10,
        "page_to": 10,
        "product_model": "",
    }
    chunks = [
        {
            **base,
            "id": "error-cell",
            "content": "Column headers: Error Message; Cell value: Timeout error has occurred while waiting for encoder input.; Row: 10; Column: 0",
            "metadata_json": {
                "product_family": "XG-X Series",
                "table_cell": True,
                "table_row": 10,
                "table_column": 0,
                "table_column_headers": ["Error Message"],
            },
        },
        {
            **base,
            "id": "cause-cell",
            "content": "Column headers: Cause; Row headers: Timeout error has occurred while waiting for encoder input.; Cell value: Encoder input timeout was detected.; Row: 10; Column: 1",
            "metadata_json": {
                "product_family": "XG-X Series",
                "table_cell": True,
                "table_row": 10,
                "table_column": 1,
                "table_column_headers": ["Cause"],
                "table_row_headers": ["Timeout error has occurred while waiting for encoder input."],
            },
        },
        {
            **base,
            "id": "action-cell",
            "content": "Column headers: Corrective Action; Row headers: Timeout error has occurred while waiting for encoder input. > Encoder input timeout was detected.; Cell value: Check the encoder connection and change the Detect Timeout time.; Row: 10; Column: 2",
            "metadata_json": {
                "product_family": "XG-X Series",
                "table_cell": True,
                "table_row": 10,
                "table_column": 2,
                "table_column_headers": ["Corrective Action"],
                "table_row_headers": [
                    "Timeout error has occurred while waiting for encoder input.",
                    "Encoder input timeout was detected.",
                ],
            },
        },
    ]

    cases = build_multi_step_eval_cases_from_chunks(chunks, max_cases=5)

    assert len(cases) == 1
    assert cases[0].retrieval_task == "multi_step_retrieval"
    assert "what causes timeout error" in cases[0].query.lower()
    assert cases[0].expected_source_chunk_ids == ["error-cell", "cause-cell", "action-cell"]
    assert len(cases[0].expected_evidence or []) == 3


def test_build_multi_step_eval_cases_from_contextual_procedure_section():
    base = {
        "source_document_id": "doc-cvx",
        "document_version_id": "ver-cvx",
        "title": "CV-X",
        "source_filename": "cvx.pdf",
        "section_path_text": "PLC-Link Ethernet setup",
        "page_from": 48,
        "page_to": 48,
        "product_model": "CV-X482",
    }
    chunks = [
        {
            **base,
            "id": "procedure-cell",
            "chunk_type": "procedure_record",
            "chunk_level": 1,
            "content": "Procedure step 3: 3. Connect the PLC-Link Ethernet cable and set the link unit.",
            "metadata_json": {
                "product_family": "CV-X Series",
                "procedure_flag": True,
                "local_rerank_context": "Connect the PLC-Link Ethernet cable and set the link unit. Port 9010 is reserved.",
            },
        },
        {
            **base,
            "id": "constraint-cell",
            "chunk_type": "atomic_text",
            "chunk_level": 1,
            "content": "The port number for communication port settings, '9010', cannot be used because it is reserved for the controller.",
            "metadata_json": {
                "product_family": "CV-X Series",
                "local_rerank_context": "Procedure step 3: Connect the PLC-Link Ethernet cable and set the link unit. Port 9010 is reserved.",
            },
        },
    ]

    cases = build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="contextual_section")

    assert len(cases) == 1
    assert cases[0].retrieval_task == "multi_step_retrieval"
    assert cases[0].generation_method == "contextual_procedure_plus_section_evidence"
    assert cases[0].query == (
        "For CV-X482, when you need to connect the PLC-Link Ethernet cable and set the link unit, "
        "what should be checked about port number for communication port settings, '9010', cannot be used?"
    )
    assert "9010" in cases[0].query
    assert cases[0].expected_source_chunk_ids == ["procedure-cell", "constraint-cell"]


def test_contextual_procedure_questions_do_not_duplicate_when_prefix():
    base = {
        "source_document_id": "doc-xgx",
        "document_version_id": "ver-xgx",
        "title": "XG-X",
        "source_filename": "xgx.pdf",
        "section_path_text": "Asynchronous capture",
        "page_from": 15,
        "page_to": 15,
        "product_family": "XG-X Series",
    }
    chunks = [
        {
            **base,
            "id": "procedure-cell",
            "chunk_type": "procedure_record",
            "chunk_level": 1,
            "content": "Procedure step 2: 2. When multiple capture units are used.",
            "metadata_json": {
                "product_family": "XG-X Series",
                "procedure_flag": True,
                "local_rerank_context": "When multiple capture units are used, branch by the passing status.",
            },
        },
        {
            **base,
            "id": "support-cell",
            "chunk_type": "atomic_text",
            "chunk_level": 1,
            "content": "On a flowchart branched by the passing status of the first capture unit, image capture is processed independently.",
            "metadata_json": {
                "product_family": "XG-X Series",
                "local_rerank_context": "When multiple capture units are used, branch by the passing status.",
            },
        },
    ]

    cases = build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="contextual_section")

    assert len(cases) == 1
    assert "when when" not in cases[0].query.lower()
    assert "what related" not in cases[0].query.lower()
    assert cases[0].query == (
        "For XG-X Series, when multiple capture units are used, "
        "what should be checked about flowchart branched by the passing status of the first capture?"
    )


def test_build_multi_step_eval_cases_from_warning_step_neighborhood():
    base = {
        "source_document_id": "doc-iv",
        "document_version_id": "ver-iv",
        "title": "IV4",
        "source_filename": "iv4.pdf",
        "product_model": "IV4-G600CA",
    }
    chunks = [
        {
            **base,
            "id": "procedure-cell",
            "chunk_type": "procedure_record",
            "chunk_level": 1,
            "section_path_text": "Connection check",
            "page_from": 12,
            "page_to": 12,
            "content": "Procedure step 2: Connect the EtherNet/IP cable and check the controller connection status.",
            "metadata_json": {
                "product_family": "IV4 Series",
                "procedure_flag": True,
                "local_rerank_context": "Before connecting the EtherNet/IP cable, turn off power and check controller wiring.",
            },
        },
        {
            **base,
            "id": "warning-cell",
            "chunk_type": "warning_record",
            "chunk_level": 1,
            "section_path_text": "Wiring caution",
            "page_from": 11,
            "page_to": 11,
            "content": "Caution: Turn off the power before connecting or disconnecting the EtherNet/IP cable.",
            "metadata_json": {
                "product_family": "IV4 Series",
                "local_rerank_context": "Connect the EtherNet/IP cable only after turning off power.",
            },
        },
    ]

    cases = build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="warning_step")

    assert len(cases) == 1
    assert cases[0].retrieval_task == "multi_step_retrieval"
    assert cases[0].generation_method == "warning_plus_step_evidence"
    assert "warning or caution" in cases[0].query.lower()
    assert cases[0].query.startswith("What warning or caution")
    assert "applies when connect the ether" in cases[0].query.lower()
    assert cases[0].expected_source_chunk_ids == ["procedure-cell", "warning-cell"]


def test_warning_step_cases_skip_generic_manual_labels_and_prohibitions():
    base = {
        "source_document_id": "doc-lj",
        "document_version_id": "ver-lj",
        "title": "LJ-X",
        "source_filename": "lj-x.pdf",
    }
    chunks = [
        {
            **base,
            "id": "prohibition",
            "chunk_type": "atomic_text",
            "chunk_level": 1,
            "section_path_text": "Mounting",
            "page_from": 12,
            "page_to": 12,
            "content": "Do not install the controller in a location with lots of dust or water vapor.",
            "metadata_json": {
                "product_model": "User's Manual (3D mode)",
                "product_family": "LJ: X8000 Series",
                "local_rerank_context": "Controller mounting cautions for LJ-X8000.",
            },
        },
        {
            **base,
            "id": "step",
            "chunk_type": "atomic_text",
            "chunk_level": 1,
            "section_path_text": "Mounting",
            "page_from": 12,
            "page_to": 12,
            "content": "Install the controller to the DIN rail, or use the holes on the bottom to secure it with screws.",
            "metadata_json": {
                "product_model": "User's Manual (3D mode)",
                "product_family": "LJ: X8000 Series",
                "local_rerank_context": "Controller mounting cautions for LJ-X8000.",
            },
        },
        {
            **base,
            "id": "warning",
            "chunk_type": "warning_record",
            "chunk_level": 1,
            "section_path_text": "Mounting",
            "page_from": 12,
            "page_to": 12,
            "content": "Caution: Caution on direction of controller mounting",
            "metadata_json": {
                "product_model": "User's Manual (3D mode)",
                "product_family": "LJ: X8000 Series",
                "local_rerank_context": "Controller mounting cautions for LJ-X8000.",
            },
        },
    ]

    cases = build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="warning_step")

    assert len(cases) == 1
    assert cases[0].expected_source_chunk_ids == ["step", "warning"]
    assert "User's Manual" not in cases[0].query
    assert "LJ: X8000 Series" in cases[0].query
    assert "When Do not" not in cases[0].query


def test_build_multi_step_eval_cases_from_cross_document_same_field_values():
    left_base = {
        "source_document_id": "doc-iv",
        "document_version_id": "ver-iv",
        "title": "IV4",
        "source_filename": "iv4.pdf",
        "section_path_text": "Electrical specifications",
        "page_from": 7,
        "page_to": 7,
        "product_model": "IV4-G120",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "IV4-G120",
            "table_cell": True,
            "table_column_headers": ["Power supply voltage"],
            "table_row_headers": ["Controller"],
        },
    }
    right_base = {
        "source_document_id": "doc-lj",
        "document_version_id": "ver-lj",
        "title": "LJ-X8000",
        "source_filename": "ljx.pdf",
        "section_path_text": "Electrical specifications",
        "page_from": 8,
        "page_to": 8,
        "product_model": "LJ-X8000",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "LJ-X8000",
            "table_cell": True,
            "table_column_headers": ["Power supply voltage"],
            "table_row_headers": ["Controller"],
        },
    }
    chunks = [
        {
            **left_base,
            "id": "iv-voltage",
            "chunk_type": "table_record",
            "content": "Row headers: Controller; Column headers: Power supply voltage; Cell value: 24 VDC",
        },
        {
            **right_base,
            "id": "lj-voltage",
            "chunk_type": "table_record",
            "content": "Row headers: Controller; Column headers: Power supply voltage; Cell value: 24 VDC +/-10%",
        },
    ]

    cases = build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="cross_document")

    assert len(cases) == 1
    assert cases[0].retrieval_task == "multi_step_retrieval"
    assert cases[0].generation_method == "cross_document_same_field_evidence"
    assert "power supply voltage entries" in cases[0].query.lower()
    assert "controller" in cases[0].query.lower()
    assert "IV4-G120" in cases[0].query
    assert "LJ-X8000" in cases[0].query
    assert cases[0].expected_source_chunk_ids == ["iv-voltage", "lj-voltage"]
    assert cases[0].expected_evidence[0]["source_document_id"] == "doc-iv"
    assert cases[0].expected_evidence[1]["source_document_id"] == "doc-lj"
    assert "iv4g120" in cases[0].expected_evidence[0]["product_identifiers"]


def test_build_cross_document_cases_skip_generic_item_fields():
    left_base = {
        "source_document_id": "doc-left",
        "document_version_id": "ver-left",
        "title": "Left",
        "source_filename": "left.pdf",
        "section_path_text": "Settings",
        "page_from": 7,
        "page_to": 7,
        "product_model": "MODEL-A",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-A",
            "table_cell": True,
            "table_column_headers": ["Item"],
            "table_row_headers": ["Mode"],
        },
    }
    right_base = {
        "source_document_id": "doc-right",
        "document_version_id": "ver-right",
        "title": "Right",
        "source_filename": "right.pdf",
        "section_path_text": "Settings",
        "page_from": 8,
        "page_to": 8,
        "product_model": "MODEL-B",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-B",
            "table_cell": True,
            "table_column_headers": ["Items"],
            "table_row_headers": ["Mode"],
        },
    }
    chunks = [
        {**left_base, "id": "left-item", "chunk_type": "table_record", "content": "Column headers: Item; Row headers: Mode; Cell value: Operation mode"},
        {**right_base, "id": "right-item", "chunk_type": "table_record", "content": "Column headers: Items; Row headers: Mode; Cell value: Batch mode"},
    ]

    assert build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="cross_document") == []


def test_build_cross_document_cases_reject_parser_artifact_subjects():
    left_base = {
        "source_document_id": "doc-left",
        "document_version_id": "ver-left",
        "title": "Left",
        "source_filename": "left.pdf",
        "section_path_text": "Communication",
        "page_from": 7,
        "page_to": 7,
        "product_model": "MODEL-A",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-A",
            "table_cell": True,
            "table_column_headers": ["Form of measured data"],
            "table_row_headers": ["PMSR[]. \uf020 DC4WPOSM[]"],
        },
    }
    right_base = {
        "source_document_id": "doc-right",
        "document_version_id": "ver-right",
        "title": "Right",
        "source_filename": "right.pdf",
        "section_path_text": "Communication",
        "page_from": 8,
        "page_to": 8,
        "product_model": "MODEL-B",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-B",
            "table_cell": True,
            "table_column_headers": ["Form of measured data"],
            "table_row_headers": ["Controller"],
        },
    }
    chunks = [
        {
            **left_base,
            "id": "left-artifact",
            "chunk_type": "table_record",
            "content": "Column headers: Form of measured data; Row headers: PMSR[]. \uf020 DC4WPOSM[]; Cell value: Sign, Integer 5 digits and decimal 3 digits",
        },
        {
            **right_base,
            "id": "right-clean",
            "chunk_type": "table_record",
            "content": "Column headers: Form of measured data; Row headers: Controller; Cell value: Integer 3 digits, 3 digits after the decimal point",
        },
    ]

    assert build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="cross_document") == []


def test_build_cross_document_cases_reject_page_reference_subjects():
    left_base = {
        "source_document_id": "doc-left",
        "document_version_id": "ver-left",
        "title": "Left",
        "source_filename": "left.pdf",
        "section_path_text": "Settings",
        "page_from": 7,
        "page_to": 7,
        "product_model": "MODEL-A",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-A",
            "table_cell": True,
            "table_column_headers": ["Item ID"],
            "table_row_headers": ["Color grouping (page 2-355) color sorting"],
        },
    }
    right_base = {
        "source_document_id": "doc-right",
        "document_version_id": "ver-right",
        "title": "Right",
        "source_filename": "right.pdf",
        "section_path_text": "Settings",
        "page_from": 8,
        "page_to": 8,
        "product_model": "MODEL-B",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-B",
            "table_cell": True,
            "table_column_headers": ["Item ID"],
            "table_row_headers": ["Pattern search (page 5-85) with or without shading"],
        },
    }
    chunks = [
        {**left_base, "id": "left-page-ref", "chunk_type": "table_record", "content": "Column headers: Item ID; Row headers: Color grouping (page 2-355) color sorting; Cell value: 9553"},
        {**right_base, "id": "right-page-ref", "chunk_type": "table_record", "content": "Column headers: Item ID; Row headers: Pattern search (page 5-85) with or without shading; Cell value: 9554"},
    ]

    assert build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="cross_document") == []


def test_build_cross_document_cases_reject_truncated_setting_subjects():
    left_base = {
        "source_document_id": "doc-left",
        "document_version_id": "ver-left",
        "title": "Left",
        "source_filename": "left.pdf",
        "section_path_text": "Settings",
        "page_from": 7,
        "page_to": 7,
        "product_model": "MODEL-A",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-A",
            "table_key_value": True,
        },
    }
    right_base = {
        "source_document_id": "doc-right",
        "document_version_id": "ver-right",
        "title": "Right",
        "source_filename": "right.pdf",
        "section_path_text": "Settings",
        "page_from": 8,
        "page_to": 8,
        "product_model": "MODEL-B",
        "chunk_level": 1,
        "metadata_json": {
            "product_model": "MODEL-B",
            "table_key_value": True,
        },
    }
    chunks = [
        {
            **left_base,
            "id": "left-truncated",
            "chunk_type": "table_record",
            "content": "Setting item: Condition list; Settings: A maximum of 16 reference conditions can be set. By setting multiple reference c",
        },
        {
            **right_base,
            "id": "right-truncated",
            "chunk_type": "table_record",
            "content": "Setting item: Reference Height; Settings: Specify the reference method of the height data to use in detecting the plane. Average detects the plane wit",
        },
    ]

    assert build_multi_step_eval_cases_from_chunks(chunks, max_cases=5, case_family="cross_document") == []


def test_build_eval_cases_skips_low_signal_atomic_queries():
    chunks = [
        {
            "id": "chunk-2",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "atomic_text",
            "title": "Manual",
            "source_filename": "Manual.pdf",
            "section_path_text": "Document",
            "page_from": 2,
            "page_to": 2,
            "content": "Click the button to use the displayed value.",
            "metadata_json": {"product_model": "MODEL-1"},
            "product_model": "MODEL-1",
        }
    ]
    assert build_eval_cases_from_chunks(chunks, max_cases=5, use_llm_generation=False) == []


def test_build_eval_cases_skips_legal_boilerplate_chunks():
    chunks = [
        {
            "id": "chunk-legal",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "atomic_text",
            "title": "Manual",
            "source_filename": "Manual.pdf",
            "section_path_text": "Legal",
            "page_from": 2,
            "page_to": 2,
            "content": (
                "OTHER THAN AS STATED HEREIN, THE PRODUCTS/SAMPLES ARE PROVIDED WITH NO OTHER "
                "WARRANTIES WHATSOEVER. ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING "
                "MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE DISCLAIMED."
            ),
            "metadata_json": {"product_model": "MODEL-1"},
            "product_model": "MODEL-1",
        }
    ]
    assert build_eval_cases_from_chunks(chunks, max_cases=5, use_llm_generation=False) == []


def test_build_eval_cases_skips_toc_style_procedure_chunks():
    chunks = [
        {
            "id": "chunk-toc",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "procedure_record",
            "title": "Manual",
            "source_filename": "Manual.pdf",
            "section_path_text": "Contents",
            "page_from": 9,
            "page_to": 9,
            "content": "Procedure step 2: 2. Wiring for the PLC-Link and setting the PLC side (Ethernet). . . . . . . . .9-32",
            "metadata_json": {"product_model": "MODEL-1"},
            "product_model": "MODEL-1",
        }
    ]
    assert build_eval_cases_from_chunks(chunks, max_cases=5, use_llm_generation=False) == []


def test_validate_eval_case_rejects_mechanical_source_dump_query():
    chunk = {
        "id": "chunk-source-dump",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "table_record",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Status table",
        "page_from": 2,
        "page_to": 2,
        "content": (
            "Column headers: Description of value; Row headers: PMSR[].PCNT[] PMSR[].MXPCNT "
            "PMSR[].MNPCNT PMSR[].AVPCNT PMSR[].DVPCNT; Cell value: Measurement values"
        ),
        "metadata_json": {
            "product_family": "LJ: S8000 Series",
            "table_cell": True,
            "table_row_headers": ["PMSR[].PCNT[] PMSR[].MXPCNT PMSR[].MNPCNT PMSR[].AVPCNT PMSR[].DVPCNT"],
            "table_column_headers": ["Description of value"],
        },
    }

    valid, reason = validate_eval_case(
        (
            "What PMSR[].PCNT[] PMSR[].MXPCNT PMSR[].MNPCNT PMSR[].AVPCNT PMSR[].DVPCNT "
            "Description of value applies to LJ: S8000 Series?"
        ),
        chunk,
        ["pmsr", "pcnt", "measurement", "values"],
    )

    assert not valid
    assert reason == "mechanical_query"


def test_validate_eval_case_rejects_toc_and_file_list_questions():
    chunk = {
        "id": "chunk-mechanical",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "table_record",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Status table",
        "page_from": 2,
        "page_to": 2,
        "content": "Column headers: Contained Data; Row headers: Output file name; Cell value: YYMMDD_HHMMSS.bmp / Sequential No._Specified string.jpg",
        "metadata_json": {
            "product_model": "MODEL-1",
            "table_cell": True,
            "table_row_headers": ["Output file name"],
            "table_column_headers": ["Contained Data"],
        },
    }

    toc_valid, toc_reason = validate_eval_case(
        "What 9-44 MCR Read Measured Value Correction . . . . . .9-16 value applies to MODEL-1?",
        chunk,
        ["mcr", "measured", "value", "correction"],
    )
    file_valid, file_reason = validate_eval_case(
        "What YYMMDD_HHMMSS.bmp and Sequential No._Specified string.jpg contained data applies to MODEL-1?",
        chunk,
        ["yymmdd", "hhmmss", "specified", "string"],
    )

    assert not toc_valid
    assert toc_reason == "mechanical_query"
    assert not file_valid
    assert file_reason == "mechanical_query"


def test_validate_eval_case_rejects_generic_short_item_queries():
    chunk = {
        "id": "chunk-items",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "table_record",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Settings",
        "page_from": 2,
        "page_to": 2,
        "content": "Column headers: Items for Setting contents; Row headers: MODEL-1; Cell value: 100 ms",
        "metadata_json": {
            "product_model": "MODEL-1",
            "table_cell": True,
            "table_row_headers": ["MODEL-1"],
            "table_column_headers": ["Items for Setting contents"],
        },
    }

    valid, reason = validate_eval_case(
        "For MODEL-1, what Items for Setting contents 100?",
        chunk,
        ["items", "setting", "contents", "100"],
    )

    assert not valid
    assert reason == "mechanical_query"


def test_validate_eval_case_rejects_source_shaped_table_coordinate_queries():
    chunk = {
        "id": "chunk-table-coordinate",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "table_record",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Command table",
        "page_from": 2,
        "page_to": 2,
        "content": "Column headers: 6bit; Row headers: 0028 65.0 > Command output area; Cell value: Command Result",
        "metadata_json": {
            "product_model": "CV-X482",
            "table_cell": True,
            "table_row_headers": ["0028 65.0", "Command output area"],
            "table_column_headers": ["6bit"],
        },
    }

    valid, reason = validate_eval_case(
        "What command 0028 65.0 Command output area 6bit value applies to CV-X482?",
        chunk,
        ["command", "result", "0028", "65.0"],
    )

    assert not valid
    assert reason == "mechanical_query"


def test_validate_eval_case_rejects_toc_like_described_queries():
    chunk = {
        "id": "chunk-contents",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "atomic_text",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Contents",
        "page_from": 16,
        "page_to": 16,
        "content": "Contents Output Assembly Address 6 to 11: FTP/SD Save File Name.",
        "metadata_json": {"product_model": "IV4-G120"},
    }

    valid, reason = validate_eval_case(
        "What contents is described for IV4-G120?",
        chunk,
        ["contents", "assembly", "address", "ftp/sd"],
    )

    assert not valid
    assert reason == "mechanical_query"


def test_validate_eval_case_rejects_generic_applies_phrasing():
    chunk = {
        "id": "chunk-generic-applies",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "table_record",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Number format",
        "page_from": 2,
        "page_to": 2,
        "content": "Column headers: Number Format > Integer Digits; Row headers: Fail Color: Red; Cell value: 3",
        "metadata_json": {
            "product_family": "VS Series",
            "table_cell": True,
            "table_row_headers": ["Fail Color: Red"],
            "table_column_headers": ["Number Format", "Integer Digits"],
        },
    }

    valid, reason = validate_eval_case(
        "What format integer applies to VS Series?",
        chunk,
        ["inspection", "region", "color", "input.graphic.region.mask.colorfail.red"],
    )

    assert not valid
    assert reason == "mechanical_query"


def test_numbered_click_step_fragments_are_not_single_step_queryworthy():
    chunk = {
        "id": "chunk-numbered-click",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "atomic_text",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Command setup",
        "page_from": 2,
        "page_to": 2,
        "content": "10After completing the setting, left-click [OK]. 11Restart the controller.",
        "metadata_json": {"product_model": "CV-X482"},
    }

    assert not chunk_is_queryworthy(chunk, ["10after", "completing", "left-click", "11restart"])


def test_table_fallback_queries_are_user_style_not_table_coordinate_dumps():
    chunk = {
        "id": "chunk-table",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "table_record",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Output symbols",
        "page_from": 2,
        "page_to": 2,
        "content": "Column headers: Settings; Row headers: Output Symbol Identifier; Cell value: When enabled, a symbol identifier is added.",
        "metadata_json": {
            "product_family": "LJ-S8000 Series",
            "table_cell": True,
            "table_row_headers": ["Output Symbol Identifier"],
            "table_column_headers": ["Settings"],
        },
    }

    cases = build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False)

    assert cases
    assert all(" value applies " not in case.query.lower() for case in cases)
    assert any("symbol identifier" in case.query.lower() for case in cases)
    assert all("output symbol identifier" not in case.query.lower() for case in cases)


def test_build_eval_cases_prefers_llm_rewritten_queries(monkeypatch):
    call_count = 0

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What power supply voltage is required for CA-EN100U?","intent":"spec_lookup","reason":"natural spec lookup"},'
                    '{"query":"What voltage is required for CA-EN100U?","intent":"spec_lookup","reason":"alternate phrasing"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunks = [
        {
            "id": "chunk-llm",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "datasheet_record",
            "title": "CA-EN100U Datasheet",
            "source_filename": "CA-EN100U_Datasheet.pdf",
            "section_path_text": "Specifications",
            "page_from": 1,
            "page_to": 1,
            "content": "Power supply voltage: 24 VDC for standard operation.",
            "metadata_json": {"product_model": "CA-EN100U"},
            "product_model": "CA-EN100U",
        }
    ]

    cases = build_eval_cases_from_chunks(chunks, max_cases=2)

    assert [case.query for case in cases] == ["What power supply voltage is required for CA-EN100U?"]
    assert call_count == 1


def test_build_eval_cases_allows_repeated_content_but_dedupes_questions(monkeypatch):
    call_count = 0

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What power supply voltage is required for CA-EN100U?","intent":"spec_lookup","reason":"natural spec lookup"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    base_chunk = {
        "id": "chunk-llm-1",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "datasheet_record",
        "title": "CA-EN100U Datasheet",
        "source_filename": "CA-EN100U_Datasheet.pdf",
        "section_path_text": "Specifications",
        "page_from": 1,
        "page_to": 1,
        "content": "Power supply voltage: 24 VDC for standard operation.",
        "metadata_json": {
            "product_model": "CA-EN100U",
            "local_rerank_context": "Specifications also list current draw and operating temperature for CA-EN100U.",
        },
        "product_model": "CA-EN100U",
    }
    duplicate_chunk = {
        **base_chunk,
        "id": "chunk-llm-2",
        "section_path_text": "Electrical Specifications",
        "page_from": 2,
        "page_to": 2,
    }

    cases = build_eval_cases_from_chunks([base_chunk, duplicate_chunk], max_cases=4)

    assert [case.source_chunk_id for case in cases] == ["chunk-llm-1", "chunk-llm-2"]
    assert call_count == 2


def test_build_eval_cases_passes_previous_chunk_questions_to_generator(monkeypatch):
    prompts = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What current draw is specified for CA-EN100U?","intent":"spec_lookup","reason":"new facet"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            prompts.append(kwargs["json"]["prompt"])
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunk = {
        "id": "chunk-history",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "datasheet_record",
        "title": "CA-EN100U Datasheet",
        "source_filename": "CA-EN100U_Datasheet.pdf",
        "section_path_text": "Specifications",
        "page_from": 1,
        "page_to": 1,
        "content": "Power supply voltage: 24 VDC. Current draw: 120 mA.",
        "metadata_json": {"product_model": "CA-EN100U"},
        "product_model": "CA-EN100U",
    }

    cases = build_eval_cases_from_chunks(
        [chunk],
        max_cases=2,
        previous_questions_by_chunk_id={
            "chunk-history": ["What power supply voltage is required for CA-EN100U?"],
        },
    )

    assert [case.query for case in cases] == ["What current draw is specified for CA-EN100U?"]
    assert "previous_questions_for_this_section" in prompts[0]
    assert "What power supply voltage is required for CA-EN100U?" in prompts[0]


def test_build_eval_cases_passes_previous_section_questions_to_generator(monkeypatch):
    prompts = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What current draw is specified for CA-EN100U?","intent":"spec_lookup","reason":"new facet"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            prompts.append(kwargs["json"]["prompt"])
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunk = {
        "id": "chunk-section-new",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "datasheet_record",
        "title": "CA-EN100U Datasheet",
        "source_filename": "CA-EN100U_Datasheet.pdf",
        "section_path_text": "Specifications",
        "page_from": 1,
        "page_to": 1,
        "content": "Power supply voltage: 24 VDC. Current draw: 120 mA.",
        "metadata_json": {"product_model": "CA-EN100U"},
        "product_model": "CA-EN100U",
    }

    cases = build_eval_cases_from_chunks(
        [chunk],
        max_cases=2,
        per_chunk_limit=1,
        previous_questions_by_section_key={
            "doc-1\x1fver-1\x1fSpecifications": ["What voltage does CA-EN100U need for power?"],
        },
    )

    assert [case.query for case in cases] == ["What current draw is specified for CA-EN100U?"]
    assert "What voltage does CA-EN100U need for power?" in prompts[0]


def test_build_eval_cases_tracks_questions_generated_for_same_section(monkeypatch):
    prompts = []

    class FakeResponse:
        def __init__(self, query):
            self.query = query

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    f'{{"query":"{self.query}","intent":"spec_lookup","reason":"new facet"}}'
                    ']}'
                )
            }

    class FakeClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            prompts.append(kwargs["json"]["prompt"])
            self.__class__.calls += 1
            query = (
                "What current draw is specified for CA-EN100U?"
                if self.__class__.calls == 1
                else "What operating temperature applies to CA-EN100U?"
            )
            return FakeResponse(query)

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    base_chunk = {
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "datasheet_record",
        "title": "CA-EN100U Datasheet",
        "source_filename": "CA-EN100U_Datasheet.pdf",
        "section_path_text": "Specifications",
        "page_from": 1,
        "page_to": 1,
        "metadata_json": {"product_model": "CA-EN100U"},
        "product_model": "CA-EN100U",
    }
    chunks = [
        {
            **base_chunk,
            "id": "chunk-current",
            "content": "Power supply current draw: 120 mA during standard operation for CA-EN100U.",
        },
        {
            **base_chunk,
            "id": "chunk-temp",
            "content": "Operating temperature range: 0 to 50 C during standard operation for CA-EN100U.",
        },
    ]

    cases = build_eval_cases_from_chunks(chunks, max_cases=2, per_chunk_limit=1)

    assert [case.query for case in cases] == [
        "What current draw is specified for CA-EN100U?",
        "What operating temperature applies to CA-EN100U?",
    ]
    assert "What current draw is specified for CA-EN100U?" in prompts[1]


def test_build_eval_cases_honors_none_from_llm_generation(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "NONE"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunk = {
        "id": "chunk-none",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "datasheet_record",
        "title": "CA-EN100U Datasheet",
        "source_filename": "CA-EN100U_Datasheet.pdf",
        "section_path_text": "Specifications",
        "page_from": 1,
        "page_to": 1,
        "content": "Power supply voltage: 24 VDC for standard operation.",
        "metadata_json": {
            "product_model": "CA-EN100U",
            "local_rerank_context": "Specifications also list current draw and operating temperature for CA-EN100U.",
        },
        "product_model": "CA-EN100U",
    }

    assert build_eval_cases_from_chunks([chunk], max_cases=2) == []


def test_llm_generation_prompt_uses_generic_few_shot_examples(monkeypatch):
    prompts = []
    request_bodies = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What voltage does CA-EN100U need for power?","intent":"spec_lookup","reason":"natural user wording"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            request_bodies.append(kwargs["json"])
            prompts.append(kwargs["json"]["prompt"])
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunk = {
        "id": "chunk-few-shot",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "datasheet_record",
        "title": "CA-EN100U Datasheet",
        "source_filename": "CA-EN100U_Datasheet.pdf",
        "section_path_text": "Specifications",
        "page_from": 1,
        "page_to": 1,
        "content": "Power supply voltage: 24 VDC for standard operation.",
        "metadata_json": {
            "product_model": "CA-EN100U",
            "local_rerank_context": "Specifications also list current draw and operating temperature for CA-EN100U.",
        },
        "product_model": "CA-EN100U",
    }

    cases = build_eval_cases_from_chunks([chunk], max_cases=1)

    assert cases
    assert USER_STYLE_QUERY_FEW_SHOT_EXAMPLES in prompts[0]
    assert "fallback_examples" not in prompts[0]
    assert "Good query: What voltage does MODEL-A need for power?" in prompts[0]
    assert "Bad query: Which disconnect all other devices detail is needed?" in prompts[0]
    assert "section_context_excerpt" in prompts[0]
    assert "current draw and operating temperature" in prompts[0]
    assert request_bodies[0]["think"] is False
    assert request_bodies[0]["model"] == "qwen3.5:27b"
    assert request_bodies[0]["options"]["num_ctx"] == 32768
    assert "format" not in request_bodies[0]


def test_parse_generated_queries_accepts_common_model_json_wrappers():
    payload = """
    <think>I should produce JSON only.</think>
    ```json
    {"queries":[{"query":"What voltage does MODEL-A need for power?","intent":"spec_lookup","reason":"natural"}]}
    ```
    """

    parsed = _parse_generated_queries(payload)

    assert parsed == [
        {
            "query": "What voltage does MODEL-A need for power?",
            "intent": "spec_lookup",
            "reason": "natural",
        }
    ]


def test_parse_generated_queries_accepts_top_level_query_arrays():
    parsed = _parse_generated_queries(
        '[{"query":"Which endian setting matches my PLC byte order?","intent":"setting_lookup","reason":"natural"}]'
    )

    assert parsed == [
        {
            "query": "Which endian setting matches my PLC byte order?",
            "intent": "setting_lookup",
            "reason": "natural",
        }
    ]


def test_parse_generated_queries_rejects_empty_model_response():
    with pytest.raises(ValueError, match="empty generated-query response"):
        _parse_generated_queries("")


def test_validate_eval_case_rejects_source_address_syntax_queries():
    chunk = {
        "id": "chunk-command-flag",
        "source_document_id": "doc-ljx",
        "document_version_id": "ver-ljx",
        "chunk_type": "datasheet_record",
        "title": "LJ-X8000",
        "source_filename": "ljx.pdf",
        "section_path_text": "Command execution",
        "page_from": 45,
        "page_to": 45,
        "content": "Check whether the tag (LJX3D: I.Data[0].1) to which the Command error flag has been assigned is ON or OFF.",
        "metadata_json": {"product_model": "LJ-X8000"},
        "product_model": "LJ-X8000",
    }
    anchors = ["check", "whether", "ljx3d", "i.data"]

    assert validate_eval_case("Is LJX3D: I.Data0.1 the correct indicator for Command errors?", chunk, anchors) == (
        False,
        "source_address_syntax_query",
    )
    assert validate_eval_case("How do I determine if a command error occurred using the LJX3D tag?", chunk, anchors) == (
        False,
        "source_address_syntax_query",
    )
    assert validate_eval_case("What is the Input.ToolID for the first command in Command Settings?", chunk, anchors) == (
        False,
        "source_address_syntax_query",
    )
    assert validate_eval_case("Which tag indicates whether a command error occurred?", chunk, anchors) == (
        True,
        "validated",
    )


def test_validate_eval_case_rejects_table_artifact_queries():
    chunk = {
        "id": "chunk-table-artifact",
        "source_document_id": "doc-1",
        "document_version_id": "ver-1",
        "chunk_type": "table_record",
        "title": "Manual",
        "source_filename": "Manual.pdf",
        "section_path_text": "Measurements",
        "page_from": 2,
        "page_to": 2,
        "content": "Column headers: Description of measurement; Row headers: XYT; Cell value: Position XY / Detected Angle; Row: 25; Column: 2",
        "metadata_json": {
            "product_model": "CV-X482",
            "table_cell": True,
            "table_row": 25,
            "table_column": 2,
            "table_row_headers": ["XYT"],
            "table_column_headers": ["Description of measurement"],
        },
    }
    anchors = ["position", "detected", "angle", "description"]

    assert validate_eval_case("What measurement does XYT row 25 report in column 2?", chunk, anchors) == (
        False,
        "table_artifact_syntax_query",
    )
    assert validate_eval_case("What row number holds the entry for XYT detected angle?", chunk, anchors) == (
        False,
        "table_artifact_syntax_query",
    )
    assert validate_eval_case("What is the Description of measurement XYT for CV-X482?", chunk, anchors) == (
        False,
        "table_artifact_syntax_query",
    )
    assert validate_eval_case("What description measurement position applies to CV-X482?", chunk, anchors) == (
        False,
        "table_artifact_syntax_query",
    )
    assert validate_eval_case("What value is used for Y Description of separation?", chunk, anchors) == (
        False,
        "table_artifact_syntax_query",
    )
    assert validate_eval_case("What XYT measurement does CV-X482 report?", chunk, anchors) == (True, "validated")


def test_build_eval_cases_falls_back_when_llm_generation_fails(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            raise RuntimeError("generation unavailable")

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunks = [
        {
            "id": "chunk-fallback",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "datasheet_record",
            "title": "CA-EN100U Datasheet",
            "source_filename": "CA-EN100U_Datasheet.pdf",
            "section_path_text": "Specifications",
            "page_from": 1,
            "page_to": 1,
            "content": "Power supply voltage: 24 VDC for standard operation.",
            "metadata_json": {"product_model": "CA-EN100U"},
            "product_model": "CA-EN100U",
        }
    ]

    cases = build_eval_cases_from_chunks(chunks, max_cases=2)

    assert cases
    assert any(
        "ca-en100u" in case.query.lower()
        and ("power" in case.query.lower() or "voltage" in case.query.lower())
        for case in cases
    )


def test_build_eval_cases_filters_meta_llm_queries(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What specification does LJ-X8000 give for laser?","intent":"spec_lookup","reason":"bad meta phrasing"},'
                    '{"query":"What laser wavelength applies to LJ-X8000?","intent":"spec_lookup","reason":"good question phrasing"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunks = [
        {
            "id": "chunk-meta",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "spec_record",
            "title": "LJ-X8000",
            "source_filename": "LJ-X8000.pdf",
            "section_path_text": "Laser",
            "page_from": 1,
            "page_to": 1,
            "content": "Laser specification: Wavelength: 405 nm Output: 10 mW for normal operation.",
            "metadata_json": {"product_model": "LJ-X8000"},
            "product_model": "LJ-X8000",
        }
    ]

    cases = build_eval_cases_from_chunks(chunks, max_cases=2)

    queries = [case.query for case in cases]
    assert "What laser wavelength applies to LJ-X8000?" in queries
    assert all("what specification" not in query.lower() for query in queries)
    assert all(query.endswith("?") for query in queries)


def test_validate_eval_case_rejects_query_not_specific_to_source_context():
    chunk = {
        "chunk_type": "spec_record",
        "title": "LJ-X8000",
        "section_path_text": "LJ-X8080",
        "content": "Capture the shape of targets in exceptional detail with 3200 points/profile.",
        "metadata_json": {"product_model": "New LJ-X8000 Series"},
        "product_model": "New LJ-X8000 Series",
    }
    anchors = ["3200", "points/profile", "capture", "shape"]

    assert validate_eval_case("New LJ-X8000 Series 3200", chunk, anchors) == (False, "not_question_form")
    valid, reason = validate_eval_case("What detail applies to New LJ-X8000 Series?", chunk, anchors)
    assert valid is False
    assert reason in {"mechanical_query", "low_specificity", "weak_source_affinity", "weak_source_discriminator"}
    assert validate_eval_case("What 3200 points/profile applies to LJ-X8000?", chunk, anchors) == (True, "validated")


def test_table_header_chunks_are_not_queryworthy_as_standalone_questions():
    chunk = {
        "id": "header-row",
        "source_document_id": "doc-iv4",
        "document_version_id": "ver-iv4",
        "chunk_type": "table_record",
        "title": "IV4 Manual",
        "source_filename": "iv4.pdf",
        "section_path_text": "R1.20",
        "page_from": 14,
        "page_to": 14,
        "content": "Table header: If the detection becomes unstable due to the effect of; Header role: row; Row: 32; Column: 0",
        "metadata_json": {
            "chunk_family": "table_record",
            "product_model": "IV4-G120",
            "table_header": True,
            "table_header_role": "row",
            "table_row": 32,
            "table_column": 0,
        },
        "product_model": "IV4-G120",
    }
    anchors = ["detection", "becomes", "unstable"]

    assert chunk_is_queryworthy(chunk, anchors) is False
    assert build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False) == []


def test_fallback_eval_queries_include_context_anchors_for_compact_specs():
    chunks = [
        {
            "id": "chunk-specific",
            "source_document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_type": "spec_record",
            "title": "LJ-X8000",
            "source_filename": "LJ-X8000.pdf",
            "section_path_text": "LJ-X8080",
            "page_from": 11,
            "page_to": 11,
            "content": "Capture the shape of targets in exceptional detail with 3200 points/profile.",
            "metadata_json": {"product_model": "New LJ-X8000 Series"},
            "product_model": "New LJ-X8000 Series",
        }
    ]

    cases = build_eval_cases_from_chunks(chunks, max_cases=3, use_llm_generation=False)

    assert cases
    assert all("points/profile" in case.query.lower() or "capture" in case.query.lower() for case in cases)
    assert all(case.query != "New LJ-X8000 Series 3200" for case in cases)
    assert all(case.query.endswith("?") for case in cases)


def test_eval_queries_reject_ambiguous_storage_only_phrasing():
    chunk = {
        "id": "chunk-command-number",
        "source_document_id": "doc-ljx",
        "document_version_id": "ver-ljx",
        "chunk_type": "datasheet_record",
        "title": "AS_128241_LJ-X8000_SG_D48GB_WW_GB_2072_1.pdf",
        "source_filename": "AS_128241_LJ-X8000_SG_D48GB_WW_GB_2072_1.pdf",
        "section_path_text": "PLC",
        "page_from": 38,
        "page_to": 38,
        "content": "The PLC stores the number: specified-command No. in Command Number and the command parameters in Command Parameter.",
        "metadata_json": {"product_model": "D48GB"},
        "product_model": "D48GB",
    }

    valid, reason = validate_eval_case("What stores number for D48GB?", chunk, ["specified-command", "command", "parameters"])
    assert valid is False
    assert reason == "filename_artifact_query"

    cases = build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False)
    queries = [case.query.lower() for case in cases]

    assert cases
    assert all(query != "d48gb stores number" for query in queries)
    assert all(query.endswith("?") for query in queries)
    assert all("d48gb" not in query for query in queries)
    assert all("command" in query for query in queries)
    assert any("specified-command" in query for query in queries)


def test_eval_queries_create_question_form_for_disconnect_guidance():
    chunk = {
        "id": "chunk-disconnect",
        "source_document_id": "doc-ljx",
        "document_version_id": "ver-ljx",
        "chunk_type": "atomic_text",
        "title": "AS_128241_LJ-X8000_SG_D48GB_WW_GB_2072_1.pdf",
        "source_filename": "AS_128241_LJ-X8000_SG_D48GB_WW_GB_2072_1.pdf",
        "section_path_text": "EtherNet/IP",
        "page_from": 3,
        "page_to": 3,
        "content": (
            "To give priority to the checking of the EtherNet/IP connection, disconnect all devices other "
            "than the LJ-X and the PLC from the hub before establishing the connection."
        ),
        "metadata_json": {"product_model": "D48GB"},
        "product_model": "D48GB",
    }

    valid, reason = validate_eval_case("D48GB disconnect devices", chunk, ["ethernet/ip", "disconnect", "devices", "plc"])
    assert valid is False
    assert reason == "not_question_form"

    cases = build_eval_cases_from_chunks([chunk], max_cases=3, use_llm_generation=False)
    queries = [case.query for case in cases]

    assert queries
    assert "Which other devices should be disconnected?" in queries
    assert "Which devices should be disconnected before checking the EtherNet/IP connection?" in queries
    assert all("D48GB" not in query for query in queries)


def test_eval_generation_does_not_prompt_with_source_filename_artifacts(monkeypatch):
    prompts = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What specified-command number does the PLC store?","intent":"spec_lookup","reason":"snippet-grounded"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            prompts.append(kwargs["json"]["prompt"])
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunk = {
        "id": "chunk-filename-artifact",
        "source_document_id": "doc-ljx",
        "document_version_id": "ver-ljx",
        "chunk_type": "datasheet_record",
        "title": "AS_128241_LJ-X8000_SG_D48GB_WW_GB_2072_1.pdf",
        "source_filename": "AS_128241_LJ-X8000_SG_D48GB_WW_GB_2072_1.pdf",
        "section_path_text": "PLC",
        "page_from": 38,
        "page_to": 38,
        "content": "The PLC stores the number: specified-command No. in Command Number and the command parameters in Command Parameter.",
        "metadata_json": {"product_model": "D48GB"},
        "product_model": "D48GB",
    }

    cases = build_eval_cases_from_chunks([chunk], max_cases=2)

    assert cases[0].query == "What specified-command number does the PLC store?"
    assert all("D48GB" not in case.query for case in cases)
    assert prompts
    assert "source_filename" not in prompts[0]
    assert "D48GB" not in prompts[0]
    assert "AS_128241" not in prompts[0]


def test_eval_generation_rejects_copied_source_phrasing(monkeypatch):
    prompts = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What obtained authentication is specified for XG-X Series?","intent":"bad_copy","reason":"copied"},'
                    '{"query":"Which controller combination has CSA approval for XG-X Series?","intent":"fair_user_question","reason":"paraphrased"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            prompts.append(kwargs["json"]["prompt"])
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunk = {
        "id": "chunk-csa-auth",
        "source_document_id": "doc-xgx",
        "document_version_id": "ver-xgx",
        "chunk_type": "warning_record",
        "title": "Safety information for XG-X Series",
        "source_filename": "AS_151433_XG-X_UM_C84US_KA_GB_2035_8a.pdf",
        "section_path_text": "Safety information",
        "page_from": 6,
        "page_to": 6,
        "content": (
            "The obtained CSA authentication of the LJ-X8000 Series head is only for the case "
            "when it is used in combination with the LJ-X8000 Series controller."
        ),
        "metadata_json": {"product_model": "XG-X Series"},
        "product_model": "XG-X Series",
    }

    anchors = ["obtained", "authentication", "x8000", "series"]

    valid, reason = validate_eval_case("What obtained authentication is specified for XG-X Series?", chunk, anchors)
    assert valid is False
    assert reason == "copied_source_phrase"

    cases = build_eval_cases_from_chunks([chunk], max_cases=1)

    assert cases
    assert cases[0].query == "Which controller combination has CSA approval for XG-X Series?"
    assert "Do not copy any other exact sentence, clause, or two-or-more-word phrase" in prompts[0]
    assert "obtained authentication" in prompts[0]


def test_eval_generation_rejects_bracketed_source_label_queries(monkeypatch):
    prompts = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"What is [Luminance Output Type] for VS Series?","intent":"bad_brackets","reason":"copied label"},'
                    '{"query":"Which luminance signal should the VS Series output?","intent":"fair_user_question","reason":"natural phrasing"}'
                    ']}'
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            prompts.append(kwargs["json"]["prompt"])
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_evals.retrieval_eval.httpx.Client", FakeClient)

    chunk = {
        "id": "chunk-luminance-output",
        "source_document_id": "doc-vs",
        "document_version_id": "ver-vs",
        "chunk_type": "table_record",
        "title": "VS Series",
        "source_filename": "vs-series.pdf",
        "section_path_text": "Output settings",
        "page_from": 12,
        "page_to": 12,
        "content": "[Luminance Output Type]: Analog output; Output voltage: 0 to 10 V.",
        "metadata_json": {
            "product_model": "VS Series",
            "table_column_headers": ["Luminance Output Type", "Output voltage"],
        },
        "product_model": "VS Series",
    }
    anchors = ["luminance", "output", "analog", "voltage"]

    assert validate_eval_case("What is [Luminance Output Type] for VS Series?", chunk, anchors) == (
        False,
        "bracketed_source_label_query",
    )
    assert validate_eval_case("What is the Luminance Output Type for VS Series?", chunk, anchors) == (
        False,
        "bracketed_source_label_query",
    )

    cases = build_eval_cases_from_chunks([chunk], max_cases=1)

    assert cases
    assert cases[0].query == "Which analog signal voltage should VS Series output?"
    assert prompts
    assert "[Luminance Output Type]" not in prompts[0]
    assert "Do not include square brackets" in prompts[0]


def test_validate_eval_case_accepts_access_control_user_question():
    chunk = {
        "chunk_type": "atomic_text",
        "title": "VS Series",
        "source_filename": "vs-series.pdf",
        "section_path_text": "Program setting protection",
        "content": (
            "To allow the use of a program setting saved in the computer only for a restricted set of "
            "VS cameras, set a password and the MAC addresses of the target VS cameras."
        ),
        "metadata_json": {"product_family": "VS Series"},
        "product_model": "VS Series",
    }

    valid, reason = validate_eval_case(
        "How do I limit a saved VS program setting so only selected cameras can use it?",
        chunk,
        ["password", "mac", "addresses", "program"],
    )

    assert (valid, reason) == (True, "validated")


def test_score_search_results_passes_on_same_document_term_overlap():
    case = RetrievalEvalCase(
        case_id="c1",
        query="What is the voltage specification for CA-EN100U?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="CA-EN100U Datasheet",
        source_filename="CA-EN100U_Datasheet.pdf",
        chunk_type="datasheet_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["voltage", "24", "ca-en100u"],
        expected_snippet="Voltage: 24 V",
        generation_method="spec_primary",
        source_metadata={"product_model": "CA-EN100U"},
    )
    results = [
        {
            "chunk_id": "chunk-x",
            "source_document_id": "doc-1",
            "section_path": ["Specifications"],
            "content": "The CA-EN100U voltage specification is 24 V.",
        }
    ]
    evaluation = score_search_results(case, results)
    assert evaluation["passed"] is True
    assert evaluation["rank"] == 1
    assert evaluation["candidate_recall"] is True
    assert evaluation["metadata_document_selection"]["attempted"] is False


def test_score_search_results_requires_code_anchors_for_same_document_overlap():
    case = RetrievalEvalCase(
        case_id="c-code-anchors",
        query="In the CV-X482 cyclic communication allocation table, what is PID 428 at address bytes 0016-0018 named?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="expected-pid-row",
        source_title="CV-X Manual",
        source_filename="cv-x.pdf",
        chunk_type="atomic_text",
        section_path="Cyclic communication",
        page_from=919,
        page_to=919,
        expected_terms=["pid", "428", "0016", "0018", "total", "count"],
        expected_snippet="Measurement count area | 0016 0017 0018 | PID 428 | Total count",
        generation_method="manual_guardrail_curated_single_step_replacement",
        source_metadata={"product_model": "CV-X482"},
    )
    wrong_same_document = [
        {
            "chunk_id": "wrong-inspection-result-row",
            "source_document_id": "doc-cvx",
            "section_path": ["Cyclic communication"],
            "content": "The inspection result area uses address bytes 0016 through 0018 and displays the total measurement count.",
        }
    ]
    correct_same_document = [
        {
            "chunk_id": "equivalent-pid-row",
            "source_document_id": "doc-cvx",
            "section_path": ["Cyclic communication"],
            "content": "Measurement count area | 0016 0017 0018 | PID 428 | Total count.",
        }
    ]

    wrong = score_search_results(case, wrong_same_document)
    correct = score_search_results(case, correct_same_document)

    assert wrong["passed"] is False
    assert wrong["failure_category"] == "ranking_or_context_loss"
    assert wrong["match_reason"] == "no_match"
    assert correct["passed"] is True
    assert correct["match_reason"] == "same_section_term_overlap"


def test_score_search_results_requires_compound_code_prefix_binding():
    case = RetrievalEvalCase(
        case_id="c-code-prefix-binding",
        query="In the CV-X482 cyclic communication allocation table, what is PID 428 at address bytes 0016-0018 named?",
        source_document_id="doc-cvx",
        document_version_id="ver-cvx",
        source_chunk_id="expected-pid-row",
        source_title="CV-X Manual",
        source_filename="cv-x.pdf",
        chunk_type="atomic_text",
        section_path="Cyclic communication",
        page_from=919,
        page_to=919,
        expected_terms=["pid", "428", "0016", "0018", "total", "count"],
        expected_snippet="Measurement count area | 0016 0017 0018 | PID 428 | Total count",
        generation_method="manual_guardrail_curated_single_step_replacement",
        source_metadata={"product_model": "CV-X482"},
    )

    adverse_results = [
        (
            "wrong-prefix",
            "Measurement count area | 0016 0017 0018 | PQD428 | Total count.",
        ),
        (
            "missing-prefix",
            "Measurement count area | 0016 0017 0018 | 428 | Total count.",
        ),
        (
            "wrong-number",
            "Measurement count area | 0016 0017 0018 | PID 429 | Total count.",
        ),
    ]

    for chunk_id, content in adverse_results:
        scored = score_search_results(
            case,
            [
                {
                    "chunk_id": chunk_id,
                    "source_document_id": "doc-cvx",
                    "section_path": ["Cyclic communication"],
                    "content": content,
                }
            ],
        )
        assert scored["passed"] is False
        assert scored["failure_category"] == "ranking_or_context_loss"
        assert scored["match_reason"] == "no_match"

    correct = score_search_results(
        case,
        [
            {
                "chunk_id": "correct-prefix",
                "source_document_id": "doc-cvx",
                "section_path": ["Cyclic communication"],
                "content": "Measurement count area | 0016 0017 0018 | PID 428 | Total count.",
            }
        ],
    )
    assert correct["passed"] is True
    assert correct["match_reason"] == "same_section_term_overlap"


def test_score_search_results_rejects_same_page_without_source_terms():
    case = RetrievalEvalCase(
        case_id="c-page",
        query="What voltage does MODEL-1 use?",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-expected",
        source_title="MODEL-1 Manual",
        source_filename="model-1.pdf",
        chunk_type="spec_record",
        section_path="Specifications",
        page_from=12,
        page_to=12,
        expected_terms=["24", "vdc"],
        expected_snippet="Power supply voltage: 24 VDC",
        generation_method="spec_primary",
        source_metadata={"product_model": "MODEL-1"},
    )
    results = [
        {
            "chunk_id": "chunk-same-page",
            "source_document_id": "doc-1",
            "pages": [12],
            "content": "Related voltage information from the same manual page.",
            "metadata": {"chunk_type": "spec_record"},
        }
    ]

    evaluation = score_search_results(case, results)

    assert evaluation["passed"] is False
    assert evaluation["failure_category"] == "ranking_or_context_loss"
    assert evaluation["match_reason"] == "no_match"


def test_score_document_selection_passes_when_expected_document_is_selected():
    case = RetrievalEvalCase(
        case_id="c-selection",
        query="AX-1200 pressure repeatability",
        source_document_id="doc-expected",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="AX-1200 Manual",
        source_filename="ax-1200.pdf",
        chunk_type="table_record",
        section_path="Specifications",
        page_from=1,
        page_to=1,
        expected_terms=["ax-1200", "pressure", "repeatability"],
        expected_snippet="AX-1200 pressure repeatability",
        generation_method="table_primary",
        source_metadata={"product_model": "AX-1200"},
    )
    results = [
        {
            "chunk_id": "chunk-x",
            "source_document_id": "doc-expected",
            "section_path": ["Specifications"],
            "content": "AX-1200 pressure repeatability is 0.02 kPa.",
            "metadata": {
                "selected_document_metadata_hits": [
                    {"source_document_id": "doc-expected", "score": 0.2},
                    {"source_document_id": "doc-other", "score": 0.1},
                ]
            },
        }
    ]

    evaluation = score_search_results(case, results)

    assert evaluation["passed"] is True
    assert evaluation["metadata_document_selection"] == {
        "attempted": True,
        "passed": True,
        "rank": 1,
        "expected_source_document_id": "doc-expected",
        "selected_source_document_ids": ["doc-expected", "doc-other"],
        "hit_count": 2,
        "failure_category": None,
    }


def test_score_document_selection_reports_metadata_document_miss():
    case = RetrievalEvalCase(
        case_id="c-selection-miss",
        query="MTR-700 bearing preload",
        source_document_id="doc-expected",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="MTR-700 Manual",
        source_filename="mtr-700.pdf",
        chunk_type="atomic_text",
        section_path="Maintenance",
        page_from=1,
        page_to=1,
        expected_terms=["mtr-700", "bearing", "preload"],
        expected_snippet="MTR-700 bearing preload",
        generation_method="general_multi",
        source_metadata={"product_model": "MTR-700"},
    )

    selection = score_document_selection(
        case,
        [
            {
                "metadata": {
                    "selected_document_metadata_hits": [
                        {"source_document_id": "doc-other", "score": 0.4},
                    ]
                }
            }
        ],
    )

    assert selection["attempted"] is True
    assert selection["passed"] is False
    assert selection["rank"] is None
    assert selection["failure_category"] == "metadata_document_miss"


def test_score_search_results_handles_table_style_queries():
    case = RetrievalEvalCase(
        case_id="c2",
        query="In IV-H, what value is given for after?",
        source_document_id="doc-2",
        document_version_id="ver-2",
        source_chunk_id="chunk-table",
        source_title="IV-H Manual",
        source_filename="IV-H.pdf",
        chunk_type="table_record",
        section_path="Document",
        page_from=10,
        page_to=10,
        expected_terms=["after", "setting", "completed", "click"],
        expected_snippet="After the setting is completed, click the OK button.",
        generation_method="spec_value",
        source_metadata={"product_model": "IV-H"},
    )
    results = [
        {
            "chunk_id": "other",
            "source_document_id": "doc-2",
            "section_path": [],
            "content": "After the setting is completed, click the OK button.",
        }
    ]
    evaluation = score_search_results(case, results)
    assert evaluation["passed"] is True
    assert evaluation["match_reason"] == "same_document_term_overlap"


def test_score_search_results_matches_slash_terms_across_table_evidence():
    case = RetrievalEvalCase(
        case_id="c-slash",
        query="New LJ-X8000 Series 3200 points/profile",
        source_document_id="doc-lj",
        document_version_id="ver-lj",
        source_chunk_id="chunk-source",
        source_title="LJ-X8000",
        source_filename="lj-x8000.pdf",
        chunk_type="spec_record",
        section_path="LJ-X8080",
        page_from=11,
        page_to=11,
        expected_terms=["3200", "points/profile", "linearity", "significantly"],
        expected_snippet="With 3200 points/profile, X: axis linearity has been significantly improved.",
        generation_method="spec_primary_multi",
        source_metadata={"product_model": "New LJ-X8000 Series"},
    )
    results = [
        {
            "chunk_id": "table-profile-count",
            "source_document_id": "doc-lj",
            "section_path": ["HALCON"],
            "content": (
                "Column headers: LJ-X8020 > LJ-X8060 > LJ-X8080; "
                "Row headers: Profile data count; Cell value: 3200 points"
            ),
            "metadata": {"chunk_type": "table_record", "product_model": "New LJ-X8000 Series"},
        }
    ]

    evaluation = score_search_results(case, results)

    assert evaluation["passed"] is True
    assert evaluation["overlap_terms"] == 2
    assert evaluation["query_overlap_terms"] >= 2
    assert evaluation["match_reason"] == "same_document_answerable_evidence"


def test_score_search_results_accepts_applicable_equivalent_table_evidence():
    case = RetrievalEvalCase(
        case_id="c-equivalent",
        query="What 02h PDO Source Sub Index value applies to IV4-G600CA?",
        source_document_id="doc-specific",
        document_version_id="ver-specific",
        source_chunk_id="chunk-specific",
        source_title="IV4-G600CA Manual",
        source_filename="iv4-g600ca.pdf",
        chunk_type="table_record",
        section_path="9-104",
        page_from=386,
        page_to=386,
        expected_terms=["02h", "m-1", "process", "object"],
        expected_snippet="Cell value: 02h+(M-1)xAh",
        generation_method="table_row_column_value",
        source_metadata={"product_model": "IV4-G600CA"},
    )
    results = [
        {
            "chunk_id": "chunk-family",
            "source_document_id": "doc-family",
            "section_path": ["9-106"],
            "content": (
                "Column headers: Process data object content (PDO Content) > Source Sub Index (HEX); "
                "Cell value: 02h + (M-1) x Ah"
            ),
            "metadata": {
                "chunk_type": "table_record",
                "product_model": "IV4-G120",
                "devices": ["IV4-G120", "IV4-G600CA"],
            },
        }
    ]

    evaluation = score_search_results(case, results)

    assert evaluation["passed"] is True
    assert evaluation["candidate_recall"] is True
    assert evaluation["match_reason"] == "applicable_equivalent_answer_evidence"


def test_score_search_results_requires_multi_step_expected_evidence():
    case = RetrievalEvalCase(
        case_id="c-multi",
        query="What causes timeout error, and how should it be corrected?",
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=10,
        page_to=10,
        expected_terms=["timeout", "encoder", "connection", "detect"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_source_chunk_ids=["error-cell", "cause-cell", "action-cell"],
        expected_evidence=[
            {"chunk_id": "error-cell", "expected_terms": ["timeout", "encoder"]},
            {"chunk_id": "cause-cell", "expected_terms": ["encoder", "timeout"]},
            {"chunk_id": "action-cell", "expected_terms": ["connection", "detect"]},
        ],
    )
    missing_action_results = [
        {
            "chunk_id": "error-cell",
            "source_document_id": "doc-xgx",
            "section_path": ["Troubleshooting"],
            "content": "Timeout error has occurred while waiting for encoder input.",
        },
        {
            "chunk_id": "cause-cell",
            "source_document_id": "doc-xgx",
            "section_path": ["Troubleshooting"],
            "content": "Encoder input timeout was detected.",
        },
    ]
    complete_results = [
        *missing_action_results,
        {
            "chunk_id": "action-cell",
            "source_document_id": "doc-xgx",
            "section_path": ["Troubleshooting"],
            "content": "Check the encoder connection and change the Detect Timeout time.",
        },
    ]

    missing = score_search_results(case, missing_action_results)
    complete = score_search_results(case, complete_results)

    assert missing["passed"] is False
    assert missing["failure_category"] == "ranking_or_context_loss"
    assert missing["missing_evidence"][0]["chunk_id"] == "action-cell"
    assert complete["passed"] is True
    assert complete["match_reason"] == "multi_step_expected_evidence"


def test_score_search_results_counts_table_row_group_context_as_multi_step_evidence():
    case = RetrievalEvalCase(
        case_id="c-row-context",
        query="What causes link error, and how should it be corrected?",
        source_document_id="doc-xgx",
        document_version_id="ver-xgx",
        source_chunk_id="error-cell",
        source_title="XG-X",
        source_filename="xgx.pdf",
        chunk_type="table_record",
        section_path="Troubleshooting",
        page_from=10,
        page_to=10,
        expected_terms=["link", "cable", "check"],
        expected_snippet="Error, cause, and corrective action",
        generation_method="table_sibling_error_cause_action",
        source_metadata={"product_family": "XG-X Series"},
        retrieval_task="multi_step_retrieval",
        expected_source_chunk_ids=["error-cell", "cause-cell", "action-cell"],
        expected_evidence=[
            {"chunk_id": "error-cell", "expected_terms": ["link", "error"]},
            {"chunk_id": "cause-cell", "expected_terms": ["cable", "disconnected"]},
            {"chunk_id": "action-cell", "expected_terms": ["check", "cable"]},
        ],
    )
    results = [
        {
            "chunk_id": "error-cell",
            "source_document_id": "doc-xgx",
            "section_path": ["Troubleshooting"],
            "content": "Column headers: Error Message; Cell value: Link error.",
            "metadata": {
                "chunk_type": "table_record",
                "context_window": "Error Message: Link error; Cause: Cable disconnected; Corrective Action: Check the cable.",
            },
        }
    ]

    evaluation = score_search_results(case, results)

    assert evaluation["passed"] is True
    assert evaluation["match_reason"] == "multi_step_expected_evidence"


def test_score_search_results_accepts_cross_document_same_field_equivalent_evidence():
    case = RetrievalEvalCase(
        case_id="c-cross-equivalent",
        query="What power supply voltage values apply for IV4-G120 and LJ-X8000?",
        source_document_id="doc-iv",
        document_version_id="ver-iv",
        source_chunk_id="iv-voltage",
        source_title="IV4",
        source_filename="iv.pdf",
        chunk_type="table_record",
        section_path="Electrical specifications",
        page_from=7,
        page_to=8,
        expected_terms=["24", "vdc"],
        expected_snippet="Power supply voltage evidence from both documents",
        generation_method="cross_document_same_field_evidence",
        source_metadata={"product_model": "IV4-G120"},
        retrieval_task="multi_step_retrieval",
        expected_source_chunk_ids=["iv-voltage", "lj-voltage"],
        expected_evidence=[
            {
                "chunk_id": "iv-voltage",
                "source_document_id": "doc-iv",
                "field": "power supply voltage",
                "product_identifiers": ["iv4g120"],
                "expected_terms": ["24", "vdc"],
            },
            {
                "chunk_id": "lj-voltage",
                "field": "power supply voltage",
                "product_identifiers": ["ljx8000"],
                "expected_terms": ["24", "vdc"],
            },
        ],
    )
    results = [
        {
            "chunk_id": "iv-voltage",
            "source_document_id": "doc-iv",
            "content": "Column headers: Power supply voltage; Cell value: 24 VDC",
            "metadata": {"chunk_type": "table_record", "table_column_headers": ["Power supply voltage"], "product_model": "IV4-G120"},
        },
        {
            "chunk_id": "lj-voltage-equivalent",
            "source_document_id": "doc-lj-family",
            "content": "Column headers: Power supply voltage; Cell value: 24 VDC +/-10%",
            "metadata": {"chunk_type": "table_record", "table_column_headers": ["Power supply voltage"], "product_family": "LJ-X8000"},
        },
    ]

    evaluation = score_search_results(case, results)

    assert evaluation["passed"] is True
    assert evaluation["matched_evidence"][1]["chunk_id"] == "lj-voltage"


def test_score_search_results_categorizes_candidate_miss():
    case = RetrievalEvalCase(
        case_id="c3",
        query="What does the manual say about encoder timing?",
        source_document_id="doc-9",
        document_version_id="ver-9",
        source_chunk_id="chunk-9",
        source_title="Manual",
        source_filename="manual.pdf",
        chunk_type="atomic_text",
        section_path="Timing",
        page_from=3,
        page_to=3,
        expected_terms=["encoder", "timing", "trigger"],
        expected_snippet="Encoder timing trigger settings",
        generation_method="general_multi",
        source_metadata={},
    )
    results = [{"chunk_id": "other", "source_document_id": "doc-2", "section_path": ["Other"], "content": "Voltage specification 24 V"}]
    evaluation = score_search_results(case, results)
    assert evaluation["passed"] is False
    assert evaluation["failure_category"] == "candidate_miss"
