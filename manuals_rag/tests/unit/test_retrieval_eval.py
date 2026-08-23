from manuals_rag_evals.retrieval_eval import (
    RetrievalEvalCase,
    build_eval_cases_from_chunks,
    score_document_selection,
    score_search_results,
    validate_eval_case,
)


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
        "metadata_json": {"product_model": "CA-EN100U"},
        "product_model": "CA-EN100U",
    }
    duplicate_chunk = {
        **base_chunk,
        "id": "chunk-llm-2",
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
    assert "previous_questions_for_this_chunk" in prompts[0]
    assert "What power supply voltage is required for CA-EN100U?" in prompts[0]


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
    assert reason in {"low_specificity", "weak_source_affinity", "weak_source_discriminator"}
    assert validate_eval_case("What 3200 points/profile applies to LJ-X8000?", chunk, anchors) == (True, "validated")


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
