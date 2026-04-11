from manuals_rag_evals.retrieval_eval import RetrievalEvalCase, build_eval_cases_from_chunks, score_search_results


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
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"queries":['
                    '{"query":"power supply voltage for CA-EN100U","intent":"spec_lookup","reason":"natural spec lookup"},'
                    '{"query":"CA-EN100U required voltage","intent":"spec_lookup","reason":"alternate phrasing"}'
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

    assert [case.query for case in cases] == [
        "power supply voltage for CA-EN100U",
        "CA-EN100U required voltage",
    ]


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
                    '{"query":"LJ-X8000 laser wavelength","intent":"spec_lookup","reason":"good search phrasing"}'
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
    assert "LJ-X8000 laser wavelength" in queries
    assert all("what specification" not in query.lower() for query in queries)


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
