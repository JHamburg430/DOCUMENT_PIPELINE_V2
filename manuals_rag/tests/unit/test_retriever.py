from types import SimpleNamespace

import pytest

from manuals_rag_retrieval.document_metadata import enrich_document_metadata_with_chunk_signals
from manuals_rag_retrieval.qdrant_store import QdrantStore
from manuals_rag_retrieval import retriever
from manuals_rag_retrieval.retriever import _resolve_rerank_device, fuse_results, rerank_results, run_dense_search, run_sparse_search, run_table_search
from manuals_rag_retrieval.query_analysis import QueryAnalysis, analyze_query
from manuals_rag_schemas.documents import SearchResult
from qdrant_client.http.exceptions import UnexpectedResponse


def test_fuse_rrf_prefers_documents_present_in_multiple_result_sets():
    dense = [
        SearchResult(
            chunk_id="shared",
            score=0.1,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Section"],
            content="shared result",
            metadata={},
        ),
        SearchResult(
            chunk_id="dense-only",
            score=0.2,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-2",
            pages=[1],
            section_path=["Section"],
            content="dense result",
            metadata={},
        ),
    ]
    sparse = [
        SearchResult(
            chunk_id="shared",
            score=0.05,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Section"],
            content="shared result",
            metadata={},
        )
    ]
    fused = QdrantStore.fuse_rrf([dense, sparse], limit=2)
    assert fused[0].chunk_id == "shared"


def test_rerank_results_uses_haystack_ranker_order(monkeypatch):
    results = [
        SearchResult(
            chunk_id="first",
            score=0.09,
            title="Doc 1",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Section"],
            content="first",
            metadata={},
        ),
        SearchResult(
            chunk_id="second",
            score=0.08,
            title="Doc 2",
            document_version_id="ver-1",
            source_document_id="doc-2",
            pages=[1],
            section_path=["Section"],
            content="second",
            metadata={},
        ),
    ]

    class FakeRanker:
        def run(self, query: str, documents: list[object]) -> dict[str, list[object]]:
            assert query == "test query"
            assert [document.id for document in documents] == ["first", "second"]
            return {
                "documents": [
                    SimpleNamespace(id="second", score=0.91, meta={"chunk_id": "second"}),
                    SimpleNamespace(id="first", score=0.55, meta={"chunk_id": "first"}),
                ]
            }

    monkeypatch.setattr(retriever, "_to_rerank_documents", lambda items: [SimpleNamespace(id=item.chunk_id) for item in items])
    monkeypatch.setattr(retriever, "_get_reranker", lambda: FakeRanker())

    reranked = rerank_results(results, "test query", limit=2)
    assert [item.chunk_id for item in reranked] == ["second", "first"]
    assert reranked[0].metadata["rerank_score"] == 0.91
    assert reranked[0].score > reranked[1].score


def test_rerank_results_falls_back_to_fused_order_when_ranker_fails(monkeypatch):
    results = [
        SearchResult(
            chunk_id="first",
            score=0.09,
            title="Doc 1",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Section"],
            content="first",
            metadata={},
        ),
        SearchResult(
            chunk_id="second",
            score=0.08,
            title="Doc 2",
            document_version_id="ver-1",
            source_document_id="doc-2",
            pages=[1],
            section_path=["Section"],
            content="second",
            metadata={},
        ),
    ]

    monkeypatch.setattr(retriever, "_to_rerank_documents", lambda items: (_ for _ in ()).throw(RuntimeError("boom")))

    reranked = rerank_results(results, "test query", limit=2)
    assert [item.chunk_id for item in reranked] == ["first", "second"]


def test_resolve_rerank_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(retriever, "settings", SimpleNamespace(haystack_rerank_device="auto"))
    monkeypatch.setattr(retriever, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))

    class FakeComponentDevice:
        @staticmethod
        def from_str(value: str) -> str:
            return value

    import sys

    monkeypatch.setitem(sys.modules, "haystack.utils", SimpleNamespace(ComponentDevice=FakeComponentDevice))
    assert _resolve_rerank_device() == "cuda:0"


def test_resolve_rerank_device_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(retriever, "settings", SimpleNamespace(haystack_rerank_device="auto"))
    monkeypatch.setattr(retriever, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert _resolve_rerank_device() is None


def test_query_analysis_prefers_section_windows_for_command_flow_queries():
    analysis = analyze_query("What does the manual say about command timing and handshake flags?")
    assert "operational_flow" in analysis.query_types
    assert "section_window" in analysis.preferred_chunk_types


def test_query_analysis_handles_revision_history_queries():
    analysis = analyze_query("What is the latest revision history for LJ-X8000?")
    assert "revision_history" in analysis.query_types
    assert analysis.preferred_metadata_filters["version_signal"] == "true"


def test_query_analysis_marks_applies_to_field_questions_as_structured_lookup():
    analysis = analyze_query("What address 1041 applies to Model-120?")
    assert "structured_lookup" in analysis.query_types
    assert "table_record" in analysis.preferred_chunk_types
    assert "spec_record" in analysis.preferred_chunk_types


def test_query_analysis_marks_detection_applies_questions_as_structured_lookup():
    analysis = analyze_query("What detection applies to IV4-G120?")

    assert "structured_lookup" in analysis.query_types
    assert analysis.product_model == "IV4-G120"
    assert "comparison" not in analysis.query_types


def test_query_analysis_marks_data_number_applies_questions_as_structured_lookup():
    analysis = analyze_query("What 0068 data4 applies to VS Series Vision System with Built-in AI?")

    assert "structured_lookup" in analysis.query_types
    assert "table_record" in analysis.preferred_chunk_types
    assert "comparison" not in analysis.query_types


def test_query_analysis_marks_summary_applies_questions_as_structured_lookup():
    analysis = analyze_query("What summary average applies to XG-X Series?")

    assert "structured_lookup" in analysis.query_types
    assert "table_record" in analysis.preferred_chunk_types


def test_query_analysis_marks_specified_for_model_questions_as_spec_lookup():
    analysis = analyze_query("What average density is specified for CV-X482?")

    assert "spec_lookup" in analysis.query_types
    assert analysis.product_model == "CV-X482"


def test_query_analysis_does_not_treat_vs_series_as_comparison():
    analysis = analyze_query("What Display Settings value applies to VS Series Vision System?")

    assert "structured_lookup" in analysis.query_types
    assert "comparison" not in analysis.query_types
    assert analysis.product_family == "VS"


def test_query_analysis_marks_cause_and_correction_questions_as_structured_lookup():
    analysis = analyze_query("What causes EtherNet/IP output buffer is full, and how should it be corrected?")
    assert "structured_lookup" in analysis.query_types
    assert "troubleshooting" in analysis.query_types
    assert "table_record" in analysis.preferred_chunk_types


def test_query_analysis_treats_letter_number_series_as_product_family_not_error_code():
    analysis = analyze_query("When controlling image capture timing for X8000 Series, what detail should be used?")

    assert analysis.product_family == "X8000"
    assert analysis.error_code is None


def test_query_analysis_recognizes_models_with_numbered_prefixes():
    analysis = analyze_query("What PDO Source Sub Index value applies to IV4-G600CA?")

    assert analysis.product_model == "IV4-G600CA"
    assert analysis.error_code is None


def test_query_analysis_does_not_extract_model_suffix_as_error_code():
    analysis = analyze_query("What warning applies to IV4-G120 when reading warning status?")

    assert analysis.product_model == "IV4-G120"
    assert analysis.error_code is None


def test_table_search_route_skips_safety_procedure_questions():
    analysis = analyze_query(
        "When installing the controller for LJ-X8000, what warning or caution about controller mounting should be followed?"
    )

    assert analysis.safety_intent is True
    assert retriever._should_run_table_search(analysis) is False


def test_table_search_route_keeps_structured_value_questions():
    analysis = analyze_query("What Display Settings Green Lower Limit Value value applies to VS Series Vision System?")

    assert "structured_lookup" in analysis.query_types
    assert retriever._should_run_table_search(analysis) is True


def test_special_routes_do_not_duplicate_how_to_route_for_safety_questions():
    analysis = analyze_query(
        "When installing the controller for LJ-X8000, what warning or caution about controller mounting should be followed?"
    )

    routes = retriever._special_route_filters({"is_active": True}, analysis)
    chunk_type_sets = [set(route["chunk_type"]) for route in routes if "chunk_type" in route]

    assert {"warning_record"} not in chunk_type_sets
    assert chunk_type_sets.count({"warning_record", "procedure_record"}) == 1
    assert {"procedure_record", "section_window"} not in chunk_type_sets


def test_special_routes_include_action_text_for_warning_applies_questions():
    analysis = analyze_query(
        "What warning or caution about warning status for IV4-G120 applies when the number of remaining buffers is checked?"
    )

    routes = retriever._special_route_filters({"is_active": True}, analysis)
    chunk_type_sets = [set(route["chunk_type"]) for route in routes if "chunk_type" in route]

    assert {"warning_record", "procedure_record", "atomic_text"} in chunk_type_sets


def test_structured_lookup_special_routes_stay_table_focused():
    analysis = analyze_query("What Display Settings Green Lower Limit Value value applies to VS Series Vision System?")

    routes = retriever._special_route_filters({"is_active": True}, analysis)
    chunk_type_sets = [set(route["chunk_type"]) for route in routes if "chunk_type" in route]

    assert {"table_record", "spec_record", "section_window"} in chunk_type_sets
    assert {"procedure_record", "section_window"} not in chunk_type_sets
    assert all("procedure_record" not in chunk_types for chunk_types in chunk_type_sets)


def test_structured_lookup_uses_focused_vector_route_instead_of_duplicate_broad_routes():
    analysis = analyze_query(
        "What Vibration resistance Compliant with JIS B 3502 and IEC 61131-2 KV-NC32T value applies to SV2 Series?"
    )

    assert "structured_lookup" in analysis.query_types
    assert retriever._should_run_broad_vector_search(analysis) is False
    assert retriever._should_run_extra_table_vector_search(analysis) is True
    assert retriever._should_run_table_lexical_search(analysis) is False


def test_structured_lookup_with_product_family_skips_table_lexical_search():
    analysis = analyze_query("What Paste target not found. Code 101. Error Detail value applies to XG-X Series?")

    assert "structured_lookup" in analysis.query_types
    assert analysis.product_family == "XG-X"
    assert retriever._should_run_extra_table_vector_search(analysis) is True
    assert retriever._should_run_table_lexical_search(analysis) is False


def test_structured_lookup_without_explicit_identifier_keeps_table_lexical_search():
    analysis = analyze_query("What message profinetunit applies to User's Manual (3D mode)?")

    assert "structured_lookup" in analysis.query_types
    assert retriever._should_run_table_lexical_search(analysis) is True


def test_structured_lookup_skips_contextual_lexical_search_terms():
    analysis = analyze_query("What Display Settings Green Lower Limit Value value applies to VS Series Vision System?")

    assert retriever._lexical_context_terms(analysis.raw_query, analysis) == []


def test_table_lexical_content_terms_ignore_generic_value_and_keep_only_strong_fields():
    terms = retriever._lexical_table_terms(
        "What data7 M_DATA7 M_DATA9 Scaling target value applies to LJ: S8000 Series?",
        analyze_query("What data7 M_DATA7 M_DATA9 Scaling target value applies to LJ: S8000 Series?"),
    )

    assert retriever._lexical_table_content_terms(terms) == ["scaling"]
    symbol_terms = retriever._lexical_table_symbol_terms(terms)
    assert "data7" in symbol_terms
    assert "mdata9" in symbol_terms


def test_dedupe_results_keeps_distinct_chunks_in_same_section():
    analysis = analyze_query("What does LJ-X8000 say about trigger timing?")
    first = SearchResult(
        chunk_id="chunk-1",
        score=0.8,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Timing"],
        content="First content",
        metadata={"chunk_type": "atomic_text"},
    )
    second = SearchResult(
        chunk_id="chunk-2",
        score=0.7,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Timing"],
        content="Second content",
        metadata={"chunk_type": "atomic_text"},
    )
    deduped = retriever._dedupe_results([first, second], analysis)
    assert [item.chunk_id for item in deduped] == ["chunk-1", "chunk-2"]


def test_family_scoring_promotes_spec_chunks_for_spec_queries():
    analysis = QueryAnalysis(raw_query="structured lookup", query_types=["spec_lookup"])
    atomic = SearchResult(
        chunk_id="atomic",
        score=0.5,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="Voltage",
        metadata={"chunk_type": "atomic_text"},
    )
    spec = SearchResult(
        chunk_id="spec",
        score=0.48,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="Voltage: 24 V",
        metadata={"chunk_type": "spec_record"},
    )
    rescored = retriever._apply_family_scoring([atomic, spec], analysis, stage="family_scored")
    assert [item.chunk_id for item in rescored][:1] == ["spec"]


def test_family_scoring_uses_scalar_product_family_identifier():
    analysis = analyze_query("VS Series capture settings value upper limit")
    generic = SearchResult(
        chunk_id="generic",
        score=0.5,
        title="Capture Settings",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Capture Settings"],
        content="Capture Settings Value Upper Limit: 100",
        metadata={"chunk_type": "table_record", "product_family": "CV-X Series"},
    )
    family_match = SearchResult(
        chunk_id="family-match",
        score=0.49,
        title="Capture Settings",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[1],
        section_path=["Capture Settings"],
        content="Capture Settings Value Upper Limit: 88",
        metadata={"chunk_type": "table_record", "product_family": "VS Series Vision System"},
    )

    rescored = retriever._apply_family_scoring([generic, family_match], analysis, stage="family_scored")

    assert rescored[0].chunk_id == "family-match"


def test_family_scoring_strongly_uses_letter_number_product_identifier():
    analysis = analyze_query("X8000 Series image capture timing detail")
    generic = SearchResult(
        chunk_id="generic",
        score=0.5,
        title="Image Capture",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Image Capture"],
        content="The image capture timing detail is configured here.",
        metadata={"chunk_type": "atomic_text", "product_family": "XG-X Series"},
    )
    family_match = SearchResult(
        chunk_id="family-match",
        score=0.4,
        title="Image Capture",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[1],
        section_path=["Image Capture"],
        content="The image capture timing detail is configured here.",
        metadata={"chunk_type": "atomic_text", "product_family": "X8000 Series - 3D", "product_models": ["LJ-X8000"]},
    )

    rescored = retriever._apply_family_scoring([generic, family_match], analysis, stage="family_scored")

    assert rescored[0].chunk_id == "family-match"


def test_contextual_lexical_search_scores_local_section_context(monkeypatch):
    analysis = analyze_query("When image capture timing for X8000 Series, what Ethernet/IP memory monitor detail should be used?")

    def fake_fetch_all(query, params):
        assert "local_rerank_context" in query
        assert "order by" in query.lower()
        assert params[0] == ["procedure_record", "atomic_text", "section_window"]
        return [
            {
                "id": "procedure",
                "document_version_id": "ver-1",
                "source_document_id": "doc-1",
                "title": "Setup Guide",
                "section_path_text": "Checking the Connection",
                "page_from": 13,
                "page_to": 13,
                "content": "Procedure step 2: Controlling the Image Capture Timing",
                "chunk_type": "procedure_record",
                "metadata_json": {
                    "product_family": "X8000 Series - 3D",
                    "product_models": ["LJ-X8000"],
                    "local_rerank_context": "Use the EtherNet/IP memory monitor of the LJ-X8000 to check the connection.",
                },
                "priority_score": 1.0,
            },
            {
                "id": "generic",
                "document_version_id": "ver-2",
                "source_document_id": "doc-2",
                "title": "Manual",
                "section_path_text": "Capture",
                "page_from": 99,
                "page_to": 99,
                "content": "Image capture timing settings",
                "chunk_type": "atomic_text",
                "metadata_json": {"product_family": "XG-X Series"},
                "priority_score": 1.0,
            },
        ]

    monkeypatch.setattr(retriever, "fetch_all", fake_fetch_all)

    results = retriever.run_contextual_lexical_search(
        "When image capture timing for X8000 Series, what Ethernet/IP memory monitor detail should be used?",
        ["manuals_vendor_keyence"],
        {"is_active": True},
        analysis,
    )

    assert results[0].chunk_id == "procedure"


def test_contextual_lexical_search_promotes_product_family_context(monkeypatch):
    analysis = analyze_query(
        "When Controlling the Image Capture Timing for X8000 Series - 3D, what related operations performed speed detail should be used?"
    )

    def fake_fetch_all(query, params):
        assert "local_rerank_context" in query
        assert "order by" in query.lower()
        assert "metadata_json::text ilike" in query
        assert any(param == "%x8000%" for param in params)
        return [
            {
                "id": "expected-section",
                "document_version_id": "ver-1",
                "source_document_id": "doc-1",
                "title": "Setup Guide",
                "section_path_text": "1.2 Checking the Connection",
                "page_from": 10,
                "page_to": 11,
                "content": "Controlling the Image Capture Timing. Operations are performed at high speed with images.",
                "chunk_type": "section_window",
                "metadata_json": {
                    "product_family": "X8000 Series - 3D",
                    "product_models": ["LJ-X8000"],
                    "local_rerank_context": "Use discreet I/O on the terminal block to apply the triggers.",
                    "section_path": ["1.2 Checking the Connection"],
                },
                "priority_score": 0.0,
            },
            {
                "id": "generic-section",
                "document_version_id": "ver-2",
                "source_document_id": "doc-2",
                "title": "Manual",
                "section_path_text": "3D Capture",
                "page_from": 20,
                "page_to": 20,
                "content": "Controlling image capture timing operations are performed at high speed with images.",
                "chunk_type": "section_window",
                "metadata_json": {
                    "product_family": "XG-X Series",
                    "local_rerank_context": "Specify the shutter speed for the 3D image to be captured.",
                    "section_path": ["3D Capture"],
                },
                "priority_score": 0.0,
            },
        ]

    monkeypatch.setattr(retriever, "fetch_all", fake_fetch_all)

    results = retriever.run_contextual_lexical_search(analysis.raw_query, ["corpus-1"], {"is_active": True}, analysis)

    assert results[0].chunk_id == "expected-section"


def test_family_scoring_demotes_spec_chunks_for_general_prose_queries():
    analysis = analyze_query("Where does the manual discuss command completion and successful execution?")
    prose = SearchResult(
        chunk_id="prose",
        score=0.5,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Command Execution"],
        content="When the command is successful, the completion flag remains off.",
        metadata={"chunk_type": "atomic_text"},
    )
    spec = SearchResult(
        chunk_id="spec",
        score=0.52,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specifications"],
        content="Successful command bit: OFF",
        metadata={"chunk_type": "spec_record"},
    )
    rescored = retriever._apply_family_scoring([spec, prose], analysis, stage="family_scored")
    assert rescored[0].chunk_id == "prose"


def test_query_alignment_promotes_candidate_with_better_query_term_coverage():
    analysis = analyze_query("Where does the manual discuss command completion and successful execution?")
    partial = SearchResult(
        chunk_id="partial",
        score=0.5,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Command"],
        content="The command remains off.",
        metadata={"chunk_type": "atomic_text"},
    )
    aligned = SearchResult(
        chunk_id="aligned",
        score=0.49,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Command Execution"],
        content="When the command is successful, the completion signal remains off after execution.",
        metadata={"chunk_type": "atomic_text"},
    )
    rescored = retriever._apply_query_alignment([partial, aligned], analysis, stage="query_aligned")
    assert rescored[0].chunk_id == "aligned"


def test_query_alignment_normalizes_hyphenated_axis_terms():
    analysis = analyze_query("z axis repeatability")
    partial = SearchResult(
        chunk_id="partial",
        score=0.5,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="Z axis measurement range",
        metadata={"chunk_type": "atomic_text"},
    )
    aligned = SearchResult(
        chunk_id="aligned",
        score=0.49,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="Z-axis repeatability is listed in the precision table.",
        metadata={"chunk_type": "atomic_text"},
    )
    rescored = retriever._apply_query_alignment([partial, aligned], analysis, stage="query_aligned")
    assert rescored[0].chunk_id == "aligned"


def test_query_alignment_does_not_match_identifier_fragments():
    analysis = analyze_query("LJ-X8080 z axis repeatability")
    wrong_identifier = SearchResult(
        chunk_id="wrong",
        score=0.5,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="LJ-X8020 Z-axis repeatability is listed.",
        metadata={"chunk_type": "table_record"},
    )
    aligned = SearchResult(
        chunk_id="aligned",
        score=0.49,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-2",
        pages=[1],
        section_path=["Specs"],
        content="LJ-X8080 Z-axis repeatability is listed.",
        metadata={"chunk_type": "table_record"},
    )
    rescored = retriever._apply_query_alignment([wrong_identifier, aligned], analysis, stage="query_aligned")
    assert rescored[0].chunk_id == "aligned"


def test_query_alignment_promotes_generic_mixed_vendor_spec_lookup():
    analysis = analyze_query("AX-1200 pressure repeatability")
    wrong_document = SearchResult(
        chunk_id="wrong",
        score=0.5,
        title="QN-42A Pressure Sensor",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specifications"],
        content="QN-42A pressure repeatability is listed as 0.04 kPa.",
        metadata={"chunk_type": "table_record"},
    )
    aligned = SearchResult(
        chunk_id="aligned",
        score=0.49,
        title="AX-1200 Pressure Controller",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[3],
        section_path=["Specifications"],
        content="AX-1200 pressure repeatability is listed as 0.02 kPa.",
        metadata={"chunk_type": "table_record"},
    )

    rescored = retriever._apply_query_alignment([wrong_document, aligned], analysis, stage="query_aligned")

    assert rescored[0].chunk_id == "aligned"


def test_query_alignment_promotes_structured_lookup_exact_subject():
    analysis = analyze_query("What message profinetunit applies to User's Manual (3D mode)?")
    generic_message = SearchResult(
        chunk_id="generic",
        score=0.5,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Errors"],
        content="Message: Failed to register the pattern. Cause: The pattern region extends beyond the image.",
        metadata={"chunk_type": "table_record"},
    )
    exact_subject = SearchResult(
        chunk_id="exact",
        score=0.45,
        title="Manual",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[2],
        section_path=["Errors"],
        content="Message: The PROFINETunit cannot be recognized. Cause: The PROFINET unit is not recognized.",
        metadata={"chunk_type": "table_record"},
    )

    rescored = retriever._apply_query_alignment([generic_message, exact_subject], analysis, stage="query_aligned")

    assert rescored[0].chunk_id == "exact"


def test_query_alignment_prefers_structured_lookup_field_with_subject():
    analysis = analyze_query("What message profinetunit applies to User's Manual (3D mode)?")
    subject_only = SearchResult(
        chunk_id="trigger-mode",
        score=0.5,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Trigger Mode"],
        content="Column headers: Trigger Mode; Row headers: PROFINET Unit; Cell value: cyclic communication trigger.",
        metadata={
            "chunk_type": "table_record",
            "rerank_document": "Column headers: Trigger Mode; Row headers: PROFINET Unit; Cell value: cyclic communication trigger.",
        },
    )
    message_row = SearchResult(
        chunk_id="message-row",
        score=0.48,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[2],
        section_path=["Error Messages"],
        content="Message: The PROFINETunit cannot be recognized. Cause: The PROFINET unit is not recognized.",
        metadata={
            "chunk_type": "table_record",
            "rerank_document": "Message: The PROFINETunit cannot be recognized. Cause: The PROFINET unit is not recognized.",
        },
    )

    rescored = retriever._apply_query_alignment([subject_only, message_row], analysis, stage="query_aligned")

    assert rescored[0].chunk_id == "message-row"


def test_query_alignment_keeps_generic_procedure_search_ahead_of_unrelated_spec_table():
    analysis = analyze_query("configure ethernet scanner steps")
    unrelated_spec = SearchResult(
        chunk_id="spec",
        score=0.5,
        title="Scanner Specifications",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specifications"],
        content="Ethernet scanner supply voltage: 24 VDC.",
        metadata={"chunk_type": "table_record"},
    )
    procedure = SearchResult(
        chunk_id="procedure",
        score=0.49,
        title="Scanner Setup Guide",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[5],
        section_path=["Setup"],
        content="Configure the Ethernet scanner by opening network setup and following the steps.",
        metadata={"chunk_type": "procedure_record"},
    )

    rescored = retriever._apply_query_alignment([unrelated_spec, procedure], analysis, stage="query_aligned")

    assert rescored[0].chunk_id == "procedure"


def test_family_selection_keeps_preferred_family_for_general_queries():
    analysis = analyze_query("Where does the manual discuss command completion and successful execution?")
    candidates = [
        SearchResult(
            chunk_id="spec",
            score=0.9,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Specifications"],
            content="Completion output",
            metadata={"family_bucket": "spec", "chunk_type": "spec_record"},
        ),
        SearchResult(
            chunk_id="prose",
            score=0.7,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Command Execution"],
            content="When the command is successful, the completion signal remains off.",
            metadata={"family_bucket": "prose", "chunk_type": "atomic_text"},
        ),
    ]
    selected = retriever._select_family_candidates(candidates, analysis, limit=5)
    assert "prose" in [item.chunk_id for item in selected]


def test_dense_search_returns_empty_on_vector_dimension_mismatch(monkeypatch):
    store = QdrantStore()
    monkeypatch.setattr("manuals_rag_retrieval.qdrant_store.embed_dense", lambda texts: [[0.1, 0.2, 0.3, 0.4]])

    class FailingClient:
        def search(self, **_kwargs):
            raise UnexpectedResponse(status_code=400, reason_phrase="Bad Request", content=b"Vector dimension error: expected dim: 768, got 1024", headers={})

    store.client = FailingClient()
    results = store.search_dense("manuals_eval_legacy", "test query", {"is_active": True}, limit=5)
    assert results == []


def test_enrich_candidates_for_rerank_adds_grouped_procedure_context(monkeypatch):
    result = SearchResult(
        chunk_id="step-1",
        score=0.7,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[3],
        section_path=["Procedure"],
        content="Step 1: Connect cable",
        metadata={"chunk_type": "procedure_record", "content_for_rerank": "Step 1: Connect cable"},
    )
    monkeypatch.setattr(
        retriever,
        "fetch_all",
        lambda *_args, **_kwargs: [
            {
                "document_version_id": "ver-1",
                "section_path_text": "Procedure",
                "chunk_type": "procedure_record",
                "chunk_level": 2,
                "page_from": 3,
                "page_to": 4,
                "content": "Step 1: Connect cable\nStep 2: Enable power",
                "metadata_json": {"grouped_procedure": True},
            },
            {
                "document_version_id": "ver-1",
                "section_path_text": "Procedure",
                "chunk_type": "parent_section",
                "chunk_level": 3,
                "page_from": 3,
                "page_to": 5,
                "content": "Full procedure block",
                "metadata_json": {},
            },
        ],
    )
    enriched = retriever.enrich_candidates_for_rerank([result], analyze_query("How do I connect the cable?"), limit=5)
    rerank_document = enriched[0].metadata["rerank_document"]
    assert "Enable power" in rerank_document
    assert "Full procedure block" in rerank_document


def test_assemble_context_uses_nearest_table_row_group_for_table_cells(monkeypatch):
    result = SearchResult(
        chunk_id="cell-action",
        score=0.7,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[10],
        section_path=["Troubleshooting"],
        content="Column headers: Corrective Action; Cell value: Check the cable.",
        metadata={"chunk_type": "table_record", "table_cell": True},
    )
    monkeypatch.setattr(
        retriever,
        "fetch_all",
        lambda *_args, **_kwargs: [
            {
                "document_version_id": "ver-1",
                "section_path_text": "Troubleshooting",
                "chunk_type": "table_record",
                "chunk_level": 1,
                "page_from": 10,
                "page_to": 10,
                "content": "Error Message: Link error; Cause: Cable disconnected; Corrective Action: Check the cable.",
                "metadata_json": {"table_row_group": True},
            },
            {
                "document_version_id": "ver-1",
                "section_path_text": "Troubleshooting",
                "chunk_type": "section_window",
                "chunk_level": 2,
                "page_from": 10,
                "page_to": 10,
                "content": "Troubleshooting section overview",
                "metadata_json": {},
            },
        ],
    )

    assembled = retriever.assemble_context([result], limit=1)

    assert assembled[0].metadata["context_window"].startswith("Error Message: Link error")
    assert assembled[0].metadata["table_row_group_context"] == assembled[0].metadata["context_window"]


def test_semantic_completeness_penalizes_heading_like_fragments():
    result = SearchResult(
        chunk_id="heading",
        score=0.6,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Section"],
        content="External Input/Output Settings",
        metadata={"chunk_type": "atomic_text"},
    )
    annotated = retriever._annotate_completeness([result])[0]
    assert annotated.metadata["is_heading_like"] is True
    assert annotated.metadata["semantic_completeness_score"] < 0


def test_select_family_candidates_prefers_procedure_family_for_how_to_queries():
    analysis = analyze_query("How do I configure the communication settings?")
    procedure = SearchResult(
        chunk_id="proc",
        score=0.6,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[2],
        section_path=["Setup"],
        content="Step 1: Open Settings",
        metadata={"chunk_type": "procedure_record", "family_bucket": "procedure"},
    )
    section = SearchResult(
        chunk_id="ctx",
        score=0.7,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[2],
        section_path=["Setup"],
        content="Communication Settings",
        metadata={"chunk_type": "section_window", "family_bucket": "context"},
    )
    chosen = retriever._select_family_candidates([section, procedure], analysis, limit=4)
    assert chosen[0].chunk_id == "proc"


def test_select_family_candidates_prefers_spec_family_for_spec_lookup():
    analysis = analyze_query("What voltage specification is listed for the module?")
    prose = SearchResult(
        chunk_id="prose",
        score=0.7,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="Voltage",
        metadata={"chunk_type": "atomic_text", "family_bucket": "prose"},
    )
    spec = SearchResult(
        chunk_id="spec",
        score=0.6,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="Voltage: 24 V",
        metadata={"chunk_type": "spec_record", "family_bucket": "spec"},
    )
    chosen = retriever._select_family_candidates([prose, spec], analysis, limit=4)
    assert chosen[0].chunk_id == "spec"
    assert "spec" in [item.chunk_id for item in chosen]


def test_select_family_candidates_prefers_table_family_for_structured_lookup():
    analysis = analyze_query("What error message value applies to Model-120?")
    prose = SearchResult(
        chunk_id="prose",
        score=0.8,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Notes"],
        content="An error occurs if the input is invalid.",
        metadata={"chunk_type": "atomic_text", "family_bucket": "prose"},
    )
    table = SearchResult(
        chunk_id="table",
        score=0.6,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[5],
        section_path=["Error Messages"],
        content="Column headers: Error Message; Cell value: Library conversion was interrupted.",
        metadata={"chunk_type": "table_record", "family_bucket": "table"},
    )

    chosen = retriever._select_family_candidates([prose, table], analysis, limit=4)

    assert chosen[0].chunk_id == "table"
    assert "prose" not in [item.chunk_id for item in chosen]


def test_structured_lookup_family_scoring_demotes_table_header_chunks():
    analysis = analyze_query("What address applies to IV4-G120?")
    header = SearchResult(
        chunk_id="header",
        score=0.5,
        title="IV4 Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Data Allocation"],
        content="Table header: IV4-G120; Header role: column; Row: 0; Column: 2",
        metadata={"chunk_type": "table_record", "table_header": True, "product_model": "IV4-G120"},
    )
    row = SearchResult(
        chunk_id="row",
        score=0.47,
        title="IV4 Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Data Allocation"],
        content="Address: 6 to 7 (WORD); Stored data: 1041",
        metadata={"chunk_type": "table_record", "table_key_value": True, "product_model": "IV4-G120"},
    )

    rescored = retriever._apply_family_scoring([header, row], analysis, stage="family_scored")

    assert rescored[0].chunk_id == "row"


def test_family_scoring_promotes_exact_product_model_over_similar_model():
    analysis = analyze_query("What PDO Source Sub Index value applies to IV4-G600CA?")
    expected = SearchResult(
        chunk_id="expected",
        score=0.5,
        title="IV4 Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["PDO"],
        content="Column headers: Source Sub Index*1 (HEX); Cell value: 02h+(M-1)xAh",
        metadata={"chunk_type": "table_record", "product_model": "IV4-G600CA"},
    )
    similar = SearchResult(
        chunk_id="similar",
        score=0.58,
        title="IV4 Manual",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[1],
        section_path=["PDO"],
        content="Column headers: Source Sub Index *1 (HEX); Cell value: 02h + (L-1) x 28h",
        metadata={"chunk_type": "table_record", "product_model": "IV4-G120"},
    )

    rescored = retriever._apply_family_scoring([similar, expected], analysis, stage="family_scored")

    assert rescored[0].chunk_id == "expected"


def test_safety_action_terms_boost_specific_step_evidence_over_broad_context():
    analysis = analyze_query(
        "What warning or caution about warning status applies when obtaining the master number, set total status condition?"
    )
    step = SearchResult(
        chunk_id="step",
        score=0.5,
        title="IV4 Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Status"],
        content="When obtaining the master number, set Total status condition as shown below.",
        metadata={"chunk_type": "atomic_text"},
    )
    context = SearchResult(
        chunk_id="context",
        score=0.5,
        title="IV4 Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Status"],
        content="Warning status can be cleared with a warning clear request.",
        metadata={"chunk_type": "section_window"},
    )

    assert retriever._query_alignment_score(step, analysis) > retriever._query_alignment_score(context, analysis)


def test_run_dense_search_queries_each_corpus_and_returns_dense_hits():
    calls: list[tuple[str, str, dict[str, object], int]] = []

    class FakeStore:
        def search_dense(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            calls.append((corpus_id, query, filters, limit))
            return [
                SearchResult(
                    chunk_id=f"dense-{corpus_id}",
                    score=0.7,
                    title=f"Doc {corpus_id}",
                    document_version_id="ver-1",
                    source_document_id=f"src-{corpus_id}",
                    pages=[1],
                    section_path=["Specs"],
                    content="Voltage: 24 V",
                    metadata={"chunk_type": "spec_record", "retrieval_stage": "dense"},
                )
            ]

    results = run_dense_search(FakeStore(), "voltage spec", ["c1", "c2"], {"product_model": "LJ-X8000"}, limit=5)

    assert [item.chunk_id for item in results] == ["dense-c1", "dense-c2"]
    assert calls == [
        ("c1", "voltage spec", {"product_model": "LJ-X8000"}, 5),
        ("c2", "voltage spec", {"product_model": "LJ-X8000"}, 5),
    ]


def test_run_dense_search_skips_failed_corpus_and_keeps_other_hits():
    class FakeStore:
        def search_dense(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            if corpus_id == "bad":
                raise RuntimeError("embedding service failed")
            return [
                SearchResult(
                    chunk_id=f"dense-{corpus_id}",
                    score=0.7,
                    title=f"Doc {corpus_id}",
                    document_version_id="ver-1",
                    source_document_id=f"src-{corpus_id}",
                    pages=[1],
                    section_path=["Specs"],
                    content="Voltage: 24 V",
                    metadata={"chunk_type": "spec_record", "retrieval_stage": "dense"},
                )
            ]

    results = run_dense_search(FakeStore(), "voltage spec", ["bad", "good"], {}, limit=5)

    assert [item.chunk_id for item in results] == ["dense-good"]


def test_run_sparse_search_queries_each_corpus_and_returns_sparse_hits():
    calls: list[tuple[str, str, dict[str, object], int]] = []

    class FakeStore:
        def search_sparse(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            calls.append((corpus_id, query, filters, limit))
            return [
                SearchResult(
                    chunk_id=f"sparse-{corpus_id}",
                    score=0.65,
                    title=f"Doc {corpus_id}",
                    document_version_id="ver-1",
                    source_document_id=f"src-{corpus_id}",
                    pages=[2],
                    section_path=["Timing"],
                    content="Command timing and handshake flags",
                    metadata={"chunk_type": "atomic_text", "retrieval_stage": "sparse"},
                )
            ]

    results = run_sparse_search(FakeStore(), "handshake flags", ["c1", "c2"], {"document_kind": "manual"}, limit=4)

    assert [item.chunk_id for item in results] == ["sparse-c1", "sparse-c2"]
    assert calls == [
        ("c1", "handshake flags", {"document_kind": "manual"}, 4),
        ("c2", "handshake flags", {"document_kind": "manual"}, 4),
    ]


def test_run_table_search_queries_table_records_with_dense_and_sparse():
    calls: list[tuple[str, str, dict[str, object], int]] = []

    class FakeStore:
        def search_dense(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            calls.append((f"dense-{corpus_id}", query, filters, limit))
            return [
                SearchResult(
                    chunk_id=f"dense-table-{corpus_id}",
                    score=0.7,
                    title=f"Doc {corpus_id}",
                    document_version_id="ver-1",
                    source_document_id=f"src-{corpus_id}",
                    pages=[1],
                    section_path=["Specs"],
                    content="Column headers: Model; Row headers: Property; Cell value: 1",
                    metadata={"chunk_type": "table_record"},
                )
            ]

        def search_sparse(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            calls.append((f"sparse-{corpus_id}", query, filters, limit))
            return [
                SearchResult(
                    chunk_id=f"sparse-table-{corpus_id}",
                    score=0.65,
                    title=f"Doc {corpus_id}",
                    document_version_id="ver-1",
                    source_document_id=f"src-{corpus_id}",
                    pages=[1],
                    section_path=["Specs"],
                    content="Column headers: Model; Row headers: Property; Cell value: 1",
                    metadata={"chunk_type": "table_record"},
                )
            ]

        @staticmethod
        def fuse_rrf(result_sets: list[list[SearchResult]], *, limit: int, k: int = 60) -> list[SearchResult]:
            return QdrantStore.fuse_rrf(result_sets, limit=limit, k=k)

    results = run_table_search(FakeStore(), "any query", ["c1", "c2"], {"is_active": True}, limit=3)

    assert [item.chunk_id for item in results] == ["dense-table-c1", "sparse-table-c1", "dense-table-c2", "sparse-table-c2"]
    assert calls == [
        ("dense-c1", "any query", {"is_active": True, "chunk_type": ["table_record"]}, 3),
        ("sparse-c1", "any query", {"is_active": True, "chunk_type": ["table_record"]}, 3),
        ("dense-c2", "any query", {"is_active": True, "chunk_type": ["table_record"]}, 3),
        ("sparse-c2", "any query", {"is_active": True, "chunk_type": ["table_record"]}, 3),
    ]


def test_run_table_search_uses_sparse_when_dense_fails():
    class FakeStore:
        def search_dense(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            raise RuntimeError("embedding service failed")

        def search_sparse(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            return [
                SearchResult(
                    chunk_id=f"sparse-table-{corpus_id}",
                    score=0.65,
                    title=f"Doc {corpus_id}",
                    document_version_id="ver-1",
                    source_document_id=f"src-{corpus_id}",
                    pages=[1],
                    section_path=["Specs"],
                    content="Column headers: Model; Row headers: Property; Cell value: 1",
                    metadata={"chunk_type": "table_record"},
                )
            ]

        @staticmethod
        def fuse_rrf(result_sets: list[list[SearchResult]], *, limit: int, k: int = 60) -> list[SearchResult]:
            return QdrantStore.fuse_rrf(result_sets, limit=limit, k=k)

    results = run_table_search(FakeStore(), "any query", ["c1"], {"is_active": True}, limit=3)

    assert [item.chunk_id for item in results] == ["sparse-table-c1"]


def test_metadata_document_selection_adds_document_scope_before_chunk_search():
    calls: list[tuple[str, str, dict[str, object], int]] = []

    class FakeStore:
        def search_document_metadata(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
            calls.append((corpus_id, query, filters, limit))
            return [
                {"source_document_id": f"doc-{corpus_id}", "score": 0.9, "payload": {"title": f"Doc {corpus_id}"}},
                {"source_document_id": "duplicate", "score": 0.8, "payload": {"title": "Duplicate"}},
            ]

    filters, hits = retriever.select_documents_from_metadata(
        FakeStore(),
        "z axis repeatability",
        ["c1", "c2"],
        {"is_active": True},
        limit=3,
    )

    assert filters == {"is_active": True, "source_document_id": ["doc-c1", "doc-c2", "duplicate"]}
    assert [hit["source_document_id"] for hit in hits] == ["doc-c1", "doc-c2", "duplicate"]
    assert calls == [
        ("c1", "z axis repeatability", {"is_active": True}, 3),
        ("c2", "z axis repeatability", {"is_active": True}, 3),
    ]


def test_metadata_document_selection_respects_explicit_document_scope():
    class FakeStore:
        def search_document_metadata(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
            raise AssertionError("metadata selection should be skipped")

    original_filters = {"is_active": True, "source_document_id": "doc-1"}
    filters, hits = retriever.select_documents_from_metadata(FakeStore(), "query", ["c1"], original_filters)

    assert filters == original_filters
    assert hits == []


def test_metadata_document_selection_falls_back_when_metadata_index_has_no_hits():
    class FakeStore:
        def search_document_metadata(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
            return []

    original_filters = {"is_active": True}
    filters, hits = retriever.select_documents_from_metadata(FakeStore(), "query", ["c1"], original_filters)

    assert filters == original_filters
    assert hits == []


def test_document_metadata_index_aggregates_generic_chunk_metadata_signals():
    document = {"metadata_json": {"product_models": ["Series-100"]}}
    chunk_rows = [
        {
            "metadata_json": {
                "product_models": ["Series-100", "Model-101"],
                "table_column_headers": ["Model-101"],
                "table_row_headers": ["Repeatability"],
                "section_path": ["Specifications"],
                "local_rerank_context": "raw chunk text is not a configured document metadata signal",
            }
        }
    ]

    enriched = enrich_document_metadata_with_chunk_signals(document, chunk_rows)

    signals = enriched["metadata_json"]["chunk_metadata_signals"]
    assert signals["product_models"] == ["Series-100", "Model-101"]
    assert signals["table_column_headers"] == ["Model-101"]
    assert signals["table_row_headers"] == ["Repeatability"]
    assert signals["section_path"] == ["Specifications"]
    assert "local_rerank_context" not in signals


def test_qdrant_store_search_sparse_uses_native_sparse_vector(monkeypatch):
    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def search(
            self,
            *,
            collection_name: str,
            query_vector: object,
            query_filter: dict[str, object] | None,
            limit: int,
            with_payload: bool,
        ) -> list[SimpleNamespace]:
            assert collection_name == "manuals_corpus-1"
            assert getattr(query_vector, "name") == "sparse"
            assert query_filter == {"must": [{"key": "document_kind", "match": {"value": "manual"}}]}
            assert limit == 2
            assert with_payload is True
            return [
                SimpleNamespace(
                    id="identity",
                    score=0.42,
                    payload={
                        "title": "CA-EN100U Datasheet",
                        "document_version_id": "ver-1",
                        "source_document_id": "doc-1",
                        "page_from": 1,
                        "page_to": 1,
                        "section_path": ["Overview"],
                        "content": "CA-EN100U encoder relay unit",
                        "chunk_type": "datasheet_record",
                        "priority_score": 0.0,
                        "document_kind": "manual",
                    },
                )
            ]

        def scroll(self, **_kwargs: object) -> tuple[list[object], None]:
            raise AssertionError("native sparse search should not scroll payloads")

    monkeypatch.setattr("manuals_rag_retrieval.qdrant_store.QdrantClient", FakeClient)
    store = QdrantStore()
    results = store.search_sparse("corpus-1", "What product is the CA-EN100U?", {"document_kind": "manual"}, limit=2)
    assert [item.chunk_id for item in results] == ["identity"]


def test_qdrant_store_search_sparse_falls_back_to_bm25_ranking(monkeypatch):
    class FakePoint:
        def __init__(self, point_id: str, payload: dict[str, object]) -> None:
            self.id = point_id
            self.payload = payload

    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def search(self, **_kwargs: object) -> list[object]:
            raise RuntimeError("sparse vector unavailable")

        def scroll(
            self,
            *,
            collection_name: str,
            scroll_filter: dict[str, object] | None,
            limit: int,
            offset: object,
            with_payload: bool,
            with_vectors: bool,
        ) -> tuple[list[FakePoint], None]:
            assert collection_name == "manuals_corpus-1"
            assert scroll_filter == {"must": [{"key": "document_kind", "match": {"value": "manual"}}]}
            return (
                [
                    FakePoint(
                        "identity",
                        {
                            "title": "CA-EN100U Datasheet",
                            "document_version_id": "ver-1",
                            "source_document_id": "doc-1",
                            "page_from": 1,
                            "page_to": 1,
                            "section_path": ["Overview"],
                            "content": "CA-EN100U encoder relay unit",
                            "chunk_type": "datasheet_record",
                            "priority_score": 0.0,
                            "document_kind": "manual",
                        },
                    ),
                    FakePoint(
                        "contact",
                        {
                            "title": "KEYENCE AMERICA",
                            "document_version_id": "ver-1",
                            "source_document_id": "doc-1",
                            "page_from": 1,
                            "page_to": 1,
                            "section_path": ["Contact"],
                            "content": "https://www.keyence.com Phone 1-888-539-3623",
                            "chunk_type": "atomic_text",
                            "priority_score": 0.0,
                            "document_kind": "manual",
                        },
                    ),
                ],
                None,
            )

    monkeypatch.setattr("manuals_rag_retrieval.qdrant_store.QdrantClient", FakeClient)
    store = QdrantStore()
    results = store.search_sparse("corpus-1", "What product is the CA-EN100U?", {"document_kind": "manual"}, limit=2)
    assert [item.chunk_id for item in results] == ["identity"]


def test_qdrant_store_search_sparse_reraises_query_timeout(monkeypatch):
    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def search(self, **_kwargs: object) -> list[object]:
            raise RuntimeError("Search exceeded per-query timeout of 8 seconds.")

        def scroll(self, **_kwargs: object) -> tuple[list[object], None]:
            raise AssertionError("timeout exceptions should not trigger fallback scrolling")

    monkeypatch.setattr("manuals_rag_retrieval.qdrant_store.QdrantClient", FakeClient)
    store = QdrantStore()

    with pytest.raises(RuntimeError, match="Search exceeded per-query timeout"):
        store.search_sparse("corpus-1", "What product is the CA-EN100U?", {"document_kind": "manual"}, limit=2)


def test_qdrant_payload_matching_supports_list_metadata_filters(monkeypatch):
    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

    monkeypatch.setattr("manuals_rag_retrieval.qdrant_store.QdrantClient", FakeClient)
    store = QdrantStore()
    payload = {
        "product_model": "LJ-X8000",
        "product_models": ["LJ-X8000", "LJ-X8080"],
        "product_families": ["LJ", "LJ-X"],
        "part_numbers": ["OP-88310"],
    }

    assert store._payload_matches(payload, {"product_models": "LJ-X8080"})
    assert store._payload_matches(payload, {"product_model": "LJ-X8080"})
    assert store._payload_matches(payload, {"product_families": "LJ-X", "part_numbers": "OP-88310"})
    assert not store._payload_matches(payload, {"product_models": "KV-8000"})


def test_qdrant_store_delete_document_chunks_uses_document_filters(monkeypatch):
    deleted: dict[str, object] = {}

    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def collection_exists(self, name: str) -> bool:
            return name == "manuals_corpus-1"

        def delete(self, *, collection_name: str, points_selector: object, wait: bool) -> None:
            deleted["collection_name"] = collection_name
            deleted["points_selector"] = points_selector
            deleted["wait"] = wait

    monkeypatch.setattr("manuals_rag_retrieval.qdrant_store.QdrantClient", FakeClient)
    store = QdrantStore()
    store.delete_document_chunks("corpus-1", source_document_id="doc-1", document_version_id="ver-1")
    assert deleted["collection_name"] == "manuals_corpus-1"
    assert deleted["wait"] is True
    selector = deleted["points_selector"]
    assert [condition.key for condition in selector.filter.must] == ["source_document_id", "document_version_id"]


def test_fuse_results_combines_dense_and_sparse_candidates_via_rrf():
    dense = [
        SearchResult(
            chunk_id="shared",
            score=0.9,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Specs"],
            content="Voltage: 24 V",
            metadata={"chunk_type": "spec_record"},
        ),
        SearchResult(
            chunk_id="dense-only",
            score=0.8,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-2",
            pages=[1],
            section_path=["Specs"],
            content="Current: 1 A",
            metadata={"chunk_type": "spec_record"},
        ),
    ]
    sparse = [
        SearchResult(
            chunk_id="shared",
            score=0.7,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Specs"],
            content="Voltage: 24 V",
            metadata={"chunk_type": "spec_record"},
        ),
        SearchResult(
            chunk_id="sparse-only",
            score=0.6,
            title="Doc",
            document_version_id="ver-1",
            source_document_id="doc-3",
            pages=[2],
            section_path=["Timing"],
            content="Handshake flags",
            metadata={"chunk_type": "atomic_text"},
        ),
    ]

    class FakeStore:
        @staticmethod
        def fuse_rrf(result_sets: list[list[SearchResult]], *, limit: int, k: int = 60) -> list[SearchResult]:
            return QdrantStore.fuse_rrf(result_sets, limit=limit, k=k)

    fused = fuse_results(FakeStore(), [dense, sparse], limit=3)

    assert [item.chunk_id for item in fused] == ["shared", "dense-only", "sparse-only"]


def test_rerank_results_reorders_fused_candidates_using_rerank_documents(monkeypatch):
    results = [
        SearchResult(
            chunk_id="dense-best",
            score=0.09,
            title="Doc 1",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Specs"],
            content="Voltage",
            metadata={"rerank_document": "Voltage heading only"},
        ),
        SearchResult(
            chunk_id="sparse-best",
            score=0.08,
            title="Doc 2",
            document_version_id="ver-1",
            source_document_id="doc-2",
            pages=[2],
            section_path=["Specs"],
            content="Voltage: 24 V",
            metadata={"rerank_document": "Voltage: 24 V nominal input specification"},
        ),
    ]

    class FakeRanker:
        def run(self, query: str, documents: list[object]) -> dict[str, list[object]]:
            assert query == "What is the voltage?"
            assert [document.id for document in documents] == ["dense-best", "sparse-best"]
            return {
                "documents": [
                    SimpleNamespace(id="sparse-best", score=0.97, meta={"chunk_id": "sparse-best"}),
                    SimpleNamespace(id="dense-best", score=0.31, meta={"chunk_id": "dense-best"}),
                ]
            }

    monkeypatch.setattr(
        retriever,
        "_to_rerank_documents",
        lambda items: [SimpleNamespace(id=item.chunk_id, content=item.metadata.get("rerank_document")) for item in items],
    )
    monkeypatch.setattr(retriever, "_get_reranker", lambda: FakeRanker())

    reranked = rerank_results(results, "What is the voltage?", limit=2)

    assert [item.chunk_id for item in reranked] == ["sparse-best", "dense-best"]
    assert reranked[0].metadata["rerank_score"] == 0.97


def test_rerank_results_keeps_query_aligned_candidate_ahead_of_unrelated_cross_encoder_hit(monkeypatch):
    results = [
        SearchResult(
            chunk_id="aligned",
            score=0.08,
            title="Doc 1",
            document_version_id="ver-1",
            source_document_id="doc-1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool settings and operation",
            metadata={"rerank_document": "Defect Tool settings and operation"},
        ),
        SearchResult(
            chunk_id="unrelated",
            score=0.07,
            title="Doc 2",
            document_version_id="ver-1",
            source_document_id="doc-2",
            pages=[2],
            section_path=["Output"],
            content="Judge output status monitor",
            metadata={"rerank_document": "Judge output status monitor"},
        ),
    ]

    class FakeRanker:
        def run(self, query: str, documents: list[object]) -> dict[str, list[object]]:
            assert query == "Defect Tool"
            return {
                "documents": [
                    SimpleNamespace(id="unrelated", score=0.99, meta={"chunk_id": "unrelated"}),
                    SimpleNamespace(id="aligned", score=0.51, meta={"chunk_id": "aligned"}),
                ]
            }

    monkeypatch.setattr(
        retriever,
        "_to_rerank_documents",
        lambda items: [SimpleNamespace(id=item.chunk_id, content=item.metadata.get("rerank_document")) for item in items],
    )
    monkeypatch.setattr(retriever, "_get_reranker", lambda: FakeRanker())

    reranked = rerank_results(results, "Defect Tool", limit=2)

    assert [item.chunk_id for item in reranked] == ["aligned"]
    assert reranked[0].metadata["rerank_query_alignment"] > 0


def test_retrieve_uses_metadata_document_selection_before_chunk_search(monkeypatch):
    selected_filters: list[dict[str, object]] = []
    result = SearchResult(
        chunk_id="selected-chunk",
        score=0.9,
        title="Selected Doc",
        document_version_id="ver-1",
        source_document_id="doc-selected",
        pages=[36],
        section_path=["Specs"],
        content="Column headers: Model-101; Row headers: Repeatability; Cell value: 0.5 um",
        metadata={"chunk_type": "table_record"},
    )

    class FakeStore:
        def search_document_metadata(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
            assert query == "Model-101 z axis repeatability"
            return [
                {
                    "source_document_id": "doc-selected",
                    "score": 0.9,
                    "retrieval_stage": "metadata_dense+metadata_sparse",
                    "payload": {"title": "Selected Doc", "source_filename": "selected.pdf"},
                }
            ]

        def search_dense(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            selected_filters.append(filters)
            return [result]

        def search_sparse(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            selected_filters.append(filters)
            return [result]

        @staticmethod
        def fuse_rrf(result_sets: list[list[SearchResult]], *, limit: int, k: int = 60) -> list[SearchResult]:
            return QdrantStore.fuse_rrf(result_sets, limit=limit, k=k)

    monkeypatch.setattr(retriever, "QdrantStore", FakeStore)
    monkeypatch.setattr(retriever, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retriever, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "assemble_context", lambda results, **_kwargs: results)

    results = retriever.retrieve("Model-101 z axis repeatability", ["corpus-1"], {"is_active": True}, limit=5)

    assert results[0].source_document_id == "doc-selected"
    assert all(filters["source_document_id"] == ["doc-selected"] for filters in selected_filters)
    assert results[0].metadata["document_selection_stage"] == "metadata_embedding"
    assert results[0].metadata["selected_document_metadata_hits"][0]["source_document_id"] == "doc-selected"


def test_product_family_structured_lookup_does_not_hard_scope_chunk_search_to_metadata_selection(monkeypatch):
    selected_filters: list[dict[str, object]] = []
    result = SearchResult(
        chunk_id="table-row",
        score=0.9,
        title="LJ-X8000 Manual",
        document_version_id="ver-1",
        source_document_id="doc-expected",
        pages=[505],
        section_path=["PLC link"],
        content="Column headers: Link unit; Row headers: SYSMAC CPM2A; Cell value: CPM1-C1F01",
        metadata={"chunk_type": "table_record", "product_family": "LJ: X8000 Series"},
    )

    class FakeStore:
        def search_document_metadata(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
            return [
                {
                    "source_document_id": "doc-wrong",
                    "score": 0.9,
                    "retrieval_stage": "metadata_dense+metadata_sparse",
                    "payload": {"title": "X8000 Brochure", "source_filename": "brochure.pdf"},
                }
            ]

        def search_dense(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            selected_filters.append(filters)
            return [result] if filters == {"is_active": True, "chunk_type": ["table_record"]} else []

        def search_sparse(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            selected_filters.append(filters)
            return []

        @staticmethod
        def fuse_rrf(result_sets: list[list[SearchResult]], *, limit: int, k: int = 60) -> list[SearchResult]:
            return QdrantStore.fuse_rrf(result_sets, limit=limit, k=k)

    monkeypatch.setattr(retriever, "QdrantStore", FakeStore)
    monkeypatch.setattr(retriever, "run_contextual_lexical_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(retriever, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "assemble_context", lambda results, **_kwargs: results)

    results = retriever.retrieve(
        "What cpm1-c1f01 SYSMAC CPM2A Link unit value applies to LJ: X8000 Series?",
        ["corpus-1"],
        {"is_active": True},
        limit=5,
    )

    assert [item.chunk_id for item in results] == ["table-row"]
    assert {"is_active": True, "chunk_type": ["table_record"]} in selected_filters
    assert all(filters.get("source_document_id") != ["doc-wrong"] for filters in selected_filters)
    assert results[0].metadata["selected_document_metadata_hits"][0]["source_document_id"] == "doc-wrong"


def test_retrieve_keeps_table_chunks_available_for_safety_queries_without_table_only_route(monkeypatch):
    query = "When installing the controller for LJ-X8000, what warning or caution about controller mounting should be followed?"
    base_search_filters: list[dict[str, object]] = []
    table_backed_warning = SearchResult(
        chunk_id="table-backed-warning",
        score=0.9,
        title="LJ-X8000 Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[12],
        section_path=["Mounting the controller"],
        content="Warning: install the controller in the orientation listed in the mounting precautions table.",
        metadata={"chunk_type": "table_record", "safety_flag": True},
    )

    class FakeStore:
        def search_document_metadata(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
            return []

        def search_dense(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            base_search_filters.append(filters)
            assert "chunk_type" not in filters
            return [table_backed_warning]

        def search_sparse(self, corpus_id: str, query: str, filters: dict[str, object], limit: int = 40) -> list[SearchResult]:
            base_search_filters.append(filters)
            assert "chunk_type" not in filters
            return []

        @staticmethod
        def fuse_rrf(result_sets: list[list[SearchResult]], *, limit: int, k: int = 60) -> list[SearchResult]:
            return QdrantStore.fuse_rrf(result_sets, limit=limit, k=k)

    monkeypatch.setattr(retriever, "QdrantStore", FakeStore)
    monkeypatch.setattr(
        retriever,
        "run_table_search",
        lambda *_args, **_kwargs: pytest.fail("safety procedure queries should not run the extra table-only route"),
    )
    monkeypatch.setattr(retriever, "run_table_lexical_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(retriever, "run_contextual_lexical_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(retriever, "run_special_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(retriever, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retriever, "_apply_query_alignment", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retriever, "assemble_context", lambda results, **_kwargs: results)

    results = retriever.retrieve(query, ["manuals_vendor_keyence"], {"is_active": True}, limit=5)

    assert [result.chunk_id for result in results] == ["table-backed-warning"]
    assert base_search_filters == [{"is_active": True}, {"is_active": True}]


def test_repeatability_query_keeps_table_candidate_after_family_selection():
    analysis = analyze_query("Model-101 z axis repeatability")
    assert analysis.query_types == ["general"]
    assert analysis.preferred_chunk_types == []
    table = SearchResult(
        chunk_id="table",
        score=0.38,
        title="Spec Sheet",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[36],
        section_path=["Specifications"],
        content="Column headers: Model-101; Row headers: Repeatability > Z-axis; Cell value: 0.5 um",
        metadata={
            "chunk_type": "table_record",
            "family_bucket": "table",
            "table_column_headers": ["Model-101"],
            "table_row_headers": ["Repeatability", "Z-axis"],
        },
    )
    prose = SearchResult(
        chunk_id="prose",
        score=0.32,
        title="Spec Sheet",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[34],
        section_path=["Notes"],
        content="For Model-101 connection, measurement range changes when binning is enabled.",
        metadata={"chunk_type": "atomic_text", "family_bucket": "prose"},
    )

    selected = retriever._select_family_candidates([table, prose], analysis, limit=5)

    assert selected[0].chunk_id == "table"


def test_structured_lookup_keeps_table_candidates_when_query_also_says_how():
    analysis = analyze_query(
        "What causes Failed to back up settings to VisionDatabase. for XG-X Series, and how should it be corrected?"
    )
    assert "structured_lookup" in analysis.query_types
    assert "how_to" in analysis.query_types
    table = SearchResult(
        chunk_id="table",
        score=0.8,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[10],
        section_path=["Errors"],
        content="Error Message: Failed to back up settings to VisionDatabase.; Cause: FTP output failed.; Corrective Action: Check disk space.",
        metadata={"chunk_type": "table_record", "family_bucket": "table"},
    )
    context = SearchResult(
        chunk_id="context",
        score=0.9,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[9],
        section_path=["Saving"],
        content="Use this screen to back up settings.",
        metadata={"chunk_type": "section_window", "family_bucket": "context"},
    )

    selected = retriever._select_family_candidates([context, table], analysis, limit=5)

    assert any(result.chunk_id == "table" for result in selected)
    assert selected[0].chunk_id == "table"


def test_safety_queries_keep_warning_records_as_primary_family():
    analysis = analyze_query(
        "What warning or caution about Warning status can be cleared with a warning clear request. "
        "for IV4-G120 applies when obtaining the master number?"
    )
    warning = SearchResult(
        chunk_id="warning",
        score=0.62,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[10],
        section_path=["Warnings"],
        content="Warning: Warning status can be cleared with a warning clear request.",
        metadata={"chunk_type": "warning_record", "family_bucket": "safety"},
    )
    action = SearchResult(
        chunk_id="action",
        score=0.9,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[11],
        section_path=["Status"],
        content="When obtaining the master number, set [Total status condition] as shown below.",
        metadata={"chunk_type": "atomic_text", "family_bucket": "prose"},
    )

    selected = retriever._select_family_candidates([action, warning], analysis, limit=5)

    assert selected[0].chunk_id == "warning"
    assert any(result.chunk_id == "action" for result in selected)


def test_safety_warning_phrase_alignment_boosts_exact_warning_record():
    analysis = analyze_query(
        "What warning or caution about Warning status can be cleared with a warning clear request. "
        "for IV4-G120 applies when obtaining the master number?"
    )
    warning = SearchResult(
        chunk_id="warning",
        score=0.4,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[10],
        section_path=["Warnings"],
        content="Warning: Warning status can be cleared with a warning clear request.",
        metadata={"chunk_type": "warning_record"},
    )
    generic = SearchResult(
        chunk_id="generic",
        score=0.4,
        title="Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[10],
        section_path=["Warnings"],
        content="Warning: Check the controller status before clearing errors.",
        metadata={"chunk_type": "warning_record"},
    )

    rescored = retriever._apply_query_alignment([generic, warning], analysis, stage="query_aligned")

    assert rescored[0].chunk_id == "warning"
    assert rescored[0].score > rescored[1].score


def test_table_lexical_search_scores_structured_troubleshooting_row_groups(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fetch_all(query: str, params: tuple[object, ...]) -> list[dict[str, object]]:
        captured["query"] = query
        captured["params"] = params
        return [
            {
                "id": "cell-1",
                "document_version_id": "ver-1",
                "source_document_id": "doc-1",
                "title": "Manual",
                "section_path_text": "A-1",
                "page_from": 12,
                "page_to": 12,
                "content": "Column headers: Error Message; Cell value: Failed to back up settings to VisionDatabase.",
                "metadata_json": {
                    "corpus_id": "corpus-1",
                    "chunk_type": "table_record",
                    "table_cell": True,
                    "product_family": "XG-X Series",
                    "section_path": ["A-1"],
                },
                "priority_score": 0.0,
            },
            {
                "id": "row-group-1",
                "document_version_id": "ver-1",
                "source_document_id": "doc-1",
                "title": "Manual",
                "section_path_text": "A-1",
                "page_from": 12,
                "page_to": 12,
                "content": (
                    "Error Message: Failed to back up settings to VisionDatabase.; "
                    "Cause: Failed to output the backup settings file to the VisionDatabase destination device.; "
                    "Corrective Action: Check the available disk space and connection condition."
                ),
                "metadata_json": {
                    "corpus_id": "corpus-1",
                    "chunk_type": "table_record",
                    "table_row_group": True,
                    "product_family": "XG-X Series",
                    "section_path": ["A-1"],
                },
                "priority_score": 0.0,
            },
        ]

    monkeypatch.setattr(retriever, "fetch_all", fake_fetch_all)

    analysis = analyze_query(
        "What causes Failed to back up settings to VisionDatabase. for XG-X Series, and how should it be corrected?"
    )
    results = retriever.run_table_lexical_search(
        analysis.raw_query,
        ["corpus-1"],
        {"source_document_id": ["doc-1"]},
        analysis,
        limit=2,
    )

    assert results[0].chunk_id == "row-group-1"
    assert results[0].metadata["table_row_group"] is True
    assert results[0].section_path == ["A-1"]
    assert "source_document_id = any" in str(captured["query"])
    assert "order by priority_score desc, id" in str(captured["query"])
    assert captured["params"][0] == ["corpus-1"]
    assert captured["params"][1] == ["doc-1"]


def test_table_lexical_search_skips_unstructured_general_queries(monkeypatch):
    monkeypatch.setattr(retriever, "fetch_all", lambda *_args, **_kwargs: pytest.fail("fetch_all should not run"))

    results = retriever.run_table_lexical_search(
        "Tell me about the software",
        ["corpus-1"],
        {},
        analyze_query("Tell me about the software"),
    )

    assert results == []
