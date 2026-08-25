from manuals_rag_evals import retrieval_debug
from manuals_rag_schemas.documents import SearchResult


def test_debug_report_includes_all_stages(monkeypatch):
    result = SearchResult(
        chunk_id="chunk-1",
        score=0.8,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Specs"],
        content="Voltage: 24 V",
        metadata={"chunk_type": "spec_record"},
    )

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(retrieval_debug, "analyze_query", lambda query: type("A", (), {"query_types": ["spec_lookup"], "preferred_chunk_types": ["spec_record"]})())
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(
        retrieval_debug,
        "select_documents_from_metadata",
        lambda store, query, corpus_ids, filters: (
            {**filters, "source_document_id": ["doc-1"]},
            [{"source_document_id": "doc-1", "score": 0.5, "retrieval_stage": "metadata_dense", "payload": {"title": "Doc"}}],
        ),
    )
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [result])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [result])
    monkeypatch.setattr(retrieval_debug, "run_table_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda *args, **kwargs: [result])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_promote_comparison_table_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results)

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["What is the voltage?"],
        request_filters={"product_model": "CA-EN100U"},
        top_k=3,
    )

    assert report["query_count"] == 1
    case = report["cases"][0]
    assert case["query_types"] == ["spec_lookup"]
    assert case["filters"]["source_document_id"] == ["doc-1"]
    assert case["chunk_search_filters"]["source_document_id"] == ["doc-1"]
    assert [stage["name"] for stage in case["stages"]] == [
        "metadata_document_selection",
        "dense",
        "sparse",
        "table",
        "table_lexical",
        "contextual_lexical",
        "special",
        "fused",
        "family_scored",
        "completeness_scored",
        "query_aligned",
        "family_selected",
        "reranked",
        "comparison_table_promoted",
        "deduped",
        "assembled",
    ]
    assert case["stages"][0]["results"][0]["source_document_id"] == "doc-1"
    assert case["stages"][1]["results"][0]["chunk_type"] == "spec_record"
    assert "summary" in report
    assert report["summary"]["cases_with_metadata_document_selection"] == 1


def test_debug_report_summary_flags_low_information_stage_regressions(monkeypatch):
    low_info = SearchResult(
        chunk_id="chunk-url",
        score=0.9,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Contact"],
        content="https://example.com",
        metadata={"chunk_type": "datasheet_record"},
    )
    strong = SearchResult(
        chunk_id="chunk-good",
        score=0.8,
        title="Doc",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[1],
        section_path=["Overview"],
        content="CA-EN100U encoder relay unit",
        metadata={"chunk_type": "section_window"},
    )

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(retrieval_debug, "analyze_query", lambda query: type("A", (), {"query_types": ["spec_lookup"], "preferred_chunk_types": ["spec_record"]})())
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(
        retrieval_debug,
        "select_documents_from_metadata",
        lambda store, query, corpus_ids, filters: (filters, []),
    )
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [low_info])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [strong])
    monkeypatch.setattr(retrieval_debug, "run_table_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda *args, **kwargs: [strong])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: [low_info])
    monkeypatch.setattr(retrieval_debug, "_promote_comparison_table_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results)

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["What is described in the datasheet?"],
        request_filters={"product_model": "CA-EN100U"},
        top_k=3,
    )

    assert report["summary"]["cases_with_low_information_top_hit"] == 1
    assert report["summary"]["cases_with_empty_special_stage"] == 1
    assert report["summary"]["cases_where_rerank_promoted_low_information"] == 1


def test_debug_report_tracks_expected_evidence_stage_ranks(monkeypatch):
    expected = SearchResult(
        chunk_id="chunk-expected",
        score=0.9,
        title="Expected Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[8],
        section_path=["Troubleshooting"],
        content="Corrective Action: KEYENCE does not guarantee operation with commercial SD cards.",
        metadata={"chunk_type": "table_record"},
    )
    distractor = SearchResult(
        chunk_id="chunk-other",
        score=0.95,
        title="Other Manual",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[3],
        section_path=["Troubleshooting"],
        content="Corrective Action: Check write protection on the SD card.",
        metadata={"chunk_type": "table_record"},
    )

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(
        retrieval_debug,
        "analyze_query",
        lambda query: type("A", (), {"query_types": ["comparison"], "preferred_chunk_types": ["table_record"]})(),
    )
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(
        retrieval_debug,
        "select_documents_from_metadata",
        lambda store, query, corpus_ids, filters: (filters, []),
    )
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_search", lambda *args, **kwargs: [distractor])
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", lambda *args, **kwargs: [expected, distractor])
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda _store, sets, **_kwargs: [item for group in sets for item in group])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_apply_query_alignment", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_promote_comparison_table_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results)

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["Compare unsupported SD-card corrective actions."],
        top_k=5,
        expected_evidence_by_query={
            "Compare unsupported SD-card corrective actions.": [
                {
                    "chunk_id": "chunk-expected",
                    "source_document_id": "doc-1",
                    "expected_terms": ["keyence", "guarantee", "commercial"],
                }
            ]
        },
    )

    ranks = report["cases"][0]["diagnostics"]["expected_evidence_stage_ranks"]
    assert ranks["table_lexical"][0]["exact_rank"] == 1
    assert ranks["table"][0]["exact_rank"] is None
    assert ranks["assembled"][0]["same_document_best_overlap"] == 3


def test_debug_report_expected_evidence_ranks_scan_beyond_stage_preview(monkeypatch):
    expected = SearchResult(
        chunk_id="chunk-expected",
        score=0.1,
        title="Expected Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[8],
        section_path=["Troubleshooting"],
        content="Corrective Action: KEYENCE does not guarantee operation with commercial SD cards.",
        metadata={"chunk_type": "table_record"},
    )
    distractors = [
        SearchResult(
            chunk_id=f"chunk-other-{index}",
            score=0.9 - index / 100.0,
            title="Other Manual",
            document_version_id="ver-2",
            source_document_id="doc-2",
            pages=[index + 1],
            section_path=["Troubleshooting"],
            content="Corrective Action: Check SD card write protection.",
            metadata={"chunk_type": "table_record"},
        )
        for index in range(3)
    ]

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(
        retrieval_debug,
        "analyze_query",
        lambda query: type("A", (), {"query_types": ["comparison"], "preferred_chunk_types": ["table_record"]})(),
    )
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(retrieval_debug, "select_documents_from_metadata", lambda store, query, corpus_ids, filters: (filters, []))
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", lambda *args, **kwargs: [*distractors, expected])
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda _store, sets, **_kwargs: [item for group in sets for item in group])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_apply_query_alignment", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_promote_comparison_table_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results[:2])

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["Compare unsupported SD-card corrective actions."],
        top_k=2,
        expected_evidence_by_query={
            "Compare unsupported SD-card corrective actions.": [
                {
                    "chunk_id": "chunk-expected",
                    "source_document_id": "doc-1",
                    "expected_terms": ["keyence", "guarantee", "commercial"],
                }
            ]
        },
    )

    case = report["cases"][0]
    assert [result["chunk_id"] for result in case["stages"][4]["results"]] == ["chunk-other-0", "chunk-other-1"]
    ranks = case["diagnostics"]["expected_evidence_stage_ranks"]
    assert ranks["table_lexical"][0]["exact_rank"] == 4
    assert ranks["table_lexical"][0]["rank_search_depth"] == 4
    assert ranks["assembled"][0]["exact_rank"] is None
    outcomes = case["diagnostics"]["expected_evidence_top_k_outcomes"]
    assert outcomes["table_lexical"]["4"]["passed"] is True
    assert outcomes["table_lexical"]["5"]["passed"] is True
    assert outcomes["assembled"]["2"]["passed"] is False


def test_debug_report_compares_normal_top_k_with_deeper_expected_evidence(monkeypatch):
    expected = SearchResult(
        chunk_id="chunk-expected",
        score=0.1,
        title="Expected Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[8],
        section_path=["Troubleshooting"],
        content="Corrective Action: KEYENCE does not guarantee operation with commercial SD cards.",
        metadata={"chunk_type": "table_record"},
    )
    distractors = [
        SearchResult(
            chunk_id=f"chunk-other-{index}",
            score=0.9 - index / 100.0,
            title="Other Manual",
            document_version_id="ver-2",
            source_document_id="doc-2",
            pages=[index + 1],
            section_path=["Troubleshooting"],
            content="Corrective Action: Check SD card write protection.",
            metadata={"chunk_type": "table_record"},
        )
        for index in range(6)
    ]

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(
        retrieval_debug,
        "analyze_query",
        lambda query: type("A", (), {"query_types": ["comparison"], "preferred_chunk_types": ["table_record"]})(),
    )
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(retrieval_debug, "select_documents_from_metadata", lambda store, query, corpus_ids, filters: (filters, []))
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", lambda *args, **kwargs: [*distractors, expected])
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda _store, sets, **_kwargs: [item for group in sets for item in group])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_apply_query_alignment", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_promote_comparison_table_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results[:7])

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["Compare unsupported SD-card corrective actions."],
        top_k=7,
        expected_evidence_by_query={
            "Compare unsupported SD-card corrective actions.": [
                {
                    "chunk_id": "chunk-expected",
                    "source_document_id": "doc-1",
                    "expected_terms": ["keyence", "guarantee", "commercial"],
                }
            ]
        },
    )

    outcomes = report["cases"][0]["diagnostics"]["expected_evidence_top_k_outcomes"]
    assert outcomes["assembled"]["5"]["passed"] is False
    assert outcomes["assembled"]["7"]["passed"] is True
    assert outcomes["assembled"]["5"]["missing_evidence"][0]["exact_rank"] == 7


def test_debug_report_explains_same_document_crowding_for_missing_evidence(monkeypatch):
    expected = SearchResult(
        chunk_id="chunk-expected",
        score=0.1,
        title="Expected Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[8],
        section_path=["Settings"],
        content="Setting item: RTO2L; Settings: Equivalent Oval Aspect Ratio Min.",
        metadata={
            "chunk_type": "table_record",
            "table_column_headers": ["Description of measurement item selection"],
            "table_row_headers": ["RTO2L"],
        },
    )
    weak_same_document = SearchResult(
        chunk_id="chunk-weak",
        score=0.9,
        title="Expected Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[7],
        section_path=["Settings"],
        content="Setting item: RTO1L; Settings: Equivalent Oval Diameter.",
        metadata={
            "chunk_type": "table_record",
            "table_column_headers": ["Description of measurement item selection"],
            "table_row_headers": ["RTO1L"],
        },
    )
    other_document = SearchResult(
        chunk_id="chunk-other",
        score=0.8,
        title="Other Manual",
        document_version_id="ver-2",
        source_document_id="doc-2",
        pages=[3],
        section_path=["Settings"],
        content="Setting item: PMSR DC2LAR; Settings: Cross-sectionArea Surrounded by a Straight Line.",
        metadata={"chunk_type": "table_record"},
    )

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(
        retrieval_debug,
        "analyze_query",
        lambda query: type("A", (), {"query_types": ["comparison"], "preferred_chunk_types": ["table_record"]})(),
    )
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(retrieval_debug, "select_documents_from_metadata", lambda store, query, corpus_ids, filters: (filters, []))
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", lambda *args, **kwargs: [weak_same_document, other_document])
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda _store, sets, **_kwargs: [item for group in sets for item in group])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_apply_query_alignment", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_promote_comparison_table_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results)

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["Compare PMSR DC2LAR and RTO2L."],
        top_k=5,
        expected_evidence_by_query={
            "Compare PMSR DC2LAR and RTO2L.": [
                {
                    "chunk_id": expected.chunk_id,
                    "source_document_id": expected.source_document_id,
                    "expected_terms": ["equivalent", "aspect", "ratio"],
                }
            ]
        },
    )

    crowding = report["cases"][0]["diagnostics"]["expected_evidence_same_document_crowding"]
    same_doc_candidates = crowding["table_lexical"][0]["same_document_candidates"]
    assert same_doc_candidates[0]["chunk_id"] == "chunk-weak"
    assert same_doc_candidates[0]["matched_terms"] == ["equivalent"]
    assert same_doc_candidates[0]["missing_terms"] == ["aspect", "ratio"]
    assert same_doc_candidates[0]["table_row_headers"] == ["RTO1L"]


def test_debug_report_probes_expected_evidence_lexical_discovery(monkeypatch):
    expected = SearchResult(
        chunk_id="chunk-expected",
        score=0.1,
        title="Expected Manual",
        document_version_id="ver-1",
        source_document_id="doc-1",
        pages=[8],
        section_path=["Settings"],
        content="Setting item: RTO2L; Settings: Equivalent Oval Aspect Ratio Min.",
        metadata={
            "chunk_type": "table_record",
            "table_column_headers": ["Description of measurement item selection"],
            "table_row_headers": ["RTO2L"],
            "product_family": "LJ-S8000 Series",
            "content_for_rerank": "Setting item: RTO2L; Settings: Equivalent Oval Aspect Ratio Min.",
        },
    )
    analysis = type(
        "A",
        (),
        {
            "query_types": ["comparison"],
            "preferred_chunk_types": ["table_record"],
            "product_identifiers": ["LJ-S8000", "LJ-X8000"],
        },
    )()

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(retrieval_debug, "analyze_query", lambda query: analysis)
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(retrieval_debug, "select_documents_from_metadata", lambda store, query, corpus_ids, filters: (filters, []))
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", lambda *args, **kwargs: [expected])
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda _store, sets, **_kwargs: [item for group in sets for item in group])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_apply_query_alignment", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_promote_comparison_table_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results)
    monkeypatch.setattr(
        retrieval_debug,
        "fetch_all",
        lambda *_args, **_kwargs: [
            {
                "id": "chunk-expected",
                "document_version_id": "ver-1",
                "source_document_id": "doc-1",
                "title": "Expected Manual",
                "section_path_text": "Settings",
                "page_from": 8,
                "page_to": 8,
                "content": "Setting item: RTO2L; Settings: Equivalent Oval Aspect Ratio Min.",
                "metadata_json": {
                    "chunk_type": "table_record",
                    "table_column_headers": ["Description of measurement item selection"],
                    "table_row_headers": ["RTO2L"],
                    "product_family": "LJ-S8000 Series",
                },
                "priority_score": 0.0,
            }
        ],
    )

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["For LJ-X8000 and LJ-S8000, compare what PMSR DC2LAR and RTO2L represent."],
        top_k=5,
        expected_evidence_by_query={
            "For LJ-X8000 and LJ-S8000, compare what PMSR DC2LAR and RTO2L represent.": [
                {
                    "chunk_id": "chunk-expected",
                    "source_document_id": "doc-1",
                    "expected_terms": ["equivalent", "aspect", "ratio"],
                }
            ]
        },
    )

    probe = report["cases"][0]["diagnostics"]["expected_evidence_lexical_probe"]
    assert "rto2l" in probe["lexical_terms"]
    assert probe["items"][0]["found_in_database"] is True
    assert probe["items"][0]["expected_terms_matched_in_content"] == ["equivalent", "aspect", "ratio"]
    assert probe["items"][0]["matched_query_identifiers"] == ["LJ-S8000"]
    assert probe["items"][0]["stage_exact_ranks"]["table_lexical"] == 1


def test_debug_report_uses_stage_candidate_limit_for_deeper_stage_ranks(monkeypatch):
    captured_limits: dict[str, int] = {}

    def _record_limit(name):
        def _inner(*_args, **kwargs):
            captured_limits[name] = kwargs["limit"]
            return []

        return _inner

    monkeypatch.setattr(retrieval_debug, "build_filters", lambda query, request_filters: {"is_active": True, **request_filters})
    monkeypatch.setattr(
        retrieval_debug,
        "analyze_query",
        lambda query: type("A", (), {"query_types": ["comparison"], "preferred_chunk_types": ["table_record"]})(),
    )
    monkeypatch.setattr(retrieval_debug, "QdrantStore", lambda: object())
    monkeypatch.setattr(retrieval_debug, "_chunk_search_filters", lambda filters, metadata_filters, _analysis: metadata_filters)
    monkeypatch.setattr(retrieval_debug, "select_documents_from_metadata", lambda store, query, corpus_ids, filters: (filters, []))
    monkeypatch.setattr(retrieval_debug, "run_dense_search", _record_limit("dense"))
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", _record_limit("sparse"))
    monkeypatch.setattr(retrieval_debug, "run_table_search", _record_limit("table"))
    monkeypatch.setattr(retrieval_debug, "run_table_lexical_search", _record_limit("table_lexical"))
    monkeypatch.setattr(retrieval_debug, "run_contextual_lexical_search", _record_limit("contextual_lexical"))
    monkeypatch.setattr(retrieval_debug, "run_special_search", _record_limit("special"))
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda _store, sets, **_kwargs: [item for group in sets for item in group])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_apply_query_alignment", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **kwargs: captured_limits.setdefault("family_selected", kwargs["limit"]) and results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **kwargs: captured_limits.setdefault("enriched", kwargs["limit"]) and results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **kwargs: captured_limits.setdefault("reranked", kwargs["limit"]) and results)
    monkeypatch.setattr(
        retrieval_debug,
        "_promote_comparison_table_candidates",
        lambda results, *_args, **kwargs: captured_limits.setdefault("comparison_promoted", kwargs["limit"]) and results,
    )
    monkeypatch.setattr(retrieval_debug, "_dedupe_results", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "assemble_context", lambda results, **_kwargs: results)

    report = retrieval_debug.debug_retrieval_report(
        corpus_ids=["manuals_vendor_keyence"],
        queries=["Compare setting rows."],
        top_k=2,
        stage_candidate_limit=25,
    )

    assert report["stage_candidate_limit"] == 25
    assert captured_limits == {
        "dense": 25,
        "sparse": 25,
        "table": 25,
        "table_lexical": 25,
        "contextual_lexical": 25,
        "special": 25,
        "family_selected": 25,
        "enriched": 25,
        "reranked": 25,
        "comparison_promoted": 25,
    }
