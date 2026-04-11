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
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [result])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [result])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda *args, **kwargs: [result])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: results)
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
    assert [stage["name"] for stage in case["stages"]] == [
        "dense",
        "sparse",
        "special",
        "fused",
        "family_scored",
        "completeness_scored",
        "query_aligned",
        "family_selected",
        "reranked",
        "assembled",
    ]
    assert case["stages"][0]["results"][0]["chunk_type"] == "spec_record"
    assert "summary" in report


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
    monkeypatch.setattr(retrieval_debug, "run_dense_search", lambda *args, **kwargs: [low_info])
    monkeypatch.setattr(retrieval_debug, "run_sparse_search", lambda *args, **kwargs: [strong])
    monkeypatch.setattr(retrieval_debug, "run_special_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval_debug, "fuse_results", lambda *args, **kwargs: [strong])
    monkeypatch.setattr(retrieval_debug, "_apply_family_scoring", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "_annotate_completeness", lambda results: results)
    monkeypatch.setattr(retrieval_debug, "_select_family_candidates", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "enrich_candidates_for_rerank", lambda results, *_args, **_kwargs: results)
    monkeypatch.setattr(retrieval_debug, "rerank_results", lambda results, *_args, **_kwargs: [low_info])
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
