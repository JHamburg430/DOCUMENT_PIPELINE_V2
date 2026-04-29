from manuals_rag_retrieval.query_analysis import analyze_query
from manuals_rag_retrieval.retriever import build_filters


def test_build_filters_adds_active_default_without_query_identifier_filters():
    filters = build_filters("How do I set up KV-8000?", {})
    assert filters["is_active"] is True
    assert "product_models" not in filters
    assert "part_numbers" not in filters


def test_build_filters_does_not_add_document_selection_filters_from_query():
    filters = build_filters("LJ-X8080 z axis repeatability", {})
    assert filters == {"is_active": True}


def test_build_filters_preserves_explicit_request_filters():
    filters = build_filters("LJ-X8080 z axis repeatability", {"product_models": "LJ-X8080"})
    assert filters["product_models"] == "LJ-X8080"
    assert filters["is_active"] is True
    assert "part_numbers" not in filters


def test_query_analysis_identifies_requested_datasheet_kind_without_structured_term_routing():
    analysis = analyze_query("What is the voltage spec for CA-EN100U datasheet?")
    assert analysis.requested_doc_kind == "datasheet"
    assert "spec_lookup" not in analysis.query_types
    assert analysis.preferred_chunk_types == []


def test_query_analysis_keeps_value_query_terms_for_embedding_alignment():
    analysis = analyze_query("In LJ-S8000, what value is given for 1-line cross-section area?")
    assert "spec_lookup" not in analysis.query_types
    assert analysis.preferred_chunk_types == []
    assert "1-line" in analysis.normalized_terms


def test_query_analysis_keeps_axis_terms_for_alignment():
    analysis = analyze_query("LJ-X8080 z axis repeatability")
    assert analysis.query_types == ["general"]
    assert analysis.preferred_chunk_types == []
    assert "z" in analysis.normalized_terms
    assert "axis" in analysis.normalized_terms
    assert "repeatability" in analysis.normalized_terms


def test_query_analysis_does_not_attach_single_vendor_manufacturer_filters():
    analysis = analyze_query("keyence setup guide")
    assert not hasattr(analysis, "manufacturer")
    assert analysis.filter_strictness == "loose"


def test_build_filters_does_not_add_family_or_part_number_filters_from_query():
    family_filters = build_filters("Show the LJ-X series communication setup", {})
    assert family_filters == {"is_active": True}

    part_filters = build_filters("Which manual covers OP-88310 wiring?", {})
    assert part_filters == {"is_active": True}
