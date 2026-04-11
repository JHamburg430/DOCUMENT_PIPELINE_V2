from manuals_rag_retrieval.query_analysis import analyze_query
from manuals_rag_retrieval.retriever import build_filters


def test_build_filters_adds_model_and_active_default():
    filters = build_filters("How do I set up KV-8000?", {})
    assert filters["is_active"] is True
    assert filters["product_models"] == "KV-8000"


def test_query_analysis_prefers_structured_chunks_for_specs():
    analysis = analyze_query("What is the voltage spec for CA-EN100U datasheet?")
    assert analysis.requested_doc_kind == "datasheet"
    assert "datasheet_record" in analysis.preferred_chunk_types
    assert "table_record" in analysis.preferred_chunk_types


def test_query_analysis_treats_value_queries_as_structured_spec_lookups():
    analysis = analyze_query("In LJ-S8000, what value is given for 1-line cross-section area?")
    assert "spec_lookup" in analysis.query_types
    assert "table_record" in analysis.preferred_chunk_types
    assert "1-line" in analysis.normalized_terms


def test_build_filters_adds_family_and_part_number_filters():
    family_filters = build_filters("Show the LJ-X series communication setup", {})
    assert family_filters["product_families"] == "LJ-X"
    assert family_filters["is_active"] is True

    part_filters = build_filters("Which manual covers OP-88310 wiring?", {})
    assert part_filters["part_numbers"] == "OP-88310"
