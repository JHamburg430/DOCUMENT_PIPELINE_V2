from __future__ import annotations

import pytest

from manuals_rag_evals.pipeline_health import (
    check_chunking_stage,
    check_chunk_quality_stage,
    check_document_metadata_index_stage,
    check_embedding_stage,
    check_fixture_path,
    check_fragmentation_stage,
    check_local_retrieval_stage,
    check_metadata_stage,
    check_normalize_stage,
    check_parse_stage,
    check_page_provenance_stage,
    check_production_retrieval_stage,
    check_query_analysis_stage,
    PIPELINE_HEALTH_DOCUMENT_ID,
)
from tests.helpers import fixture_pdf_path


FIXTURE = fixture_pdf_path()


@pytest.fixture(scope="module")
def parsed_document():
    result, parsed = check_parse_stage(FIXTURE)
    assert result.status == "pass"
    assert parsed is not None
    return parsed


@pytest.fixture(scope="module")
def inferred_metadata(parsed_document):
    result, metadata = check_metadata_stage(FIXTURE, parsed_document)
    assert result.status == "pass"
    assert metadata is not None
    return metadata


@pytest.fixture(scope="module")
def normalized_nodes(parsed_document):
    result, normalized = check_normalize_stage(parsed_document)
    assert result.status == "pass"
    assert normalized is not None
    return normalized


@pytest.fixture(scope="module")
def built_chunks(inferred_metadata, normalized_nodes):
    result, chunks = check_chunking_stage(inferred_metadata, normalized_nodes)
    assert result.status == "pass"
    assert chunks is not None
    return chunks


def test_pipeline_health_fixture_stage():
    result = check_fixture_path(FIXTURE)
    assert result.status == "pass"


def test_pipeline_health_parse_stage(parsed_document):
    assert parsed_document is not None


def test_pipeline_health_metadata_stage(parsed_document):
    result, metadata = check_metadata_stage(FIXTURE, parsed_document)
    assert result.status == "pass"
    assert metadata is not None
    assert metadata.product_model == "CA-EN100U"


def test_pipeline_health_page_provenance_stage(parsed_document):
    result = check_page_provenance_stage(parsed_document)
    assert result.status == "pass"
    assert result.details["distinct_logical_pages"] == result.details["page_count"]


def test_pipeline_health_normalize_stage(parsed_document):
    result, normalized = check_normalize_stage(parsed_document)
    assert result.status == "pass"
    assert normalized is not None
    assert normalized


def test_pipeline_health_chunking_stage(inferred_metadata, normalized_nodes):
    result, chunks = check_chunking_stage(inferred_metadata, normalized_nodes)
    assert result.status == "pass"
    assert chunks is not None
    assert chunks


def test_pipeline_health_fragmentation_stage(parsed_document, normalized_nodes, built_chunks):
    result = check_fragmentation_stage(parsed_document, normalized_nodes, built_chunks)
    assert result.status == "pass"
    assert result.details["distinct_chunk_pages"] > 0
    assert result.details["avg_chunks_per_page"] > 0


def test_pipeline_health_chunk_quality_stage(built_chunks):
    result = check_chunk_quality_stage(built_chunks)
    assert result.details["chunk_count"] > 0
    assert result.details["low_information_ratio"] >= 0.0
    assert result.details["structured_low_information_ratio"] >= 0.0


def test_pipeline_health_embedding_stage(built_chunks):
    result = check_embedding_stage(built_chunks)
    if result.status == "fail" and result.error and "/api/embed" in result.error:
        pytest.skip("Embedding backend unavailable in the test runtime.")
    assert result.status == "pass"


def test_pipeline_health_query_analysis_stage():
    result = check_query_analysis_stage()
    assert result.status == "pass"
    assert "section_window" in result.details["preferred_chunk_types"]


def test_pipeline_health_local_retrieval_stage(built_chunks, inferred_metadata):
    result = check_local_retrieval_stage(built_chunks, inferred_metadata.product_model)
    if result.status == "fail" and result.error and "/api/embed" in result.error:
        pytest.skip("Embedding backend unavailable in the test runtime.")
    assert result.status == "pass"


def test_pipeline_health_document_metadata_index_stage(built_chunks, inferred_metadata):
    result = check_document_metadata_index_stage(built_chunks, inferred_metadata.product_model)
    if result.status == "fail" and result.error and "/api/embed" in result.error:
        pytest.skip("Embedding backend unavailable in the test runtime.")
    assert result.status == "pass"
    assert result.details["top_source_document_id"] == PIPELINE_HEALTH_DOCUMENT_ID


def test_pipeline_health_production_retrieval_stage(built_chunks, inferred_metadata):
    result = check_production_retrieval_stage(built_chunks, inferred_metadata.product_model)
    if result.status == "fail" and result.error and "/api/embed" in result.error:
        pytest.skip("Embedding backend unavailable in the test runtime.")
    assert result.status == "pass"
    assert result.details["document_selection_stage"] == "metadata_embedding"
