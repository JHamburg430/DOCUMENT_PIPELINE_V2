from manuals_rag_retrieval.retriever import _attach_document_selection
from manuals_rag_schemas.documents import SearchResult


def test_routing_evidence_record_is_attached_to_each_result():
    result = SearchResult(
        chunk_id="chunk-1", source_document_id="doc-1", document_version_id="ver-1",
        title="IV-H2000 manual", content="The search area excludes unwanted outlines.",
        score=1.0, pages=[41], section_path=["Outline"], metadata={}
    )
    hit = {"source_document_id":"doc-1", "score":0.9, "payload":{"title":"IV-H2000 manual", "source_filename":"manual.pdf", "document_version_id":"ver-1"}}
    attached = _attach_document_selection([result], [hit], query="How should IV-H2000 configure the search area?")
    record = attached[0].metadata["routing_evidence_record"]
    assert record["schema"] == "rag-routing-evidence-v1"
    assert record["selected_document_ids"] == ["doc-1"]
    assert record["evidence_candidates"][0]["chunk_id"] == "chunk-1"
    assert "Verify each requested facet" in record["model_instruction"]
