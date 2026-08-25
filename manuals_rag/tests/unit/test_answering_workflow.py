from manuals_rag_answering import workflow
from manuals_rag_schemas.documents import AnswerResponse, SearchResult


def _result(chunk_id: str = "chunk-1") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=1.0,
        title="Manual",
        document_version_id="version-1",
        source_document_id="document-1",
        pages=[1],
        section_path=["Section"],
        content="Grounded evidence",
        metadata={"chunk_type": "table_record"},
    )


def test_answer_workflow_uses_integrated_retriever(monkeypatch):
    calls = []

    def fake_retrieve(query, corpus_ids, filters):
        calls.append((query, corpus_ids, filters))
        return [_result()]

    def fake_generate_answer(query, results):
        return AnswerResponse(
            answer=f"Answered from {results[0].content}",
            confidence="high",
            used_documents=[
                {
                    "document_id": results[0].source_document_id,
                    "title": results[0].title,
                    "version": results[0].document_version_id,
                    "pages": results[0].pages,
                    "section_path": results[0].section_path,
                }
            ],
            citations=[
                {
                    "chunk_id": results[0].chunk_id,
                    "document_id": results[0].source_document_id,
                    "pages": results[0].pages,
                    "quote_span": None,
                }
            ],
            warnings=[],
            followup_questions=[],
            insufficient_evidence=False,
        )

    monkeypatch.setattr(workflow, "retrieve", fake_retrieve)
    monkeypatch.setattr(workflow, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(
        workflow,
        "run_dense_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy dense path should not run")),
    )

    result = workflow.build_workflow().invoke(
        {
            "query": "compare these documents",
            "corpus_ids": ["manuals"],
            "filters": {"document_kind": "manual"},
        }
    )

    assert calls == [("compare these documents", ["manuals"], {"document_kind": "manual", "is_active": True})]
    assert result["retrieval_results"][0]["chunk_id"] == "chunk-1"
    assert result["answer"]["citations"][0]["document_id"] == "document-1"
    assert "retrieve_documents" in result["step_timings_ms"]
    assert "run_dense_search" not in result["step_timings_ms"]


def test_debug_workflow_keeps_stepwise_retrieval_path(monkeypatch):
    monkeypatch.setattr(workflow, "run_dense_search", lambda *_args, **_kwargs: [_result("dense")])
    monkeypatch.setattr(workflow, "run_sparse_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(workflow, "run_special_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(workflow, "fuse_results", lambda _store, result_sets, limit: [item for items in result_sets for item in items])
    monkeypatch.setattr(workflow, "rerank_results", lambda results, query, limit: results)
    monkeypatch.setattr(workflow, "assemble_context", lambda results: results)

    result = workflow.build_workflow(include_answer=False).invoke(
        {
            "query": "lookup",
            "corpus_ids": ["manuals"],
            "filters": {},
        }
    )

    assert result["retrieval_results"][0]["chunk_id"] == "dense"
    assert "run_dense_search" in result["step_timings_ms"]
    assert "retrieve_documents" not in result["step_timings_ms"]
