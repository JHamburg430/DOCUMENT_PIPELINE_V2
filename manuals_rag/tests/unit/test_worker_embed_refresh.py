from types import SimpleNamespace
import pytest
import apps.worker_embed.main as worker


@pytest.mark.parametrize("fail_upsert", [False, True])
def test_refresh_preserves_old_index_until_upsert_succeeds(monkeypatch, fail_upsert):
    events = []
    chunk = dict(id="chunk", document_version_id="version", source_document_id="document",
                 logical_node_ids_json=[], metadata_json={})
    monkeypatch.setattr(worker, "fetch_all", lambda query, params: [chunk] if "retrieval_chunks" in query else [{"corpus_id":"corpus"}])
    monkeypatch.setattr(worker, "RetrievalChunk", SimpleNamespace(model_validate=lambda row:SimpleNamespace(id=row["id"])))
    for name in ("execute", "start_ingestion_step", "complete_ingestion_step", "fail_ingestion_step"):
        monkeypatch.setattr(worker, name, lambda *a, **kw:None)
    monkeypatch.setattr(worker, "_fetch_document_metadata_record", lambda doc:{"source_document_id":doc})
    class Store:
        def __init__(self, **kw):
            assert kw["timeout"] == 30
        def upsert_chunks(self, *a):
            events.append("upsert")
            if fail_upsert:
                raise RuntimeError("embedding unavailable")
        def delete_document_chunks(self, corpus, **kw):
            assert kw["exclude_chunk_ids"] == ["chunk"]
            events.append("prune")
        def upsert_document_metadata(self, *a):
            events.append("selector_upsert")
    monkeypatch.setattr(worker, "QdrantStore", Store)
    job=dict(run_id="run",document_id="document",version_id="version")
    if fail_upsert:
        with pytest.raises(RuntimeError, match="embedding unavailable"):
            worker.process_job(job)
        assert events == ["upsert"]
    else:
        worker.process_job(job)
        assert events == ["upsert", "prune", "selector_upsert"]


def test_empty_refresh_does_not_delete_existing_index(monkeypatch):
    monkeypatch.setattr(worker,"fetch_all",lambda *a:[])
    monkeypatch.setattr(worker,"start_ingestion_step",lambda *a:None)
    monkeypatch.setattr(worker,"fail_ingestion_step",lambda *a:None)
    monkeypatch.setattr(worker,"QdrantStore",lambda **kw:pytest.fail("Empty refresh must not touch Qdrant"))
    with pytest.raises(ValueError,match="No chunks"):
        worker.process_job(dict(run_id="run",document_id="doc",version_id="version"))


def test_pruning_filter_excludes_replacement_point_ids():
    from manuals_rag_retrieval.qdrant_store import QdrantStore
    captured={}
    store=object.__new__(QdrantStore)
    store.client=SimpleNamespace(collection_exists=lambda name:True,delete=lambda **kw:captured.update(kw))
    point='11111111-1111-1111-1111-111111111111'
    store.delete_document_chunks('corpus',source_document_id='doc',document_version_id='version',exclude_chunk_ids=[point])
    query=captured['points_selector'].filter
    assert query.must_not[0].has_id==[point]
    assert {c.key for c in query.must}=={'source_document_id','document_version_id'}
