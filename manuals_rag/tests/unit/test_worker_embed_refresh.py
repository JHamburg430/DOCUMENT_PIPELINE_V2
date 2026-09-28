from types import SimpleNamespace
import pytest
import apps.worker_embed.main as worker


@pytest.mark.parametrize("fail_upsert", [False, True])
def test_refresh_preserves_old_index_until_upsert_succeeds(monkeypatch, fail_upsert):
    events = []
    chunk = dict(id="chunk", document_version_id="version", source_document_id="document",
                 logical_node_ids_json=[], metadata_json={})
    def fetch(query, params):
        if "count(*)" in query:
            return [{"count": 1}]
        if "select * from retrieval_chunks" in query:
            return [chunk]
        return [{"corpus_id": "corpus"}]
    monkeypatch.setattr(worker, "fetch_all", fetch)
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


def test_metadata_only_refresh_updates_payload_without_reembedding(monkeypatch):
    events = []
    def fetch(query, params):
        if "count(*)" in query:
            return [{"count": 38340}]
        return [{"corpus_id": "corpus"}]
    monkeypatch.setattr(worker, "fetch_all", fetch)
    for name in ("execute", "start_ingestion_step", "complete_ingestion_step", "fail_ingestion_step"):
        monkeypatch.setattr(worker, name, lambda *a, **kw: None)
    monkeypatch.setattr(worker, "_fetch_document_metadata_record", lambda doc: {
        "source_document_id": doc,
        "metadata_json": {
            "metadata_pipeline_version": "v7",
            "routing_product_models": ["XG-X2700"],
        },
    })
    class Store:
        def __init__(self, **kw):
            pass
        def refresh_document_chunk_payload(self, corpus, **kwargs):
            events.append(("payload", corpus, kwargs))
        def upsert_document_metadata(self, corpus, records):
            events.append(("selector", corpus, records))
        def upsert_chunks(self, *args):
            pytest.fail("metadata-only refresh must not recompute dense vectors")
    monkeypatch.setattr(worker, "QdrantStore", Store)

    worker.process_job({
        "run_id": "run",
        "document_id": "document",
        "version_id": "version",
        "metadata_only": True,
    })

    assert [event[0] for event in events] == ["payload", "selector"]
    assert events[0][2]["payload"]["routing_product_models"] == ["XG-X2700"]


def test_qdrant_payload_refresh_uses_document_filter_for_dense_and_bm25(monkeypatch):
    import manuals_rag_retrieval.qdrant_store as qdrant_store
    from manuals_rag_retrieval.qdrant_store import QdrantStore
    captured = []
    store = object.__new__(QdrantStore)
    store.client = SimpleNamespace(
        collection_exists=lambda name: True,
        set_payload=lambda **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(
        qdrant_store,
        "settings",
        SimpleNamespace(indexed_bm25_enabled=True),
    )

    store.refresh_document_chunk_payload(
        "corpus",
        source_document_id="document",
        payload={"metadata_pipeline_version": "v7"},
    )

    assert len(captured) == 2
    assert {call["collection_name"] for call in captured} == {
        "manuals_corpus",
        "manuals_corpus_bm25",
    }
    assert all(call["points"].must[0].match.value == "document" for call in captured)


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
