from concurrent.futures import ThreadPoolExecutor
from threading import Event

from manuals_rag_retrieval import qdrant_store


class _Point:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


def test_exact_document_reference_search_matches_normalized_title(monkeypatch):
    class FakeClient:
        @staticmethod
        def collection_exists(_name):
            return True

        @staticmethod
        def scroll(**_kwargs):
            return (
                [
                    _Point(
                        "doc-exact",
                        {
                            "source_document_id": "doc-exact",
                            "title": "AS_160148_XG-X_C_689246_KA_US_2085_1",
                            "source_filename": "camera.pdf",
                        },
                    ),
                    _Point(
                        "doc-prefix",
                        {
                            "source_document_id": "doc-prefix",
                            "title": "AS_1601489_XG-X_C",
                        },
                    ),
                    _Point("doc-other", {"source_document_id": "doc-other", "title": "XG-X User Manual"}),
                ],
                None,
            )

    store = object.__new__(qdrant_store.QdrantStore)
    store.client = FakeClient()
    monkeypatch.setattr(store, "_build_filter", lambda filters: filters)

    hits = store.search_document_metadata_exact_references(
        "manuals_vendor_keyence",
        ["as160148"],
        {"is_active": True},
    )

    assert [hit["source_document_id"] for hit in hits] == ["doc-exact"]
    assert hits[0]["retrieval_stage"] == "metadata_exact_reference"


def test_dense_query_vector_is_cached_per_query_and_instruction(monkeypatch):
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(qdrant_store, "QdrantClient", lambda **_kwargs: object())
    monkeypatch.setattr(
        qdrant_store,
        "embed_dense",
        lambda texts: calls.append(("dense", texts[0])) or [[1.0, 2.0]],
    )
    monkeypatch.setattr(
        qdrant_store,
        "embed_query_dense",
        lambda query, instruction: calls.append((instruction, query)) or [3.0, 4.0],
    )
    store = qdrant_store.QdrantStore()

    assert store._dense_query_vector("same query") == [1.0, 2.0]
    assert store._dense_query_vector("same query") == [1.0, 2.0]
    assert store._dense_query_vector("same query", instruction="retrieve") == [3.0, 4.0]
    assert store._dense_query_vector("same query", instruction="retrieve") == [3.0, 4.0]
    assert calls == [("dense", "same query"), ("retrieve", "same query")]


def test_dense_query_vector_cache_is_thread_safe(monkeypatch):
    calls: list[str] = []
    embed_started = Event()
    release_embed = Event()

    monkeypatch.setattr(qdrant_store, "QdrantClient", lambda **_kwargs: object())

    def embed(texts):
        calls.append(texts[0])
        embed_started.set()
        assert release_embed.wait(timeout=2)
        return [[1.0, 2.0]]

    monkeypatch.setattr(qdrant_store, "embed_dense", embed)
    store = qdrant_store.QdrantStore()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(store._dense_query_vector, "same query")
        assert embed_started.wait(timeout=2)
        second = executor.submit(store._dense_query_vector, "same query")
        release_embed.set()
        assert first.result() == [1.0, 2.0]
        assert second.result() == [1.0, 2.0]

    assert calls == ["same query"]
