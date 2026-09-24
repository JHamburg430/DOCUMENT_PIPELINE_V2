from concurrent.futures import ThreadPoolExecutor
from threading import Event

from manuals_rag_retrieval import qdrant_store


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
