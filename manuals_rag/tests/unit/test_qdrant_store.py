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
