from manuals_rag_retrieval.embeddings import EMBED_BATCH_SIZE, embed_dense, normalize_for_embedding


def test_normalize_for_embedding_clips_long_text():
    text = ("alpha " * 2000).strip()
    normalized = normalize_for_embedding(text, max_chars=120)
    assert len(normalized) <= 120
    assert normalized.startswith("alpha")


def test_normalize_for_embedding_adds_search_term_variants():
    normalized = normalize_for_embedding("Find CA-EN100U 1-line cross-section values.")
    assert "ca en100u" in normalized.lower()
    assert "1 line" in normalized.lower()


def test_embed_dense_batches_requests(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            batch = calls[-1]["input"]
            return {"embeddings": [[float(index)] for index, _ in enumerate(batch, start=1)]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, path, json):
            calls.append(json)
            return FakeResponse()

    monkeypatch.setattr("manuals_rag_retrieval.embeddings.httpx.Client", FakeClient)

    texts = [f"text {index}" for index in range(EMBED_BATCH_SIZE + 3)]
    vectors = embed_dense(texts)

    assert len(calls) == 2
    assert all(isinstance(call["input"], list) for call in calls)
    assert len(vectors) == len(texts)


def test_embed_dense_falls_back_to_single_requests(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"bad status {self.status_code}")

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, path, json):
            calls.append(json)
            if isinstance(json["input"], list):
                raise RuntimeError("batch failure")
            return FakeResponse(200, {"embedding": [1.0, 2.0, 3.0]})

    monkeypatch.setattr("manuals_rag_retrieval.embeddings.httpx.Client", FakeClient)

    vectors = embed_dense(["one", "two"])

    assert len(vectors) == 2
    assert isinstance(calls[0]["input"], list)
    assert "one" in calls[0]["input"][0]
    assert "two" in calls[0]["input"][1]
    assert isinstance(calls[1]["input"], str) and "one" in calls[1]["input"]
    assert isinstance(calls[2]["input"], str) and "two" in calls[2]["input"]
