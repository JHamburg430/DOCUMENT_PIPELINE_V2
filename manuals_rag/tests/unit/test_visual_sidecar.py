from types import SimpleNamespace

import pytest

from manuals_rag_retrieval.visual_sidecar import (
    VisualPage,
    VisualRetrievalSidecar,
    _as_matrix,
    _mean_pool,
    visual_collection_name,
    visual_page_id,
)


class FakeEncoder:
    def encode_document(self, images):
        return [[[1.0, 0.0], [0.0, 1.0]] for _ in images]

    def encode_query(self, queries):
        return [[[0.75, 0.25], [0.25, 0.75]] for _ in queries]


class FakeClient:
    def __init__(self):
        self.created = []
        self.upserts = []
        self.queries = []

    def collection_exists(self, _name):
        return False

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(id="point-1", score=3.5, payload={"page": 7})])


def test_visual_collection_and_point_ids_are_deterministic():
    assert visual_collection_name("vendor/acme manuals") == "manuals_vendor_acme_manuals_visual_pages"
    assert visual_page_id("version-1", 3) == visual_page_id("version-1", 3)
    assert visual_page_id("version-1", 3) != visual_page_id("version-1", 4)


def test_multivector_validation_and_pooling():
    assert _mean_pool([[1.0, 0.0], [0.0, 1.0]]) == [0.5, 0.5]
    with pytest.raises(ValueError, match="ragged"):
        _as_matrix([[1.0], [1.0, 2.0]])


def test_sidecar_uses_pooled_prefetch_and_late_interaction():
    client = FakeClient()
    sidecar = VisualRetrievalSidecar(client=client, encoder=FakeEncoder())
    page = VisualPage(
        source_document_id="document-1",
        document_version_id="version-1",
        page=7,
        image_uri="s3://bucket/page-0007.png",
        title="Manual",
        image=object(),
    )

    assert sidecar.upsert_pages("corpus-1", [page]) == 1
    results = sidecar.search("corpus-1", "Where is the connector pin?", source_document_ids=["document-1"])

    assert client.created[0]["vectors_config"]["late"].hnsw_config.m == 0
    assert len(client.upserts[0]["points"][0].vector["late"]) == 2
    assert client.queries[0]["prefetch"].using == "pooled"
    assert client.queries[0]["using"] == "late"
    assert results == [{"score": 3.5, "page": 7, "visual_point_id": "point-1"}]
