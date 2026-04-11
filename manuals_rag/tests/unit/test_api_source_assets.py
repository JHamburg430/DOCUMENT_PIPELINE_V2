from fastapi.testclient import TestClient

from apps.api import main
from apps.api.main import app


client = TestClient(app)
USER_HEADERS = {"Authorization": "Bearer user-token"}


def _fake_query_result():
    return {
        "answer": {
            "answer": "Use page 2.",
            "confidence": "high",
            "used_documents": [],
            "citations": [{"chunk_id": "chunk-1", "document_id": "doc-1", "pages": [2]}],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        },
        "retrieval_results": [
            {
                "chunk_id": "chunk-1",
                "source_document_id": "doc-1",
                "document_version_id": "ver-1",
                "pages": [2],
            }
        ],
    }


def test_query_source_assets_are_opt_in(monkeypatch):
    class FakeWorkflow:
        def invoke(self, _payload):
            return _fake_query_result()

    monkeypatch.setattr(main, "workflow", FakeWorkflow())

    response = client.post(
        "/query",
        headers=USER_HEADERS,
        json={"query": "where?", "corpus_ids": ["manuals_vendor_keyence"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "source_assets" not in payload
    assert "source_assets" not in payload["citations"][0]


def test_query_can_return_pdf_download_source_assets(monkeypatch):
    class FakeWorkflow:
        def invoke(self, _payload):
            return _fake_query_result()

    class FakeStore:
        def presigned_get_url(self, bucket, object_name, *, expires):
            return f"http://minio.local/{bucket}/{object_name}?signed=1"

    def fake_fetch_all(_query, _params):
        return [
            {
                "id": "doc-1",
                "tenant_id": "tenant-1",
                "current_version_id": "ver-1",
                "title": "Manual",
                "source_filename": "Manual.pdf",
                "storage_uri": "s3://manuals-originals/tenant-1/sha256/manual.pdf",
            }
        ]

    monkeypatch.setattr(main, "workflow", FakeWorkflow())
    monkeypatch.setattr(main, "ObjectStore", FakeStore)
    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)

    response = client.post(
        "/query",
        headers=USER_HEADERS,
        json={
            "query": "where?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "include_source_assets": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_assets"]["documents"][0]["pdf_download_url"] == (
        "http://minio.local/manuals-originals/tenant-1/sha256/manual.pdf?signed=1"
    )
    assert payload["source_assets"]["citation_pages"][0]["page"] == 2
    assert payload["citations"][0]["source_assets"][0]["pdf_download_url"].endswith("?signed=1")
    assert "page_image_url" not in payload["source_assets"]["citation_pages"][0]


def test_query_can_return_stored_page_and_table_image_assets(monkeypatch):
    class FakeWorkflow:
        def invoke(self, _payload):
            return _fake_query_result()

    class FakeStore:
        def presigned_get_url(self, bucket, object_name, *, expires):
            return f"http://minio.local/{bucket}/{object_name}?signed=1"

    def fake_fetch_all(_query, _params):
        return [
            {
                "id": "doc-1",
                "tenant_id": "tenant-1",
                "current_version_id": "ver-1",
                "title": "Manual",
                "source_filename": "Manual.pdf",
                "storage_uri": "s3://manuals-originals/tenant-1/sha256/manual.pdf",
                "docling_artifact_uri": "s3://manuals-artifacts/tenant-1/artifact.json",
            }
        ]

    def fake_image_assets(_store, _document):
        return {
            "page_images": [{"page": 2, "uri": "s3://manuals-artifacts/tenant-1/page-0002.png"}],
            "table_images": [
                {
                    "page": 2,
                    "table_index": 1,
                    "uri": "s3://manuals-artifacts/tenant-1/table-0001.png",
                    "bbox": {"l": 1, "t": 2, "r": 3, "b": 4},
                }
            ],
        }

    monkeypatch.setattr(main, "workflow", FakeWorkflow())
    monkeypatch.setattr(main, "ObjectStore", FakeStore)
    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(main, "_artifact_image_assets", fake_image_assets)

    response = client.post(
        "/query",
        headers=USER_HEADERS,
        json={
            "query": "where?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "include_page_images": True,
            "include_table_images": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_assets"]["citation_pages"][0]["page_image_url"] == (
        "http://minio.local/manuals-artifacts/tenant-1/page-0002.png?signed=1"
    )
    assert payload["source_assets"]["tables"][0]["table_image_url"] == (
        "http://minio.local/manuals-artifacts/tenant-1/table-0001.png?signed=1"
    )
    assert payload["citations"][0]["table_assets"][0]["table_index"] == 1
