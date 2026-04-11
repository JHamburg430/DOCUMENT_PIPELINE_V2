from __future__ import annotations

import time
from uuid import uuid4

import httpx
import pytest
from minio import Minio

from manuals_rag_common.config import settings
from tests.helpers import fixture_pdf_path


API_BASE = "http://127.0.0.1:8600"
ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}
USER_HEADERS = {"Authorization": "Bearer user-token"}
FIXTURE = fixture_pdf_path()
MINIO = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure,
)


def _api_available() -> bool:
    try:
        response = httpx.get(f"{API_BASE}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _api_available(), reason="Live API stack is not running.")


def _upload_and_ingest() -> str:
    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        unique_name = f"{uuid4()}_{FIXTURE.name}"
        with FIXTURE.open("rb") as handle:
            upload = client.post(
                "/documents/upload",
                headers=ADMIN_HEADERS,
                files={"files": (unique_name, handle, "application/pdf")},
            )
        upload.raise_for_status()
        document_id = upload.json()["uploaded"][0]["document_id"]
        ingest = client.post(f"/documents/{document_id}/ingest", headers=ADMIN_HEADERS)
        ingest.raise_for_status()
        run_id = ingest.json()["run_id"]
        for _ in range(240):
            run = client.get(f"/ingestion-runs/{run_id}", headers=ADMIN_HEADERS)
            run.raise_for_status()
            payload = run.json()
            if payload["status"] == "completed":
                return document_id
            if payload["status"] == "failed":
                raise AssertionError(payload["failure_reason"])
            time.sleep(1)
    raise TimeoutError("Ingestion did not complete.")


def test_live_upload_deduplicates_same_file_within_corpus():
    corpus_id = f"dedupe-{uuid4()}"
    before = {
        obj.object_name
        for obj in MINIO.list_objects("manuals-originals", recursive=True)
        if "/sha256/" in obj.object_name
    }
    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        create = client.post("/corpora", headers=ADMIN_HEADERS, json={"id": corpus_id, "name": corpus_id})
        create.raise_for_status()
        with FIXTURE.open("rb") as handle:
            first = client.post(
                "/documents/upload",
                headers=ADMIN_HEADERS,
                data={"corpus_id": corpus_id},
                files={"files": (FIXTURE.name, handle, "application/pdf")},
            )
        first.raise_for_status()
        first_item = first.json()["uploaded"][0]
        assert first_item["duplicate"] is False
        after_first = {
            obj.object_name
            for obj in MINIO.list_objects("manuals-originals", recursive=True)
            if "/sha256/" in obj.object_name
        }

        with FIXTURE.open("rb") as handle:
            second = client.post(
                "/documents/upload",
                headers=ADMIN_HEADERS,
                data={"corpus_id": corpus_id},
                files={"files": (FIXTURE.name, handle, "application/pdf")},
            )
        second.raise_for_status()
        second_item = second.json()["uploaded"][0]
        assert second_item["duplicate"] is True
        assert second_item["document_id"] == first_item["document_id"]
        assert second_item["version_id"] == first_item["version_id"]
    after = {
        obj.object_name
        for obj in MINIO.list_objects("manuals-originals", recursive=True)
        if "/sha256/" in obj.object_name
    }
    assert len(after) == len(after_first)
    assert len(after_first) in {len(before), len(before) + 1}


def test_live_upload_ingest_and_query():
    document_id = _upload_and_ingest()
    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        document = client.get(f"/documents/{document_id}", headers=ADMIN_HEADERS)
        document.raise_for_status()
        doc_payload = document.json()
        assert doc_payload["document_kind"] == "datasheet"
        assert doc_payload["manufacturer"] == "Keyence"
        assert doc_payload["product_model"] == "CA-EN100U"

        query = client.post(
            "/query",
            headers=USER_HEADERS,
            json={
                "query": "What product is described in the CA-EN100U datasheet?",
                "corpus_ids": ["manuals_vendor_keyence"],
                "filters": {"source_document_id": document_id},
                "response_mode": "answer_with_citations",
            },
        )
        query.raise_for_status()
        answer = query.json()
        assert "Encoder relay unit" in answer["answer"]
        assert answer["citations"]
        assert answer["used_documents"]


def test_live_query_rejects_admin_only_document_route_for_end_user():
    with httpx.Client(base_url=API_BASE, timeout=10.0) as client:
        response = client.get("/ingestion-runs/not-a-real-run", headers=USER_HEADERS)
    assert response.status_code == 403
