from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app


client = TestClient(app)


def test_search_requires_authentication():
    response = client.post("/search", json={"query": "test", "corpus_ids": ["manuals_vendor_keyence"]})
    assert response.status_code == 401


def test_ingestion_run_requires_privileged_role():
    response = client.get("/ingestion-runs/not-a-real-run", headers={"Authorization": "Bearer user-token"})
    assert response.status_code == 403


def test_upload_corpus_upsert_uses_general_corpus_name_and_permissions(monkeypatch):
    calls = []

    monkeypatch.setattr(api_main, "execute", lambda query, params=(): calls.append((query, params)))

    api_main._upsert_corpus(
        "manuals_vendor_keyence",
        "tenant-1",
        api_main._default_corpus_name("manuals_vendor_keyence"),
        update_on_conflict=False,
    )

    assert calls
    assert "do nothing" in calls[0][0]
    assert calls[0][1][0] == "manuals_vendor_keyence"
    assert calls[0][1][1] == "tenant-1"
    assert calls[0][1][2] == "Manuals Vendor Keyence"
    assert "Keyence" not in api_main._upsert_corpus.__code__.co_consts


def test_upload_document_path_does_not_contain_vendor_manufacturer_fallback():
    constants = " ".join(str(value) for value in api_main.upload_documents.__code__.co_consts)
    assert "manufacturer" in constants
    assert "Keyence" not in constants


def test_ingest_document_initializes_all_step_records_before_queueing(monkeypatch):
    calls = []
    monkeypatch.setattr(
        api_main,
        "fetch_one",
        lambda _query, _params: {
            "current_version_id": "version-1",
            "source_filename": "manual.pdf",
            "file_size_bytes": 123,
            "sha256": "abc",
            "storage_uri": "s3://manuals/manual.pdf",
            "corpus_id": "manuals",
        },
    )
    monkeypatch.setattr(api_main, "execute", lambda query, params=(): calls.append(("execute", query, params)))
    monkeypatch.setattr(api_main, "initialize_ingestion_steps", lambda run_id, upload_details=None: calls.append(("steps", run_id, upload_details)))
    monkeypatch.setattr(api_main, "enqueue", lambda queue, payload: calls.append(("enqueue", queue, payload)))

    response = client.post("/documents/doc-1/ingest", headers={"Authorization": "Bearer admin-token"})

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    step_index = next(index for index, call in enumerate(calls) if call[0] == "steps")
    queue_index = next(index for index, call in enumerate(calls) if call[0] == "enqueue")
    assert step_index < queue_index
    assert calls[step_index][1] == run_id
    assert calls[step_index][2]["filename"] == "manual.pdf"
    assert calls[queue_index][1] == "ingest_jobs"
