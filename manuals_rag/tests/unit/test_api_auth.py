from fastapi.testclient import TestClient

from apps.api.main import app


client = TestClient(app)


def test_search_requires_authentication():
    response = client.post("/search", json={"query": "test", "corpus_ids": ["manuals_vendor_keyence"]})
    assert response.status_code == 401


def test_ingestion_run_requires_privileged_role():
    response = client.get("/ingestion-runs/not-a-real-run", headers={"Authorization": "Bearer user-token"})
    assert response.status_code == 403
