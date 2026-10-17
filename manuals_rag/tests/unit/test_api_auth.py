import base64
import hashlib
import hmac
import json
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app
from manuals_rag_permissions import auth as auth_module


client = TestClient(app)


def test_search_requires_authentication():
    response = client.post("/search", json={"query": "test", "corpus_ids": ["manuals_vendor_keyence"]})
    assert response.status_code == 401


def test_version_exposes_runtime_provenance_without_credentials(monkeypatch):
    monkeypatch.setenv("SOURCE_REVISION", "revision-under-test")
    monkeypatch.setenv("SOURCE_BRANCH", "branch-under-test")
    monkeypatch.setenv("BUILD_ID", "build-under-test")

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["service"] == "manuals-rag-api"
    assert response.json()["revision"] == "revision-under-test"
    assert response.json()["branch"] == "branch-under-test"
    assert response.json()["build_id"] == "build-under-test"
    assert response.headers["x-request-id"]


def test_request_id_is_preserved_for_auditable_calls():
    response = client.get("/version", headers={"X-Request-ID": "trace-test-42"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "trace-test-42"


def test_rate_limiter_fails_closed_in_production_when_redis_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "settings",
        replace(api_main.settings, app_env="production", rate_limit_enabled=True),
    )
    monkeypatch.setattr(api_main, "redis_client", lambda: (_ for _ in ()).throw(RuntimeError("redis")))

    response = client.post(
        "/search",
        headers={"Authorization": "Bearer user-token"},
        json={"query": "test", "corpus_ids": ["manuals_vendor_keyence"]},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Rate-limit service unavailable."}


def test_rate_limiter_rejects_requests_over_the_fixed_window(monkeypatch):
    class Pipeline:
        def incr(self, _key):
            return self

        def expire(self, _key, _seconds):
            return self

        def execute(self):
            return [121, True]

    class FakeRedis:
        def pipeline(self, transaction=True):
            assert transaction is True
            return Pipeline()

    monkeypatch.setattr(api_main, "redis_client", lambda: FakeRedis())
    response = client.post(
        "/search",
        headers={"Authorization": "Bearer user-token"},
        json={"query": "test", "corpus_ids": ["manuals_vendor_keyence"]},
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"


def test_production_startup_rejects_local_bootstrap_defaults(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "settings",
        replace(api_main.settings, app_env="production"),
    )

    with pytest.raises(RuntimeError, match="Unsafe production configuration"):
        api_main._validate_production_configuration()


def test_hs256_jwt_auth_maps_subject_role_and_tenant(monkeypatch):
    secret = "test-secret"
    monkeypatch.setattr(
        auth_module,
        "settings",
        replace(
            auth_module.settings,
            auth_mode="jwt_hs256",
            auth_jwt_secret=secret,
            auth_jwt_issuer="issuer",
            auth_jwt_audience="manuals-rag",
        ),
    )
    encode = lambda value: base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    header = encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = encode(json.dumps({
        "sub": "employee-42",
        "role": "end_user",
        "tenant_id": "tenant-42",
        "iss": "issuer",
        "aud": "manuals-rag",
        "exp": time.time() + 60,
    }).encode())
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = encode(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
    principal = auth_module._jwt_hs256_principal(f"{header}.{payload}.{signature}")

    assert principal.subject == "employee-42"
    assert principal.role == "end_user"
    assert principal.tenant_id == "tenant-42"


def test_readiness_fails_closed_without_backing_services(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_one", lambda _query: (_ for _ in ()).throw(RuntimeError("db")))
    monkeypatch.setattr(api_main.httpx, "Client", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("qdrant")))
    monkeypatch.setattr(api_main, "redis_client", lambda: type("RedisProbe", (), {"ping": lambda self: (_ for _ in ()).throw(RuntimeError("redis"))})())
    monkeypatch.setattr(api_main, "reranker_residency_status", lambda: "unavailable")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "postgres": "unavailable", "qdrant": "unavailable", "redis": "unavailable",
            "reranker": "unavailable", "embedding_model": "unavailable",
            "retrieval_verifier": "unavailable", "answer_model": "unavailable",
        },
    }


def test_ingestion_run_requires_privileged_role():
    response = client.get("/ingestion-runs/not-a-real-run", headers={"Authorization": "Bearer user-token"})
    assert response.status_code == 403


def test_search_rejects_corpus_outside_principal_tenant(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_all", lambda _query, _params: [])

    response = client.post(
        "/search",
        headers={"Authorization": "Bearer user-token"},
        json={"query": "test", "corpus_ids": ["other-tenant-corpus"]},
    )

    assert response.status_code == 403


def test_search_returns_redacted_503_when_authorization_store_is_unavailable(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_all", lambda *_args: (_ for _ in ()).throw(RuntimeError("db secret")))

    response = client.post(
        "/search",
        headers={"Authorization": "Bearer user-token", "X-Request-ID": "auth-failure-1"},
        json={"query": "test", "corpus_ids": ["manuals_vendor_keyence"]},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Corpus authorization service is unavailable."}
    assert response.headers["x-request-id"] == "auth-failure-1"
    assert "db secret" not in response.text


def test_search_returns_redacted_503_when_retrieval_is_unavailable(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_all", lambda *_args: [{"id": "manuals_vendor_keyence"}])
    monkeypatch.setattr(api_main, "retrieve", lambda *_args: (_ for _ in ()).throw(RuntimeError("qdrant secret")))

    response = client.post(
        "/search",
        headers={"Authorization": "Bearer user-token"},
        json={"query": "test", "corpus_ids": ["manuals_vendor_keyence"]},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Retrieval service is unavailable."}
    assert "qdrant secret" not in response.text


def test_search_rejects_oversized_query_before_execution():
    response = client.post(
        "/search",
        headers={"Authorization": "Bearer user-token"},
        json={"query": "x" * 12001, "corpus_ids": ["manuals_vendor_keyence"]},
    )

    assert response.status_code == 422


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
