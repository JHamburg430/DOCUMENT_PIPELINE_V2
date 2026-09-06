from __future__ import annotations

from manuals_rag_common import ingestion_progress


def test_initialize_ingestion_steps_records_complete_upload_and_queued_pipeline(monkeypatch):
    statements: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(ingestion_progress, "ensure_ingestion_step_table", lambda: None)
    monkeypatch.setattr(ingestion_progress, "execute", lambda query, params=(): statements.append((query, params)))

    ingestion_progress.initialize_ingestion_steps(
        "run-1",
        upload_details={"filename": "manual.pdf", "size_bytes": 123},
    )

    assert len(statements) == len(ingestion_progress.INGESTION_STEPS)
    upload_params = statements[0][1]
    assert upload_params[1:5] == ("upload", 1, "Upload source", "completed")
    assert '"filename": "manual.pdf"' in str(upload_params[8])
    assert all(statement[1][4] == "queued" for statement in statements[1:])


def test_step_updates_persist_status_details_and_failure(monkeypatch):
    statements: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(ingestion_progress, "execute", lambda query, params=(): statements.append((query, params)))

    ingestion_progress.start_ingestion_step("run-1", "parse", details={"filename": "manual.pdf"})
    ingestion_progress.complete_ingestion_step("run-1", "parse", details={"page_count": 12})
    ingestion_progress.fail_ingestion_step("run-1", "index_chunks", "index unavailable")

    assert "status = 'running'" in statements[0][0]
    assert '"filename": "manual.pdf"' in str(statements[0][1][0])
    assert "status = 'completed'" in statements[1][0]
    assert '"page_count": 12' in str(statements[1][1][0])
    assert "status = 'failed'" in statements[2][0]
    assert statements[2][1] == ("index unavailable", "run-1", "index_chunks")
    assert "status = 'skipped'" in statements[3][0]
    assert statements[3][1] == ("index_chunks", "run-1", "run-1", "index_chunks")
