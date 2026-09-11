import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "backfill_document_metadata.py"
_SPEC = importlib.util.spec_from_file_location("backfill_document_metadata", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_documents_can_target_multiple_document_ids(monkeypatch):
    captured = {}

    def fake_fetch_all(query, params):
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(_MODULE, "fetch_all", fake_fetch_all)

    assert _MODULE._documents(limit=2, document_ids=["doc-a", "doc-b"]) == []
    assert "sd.id in (%s,%s)" in captured["query"]
    assert captured["query"].rstrip().endswith("limit %s")
    assert captured["params"] == ("doc-a", "doc-b", 2)
    assert "extracted_pipeline_version" in captured["query"]


def test_checkpoint_report_reuses_explicit_path(monkeypatch, tmp_path):
    monkeypatch.setattr(_MODULE, "REPORT_DIR", tmp_path)
    report_path = tmp_path / "checkpoint.json"
    first = _MODULE.BackfillResult(
        document_id="doc-a",
        version_id="version-a",
        source_filename="a.pdf",
        status="planned",
    )
    second = _MODULE.BackfillResult(
        document_id="doc-b",
        version_id="version-b",
        source_filename="b.pdf",
        status="failed",
        error="model timeout",
    )

    assert _MODULE._write_report([first], path=report_path) == report_path
    assert _MODULE._write_report([first, second], path=report_path) == report_path
    rendered = report_path.read_text(encoding="utf-8")

    assert '"document_id": "doc-a"' in rendered
    assert '"document_id": "doc-b"' in rendered
    assert '"error": "model timeout"' in rendered


def test_apply_metadata_commits_document_payload_atomically_before_enqueue(monkeypatch):
    connections = []
    queued = []

    class FakeCursor:
        def __init__(self):
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params):
            self.queries.append((query, params))

        def fetchone(self):
            return {"count": 7}

    class FakeConnection:
        def __init__(self):
            self.cursor_instance = FakeCursor()
            self.commits = 0

        def cursor(self):
            return self.cursor_instance

        def commit(self):
            self.commits += 1

    @contextmanager
    def fake_get_db():
        connection = FakeConnection()
        connections.append(connection)
        yield connection

    metadata = {
        "document_kind": "manual",
        "manufacturer": "Example",
        "companies": ["Example"],
        "product_family": "Alpha",
        "product_model": "ALPHA-1",
        "product_families": ["Alpha"],
        "product_models": ["ALPHA-1"],
        "devices": [],
        "part_numbers": [],
        "protocol_terms": [],
        "settings": [],
        "parameters": [],
        "menu_labels": [],
        "document_topics": [],
        "revision_date": None,
        "metadata_schema_version": 2,
        "metadata_pipeline_version": _MODULE.METADATA_PIPELINE_VERSION,
        "normalized_identifier_aliases": {},
        "routing_product_models": ["ALPHA-1"],
        "routing_part_numbers": [],
        "routing_protocol_terms": [],
        "firmware_applicability": [],
        "software_applicability": [],
        "title": "Alpha manual",
    }
    monkeypatch.setattr(_MODULE, "get_db", fake_get_db)
    monkeypatch.setattr(_MODULE, "enqueue", lambda queue, payload: queued.append((queue, payload)))

    chunk_count, embed_enqueued, warning = _MODULE._apply_metadata(
        {"document_id": "doc-a", "version_id": "version-a"},
        metadata,
        enqueue_embed=True,
    )

    assert chunk_count == 7
    assert embed_enqueued is True
    assert warning is None
    assert len(connections) == 2
    assert len(connections[0].cursor_instance.queries) == 4
    assert connections[0].commits == 1
    assert queued[0][0] == "embed_jobs"
