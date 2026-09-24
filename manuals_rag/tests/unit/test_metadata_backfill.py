import importlib.util
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "backfill_document_metadata.py"
_SPEC = importlib.util.spec_from_file_location("backfill_document_metadata", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_documents_can_target_multiple_document_and_corpus_ids(monkeypatch):
    captured = {}

    def fake_fetch_all(query, params):
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(_MODULE, "fetch_all", fake_fetch_all)

    assert _MODULE._documents(
        limit=2,
        document_ids=["doc-a", "doc-b"],
        corpus_ids=["corpus-a", "corpus-b"],
    ) == []
    assert "sd.id in (%s,%s)" in captured["query"]
    assert "sd.corpus_id in (%s,%s)" in captured["query"]
    assert captured["query"].rstrip().endswith("limit %s")
    assert captured["params"] == ("doc-a", "doc-b", "corpus-a", "corpus-b", 2)
    assert "extracted_pipeline_version" in captured["query"]
    assert "authoritative_manufacturer" in captured["query"]


def test_document_segments_prefer_literal_raw_text_for_provenance(monkeypatch):
    monkeypatch.setattr(
        _MODULE,
        "fetch_all",
        lambda *_args, **_kwargs: [
            {
                "text_raw": "VJ-3302 Image processing unit",
                "text_normalized": "VJ: 3302 Image processing unit",
                "page_from": 1,
                "page_to": 1,
                "section_path_json": ["Cover"],
            }
        ],
    )

    segments = _MODULE._document_segments("version-1")

    assert len(segments) == 1
    assert segments[0].text == "VJ-3302 Image processing unit"
    assert segments[0].page_from == 1
    assert segments[0].section_path == ("Cover",)


def test_authoritative_corpus_manufacturer_overrides_extracted_company_noise():
    metadata = {
        "manufacturer": "Intel Corporation",
        "companies": ["Intel Corporation", "Example PLC Vendor"],
        "metadata_evidence": [
            {
                "kind": "company",
                "value": "KEYENCE AMERICA",
                "relation": "primary_manufacturer",
            }
        ],
        "metadata_claims": [
            {
                "kind": "company",
                "value": "KEYENCE",
                "relation": "primary_manufacturer",
            },
            {
                "kind": "company",
                "value": "Example PLC Vendor",
                "relation": "primary_manufacturer",
            },
        ],
    }

    result = _MODULE._apply_authoritative_metadata_defaults(
        {"corpus_id": "manuals_vendor_keyence", "authoritative_manufacturer": "KEYENCE"},
        metadata,
    )

    assert result["manufacturer"] == "KEYENCE"
    assert result["companies"] == ["KEYENCE", "Intel Corporation", "Example PLC Vendor"]
    assert result["metadata_authoritative_defaults"] == {
        "manufacturer": "KEYENCE",
        "source": "corpus_configuration",
    }
    assert result["metadata_evidence"][0]["relation"] == "mentioned"
    assert result["metadata_claims"][0]["relation"] == "primary_manufacturer"
    assert result["metadata_claims"][1]["relation"] == "mentioned"
    assert metadata["manufacturer"] == "Intel Corporation"
    assert metadata["metadata_evidence"][0]["relation"] == "primary_manufacturer"


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


def test_load_report_rejects_duplicate_document_versions(tmp_path):
    report_path = tmp_path / "duplicate.json"
    report_path.write_text(
        json.dumps(
            [
                {"document_id": "doc-a", "version_id": "v1", "source_filename": "a.pdf", "status": "planned"},
                {"document_id": "doc-a", "version_id": "v1", "source_filename": "a.pdf", "status": "planned"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate document/version"):
        _MODULE._load_report(report_path)


@pytest.mark.parametrize(
    ("status", "apply", "enqueue_current", "already_current", "expected"),
    [
        ("planned", False, False, False, True),
        ("failed", False, False, False, False),
        ("skipped_current", False, False, False, False),
        ("skipped_current", False, False, True, True),
        ("planned", True, False, True, False),
        ("applied", True, False, True, True),
        ("applied", True, False, False, False),
        ("embed_enqueued", False, True, True, True),
        ("enqueue_failed", False, True, True, False),
    ],
)
def test_resume_status_is_mode_and_database_aware(
    status, apply, enqueue_current, already_current, expected
):
    result = _MODULE.BackfillResult(
        document_id="doc-a",
        version_id="v1",
        source_filename="a.pdf",
        status=status,
        metadata={"metadata_pipeline_version": _MODULE.METADATA_PIPELINE_VERSION},
        embed_enqueued=status == "embed_enqueued",
    )

    assert _MODULE._can_resume_result(
        result,
        apply=apply,
        enqueue_current=enqueue_current,
        already_current=already_current,
    ) is expected


def test_interrupted_dry_run_resumes_without_reextracting_checkpointed_document(
    monkeypatch, tmp_path
):
    documents = [
        {
            "document_id": "doc-a",
            "version_id": "v1",
            "source_filename": "a.pdf",
            "extracted_version_id": None,
            "extracted_pipeline_version": None,
        },
        {
            "document_id": "doc-b",
            "version_id": "v1",
            "source_filename": "b.pdf",
            "extracted_version_id": None,
            "extracted_pipeline_version": None,
        },
    ]
    extraction_attempts = []
    interrupt_once = {"b.pdf"}

    def infer(filename, *_args, **_kwargs):
        extraction_attempts.append(filename)
        if filename in interrupt_once:
            interrupt_once.remove(filename)
            raise KeyboardInterrupt
        return filename

    monkeypatch.setattr(_MODULE, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(_MODULE, "_ensure_metadata_table", lambda: None)
    monkeypatch.setattr(_MODULE, "_documents", lambda **_kwargs: documents)
    monkeypatch.setattr(_MODULE, "_document_segments", lambda version_id, **_kwargs: [version_id])
    monkeypatch.setattr(_MODULE, "infer_document_metadata_from_segments", infer)
    monkeypatch.setattr(
        _MODULE,
        "_metadata_payload",
        lambda filename: {
            "title": filename,
            "metadata_pipeline_version": _MODULE.METADATA_PIPELINE_VERSION,
        },
    )
    monkeypatch.setattr(sys, "argv", ["backfill_document_metadata.py"])

    with pytest.raises(KeyboardInterrupt):
        _MODULE.main()

    reports = list(tmp_path.glob("document_metadata_backfill_*.json"))
    assert len(reports) == 1
    report_path = reports[0]
    interrupted = json.loads(report_path.read_text(encoding="utf-8"))
    assert [(item["document_id"], item["status"]) for item in interrupted] == [
        ("doc-a", "planned")
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        ["backfill_document_metadata.py", "--resume-report", str(report_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        _MODULE.main()

    assert exc_info.value.code == 0
    assert extraction_attempts == ["a.pdf", "b.pdf", "b.pdf"]
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert [(item["document_id"], item["status"]) for item in saved] == [
        ("doc-a", "planned"),
        ("doc-b", "planned"),
    ]


def test_resume_report_rejects_results_outside_selected_corpus(monkeypatch, tmp_path):
    report_path = tmp_path / "planned.json"
    report_path.write_text(
        json.dumps([
            {
                "document_id": "validation-doc",
                "version_id": "validation-v1",
                "source_filename": "validation.pdf",
                "status": "planned",
            }
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(_MODULE, "_ensure_metadata_table", lambda: None)
    monkeypatch.setattr(
        _MODULE,
        "_documents",
        lambda **_kwargs: [
            {
                "document_id": "production-doc",
                "version_id": "production-v1",
                "source_filename": "production.pdf",
                "extracted_version_id": None,
                "extracted_pipeline_version": None,
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_document_metadata.py",
            "--corpus-id",
            "manuals_vendor_keyence",
            "--resume-report",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        _MODULE.main()

    assert exc_info.value.code == 2


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


def test_exact_planned_report_apply_uses_report_metadata_without_reextracting(monkeypatch, tmp_path):
    document = {
        "document_id": "doc-a",
        "version_id": "v1",
        "source_filename": "a.pdf",
        "extracted_version_id": None,
        "extracted_pipeline_version": None,
    }
    metadata = {
        "title": "Audited title",
        "metadata_pipeline_version": _MODULE.METADATA_PIPELINE_VERSION,
    }
    report_path = tmp_path / "planned.json"
    report_path.write_text(
        json.dumps([
            {
                "document_id": "doc-a",
                "version_id": "v1",
                "source_filename": "a.pdf",
                "status": "planned",
                "metadata": metadata,
                "error": None,
                "chunk_count": 0,
                "embed_enqueued": False,
                "warning": None,
            }
        ]),
        encoding="utf-8",
    )
    report_sha256 = _MODULE._sha256(report_path)
    result_path = tmp_path / "applied.json"
    applied = []

    monkeypatch.setattr(_MODULE, "_ensure_metadata_table", lambda: None)
    monkeypatch.setattr(_MODULE, "_documents", lambda **_kwargs: [document])
    monkeypatch.setattr(
        _MODULE,
        "infer_document_metadata_from_segments",
        lambda *_args, **_kwargs: pytest.fail("exact report apply must not re-extract"),
    )
    monkeypatch.setattr(
        _MODULE,
        "_apply_metadata",
        lambda doc, payload, **_kwargs: (applied.append((doc, payload)) or (5, True, None)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_document_metadata.py",
            "--apply",
            "--document-id",
            "doc-a",
            "--resume-report",
            str(report_path),
            "--apply-planned-report",
            "--report-sha256",
            report_sha256,
            "--result-report",
            str(result_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        _MODULE.main()

    assert exc_info.value.code == 0
    assert applied == [(document, metadata)]
    assert json.loads(report_path.read_text(encoding="utf-8"))[0]["status"] == "planned"
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved[0]["status"] == "applied"
    assert saved[0]["metadata"] == metadata


def test_exact_planned_report_apply_rejects_hash_mismatch(monkeypatch, tmp_path):
    report_path = tmp_path / "planned.json"
    report_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(_MODULE, "_ensure_metadata_table", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_document_metadata.py",
            "--apply",
            "--limit",
            "1",
            "--resume-report",
            str(report_path),
            "--apply-planned-report",
            "--report-sha256",
            "0" * 64,
            "--result-report",
            str(tmp_path / "applied.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        _MODULE.main()

    assert exc_info.value.code == 2
