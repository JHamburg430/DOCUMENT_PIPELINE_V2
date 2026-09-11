import importlib.util
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
