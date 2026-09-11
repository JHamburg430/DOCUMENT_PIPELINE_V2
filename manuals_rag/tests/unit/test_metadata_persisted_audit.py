import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "audit_persisted_metadata_mrv.py"
_SPEC = importlib.util.spec_from_file_location("audit_persisted_metadata_mrv", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_audit_can_select_a_complete_corpus(monkeypatch):
    captured = {}

    def fake_fetch_all(query, params):
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(_MODULE, "fetch_all", fake_fetch_all)

    report = _MODULE.run(corpus_id="manuals_canary")

    assert "sd.corpus_id = %s" in captured["query"]
    assert captured["params"] == (_MODULE.PIPELINE, "manuals_canary")
    assert report["document_count"] == 0
    assert report["checks_passed"] is False
    assert report["scope_failures"] == ["no_documents_found"]


def test_audit_requires_an_explicit_scope():
    with pytest.raises(ValueError, match="document id or a corpus id"):
        _MODULE.run()
