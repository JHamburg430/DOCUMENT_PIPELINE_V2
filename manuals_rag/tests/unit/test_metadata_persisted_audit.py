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


def test_audit_rejects_unresolved_and_cross_scope_cover_routing():
    row = {
        "document_id": "doc-1",
        "source_filename": "SR-1000_guide.pdf",
        "title": "SIEMENS S7-1500 SERIES",
        "ingest_status": "indexed",
        "chunk_count": 2,
        "mrv_chunk_count": 2,
        "metadata_json": {
            "metadata_schema_version": 2,
            "metadata_pipeline_version": _MODULE.PIPELINE,
            "title": "SIEMENS S7-1500 SERIES",
            "routing_product_models": ["S7-1500"],
            "metadata_claims": [
                {
                    "value": "S7-1500",
                    "kind": "product_model",
                    "relation": "mentioned",
                    "source_method": "opening_title_candidate",
                    "verification_status": "confirmed",
                    "confidence": 0.9,
                    "grounded": True,
                    "source_quote": "SIEMENS S7-1500 SERIES",
                    "page_from": 1,
                },
                {
                    "value": "SR-1000",
                    "kind": "product_model",
                    "relation": "mentioned",
                    "source_method": "upload_identity_page_grounded",
                    "verification_status": "unresolved",
                    "confidence": 0.45,
                    "grounded": True,
                    "source_quote": "SR-1000",
                    "page_from": 1,
                },
            ],
        },
    }

    result = _MODULE._audit_document(row)

    assert "unresolved_claims_persisted" in result["failures"]
    assert "routing_identifier_without_scoped_relationship" in result["failures"]


def test_audit_rejects_late_compatible_model_as_document_routing_scope():
    row = {
        "document_id": "doc-2",
        "source_filename": "KV-X_manual.pdf",
        "title": "KV-X Manual",
        "ingest_status": "indexed",
        "chunk_count": 2,
        "mrv_chunk_count": 2,
        "metadata_json": {
            "metadata_schema_version": 2,
            "metadata_pipeline_version": _MODULE.PIPELINE,
            "title": "KV-X Manual",
            "routing_product_models": ["KV-X", "CV500"],
            "metadata_claims": [
                {
                    "value": "KV-X",
                    "kind": "product_model",
                    "relation": "primary_product",
                    "source_method": "page_aware_model_extraction",
                    "verification_status": "confirmed",
                    "confidence": 0.9,
                    "grounded": True,
                    "source_quote": "KV-X Manual",
                    "page_from": 1,
                },
                {
                    "value": "CV500",
                    "kind": "product_model",
                    "relation": "compatible_with",
                    "source_method": "page_aware_model_extraction",
                    "verification_status": "confirmed",
                    "confidence": 0.9,
                    "grounded": True,
                    "source_quote": "Compatible model CV500",
                    "page_from": 43,
                },
            ],
        },
    }

    result = _MODULE._audit_document(row)

    assert "routing_identifier_without_scoped_relationship" in result["failures"]


def test_audit_rejects_download_call_to_action_as_title():
    row = {
        "document_id": "doc-3",
        "source_filename": "VJ-H500CX_Datasheet.pdf",
        "title": "VJ-H500CX Datasheet",
        "ingest_status": "indexed",
        "chunk_count": 1,
        "mrv_chunk_count": 1,
        "metadata_json": {
            "metadata_schema_version": 2,
            "metadata_pipeline_version": _MODULE.PIPELINE,
            "title": "Download CAD file or product manual for larger image and more detail.",
            "routing_product_models": ["VJ-H500CX"],
            "metadata_claims": [
                {
                    "value": "VJ-H500CX",
                    "kind": "product_model",
                    "relation": "primary_product",
                    "source_method": "page_aware_model_extraction",
                    "verification_status": "confirmed",
                    "confidence": 0.9,
                    "grounded": True,
                    "source_quote": "Model VJ-H500CX",
                    "page_from": 1,
                }
            ],
        },
    }

    result = _MODULE._audit_document(row)

    assert "boilerplate_selected_as_title" in result["failures"]


def test_current_pipeline_stamp_does_not_mask_stale_chunk_scope(monkeypatch):
    captured = {}
    def fetch(query, params):
        captured["query"] = query
        return [{"document_id": "doc", "ingest_status": "indexed", "chunk_count": 3,
                 "mrv_chunk_count": 3, "scope_mismatch_chunk_count": 1,
                 "version_mismatch_chunk_count": 1,
                 "metadata_json": {"metadata_schema_version": 2,
                                   "metadata_pipeline_version": _MODULE.PIPELINE}}]
    monkeypatch.setattr(_MODULE, "fetch_all", fetch)
    report = _MODULE.run(["doc"])
    assert not report["checks_passed"]
    assert "chunk_scope_metadata_mismatch" in report["documents"][0]["failures"]
    assert "chunk_document_version_mismatch" in report["documents"][0]["failures"]
    for field in _MODULE.PROPAGATED_SCOPE_FIELDS:
        assert f"rc.metadata_json->'{field}' is distinct from dme.metadata_json->'{field}'" in captured["query"]
