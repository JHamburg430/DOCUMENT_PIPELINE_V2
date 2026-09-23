import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "freeze_heldout_eval_bank.py"
_SPEC = importlib.util.spec_from_file_location("freeze_heldout_eval_bank", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _case():
    return {
        "case_id": "case-1",
        "source_document_id": "heldout-doc",
        "document_version_id": "version-1",
        "source_chunk_id": "chunk-1",
        "expected_snippet": "Install the sensor 20 mm away.",
        "expected_terms": ["sensor", "20 mm"],
    }


def _chunk():
    return {
        "id": "chunk-1",
        "source_document_id": "heldout-doc",
        "document_version_id": "version-1",
        "content": "Procedure:  Install the sensor 20 mm away. Then tighten it.",
        "is_active": True,
    }


def test_freezes_only_source_verified_document_disjoint_cases():
    frozen = _MODULE.verify_and_freeze_cases(
        [_case()],
        {"chunk-1": _chunk()},
        tuning_document_ids={"different-doc"},
        verified_at="2026-09-23T00:00:00+00:00",
    )

    assert frozen[0]["evaluation_split"] == "held_out"
    assert frozen[0]["adjudication"]["status"] == "source_verified"
    assert frozen[0]["adjudication"]["human_reviewed"] is False
    assert len(frozen[0]["adjudication"]["source_chunk_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_document_id": "tuning-doc"}, "overlap"),
        ({"expected_snippet": "not in source"}, "not present"),
    ],
)
def test_rejects_overlap_or_unverifiable_snippet(mutation, message):
    case = {**_case(), **mutation}
    with pytest.raises(ValueError, match=message):
        _MODULE.verify_and_freeze_cases(
            [case],
            {"chunk-1": _chunk()},
            tuning_document_ids={"tuning-doc"},
            verified_at="2026-09-23T00:00:00+00:00",
        )
