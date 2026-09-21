import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "compare_embedding_models.py"
_SPEC = importlib.util.spec_from_file_location("compare_embedding_models", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_dense_candidates_require_nonempty_persisted_evidence():
    case = {
        "case_id": "case-1",
        "baseline": {
            "stage_snapshots": [{
                "stage": "dense",
                "results": [{"chunk_id": "chunk-1", "evidence_text": ""}],
            }],
        },
    }

    with pytest.raises(ValueError, match="empty evidence_text"):
        _MODULE._dense_candidates(case)


def test_pool_hash_covers_ids_and_exact_text():
    original = [{"chunk_id": "chunk-1", "text": "evidence"}]

    assert _MODULE._pool_hash(original) != _MODULE._pool_hash([
        {"chunk_id": "chunk-1", "text": "changed evidence"},
    ])


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensions differ"):
        _MODULE._cosine([1.0], [1.0, 0.0])
