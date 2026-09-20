import importlib.util
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "compare_rerankers.py"
_SPEC = importlib.util.spec_from_file_location("compare_rerankers", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_candidate_text_reads_persisted_stage_evidence():
    candidate = {"chunk_id": "chunk-1", "evidence_text": "source-backed evidence"}

    assert _MODULE._candidate_text(candidate, 6000) == "source-backed evidence"


def test_candidate_text_prefers_full_rerank_document_and_clips_exactly():
    candidate = {
        "content": "short",
        "evidence_text": "bounded",
        "metadata": {"rerank_document": "a" * 7000},
    }

    result = _MODULE._candidate_text(candidate, 6000)

    assert result == "a" * 6000
    assert len(result) == 6000
