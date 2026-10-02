from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
from run_onsite_agent_eval import grade, load_bank  # noqa: E402


def test_frozen_onsite_bank_has_active_source_contracts() -> None:
    rows, manifest = load_bank(
        ROOT / "tests/fixtures/keyence_onsite_agent_eval_v2.jsonl",
        ROOT / "tests/fixtures/keyence_onsite_agent_eval_v2.manifest.json",
    )
    assert len(rows) == manifest["case_count"] == 20
    assert len({row["case_id"] for row in rows}) == 20
    assert len({row["source_document_id"] for row in rows}) >= 10
    assert all(len(row["source_content_sha256"]) == 64 for row in rows)


def test_wrong_manual_citation_does_not_pass_even_with_answer_terms() -> None:
    case = {
        "source_document_id": "expected-manual",
        "source_chunk_id": "expected-chunk",
        "expected_facets": [["24 v"], ["off", "not supplied"]],
    }
    result = {
        "answer": "Check that 24 V is off.",
        "citations": [{"document_id": "different-manual", "chunk_id": "other"}],
        "insufficient_evidence": False,
        "agent_trace": {"tool_calls": [{"action": "manuals_rag"}]},
    }
    scored = grade(case, "langgraph_agent", result, [])
    assert scored["status"] == "fail"
    assert scored["checks"]["answer_facets"]
    assert not scored["checks"]["cited_expected_document"]


def test_mutated_bank_cannot_reuse_frozen_manifest(tmp_path: Path) -> None:
    dataset = ROOT / "tests/fixtures/keyence_onsite_agent_eval_v2.jsonl"
    manifest = ROOT / "tests/fixtures/keyence_onsite_agent_eval_v2.manifest.json"
    changed = tmp_path / "changed.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    rows[0]["question"] += " changed"
    changed.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="changed after freezing"):
        load_bank(changed, manifest)
