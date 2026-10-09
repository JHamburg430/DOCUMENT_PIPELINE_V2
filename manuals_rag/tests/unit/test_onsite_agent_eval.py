from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
import run_onsite_agent_eval as onsite_eval  # noqa: E402
from run_onsite_agent_eval import (  # noqa: E402
    _exclusive_output_lock,
    _select_rows,
    grade,
    load_bank,
)


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
    assert scored["retrieval_trace"] is None
    assert scored["evidence_gate"] is None


def test_grade_preserves_gate_and_retrieval_diagnostics() -> None:
    case = {
        "source_document_id": "manual",
        "source_chunk_id": "chunk",
        "expected_facets": [["24 v"]],
    }
    result = {
        "answer": "24 V",
        "citations": [{"document_id": "manual", "chunk_id": "chunk"}],
        "insufficient_evidence": False,
        "agent_trace": {"tool_calls": [{"action": "manuals_rag"}]},
        "retrieval_trace": {"context_assembly": {"retained_chunk_ids": ["chunk"]}},
        "evidence_gate": {"status": "accepted", "verified_quote_count": 1},
    }
    scored = grade(case, "langgraph_agent", result, [])
    assert scored["status"] == "pass"
    assert scored["retrieval_trace"] == result["retrieval_trace"]
    assert scored["evidence_gate"] == result["evidence_gate"]


def test_output_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    lock_path = tmp_path / "onsite.lock"
    with _exclusive_output_lock(lock_path, "first"):
        with pytest.raises(RuntimeError, match="already owned"):
            with _exclusive_output_lock(lock_path, "second"):
                pass


def test_select_rows_preserves_exact_offset_and_limit() -> None:
    rows = [{"case_id": f"case-{index}"} for index in range(6)]
    assert [row["case_id"] for row in _select_rows(rows, offset=2, limit=3)] == [
        "case-2",
        "case-3",
        "case-4",
    ]
    with pytest.raises(ValueError, match="empty"):
        _select_rows(rows, offset=6, limit=1)


def test_loopback_requests_bypass_inherited_proxy(monkeypatch) -> None:
    sentinel = object()
    opened: list[tuple[str, float]] = []

    class Opener:
        def open(self, request, timeout):
            opened.append((request.full_url, timeout))
            return sentinel

    monkeypatch.setattr(onsite_eval, "build_opener", lambda _handler: Opener())
    monkeypatch.setattr(
        onsite_eval,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("loopback request used inherited proxy opener")
        ),
    )
    request = onsite_eval.Request("http://127.0.0.1:18600/agent/stream")
    assert onsite_eval._open_request(request, timeout=12) is sentinel
    assert opened == [("http://127.0.0.1:18600/agent/stream", 12)]


def test_mutated_bank_cannot_reuse_frozen_manifest(tmp_path: Path) -> None:
    dataset = ROOT / "tests/fixtures/keyence_onsite_agent_eval_v2.jsonl"
    manifest = ROOT / "tests/fixtures/keyence_onsite_agent_eval_v2.manifest.json"
    changed = tmp_path / "changed.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    rows[0]["question"] += " changed"
    changed.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="changed after freezing"):
        load_bank(changed, manifest)
