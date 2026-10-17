from manuals_rag_retrieval.improvement_loop import build_improvement_record, classify_failure, render_handoff


def test_classifies_retrieval_failure_before_answer_failure():
    record = {"agent_evaluation": {"cells": {
        "candidate_recall": {"status": "fail"},
        "grounded_answer": {"status": "fail"},
    }}}
    assert classify_failure(record) == "retrieval_recall"


def test_builds_review_first_handoff():
    payload = build_improvement_record(
        run_id="run-1",
        dataset="questions.jsonl",
        records=[
            {"case_id": "ok", "agent_evaluation": {"passed": True}},
            {"case_id": "bad", "agent_evaluation": {"passed": False, "cells": {"doc_select": {"status": "fail"}}}, "failure_reasons": ["wrong_document"]},
        ],
        provenance={"git_commit": "abc"},
    )
    assert payload["summary"] == {"total_cases": 2, "passed_cases": 1, "failed_cases": 1, "failure_classes": {"document_selection": 1}}
    assert payload["review"]["status"] == "pending"
    assert "wrong_document" in render_handoff(payload)
