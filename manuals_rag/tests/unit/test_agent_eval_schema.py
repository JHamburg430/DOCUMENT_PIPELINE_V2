import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from manuals_rag_evals.agent_eval_schema import ExpectedEvidenceGraph, build_expected_evidence_graph
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase


def test_expected_evidence_graph_rejects_forward_dependency():
    with pytest.raises(ValidationError):
        ExpectedEvidenceGraph.model_validate(
            {
                "mode": "dependent",
                "category": "dependent_multi_hop",
                "nodes": [
                    {"node_id": "two", "claim": "second", "depends_on": ["one"]},
                    {"node_id": "one", "claim": "first"},
                ],
                "answer_requires": ["one", "two"],
            }
        )


def test_frozen_agent_bank_has_48_valid_unique_cases_and_expected_taxonomy():
    path = Path("tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    assert len(cases) == 48
    assert len({case["case_id"] for case in cases}) == 48
    assert len({" ".join(case["query"].lower().split()).rstrip("?") for case in cases}) == 48
    for case in cases:
        RetrievalEvalCase(**case)
        build_expected_evidence_graph(case)
    counts = Counter(case["expected_evidence_graph"]["category"] for case in cases)
    assert counts == {
        "single_hop_control": 6,
        "parallel_multi_part": 8,
        "dependent_multi_hop": 15,
        "cross_document": 10,
        "exact_structured_lookup": 5,
        "entity_resolution": 3,
        "unanswerable": 1,
    }
