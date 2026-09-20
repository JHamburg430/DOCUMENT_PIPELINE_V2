import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from manuals_rag_evals.agent_eval_schema import (
    ExpectedEvidenceGraph,
    attach_expected_evidence_graph,
    build_expected_evidence_graph,
)
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


def test_troubleshooting_error_text_is_binding_anchor_not_answer_requirement():
    case = attach_expected_evidence_graph(
        {
            "generation_method": "table_sibling_error_cause_action",
            "source_metadata": {"agent_case_category": "parallel_multi_part"},
            "expected_evidence": [
                {"chunk_id": "message", "field": "error message", "expected_terms": ["error"]},
                {"chunk_id": "cause", "field": "cause", "expected_terms": ["buffer", "full"]},
                {
                    "chunk_id": "action",
                    "field": "corrective action",
                    "expected_terms": ["clear", "buffer"],
                },
            ],
        }
    )

    graph = ExpectedEvidenceGraph.model_validate(case["expected_evidence_graph"])
    assert [node.required for node in graph.nodes] == [False, True, True]
    assert graph.answer_requires == ["claim_2", "claim_3"]
    assert case["expected_terms"] == ["buffer", "full", "clear"]


def test_contextual_procedure_header_is_anchor_for_one_direct_claim():
    case = attach_expected_evidence_graph(
        {
            "generation_method": "contextual_procedure_plus_section_evidence",
            "source_metadata": {"agent_case_category": "single_hop_control"},
            "expected_terms": ["procedure", "multiple", "flowchart", "branched"],
            "expected_evidence": [
                {
                    "chunk_id": "heading",
                    "expected_terms": ["procedure", "multiple"],
                },
                {
                    "chunk_id": "rule",
                    "expected_terms": ["flowchart", "branched"],
                },
            ],
        }
    )

    graph = ExpectedEvidenceGraph.model_validate(case["expected_evidence_graph"])
    assert graph.mode == "single"
    assert [node.required for node in graph.nodes] == [False, True]
    assert graph.answer_requires == ["claim_2"]
    assert case["expected_terms"] == ["flowchart", "branched"]


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
            "single_hop_control": 12,
        "parallel_multi_part": 8,
        "dependent_multi_hop": 7,
        "cross_document": 10,
        "exact_structured_lookup": 5,
            "entity_resolution": 3,
            "conflicting_evidence": 2,
            "unanswerable": 1,
    }
