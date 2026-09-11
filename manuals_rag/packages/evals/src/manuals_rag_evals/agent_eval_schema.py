from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


AgentCaseCategory = Literal[
    "single_hop_control",
    "parallel_multi_part",
    "dependent_multi_hop",
    "cross_document",
    "entity_resolution",
    "exact_structured_lookup",
    "conflicting_evidence",
    "unanswerable",
]

AGENT_CASE_CATEGORIES: frozenset[str] = frozenset(
    {
        "single_hop_control",
        "parallel_multi_part",
        "dependent_multi_hop",
        "cross_document",
        "entity_resolution",
        "exact_structured_lookup",
        "conflicting_evidence",
        "unanswerable",
    }
)


class ExpectedEvidenceNode(BaseModel):
    node_id: str
    claim: str
    answer_facet: str = "fact"
    evidence_role: str = "support"
    required: bool = True
    depends_on: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)


class ExpectedEvidenceGraph(BaseModel):
    mode: Literal["single", "parallel", "dependent", "abstain"]
    category: AgentCaseCategory
    expected_outcome: Literal["answerable", "insufficient"] = "answerable"
    nodes: list[ExpectedEvidenceNode]
    answer_requires: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "ExpectedEvidenceGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Expected evidence node IDs must be unique.")
        known: set[str] = set()
        for node in self.nodes:
            if any(dependency not in known for dependency in node.depends_on):
                raise ValueError("Expected evidence dependencies must reference earlier nodes.")
            known.add(node.node_id)
        if any(node_id not in known for node_id in self.answer_requires):
            raise ValueError("answer_requires must reference expected evidence nodes.")
        if self.expected_outcome == "answerable" and not self.answer_requires:
            raise ValueError("Answerable cases require at least one evidence node.")
        if self.mode == "dependent" and not any(node.depends_on for node in self.nodes):
            raise ValueError("Dependent cases require at least one dependency edge.")
        if self.mode == "abstain" and self.expected_outcome != "insufficient":
            raise ValueError("Abstain graphs must expect an insufficient-evidence outcome.")
        return self


def infer_agent_case_category(case: dict[str, Any]) -> AgentCaseCategory:
    metadata = case.get("source_metadata") or {}
    explicit = str(metadata.get("agent_case_category") or "")
    if explicit in AGENT_CASE_CATEGORIES:
        return explicit  # type: ignore[return-value]
    if str(case.get("retrieval_task") or "") == "unanswerable":
        return "unanswerable"
    generation = str(case.get("generation_method") or "").lower()
    evidence = case.get("expected_evidence") or []
    document_ids = {
        str(item.get("source_document_id") or "")
        for item in evidence
        if isinstance(item, dict) and item.get("source_document_id")
    }
    if len(document_ids) > 1 or "cross_document" in generation:
        return "cross_document"
    if str(case.get("retrieval_task") or "") == "multi_step_retrieval" and re.search(
        r"\b(?:then|using that|after that)\b", str(case.get("query") or ""), flags=re.I
    ):
        return "dependent_multi_hop"
    if any(token in generation for token in ("contextual", "warning_plus_step", "dependency")):
        return "dependent_multi_hop"
    if len(evidence) > 1:
        return "parallel_multi_part"
    if str(case.get("chunk_type") or "") in {"table_record", "spec_record", "procedure_record"}:
        return "exact_structured_lookup"
    return "single_hop_control"


def build_expected_evidence_graph(case: dict[str, Any]) -> ExpectedEvidenceGraph:
    existing = case.get("expected_evidence_graph")
    if isinstance(existing, dict):
        return ExpectedEvidenceGraph.model_validate(existing)

    category = infer_agent_case_category(case)
    if category == "unanswerable":
        return ExpectedEvidenceGraph(
            mode="abstain",
            category=category,
            expected_outcome="insufficient",
            nodes=[],
            answer_requires=[],
        )

    evidence = [item for item in case.get("expected_evidence") or [] if isinstance(item, dict)]
    if not evidence:
        evidence = [
            {
                "chunk_id": case.get("source_chunk_id"),
                "source_document_id": case.get("source_document_id"),
                "field": case.get("chunk_type") or "fact",
                "label": case.get("source_title") or "source evidence",
                "expected_terms": case.get("expected_terms") or [],
            }
        ]
    mode: Literal["single", "parallel", "dependent", "abstain"] = "single"
    if len(evidence) > 1:
        mode = "dependent" if category in {"dependent_multi_hop", "entity_resolution"} else "parallel"
    nodes: list[ExpectedEvidenceNode] = []
    for index, item in enumerate(evidence, start=1):
        node_id = f"claim_{index}"
        field = str(item.get("field") or item.get("evidence_role") or "fact")
        label = str(item.get("label") or item.get("snippet") or f"evidence {index}")
        nodes.append(
            ExpectedEvidenceNode(
                node_id=node_id,
                claim=f"Retrieve {field} evidence for {label}"[:500],
                answer_facet=field,
                evidence_role=str(item.get("evidence_role") or "support"),
                depends_on=[f"claim_{index - 1}"] if mode == "dependent" and index > 1 else [],
                expected_chunk_ids=[str(item["chunk_id"])] if item.get("chunk_id") else [],
                expected_document_ids=[str(item["source_document_id"])] if item.get("source_document_id") else [],
                expected_terms=[str(value) for value in item.get("expected_terms") or [] if value],
            )
        )
    return ExpectedEvidenceGraph(
        mode=mode,
        category=category,
        nodes=nodes,
        answer_requires=[node.node_id for node in nodes if node.required],
    )


def attach_expected_evidence_graph(case: dict[str, Any]) -> dict[str, Any]:
    graph = build_expected_evidence_graph(case)
    metadata = dict(case.get("source_metadata") or {})
    metadata["agent_case_category"] = graph.category
    return {
        **case,
        "source_metadata": metadata,
        "expected_evidence_graph": graph.model_dump(),
    }
