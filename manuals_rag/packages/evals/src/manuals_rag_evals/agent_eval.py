from __future__ import annotations

import re
from typing import Any

from manuals_rag_evals.agent_eval_schema import build_expected_evidence_graph


AGENT_EVALUATION_LAYERS = (
    "tool_selection",
    "candidate_recall",
    "document_retention",
    "hop_dependencies",
    "evidence_sufficiency",
    "grounded_answer",
    "latency_token_cost",
)

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
_QUANTITY_ROLES = {
    "angle",
    "count",
    "counts",
    "current",
    "distance",
    "height",
    "interval",
    "limit",
    "line",
    "lines",
    "overlap",
    "overlapping",
    "pressure",
    "range",
    "speed",
    "temperature",
    "total",
    "voltage",
    "width",
}
_VALUE_PATTERN = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"\d+(?:\.\d+)?)\s*(?:vdc|volts?|v|amps?|ma|a|lines?|mm|ms|%|hz|khz|mhz)?"
)


def _canonical_relation_value(value: str) -> str:
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-zA-Z%]+", value.lower())
    unit_aliases = {
        "volt": "v",
        "volts": "v",
        "amp": "a",
        "amps": "a",
        "line": "lines",
    }
    return " ".join(unit_aliases.get(token, _NUMBER_WORDS.get(token, token)) for token in tokens)


def _role_value_relations(text: str) -> dict[str, set[str]]:
    """Extract quantitative role/value bindings without treating values as a bag."""
    relations: dict[str, set[str]] = {}
    normalized = re.sub(r"\s+", " ", text)
    role_pattern = r"[A-Za-z][A-Za-z0-9 /_-]{0,40}?"
    patterns = (
        rf"\b(?P<role>{role_pattern})\s+(?:is|are|was|were|to|:|=)\s+(?P<value>{_VALUE_PATTERN})\b",
        rf"\b(?P<role>{role_pattern})\s+(?P<value>{_VALUE_PATTERN})\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.I):
            role_tokens = re.findall(r"[a-z]+", match.group("role").lower())
            quantity_tokens = [token for token in role_tokens if token in _QUANTITY_ROLES]
            if not quantity_tokens:
                continue
            # Preserve compound roles such as "overlap lines" while excluding
            # generic lead-in prose captured by the permissive relation regex.
            role = " ".join(quantity_tokens[-2:])
            value = _canonical_relation_value(match.group("value"))
            if value:
                relations.setdefault(role, set()).add(value)
    return relations


def _expected_relation_text(case: dict[str, Any]) -> str:
    snippets = [str(case.get("expected_snippet") or "")]
    snippets.extend(
        str(item.get("snippet") or "")
        for item in case.get("expected_evidence") or []
        if isinstance(item, dict)
    )
    return " ".join(snippet for snippet in snippets if snippet)


def _relation_grounding(case: dict[str, Any], answer_text: str) -> dict[str, Any]:
    expected = _role_value_relations(_expected_relation_text(case))
    if len(expected) < 2:
        return {"checked": False, "passed": True, "expected": expected, "answer": {}}
    actual = _role_value_relations(answer_text)
    missing_or_mismatched = {
        role: {"expected": sorted(values), "answer": sorted(actual.get(role, set()))}
        for role, values in expected.items()
        if not actual.get(role, set()).intersection(values)
    }
    return {
        "checked": True,
        "passed": not missing_or_mismatched,
        "expected": {role: sorted(values) for role, values in expected.items()},
        "answer": {role: sorted(values) for role, values in actual.items()},
        "missing_or_mismatched": missing_or_mismatched,
    }


def _cell(status: str, detail: str, **metrics: Any) -> dict[str, Any]:
    return {"status": status, "label": status.upper(), "detail": detail, "metrics": metrics}


def _normalized(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _expected_document_ids(case: dict[str, Any]) -> set[str]:
    values = {str(case.get("source_document_id") or "")}
    values.update(
        str(item.get("source_document_id") or "")
        for item in case.get("expected_evidence") or []
        if isinstance(item, dict)
    )
    return {value for value in values if value}


def _expected_chunk_ids(case: dict[str, Any]) -> set[str]:
    values = {str(value) for value in case.get("expected_source_chunk_ids") or [] if value}
    values.update(
        str(item.get("chunk_id") or "")
        for item in case.get("expected_evidence") or []
        if isinstance(item, dict)
    )
    source_chunk_id = str(case.get("source_chunk_id") or "")
    if source_chunk_id:
        values.add(source_chunk_id)
    return {value for value in values if value}


def score_agent_run(
    case: dict[str, Any],
    *,
    trace: dict[str, Any],
    results: list[dict[str, Any]],
    answer: dict[str, Any] | None = None,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    """Score one agent trace by stage so retrieval and synthesis failures remain distinguishable."""
    answer = answer or {}
    graph = build_expected_evidence_graph(case)
    ledger = trace.get("evidence_ledger") or {}
    plan = trace.get("plan") or {}
    hops = plan.get("hops") or []
    tools = [str(item.get("tool") or item.get("strategy") or "") for item in ledger.values()]
    allowed_tools = {"hybrid", "broad", "dense", "sparse", "structural"}
    expected_tools = {
        str(value)
        for value in (case.get("source_metadata") or {}).get("expected_tools") or []
        if value
    }
    tool_ok = bool(tools) and all(tool in allowed_tools for tool in tools)
    if expected_tools:
        tool_ok = tool_ok and expected_tools.issubset(set(tools))
    tool_cell = _cell(
        "pass" if tool_ok else "fail",
        f"selected {', '.join(tools) or 'no tools'}"
        + (f"; expected {', '.join(sorted(expected_tools))}" if expected_tools else ""),
        selected=tools,
        expected=sorted(expected_tools),
    )

    expected_chunks = _expected_chunk_ids(case)
    candidate_chunks = {
        str(chunk_id)
        for item in ledger.values()
        for chunk_id in item.get("chunk_ids") or []
        if chunk_id
    }
    candidate_hits = expected_chunks.intersection(candidate_chunks)
    required_nodes = [node for node in graph.nodes if node.required]
    node_candidate_coverage = {
        node.node_id: (
            not node.expected_chunk_ids
            or bool(set(node.expected_chunk_ids).intersection(candidate_chunks))
        )
        for node in required_nodes
    }
    candidate_ok = (
        not required_nodes or all(node_candidate_coverage.values())
    ) if graph.expected_outcome == "answerable" else True
    candidate_cell = _cell(
        "pass" if candidate_ok else "fail",
        f"candidate evidence retained {len(candidate_hits)}/{len(expected_chunks)} expected chunks",
        expected=len(expected_chunks),
        found=len(candidate_hits),
        missing=sorted(expected_chunks - candidate_chunks),
        claim_coverage=node_candidate_coverage,
    )

    expected_documents = _expected_document_ids(case)
    retained_documents = {str(item.get("source_document_id") or "") for item in results if item.get("source_document_id")}
    retained_hits = expected_documents.intersection(retained_documents)
    retention_ok = not expected_documents or expected_documents.issubset(retained_documents)
    retention_cell = _cell(
        "pass" if retention_ok else "fail",
        f"final context retained {len(retained_hits)}/{len(expected_documents)} expected documents",
        expected=len(expected_documents),
        found=len(retained_hits),
        missing=sorted(expected_documents - retained_documents),
    )

    mode = str(plan.get("mode") or trace.get("mode") or "")
    dependency_edges = sum(len(hop.get("depends_on") or []) for hop in hops if isinstance(hop, dict))
    hop_ok = bool(hops)
    expected_edges = sum(len(node.depends_on) for node in graph.nodes)
    if graph.mode in {"parallel", "dependent"}:
        hop_ok = len(hops) >= len(graph.nodes) and mode == graph.mode
    if graph.mode == "dependent":
        hop_ok = hop_ok and dependency_edges >= expected_edges
    if graph.mode == "abstain":
        hop_ok = bool(hops)
    dependency_cell = _cell(
        "pass" if hop_ok else "fail",
        f"mode={mode or 'missing'}, hops={len(hops)}, dependency_edges={dependency_edges}",
        mode=mode,
        hops=len(hops),
        dependency_edges=dependency_edges,
        expected_mode=graph.mode,
        expected_dependency_edges=expected_edges,
    )

    required_entries = [item for item in ledger.values() if item.get("required") and not item.get("recovery_for")]
    recovered_targets = {
        str(item.get("recovery_for"))
        for item in ledger.values()
        if item.get("recovery_for") and item.get("sufficient")
    }
    unsupported = [
        str(hop_id)
        for hop_id, item in ledger.items()
        if item.get("required") and not item.get("sufficient") and str(hop_id) not in recovered_targets
    ]
    predicted_sufficient = bool(trace.get("sufficient"))
    if graph.expected_outcome == "insufficient":
        sufficiency_ok = not predicted_sufficient
    else:
        support = trace.get("required_claim_support") or {
            str(hop_id): list((item.get("assessment") or {}).get("supporting_chunk_ids") or item.get("chunk_ids") or [])
            for hop_id, item in ledger.items()
            if item.get("required") and not item.get("recovery_for") and item.get("sufficient")
        }
        context = trace.get("context_assembly") or {}
        sufficiency_ok = (
            predicted_sufficient
            and bool(required_entries)
            and not unsupported
            and all(node_candidate_coverage.values())
            and all(bool(chunks) for chunks in support.values())
            and bool(context.get("all_required_claims_retained", True))
        )
    recoveries = [item for item in ledger.values() if item.get("recovery_for")]
    successful_recoveries = [item for item in recoveries if item.get("sufficient")]
    sufficiency_cell = _cell(
        "pass" if sufficiency_ok else "fail",
        "ledger outcome matches the expected evidence contract" if sufficiency_ok else f"unsupported or false-positive requirements: {', '.join(unsupported) or 'sufficiency mismatch'}",
        sufficient=predicted_sufficient,
        unsupported=unsupported,
        expected_outcome=graph.expected_outcome,
        recovery_attempts=len(recoveries),
        successful_recoveries=len(successful_recoveries),
    )

    answer_text = _normalized(answer.get("answer"))
    relation_grounding = _relation_grounding(case, str(answer.get("answer") or ""))
    expected_terms = [_normalized(value) for value in case.get("expected_terms") or [] if value]
    term_hits = [term for term in expected_terms if term and term in answer_text]
    citation_chunks = {
        str(item.get("chunk_id") or "")
        for item in answer.get("citations") or []
        if isinstance(item, dict) and item.get("chunk_id")
    }
    invalid_citation_chunks = citation_chunks - candidate_chunks
    node_grounding = {
        node.node_id: {
            "terms": all(_normalized(term) in answer_text for term in node.expected_terms if term),
            "citation": not node.expected_chunk_ids
            or bool(set(node.expected_chunk_ids).intersection(citation_chunks)),
        }
        for node in required_nodes
    }
    if graph.expected_outcome == "insufficient":
        grounding_ok = bool(answer.get("insufficient_evidence")) and not citation_chunks
    else:
        grounding_ok = (
            bool(answer_text)
            and (not expected_terms or len(term_hits) == len(expected_terms))
            and all(item["terms"] and item["citation"] for item in node_grounding.values())
            and relation_grounding["passed"]
            and not invalid_citation_chunks
        )
    grounded_cell = _cell(
        "pass" if grounding_ok else "fail",
        f"answer contains {len(term_hits)}/{len(expected_terms)} expected terms and cites {len(citation_chunks)} chunks",
        expected_terms=expected_terms,
        matched_terms=term_hits,
        citation_chunks=sorted(citation_chunks),
        invalid_citation_chunks=sorted(invalid_citation_chunks),
        claim_grounding=node_grounding,
        relation_grounding=relation_grounding,
        insufficient_evidence=bool(answer.get("insufficient_evidence")),
    )

    cost = trace.get("cost") or {}
    measured_elapsed = float(elapsed_ms if elapsed_ms is not None else trace.get("duration_ms") or 0.0)
    measured_tokens = int(cost.get("total_tokens") or 0)
    token_estimate = int(cost.get("llm_token_estimate") or 0)
    token_count = measured_tokens if cost.get("measured") else token_estimate
    max_latency_ms = float((case.get("source_metadata") or {}).get("max_agent_latency_ms") or 120_000)
    max_tokens = int((case.get("source_metadata") or {}).get("max_agent_token_estimate") or 4_000)
    efficiency_ok = measured_elapsed <= max_latency_ms and token_count <= max_tokens
    efficiency_cell = _cell(
        "pass" if efficiency_ok else "fail",
        f"{measured_elapsed:.0f} ms; {token_count} {'measured' if cost.get('measured') else 'estimated'} tokens; {int(cost.get('retrieval_calls') or 0)} retrieval calls",
        elapsed_ms=measured_elapsed,
        token_estimate=token_estimate,
        measured_tokens=measured_tokens,
        model_calls=int(cost.get("model_calls") or 0),
        measured=bool(cost.get("measured")),
        retrieval_calls=int(cost.get("retrieval_calls") or 0),
        recovery_attempts=len(recoveries),
        successful_recoveries=len(successful_recoveries),
        max_latency_ms=max_latency_ms,
        max_token_estimate=max_tokens,
    )

    cells = {
        "tool_selection": tool_cell,
        "candidate_recall": candidate_cell,
        "document_retention": retention_cell,
        "hop_dependencies": dependency_cell,
        "evidence_sufficiency": sufficiency_cell,
        "grounded_answer": grounded_cell,
        "latency_token_cost": efficiency_cell,
    }
    passed = all(cell["status"] == "pass" for cell in cells.values())
    return {"passed": passed, "cells": cells}
