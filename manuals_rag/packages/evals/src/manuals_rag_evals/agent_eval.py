from __future__ import annotations

import re
from typing import Any


AGENT_EVALUATION_LAYERS = (
    "tool_selection",
    "candidate_recall",
    "document_retention",
    "hop_dependencies",
    "evidence_sufficiency",
    "grounded_answer",
    "latency_token_cost",
)


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
    candidate_ok = not expected_chunks or expected_chunks.issubset(candidate_chunks)
    candidate_cell = _cell(
        "pass" if candidate_ok else "fail",
        f"candidate evidence retained {len(candidate_hits)}/{len(expected_chunks)} expected chunks",
        expected=len(expected_chunks),
        found=len(candidate_hits),
        missing=sorted(expected_chunks - candidate_chunks),
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

    multi_hop = str(case.get("retrieval_task") or "") == "multi_step_retrieval" or len(case.get("expected_evidence") or []) > 1
    mode = str(plan.get("mode") or trace.get("mode") or "")
    dependency_edges = sum(len(hop.get("depends_on") or []) for hop in hops if isinstance(hop, dict))
    hop_ok = bool(hops)
    if multi_hop:
        hop_ok = len(hops) >= 2 and mode in {"parallel", "dependent"}
        if mode == "dependent":
            hop_ok = hop_ok and dependency_edges > 0
    dependency_cell = _cell(
        "pass" if hop_ok else "fail",
        f"mode={mode or 'missing'}, hops={len(hops)}, dependency_edges={dependency_edges}",
        mode=mode,
        hops=len(hops),
        dependency_edges=dependency_edges,
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
    sufficiency_ok = bool(trace.get("sufficient")) and bool(required_entries) and not unsupported
    sufficiency_cell = _cell(
        "pass" if sufficiency_ok else "fail",
        "all required evidence requirements are supported" if sufficiency_ok else f"unsupported requirements: {', '.join(unsupported) or 'agent marked insufficient'}",
        sufficient=bool(trace.get("sufficient")),
        unsupported=unsupported,
    )

    answer_text = _normalized(answer.get("answer"))
    expected_terms = [_normalized(value) for value in case.get("expected_terms") or [] if value]
    term_hits = [term for term in expected_terms if term and term in answer_text]
    citation_chunks = {
        str(item.get("chunk_id") or "")
        for item in answer.get("citations") or []
        if isinstance(item, dict) and item.get("chunk_id")
    }
    grounding_ok = bool(answer_text) and (not expected_terms or len(term_hits) == len(expected_terms))
    if expected_chunks:
        grounding_ok = grounding_ok and bool(expected_chunks.intersection(citation_chunks))
    grounded_cell = _cell(
        "pass" if grounding_ok else "fail",
        f"answer contains {len(term_hits)}/{len(expected_terms)} expected terms and cites {len(citation_chunks)} chunks",
        expected_terms=expected_terms,
        matched_terms=term_hits,
        citation_chunks=sorted(citation_chunks),
    )

    cost = trace.get("cost") or {}
    measured_elapsed = float(elapsed_ms if elapsed_ms is not None else trace.get("duration_ms") or 0.0)
    token_estimate = int(cost.get("llm_token_estimate") or 0)
    max_latency_ms = float((case.get("source_metadata") or {}).get("max_agent_latency_ms") or 120_000)
    max_tokens = int((case.get("source_metadata") or {}).get("max_agent_token_estimate") or 4_000)
    efficiency_ok = measured_elapsed <= max_latency_ms and token_estimate <= max_tokens
    efficiency_cell = _cell(
        "pass" if efficiency_ok else "fail",
        f"{measured_elapsed:.0f} ms; estimated {token_estimate} control tokens; {int(cost.get('retrieval_calls') or 0)} retrieval calls",
        elapsed_ms=measured_elapsed,
        token_estimate=token_estimate,
        retrieval_calls=int(cost.get("retrieval_calls") or 0),
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
