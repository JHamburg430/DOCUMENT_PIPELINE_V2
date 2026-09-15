from __future__ import annotations

import re
from typing import Any

from manuals_rag_common.claim_relations import profile_is_preserved, relation_profile
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
    return {role: set(values) for role, values in relation_profile(text).role_values.items()}


def _expected_relation_text(case: dict[str, Any]) -> str:
    evidence = [item for item in case.get("expected_evidence") or [] if isinstance(item, dict)]
    if not evidence:
        return str(case.get("expected_snippet") or "")

    graph = build_expected_evidence_graph(case)
    required_chunks = {
        chunk_id
        for node in graph.nodes
        if node.required
        for chunk_id in node.expected_chunk_ids
    }
    snippets: list[str] = []
    for item in evidence:
        chunk_id = str(item.get("chunk_id") or "")
        if required_chunks and chunk_id not in required_chunks:
            continue
        snippet = str(item.get("snippet") or "").strip()
        if not snippet:
            continue
        if any(marker in snippet for marker in ("Column headers:", "Row headers:", "Cell value:")):
            # Structured fixture excerpts may be clipped to a character budget.
            # Only derive semantic relations from cells with an explicit trailing
            # row/column boundary; expected terms and exact citations continue to
            # enforce incomplete cells without hallucinating relations from their
            # truncated prefixes or serialization labels.
            match = re.search(
                r"Cell value:\s*(.+?);\s*(?:Row|Column):",
                snippet,
                flags=re.I | re.S,
            )
            if match:
                snippets.append(match.group(1).strip())
            continue
        snippets.append(snippet)
    return " ".join(snippets)


def _relation_grounding(case: dict[str, Any], answer_text: str) -> dict[str, Any]:
    expected_profile = relation_profile(_expected_relation_text(case))
    actual_profile = relation_profile(answer_text)
    checked = bool(expected_profile.role_values or expected_profile.action_polarities)
    if not checked:
        return {"checked": False, "passed": True, "expected": {}, "answer": {}}
    passed, details = profile_is_preserved(expected_profile, actual_profile)
    return {"checked": True, "passed": passed, **details}


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


def _page_set(item: dict[str, Any]) -> set[int]:
    pages = item.get("pages") or []
    if pages:
        return {int(page) for page in pages}
    page_from = item.get("page_from")
    page_to = item.get("page_to")
    if page_from is None or page_to is None:
        return set()
    return set(range(int(page_from), int(page_to) + 1))


def _structured_values(snippet: str) -> list[str]:
    """Return the values from a compact ``Label: value; Label: value`` row."""
    values = []
    for field in re.split(r";\s*", snippet or ""):
        if ":" not in field:
            continue
        _label, value = field.split(":", 1)
        normalized = _normalized(value)
        if normalized:
            values.append(normalized)
    return values


def _result_preserves_expected_evidence(
    result: dict[str, Any],
    *,
    source_document_id: str,
    expected_pages: set[int],
    snippet: str,
) -> bool:
    """Accept a larger parent chunk only when it demonstrably contains the same evidence.

    Same-document or term overlap alone is intentionally insufficient. A parent must
    overlap the expected page and either contain the complete normalized snippet or place
    every value from a structured expected row on one physical row. The only cross-page
    exception is an atomic chunk whose complete normalized content exactly duplicates the
    expected snippet; manuals can repeat the same warning verbatim in multiple sections.
    """
    if str(result.get("source_document_id") or "") != source_document_id:
        return False
    normalized_snippet = _normalized(snippet)
    if not normalized_snippet:
        return False
    content = str(result.get("content") or "")
    normalized_content = _normalized(content)
    result_pages = _page_set(result)
    if expected_pages and (not result_pages or expected_pages.isdisjoint(result_pages)):
        metadata = result.get("metadata") or {}
        chunk_type = str(metadata.get("chunk_type") or result.get("chunk_type") or "")
        return (
            chunk_type in {"atomic_text", "warning_record"}
            and normalized_content == normalized_snippet
        )
    if normalized_snippet in normalized_content:
        return True
    values = _structured_values(snippet)
    if len(values) < 2:
        return False
    return any(
        all(value in _normalized(line) for value in values)
        for line in content.splitlines()
        if line.strip()
    )


def _equivalent_chunk_ids(
    case: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Map each expected chunk to strictly verified parent/aggregate alternatives."""
    expected_pages = set(
        range(int(case.get("page_from") or 0), int(case.get("page_to") or 0) + 1)
    ) if case.get("page_from") is not None and case.get("page_to") is not None else set()
    default_snippet = str(case.get("expected_snippet") or "")
    default_document = str(case.get("source_document_id") or "")
    evidence_by_chunk = {
        str(item.get("chunk_id") or ""): item
        for item in case.get("expected_evidence") or []
        if isinstance(item, dict) and item.get("chunk_id")
    }
    equivalents: dict[str, set[str]] = {}
    for expected_chunk in _expected_chunk_ids(case):
        evidence = evidence_by_chunk.get(expected_chunk, {})
        source_document_id = str(evidence.get("source_document_id") or default_document)
        snippet = str(evidence.get("snippet") or default_snippet)
        pages = _page_set(evidence) or expected_pages
        equivalents[expected_chunk] = {
            str(result.get("chunk_id") or "")
            for result in results
            if result.get("chunk_id")
            and _result_preserves_expected_evidence(
                result,
                source_document_id=source_document_id,
                expected_pages=pages,
                snippet=snippet,
            )
        }
    return equivalents


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
    equivalent_chunks = _equivalent_chunk_ids(case, results)
    candidate_chunks = {
        str(chunk_id)
        for item in ledger.values()
        for chunk_id in item.get("chunk_ids") or []
        if chunk_id
    }
    candidate_hits = {
        chunk_id
        for chunk_id in expected_chunks
        if chunk_id in candidate_chunks
        or bool(equivalent_chunks.get(chunk_id, set()).intersection(candidate_chunks))
    }
    required_nodes = [node for node in graph.nodes if node.required]
    node_candidate_coverage = {
        node.node_id: (
            not node.expected_chunk_ids
            or all(
                chunk_id in candidate_chunks
                or bool(equivalent_chunks.get(chunk_id, set()).intersection(candidate_chunks))
                for chunk_id in node.expected_chunk_ids
            )
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
        missing=sorted(expected_chunks - candidate_hits),
        equivalent_chunks={key: sorted(value) for key, value in equivalent_chunks.items() if value},
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
        # Query-anchor nodes bind requested evidence but do not require a
        # separate retrieval branch of their own.
        hop_ok = len(hops) >= len(required_nodes) and mode == graph.mode
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
            or all(
                chunk_id in citation_chunks
                or bool(equivalent_chunks.get(chunk_id, set()).intersection(citation_chunks))
                for chunk_id in node.expected_chunk_ids
            ),
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
