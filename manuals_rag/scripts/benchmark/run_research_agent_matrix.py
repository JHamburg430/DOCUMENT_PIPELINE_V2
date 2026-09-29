#!/usr/bin/env python3
"""Run the evidence-aware research-agent validation matrix.

The controller-contract layer executes the real bounded ReAct controller with
deterministic planner/RAG doubles.  This keeps the matrix repeatable while
still exercising the controller's actual loop, duplicate handling, tool order,
and finalization code.  Live-corpus rows are deliberately reported as
``not_run`` until a corpus-backed runner is supplied; they must never be
silently counted as passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "answering" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "common" / "src"))

# The controller-contract run never contacts Ollama.  Keep this runner
# executable in the lightweight benchmark image as well as the full API
# image, where the production ollama module has optional HTTP dependencies.
if "manuals_rag_common.ollama" not in sys.modules:
    _ollama_stub = types.ModuleType("manuals_rag_common.ollama")
    _ollama_stub.chat_json = lambda **_kwargs: ({}, "")
    sys.modules["manuals_rag_common.ollama"] = _ollama_stub

from validate_research_agent_matrix import MatrixValidationError, validate_matrix  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _query_for(case: dict[str, Any], index: int, observation: dict[str, Any], prior: str) -> str:
    requirements = case["expected"].get("query_requirements") or []
    terms = requirements[min(index, len(requirements) - 1)] if requirements else []
    if not terms:
        terms = observation.get("discovered_terms") or observation.get("headings") or [observation["response_id"]]
    query = " ".join(str(term) for term in terms)
    if index == 0:
        return query or case["turns"][-1]["content"]
    if observation.get("status") == "controller_event":
        return prior
    # A recovery strategy must be observably different even when the fixture
    # has only one lexical requirement (for example an unsupported product).
    # The response id is a deterministic stand-in for the controller's
    # strategy label and prevents the harness from manufacturing a duplicate.
    return f"{case['expected']['interpretation'].get('entity') or ''} {query} {observation.get('response_id') or ''}".strip()


def _scripted_result(observation: dict[str, Any]) -> dict[str, Any]:
    status = observation.get("status")
    if status == "error":
        raise TimeoutError(str(observation.get("error_class") or "scripted RAG error"))
    if status == "controller_event":
        raise AssertionError("controller events must be handled before the RAG double")
    citations = [
        {"chunk_id": str(value), "document_id": "scripted-validation"}
        for value in observation.get("citations", [])
    ]
    satisfied = observation.get("satisfied_claims", [])
    missing = observation.get("missing_claims", [])
    return {
        "answer": "Grounded scripted evidence: " + ", ".join(satisfied or missing or ["no matching evidence"]),
        "insufficient_evidence": status != "sufficient",
        "citations": citations,
        "used_documents": [{"document_id": "scripted-validation", "version": "fixture"}],
        "warnings": [str(observation["rejected_reason"])] if observation.get("rejected_reason") else [],
        "missing_claims": missing,
        "satisfied_claims": satisfied,
        "headings": observation.get("headings", []),
        "discovered_terms": observation.get("discovered_terms", []),
    }


def _run_contract_case(case: dict[str, Any]) -> dict[str, Any]:
    from manuals_rag_answering import react_agent

    observations = list(case.get("rag_observations") or [])
    planner_index = 0
    prior_query = ""
    events: list[dict[str, Any]] = []

    def planner(**_kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal planner_index, prior_query
        if planner_index >= len(observations):
            return {"action": "finish", "query": "", "rationale": "all_required_claims_supported"}, "{}"
        item = observations[planner_index]
        query = _query_for(case, planner_index, item, prior_query)
        if item.get("status") == "controller_event" and not prior_query:
            query = case["turns"][-1]["content"]
        tags = case["expected"].get("required_strategy_tags") or []
        tag = tags[min(planner_index, len(tags) - 1)] if tags else "original"
        action = "calculator" if item.get("tool") == "calculator" else "manuals_rag"
        tag_text = " ".join(dict.fromkeys((tags if len(observations) == 1 else (["original"] if planner_index == 1 and "original" in tags else []) + [tag])))
        rationale = f"{tag_text}: retrieve the next missing evidence"
        planner_index += 1
        return {"action": action, "query": query, "rationale": rationale}, "{}"

    def rag_tool(query: str, _backend: str, _relay: Any) -> dict[str, Any]:
        nonlocal prior_query
        index = next((i for i, item in enumerate(observations) if item.get("status") != "controller_event" and _query_for(case, i, item, prior_query) == query), None)
        if index is None:
            index = min(planner_index - 1, len(observations) - 1)
        item = observations[index]
        prior_query = query
        return _scripted_result(item)

    def emit(event: dict[str, Any]) -> None:
        events.append(dict(event))

    def fake_chat_json(*, purpose: str, **_kwargs: Any) -> tuple[dict[str, Any], str]:
        if purpose == "react_agent.plan_action":
            return planner()
        if purpose == "react_agent.finalize":
            citations = []
            satisfied: set[str] = set()
            for item in observations[:planner_index]:
                satisfied.update(item.get("satisfied_claims", []))
                citations.extend({"chunk_id": str(value), "document_id": "scripted-validation"} for value in item.get("citations", []))
            supported = case["expected"]["terminal_state"] == "supported" and set(case["required_claims"]).issubset(satisfied)
            return {
                "answer": "Validated grounded answer" if supported else "I could not find enough grounded manual evidence to answer that request.",
                "confidence": "high" if supported else "low",
                "citation_indices": list(range(1, len(citations) + 1)),
                "insufficient_evidence": not supported,
                "followup_questions": [],
                "memory_updates": [],
            }, "{}"
        raise AssertionError(f"unexpected chat_json purpose: {purpose}")

    started = time.monotonic()
    try:
        with patch.object(react_agent, "chat_json", side_effect=fake_chat_json):
            result = react_agent.run_react_agent(
                query=case["turns"][-1]["content"],
                backend="scripted-validation",
                rag_tool=rag_tool,
                history=case["turns"][:-1],
                memory=case.get("initial_memory") or [],
                max_tool_calls=case["expected"]["max_rag_calls"],
                event_callback=emit,
            )
    except Exception as error:
        return {"case_id": case["case_id"], "status": "error", "passed": False, "error": f"{type(error).__name__}: {error}"}

    trace = result.get("agent_trace") or {}
    calls = [item for item in trace.get("tool_calls", []) if item.get("action") == "manuals_rag"]
    distinct = len({" ".join(str(item.get("query") or "").lower().split()) for item in calls})
    rationales = " ".join(str(item.get("rationale") or "") for item in trace.get("tool_calls", []))
    event_text = " ".join(json.dumps(event, sort_keys=True) for event in events)
    required = case["expected"]
    failures: list[str] = []
    if not calls or required["first_tool"] != "manuals_rag":
        failures.append("first_tool_not_manuals_rag")
    if not (required["min_rag_calls"] <= len(calls) <= required["max_rag_calls"]):
        failures.append(f"rag_call_count={len(calls)} outside expected range")
    if distinct < required["min_distinct_rag_queries"]:
        failures.append(f"distinct_rag_queries={distinct} below minimum")
    for tag in required.get("required_strategy_tags", []):
        if tag not in rationales and tag not in event_text:
            failures.append(f"missing_strategy_tag:{tag}")
    if required["terminal_state"] == "supported" and result.get("insufficient_evidence"):
        failures.append("supported_case_finished_insufficient")
    if required["terminal_state"] != "supported" and not result.get("insufficient_evidence") and required["terminal_state"] != "clarification":
        failures.append("unsupported_case_answered")
    if "duplicate_query_terminal" in required.get("prohibited_events", []) and "Duplicate RAG query avoided" in event_text:
        failures.append("duplicate_query_terminal")
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "category": case["category"],
        "layer": case["layer"],
        "status": "pass" if not failures else "fail",
        "passed": not failures,
        "failures": failures,
        "observed": {
            "rag_call_count": len(calls),
            "distinct_rag_queries": distinct,
            "insufficient_evidence": bool(result.get("insufficient_evidence")),
            "tool_calls": trace.get("tool_calls", []),
            "events": events,
        },
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
    }


def run(dataset: Path, manifest: Path, output: Path, layer: str = "all") -> dict[str, Any]:
    validation = validate_matrix(dataset, manifest)
    rows = _load_rows(dataset)
    records: list[dict[str, Any]] = []
    for case in rows:
        if layer != "all" and case["layer"] != layer:
            continue
        if case["layer"] == "live_corpus":
            records.append({"case_id": case["case_id"], "title": case["title"], "category": case["category"], "layer": case["layer"], "status": "not_run", "passed": False, "reason": "live corpus runner is not attached; not counted as a pass"})
            continue
        records.append(_run_contract_case(case))
    passed = sum(1 for record in records if record.get("status") == "pass")
    failed = sum(1 for record in records if record.get("status") == "fail")
    not_run = sum(1 for record in records if record.get("status") == "not_run")
    report = {
        "schema": "manuals-rag-research-agent-matrix-result-v1",
        "status": "passed" if failed == 0 and not_run == 0 else ("failed" if failed else "incomplete"),
        "dataset": str(dataset),
        "manifest": str(manifest),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "validation": validation,
        "layer": layer,
        "summary": {"total": len(records), "passed": passed, "failed": failed, "not_run": not_run},
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("tests/fixtures/research_agent_validation_matrix_v1.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("tests/fixtures/research_agent_validation_matrix_v1.manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("test_reports/research_agent_matrix_latest.json"))
    parser.add_argument("--layer", choices=["all", "controller_contract", "live_corpus"], default="all")
    args = parser.parse_args()
    try:
        report = run(args.dataset, args.manifest, args.output, args.layer)
    except (MatrixValidationError, ImportError, ModuleNotFoundError) as error:
        print(json.dumps({"status": "blocked", "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
