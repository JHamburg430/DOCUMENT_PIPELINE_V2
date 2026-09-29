from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from manuals_rag_common.config import settings
from manuals_rag_common.ollama import chat_json


EventCallback = Callable[[dict[str, Any]], None]
RagTool = Callable[[str, str, EventCallback], dict[str, Any]]


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["manuals_rag", "calculator", "finish"]},
        "query": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["action", "query", "rationale"],
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "citation_indices": {"type": "array", "items": {"type": "integer"}},
        "insufficient_evidence": {"type": "boolean"},
        "followup_questions": {"type": "array", "items": {"type": "string"}},
        "memory_updates": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "answer",
        "confidence",
        "citation_indices",
        "insufficient_evidence",
        "followup_questions",
        "memory_updates",
    ],
}


_ARITHMETIC_OPERATORS: dict[type[ast.AST], Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_calculate(expression: str) -> float:
    """Evaluate bounded arithmetic without names, calls, attributes, or indexing."""
    if len(expression) > 160:
        raise ValueError("Calculator expression is too long.")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST, depth: int = 0) -> float:
        if depth > 12:
            raise ValueError("Calculator expression is too deeply nested.")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = float(node.value)
            if abs(value) > 1e12:
                raise ValueError("Calculator operand is outside the allowed range.")
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_OPERATORS:
            value = _ARITHMETIC_OPERATORS[type(node.op)](
                evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
            )
            if abs(value) > 1e15:
                raise ValueError("Calculator result is outside the allowed range.")
            return float(value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC_OPERATORS:
            return float(_ARITHMETIC_OPERATORS[type(node.op)](evaluate(node.operand, depth + 1)))
        raise ValueError("Calculator supports arithmetic expressions only.")

    return evaluate(tree)


def _compact_history(history: list[dict[str, str]], limit: int = 8) -> str:
    rows = []
    for item in history[-limit:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            rows.append(f"{role}: {content[:1600]}")
    return "\n".join(rows) or "(none)"


def _compact_memory(memory: list[str], limit: int = 12) -> str:
    return "\n".join(f"- {str(item)[:500]}" for item in memory[-limit:] if str(item).strip()) or "(none)"


def _observation_text(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return "(none)"
    rows = []
    for index, item in enumerate(observations, start=1):
        rows.append(
            f"Observation {index} | tool={item['tool']} | input={item['input']}\n"
            f"status={item.get('status')}\n{str(item.get('summary') or '')[:5000]}"
        )
    return "\n\n".join(rows)


def _plan_action(
    *,
    query: str,
    history: list[dict[str, str]],
    memory: list[str],
    observations: list[dict[str, Any]],
    first_step: bool,
) -> dict[str, str]:
    allowed = "manuals_rag" if first_step else "manuals_rag, calculator, or finish"
    payload, _raw = chat_json(
        model=settings.ollama_answer_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the bounded ReAct controller for a technical-manuals assistant. "
                    "The manuals_rag tool is the primary and authoritative tool. It runs the existing "
                    "grounded multi-hop RAG agent and returns citations. The first tool call MUST be "
                    "manuals_rag. Use multiple manuals_rag calls when the request is compound, when the "
                    "first result is incomplete or ambiguous, or when a follow-up needs a different search "
                    "formulation. Use calculator only after manual evidence supplies the relevant numbers. "
                    "Never answer a manual fact from memory or conversation history alone. Choose exactly "
                    f"one next action: {allowed}. Keep rationale concise. If finishing, leave query empty."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current request:\n{query}\n\nRecent conversation:\n{_compact_history(history)}\n\n"
                    f"Session memory (context only, not evidence):\n{_compact_memory(memory)}\n\n"
                    f"Tool observations:\n{_observation_text(observations)}"
                ),
            },
        ],
        json_schema=ACTION_SCHEMA,
        think=False,
        timeout=60.0,
        num_predict=350,
        purpose="react_agent.plan_action",
    )
    action = str(payload.get("action") or "").strip()
    if first_step:
        action = "manuals_rag"
    return {
        "action": action,
        "query": str(payload.get("query") or query).strip(),
        "rationale": str(payload.get("rationale") or "").strip(),
    }


def _dedupe_records(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        identity = tuple(str(record.get(key) or "") for key in keys)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(record)
    return output


def _finalize(
    *,
    query: str,
    history: list[dict[str, str]],
    memory: list[str],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    rag_results = [item["result"] for item in observations if item["tool"] == "manuals_rag" and item.get("result")]
    citations = _dedupe_records(
        [citation for result in rag_results for citation in result.get("citations", [])],
        ("chunk_id", "document_id"),
    )
    used_documents = _dedupe_records(
        [document for result in rag_results for document in result.get("used_documents", [])],
        ("document_id", "version"),
    )
    warnings = list(dict.fromkeys(str(warning) for result in rag_results for warning in result.get("warnings", [])))
    sufficient = [result for result in rag_results if not result.get("insufficient_evidence")]
    evidence_rows = []
    for result_index, result in enumerate(rag_results, start=1):
        result_citations = result.get("citations", [])
        citation_indices = []
        for citation in result_citations:
            try:
                citation_indices.append(citations.index(citation) + 1)
            except ValueError:
                continue
        evidence_rows.append(
            f"RAG call {result_index}:\nanswer={str(result.get('answer') or '')[:6000]}\n"
            f"insufficient={bool(result.get('insufficient_evidence'))}\n"
            f"available_citation_indices={citation_indices}"
        )
    calculation_rows = [
        f"Calculator: {item['input']} = {item.get('summary')}"
        for item in observations
        if item["tool"] == "calculator" and item.get("status") == "ok"
    ]
    if not sufficient:
        answer = next((str(result.get("answer") or "") for result in reversed(rag_results) if result.get("answer")), "")
        return {
            "answer": answer or "I could not find enough grounded manual evidence to answer that request.",
            "confidence": "low",
            "used_documents": [],
            "citations": [],
            "warnings": warnings,
            "followup_questions": [],
            "insufficient_evidence": True,
            "memory_updates": [],
        }
    try:
        payload, _raw = chat_json(
            model=settings.ollama_answer_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Produce the final answer using only the supplied grounded RAG answers and calculator "
                        "results. Calculator results may transform numbers supplied by RAG, but they are not "
                        "independent evidence. Combine multiple calls when needed, but do not add unsupported "
                        "facts. citation_indices must "
                        "contain only indices explicitly available for the claims used. Session memory and "
                        "conversation history may resolve references but are not evidence. memory_updates may "
                        "contain only durable user context such as the product/model currently being discussed; "
                        "never store retrieved specifications, secrets, or speculative conclusions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Request:\n{query}\n\nConversation:\n{_compact_history(history)}\n\n"
                        f"Session memory:\n{_compact_memory(memory)}\n\nGrounded tool answers:\n"
                        + "\n\n".join(evidence_rows)
                        + "\n\nCalculator results:\n"
                        + ("\n".join(calculation_rows) or "(none)")
                    ),
                },
            ],
            json_schema=FINAL_SCHEMA,
            think=False,
            timeout=90.0,
            num_predict=max(600, settings.ollama_answer_num_predict),
            purpose="react_agent.finalize",
        )
        selected = [
            citations[index - 1]
            for index in payload.get("citation_indices", [])
            if isinstance(index, int) and 1 <= index <= len(citations)
        ]
        if not selected:
            selected = citations
        return {
            "answer": str(payload.get("answer") or "").strip(),
            "confidence": str(payload.get("confidence") or "medium"),
            "used_documents": used_documents,
            "citations": _dedupe_records(selected, ("chunk_id", "document_id")),
            "warnings": warnings,
            "followup_questions": [str(item) for item in payload.get("followup_questions", [])[:3]],
            "insufficient_evidence": bool(payload.get("insufficient_evidence")),
            "memory_updates": [str(item).strip() for item in payload.get("memory_updates", [])[:4] if str(item).strip()],
        }
    except Exception:
        return {
            "answer": "\n\n".join(str(result.get("answer") or "").strip() for result in sufficient),
            "confidence": min((str(result.get("confidence") or "medium") for result in sufficient), default="medium"),
            "used_documents": used_documents,
            "citations": citations,
            "warnings": warnings,
            "followup_questions": [],
            "insufficient_evidence": False,
            "memory_updates": [],
        }


def run_react_agent(
    *,
    query: str,
    backend: str,
    rag_tool: RagTool,
    history: list[dict[str, str]] | None = None,
    memory: list[str] | None = None,
    max_tool_calls: int = 4,
    event_callback: EventCallback | None = None,
) -> dict[str, Any]:
    """Run a bounded ReAct loop whose first and primary tool is the existing RAG agent."""
    emit = event_callback or (lambda _event: None)
    history = list(history or [])[-12:]
    memory = list(memory or [])[-20:]
    max_tool_calls = max(1, min(6, int(max_tool_calls)))
    observations: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    seen_rag_queries: set[str] = set()
    emit({"event": "react_started", "backend": backend, "max_tool_calls": max_tool_calls})

    for step in range(1, max_tool_calls + 1):
        try:
            action = _plan_action(
                query=query,
                history=history,
                memory=memory,
                observations=observations,
                first_step=step == 1,
            )
        except Exception:
            action = {
                "action": "manuals_rag" if step == 1 else "finish",
                "query": query,
                "rationale": "Planner fallback",
            }
        if action["action"] == "finish" and observations:
            emit({"event": "react_finish_selected", "step": step, "rationale": action["rationale"]})
            break
        if action["action"] == "calculator":
            emit({"event": "tool_call_started", "tool": "calculator", "step": step, "input": action["query"]})
            try:
                value = _safe_calculate(action["query"])
                observation = {"tool": "calculator", "input": action["query"], "status": "ok", "summary": str(value)}
            except Exception as error:
                observation = {"tool": "calculator", "input": action["query"], "status": "error", "summary": str(error)}
            observations.append(observation)
            tool_trace.append({**action, "step": step, "status": observation["status"]})
            emit({"event": "tool_call_completed", "tool": "calculator", "step": step, "observation": observation["summary"]})
            continue

        rag_query = action["query"] or query
        normalized = " ".join(rag_query.lower().split())
        if normalized in seen_rag_queries and observations:
            emit({"event": "react_finish_selected", "step": step, "rationale": "Duplicate RAG query avoided"})
            break
        seen_rag_queries.add(normalized)
        emit({"event": "tool_call_started", "tool": "manuals_rag", "step": step, "input": rag_query})

        def relay(event: dict[str, Any]) -> None:
            emit({"event": "rag_tool_event", "tool": "manuals_rag", "step": step, "source": event})

        try:
            result = rag_tool(rag_query, backend, relay)
        except Exception as error:
            observation = {
                "tool": "manuals_rag",
                "input": rag_query,
                "status": "error",
                "summary": f"RAG tool error: {error}",
            }
            observations.append(observation)
            tool_trace.append({**action, "step": step, "status": "error"})
            emit(
                {
                    "event": "tool_call_completed",
                    "tool": "manuals_rag",
                    "step": step,
                    "status": "error",
                    "error": str(error),
                }
            )
            continue
        observation = {
            "tool": "manuals_rag",
            "input": rag_query,
            "status": "insufficient" if result.get("insufficient_evidence") else "ok",
            "summary": str(result.get("answer") or ""),
            "result": result,
        }
        observations.append(observation)
        tool_trace.append({**action, "step": step, "status": observation["status"]})
        emit(
            {
                "event": "tool_call_completed",
                "tool": "manuals_rag",
                "step": step,
                "status": observation["status"],
                "citation_count": len(result.get("citations", [])),
            }
        )

    final = _finalize(query=query, history=history, memory=memory, observations=observations)
    final["agent_trace"] = {
        "mode": "react",
        "backend": backend,
        "tool_calls": tool_trace,
        "tool_call_count": len(tool_trace),
        "rag_call_count": sum(1 for item in tool_trace if item["action"] == "manuals_rag"),
        "available_tools": ["manuals_rag", "calculator", "session_memory"],
    }
    emit({"event": "react_completed", "result": final, "trace": final["agent_trace"]})
    return final
