from manuals_rag_answering import react_agent


def _rag_answer(answer: str, chunk_id: str) -> dict:
    return {
        "answer": answer,
        "confidence": "high",
        "used_documents": [{"document_id": "doc", "version": "v1", "title": "Manual"}],
        "citations": [{"chunk_id": chunk_id, "document_id": "doc", "pages": [1]}],
        "warnings": [],
        "followup_questions": [],
        "insufficient_evidence": False,
    }


def test_react_agent_uses_rag_first_and_can_call_it_multiple_times(monkeypatch):
    model_outputs = iter(
        [
            ({"action": "manuals_rag", "query": "find voltage", "rationale": "first fact"}, ""),
            ({"action": "manuals_rag", "query": "find current", "rationale": "second fact"}, ""),
            ({"action": "finish", "query": "", "rationale": "complete"}, ""),
            (
                {
                    "answer": "Voltage and current are both grounded.",
                    "confidence": "high",
                    "citation_indices": [1, 2],
                    "insufficient_evidence": False,
                    "followup_questions": [],
                    "memory_updates": ["User is working with model CA-U5."],
                },
                "",
            ),
        ]
    )
    monkeypatch.setattr(react_agent, "chat_json", lambda **_kwargs: next(model_outputs))
    calls = []
    events = []

    def rag_tool(query, backend, relay):
        calls.append((query, backend))
        relay({"event": "plan_completed", "plan": {"hops": []}})
        return _rag_answer(f"Answer for {query}", f"chunk-{len(calls)}")

    result = react_agent.run_react_agent(
        query="What are the voltage and current?",
        backend="langgraph_agent",
        rag_tool=rag_tool,
        history=[{"role": "user", "content": "We are discussing CA-U5."}],
        memory=["Current product is CA-U5."],
        max_tool_calls=4,
        event_callback=events.append,
    )

    assert calls == [("find voltage", "langgraph_agent"), ("find current", "langgraph_agent")]
    assert result["agent_trace"]["rag_call_count"] == 2
    assert [citation["chunk_id"] for citation in result["citations"]] == ["chunk-1", "chunk-2"]
    assert result["memory_updates"] == ["User is working with model CA-U5."]
    assert events[1]["event"] == "tool_call_started"
    assert events[1]["tool"] == "manuals_rag"


def test_react_agent_forces_rag_as_first_tool(monkeypatch):
    outputs = iter(
        [
            ({"action": "finish", "query": "", "rationale": "incorrect early finish"}, ""),
            ({"action": "finish", "query": "", "rationale": "done"}, ""),
            (
                {
                    "answer": "Grounded answer.",
                    "confidence": "high",
                    "citation_indices": [1],
                    "insufficient_evidence": False,
                    "followup_questions": [],
                    "memory_updates": [],
                },
                "",
            ),
        ]
    )
    monkeypatch.setattr(react_agent, "chat_json", lambda **_kwargs: next(outputs))
    calls = []

    def rag_tool(query, _backend, _relay):
        calls.append(query)
        return _rag_answer("Grounded answer.", "chunk-1")

    result = react_agent.run_react_agent(
        query="What is the rating?",
        backend="llamaindex_agent",
        rag_tool=rag_tool,
        max_tool_calls=2,
    )

    assert calls == ["What is the rating?"]
    assert result["agent_trace"]["tool_calls"][0]["action"] == "manuals_rag"


def test_calculator_rejects_code_execution():
    assert react_agent._safe_calculate("(12 * 5) / 3") == 20
    try:
        react_agent._safe_calculate("__import__('os').system('id')")
    except ValueError as error:
        assert "arithmetic" in str(error)
    else:
        raise AssertionError("Unsafe calculator input was accepted")
