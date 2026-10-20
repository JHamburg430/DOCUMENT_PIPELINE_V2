import io
import json

import provis_agent_link_mcp as agent_link_mcp


def request(method, identifier=1, params=None):
    return {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params or {}}


def test_mcp_lists_tools_without_loading_owner_capability():
    loaded = []

    def load():
        loaded.append(True)
        return {"gateway_url": "https://gateway.example", "session_token": "secret"}

    server = agent_link_mcp.McpServer(session_loader=load, caller=lambda *args: {})
    response = server.handle(request("tools/list"))
    assert [tool["name"] for tool in response["result"]["tools"]] == [
        "provis_list_published_sessions",
        "provis_message_history",
        "provis_send_message",
    ]
    assert loaded == []


def test_mcp_requires_explicit_confirmation_and_uses_exact_target():
    calls = []

    def caller(gateway, token, action, payload):
        calls.append((gateway, token, action, payload))
        return {"status": "queued"}

    server = agent_link_mcp.McpServer(
        session_loader=lambda: {"gateway_url": "https://gateway.example", "session_token": "secret"},
        caller=caller,
    )
    denied = server.handle(request("tools/call", params={
        "name": "provis_send_message",
        "arguments": {"node_id": "pc-a", "thread_id": "thread-a", "message": "hello", "confirmation": False},
    }))
    assert denied["result"]["isError"] is True
    assert calls == []

    allowed = server.handle(request("tools/call", identifier=2, params={
        "name": "provis_send_message",
        "arguments": {"node_id": "pc-a", "thread_id": "thread-a", "message": "hello", "confirmation": True},
    }))
    assert allowed["result"]["structuredContent"]["status"] == "queued"
    assert calls[0][2] == "send"
    assert calls[0][3]["node_id"] == "pc-a"
    assert calls[0][3]["thread_id"] == "thread-a"
    assert calls[0][3]["message"] == "hello"


def test_mcp_stdio_handles_initialize_tools_and_notification():
    input_stream = io.StringIO("\n".join([
        json.dumps(request("initialize", params={"protocolVersion": "2024-11-05"})),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps(request("tools/list", identifier=2)),
    ]) + "\n")
    output_stream = io.StringIO()
    agent_link_mcp.McpServer().run(input_stream, output_stream)
    rows = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert rows[0]["result"]["protocolVersion"] == "2024-11-05"
    assert rows[1]["id"] == 2
