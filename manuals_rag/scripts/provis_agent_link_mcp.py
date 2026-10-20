#!/usr/bin/env python3
"""Local stdio MCP adapter for the verified PROVIS owner Agent Link.

The node UI remains the authority boundary: this process can only use the
short-lived owner capability written after Ctrl+Shift+A verification. It does
not expose a TCP listener, publish sessions, grant approvals, or retry an
uncertain delivery.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path


if __package__ in (None, ""):
    # Make the checked-out source layout runnable without requiring users to
    # know the repository's internal PYTHONPATH. Packaged builds place the
    # package beside this script and do not need this fallback.
    source_package = Path(__file__).resolve().parents[1] / "packages/inference_fabric/src"
    if source_package.is_dir():
        sys.path.insert(0, str(source_package))

from manuals_rag_inference.agent_link_client import call
from manuals_rag_inference.agent_link_session import load_session


SERVER_NAME = "provis-agent-link"
PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}

TOOLS = [
    {
        "name": "provis_list_published_sessions",
        "description": "List Codex sessions explicitly published through the PROVIS owner panel. Call this before addressing a remote session; use the returned node_id and thread_id exactly.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "provis_message_history",
        "description": "Read recent PROVIS Agent Link delivery status. An uncertain delivery must be inspected in the target Codex session and never resent automatically.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "provis_send_message",
        "description": "Send one user-approved message to an explicitly published remote Codex session. Show the user the exact destination and message first; set confirmation true only after approval. This never grants remote approvals or executes shell commands.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "thread_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "message": {"type": "string", "minLength": 1, "maxLength": 12000},
                "confirmation": {"type": "boolean"},
            },
            "required": ["node_id", "thread_id", "message", "confirmation"],
            "additionalProperties": False,
        },
    },
]


def _tool_result(value: object, *, error: bool = False) -> dict:
    result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}]}
    if error:
        result["isError"] = True
    else:
        result["structuredContent"] = value
    return result


class McpServer:
    """Small dependency-free MCP JSON-RPC server for Codex stdio transport."""

    def __init__(self, session_loader=load_session, caller=call):
        self.session_loader = session_loader
        self.caller = caller
        self.session = None

    def _owner_call(self, action: str, payload: dict | None = None) -> dict:
        if self.session is None:
            self.session = self.session_loader()
        session = self.session
        return self.caller(session["gateway_url"], session["session_token"], action, payload or {})

    def _tool(self, name: str, arguments: object) -> dict:
        if type(arguments) is not dict:
            raise ValueError("Tool arguments must be an object")
        if name == "provis_list_published_sessions":
            return self._owner_call("targets")
        if name == "provis_message_history":
            return self._owner_call("messages")
        if name == "provis_send_message":
            required = {"node_id", "thread_id", "message", "confirmation"}
            if set(arguments) != required:
                raise ValueError("node_id, thread_id, message and confirmation are required")
            if arguments["confirmation"] is not True:
                raise PermissionError("User confirmation is required before sending")
            if type(arguments["node_id"]) is not str or not 1 <= len(arguments["node_id"]) <= 200:
                raise ValueError("Bounded node_id required")
            if type(arguments["thread_id"]) is not str or not 1 <= len(arguments["thread_id"]) <= 200:
                raise ValueError("Bounded thread_id required")
            if type(arguments["message"]) is not str or not 1 <= len(arguments["message"]) <= 12000:
                raise ValueError("Message must contain 1–12000 characters")
            payload = {"id": str(uuid.uuid4()), "node_id": arguments["node_id"], "thread_id": arguments["thread_id"], "message": arguments["message"]}
            result = self._owner_call("send", payload)
            return {"delivery_id": payload["id"], **result}
        raise ValueError("Unknown PROVIS tool")

    def handle(self, request: dict) -> dict | None:
        if type(request) is not dict or request.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None, "error": {"code": -32600, "message": "Invalid JSON-RPC request"}}
        method = request.get("method")
        identifier = request.get("id")
        if identifier is None and method == "notifications/initialized":
            return None
        if method == "initialize":
            requested = (request.get("params") or {}).get("protocolVersion")
            version = requested if requested in PROTOCOL_VERSIONS else "2025-06-18"
            return {"jsonrpc": "2.0", "id": identifier, "result": {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"}, "instructions": "Use Ctrl+Shift+A in the PROVIS node to verify the owner before using these tools."}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": identifier, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": identifier, "result": {"tools": TOOLS}}
        if method == "tools/call":
            try:
                params = request.get("params") or {}
                return {"jsonrpc": "2.0", "id": identifier, "result": _tool_result(self._tool(params.get("name"), params.get("arguments") or {}))}
            except Exception as error:
                return {"jsonrpc": "2.0", "id": identifier, "result": _tool_result({"status": "unavailable_or_unconfirmed", "detail": str(error)}, error=True)}
        return {"jsonrpc": "2.0", "id": identifier, "error": {"code": -32601, "message": "Method not found"}}

    def run(self, input_stream=None, output_stream=None) -> None:
        input_stream = input_stream or sys.stdin
        output_stream = output_stream or sys.stdout
        for line in input_stream:
            if not line.strip():
                continue
            try:
                response = self.handle(json.loads(line))
            except Exception as error:
                response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(error)}}
            if response is not None:
                output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
                output_stream.flush()


def main() -> None:
    McpServer().run()


if __name__ == "__main__":
    main()
