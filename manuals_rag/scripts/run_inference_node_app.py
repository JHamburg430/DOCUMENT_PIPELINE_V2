#!/usr/bin/env python3
"""Local Connect page followed by the inference node agent."""

from __future__ import annotations

import html
import os
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from manuals_rag_inference.node_agent import _detect_tailscale_ipv4, _load_node_config, _save_node_config, enroll_node, run_node_agent


def _field(form: dict[str, list[str]], name: str, default: str = "") -> str:
    return str((form.get(name) or [default])[0]).strip()


def _page(message: str = "", values: dict[str, str] | None = None) -> bytes:
    values = values or {}
    port_value = values.get("port") or os.getenv("INFERENCE_NODE_PORT", "8765")
    tailscale_ipv4 = _detect_tailscale_ipv4()
    tailscale_agent_url = f"http://{tailscale_ipv4}:{port_value}" if tailscale_ipv4 else ""
    fields = {
        "gateway_url": values.get("gateway_url") or os.getenv("INFERENCE_GATEWAY_URL", ""),
        "pairing_token": "",
        "node_id": values.get("node_id") or f"node-{secrets.token_hex(4)}",
        "agent_url": values.get("agent_url") or os.getenv("INFERENCE_NODE_ADVERTISE_URL", "") or tailscale_agent_url,
        "ollama_url": values.get("ollama_url") or os.getenv("OLLAMA_NODE_URL", "http://127.0.0.1:11434"),
        "tailnet_name": values.get("tailnet_name") or os.getenv("TAILNET_NAME", ""),
        "allowed_applications": values.get("allowed_applications") or os.getenv(
            "INFERENCE_NODE_ALLOWED_APPLICATIONS",
            "manuals_rag,provis_employee_agent,openclaw",
        ),
        "bind": values.get("bind") or os.getenv("INFERENCE_NODE_BIND", "") or (tailscale_ipv4 or ""),
        "port": port_value,
    }
    inputs = "\n".join(
        f'<label>{html.escape(name.replace("_", " ").title())}'
        f'<input name="{name}" value="{html.escape(value, quote=True)}" required></label>'
        for name, value in fields.items()
    )
    notice = f'<div class="message">{html.escape(message)}</div>' if message else ""
    return f"""<!doctype html><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>PROVIS Inference Node connection</title><style>
body{{font:16px system-ui;max-width:680px;margin:40px auto;padding:0 18px;background:#f5f7fa;color:#17202a}}
form{{background:white;padding:24px;border-radius:12px;box-shadow:0 2px 12px #0001}}
label{{display:block;margin:12px 0;text-transform:capitalize}}input{{display:block;width:100%;box-sizing:border-box;padding:10px;margin-top:4px}}
button{{background:#0d7183;color:#fff;border:0;padding:12px 20px;border-radius:6px;font-size:16px}}
.message{{padding:12px;background:#fff3cd;margin-bottom:16px}}
</style><h1>Connect PROVIS Inference Node</h1>
<p>This legacy local page is retained for compatibility. The desktop client uses Microsoft 365 sign-in; use a comma-separated application allowlist for production isolation.</p>
{notice}<form method='post' action='/connect'>{inputs}<button type='submit'>Connect</button></form>""".encode()


def run_setup() -> None:
    config_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", str(Path.home() / ".config/ProVIS/inference-node.json"))
    state = {"connected": False}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, _page())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/connect":
                self._send(404, b"not found")
                return
            form: dict[str, list[str]] = {}
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length > 16384:
                    raise ValueError("form too large")
                form = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
                values = {key: _field(form, key) for key in ("gateway_url", "pairing_token", "node_id", "agent_url", "ollama_url", "tailnet_name", "allowed_applications", "bind", "port")}
                result = enroll_node(
                    gateway_url=values["gateway_url"],
                    pairing_token=values["pairing_token"],
                    node_id=values["node_id"],
                    endpoint=values["ollama_url"],
                    agent_url=values["agent_url"],
                    tailnet_name=values["tailnet_name"],
                )
                result.update({
                    "gateway_url": values["gateway_url"],
                    "allowed_applications": tuple(
                        item.strip() for item in values["allowed_applications"].split(",") if item.strip()
                    ),
                    "bind": values["bind"],
                    "port": int(values["port"]),
                })
                _save_node_config(config_path, result)
                state["connected"] = True
                self._send(200, b"<h1>Connected</h1><p>Starting the inference node agent.</p>")
            except Exception as error:
                safe_values = {key: _field(form, key) for key in ("gateway_url", "node_id", "agent_url", "ollama_url", "tailnet_name", "allowed_applications", "bind", "port")}
                self._send(400, _page(f"Connection failed: {error}", safe_values))

        def log_message(self, format: str, *args) -> None:
            return

    server = HTTPServer((os.getenv("INFERENCE_SETUP_BIND", "127.0.0.1"), int(os.getenv("INFERENCE_SETUP_PORT", "8766"))), Handler)
    while not state["connected"]:
        server.handle_request()
    server.server_close()


def main() -> None:
    config_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", str(Path.home() / ".config/ProVIS/inference-node.json"))
    config = _load_node_config(config_path)
    if not config.get("shared_secret") and not os.getenv("INFERENCE_NODE_SHARED_SECRET"):
        run_setup()
    run_node_agent()


if __name__ == "__main__":
    main()
