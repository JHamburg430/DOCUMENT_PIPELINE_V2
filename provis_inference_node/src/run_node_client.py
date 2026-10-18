#!/usr/bin/env python3
"""Small Tk desktop client for connecting and operating an inference node."""

from __future__ import annotations

import os
import platform
import queue
import re
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from urllib.parse import urlparse

from provis_inference_node.node_agent import (
    _detect_tailscale_ipv4,
    _load_node_config,
    _save_node_config,
    create_node_server,
    enroll_node,
)
from provis_inference_node.entra_client import (
    enroll_node_with_entra,
    fetch_enrollment_config,
    poll_device_token,
    request_device_code,
)
from provis_inference_node.connect_diagnostics import append_connect_diagnostic
from provis_inference_node.ui_helpers import copy_endpoint_to_clipboard, local_broker_endpoints


def _default_config_path() -> str:
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return str(Path(os.environ["LOCALAPPDATA"]) / "ProVIS" / "inference-node.json")
    config_home = os.getenv("XDG_CONFIG_HOME", "").strip()
    if config_home:
        return str(Path(config_home).expanduser() / "ProVIS" / "inference-node.json")
    return str(Path.home() / ".config" / "ProVIS" / "inference-node.json")


def _is_managed_gateway_url(value: object) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme.lower() == "https" and bool(parsed.hostname) and not any(
        (parsed.username, parsed.password, parsed.query, parsed.fragment)
    )


def _managed_config_candidates() -> tuple[Path, ...]:
    """Return deterministic managed-config locations, never based on cwd.

    Source launchers may explicitly provide ``PROVIS_NODE_BOOTSTRAP_CONFIG``.
    Frozen builds use PyInstaller's extracted data directory first, then an
    administrator-controlled sidecar beside the executable.  The source/bundle
    layout checks both the script directory and its parent because the source
    launcher lives under ``src/`` while the managed JSON is at bundle root.
    """
    candidates: list[Path] = []
    frozen = bool(getattr(sys, "frozen", False))
    if not frozen:
        configured = os.getenv("PROVIS_NODE_BOOTSTRAP_CONFIG", "").strip()
        if configured:
            candidates.append(Path(configured).expanduser())
    if frozen:
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(Path(str(meipass)) / "provis-node-config.json")
        candidates.append(Path(sys.executable).resolve().with_name("provis-node-config.json"))
    script_dir = Path(__file__).resolve().parent
    candidates.extend((script_dir / "provis-node-config.json", script_dir.parent / "provis-node-config.json"))
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _load_managed_gateway_url() -> str:
    for candidate in _managed_config_candidates():
        try:
            config = _load_node_config(str(candidate))
        except (OSError, ValueError, TypeError):
            continue
        gateway_url = str(config.get("gateway_url") or "").strip()
        if _is_managed_gateway_url(gateway_url):
            return gateway_url.rstrip("/")
    if not getattr(sys, "frozen", False):
        for variable in ("PROVIS_INFERENCE_GATEWAY_URL", "INFERENCE_GATEWAY_URL"):
            gateway_url = os.getenv(variable, "").strip()
            if _is_managed_gateway_url(gateway_url):
                return gateway_url.rstrip("/")
    return ""


def _default_node_id() -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", platform.node()).strip("-")
    return value or "provis-inference-node"


class InferenceNodeClient(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PROVIS Inference Node")
        self.geometry("920x650")
        self.minsize(760, 520)
        self.protocol("WM_DELETE_WINDOW", self._close_requested)
        self.config_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", _default_config_path())
        self.connect_diagnostics_path = str(Path(self.config_path).with_name("inference-node-diagnostics.jsonl"))
        self.server = None
        self.agent = None
        self.gateway_url = ""
        self._verification_url = ""
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.fields: dict[str, tk.StringVar] = {
            key: tk.StringVar()
            for key in (
                "gateway_url", "entra_tenant_id", "entra_client_id", "entra_scope",
                "pairing_token", "node_id", "agent_url", "ollama_url", "tailnet_name",
                "allowed_applications", "port",
            )
        }
        self.status_var = tk.StringVar(value="Not connected")
        self.inference_var = tk.StringVar(value="Inference disabled")
        self.local_endpoint_var = tk.StringVar(value="Available after node connects")
        self.local_endpoint_copy_button = None
        self.device_code_var = tk.StringVar()
        self.device_message_var = tk.StringVar()
        self._build_ui()
        self._load_saved_values()
        self.after(250, self._drain_worker_queue)
        self.after(1000, self._refresh_activity)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="PROVIS Inference Node", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(root, text="Connect this machine to the private PROVIS inference gateway. No Tailscale key is entered here.").pack(anchor="w", pady=(2, 12))

        connection = ttk.LabelFrame(root, text="Connect / login", padding=10)
        connection.pack(fill="x")
        ttk.Label(connection, text="This device is preconfigured by PROVIS. Sign in with Microsoft 365 to enroll it; node settings are discovered automatically.").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self.entra_button = ttk.Button(connection, text="Sign in with Microsoft 365", command=self._entra_connect)
        self.entra_button.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(connection, textvariable=self.status_var).grid(row=1, column=1, columnspan=3, sticky="w", pady=(4, 0))
        connection.columnconfigure(1, weight=1)
        device_prompt = ttk.Frame(connection)
        device_prompt.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Label(device_prompt, text="Microsoft code:").pack(side="left")
        self.device_code_entry = ttk.Entry(device_prompt, textvariable=self.device_code_var, width=18, state="readonly")
        self.device_code_entry.pack(side="left", padx=(8, 6))
        self.copy_code_button = ttk.Button(device_prompt, text="Copy code", command=self._copy_device_code, state="disabled")
        self.copy_code_button.pack(side="left")
        self.open_signin_button = ttk.Button(device_prompt, text="Open sign-in", command=self._open_verification_url, state="disabled")
        self.open_signin_button.pack(side="left", padx=(6, 0))
        ttk.Label(connection, textvariable=self.device_message_var, wraplength=760).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

        controls = ttk.LabelFrame(root, text="Node controls", padding=10)
        controls.pack(fill="x", pady=(12, 0))
        self.enable_button = ttk.Button(controls, text="Enable inference", command=lambda: self._set_inference(True), state="disabled")
        self.enable_button.pack(side="left")
        self.disable_button = ttk.Button(controls, text="Disable inference", command=lambda: self._set_inference(False), state="disabled")
        self.disable_button.pack(side="left", padx=8)
        self.disconnect_button = ttk.Button(controls, text="Disconnect safely", command=self._disconnect, state="disabled")
        self.disconnect_button.pack(side="left")
        ttk.Label(controls, textvariable=self.inference_var).pack(side="left", padx=16)
        shared_compute = ttk.LabelFrame(root, text="PROVIS shared compute — OpenAI-compatible endpoint", padding=10)
        shared_compute.pack(fill="x", pady=(10, 0))
        ttk.Label(shared_compute, text="Use this base URL for LLM workflows and applications running on this PC:").pack(anchor="w")
        endpoint_row = ttk.Frame(shared_compute)
        endpoint_row.pack(fill="x", pady=(5, 0))
        ttk.Label(endpoint_row, text="This PC:").pack(side="left")
        ttk.Label(endpoint_row, textvariable=self.local_endpoint_var, foreground="#165d69").pack(side="left", padx=(6, 6))
        self.local_endpoint_copy_button = ttk.Button(endpoint_row, text="Copy URL", command=lambda: self._copy_inference_endpoint(self.local_endpoint_var.get()), state="disabled")
        self.local_endpoint_copy_button.pack(side="left")
        ttk.Label(shared_compute, text=(
            "Requests are forwarded by the enrolled node to PROVIS shared compute. This endpoint is loopback-only and works for apps on this PC. "
            "Docker container access is not enabled. OpenClaw can use the shared gateway when configured; gateway credentials stay inside the node."
        ), wraplength=860, justify="left", foreground="#5c6770").pack(anchor="w", pady=(7, 0))
        ttk.Label(root, textvariable=self.device_message_var, foreground="#5c6770").pack(anchor="w", pady=(4, 0))

        monitor = ttk.LabelFrame(root, text="PROVIS node activity (prompts and credentials are never displayed)", padding=8)
        monitor.pack(fill="both", expand=True, pady=(12, 0))
        columns = ("time", "operation", "model", "status", "duration", "error")
        self.activity_view = ttk.Treeview(monitor, columns=columns, show="headings", height=14)
        headings = {"time": "Time", "operation": "Operation", "model": "Model", "status": "Status", "duration": "Duration ms", "error": "Error"}
        widths = {"time": 90, "operation": 100, "model": 180, "status": 90, "duration": 90, "error": 180}
        for column in columns:
            self.activity_view.heading(column, text=headings[column])
            self.activity_view.column(column, width=widths[column], anchor="w")
        scrollbar = ttk.Scrollbar(monitor, orient="vertical", command=self.activity_view.yview)
        self.activity_view.configure(yscrollcommand=scrollbar.set)
        self.activity_view.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _load_saved_values(self) -> None:
        config = _load_node_config(self.config_path)
        tailscale_ipv4 = _detect_tailscale_ipv4()
        port = str(config.get("port") or os.getenv("INFERENCE_NODE_PORT", "8765"))
        values = {
            "gateway_url": _load_managed_gateway_url() or str(config.get("gateway_url") or ""),
            "entra_tenant_id": config.get("entra_tenant_id") or os.getenv("INFERENCE_ENTRA_TENANT_ID", ""),
            "entra_client_id": config.get("entra_client_id") or os.getenv("INFERENCE_ENTRA_CLIENT_ID", ""),
            "entra_scope": config.get("entra_scope") or os.getenv("INFERENCE_ENTRA_SCOPE", "api://provis-inference-gateway/.default"),
            "pairing_token": "",
            "node_id": config.get("node_id") or _default_node_id(),
            "agent_url": config.get("agent_url") or (f"http://{tailscale_ipv4}:{port}" if tailscale_ipv4 else ""),
            "ollama_url": config.get("endpoint") or "http://127.0.0.1:11434",
            "tailnet_name": config.get("tailnet_name") or os.getenv("TAILNET_NAME", "provisusa.com"),
            "allowed_applications": ",".join(config.get("allowed_applications") or ("manuals_rag", "provis_employee_agent", "openclaw")),
            "port": port,
        }
        for key, value in values.items():
            self.fields[key].set(str(value))
        if config.get("shared_secret"):
            self.status_var.set("Saved enrollment found; reconnecting automatically")
            self.after(600, self._resume_saved_enrollment)

    def _values(self) -> dict[str, str]:
        return {key: variable.get().strip() for key, variable in self.fields.items()}

    def _resume_saved_enrollment(self) -> None:
        if self.agent is None:
            self.status_var.set("Reconnecting saved enrollment...")
            threading.Thread(target=self._connect_worker, args=(self._values(),), daemon=True).start()

    def _entra_connect(self) -> None:
        self.entra_button.configure(state="disabled")
        self._clear_device_prompt()
        self.status_var.set("Starting Microsoft 365 device sign-in...")
        threading.Thread(target=self._entra_connect_worker, args=(self._values(),), daemon=True).start()

    def _clear_device_prompt(self) -> None:
        self._verification_url = ""
        self.device_code_var.set("")
        self.device_message_var.set("")
        self.copy_code_button.configure(state="disabled")
        self.open_signin_button.configure(state="disabled")

    def _show_device_prompt(self, payload: object) -> None:
        details = payload if isinstance(payload, dict) else {}
        self._verification_url = str(details.get("verification_url") or "")
        self.device_code_var.set(str(details.get("user_code") or ""))
        self.device_message_var.set(str(details.get("message") or "Complete Microsoft 365 sign-in in the browser."))
        self.copy_code_button.configure(state="normal" if self.device_code_var.get() else "disabled")
        self.open_signin_button.configure(state="normal" if self._verification_url else "disabled")

    def _copy_device_code(self) -> None:
        code = self.device_code_var.get().strip()
        if not code:
            return
        self.clipboard_clear()
        self.clipboard_append(code)
        self.update()
        self.status_var.set("Microsoft code copied to clipboard")

    def _copy_inference_endpoint(self, endpoint: str) -> None:
        copied = copy_endpoint_to_clipboard(
            endpoint,
            local_broker_endpoints(getattr(getattr(self.server, "local_broker_server", None), "server_port", 0)),
            clipboard_clear=self.clipboard_clear,
            clipboard_append=self.clipboard_append,
            update=self.update,
        )
        if not copied:
            self.device_message_var.set("Endpoint could not be copied.")
            return
        self.device_message_var.set("Endpoint URL copied. Local API authentication may be required by your administrator.")

    def _open_verification_url(self) -> None:
        if self._verification_url:
            webbrowser.open(self._verification_url)

    def _resolve_values(self, values: dict[str, str]) -> dict[str, str]:
        config = _load_node_config(self.config_path)
        gateway_url = _load_managed_gateway_url() or str(config.get("gateway_url") or "").strip()
        if not _is_managed_gateway_url(gateway_url):
            raise RuntimeError("this PROVIS build has no gateway address; ask the administrator for the managed installer")
        discovered = fetch_enrollment_config(gateway_url=gateway_url)
        tailscale_ipv4 = _detect_tailscale_ipv4()
        port = str(config.get("port") or discovered.get("port") or 8765)
        values.update({
            "gateway_url": gateway_url,
            "entra_tenant_id": str(discovered["tenant_id"]),
            "entra_client_id": str(discovered["client_id"]),
            "entra_scope": str(discovered["scope"]),
            "node_id": str(config.get("node_id") or values.get("node_id") or _default_node_id()),
            "agent_url": str(config.get("agent_url") or values.get("agent_url") or (f"http://{tailscale_ipv4}:{port}" if tailscale_ipv4 else "")),
            "ollama_url": str(config.get("endpoint") or values.get("ollama_url") or "http://127.0.0.1:11434"),
            "tailnet_name": str(config.get("tailnet_name") or discovered.get("tailnet_name") or ""),
            "allowed_applications": ",".join(config.get("allowed_applications") or discovered.get("allowed_applications") or ("provis_employee_agent",)),
            "port": port,
        })
        if not values["agent_url"]:
            raise RuntimeError("Tailscale is not available; the node cannot advertise its inference endpoint")
        return values

    def _persist_and_start(self, config: dict, values: dict[str, str]) -> tuple[object, object, str]:
        config.update({
            "gateway_url": values["gateway_url"] or config.get("gateway_url", ""),
            "node_id": values["node_id"] or config.get("node_id", ""),
            "endpoint": values["ollama_url"] or config.get("endpoint", ""),
            "agent_url": values["agent_url"] or config.get("agent_url", ""),
            "tailnet_name": values["tailnet_name"] or config.get("tailnet_name", ""),
            "allowed_applications": tuple(item.strip() for item in values["allowed_applications"].split(",") if item.strip()),
            "port": int(values["port"] or 8765),
            "bind": (_detect_tailscale_ipv4() or "127.0.0.1"),
            "entra_tenant_id": values.get("entra_tenant_id", ""),
            "entra_client_id": values.get("entra_client_id", ""),
            "entra_scope": values.get("entra_scope", ""),
        })
        _save_node_config(self.config_path, config)
        server, agent, gateway_url = create_node_server(self.config_path)
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="provis-inference-node-server")
        thread.start()
        server.inference_server_thread = thread
        local_broker_server = getattr(server, "local_broker_server", None)
        if local_broker_server is not None:
            broker_thread = threading.Thread(target=local_broker_server.serve_forever, daemon=True, name="provis-inference-local-broker")
            try:
                broker_thread.start()
            except Exception:
                server.inference_stop_event.set()
                server.shutdown()
                server.server_close()
                local_broker_server.server_close()
                thread.join(timeout=5)
                raise
            server.local_broker_thread = broker_thread
        return server, agent, gateway_url

    def _connect_worker(self, values: dict[str, str]) -> None:
        phase = "saved_registration_load"
        try:
            config = _load_node_config(self.config_path)
            if not config.get("shared_secret"):
                phase = "legacy_enrollment"
                if not values["gateway_url"] or not values["pairing_token"]:
                    raise RuntimeError("use Sign in with Microsoft 365 for first connection, or provide a legacy pairing token")
                result = enroll_node(
                    gateway_url=values["gateway_url"],
                    pairing_token=values["pairing_token"],
                    node_id=values["node_id"],
                    endpoint=values["ollama_url"],
                    agent_url=values["agent_url"],
                    tailnet_name=values["tailnet_name"],
                )
                config.update(result)
            phase = "local_service_start"
            self.worker_queue.put(("connected", self._persist_and_start(config, values)))
            append_connect_diagnostic(self.connect_diagnostics_path, "connect_complete")
        except Exception as error:
            append_connect_diagnostic(self.connect_diagnostics_path, phase, error=error)
            self.worker_queue.put(("error", str(error)))

    def _entra_connect_worker(self, values: dict[str, str]) -> None:
        phase = "gateway_discovery"
        try:
            values = self._resolve_values(values)
            phase = "device_code_request"
            device = request_device_code(
                tenant_id=values["entra_tenant_id"],
                client_id=values["entra_client_id"],
                scope=values["entra_scope"],
            )
            message = str(device.get("message") or "Open the Microsoft sign-in URL and enter the displayed code.")
            verification_url = str(device.get("verification_uri_complete") or device.get("verification_uri") or "")
            self.worker_queue.put(("entra_prompt", {
                "message": message,
                "user_code": str(device.get("user_code") or ""),
                "verification_url": verification_url,
            }))
            if verification_url:
                webbrowser.open(verification_url)
            phase = "device_code_poll"
            token = poll_device_token(
                tenant_id=values["entra_tenant_id"],
                client_id=values["entra_client_id"],
                device_code=str(device["device_code"]),
                interval=int(device.get("interval") or 5),
                expires_in=int(device.get("expires_in") or 900),
            )
            phase = "entra_enrollment"
            config = _load_node_config(self.config_path)
            result = enroll_node_with_entra(
                gateway_url=values["gateway_url"],
                access_token=str(token["access_token"]),
                node_id=values["node_id"],
                endpoint=values["ollama_url"],
                agent_url=values["agent_url"],
                tailnet_name=values["tailnet_name"],
            )
            config.update(result)
            config.update({"entra_tenant_id": values["entra_tenant_id"], "entra_client_id": values["entra_client_id"], "entra_scope": values["entra_scope"]})
            phase = "local_service_start"
            self.worker_queue.put(("connected", self._persist_and_start(config, values)))
            append_connect_diagnostic(self.connect_diagnostics_path, "connect_complete")
        except Exception as error:
            append_connect_diagnostic(self.connect_diagnostics_path, phase, error=error)
            self.worker_queue.put(("error", str(error)))

    def _set_inference(self, enabled: bool) -> None:
        if self.agent is None:
            return
        self.agent.set_inference_enabled(enabled)
        self._update_controls()

    def _disconnect(self) -> None:
        if self.agent is None or not self.gateway_url:
            self.status_var.set("No connected gateway")
            return
        self.disconnect_button.configure(state="disabled")
        self.status_var.set("Disabling inference and disconnecting from gateway...")
        threading.Thread(target=self._disconnect_worker, daemon=True).start()

    def _disconnect_worker(self) -> None:
        try:
            self.agent.set_inference_enabled(False)
            result = self.agent.disconnect(self.gateway_url)
            if not result.get("disconnected"):
                raise RuntimeError("gateway did not confirm disconnect")
            self.worker_queue.put(("disconnected", None))
        except Exception as error:
            self.worker_queue.put(("disconnect_error", str(error)))

    def _drain_worker_queue(self) -> None:
        try:
            while True:
                event, value = self.worker_queue.get_nowait()
                if event == "connected":
                    self.server, self.agent, self.gateway_url = value
                    broker_server = getattr(self.server, "local_broker_server", None)
                    endpoints = local_broker_endpoints(getattr(broker_server, "server_port", 0))
                    self.local_endpoint_var.set(endpoints[0] if endpoints else "Local inference endpoint disabled")
                    self.local_endpoint_copy_button.configure(state="normal" if endpoints else "disabled")
                    self._clear_device_prompt()
                    self.status_var.set(f"Connected: {self.agent.node_id}")
                    self._update_controls()
                elif event == "disconnected":
                    self._stop_server()
                    self.status_var.set("Disconnected safely from gateway")
                    self._update_controls()
                elif event == "disconnect_error":
                    self.status_var.set(f"Disconnect failed; node remains stopped from inference: {value}")
                    if self.agent:
                        self._update_controls()
                elif event == "error":
                    self.status_var.set(f"Connection failed: {value}")
                    self.entra_button.configure(state="normal")
                elif event == "entra_prompt":
                    self._show_device_prompt(value)
                    self.status_var.set("Microsoft sign-in is waiting for approval")
        except queue.Empty:
            pass
        self.after(250, self._drain_worker_queue)

    def _update_controls(self) -> None:
        connected = self.agent is not None
        self.entra_button.configure(state="disabled" if connected else "normal")
        state = "normal" if connected else "disabled"
        self.enable_button.configure(state=state)
        self.disable_button.configure(state=state)
        self.disconnect_button.configure(state=state)
        if connected:
            self.inference_var.set("Inference enabled" if self.agent.inference_enabled else "Inference disabled")
        else:
            self.inference_var.set("Inference disabled")

    def _refresh_activity(self) -> None:
        if self.agent is not None:
            events = self.agent.activity()
            heartbeat = next((event for event in reversed(events) if event.get("operation") == "heartbeat"), None)
            if heartbeat and heartbeat.get("status") == "failed":
                self.status_var.set(f"Gateway heartbeat failed: {heartbeat.get('error', 'unknown')}; retrying automatically")
            elif heartbeat and str(self.status_var.get()).startswith(("Gateway heartbeat failed:", "Connected:")):
                self.status_var.set(f"Connected: {self.agent.node_id}")
            for item in self.activity_view.get_children():
                self.activity_view.delete(item)
            for event in events[-200:]:
                timestamp = time.strftime("%H:%M:%S", time.localtime(float(event.get("timestamp") or 0)))
                self.activity_view.insert("", "end", values=(
                    timestamp,
                    event.get("operation", ""),
                    event.get("model", ""),
                    event.get("status", ""),
                    event.get("duration_ms", ""),
                    event.get("error", ""),
                ))
            if events:
                children = self.activity_view.get_children()
                if children:
                    self.activity_view.see(children[-1])
        self.after(1000, self._refresh_activity)

    def _stop_server(self) -> None:
        server = self.server
        self.server = None
        self.agent = None
        self.gateway_url = ""
        if server is not None:
            server.inference_stop_event.set()
            local_broker_server = getattr(server, "local_broker_server", None)
            broker_thread = getattr(server, "local_broker_thread", None)
            if local_broker_server is not None:
                if broker_thread is not None and broker_thread.is_alive():
                    local_broker_server.shutdown()
                local_broker_server.server_close()
                if broker_thread is not None:
                    broker_thread.join(timeout=5)
            server.shutdown()
            server.server_close()
            main_thread = getattr(server, "inference_server_thread", None)
            if main_thread is not None:
                main_thread.join(timeout=5)
        self.local_endpoint_var.set("Available after node connects")
        if self.local_endpoint_copy_button is not None:
            self.local_endpoint_copy_button.configure(state="disabled")

    def _close_requested(self) -> None:
        if self.agent is None:
            self.destroy()
            return
        if messagebox.askyesno("Disconnect node", "Safely disconnect this node from the gateway before closing?"):
            self._disconnect()
            self.after(250, self._close_after_disconnect)

    def _close_after_disconnect(self) -> None:
        if self.agent is None:
            self.destroy()
        else:
            self.after(250, self._close_after_disconnect)


def main() -> None:
    InferenceNodeClient().mainloop()


if __name__ == "__main__":
    main()
