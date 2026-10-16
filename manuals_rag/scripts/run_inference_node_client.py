#!/usr/bin/env python3
"""Small Tk desktop client for connecting and operating an inference node."""

from __future__ import annotations

import os
import platform
import queue
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

from manuals_rag_inference.node_agent import (
    _detect_tailscale_ipv4,
    _load_node_config,
    _save_node_config,
    create_node_server,
    enroll_node,
)
from manuals_rag_inference.entra_client import (
    enroll_node_with_entra,
    poll_device_token,
    request_device_code,
)


def _default_config_path() -> str:
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return str(Path(os.environ["LOCALAPPDATA"]) / "ProVIS" / "inference-node.json")
    return str(Path.home() / ".config/ProVIS/inference-node.json")


class InferenceNodeClient(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PROVIS Inference Node")
        self.geometry("920x650")
        self.minsize(760, 520)
        self.protocol("WM_DELETE_WINDOW", self._close_requested)
        self.config_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", _default_config_path())
        self.server = None
        self.agent = None
        self.gateway_url = ""
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.fields: dict[str, tk.StringVar] = {}
        self.status_var = tk.StringVar(value="Not connected")
        self.inference_var = tk.StringVar(value="Inference disabled")
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
        rows = (
            ("gateway_url", "Gateway URL"),
            ("entra_tenant_id", "Microsoft tenant ID"),
            ("entra_client_id", "Microsoft app client ID"),
            ("entra_scope", "Gateway API scope"),
            ("pairing_token", "Legacy pairing token (optional)"),
            ("node_id", "Node name"),
            ("agent_url", "Advertised node URL"),
            ("ollama_url", "Local Ollama URL"),
            ("tailnet_name", "Tailnet name"),
            ("allowed_applications", "Allowed applications"),
            ("port", "Node port"),
        )
        for index, (key, label) in enumerate(rows):
            self.fields[key] = tk.StringVar()
            ttk.Label(connection, text=label).grid(row=index // 2, column=(index % 2) * 2, sticky="w", padx=(0, 6), pady=4)
            entry = ttk.Entry(connection, textvariable=self.fields[key], show="*" if key == "pairing_token" else "")
            entry.grid(row=index // 2, column=(index % 2) * 2 + 1, sticky="ew", padx=(0, 14), pady=4)
        connection.columnconfigure(1, weight=1)
        connection.columnconfigure(3, weight=1)
        action_row = (len(rows) + 1) // 2
        self.entra_button = ttk.Button(connection, text="Sign in with Microsoft 365", command=self._entra_connect)
        self.entra_button.grid(row=action_row, column=0, sticky="w", pady=(8, 0))
        self.connect_button = ttk.Button(connection, text="Connect saved enrollment", command=self._connect)
        self.connect_button.grid(row=action_row, column=2, sticky="w", pady=(8, 0))
        ttk.Label(connection, textvariable=self.status_var).grid(row=action_row + 1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        controls = ttk.LabelFrame(root, text="Node controls", padding=10)
        controls.pack(fill="x", pady=(12, 0))
        self.enable_button = ttk.Button(controls, text="Enable inference", command=lambda: self._set_inference(True), state="disabled")
        self.enable_button.pack(side="left")
        self.disable_button = ttk.Button(controls, text="Disable inference", command=lambda: self._set_inference(False), state="disabled")
        self.disable_button.pack(side="left", padx=8)
        self.disconnect_button = ttk.Button(controls, text="Disconnect safely", command=self._disconnect, state="disabled")
        self.disconnect_button.pack(side="left")
        ttk.Label(controls, textvariable=self.inference_var).pack(side="left", padx=16)

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
            "gateway_url": config.get("gateway_url") or os.getenv("INFERENCE_GATEWAY_URL", ""),
            "entra_tenant_id": config.get("entra_tenant_id") or os.getenv("INFERENCE_ENTRA_TENANT_ID", ""),
            "entra_client_id": config.get("entra_client_id") or os.getenv("INFERENCE_ENTRA_CLIENT_ID", ""),
            "entra_scope": config.get("entra_scope") or os.getenv("INFERENCE_ENTRA_SCOPE", "api://provis-inference-gateway/.default"),
            "pairing_token": "",
            "node_id": config.get("node_id") or platform.node().replace(" ", "-") or "inference-node",
            "agent_url": config.get("agent_url") or (f"http://{tailscale_ipv4}:{port}" if tailscale_ipv4 else ""),
            "ollama_url": config.get("endpoint") or "http://127.0.0.1:11434",
            "tailnet_name": config.get("tailnet_name") or os.getenv("TAILNET_NAME", "provisusa.com"),
            "allowed_applications": ",".join(config.get("allowed_applications") or ("manuals_rag", "provis_employee_agent", "openclaw")),
            "port": port,
        }
        for key, value in values.items():
            self.fields[key].set(str(value))
        if config.get("shared_secret"):
            self.status_var.set("Saved enrollment found; press Connect to resume")

    def _values(self) -> dict[str, str]:
        return {key: variable.get().strip() for key, variable in self.fields.items()}

    def _connect(self) -> None:
        self.connect_button.configure(state="disabled")
        self.status_var.set("Connecting and checking Ollama...")
        threading.Thread(target=self._connect_worker, args=(self._values(),), daemon=True).start()

    def _entra_connect(self) -> None:
        self.entra_button.configure(state="disabled")
        self.connect_button.configure(state="disabled")
        self.status_var.set("Starting Microsoft 365 device sign-in...")
        threading.Thread(target=self._entra_connect_worker, args=(self._values(),), daemon=True).start()

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
        return server, agent, gateway_url

    def _connect_worker(self, values: dict[str, str]) -> None:
        try:
            config = _load_node_config(self.config_path)
            if not config.get("shared_secret"):
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
            self.worker_queue.put(("connected", self._persist_and_start(config, values)))
        except Exception as error:
            self.worker_queue.put(("error", str(error)))

    def _entra_connect_worker(self, values: dict[str, str]) -> None:
        try:
            required = (values["gateway_url"], values["entra_tenant_id"], values["entra_client_id"], values["entra_scope"])
            if not all(required):
                raise RuntimeError("gateway URL, tenant ID, client ID, and gateway API scope are required")
            device = request_device_code(
                tenant_id=values["entra_tenant_id"],
                client_id=values["entra_client_id"],
                scope=values["entra_scope"],
            )
            message = str(device.get("message") or "Open the Microsoft sign-in URL and enter the displayed code.")
            self.worker_queue.put(("entra_prompt", message))
            verification_url = str(device.get("verification_uri_complete") or device.get("verification_uri") or "")
            if verification_url:
                webbrowser.open(verification_url)
            token = poll_device_token(
                tenant_id=values["entra_tenant_id"],
                client_id=values["entra_client_id"],
                device_code=str(device["device_code"]),
                interval=int(device.get("interval") or 5),
                expires_in=int(device.get("expires_in") or 900),
            )
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
            self.worker_queue.put(("connected", self._persist_and_start(config, values)))
        except Exception as error:
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
                    self.connect_button.configure(state="normal")
                    self.entra_button.configure(state="normal")
                elif event == "entra_prompt":
                    self.status_var.set(f"Microsoft sign-in: {value}")
        except queue.Empty:
            pass
        self.after(250, self._drain_worker_queue)

    def _update_controls(self) -> None:
        connected = self.agent is not None
        self.connect_button.configure(state="disabled" if connected else "normal")
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
            server.shutdown()
            server.server_close()

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
