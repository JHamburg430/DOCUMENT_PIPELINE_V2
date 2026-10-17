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

from manuals_rag_inference.node_agent import (
    _detect_tailscale_ipv4,
    _gpu_snapshots,
    _load_node_config,
    _save_node_config,
    create_node_server,
    enroll_node,
)
from manuals_rag_inference.entra_client import (
    enroll_node_with_entra,
    fetch_enrollment_config,
    poll_device_token,
    request_device_code,
)


def _default_config_path() -> str:
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return str(Path(os.environ["LOCALAPPDATA"]) / "ProVIS" / "inference-node.json")
    return str(Path.home() / ".config/ProVIS/inference-node.json")


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


from manuals_rag_inference.node_presentation import public_events, utilization
from manuals_rag_inference.node_tray import NodeTray, SingleInstance


class InferenceNodeClient(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PROVIS Inference Node")
        self.geometry("440x740")
        self.resizable(False, False)
        self.configure(bg="#f3f6fb")
        self.protocol("WM_DELETE_WINDOW", self._close_requested)
        self.config_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", _default_config_path())
        self.server = None
        self.agent = None
        self.gateway_url = ""
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
        self.signin_code = ""
        self.inference_var = tk.StringVar(value="Inference disabled")
        self._quitting = False
        self._last_rows = None
        self._metrics_stop = threading.Event()
        self._build_ui()
        self.tray = NodeTray(lambda event, value: self.worker_queue.put((event, value)))
        self._load_saved_values()
        threading.Thread(target=self._metrics_worker, daemon=True, name="provis-utilization").start()
        self.after(800, self._position_panel)
        self.after(250, self._drain_worker_queue)
        self.after(1000, self._refresh_activity)

    def _build_ui(self) -> None:
        navy, muted, blue = "#122743", "#65758c", "#225fe5"
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 8), background="#edf2fa", foreground=navy, borderwidth=0)
        style.map("TButton", background=[("active", "#dbe7fa")], foreground=[("disabled", "#96a1b2")])
        style.configure("Primary.TButton", background=blue, foreground="white")
        style.map("Primary.TButton", background=[("active", "#174fc9"), ("disabled", "#b3c8ef")])
        style.configure("Node.Treeview", font=("Segoe UI", 9), rowheight=34, background="white", fieldbackground="white", foreground=navy, borderwidth=0)
        style.configure("Node.Treeview.Heading", font=("Segoe UI", 9), background="#f3f6fb", foreground=muted, relief="flat")
        style.configure("Node.Horizontal.TProgressbar", background="#1c9b8a", troughcolor="#e8edf5", borderwidth=0)
        header = tk.Frame(self, bg=navy, padx=22, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="PROVIS", bg=navy, fg="white", font=("Segoe UI", 21, "bold")).pack(anchor="w")
        tk.Label(header, text="PRIVATE COMPUTE", bg=navy, fg="#96b7eb", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(1, 0))
        root = tk.Frame(self, bg="#f3f6fb", padx=18, pady=14)
        root.pack(fill="both", expand=True)
        connection = tk.Frame(root, bg="white", padx=14, pady=12)
        connection.pack(fill="x")
        tk.Label(connection, text="Your device", bg="white", fg=navy, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(connection, textvariable=self.status_var, wraplength=315, justify="left", bg="white", fg=muted, font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=(4, 8))
        self.copy_code_button = ttk.Button(connection, text="Copy", command=self._copy_signin_code)
        self.copy_code_button.grid(row=1, column=1, padx=(6, 0))
        self.copy_code_button.grid_remove()
        self.entra_button = ttk.Button(connection, text="Sign in with Microsoft 365", style="Primary.TButton", command=self._entra_connect)
        self.entra_button.grid(row=2, column=0, columnspan=2, sticky="ew")
        metrics = tk.Frame(root, bg="white", padx=14, pady=12)
        metrics.pack(fill="x", pady=(10, 0))
        tk.Label(metrics, text="Device utilization", bg="white", fg=navy, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.gpu_var = tk.StringVar(value="GPU  •  Checking…")
        self.memory_var = tk.StringVar(value="GPU memory  •  Checking…")
        self.work_var = tk.StringVar(value="No active work")
        tk.Label(metrics, textvariable=self.gpu_var, bg="white", fg=muted, font=("Segoe UI", 10)).pack(anchor="w", pady=(8, 3))
        self.gpu_bar = ttk.Progressbar(metrics, style="Node.Horizontal.TProgressbar", maximum=100)
        self.gpu_bar.pack(fill="x")
        tk.Label(metrics, textvariable=self.memory_var, bg="white", fg=muted, font=("Segoe UI", 10)).pack(anchor="w", pady=(8, 3))
        self.memory_bar = ttk.Progressbar(metrics, style="Node.Horizontal.TProgressbar", maximum=100)
        self.memory_bar.pack(fill="x")
        tk.Label(metrics, textvariable=self.work_var, bg="white", fg=muted, font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
        controls = tk.Frame(root, bg="#f3f6fb")
        controls.pack(fill="x", pady=10)
        self.enable_button = ttk.Button(controls, text="Resume", command=lambda: self._set_inference(True), state="disabled")
        self.enable_button.pack(side="left")
        self.disable_button = ttk.Button(controls, text="Pause", command=lambda: self._set_inference(False), state="disabled")
        self.disable_button.pack(side="left", padx=6)
        self.disconnect_button = ttk.Button(controls, text="Disconnect", command=self._disconnect, state="disabled")
        self.disconnect_button.pack(side="right")
        status = tk.Frame(root, bg="#f3f6fb")
        status.pack(fill="x", pady=(0, 8))
        tk.Label(status, textvariable=self.inference_var, bg="#f3f6fb", fg=muted, font=("Segoe UI", 9)).pack(side="left")
        tk.Button(status, text="Quit", command=self._quit_requested, bg="#f3f6fb", fg=muted, relief="flat", font=("Segoe UI", 9)).pack(side="right")
        tk.Label(root, text="Recent events", bg="#f3f6fb", fg=navy, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.activity_view = ttk.Treeview(root, style="Node.Treeview", columns=("time", "event", "status"), show="headings", height=4)
        for column, label, width in (("time", "Time", 66), ("event", "Event", 197), ("status", "Status", 101)):
            self.activity_view.heading(column, text=label)
            self.activity_view.column(column, width=width, stretch=False)
        self.activity_view.pack(fill="both", expand=True)
        tk.Label(root, text="Private by design. Activity and utilization only.", bg="#f3f6fb", fg=muted, font=("Segoe UI", 8)).pack(anchor="w", pady=(10, 0))

    def _position_panel(self):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            area = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(area), 0):
                height = min(740, area.bottom - area.top - 48)
                self.geometry(f"440x{height}+{max(area.left, area.right - 456)}+{max(area.top, area.bottom - height - 40)}")

    def _show_panel(self):
        if self.state() == "withdrawn":
            self._position_panel()
            self.deiconify()
            self.lift()
            self.focus_force()
        else:
            self.withdraw()

    def _metrics_worker(self):
        while not self._metrics_stop.is_set():
            try:
                self.worker_queue.put(("metrics", utilization(_gpu_snapshots())))
            except Exception:
                self.worker_queue.put(("metrics", (0, 0, 0)))
            self._metrics_stop.wait(3)

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
        self._clear_signin_code()
        self.entra_button.configure(state="disabled")
        self.status_var.set("Starting Microsoft 365 device sign-in...")
        threading.Thread(target=self._entra_connect_worker, args=(self._values(),), daemon=True).start()

    def _show_signin_code(self, code: str) -> None:
        self.signin_code = code
        self.status_var.set(code)
        self.copy_code_button.configure(text="Copy", state="normal")
        self.copy_code_button.grid()

    def _clear_signin_code(self) -> None:
        self.signin_code = ""
        self.copy_code_button.configure(text="Copy", state="disabled")
        self.copy_code_button.grid_remove()

    def _copy_signin_code(self) -> None:
        code = self.signin_code
        if not code:
            return
        self.clipboard_clear()
        self.clipboard_append(code)
        self.copy_code_button.configure(text="Copied")
        self.after(1500, lambda: self.copy_code_button.configure(text="Copy") if self.signin_code == code else None)

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
            values = self._resolve_values(values)
            device = request_device_code(
                tenant_id=values["entra_tenant_id"],
                client_id=values["entra_client_id"],
                scope=values["entra_scope"],
            )
            user_code = str(device.get("user_code") or "").strip()
            if not user_code:
                raise RuntimeError("Microsoft sign-in did not return a user code")
            self.worker_queue.put(("entra_prompt", user_code))
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
                if event == "tray_show":
                    self._show_panel()
                elif event == "show_existing":
                    self._position_panel()
                    self.deiconify()
                    self.lift()
                    self.focus_force()
                elif event == "tray_pause":
                    self._set_inference(False)
                elif event == "tray_resume":
                    self._set_inference(True)
                elif event == "tray_disconnect":
                    self._disconnect()
                elif event == "tray_quit":
                    self._quit_requested()
                elif event == "tray_error":
                    self.deiconify()
                    self.status_var.set("Tray unavailable. Keep this panel open while sharing.")
                elif event == "metrics":
                    load, used, total = value
                    self.gpu_var.set(f"GPU load  •  {load:.0f}%" if total else "GPU metrics unavailable")
                    self.memory_var.set(f"GPU memory  •  {used / 1024:.1f} / {total / 1024:.1f} GB" if total else "GPU memory unavailable")
                    self.gpu_bar["value"] = load
                    self.memory_bar["value"] = 100 * used / total if total else 0
                elif event == "connected":
                    self._clear_signin_code()
                    self.server, self.agent, self.gateway_url = value
                    self.status_var.set("Connected to PROVIS")
                    self._update_controls()
                elif event == "disconnected":
                    self._clear_signin_code()
                    self._stop_server()
                    self.status_var.set("Disconnected safely from gateway")
                    self._update_controls()
                elif event == "disconnect_error":
                    self.status_var.set("Sharing paused. Disconnect could not be confirmed; try again.")
                    if self.agent:
                        self._update_controls()
                elif event == "error":
                    self._clear_signin_code()
                    self.status_var.set("Unable to connect. Check your connection and sign in again.")
                    self.entra_button.configure(state="normal")
                elif event == "entra_prompt":
                    self._show_signin_code(value)
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
            if heartbeat:
                self.status_var.set("Reconnecting to PROVIS…" if heartbeat.get("status") == "failed" else "Connected to PROVIS")
            rows = public_events(events)
            if rows != self._last_rows:
                self._last_rows = rows
                for item in self.activity_view.get_children():
                    self.activity_view.delete(item)
                for row in rows:
                    self.activity_view.insert("", "end", values=row)
                children = self.activity_view.get_children()
                if children:
                    self.activity_view.see(children[-1])
            # Only counts cross the presentation boundary; model keys stay private.
            with self.agent._request_lock:
                active = sum(self.agent._active_by_model.values())
            self.work_var.set(f"{active} active request" + ("s" if active != 1 else "") if active else "No active work")
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
        if self.tray.ready.is_set() and self.tray.error is None:
            self.withdraw()
        else:
            self._quit_requested()

    def _quit_requested(self) -> None:
        if self._quitting:
            return
        if self.agent is not None:
            if not messagebox.askyesno("Quit PROVIS", "Pause sharing and safely disconnect this device?", parent=self):
                return
            self._quitting = True
            self._disconnect()
            self.after(250, self._close_after_disconnect)
        else:
            self._finish_quit()

    def _finish_quit(self):
        self._metrics_stop.set()
        self.tray.close()
        self.destroy()

    def _close_after_disconnect(self) -> None:
        if self.agent is None:
            self._finish_quit()
        elif str(self.status_var.get()).startswith("Sharing paused."):
            self._quitting = False
            self.deiconify()
        else:
            self.after(250, self._close_after_disconnect)


def main() -> None:
    instance = SingleInstance()
    if instance.existing:
        instance.close()
        return
    try:
        app = InferenceNodeClient()
        instance.watch(lambda event, value: app.worker_queue.put((event, value)))
        app.mainloop()
    finally:
        instance.close()


if __name__ == "__main__":
    main()
