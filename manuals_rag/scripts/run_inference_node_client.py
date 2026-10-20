#!/usr/bin/env python3
"""Small Tk desktop client for connecting and operating an inference node."""

from __future__ import annotations

import os
import platform
import queue
import re
import json
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
    returning_node_registration,
    verify_node_registration_with_entra,
)
from manuals_rag_inference.connect_diagnostics import append_connect_diagnostic


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
    # Linux zipapps carry the same nonsecret bootstrap JSON as the Windows EXE.
    if not getattr(sys, 'frozen', False) and Path(sys.argv[0]).suffix == '.pyz':
        import json
        import zipfile
        try:
            with zipfile.ZipFile(sys.argv[0]) as archive:
                config = json.loads(archive.read('provis-node-config.json'))
            gateway_url = str(config.get('gateway_url') or '').strip()
            if _is_managed_gateway_url(gateway_url):
                return gateway_url.rstrip('/')
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            pass
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


def _managed_update_settings():
    import json
    from manuals_rag_inference.node_updater import bootstrap
    if not getattr(sys, 'frozen', False) and Path(sys.argv[0]).suffix == '.pyz':
        import zipfile
        with zipfile.ZipFile(sys.argv[0]) as archive:
            return bootstrap(json.loads(archive.read('provis-node-config.json')))
    for candidate in _managed_config_candidates():
        if candidate.is_file(): return bootstrap(_load_node_config(str(candidate)))
    return bootstrap({})


def _default_node_id() -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", platform.node()).strip("-")
    return value or "provis-inference-node"


def _application_asset_path(name: str) -> Path:
    """Resolve packaged or source-tree visual assets deterministically."""
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        frozen_root = getattr(sys, "_MEIPASS", "")
        if frozen_root:
            candidates.append(Path(str(frozen_root)) / "assets" / "brand" / name)
    script_dir = Path(__file__).resolve().parent
    candidates.extend((script_dir / "assets" / "brand" / name,
                       script_dir.parent / "assets" / "brand" / name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"PROVIS desktop asset is missing: {name}")


from manuals_rag_inference.node_presentation import public_events, utilization
from manuals_rag_inference.node_system import (
    OLLAMA_DOWNLOAD_PAGE,
    TAILSCALE_DOWNLOAD_PAGE,
    launch_managed_installer,
    managed_installer,
    official_installer_url,
    system_status,
    tailscale_disconnect,
    tailscale_reconnect,
    vendor_driver_page,
)
from manuals_rag_inference.node_tray import NodeTray, SingleInstance
from manuals_rag_inference.node_theme import PALETTES, load_theme, save_theme
from manuals_rag_inference.endpoint_ui import copy_endpoint_to_clipboard, local_broker_endpoints


class InferenceNodeClient(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PROVIS Inference Node")
        self._application_icon = None
        try:
            self._application_icon = tk.PhotoImage(file=str(_application_asset_path("provis-lens.png")))
            self.iconphoto(True, self._application_icon)
        except (OSError, tk.TclError):
            # Keep startup usable in source-only or deliberately minimal builds.
            self._application_icon = None
        self.geometry("440x740")
        self.resizable(False, False)
        self.configure(bg="#f3f6fb")
        self.protocol("WM_DELETE_WINDOW", self._close_requested)
        self.config_path = os.getenv("INFERENCE_NODE_CONFIG_PATH", _default_config_path())
        self.connect_diagnostics_path = str(Path(self.config_path).with_name("inference-node-diagnostics.jsonl"))
        self.theme_path = Path(self.config_path).with_name('inference-node-ui.json')
        self.theme = load_theme(self.theme_path)
        self.ui_font = 'Segoe UI' if os.name == 'nt' else 'DejaVu Sans'
        self.theme_button_var = tk.StringVar()
        self.updater = None
        self.update_settings = _managed_update_settings()
        self.server = None
        self.agent = None
        self.attached = False
        self.attached_url = ""
        self.attached_status: dict[str, object] = {}
        self.attach_existing = os.getenv("INFERENCE_NODE_ATTACH_EXISTING", "false").lower() == "true"
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
        self.local_endpoint_var = tk.StringVar(value="Available after node connects")
        self.local_endpoint_copy_button = None
        self.endpoint_copy_var = tk.StringVar(value="")
        self._quitting = False
        self._last_rows = None
        self._metrics_stop = threading.Event()
        self._build_ui()
        from manuals_rag_inference.owner_drawer import OwnerDrawer
        self.owner_drawer = OwnerDrawer(self)
        self.owner_panel = None
        self.settings_panel = None
        self.settings_vars: dict[str, tk.StringVar] = {}
        self.settings_buttons: dict[str, ttk.Button] = {}
        self.bind_all('<Control-Shift-A>', self._open_owner_panel)
        self._apply_theme()
        self.tray = NodeTray(lambda event, value: self.worker_queue.put((event, value)))
        self._load_saved_values()
        if self.attach_existing:
            self.after(600, self._attach_existing_node)
        threading.Thread(target=self._metrics_worker, daemon=True, name="provis-utilization").start()
        self.after(800, self._position_panel)
        self.after(250, self._drain_worker_queue)
        self.after(1000, self._refresh_activity)

    def _attached_request(self, path: str, method: str = "GET", payload: dict | None = None) -> dict:
        config = _load_node_config(self.config_path)
        advertised = urlparse(str(config.get('agent_url') or ''))
        port = int(advertised.port or config.get("port") or os.getenv("INFERENCE_NODE_PORT", "8765"))
        host = str(config.get("bind") or "").strip()
        if not host or host in {"0.0.0.0", "::", "::0"}:
            host = urlparse(str(config.get("agent_url") or "")).hostname or "127.0.0.1"
        self.attached_url = f"http://{'['+host+']' if ':' in host else host}:{port}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            token = str(config.get("local_control_token") or "").strip()
            if token:
                headers["X-Node-Local-Token"] = token
        request = Request(f"{self.attached_url}{path}", data=body, method=method, headers=headers)
        from urllib.request import build_opener
        from manuals_rag_inference.node_updater import NoRedirect
        with build_opener(NoRedirect()).open(request, timeout=2) as response:
            raw = response.read(262145)
        if len(raw) > 262144:
            raise ValueError('Local node response exceeded its limit')
        return json.loads(raw.decode())

    def _attach_existing_node(self) -> None:
        try:
            status = self._attached_request("/v1/node/status")
            if status.get('node_id') != _load_node_config(self.config_path).get('node_id'):
                raise ValueError('Local node identity mismatch')
            self.attached_status = status
            self.attached = True
            self.status_var.set("Connected to PROVIS")
            self._update_controls()
        except Exception:
            self.attached = False
            self.status_var.set("PROVIS service is not available")
            self._update_controls()
            self.after(3000, self._attach_existing_node)

    def _build_ui(self) -> None:
        navy, muted, blue = "#122743", "#65758c", "#225fe5"
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=(self.ui_font, 10), padding=(10, 8), background="#edf2fa", foreground=navy, borderwidth=0)
        style.map("TButton", background=[("active", "#dbe7fa")], foreground=[("disabled", "#96a1b2")])
        style.configure("Primary.TButton", background=blue, foreground="white")
        style.map("Primary.TButton", background=[("active", "#174fc9"), ("disabled", "#b3c8ef")])
        style.configure("Node.Treeview", font=(self.ui_font, 9), rowheight=34, background="white", fieldbackground="white", foreground=navy, borderwidth=0)
        style.configure("Node.Treeview.Heading", font=(self.ui_font, 9), background="#f3f6fb", foreground=muted, relief="flat")
        style.configure("Node.Horizontal.TProgressbar", background="#1c9b8a", troughcolor="#e8edf5", borderwidth=0)
        header = tk.Frame(self, bg=navy, padx=22, pady=18)
        header.pack(fill="x")
        tk.Button(header, text="Settings", command=self._open_settings, bg=navy, fg='white', activebackground='#29394d', activeforeground='white', relief='flat', font=(self.ui_font, 9), padx=8, pady=6).pack(side='right', padx=(4, 0))
        tk.Button(header, textvariable=self.theme_button_var, command=self._toggle_theme, bg=navy, fg='white', activebackground='#29394d', activeforeground='white', relief='flat', font=(self.ui_font, 9), padx=8, pady=6).pack(side='right')
        tk.Label(header, text="PROVIS", bg=navy, fg="white", font=(self.ui_font, 21, "bold")).pack(anchor="w")
        tk.Label(header, text="PRIVATE COMPUTE", bg=navy, fg="#96b7eb", font=(self.ui_font, 9, "bold")).pack(anchor="w", pady=(1, 0))
        from manuals_rag_inference.scrollable_body import ScrollableBody
        self.body_scroll = ScrollableBody(self)
        self.body_scroll.pack(fill="both", expand=True)
        root = self.main_body = self.body_scroll.content
        connection = tk.Frame(root, bg="white", padx=14, pady=12)
        connection.pack(fill="x")
        tk.Label(connection, text="Your device", bg="white", fg=navy, font=(self.ui_font, 12, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(connection, textvariable=self.status_var, wraplength=315, justify="left", bg="white", fg=muted, font=(self.ui_font, 10)).grid(row=1, column=0, sticky="w", pady=(4, 8))
        self.copy_code_button = ttk.Button(connection, text="Copy", command=self._copy_signin_code)
        self.copy_code_button.grid(row=1, column=1, padx=(6, 0))
        self.copy_code_button.grid_remove()
        self.entra_button = ttk.Button(connection, text="Sign in with Microsoft 365", style="Primary.TButton", command=self._entra_connect)
        self.entra_button.grid(row=2, column=0, columnspan=2, sticky="ew")
        shared_compute = tk.LabelFrame(connection, text="PROVIS shared compute — OpenAI-compatible endpoint", bg="white", fg=navy, padx=8, pady=7, font=(self.ui_font, 9, "bold"))
        shared_compute.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        tk.Label(shared_compute, text="Use as the OpenAI base URL for LLM workflows and apps on this PC:", wraplength=340, justify="left", bg="white", fg=muted, font=(self.ui_font, 9)).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Label(shared_compute, text="This PC", bg="white", fg=navy, font=(self.ui_font, 9, "bold")).grid(row=1, column=0, sticky="w", pady=(5, 0))
        tk.Label(shared_compute, textvariable=self.local_endpoint_var, bg="white", fg="#165d69", font=(self.ui_font, 9)).grid(row=1, column=1, sticky="w", padx=(6, 6), pady=(5, 0))
        self.local_endpoint_copy_button = ttk.Button(shared_compute, text="Copy URL", command=lambda: self._copy_inference_endpoint(self.local_endpoint_var.get()), state="disabled")
        self.local_endpoint_copy_button.grid(row=1, column=2, pady=(5, 0))
        tk.Label(shared_compute, text=(
            "The enrolled node forwards requests to PROVIS shared compute. This endpoint is loopback-only and works for apps on this PC. "
            "Docker container access is not enabled. OpenClaw can use the shared gateway when configured; gateway credentials stay inside the node."
        ), wraplength=360, justify="left", bg="white", fg=muted, font=(self.ui_font, 8)).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        tk.Label(connection, textvariable=self.endpoint_copy_var, wraplength=315, justify="left", bg="white", fg=muted, font=(self.ui_font, 8)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(3, 0))
        shared_compute.grid_columnconfigure(1, weight=1)
        metrics = tk.Frame(root, bg="white", padx=14, pady=12)
        metrics.pack(fill="x", pady=(10, 0))
        tk.Label(metrics, text="Device utilization", bg="white", fg=navy, font=(self.ui_font, 12, "bold")).pack(anchor="w")
        self.gpu_var = tk.StringVar(value="GPU  •  Checking…")
        self.memory_var = tk.StringVar(value="GPU memory  •  Checking…")
        self.work_var = tk.StringVar(value="No active work")
        tk.Label(metrics, textvariable=self.gpu_var, bg="white", fg=muted, font=(self.ui_font, 10)).pack(anchor="w", pady=(8, 3))
        self.gpu_bar = ttk.Progressbar(metrics, style="Node.Horizontal.TProgressbar", maximum=100)
        self.gpu_bar.pack(fill="x")
        tk.Label(metrics, textvariable=self.memory_var, bg="white", fg=muted, font=(self.ui_font, 10)).pack(anchor="w", pady=(8, 3))
        self.memory_bar = ttk.Progressbar(metrics, style="Node.Horizontal.TProgressbar", maximum=100)
        self.memory_bar.pack(fill="x")
        tk.Label(metrics, textvariable=self.work_var, bg="white", fg=muted, font=(self.ui_font, 9)).pack(anchor="w", pady=(8, 0))
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
        tk.Label(status, textvariable=self.inference_var, bg="#f3f6fb", fg=muted, font=(self.ui_font, 9)).pack(side="left")
        tk.Button(status, text="Quit", command=self._quit_requested, bg="#f3f6fb", fg=muted, relief="flat", font=(self.ui_font, 9)).pack(side="right")
        tk.Label(root, text="Recent events", bg="#f3f6fb", fg=navy, font=(self.ui_font, 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.activity_view = ttk.Treeview(root, style="Node.Treeview", columns=("time", "event", "status"), show="headings", height=4)
        for column, label, width in (("time", "Time", 66), ("event", "Event", 197), ("status", "Status", 101)):
            self.activity_view.heading(column, text=label)
            self.activity_view.column(column, width=width, stretch=False)
        self.activity_view.pack(fill="x")
        tk.Label(root, text="Activity and utilization only.", bg="#f3f6fb", fg=muted, font=(self.ui_font, 8)).pack(anchor="w", pady=(8, 0))

    def _toggle_theme(self):
        self.theme = 'dark' if self.theme == 'light' else 'light'
        self._apply_theme()
        try:
            save_theme(self.theme_path, self.theme)
        except OSError:
            self.status_var.set('Appearance changed for this session; preference could not be saved.')

    def _open_owner_panel(self, _event=None):
        from manuals_rag_inference.agent_link_ui import OwnerPanel
        if self.owner_panel is None or not self.owner_panel.winfo_exists():
            self.owner_panel = OwnerPanel(self)
        if self.owner_panel.runtime:
            self.owner_drawer.toggle()
        else:
            self.owner_panel.deiconify();self.owner_panel.lift()

    def _open_settings(self):
        if self.settings_panel is not None and self.settings_panel.winfo_exists():
            self.settings_panel.deiconify(); self.settings_panel.lift()
            self._refresh_settings()
            return
        panel = self.settings_panel = tk.Toplevel(self)
        panel.title("PROVIS • Settings")
        panel.geometry("560x620")
        panel.resizable(False, True)
        panel.protocol("WM_DELETE_WINDOW", panel.withdraw)
        colors = PALETTES[self.theme]
        panel.configure(bg=colors["background"])
        from manuals_rag_inference.scrollable_body import ScrollableBody
        panel.theme = self.theme
        self.settings_scroll = ScrollableBody(panel)
        self.settings_scroll.pack(fill='both', expand=True)
        body = self.settings_scroll.content
        tk.Label(body, text="Settings", bg=colors["background"], fg=colors["text"], font=(self.ui_font, 18, "bold")).pack(anchor="w")
        tk.Label(body, text="Local prerequisites and private-network controls", bg=colors["background"], fg=colors["muted"], font=(self.ui_font, 10)).pack(anchor="w", pady=(3, 14))
        refresh = ttk.Button(body, text="Refresh status", command=self._refresh_settings)
        refresh.pack(anchor="e", pady=(0, 8))

        def row(key: str, title: str, description: str):
            card = tk.Frame(body, bg=colors["card"], padx=12, pady=10)
            card.pack(fill="x", pady=5)
            top = tk.Frame(card, bg=colors["card"])
            top.pack(fill="x")
            tk.Label(top, text=title, bg=colors["card"], fg=colors["text"], font=(self.ui_font, 11, "bold")).pack(side="left")
            status_var = tk.StringVar(value="Checking…")
            tk.Label(top, textvariable=status_var, bg=colors["card"], fg=colors["muted"], font=(self.ui_font, 10)).pack(side="right")
            detail_var = tk.StringVar(value=description)
            tk.Label(card, textvariable=detail_var, bg=colors["card"], fg=colors["muted"], justify="left", wraplength=500, font=(self.ui_font, 9)).pack(anchor="w", pady=(4, 7))
            controls = tk.Frame(card, bg=colors["card"])
            controls.pack(fill="x")
            return card, status_var, detail_var, controls

        _, status, detail, controls = row('onnx', 'ONNX / Intel acceleration', 'Included automatically in the signed app; detects available Intel accelerators')
        self.settings_vars.update({'onnx_status': status, 'onnx_detail': detail})
        self.onnx_device_var = tk.StringVar(value=str(_load_node_config(self.config_path).get('onnx_device') or 'AUTO'))
        ttk.Label(controls, text='Device').pack(side='left', padx=(0, 8))
        self.onnx_device_choice = ttk.Combobox(controls, textvariable=self.onnx_device_var, values=('AUTO',), state='disabled', width=12)
        self.onnx_device_choice.pack(side='left')
        self.onnx_device_choice.bind('<<ComboboxSelected>>', self._set_onnx_device)
        ttk.Button(controls, text='Test device', command=self._test_onnx_device).pack(side='left', padx=8)
        from manuals_rag_inference.onnx_runtime import INTEL_DRIVER_PAGE
        ttk.Button(controls, text='Intel driver setup', command=lambda: self._open_settings_link(INTEL_DRIVER_PAGE, 'Intel driver setup opened. The operating system may require administrator approval.')).pack(side='right')

        _, status, detail, controls = row("ollama", "Ollama", "Local inference engine; Install downloads the latest official installer")
        self.settings_vars.update({"ollama_status": status, "ollama_detail": detail})
        self.settings_buttons["ollama_install"] = ttk.Button(controls, text="Install", command=lambda: self._start_managed_install("ollama"), state="disabled")
        self.settings_buttons["ollama_install"].pack(side="left")
        ttk.Button(controls, text="Official download", command=lambda: self._open_settings_link(OLLAMA_DOWNLOAD_PAGE, "Ollama download page opened.")).pack(side="left", padx=8)

        _, status, detail, controls = row("gpu", "GPU driver", "NVIDIA driver and accelerator availability")
        self.settings_vars.update({"gpu_status": status, "gpu_detail": detail})
        ttk.Button(controls, text="Check for updates", command=self._check_driver_updates).pack(side="left")
        self.settings_buttons["gpu_install"] = ttk.Button(controls, text="Install driver", command=lambda: self._start_managed_install("gpu"), state="disabled")
        self.settings_buttons["gpu_install"].pack(side="left", padx=8)

        _, status, detail, controls = row("tailscale", "Tailscale", "Private network connection; Install downloads the latest official client")
        self.settings_vars.update({"tailscale_status": status, "tailscale_detail": detail})
        self.settings_buttons["tailscale_install"] = ttk.Button(controls, text="Install", command=lambda: self._start_managed_install("tailscale"), state="disabled")
        self.settings_buttons["tailscale_install"].pack(side="left")
        self.settings_buttons["tailscale_disconnect"] = ttk.Button(controls, text="Disconnect", command=lambda: self._tailscale_action("disconnect"), state="disabled")
        self.settings_buttons["tailscale_disconnect"].pack(side="left", padx=8)
        self.settings_buttons["tailscale_reconnect"] = ttk.Button(controls, text="Reconnect", command=lambda: self._tailscale_action("reconnect"), state="disabled")
        self.settings_buttons["tailscale_reconnect"].pack(side="left")
        ttk.Button(controls, text="Official download", command=lambda: self._open_settings_link(TAILSCALE_DOWNLOAD_PAGE, "Tailscale download page opened.")).pack(side="right")

        note = tk.StringVar(value="Install actions are enabled only when an administrator-bundled installer is present.")
        self.settings_vars["settings_note"] = note
        tk.Label(body, textvariable=note, bg=colors["background"], fg=colors["muted"], justify="left", wraplength=510, font=(self.ui_font, 9)).pack(anchor="w", pady=(14, 0))
        self._apply_theme()
        self._refresh_settings()

    def _open_settings_link(self, url: str, message: str) -> None:
        webbrowser.open(url)
        if "settings_note" in self.settings_vars:
            self.settings_vars["settings_note"].set(message)

    def _refresh_settings(self) -> None:
        if self.settings_panel is None or not self.settings_panel.winfo_exists():
            return
        for key in ("ollama_status", "gpu_status", "tailscale_status", 'onnx_status'):
            if key in self.settings_vars:
                self.settings_vars[key].set("Checking…")
        endpoint = self.fields["ollama_url"].get() if "ollama_url" in self.fields else "http://127.0.0.1:11434"
        threading.Thread(target=self._settings_status_worker, args=(endpoint,), daemon=True, name="provis-settings-status").start()

    def _settings_status_worker(self, endpoint: str) -> None:
        try:
            result = system_status(endpoint)
            self.worker_queue.put(("settings_status", result))
        except Exception as error:
            self.worker_queue.put(("settings_status_error", error.__class__.__name__))

    def _render_settings(self, statuses: dict[str, dict[str, object]]) -> None:
        if self.settings_panel is None or not self.settings_panel.winfo_exists():
            return
        ollama = statuses.get("ollama", {})
        gpu = statuses.get("gpu", {})
        tailscale = statuses.get("tailscale", {})
        from manuals_rag_inference.onnx_runtime import device_choices
        onnx = statuses.get('onnx', {})
        labels = {'intel_gpu_ready': 'Intel GPU ready', 'cpu_ready': 'CPU ready · no Intel GPU detected',
                  'not_installed': 'Updated signed app required', 'provider_missing': 'OpenVINO provider missing'}
        self.settings_vars['onnx_status'].set(labels.get(onnx.get('status'), 'Unavailable'))
        device_names = ', '.join(item['name'] for item in onnx.get('devices', [])) or 'No devices detected'
        self.settings_vars['onnx_detail'].set(('Installed automatically with this app' if onnx.get('bundled') and onnx.get('installed') else 'Installed in this environment' if onnx.get('installed') else 'Install the current signed PROVIS app to include the runtime')
            + (f" · Runtime {onnx['version']}" if onnx.get('version') else '') + '\n' + device_names
            + '\nIntel GPU availability requires a compatible operating-system driver. ONNX models are assigned separately from Ollama models.')
        self.onnx_device_choice.configure(values=device_choices(onnx), state='readonly' if onnx.get('installed') and not os.getenv('INFERENCE_ONNX_DEVICE') else 'disabled')
        ollama_installed = bool(ollama.get("installed"))
        tailscale_installed = bool(tailscale.get("installed"))
        self.settings_vars["ollama_status"].set(str(ollama.get("status", "unknown")).replace("_", " ").title())
        ollama_detail = str(ollama.get("version") or "Local API is not responding")
        self.settings_vars["ollama_detail"].set(f"Version: {ollama_detail}")
        self.settings_vars["gpu_status"].set(str(gpu.get("status", "unknown")).replace("_", " ").title())
        gpu_names = ", ".join(str(name) for name in gpu.get("gpus", [])[:3])
        self.settings_vars["gpu_detail"].set(f"Driver: {gpu.get('driver') or 'not detected'}" + (f" • {gpu_names}" if gpu_names else ""))
        self.settings_vars["tailscale_status"].set(str(tailscale.get("status", "unknown")).replace("_", " ").title())
        tail_detail = str(tailscale.get("version") or "")
        if tailscale.get("ipv4"):
            tail_detail += (" • " if tail_detail else "") + f"IPv4: {tailscale['ipv4']}"
        if not tailscale_installed:
            tail_detail = "Tailscale is not installed"
        elif tailscale.get("status") == "unknown":
            tail_detail = "Could not confirm Tailscale status; check permissions or refresh status"
        self.settings_vars["tailscale_detail"].set(tail_detail)
        self.settings_buttons["ollama_install"].configure(state="disabled" if ollama_installed or (managed_installer("ollama") is None and official_installer_url("ollama") is None) else "normal")
        self.settings_buttons["gpu_install"].configure(state="normal" if managed_installer("gpu") is not None else "disabled")
        self.settings_buttons["tailscale_install"].configure(state="disabled" if tailscale_installed or (managed_installer("tailscale") is None and official_installer_url("tailscale") is None) else "normal")
        self.settings_buttons["tailscale_disconnect"].configure(state="normal" if tailscale.get("connected") else "disabled")
        self.settings_buttons["tailscale_reconnect"].configure(state="normal" if tailscale_installed and tailscale.get("status") == "disconnected" else "disabled")

    def _start_managed_install(self, component: str) -> None:
        if managed_installer(component) is None and official_installer_url(component) is None:
            self.settings_vars["settings_note"].set("No supported official installer is available for this platform.")
            return
        source = "bundled" if managed_installer(component) is not None else "official online"
        if not messagebox.askyesno("Install component", f"Download and launch the {source} {component} installer? The operating system may request administrator approval.", parent=self.settings_panel):
            return
        threading.Thread(target=self._installer_worker, args=(component,), daemon=True, name=f"provis-install-{component}").start()

    def _set_onnx_device(self, event=None) -> None:
        device = self.onnx_device_var.get()
        try:
            if self.agent is not None:
                self.agent.set_onnx_device(device)
            elif self.attached:
                self._attached_request('/v1/node/settings', method='POST', payload={'action': 'set_onnx_device', 'device': device})
            else:
                from manuals_rag_inference.onnx_runtime import runtime_status, device_choices
                if device not in device_choices(runtime_status()):
                    raise ValueError('Selected device is unavailable')
                config = _load_node_config(self.config_path)
                config['onnx_device'] = device
                _save_node_config(self.config_path, config)
            self.settings_vars['settings_note'].set('Accelerator preference saved. AUTO selects an available device for assigned ONNX models.')
        except Exception as error:
            self.settings_vars['settings_note'].set(f'Accelerator preference was not changed: {error.__class__.__name__}. Refresh status and wait for active work to finish.')
            self.onnx_device_var.set(str(_load_node_config(self.config_path).get('onnx_device') or 'AUTO'))

    def _test_onnx_device(self):
        self.settings_vars['settings_note'].set('Testing the selected device with a synthetic tensor…')
        threading.Thread(target=self._onnx_probe_worker, args=(self.onnx_device_var.get(),), daemon=True).start()

    def _onnx_probe_worker(self, device):
        from manuals_rag_inference.onnx_runtime import probe_runtime
        try:
            result = probe_runtime(device)
            self.worker_queue.put(('onnx_probe', f"ONNX test passed on {result['device']} using OpenVINO."))
        except Exception as error:
            self.worker_queue.put(('onnx_probe', f'ONNX test failed: {error.__class__.__name__}. Refresh status and check the selected device driver.'))

    def _installer_worker(self, component: str) -> None:
        self.worker_queue.put(("settings_action", (component, launch_managed_installer(component))))

    def _check_driver_updates(self) -> None:
        self._open_settings_link(vendor_driver_page(), "The vendor driver page was opened. Review the offered driver before installing it.")

    def _tailscale_action(self, action: str) -> None:
        if action == "disconnect" and self.agent is not None:
            self.settings_vars["settings_note"].set("Disconnect the PROVIS node from the gateway first, then disconnect Tailscale.")
            return
        if not messagebox.askyesno("Tailscale", f"{action.title()} the private Tailscale connection?", parent=self.settings_panel):
            return
        threading.Thread(target=self._tailscale_worker, args=(action,), daemon=True, name=f"provis-tailscale-{action}").start()

    def _tailscale_worker(self, action: str) -> None:
        result = tailscale_disconnect() if action == "disconnect" else tailscale_reconnect()
        self.worker_queue.put(("settings_action", (f"Tailscale {action}", result)))

    def _apply_theme(self):
        colors = PALETTES[self.theme]
        self.theme_button_var.set('Light mode' if self.theme == 'dark' else 'Dark mode')
        background_roles = {'#f3f6fb': 'background', 'white': 'card', '#ffffff': 'card', '#122743': 'header'}
        foreground_roles = {'#122743': 'text', '#65758c': 'muted', '#165d69': 'link'}
        for palette in PALETTES.values():
            background_roles.update({palette[role]: role for role in ('background', 'card', 'header')})
            foreground_roles.update({palette[role]: role for role in ('text', 'muted', 'link')})
        def visit(widget):
            if isinstance(widget, (tk.Tk, tk.Toplevel, tk.Frame, tk.LabelFrame, tk.Label, tk.Button, tk.Text, tk.Canvas)):
                if not hasattr(widget, '_appearance_roles'):
                    roles = {}
                    for option, mapping in [('background', background_roles), ('foreground', foreground_roles)]:
                        try:
                            role = mapping.get(widget.cget(option))
                            if role: roles[option] = role
                        except tk.TclError:
                            pass
                    widget._appearance_roles = roles
                widget.configure(**{option: colors[role] for option, role in widget._appearance_roles.items()})
                if isinstance(widget, tk.Text): widget.configure(insertbackground=colors['text'])
            for child in widget.winfo_children(): visit(child)
        visit(self)
        if self.settings_panel is not None and self.settings_panel.winfo_exists():
            visit(self.settings_panel)
        style = ttk.Style(self)
        style.configure('TButton', background=colors['button'], foreground=colors['text'])
        style.map('TButton', background=[('active', colors['hover'])], foreground=[('disabled', colors['muted'])])
        style.configure('Primary.TButton', background='#225fe5', foreground='white')
        style.map('Primary.TButton', background=[('active', '#174fc9'), ('disabled', colors['button'])], foreground=[('disabled', colors['muted'])])
        style.configure('Node.Treeview', background=colors['card'], fieldbackground=colors['card'], foreground=colors['text'])
        style.map('Node.Treeview', background=[('selected', '#225fe5')], foreground=[('selected', 'white')])
        style.configure('Node.Treeview.Heading', background=colors['background'], foreground=colors['muted'])
        style.configure('Node.Horizontal.TProgressbar', troughcolor=colors['track'], background='#1c9b8a')

    def _position_panel(self):
        expanded = bool(getattr(self, 'owner_drawer', None) and self.owner_drawer.expanded)
        width, target_height = (760, 1040) if expanded else (440, 740)
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            area = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(area), 0):
                height = min(target_height, area.bottom - area.top - 48)
                self.geometry(f"{width}x{height}+{max(area.left, area.right - width - 16)}+{max(area.top, area.bottom - height - 40)}")
        else:
            height = min(target_height, self.winfo_screenheight() - 80)
            self.geometry(f"{width}x{height}+{max(0, self.winfo_screenwidth() - width - 16)}+{max(0, self.winfo_screenheight() - height - 64)}")

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
        if config.get("shared_secret") and not self.attach_existing:
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

    def _copy_inference_endpoint(self, endpoint: str) -> None:
        copied = copy_endpoint_to_clipboard(
            endpoint,
            local_broker_endpoints(getattr(getattr(self.server, "local_broker_server", None), "server_port", 0)),
            clipboard_clear=self.clipboard_clear,
            clipboard_append=self.clipboard_append,
            update=self.update,
        )
        self.endpoint_copy_var.set(
            "Endpoint URL copied. Local API authentication may be required by your administrator."
            if copied else "Endpoint could not be copied."
        )

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
        from manuals_rag_inference.node_updater import platform_id
        plan = getattr(self, 'update_plan', None)
        startup_enabled = plan['previous_enabled'] if plan else True
        metadata = {'node_app_version': self.update_settings['version'], 'node_app_platform': platform_id(),
                    'remote_updates_enabled': self.update_settings['enabled'] and (bool(getattr(sys, 'frozen', False)) or Path(sys.argv[0]).suffix == '.pyz')}
        server, agent, gateway_url = create_node_server(self.config_path, app_metadata=metadata, startup_enabled=startup_enabled)
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
            user_code = str(device.get("user_code") or "").strip()
            if not user_code:
                raise RuntimeError("Microsoft sign-in did not return a user code")
            self.worker_queue.put(("entra_prompt", user_code))
            verification_url = str(device.get("verification_uri_complete") or device.get("verification_uri") or "")
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
            config = _load_node_config(self.config_path)
            phase = "entra_registration_check"
            result = returning_node_registration(
                config=config,
                gateway_url=values["gateway_url"],
                tenant_id=values["entra_tenant_id"],
                client_id=values["entra_client_id"],
                scope=values["entra_scope"],
                access_token=str(token["access_token"]),
                node_id=values["node_id"],
                verify=verify_node_registration_with_entra,
                enroll=lambda: enroll_node_with_entra(
                    gateway_url=values["gateway_url"],
                    access_token=str(token["access_token"]),
                    node_id=values["node_id"],
                    endpoint=values["ollama_url"],
                    agent_url=values["agent_url"],
                    tailnet_name=values["tailnet_name"],
                ),
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
        if self.updater and self.updater.applying: return
        if self.agent is not None:
            self.agent.set_inference_enabled(enabled)
            self._update_controls()
            return
        if not self.attached:
            return
        try:
            result = self._attached_request(
                "/v1/node/settings",
                method="POST",
                payload={"action": "enable_inference" if enabled else "disable_inference"},
            )
            self.attached_status.update(result)
            self._update_controls()
        except Exception as error:
            self.status_var.set(f"Node control failed: {error.__class__.__name__}")

    def _disconnect(self) -> None:
        if self.owner_panel is not None: self.owner_panel._sign_out()
        if self.attached and self.agent is None:
            self._set_inference(False)
            self.status_var.set("Sharing paused on the connected node")
            return
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
                elif event == "settings_status":
                    self._render_settings(value)
                elif event == "settings_status_error":
                    if self.settings_panel is not None and self.settings_panel.winfo_exists():
                        for key in ("ollama_status", "gpu_status", "tailscale_status"):
                            self.settings_vars[key].set("Unavailable")
                        self.settings_vars["settings_note"].set("Local prerequisite checks could not complete; try Refresh status.")
                elif event == "settings_action":
                    action, result = value
                    status = str(result.get("status", "failed")) if isinstance(result, dict) else "failed"
                    if self.settings_panel is not None and self.settings_panel.winfo_exists():
                        self.settings_vars["settings_note"].set(f"{action}: {status}.")
                        self.after(1200, self._refresh_settings)
                elif event == 'onnx_probe':
                    if self.settings_panel is not None and self.settings_panel.winfo_exists():
                        self.settings_vars['settings_note'].set(value)
                elif event == "connected":
                    self._clear_signin_code()
                    self.server, self.agent, self.gateway_url = value
                    broker_server = getattr(self.server, "local_broker_server", None)
                    endpoints = local_broker_endpoints(getattr(broker_server, "server_port", 0))
                    self.local_endpoint_var.set(endpoints[0] if endpoints else "Local inference endpoint disabled")
                    self.local_endpoint_copy_button.configure(state="normal" if endpoints else "disabled")
                    self.attached = False
                    from manuals_rag_inference.node_updater import UpdateWorker
                    if self.update_settings['enabled'] and self.agent.app_metadata['remote_updates_enabled']:
                        self.updater = UpdateWorker(self.agent, self.gateway_url, self.config_path, self.update_settings,
                            self._owner_busy,
                            lambda plan: self.worker_queue.put(('app_update_apply', plan)))
                    self.status_var.set("Connected to PROVIS")
                    self._update_controls()
                elif event == 'app_update_apply':
                    from manuals_rag_inference.node_updater import command
                    import subprocess
                    flags = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                    subprocess.Popen(command()+['--update-helper',str(value)], **flags)
                    self._stop_server()
                    self._finish_quit()
                    return
                elif event == "disconnected":
                    self._clear_signin_code()
                    self._stop_server()
                    self.attached = False
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
        connected = self.agent is not None or self.attached
        self.entra_button.configure(state="disabled" if connected else "normal")
        state = "normal" if connected and not (self.updater and self.updater.applying) else "disabled"
        self.enable_button.configure(state=state)
        self.disable_button.configure(state=state)
        self.disconnect_button.configure(state="normal" if self.agent is not None else "disabled")
        if self.agent is not None:
            self.inference_var.set("Inference enabled" if self.agent.inference_enabled else "Inference disabled")
        elif self.attached:
            self.inference_var.set("Inference enabled" if self.attached_status.get("inference_enabled") else "Inference disabled")
        else:
            self.inference_var.set("Inference disabled")

    def _refresh_activity(self) -> None:
        self._update_controls()
        events = []
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
        elif self.attached:
            try:
                status = self._attached_request("/v1/node/status")
                self.attached_status.update(status)
                events = self._attached_request("/v1/node/activity").get("events", [])
                self.status_var.set("Connected to PROVIS")
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
                self.work_var.set("No active work")
            except Exception:
                self.attached = False
                self.status_var.set("PROVIS service is not available")
        self.after(1000, self._refresh_activity)

    def _owner_busy(self):
        runtime = self.owner_panel.runtime if self.owner_panel else None
        if runtime:
            with runtime.delivery_lock: return runtime.busy.is_set()
        return False

    def _stop_server(self) -> None:
        if self.updater: self.updater.stop.set();self.updater = None
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
        if self.updater: self.updater.stop.set()
        if self.owner_panel is not None: self.owner_panel.stop()
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
    if not getattr(sys, 'frozen', False) and len(sys.argv) > 1 and sys.argv[1] == '--agent-link':
        sys.argv.pop(1)
        from provis_agent_link import main as owner_tool
        owner_tool()
        return
    from manuals_rag_inference.node_updater import helper, active_app, command, validate_plan, private_json, _report_plan
    import subprocess
    settings = _managed_update_settings()
    config_path = os.getenv('INFERENCE_NODE_CONFIG_PATH', _default_config_path())
    mode = sys.argv[1] if len(sys.argv)>1 else ''
    if mode == '--self-test':
        from manuals_rag_inference.onnx_runtime import require_bundled_runtime, probe_runtime
        onnx_status = require_bundled_runtime()
        onnx_probe = probe_runtime('CPU')
        print(json.dumps({
            'gateway_configured': bool(_load_managed_gateway_url()),
            'updater_enabled': bool(settings['enabled']),
            'python_embedded': bool(getattr(sys, 'frozen', False)),
            'onnx_runtime_version': onnx_status['version'],
            'openvino_provider': 'OpenVINOExecutionProvider' in onnx_status['providers'],
            'intel_gpu_count': len(onnx_status['intel_gpus']),
            'onnx_probe': onnx_probe,
        }, sort_keys=True))
        return
    if mode in ('--update-helper', '--update-probe', '--update-rollback') and (not settings['enabled'] or len(sys.argv)!=3):
        raise PermissionError('Trusted updater bootstrap required')
    if mode == '--update-helper':
        helper(sys.argv[2], config_path, settings)
        return
    probe = None
    if mode in ('--update-probe','--update-rollback'):
        probe = validate_plan(sys.argv[2], config_path, settings)
    elif settings['enabled']:
        target = active_app(config_path, settings)
        if target:
            subprocess.Popen(command(target), creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            return
    instance = SingleInstance()
    if instance.existing:
        instance.close()
        return
    try:
        app = InferenceNodeClient()
        if probe:
            path, plan, manifest = probe
            app.update_plan = plan
            def check_update_health():
                if app.agent and app.server and app.agent.heartbeat_accepted.is_set() and app.agent.node_id == plan['node_id']:
                    if mode == '--update-probe':
                        if app.update_settings['version'] != manifest['version']:
                            app._stop_server();app._finish_quit();return
                        private_json(path.parent/'healthy.json', {'nonce':plan['probe_nonce'], 'version':manifest['version'], 'node_id':plan['node_id']})
                    else: _report_plan(plan,'rolled_back')
                    return
                if time.monotonic() > health_deadline:
                    app._stop_server();app._finish_quit();return
                app.after(500, check_update_health)
            health_deadline = time.monotonic()+75
            app.after(500,check_update_health)
        instance.watch(lambda event, value: app.worker_queue.put((event, value)))
        app.mainloop()
    finally:
        instance.close()


if __name__ == "__main__":
    main()
