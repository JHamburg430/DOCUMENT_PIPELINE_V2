"""Small Tk GUI for the standalone Windows installer."""

from __future__ import annotations

import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from provis_node_windows_installer import (
    APP_EXE_NAME,
    InstallError,
    WindowsNodeInstaller,
    resolve_known_folders,
)


def run_gui(icon_png: bytes, *, smoke_test: bool = False) -> int:
    root = tk.Tk()
    root.title("PROVIS Inference Node Setup")
    root.geometry("520x260")
    root.minsize(460, 240)
    root.resizable(False, False)
    root.configure(background="#101820")
    # Tk uses its platform's named default font. Passing a family with spaces
    # as a flat Tcl font string (for example, "Segoe UI 10") is parsed as
    # separate fields and can fail before the installer window is created.

    frame = ttk.Frame(root, padding=22)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="PROVIS Inference Node", font=("Segoe UI", 17, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Install or repair this app for the current Windows user.").pack(anchor="w", pady=(3, 16))
    status = tk.StringVar(value="Ready. Your enrollment and settings are kept.")
    ttk.Label(frame, textvariable=status, wraplength=470).pack(anchor="w", pady=(0, 10))
    progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
    progress.pack(fill="x", pady=(0, 14))
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")

    events: queue.Queue = queue.Queue()
    running = False
    installed_path = None

    def process_events():
        nonlocal running, installed_path
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == "progress":
                    label, current, total = value
                    status.set(label)
                    if total:
                        progress.configure(mode="determinate", maximum=total, value=current)
                    else:
                        progress.configure(mode="indeterminate")
                        progress.start(10)
                elif kind == "done":
                    running = False
                    for child in buttons.winfo_children():
                        child.configure(state="normal")
                    progress.stop()
                    progress.configure(mode="determinate", value=0)
                    action, result = value
                    if action == "install":
                        installed_path = result
                        status.set("Verified app installed. Use the shortcut or select Launch below.")
                        launch_button.configure(state="normal")
                        close_button.configure(text="Close")
                    elif action == "uninstall":
                        status.set(result)
                        launch_button.configure(state="disabled")
                    else:
                        status.set("Ready.")
                elif kind == "error":
                    running = False
                    for child in buttons.winfo_children():
                        child.configure(state="normal")
                    progress.stop()
                    progress.configure(mode="determinate", value=0)
                    status.set("The operation stopped. Your settings were not intentionally changed.")
                    messagebox.showerror("PROVIS Inference Node Setup", value, parent=root)
        except queue.Empty:
            pass
        if root.winfo_exists():
            root.after(100, process_events)

    def run_operation(action: str):
        nonlocal running
        if running:
            return
        if action == "uninstall" and not messagebox.askyesno(
                "Remove PROVIS Inference Node?",
                "Remove this installer's app files and shortcuts? Enrollment and settings will remain.",
                parent=root):
            return
        running = True
        launch_button.configure(state="disabled")
        for child in buttons.winfo_children():
            child.configure(state="disabled")
        status.set("Preparing…")
        progress.configure(mode="indeterminate")
        progress.start(10)

        def worker():
            try:
                folders = resolve_known_folders()
                installer = WindowsNodeInstaller(folders=folders, icon_png=icon_png)
                callback = lambda label, current, total: events.put(("progress", (label, current, total)))
                if action == "uninstall":
                    removed = installer.uninstall_owned(progress=callback)
                    if removed:
                        suffix = " Custom files were preserved." if installer.version_root.exists() else ""
                        events.put(("done", (action, "Installer-owned app files and shortcuts removed; settings kept." + suffix)))
                    else:
                        events.put(("done", (action, "No installer-owned release was found; nothing was removed.")))
                else:
                    events.put(("progress", ("Verifying the pinned release", 0, 0)))
                    target = installer.install_or_repair(progress=callback)
                    events.put(("done", (action, target)))
            except Exception as error:
                events.put(("error", str(error)))

        threading.Thread(target=worker, name="provis-installer", daemon=True).start()

    def launch_app():
        if installed_path is None:
            return
        try:
            folders = resolve_known_folders()
            installer = WindowsNodeInstaller(folders=folders, icon_png=icon_png)
            installer._verify_installed()
            process = subprocess.Popen([str(installed_path)], cwd=str(installed_path.parent),
                                       close_fds=True)
            status.set(f"Launched {APP_EXE_NAME} (process {process.pid}).")
        except Exception as error:
            messagebox.showerror("Could not launch PROVIS Inference Node", str(error), parent=root)

    def request_close():
        if running:
            messagebox.showinfo(
                "Setup is still working",
                "Wait for the current install or removal to finish before closing this window.",
                parent=root,
            )
            return
        root.destroy()

    install_button = ttk.Button(buttons, text="Install / Repair", command=lambda: run_operation("install"))
    install_button.pack(side="left")
    uninstall_button = ttk.Button(buttons, text="Uninstall", command=lambda: run_operation("uninstall"))
    uninstall_button.pack(side="left", padx=8)
    launch_button = ttk.Button(buttons, text="Launch", command=launch_app, state="disabled")
    launch_button.pack(side="left")
    close_button = ttk.Button(buttons, text="Close", command=request_close)
    close_button.pack(side="right")
    root.protocol("WM_DELETE_WINDOW", request_close)
    root.after(100, process_events)
    if smoke_test:
        # Exercise the real window/widget initialization path in CI without
        # requiring a click or ever starting an install operation.
        root.after(250, root.destroy)
    root.mainloop()
    return 0
