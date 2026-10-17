"""Allowlisted, content-free view of node activity for the desktop panel."""
from __future__ import annotations

import math
import time

LABELS = {
    "app_update": "App update",
    "heartbeat": "Gateway connection", "ollama": "Local engine", "control": "Compute sharing",
    "model_inventory": "Resources checked", "model_inventory_check": "Checking resources",
    "model_already_installed": "Local resources available", "model_pull_started": "Downloading resources",
    "model_pull_failed": "Resource download failed", "model_warm_started": "Preparing compute",
    "model_warm_completed": "Compute ready", "model_warm_failed": "Compute preparation failed",
    "chat": "Compute request", "generate": "Compute request", "embeddings": "Compute request",
    "inference": "Compute request", "disconnect": "Gateway disconnect",
    "tailscale": "Private connection", "pull_model": "Preparing resources",
}
STATES = {
    "downloading": "Downloading", "verified": "Verified", "waiting_for_idle": "Waiting for active work",
    "restarting": "Restarting app", "rolled_back": "Previous app restored", "interrupted": "Needs attention",
    "accepted": "Connected", "healthy": "Available", "available": "Available", "completed": "Complete",
    "success": "Complete", "started": "In progress", "failed": "Needs attention", "unavailable": "Unavailable",
    "enabled": "Enabled", "disabled": "Paused", "skipped": "Already available", "connected": "Connected",
    "disconnected": "Disconnected", "rejected": "Declined", "cancelled": "Cancelled",
}


def public_events(events: list[dict]) -> list[tuple[str, str, str]]:
    """Never render arbitrary upstream fields, model names or error text."""
    rows = []
    previous_heartbeat = None
    for event in events:
        operation = event.get("operation")
        state = STATES.get(event.get("status"), "Updated")
        if operation == "heartbeat":
            if previous_heartbeat == state:
                continue
            previous_heartbeat = state
        try:
            stamp = float(event.get("timestamp", 0))
            clock = time.strftime("%H:%M:%S", time.localtime(stamp)) if math.isfinite(stamp) else "—"
        except (TypeError, ValueError, OverflowError, OSError):
            clock = "—"
        rows.append((clock, LABELS.get(operation, "Node activity"), state))
    return rows[-30:]


def utilization(gpus) -> tuple[float, int, int]:
    total = sum(max(0, gpu.total_vram_mb) for gpu in gpus)
    used = sum(max(0, min(gpu.used_vram_mb, gpu.total_vram_mb)) for gpu in gpus)
    load = max((max(0, min(100, gpu.utilization_percent)) for gpu in gpus), default=0)
    return load, used, total
