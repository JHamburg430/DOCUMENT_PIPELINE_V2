#!/usr/bin/env python3
"""Monitor a UI-backed RAG run and persist terminal state for Codex follow-through."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    terminal = {"completed", "failed", "cancelled", "canceled", "stopped"}
    state_path = Path(args.state_file)
    while True:
        with urlopen(args.status_url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        runs = payload.get("runs") or []
        run = next((item for item in runs if str(item.get("run_id")) == args.run_id), None)
        snapshot = {"run_id": args.run_id, "observed_at": time.time(), "run": run, "payload": payload}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
        if run and str(run.get("status", "")).lower() in terminal:
            return 0 if run.get("status") == "completed" else 1
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
