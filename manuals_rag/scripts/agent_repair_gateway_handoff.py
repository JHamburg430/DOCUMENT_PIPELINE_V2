"""Deliver queued PostgreSQL repair cases to visible OpenClaw Gateway sessions.

Runs on the Gateway host, where the OpenClaw CLI has its normal operator identity.
The UI container never receives Gateway credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


REPO = Path(__file__).resolve().parents[2]
DSN = os.environ.get("MANUALS_RAG_GATEWAY_REPAIR_DSN", "postgresql://manuals:manuals@127.0.0.1:5433/manuals_rag")


def deliver_one() -> bool:
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        with conn.transaction():
            row = conn.execute(
                """select request_id, source_run_id from agent_repair_cases
                   where gateway_handoff_status = 'pending'
                   order by created_at for update skip locked limit 1"""
            ).fetchone()
            if row is None:
                return False
            run_id = row["source_run_id"]
            key = f"agent:main:manuals-repair:{run_id}"
            prompt = (
                f"Diagnose Manuals RAG repair case {row['request_id']} for run {run_id}. "
                f"Use the server-owned PostgreSQL case and full run trace in {REPO / 'manuals_rag'}; "
                "follow manuals_rag/docs/agent_repair_history.md and the manuals-rag-repair-history skill. "
                "Reproduce before fixing, link any warranted fix and verification to the case, "
                "and report a no-fix conclusion when the failure is transient. "
                "Keep this visible session as the case's handoff and report evidence and remaining blockers."
            )
            params = {
                "key": key,
                "displayName": f"Manuals repair {run_id.removeprefix('agent-run-')}",
                "category": "Technical Docs RAG",
                "cwd": str(REPO),
            }
            try:
                described = subprocess.run(
                    ["openclaw", "gateway", "call", "sessions.describe", "--json",
                     "--timeout", "30000", "--params", json.dumps({"key": key})],
                    capture_output=True, text=True, timeout=40, check=True,
                )
                if not json.loads(described.stdout).get("session"):
                    command = ["openclaw", "gateway", "call", "sessions.create", "--json",
                               "--timeout", "240000", "--params", json.dumps(params)]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=250, check=True)
                    receipt = json.loads(result.stdout)
                    if not receipt.get("ok") or receipt.get("key") != key:
                        raise RuntimeError(str(receipt.get("error") or "Gateway did not create the case session"))
                conn.execute(
                    "update agent_repair_cases set gateway_session_key = %s where request_id = %s",
                    (key, row["request_id"]),
                )
                send = subprocess.run(
                    ["openclaw", "gateway", "call", "sessions.send", "--json",
                     "--timeout", "30000", "--params", json.dumps({"key": key, "message": prompt})],
                    capture_output=True, text=True, timeout=40, check=True,
                )
                delivery = json.loads(send.stdout)
                if delivery.get("status") not in {"started", "queued", "accepted"} and not delivery.get("runId"):
                    raise RuntimeError(str(delivery.get("error") or "Gateway did not accept the case message"))
            except subprocess.CalledProcessError as exc:
                detail = f"Gateway CLI exited {exc.returncode}: {(exc.stderr or '').strip()[:350]}"
                conn.execute(
                    """update agent_repair_cases set gateway_handoff_status = 'failed',
                       gateway_handoff_error = %s, updated_at = now() where request_id = %s""",
                    (detail, row["request_id"]),
                )
                print(f"handoff failed: {row['request_id']}: {detail}", file=sys.stderr, flush=True)
            except (subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
                detail = "Gateway handoff timed out" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)[:500]
                conn.execute(
                    """update agent_repair_cases set gateway_handoff_status = 'failed',
                       gateway_handoff_error = %s, updated_at = now() where request_id = %s""",
                    (detail, row["request_id"]),
                )
                print(f"handoff failed: {row['request_id']}: {detail}", file=sys.stderr, flush=True)
            else:
                conn.execute(
                    """update agent_repair_cases set gateway_session_key = %s,
                       gateway_handoff_status = 'sent', gateway_handoff_error = null,
                       updated_at = now() where request_id = %s""",
                    (key, row["request_id"]),
                )
                print(f"handoff sent: {row['request_id']} -> {key}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process at most one queued case")
    args = parser.parse_args()
    while True:
        try:
            processed = deliver_one()
        except psycopg.Error as exc:
            print(f"repair database unavailable: {exc.__class__.__name__}", file=sys.stderr, flush=True)
            processed = False
        if args.once:
            return
        if not processed:
            time.sleep(2)


if __name__ == "__main__":
    main()
