"""Pure helpers for Last-Event-ID replay and Server-Sent Events encoding."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

try:
    from .event_envelope import EventEnvelope, make_event
except ImportError:  # pragma: no cover - direct server.py execution
    from event_envelope import EventEnvelope, make_event


class ReplayRequestError(ValueError):
    pass


def parse_replay_cursor(last_event_id: str | None, after: str | None) -> int:
    """Return a validated cursor, preferring the standard SSE header."""
    raw = last_event_id if last_event_id not in (None, "") else after
    if raw in (None, ""):
        return 0
    try:
        cursor = int(raw)
    except (TypeError, ValueError) as error:
        raise ReplayRequestError("Last-Event-ID/after must be an integer") from error
    if cursor < 0:
        raise ReplayRequestError("Last-Event-ID/after must be non-negative")
    return cursor


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        return value
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def envelope_from_persisted_row(
    run_id: str,
    row: Mapping[str, Any],
    *,
    workflow: str = "application",
) -> EventEnvelope:
    """Wrap an existing app_run_events row without changing its stored payload."""
    sequence = row.get("event_index")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ReplayRequestError("persisted event_index must be a positive integer")
    raw = row.get("event_json")
    if not isinstance(raw, dict):
        raise ReplayRequestError("persisted event_json must be an object")
    event_name = str(raw.get("event") or raw.get("type") or "progress")
    raw_status = str(raw.get("status") or "running")
    completed = bool(raw.get("completed")) or raw_status in {"completed", "succeeded", "failed", "cancelled"}
    stable_id = str(uuid5(NAMESPACE_URL, f"manuals-rag:{run_id}:{sequence}"))
    entity_id = str(raw.get("entity_id") or raw.get("case_id") or raw.get("question_id") or run_id)
    return make_event(
        event_id=stable_id,
        sequence=sequence,
        timestamp=_timestamp(row.get("created_at")),
        run_id=run_id,
        parent_run_id=raw.get("parent_run_id") if isinstance(raw.get("parent_run_id"), str) else None,
        workflow=str(raw.get("workflow") or workflow),
        phase=str(raw.get("phase") or event_name),
        entity_type=str(raw.get("entity_type") or "run"),
        entity_id=entity_id,
        status=raw_status,
        completed=completed,
        total=raw.get("total") if isinstance(raw.get("total"), int) and not isinstance(raw.get("total"), bool) else None,
        provisional=bool(raw.get("provisional", not completed)),
        payload=dict(raw),
        artifact_ref=raw.get("artifact_ref") if isinstance(raw.get("artifact_ref"), str) else None,
    )


def encode_sse(event: EventEnvelope, *, event_name: str = "run-event") -> bytes:
    data = json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=False)
    return f"id: {event.sequence}\nevent: {event_name}\ndata: {data}\n\n".encode("utf-8")


def encode_snapshot(snapshot: Mapping[str, Any]) -> bytes:
    data = json.dumps(dict(snapshot), separators=(",", ":"), ensure_ascii=False, default=str)
    return f"event: snapshot\ndata: {data}\n\n".encode("utf-8")


def encode_heartbeat() -> bytes:
    return b": keep-alive\n\n"
