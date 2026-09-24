"""Update-safe bounded event journal backed by a private SQLite file.

This module never creates, alters, or deletes PostgreSQL objects. The existing
``app_run_events`` table remains an immutable application-owned event source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any
from uuid import NAMESPACE_URL, uuid5

try:
    from .event_envelope import EventEnvelope, EventValidationError, make_event
except ImportError:  # pragma: no cover - direct server.py execution
    from event_envelope import EventEnvelope, EventValidationError, make_event


class JournalError(RuntimeError):
    pass


class JournalConflictError(JournalError):
    pass


@dataclass(frozen=True, slots=True)
class ReplayPage:
    events: tuple[EventEnvelope, ...]
    oldest_sequence: int | None
    newest_sequence: int | None
    truncated: bool


class SQLiteEventJournal:
    """A per-run bounded, ordered, idempotent journal.

    The database and schema are owned solely by this component, making the
    journal safe to add without coupling to application PostgreSQL migrations.
    """

    def __init__(self, path: str | Path, *, max_events_per_run: int = 2000) -> None:
        if max_events_per_run < 1:
            raise ValueError("max_events_per_run must be positive")
        self.path = Path(path)
        self.max_events_per_run = max_events_per_run
        self._lock = RLock()
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute("pragma journal_mode = wal")
            self._connection.execute(
                """
                create table if not exists events (
                    event_id text primary key,
                    run_id text not null,
                    sequence integer not null,
                    event_json text not null,
                    created_at text not null,
                    unique (run_id, sequence)
                )
                """
            )
            self._connection.execute(
                "create index if not exists events_run_sequence on events(run_id, sequence)"
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def append(self, event: EventEnvelope) -> EventEnvelope:
        encoded = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock, self._connection:
            existing = self._connection.execute(
                "select event_json from events where event_id = ?", (event.event_id,)
            ).fetchone()
            if existing:
                if existing["event_json"] != encoded:
                    raise JournalConflictError("event_id already exists with different content")
                return EventEnvelope.from_mapping(json.loads(existing["event_json"]))
            try:
                self._connection.execute(
                    "insert into events(event_id, run_id, sequence, event_json, created_at) values (?, ?, ?, ?, ?)",
                    (event.event_id, event.run_id, event.sequence, encoded, event.timestamp),
                )
            except sqlite3.IntegrityError as error:
                raise JournalConflictError("run sequence already belongs to another event") from error
            self._connection.execute(
                """
                delete from events
                where run_id = ? and sequence <= (
                    select coalesce(max(sequence), 0) - ? from events where run_id = ?
                )
                """,
                (event.run_id, self.max_events_per_run, event.run_id),
            )
        return event

    def publish(
        self,
        *,
        run_id: str,
        workflow: str,
        phase: str,
        entity_type: str,
        entity_id: str,
        status: str,
        completed: bool = False,
        total: int | None = None,
        provisional: bool = False,
        payload: dict[str, Any] | None = None,
        artifact_ref: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> EventEnvelope:
        with self._lock:
            if event_id:
                existing = self._connection.execute(
                    "select event_json from events where event_id = ?", (event_id,)
                ).fetchone()
                if existing:
                    saved = EventEnvelope.from_mapping(json.loads(existing["event_json"]))
                    requested = {
                        "run_id": run_id,
                        "parent_run_id": parent_run_id,
                        "workflow": workflow,
                        "phase": phase,
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "status": status,
                        "completed": completed,
                        "total": total,
                        "provisional": provisional,
                        "payload": payload or {},
                        "artifact_ref": artifact_ref,
                    }
                    if any(getattr(saved, key) != value for key, value in requested.items()):
                        raise JournalConflictError("event_id already exists with different content")
                    return saved
            row = self._connection.execute(
                "select coalesce(max(sequence), 0) as value from events where run_id = ?", (run_id,)
            ).fetchone()
            event = make_event(
                sequence=int(row["value"]) + 1,
                run_id=run_id,
                workflow=workflow,
                phase=phase,
                entity_type=entity_type,
                entity_id=entity_id,
                status=status,
                completed=completed,
                total=total,
                provisional=provisional,
                payload=payload,
                artifact_ref=artifact_ref,
                parent_run_id=parent_run_id,
                event_id=event_id,
                timestamp=timestamp,
            )
            return self.append(event)

    def replay(self, run_id: str, *, after: int = 0, limit: int = 500) -> ReplayPage:
        if after < 0 or limit < 1:
            raise ValueError("after must be non-negative and limit must be positive")
        bounded_limit = min(limit, self.max_events_per_run)
        with self._lock:
            bounds = self._connection.execute(
                "select min(sequence) as oldest, max(sequence) as newest from events where run_id = ?",
                (run_id,),
            ).fetchone()
            rows = self._connection.execute(
                "select event_json from events where run_id = ? and sequence > ? order by sequence limit ?",
                (run_id, after, bounded_limit),
            ).fetchall()
        oldest = bounds["oldest"]
        newest = bounds["newest"]
        return ReplayPage(
            events=tuple(EventEnvelope.from_mapping(json.loads(row["event_json"])) for row in rows),
            oldest_sequence=oldest,
            newest_sequence=newest,
            truncated=oldest is not None and after + 1 < oldest,
        )


class ProgressJsonlBridge:
    """Fail-closed adapter for newline-delimited CLI progress objects."""

    def __init__(self, journal: SQLiteEventJournal, *, run_id: str, workflow: str) -> None:
        self.journal = journal
        self.run_id = run_id
        self.workflow = workflow

    def consume(self, line: str) -> EventEnvelope:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EventValidationError("progress JSONL line is not valid JSON") from error
        if not isinstance(raw, dict):
            raise EventValidationError("progress JSONL line must contain an object")
        event_name = raw.get("event")
        if not isinstance(event_name, str) or not event_name.strip():
            raise EventValidationError("progress JSONL object must contain a non-empty event")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        event_id = str(uuid5(NAMESPACE_URL, f"{self.run_id}:{digest}"))
        status = str(raw.get("status") or ("completed" if raw.get("completed") else "running"))
        return self.journal.publish(
            run_id=self.run_id,
            workflow=self.workflow,
            phase=event_name,
            entity_type=str(raw.get("entity_type") or "run"),
            entity_id=str(raw.get("entity_id") or self.run_id),
            status=status,
            completed=bool(raw.get("completed", status in {"completed", "succeeded", "failed", "cancelled"})),
            total=raw.get("total") if isinstance(raw.get("total"), int) else None,
            provisional=bool(raw.get("provisional", True)),
            payload=raw,
            artifact_ref=raw.get("artifact_ref") if isinstance(raw.get("artifact_ref"), str) else None,
            event_id=event_id,
        )
