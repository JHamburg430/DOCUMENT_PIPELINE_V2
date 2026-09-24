from __future__ import annotations

from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from threading import Thread
from time import monotonic
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest

from apps.ui.durable_journal import JournalConflictError, ProgressJsonlBridge, SQLiteEventJournal
from apps.ui.event_envelope import EventEnvelope, EventValidationError, make_event
from apps.ui.run_registry import RunRegistry
from apps.ui import server as ui_server
from apps.ui.sse_replay import (
    ReplayRequestError,
    encode_sse,
    envelope_from_persisted_row,
    parse_replay_cursor,
)


def _event(sequence: int, *, run_id: str = "run-1", event_id: str | None = None) -> EventEnvelope:
    return make_event(
        event_id=event_id,
        sequence=sequence,
        timestamp="2026-09-24T12:00:00.000Z",
        run_id=run_id,
        workflow="evaluation",
        phase="case_completed",
        entity_type="question",
        entity_id=f"question-{sequence}",
        status="running",
        payload={"value": sequence},
    )


def test_event_envelope_round_trip_is_complete_and_defensive():
    event = _event(1)
    data = event.to_dict()
    assert set(data) == {
        "event_id", "sequence", "timestamp", "run_id", "parent_run_id", "workflow",
        "phase", "entity_type", "entity_id", "status", "completed", "total",
        "provisional", "payload", "artifact_ref",
    }
    data["payload"]["value"] = 99
    assert event.payload == {"value": 1}
    assert EventEnvelope.from_mapping(event.to_dict()) == event


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"event_id": "not-a-uuid"}, "event_id"),
        ({"sequence": 0}, "sequence"),
        ({"timestamp": "2026-09-24T12:00:00"}, "timezone"),
        ({"payload": []}, "payload"),
        ({"completed": "false"}, "completed"),
    ],
)
def test_event_envelope_fails_closed(mutation, message):
    data = _event(1).to_dict()
    data.update(mutation)
    with pytest.raises(EventValidationError, match=message):
        EventEnvelope.from_mapping(data)


def test_event_envelope_rejects_missing_and_unknown_fields():
    data = _event(1).to_dict()
    data.pop("artifact_ref")
    with pytest.raises(EventValidationError, match="missing required"):
        EventEnvelope.from_mapping(data)
    data = _event(1).to_dict()
    data["surprise"] = True
    with pytest.raises(EventValidationError, match="unknown fields"):
        EventEnvelope.from_mapping(data)


def test_sqlite_journal_is_ordered_idempotent_and_bounded(tmp_path):
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3", max_events_per_run=2)
    first = _event(1)
    assert journal.append(first) == first
    assert journal.append(first) == first
    journal.append(_event(2))
    journal.append(_event(3))
    page = journal.replay("run-1", after=0, limit=20)
    assert [event.sequence for event in page.events] == [2, 3]
    assert page.oldest_sequence == 2
    assert page.newest_sequence == 3
    assert page.truncated is True
    journal.close()


def test_sqlite_journal_rejects_event_id_and_sequence_conflicts(tmp_path):
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3")
    first = _event(1)
    journal.append(first)
    with pytest.raises(JournalConflictError, match="event_id"):
        journal.append(_event(2, event_id=first.event_id))
    with pytest.raises(JournalConflictError, match="sequence"):
        journal.append(_event(1))


def test_progress_jsonl_bridge_is_deterministic_and_fail_closed(tmp_path):
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3")
    bridge = ProgressJsonlBridge(journal, run_id="cli-run", workflow="evaluation")
    line = '{"event":"case_completed","entity_id":"case-1","total":3}'
    first = bridge.consume(line)
    replayed = bridge.consume(line)
    assert replayed == first
    assert journal.replay("cli-run").events == (first,)
    with pytest.raises(EventValidationError, match="valid JSON"):
        bridge.consume("not-json")
    with pytest.raises(EventValidationError, match="non-empty event"):
        bridge.consume("{}")


def test_registry_tracks_subscribers_without_claiming_artifact_authority():
    registry = RunRegistry()
    registry.observe("run-1", workflow="evaluation", status="running", last_sequence=4)
    assert registry.subscriber_opened("run-1").subscribers == 1
    completed = registry.observe("run-1", status="completed", last_sequence=5)
    assert completed.terminal is True
    assert completed.metadata == {}
    assert registry.subscriber_closed("run-1").subscribers == 0


def test_replay_cursor_prefers_last_event_id_and_fails_closed():
    assert parse_replay_cursor("7", "3") == 7
    assert parse_replay_cursor(None, "3") == 3
    assert parse_replay_cursor(None, None) == 0
    with pytest.raises(ReplayRequestError):
        parse_replay_cursor("bad", None)
    with pytest.raises(ReplayRequestError):
        parse_replay_cursor("-1", None)


def test_persisted_row_adapter_and_sse_encoding_preserve_payload():
    row = {
        "event_index": 9,
        "event_json": {"event": "case_completed", "case_id": "case-9", "status": "completed"},
        "created_at": datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
    }
    event = envelope_from_persisted_row("run-9", row, workflow="evaluation")
    encoded = encode_sse(event).decode()
    assert event.sequence == 9
    assert event.payload == row["event_json"]
    assert encoded.startswith("id: 9\nevent: run-event\ndata: ")


class EventHandler(ui_server.ManualsRagUiHandler):
    event_rows = [
        {
            "event_index": 1,
            "event_json": {"event": "eval_started", "status": "running"},
            "created_at": "2026-09-24T12:00:00Z",
        },
        {
            "event_index": 2,
            "event_json": {"event": "eval_completed", "status": "completed", "completed": True},
            "created_at": "2026-09-24T12:00:00.050Z",
        },
    ]

    def _query_run_events(self, run_id, *, after, limit):
        assert run_id == "run-http"
        return [dict(row) for row in self.event_rows if row["event_index"] > after][:limit]

    def _query_run_snapshot(self, run_id):
        return {"id": run_id, "run_type": "evaluation", "status": "completed"}

    def log_message(self, *_args):
        return


def _serve():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), EventHandler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_snapshot_endpoint_preserves_poll_shape_and_adds_envelope():
    httpd = _serve()
    try:
        request = Request(
            f"http://127.0.0.1:{httpd.server_port}/local/run-events?run_id=run-http&after=1"
        )
        with urlopen(request, timeout=5) as response:
            rows = json.load(response)
            assert response.headers["X-Event-Replay-After"] == "1"
        assert [row["event_index"] for row in rows] == [2]
        assert rows[0]["event_json"]["event"] == "eval_completed"
        assert rows[0]["envelope"]["sequence"] == 2
    finally:
        httpd.shutdown()


def test_sse_endpoint_replays_from_last_event_id_with_subsecond_transport():
    httpd = _serve()
    started = monotonic()
    try:
        request = Request(
            f"http://127.0.0.1:{httpd.server_port}/local/run-events/subscribe?run_id=run-http",
            headers={"Last-Event-ID": "1", "Accept": "text/event-stream"},
        )
        with urlopen(request, timeout=5) as response:
            body = response.read().decode()
            assert response.headers["Content-Type"].startswith("text/event-stream")
            assert response.headers["X-Event-Replay-After"] == "1"
        elapsed = monotonic() - started
        assert "event: snapshot" in body
        assert "id: 2\nevent: run-event" in body
        assert "id: 1\n" not in body
        assert elapsed < 1.0
    finally:
        httpd.shutdown()


def test_http_endpoint_rejects_malformed_last_event_id():
    httpd = _serve()
    try:
        request = Request(
            f"http://127.0.0.1:{httpd.server_port}/local/run-events/subscribe?run_id=run-http",
            headers={"Last-Event-ID": "nope"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 400
    finally:
        httpd.shutdown()
