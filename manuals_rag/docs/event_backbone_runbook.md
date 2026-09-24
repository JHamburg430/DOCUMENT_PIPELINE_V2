# Real-time event backbone

## Scope and authority

The backbone is transport infrastructure only. REST remains the command path,
immutable result artifacts remain acceptance authority, and polling remains a
fallback. This change does not alter retrieval, evaluation, or UI semantics.

The application already owns these PostgreSQL tables:

- `app_runs`
- `app_run_events(run_id, event_index, event_json, created_at)`

The UI server reads them but performs no DDL and never trims or rewrites their
rows. `SQLiteEventJournal` is a separate, update-safe journal for future local
or external progress producers. Its file location must be supplied by the
caller; importing the module creates no files.

## Contract

Every transported event contains exactly:

`event_id`, `sequence`, `timestamp`, `run_id`, `parent_run_id`, `workflow`,
`phase`, `entity_type`, `entity_id`, `status`, `completed`, `total`,
`provisional`, `payload`, and `artifact_ref`.

Validation is fail closed: UUID, positive sequence, timezone-aware timestamp,
required text fields, booleans, totals, payload type, missing fields, and
unknown fields are checked before an event is accepted.

Persisted legacy `event_json` objects are wrapped deterministically at the
transport boundary. Their original object is preserved under `payload`; the
stable envelope `event_id` is derived from `(run_id, event_index)`.

## HTTP API

### Snapshot / polling fallback

```text
GET /local/run-events?run_id=<id>&after=<sequence>&limit=<1..2000>
```

Returns the existing array of rows (`event_index`, `event_json`, `created_at`)
plus an `envelope` field. This is backward-compatible with the current client.

### SSE subscription

```text
GET /local/run-events/subscribe?run_id=<id>&after=<sequence>&limit=<1..2000>
Last-Event-ID: <sequence>
Accept: text/event-stream
```

`Last-Event-ID` takes precedence over `after`. Invalid or negative cursors are
rejected with HTTP 400. The stream first emits a `snapshot` event, then ordered
`run-event` messages with `id: <sequence>`. Reconnect with the last received ID
to replay strictly newer events. The stream sends comments as heartbeats and
closes after the persisted run reaches a terminal state and its tail is empty.

Expected local transport visibility is under one second; the default database
poll interval is 200 ms. This target covers local transport only, not producer,
database, model, retrieval, or shared-stack latency.

## Bounded durable journal

`SQLiteEventJournal(path, max_events_per_run=2000)` provides:

- unique `event_id` idempotency;
- unique ordered `(run_id, sequence)` values;
- bounded per-run retention;
- replay metadata (`oldest_sequence`, `newest_sequence`, `truncated`);
- conflict detection instead of silent overwrite.

The SQLite schema belongs only to this module. Do not point the journal at an
application database or treat it as result authority.

## Progress-JSONL bridge

`ProgressJsonlBridge` accepts one external CLI progress JSON object per line,
requires a non-empty `event`, derives a deterministic UUID from the canonical
line, and publishes to SQLite. Re-reading the same line is idempotent. Invalid
JSON or malformed objects are rejected. The bridge stores the complete source
object in `payload` and only references immutable output through `artifact_ref`.

This lane intentionally does not modify the frozen benchmark evaluator. A
future launcher can consume its `--progress-jsonl` stdout and feed each line to
the bridge without changing evaluator semantics.

## Verification

```bash
pytest -q tests/unit/test_event_backbone.py tests/unit/test_ui_server.py
python -m py_compile apps/ui/event_envelope.py apps/ui/durable_journal.py \
  apps/ui/run_registry.py apps/ui/sse_replay.py apps/ui/server.py
git diff --check
```

The focused HTTP test starts a loopback `ThreadingHTTPServer`, exercises
snapshot and SSE replay, and measures transport elapsed time without touching
live services, PostgreSQL, Qdrant, or production artifacts.
