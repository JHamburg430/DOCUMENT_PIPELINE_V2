from __future__ import annotations

import fcntl
import hashlib
import os
import json
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from json import dumps
from http.client import RemoteDisconnected
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import timeout as SocketTimeout
from threading import Lock
from threading import Thread
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import parse_qs, urlparse

import psycopg
from psycopg.rows import dict_row

try:
    from .durable_journal import ProgressJsonlBridge, SQLiteEventJournal
    from .run_registry import RunRegistry
    from .sse_replay import (
        ReplayRequestError,
        encode_heartbeat,
        encode_snapshot,
        encode_sse,
        envelope_from_persisted_row,
        parse_replay_cursor,
    )
except ImportError:  # pragma: no cover - direct ``python apps/ui/server.py`` execution
    from durable_journal import ProgressJsonlBridge, SQLiteEventJournal
    from run_registry import RunRegistry
    from sse_replay import (
        ReplayRequestError,
        encode_heartbeat,
        encode_snapshot,
        encode_sse,
        envelope_from_persisted_row,
        parse_replay_cursor,
    )


API_BASE = os.getenv("MANUALS_RAG_API_BASE", "http://api:8600").rstrip("/")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://manuals:manuals@postgres:5432/manuals_rag")
STATIC_DIR = Path(__file__).resolve().parent
MANUALS_ROOT = STATIC_DIR.parents[1]
TEST_REPORTS_DIR = MANUALS_ROOT / "test_reports"
DEFAULT_CORPUS_ID = os.getenv("MANUALS_RAG_DEFAULT_CORPUS", "manuals_vendor_keyence")
MATRIX_JOB_TIMEOUT_SECONDS = int(os.getenv("MATRIX_JOB_TIMEOUT_SECONDS", "7200"))
DEFAULT_QUESTION_GENERATION_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_EVAL_QUESTION_TIMEOUT_SECONDS", "180"))

MATRIX_JOBS: dict[str, dict] = {}
MATRIX_PROCESSES: dict[str, subprocess.Popen] = {}
MATRIX_JOBS_LOCK = Lock()
MATRIX_JOBS_LOADED = False
MATRIX_JOB_EVENT_TAIL_LIMIT = 200
MATRIX_JOB_PERSISTED_EVENT_TAIL_LIMIT = 50
PROXY_RETRYABLE_METHODS = {"GET", "HEAD"}
PROXY_RETRYABLE_ERRORS = (ConnectionError, RemoteDisconnected, SocketTimeout, URLError, TimeoutError)
ANSWER_STAGE_KEYS = {"relevance", "summaries", "generation", "answer_docs", "citations", "terms", "answer"}
MATRIX_STAGE_KEYS = [
    "query_classify",
    "filters",
    "dense",
    "sparse",
    "special",
    "fuse",
    "rerank",
    "assemble",
    "metadata",
    "retrieval",
    "relevance",
    "summaries",
    "generation",
    "answer_docs",
    "citations",
    "terms",
    "answer",
]
DEBUG_STEP_TO_MATRIX_KEY = {
    "classify_query": "query_classify",
    "build_filters": "filters",
    "run_dense_search": "dense",
    "run_sparse_search": "sparse",
    "run_special_search": "special",
    "fuse_results": "fuse",
    "rerank_results": "rerank",
    "assemble_context": "assemble",
    "judge_answer_inputs": "relevance",
    "summarize_answer_inputs": "summaries",
    "generate_answer": "generation",
}
RESULT_STAGE_STEPS = {
    "run_dense_search",
    "run_sparse_search",
    "run_special_search",
    "fuse_results",
    "rerank_results",
    "assemble_context",
}
AGENT_MATRIX_REPORT = TEST_REPORTS_DIR / "agent_evaluation_matrix_latest.json"
AGENT_MATRIX_JOBS: dict[str, dict] = {}
AGENT_MATRIX_LOCK = Lock()
AGENT_LIVE_JOBS: dict[str, dict] = {}
AGENT_LIVE_LOCK = Lock()
AGENT_LIVE_LATEST_ID: str | None = None
AGENT_LIVE_JOB_LIMIT = 10
UI_AUTH_TOKEN = os.getenv("MANUALS_RAG_AUTH_TOKEN", "admin-token")
RUN_EVENT_REGISTRY = RunRegistry()
RUN_EVENT_POLL_SECONDS = max(0.05, float(os.getenv("MANUALS_RAG_EVENT_POLL_SECONDS", "0.2")))
RUN_EVENT_HEARTBEAT_SECONDS = max(1.0, float(os.getenv("MANUALS_RAG_EVENT_HEARTBEAT_SECONDS", "15")))
RUN_EVENT_STREAM_MAX_SECONDS = max(1.0, float(os.getenv("MANUALS_RAG_EVENT_STREAM_MAX_SECONDS", "3600")))
UI_EVENT_JOURNAL_PATH = os.getenv("MANUALS_RAG_UI_EVENT_JOURNAL")
UI_EVENT_JOURNAL_MAX_EVENTS = max(100, int(os.getenv("MANUALS_RAG_UI_EVENT_MAX_EVENTS", "4000")))
UI_EVENT_JOURNAL: SQLiteEventJournal | None = None
UI_EVENT_JOURNAL_LOCK = Lock()
EXTERNAL_EVAL_RUN_PREFIX = "external-eval-"
LOCAL_EVENT_RUN_PREFIXES = ("matrix-", "agent-run-", "agent-matrix-", EXTERNAL_EVAL_RUN_PREFIX)
EXTERNAL_EVAL_OBSERVATIONS: dict[str, tuple[int, str]] = {}
EXTERNAL_EVAL_FILE_COUNTS: dict[str, tuple[int, int, int]] = {}
EXTERNAL_EVAL_LOCK = Lock()
TERMINAL_RUN_STATUSES = {"completed", "succeeded", "failed", "cancelled", "canceled", "stopped"}


class MatrixJobCancelled(RuntimeError):
    pass


class MatrixAnswerFailure(RuntimeError):
    pass


def _ui_event_journal() -> SQLiteEventJournal:
    """Open the private UI journal lazily; importing this module creates no files."""
    global UI_EVENT_JOURNAL
    with UI_EVENT_JOURNAL_LOCK:
        if UI_EVENT_JOURNAL is None:
            UI_EVENT_JOURNAL = SQLiteEventJournal(
                Path(UI_EVENT_JOURNAL_PATH) if UI_EVENT_JOURNAL_PATH else TEST_REPORTS_DIR / ".ui_run_events.sqlite3",
                max_events_per_run=UI_EVENT_JOURNAL_MAX_EVENTS,
            )
        return UI_EVENT_JOURNAL


def _is_component_event_run(run_id: str) -> bool:
    return run_id.startswith(LOCAL_EVENT_RUN_PREFIXES)


def _component_run_snapshot(run_id: str) -> dict | None:
    if run_id.startswith(EXTERNAL_EVAL_RUN_PREFIX):
        return _external_eval_run(run_id)
    if run_id.startswith("agent-matrix-"):
        with AGENT_MATRIX_LOCK:
            job = deepcopy(AGENT_MATRIX_JOBS.get(run_id))
        return job
    if run_id.startswith("agent-run-"):
        with AGENT_LIVE_LOCK:
            job = deepcopy(AGENT_LIVE_JOBS.get(run_id))
        return job
    if run_id.startswith("matrix-"):
        _load_question_matrix_jobs_if_needed()
        with MATRIX_JOBS_LOCK:
            job = deepcopy(MATRIX_JOBS.get(run_id))
        return job
    return None


def _component_workflow(run_id: str, snapshot: dict | None = None) -> str:
    if run_id.startswith(EXTERNAL_EVAL_RUN_PREFIX):
        return str((snapshot or {}).get("workflow") or "evaluation")
    if run_id.startswith("agent-matrix-"):
        return "agent_matrix"
    if run_id.startswith("agent-run-"):
        return "agent_chat" if (snapshot or {}).get("surface") == "chat" else "agent_lab"
    return "question_matrix"


def _publish_component_event(
    run_id: str,
    *,
    workflow: str,
    phase: str,
    status: str,
    entity_type: str = "run",
    entity_id: str | None = None,
    provisional: bool = True,
    payload: dict | None = None,
    total: int | None = None,
    artifact_ref: str | None = None,
) -> None:
    completed = status.lower() in TERMINAL_RUN_STATUSES
    event = _ui_event_journal().publish(
        run_id=run_id,
        workflow=workflow,
        phase=phase,
        entity_type=entity_type,
        entity_id=entity_id or run_id,
        status=status,
        completed=completed,
        total=total,
        provisional=provisional and not completed,
        payload=payload or {},
        artifact_ref=artifact_ref,
    )
    RUN_EVENT_REGISTRY.observe(
        run_id,
        workflow=workflow,
        status=status,
        last_sequence=event.sequence,
        metadata=_component_run_snapshot(run_id) or {},
    )


def _try_publish_component_event(run_id: str, **kwargs: object) -> None:
    """Keep observability failures from changing evaluator outcomes."""
    try:
        _publish_component_event(run_id, **kwargs)
    except Exception as error:  # pragma: no cover - exercised through degraded-host behavior
        print(
            f"UI event journal publish failed for {run_id}: {error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )


def _component_event_rows(run_id: str, *, after: int, limit: int) -> list[dict]:
    page = _ui_event_journal().replay(run_id, after=after, limit=limit)
    return [
        {
            "event_index": event.sequence,
            "event_json": {"event": event.phase, **deepcopy(event.payload)},
            "created_at": event.timestamp,
            "envelope": event.to_dict(),
        }
        for event in page.events
    ]


class ManualsRagUiHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/local/agent-chat/current":
            self._local_current_agent_chat_job()
            return
        if parsed.path.startswith("/local/agent-runs/jobs/"):
            self._local_agent_live_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path == "/local/agent-runs/current":
            self._local_current_agent_live_job()
            return
        if parsed.path.startswith("/local/agent-matrix/jobs/"):
            self._local_agent_matrix_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path == "/local/agent-matrix":
            self._local_agent_matrix()
            return
        if parsed.path.startswith("/local/question-matrix/jobs/"):
            self._local_question_matrix_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path == "/local/question-matrix":
            self._local_question_matrix()
            return
        if parsed.path.startswith("/local/run-events"):
            self._local_run_events()
            return
        if self.path.startswith("/api/"):
            self._proxy()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self.path.startswith("/api/"):
            self._proxy()
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/local/agent-chat/run":
            self._start_local_agent_chat_run()
            return
        if parsed.path == "/local/agent-runs/run":
            self._start_local_agent_live_run()
            return
        if parsed.path == "/local/agent-matrix/run":
            self._start_local_agent_matrix_run()
            return
        if parsed.path == "/local/question-matrix/run":
            self._start_local_question_matrix_run()
            return
        if parsed.path == "/local/question-matrix/generate":
            self._start_local_question_generation()
            return
        if parsed.path == "/local/question-matrix/clear":
            self._clear_local_question_matrix_results()
            return
        if parsed.path == "/local/question-matrix/questions/clear":
            self._clear_local_question_matrix_questions()
            return
        if parsed.path == "/local/question-matrix/bank/reset":
            self._reset_local_question_matrix_bank()
            return
        if parsed.path.startswith("/local/question-matrix/jobs/") and parsed.path.endswith("/stop"):
            self._stop_local_question_matrix_job(parsed.path.split("/")[-2])
            return
        if self.path.startswith("/api/"):
            self._proxy()
            return
        self.send_error(404, "Not found")

    def _proxy(self) -> None:
        upstream_path = self.path.removeprefix("/api")
        target = f"{API_BASE}{upstream_path}"
        body = None
        if self.command in {"POST", "PUT", "PATCH"}:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else None

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        request = Request(target, data=body, headers=headers, method=self.command)
        attempts = 2 if self.command in PROXY_RETRYABLE_METHODS else 1
        try:
            for attempt in range(attempts):
                try:
                    response = urlopen(request, timeout=900)
                    break
                except PROXY_RETRYABLE_ERRORS:
                    if attempt + 1 >= attempts:
                        raise
                    time.sleep(0.25)
            with response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in {"transfer-encoding", "connection", "content-encoding"}:
                        self.send_header(key, value)
                self.end_headers()
                if self.command == "HEAD":
                    return
                if self._is_streaming_response(response):
                    for line in response:
                        if not self._write(line):
                            break
                    return
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    if not self._write(chunk):
                        break
        except HTTPError as error:
            self.send_response(error.code)
            for key, value in error.headers.items():
                if key.lower() not in {"transfer-encoding", "connection", "content-encoding"}:
                    self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self._write(error.read())
        except (ConnectionError, SocketTimeout, URLError, TimeoutError, OSError) as error:
            if self.command == "HEAD":
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                return
            payload = dumps({
                "detail": f"Manuals RAG API proxy failed: {error.__class__.__name__}: {error}"
            }).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self._write(payload)
            except OSError:
                pass

    def _local_run_events(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        run_id = (query.get("run_id") or [""])[0]
        if not run_id:
            self.send_error(400, "run_id is required")
            return
        try:
            after = parse_replay_cursor(
                self.headers.get("Last-Event-ID"),
                (query.get("after") or [None])[0],
            )
            limit = max(1, min(int((query.get("limit") or ["1000"])[0]), 2000))
        except (ReplayRequestError, ValueError) as error:
            self.send_error(400, str(error))
            return
        if parsed.path == "/local/run-events/subscribe":
            self._local_run_event_stream(run_id, after=after, limit=limit)
            return
        if parsed.path != "/local/run-events":
            self.send_error(404, "Not found")
            return
        try:
            if _is_component_event_run(run_id):
                if run_id.startswith(EXTERNAL_EVAL_RUN_PREFIX):
                    _sync_external_eval_event(run_id)
                snapshot = _component_run_snapshot(run_id)
                rows = _component_event_rows(run_id, after=after, limit=limit)
                workflow = _component_workflow(run_id, snapshot)
            else:
                rows = self._query_run_events(run_id, after=after, limit=limit)
                snapshot = self._query_run_snapshot(run_id)
                workflow = str((snapshot or {}).get("run_type") or "application")
                for row in rows:
                    row["envelope"] = envelope_from_persisted_row(run_id, row, workflow=workflow).to_dict()
            last_sequence = int(rows[-1]["event_index"]) if rows else after
            RUN_EVENT_REGISTRY.observe(
                run_id,
                workflow=workflow,
                status=str((snapshot or {}).get("status") or "unknown"),
                last_sequence=last_sequence,
                metadata=snapshot or {},
            )
            payload = json.dumps(rows, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Event-Replay-After", str(after))
            self.send_header("X-Event-Replay-Cursor", str(last_sequence))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Local run event lookup failed: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _query_run_events(self, run_id: str, *, after: int, limit: int) -> list[dict]:
        """Read the inspected application-owned event schema without mutating it."""
        with psycopg.connect(POSTGRES_DSN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select event_index, event_json, created_at
                    from app_run_events
                    where run_id = %s
                      and event_index > %s
                      and coalesce(event_json #>> '{query_event,event}', '') <> 'llm_token'
                    order by event_index asc
                    limit %s
                    """,
                    (run_id, after, limit),
                )
                return [dict(row) for row in cur.fetchall()]

    def _query_run_snapshot(self, run_id: str) -> dict | None:
        with psycopg.connect(POSTGRES_DSN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, run_type, status, progress_json, error, created_at, updated_at
                    from app_runs
                    where id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def _local_run_event_stream(self, run_id: str, *, after: int, limit: int) -> None:
        try:
            if run_id.startswith(EXTERNAL_EVAL_RUN_PREFIX):
                _sync_external_eval_event(run_id)
            snapshot = (
                _component_run_snapshot(run_id)
                if _is_component_event_run(run_id)
                else self._query_run_snapshot(run_id)
            )
        except Exception as error:
            self.send_error(500, f"Run snapshot lookup failed: {error.__class__.__name__}: {error}")
            return
        if snapshot is None:
            self.send_error(404, "Run not found")
            return

        workflow = (
            _component_workflow(run_id, snapshot)
            if _is_component_event_run(run_id)
            else str(snapshot.get("run_type") or "application")
        )
        registry_snapshot = RUN_EVENT_REGISTRY.observe(
            run_id,
            workflow=workflow,
            status=str(snapshot.get("status") or "unknown"),
            last_sequence=after,
            metadata=snapshot,
        )
        RUN_EVENT_REGISTRY.subscriber_opened(run_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Event-Replay-After", str(after))
        self.end_headers()

        cursor = after
        started = time.monotonic()
        last_write = started
        try:
            if not self._write(encode_snapshot(registry_snapshot.to_dict())):
                return
            while time.monotonic() - started < RUN_EVENT_STREAM_MAX_SECONDS:
                if run_id.startswith(EXTERNAL_EVAL_RUN_PREFIX):
                    _sync_external_eval_event(run_id)
                if _is_component_event_run(run_id):
                    envelopes = list(_ui_event_journal().replay(run_id, after=cursor, limit=limit).events)
                else:
                    rows = self._query_run_events(run_id, after=cursor, limit=limit)
                    envelopes = [envelope_from_persisted_row(run_id, row, workflow=workflow) for row in rows]
                for envelope in envelopes:
                    if not self._write(encode_sse(envelope)):
                        return
                    cursor = envelope.sequence
                    last_write = time.monotonic()
                snapshot = (
                    _component_run_snapshot(run_id)
                    if _is_component_event_run(run_id)
                    else self._query_run_snapshot(run_id)
                ) or snapshot
                observed = RUN_EVENT_REGISTRY.observe(
                    run_id,
                    workflow=workflow,
                    status=str(snapshot.get("status") or "unknown"),
                    last_sequence=cursor,
                    metadata=snapshot,
                )
                if observed.terminal and not envelopes:
                    return
                if time.monotonic() - last_write >= RUN_EVENT_HEARTBEAT_SECONDS:
                    if not self._write(encode_heartbeat()):
                        return
                    last_write = time.monotonic()
                time.sleep(RUN_EVENT_POLL_SECONDS)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            error_payload = json.dumps(
                {"detail": f"Run event stream failed: {error.__class__.__name__}: {error}"},
                separators=(",", ":"),
            )
            self._write(f"event: error\ndata: {error_payload}\n\n".encode("utf-8"))
        finally:
            RUN_EVENT_REGISTRY.subscriber_closed(run_id)
            self.close_connection = True

    def _local_question_matrix(self) -> None:
        try:
            payload = json.dumps(_build_question_matrix(), default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Question matrix lookup failed: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _local_agent_matrix(self) -> None:
        try:
            payload = json.dumps(_build_agent_matrix(), default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Agent matrix lookup failed: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _local_current_agent_live_job(self) -> None:
        with AGENT_LIVE_LOCK:
            job = next(
                (
                    deepcopy(item)
                    for item in reversed(list(AGENT_LIVE_JOBS.values()))
                    if item.get("surface", "lab") == "lab"
                ),
                {},
            )
        payload = json.dumps({"job": job or None}, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _local_current_agent_chat_job(self) -> None:
        with AGENT_LIVE_LOCK:
            job = next(
                (deepcopy(item) for item in reversed(list(AGENT_LIVE_JOBS.values())) if item.get("surface") == "chat"),
                {},
            )
        payload = json.dumps({"job": job or None}, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _local_agent_live_job(self, job_id: str) -> None:
        with AGENT_LIVE_LOCK:
            job = deepcopy(AGENT_LIVE_JOBS.get(job_id) or {})
        if not job:
            self.send_error(404, "Agent run job not found")
            return
        payload = json.dumps(job, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _start_local_agent_live_run(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else b"{}"
            job = _start_agent_live_job(json.loads(body.decode("utf-8") or "{}"))
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Agent run failed to start: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _start_local_agent_chat_run(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else b"{}"
            job = _start_agent_live_job(json.loads(body.decode("utf-8") or "{}"), surface="chat")
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Agent chat failed to start: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _local_agent_matrix_job(self, job_id: str) -> None:
        if job_id.startswith(EXTERNAL_EVAL_RUN_PREFIX):
            job = dict(_external_agent_matrix_run(job_id) or {})
        else:
            with AGENT_MATRIX_LOCK:
                job = dict(AGENT_MATRIX_JOBS.get(job_id) or {})
        if not job:
            self.send_error(404, "Agent matrix job not found")
            return
        payload = json.dumps(job, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _start_local_agent_matrix_run(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else b"{}"
            job = _start_agent_matrix_job(json.loads(body.decode("utf-8") or "{}"))
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Agent matrix run failed to start: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _local_question_matrix_job(self, job_id: str) -> None:
        _load_question_matrix_jobs_if_needed()
        with MATRIX_JOBS_LOCK:
            job = dict(MATRIX_JOBS.get(job_id) or {})
        if not job:
            self.send_error(404, "Job not found")
            return
        payload = json.dumps(job, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _start_local_question_matrix_run(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else b"{}"
            request_payload = json.loads(body.decode("utf-8") or "{}")
            job = _start_question_matrix_job(request_payload)
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Question matrix run failed to start: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _start_local_question_generation(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else b"{}"
            request_payload = json.loads(body.decode("utf-8") or "{}")
            job = _start_question_generation_job(request_payload)
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except Exception as error:
            payload = dumps({"detail": f"Question generation failed to start: {error.__class__.__name__}: {error}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _stop_local_question_matrix_job(self, job_id: str) -> None:
        try:
            job = _stop_question_matrix_job(job_id)
            payload = json.dumps(job, default=str).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except KeyError:
            self.send_error(404, "Job not found")
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _clear_local_question_matrix_results(self) -> None:
        try:
            result = _clear_question_matrix_results()
            payload = json.dumps(result, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _clear_local_question_matrix_questions(self) -> None:
        try:
            result = _clear_question_matrix_generated_questions()
            payload = json.dumps(result, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _reset_local_question_matrix_bank(self) -> None:
        try:
            result = _reset_question_matrix_bank()
            payload = json.dumps(result, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)
        except ValueError as error:
            payload = dumps({"detail": str(error)}).encode("utf-8")
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write(payload)

    def _write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _is_streaming_response(self, response) -> bool:
        content_type = (response.headers.get("Content-Type") or "").lower()
        return "application/x-ndjson" in content_type or self.path.startswith("/api/eval/end-to-end-stream")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _external_agent_matrix_report(snapshot: dict) -> dict:
    merged_items: dict[str, dict] = {}
    summary: dict = {}
    category_summary: dict = {}
    for relative_path in snapshot.get("report_paths") or []:
        path = MANUALS_ROOT / str(relative_path)
        report = _agent_matrix_artifact_json(path) or {}
        for item in report.get("items") or []:
            if isinstance(item, dict) and item.get("case_id"):
                merged_items[str(item["case_id"])] = deepcopy(item)
        if report.get("complete"):
            summary = deepcopy(report.get("summary") or summary)
            category_summary = deepcopy(report.get("category_summary") or category_summary)
    provisional_ids = set(snapshot.get("provisional_case_ids") or [])
    if snapshot.get("status") == "running":
        for case_id in provisional_ids:
            item = merged_items.get(str(case_id)) or {}
            for backend in ("langgraph", "llamaindex"):
                cells = (((item.get(backend) or {}).get("agent_evaluation") or {}).get("cells") or {})
                for cell in cells.values():
                    if not isinstance(cell, dict) or cell.get("status") not in {"pass", "fail"}:
                        continue
                    cell["final_status"] = cell["status"]
                    cell["status"] = "provisional"
                    cell["label"] = "LIVE"
                    cell["detail"] = f"{cell.get('detail') or cell['final_status']} · provisional until terminal reconciliation"
    category_counts: dict[str, int] = {}
    for item in merged_items.values():
        category = str(item.get("agent_case_category") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    for backend in ("baseline", "langgraph", "llamaindex"):
        backend_items = [item.get(backend) for item in merged_items.values() if isinstance(item.get(backend), dict)]
        backend_summary = dict(summary.get(backend) or {})
        backend_summary["cases"] = len(backend_items)
        backend_summary["agent_matrix_passed"] = sum(
            1 for item in backend_items if (item.get("agent_evaluation") or {}).get("passed")
        )
        summary[backend] = backend_summary
    return {
        "dataset": snapshot["dataset_path"],
        "items": list(merged_items.values()),
        "summary": summary,
        "category_summary": category_summary,
        "category_counts": category_counts,
        "dataset_sha256": None,
    }


def _agent_matrix_ui_result(item: dict | None) -> dict | None:
    """Project a persisted matrix item down to fields rendered by the browser.

    Full evaluation artifacts include retrieval results and stage snapshots that can
    make the 200-row response exceed 100 MB.  The matrix only needs terminal cells,
    the generated answer, elapsed time, and the compact trace shown in row details.
    """
    if not isinstance(item, dict):
        return None
    projected = {
        key: deepcopy(item[key])
        for key in ("case_id", "query", "retrieval_task", "agent_case_category")
        if key in item
    }
    for backend in ("langgraph", "llamaindex"):
        result = item.get(backend)
        if not isinstance(result, dict):
            continue
        projected[backend] = {
            key: deepcopy(result[key])
            for key in ("agent_evaluation", "answer", "elapsed_ms", "trace")
            if key in result
        }
    return projected


def _build_agent_matrix() -> dict:
    external_snapshot = _external_agent_matrix_run()
    report = (
        _external_agent_matrix_report(external_snapshot)
        if external_snapshot is not None
        else (_read_json(AGENT_MATRIX_REPORT) if AGENT_MATRIX_REPORT.exists() else {})
    )
    dataset_value = str(report.get("dataset") or "tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl")
    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = MANUALS_ROOT / dataset_path
    cases = _read_jsonl(dataset_path) if dataset_path.exists() else []
    report_items = {str(item.get("case_id") or ""): item for item in report.get("items") or []}
    with AGENT_MATRIX_LOCK:
        active_job = next(
            (deepcopy(job) for job in AGENT_MATRIX_JOBS.values() if job.get("status") in {"queued", "running"}),
            None,
        )
    if active_job is None and external_snapshot is not None and external_snapshot.get("status") == "running":
        active_job = deepcopy(external_snapshot)
    rows = []
    for number, record in enumerate(cases, start=1):
        case = record.get("case") if isinstance(record.get("case"), dict) else record
        case_id = str(case.get("case_id") or f"question-{number}")
        report_item = report_items.get(case_id)
        report_graph = (
            report_item.get("expected_evidence_graph")
            if isinstance(report_item, dict) and isinstance(report_item.get("expected_evidence_graph"), dict)
            else {}
        )
        rows.append(
            {
                "number": number,
                "case_id": case_id,
                "question": case.get("query") or "",
                "retrieval_task": case.get("retrieval_task") or "",
                "agent_case_category": (
                    (case.get("expected_evidence_graph") or {}).get("category")
                    or (case.get("source_metadata") or {}).get("agent_case_category")
                    or (report_item or {}).get("agent_case_category")
                    or report_graph.get("category")
                    or "unknown"
                ),
                "expected_graph_mode": (
                    (case.get("expected_evidence_graph") or {}).get("mode")
                    or report_graph.get("mode")
                    or ""
                ),
                "expected_evidence_count": len(case.get("expected_evidence") or []),
                "expected_document_count": len(
                    {
                        str(case.get("source_document_id") or ""),
                        *[
                            str(item.get("source_document_id") or "")
                            for item in case.get("expected_evidence") or []
                            if isinstance(item, dict)
                        ],
                    }
                    - {""}
                ),
                "result": _agent_matrix_ui_result(report_item),
            }
        )
    return {
        "schema": "manuals-rag-agent-evaluation-matrix-v2",
        "dataset": str(dataset_path.relative_to(MANUALS_ROOT)) if dataset_path.is_relative_to(MANUALS_ROOT) else str(dataset_path),
        "generated_at": (
            external_snapshot.get("updated_at")
            if external_snapshot is not None
            else (AGENT_MATRIX_REPORT.stat().st_mtime if AGENT_MATRIX_REPORT.exists() else None)
        ),
        "summary": report.get("summary") or {},
        "category_summary": report.get("category_summary") or {},
        "category_counts": report.get("category_counts") or {},
        "dataset_sha256": report.get("dataset_sha256"),
        "active_job": active_job,
        "rows": rows,
        "layers": [
            "tool_selection",
            "candidate_recall",
            "document_retention",
            "hop_dependencies",
            "evidence_sufficiency",
            "grounded_answer",
            "latency_token_cost",
        ],
    }


def _start_agent_matrix_job(payload: dict) -> dict:
    dataset_value = str(payload.get("dataset") or "tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl")
    dataset_path = (MANUALS_ROOT / dataset_value).resolve()
    if not dataset_path.is_relative_to(MANUALS_ROOT) or not dataset_path.exists():
        raise ValueError("Agent matrix dataset must be an existing file inside the repository.")
    limit = _as_positive_int(payload.get("limit"), 10, maximum=1000)
    max_hops = _as_positive_int(payload.get("max_hops"), 6, maximum=8)
    corpus_id = str(payload.get("corpus_id") or DEFAULT_CORPUS_ID)
    no_llm = bool(payload.get("no_llm"))
    with AGENT_MATRIX_LOCK:
        active = next((job for job in AGENT_MATRIX_JOBS.values() if job.get("status") in {"queued", "running"}), None)
        if active:
            raise ValueError(f"Agent matrix job {active['id']} is already running.")
        job_id = f"agent-matrix-{uuid.uuid4().hex[:12]}"
        job = {
            "id": job_id,
            "status": "queued",
            "dataset": str(dataset_path.relative_to(MANUALS_ROOT)),
            "limit": limit,
            "max_hops": max_hops,
            "corpus_id": corpus_id,
            "no_llm": no_llm,
            "completed_questions": 0,
            "current_case_id": None,
            "live_results": {},
            "error": None,
            "started_at": None,
            "completed_at": None,
        }
        AGENT_MATRIX_JOBS[job_id] = job
    _try_publish_component_event(
        job_id,
        workflow="agent_matrix",
        phase="job_queued",
        status="queued",
        payload={"dataset": job["dataset"], "limit": limit, "max_hops": max_hops},
        total=limit,
    )
    Thread(target=_run_agent_matrix_job, args=(job_id, dataset_path), daemon=True).start()
    return dict(job)


def _start_agent_live_job(payload: dict, *, surface: str = "lab") -> dict:
    global AGENT_LIVE_LATEST_ID
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("Agent query is required.")
    requested_backends = payload.get("backends") or ["langgraph_agent", "llamaindex_agent"]
    if isinstance(requested_backends, str):
        requested_backends = [requested_backends]
    allowed_backends = {"langgraph_agent", "llamaindex_agent"}
    backends = [str(item) for item in requested_backends if str(item) in allowed_backends]
    if not backends or len(backends) != len(requested_backends):
        raise ValueError("Agent backends must be langgraph_agent and/or llamaindex_agent.")
    corpus_ids = payload.get("corpus_ids") or [DEFAULT_CORPUS_ID]
    if isinstance(corpus_ids, str):
        corpus_ids = [item.strip() for item in corpus_ids.split(",") if item.strip()]
    corpus_ids = [str(item).strip() for item in corpus_ids if str(item).strip()]
    if not corpus_ids:
        corpus_ids = [DEFAULT_CORPUS_ID]
    max_hops = _as_positive_int(payload.get("max_retrieval_hops"), 6, maximum=8)
    with AGENT_LIVE_LOCK:
        active = next((job for job in AGENT_LIVE_JOBS.values() if job.get("status") in {"queued", "running"}), None)
        if active:
            active_surface = "Agent page" if active.get("surface") == "chat" else "Agent Lab"
            raise ValueError(f"Agent run {active['id']} is already active in {active_surface}.")
        job_id = f"agent-run-{uuid.uuid4().hex[:12]}"
        started_epoch = time.time()
        job = {
            "id": job_id,
            "status": "queued",
            "surface": surface,
            "query": query,
            "corpus_ids": corpus_ids,
            "backends": backends,
            "max_retrieval_hops": max_hops,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_epoch)),
            "started_at_epoch": started_epoch,
            "completed_at": None,
            "completed_at_epoch": None,
            "runs": {
                backend: {"backend": backend, "status": "queued", "events": [], "error": None}
                for backend in backends
            },
            "error": None,
        }
        AGENT_LIVE_JOBS[job_id] = job
        AGENT_LIVE_LATEST_ID = job_id
        while len(AGENT_LIVE_JOBS) > AGENT_LIVE_JOB_LIMIT:
            oldest = next(iter(AGENT_LIVE_JOBS))
            if oldest == job_id:
                break
            AGENT_LIVE_JOBS.pop(oldest, None)
        snapshot = deepcopy(job)
    workflow = "agent_chat" if surface == "chat" else "agent_lab"
    _try_publish_component_event(
        job_id,
        workflow=workflow,
        phase="job_queued",
        status="queued",
        payload={"surface": surface, "backends": backends, "query": query},
        total=len(backends),
    )
    Thread(target=_run_agent_live_job, args=(job_id,), daemon=True).start()
    return snapshot


def _run_agent_live_backend(job_id: str, backend: str, request_payload: dict) -> None:
    with AGENT_LIVE_LOCK:
        run = AGENT_LIVE_JOBS[job_id]["runs"][backend]
        run["status"] = "running"
        surface = AGENT_LIVE_JOBS[job_id].get("surface", "lab")
    workflow = "agent_chat" if surface == "chat" else "agent_lab"
    _try_publish_component_event(
        job_id,
        workflow=workflow,
        phase="backend_started",
        status="running",
        entity_type="agent_backend",
        entity_id=backend,
        payload={"backend": backend},
    )
    body = json.dumps({**request_payload, "retrieval_orchestrator": backend}).encode("utf-8")
    request = Request(
        f"{API_BASE}/query/stream",
        data=body,
        headers={
            "Authorization": f"Bearer {UI_AUTH_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=900) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event["received_at_epoch"] = time.time()
                with AGENT_LIVE_LOCK:
                    run = AGENT_LIVE_JOBS[job_id]["runs"][backend]
                    run["events"].append(event)
                    if event.get("event") == "run_completed":
                        run["status"] = "completed"
                    elif event.get("event") == "run_failed":
                        run["status"] = "failed"
                        run["error"] = event.get("error")
                _try_publish_component_event(
                    job_id,
                    workflow=workflow,
                    phase=str(event.get("event") or "agent_event"),
                    status="running",
                    entity_type="agent_backend",
                    entity_id=backend,
                    provisional=True,
                    payload={"backend": backend, "source": deepcopy(event)},
                )
        failure_event = None
        with AGENT_LIVE_LOCK:
            run = AGENT_LIVE_JOBS[job_id]["runs"][backend]
            if run["status"] == "running":
                run["status"] = "failed"
                run["error"] = "Agent stream ended without a completion event."
                run["events"].append({
                    "event": "run_failed",
                    "error": run["error"],
                    "received_at_epoch": time.time(),
                })
                failure_event = deepcopy(run["events"][-1])
        if failure_event:
            _try_publish_component_event(
                job_id,
                workflow=workflow,
                phase="run_failed",
                status="running",
                entity_type="agent_backend",
                entity_id=backend,
                provisional=True,
                payload={"backend": backend, "source": failure_event},
            )
    except Exception as error:
        failure = f"{error.__class__.__name__}: {error}"
        with AGENT_LIVE_LOCK:
            run = AGENT_LIVE_JOBS[job_id]["runs"][backend]
            run["status"] = "failed"
            run["error"] = failure
            run["events"].append({"event": "run_failed", "error": failure, "received_at_epoch": time.time()})
            failure_event = deepcopy(run["events"][-1])
        _try_publish_component_event(
            job_id,
            workflow=workflow,
            phase="run_failed",
            status="running",
            entity_type="agent_backend",
            entity_id=backend,
            provisional=True,
            payload={"backend": backend, "source": failure_event},
        )


def _run_agent_live_job(job_id: str) -> None:
    with AGENT_LIVE_LOCK:
        job = AGENT_LIVE_JOBS[job_id]
        job["status"] = "running"
        request_payload = {
            "query": job["query"],
            "corpus_ids": job["corpus_ids"],
            "filters": {},
            "response_mode": "answer_with_citations",
            "max_retrieval_hops": job["max_retrieval_hops"],
        }
        backends = list(job["backends"])
        workflow = "agent_chat" if job.get("surface") == "chat" else "agent_lab"
    _try_publish_component_event(
        job_id,
        workflow=workflow,
        phase="job_started",
        status="running",
        payload={"backends": backends},
        total=len(backends),
    )
    workers = [
        Thread(target=_run_agent_live_backend, args=(job_id, backend, request_payload), daemon=True)
        for backend in backends
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    completed_epoch = time.time()
    with AGENT_LIVE_LOCK:
        job = AGENT_LIVE_JOBS[job_id]
        job["status"] = "completed" if all(
            run.get("status") == "completed" for run in job["runs"].values()
        ) else "failed"
        job["completed_at_epoch"] = completed_epoch
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_epoch))
        failures = [run.get("error") for run in job["runs"].values() if run.get("error")]
        job["error"] = "; ".join(failures) if failures else None
        final_status = job["status"]
    _try_publish_component_event(
        job_id,
        workflow=workflow,
        phase="job_completed" if final_status == "completed" else "job_failed",
        status=final_status,
        provisional=False,
        payload={"backends": backends},
        total=len(backends),
    )


def _run_agent_matrix_job(job_id: str, dataset_path: Path) -> None:
    with AGENT_MATRIX_LOCK:
        job = AGENT_MATRIX_JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        snapshot = dict(job)
    _try_publish_component_event(
        job_id,
        workflow="agent_matrix",
        phase="job_started",
        status="running",
        payload={"dataset": snapshot["dataset"]},
        total=int(snapshot["limit"]),
    )
    try:
        progress_bridge = ProgressJsonlBridge(_ui_event_journal(), run_id=job_id, workflow="agent_matrix")
    except Exception as error:  # journal failure must not change benchmark behavior
        progress_bridge = None
        print(f"Agent matrix progress bridge unavailable: {error}", file=sys.stderr, flush=True)
    cmd = [
        sys.executable,
        str(MANUALS_ROOT / "scripts" / "benchmark" / "compare_agentic_retrieval.py"),
        "--dataset",
        str(dataset_path),
        "--corpus-id",
        str(snapshot["corpus_id"]),
        "--limit",
        str(snapshot["limit"]),
        "--max-hops",
        str(snapshot["max_hops"]),
        "--output",
        str(AGENT_MATRIX_REPORT),
        "--progress-jsonl",
        "--quiet",
    ]
    if snapshot.get("no_llm"):
        cmd.append("--no-llm")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=MANUALS_ROOT,
            env={**os.environ, "API_BASE": API_BASE},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if progress_bridge is not None:
                try:
                    bridged = progress_bridge.consume(line)
                    component_snapshot = _component_run_snapshot(job_id) or {}
                    RUN_EVENT_REGISTRY.observe(
                        job_id,
                        workflow="agent_matrix",
                        status=str(component_snapshot.get("status") or "running"),
                        last_sequence=bridged.sequence,
                        metadata=component_snapshot,
                    )
                except Exception as error:
                    print(f"Agent matrix progress bridge rejected a line: {error}", file=sys.stderr, flush=True)
            if event.get("event") != "agent_case_completed":
                continue
            with AGENT_MATRIX_LOCK:
                job = AGENT_MATRIX_JOBS[job_id]
                job["current_case_id"] = event.get("case_id")
                job["completed_questions"] = int(event.get("question_number") or 0)
                job["live_results"] = {
                    **job.get("live_results", {}),
                    str(event.get("case_id") or ""): {
                        "langgraph": event.get("langgraph"),
                        "llamaindex": event.get("llamaindex"),
                    },
                }
        returncode = process.wait(timeout=MATRIX_JOB_TIMEOUT_SECONDS)
        if returncode != 0:
            raise RuntimeError(f"Agent evaluation exited with status {returncode}")
        with AGENT_MATRIX_LOCK:
            job = AGENT_MATRIX_JOBS[job_id]
            job["status"] = "completed"
            job["current_case_id"] = None
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _try_publish_component_event(
            job_id,
            workflow="agent_matrix",
            phase="job_completed",
            status="completed",
            provisional=False,
            payload={"completed_questions": int(job.get("completed_questions") or 0)},
            total=int(snapshot["limit"]),
            artifact_ref=str(AGENT_MATRIX_REPORT.relative_to(MANUALS_ROOT)),
        )
    except Exception as error:
        with AGENT_MATRIX_LOCK:
            job = AGENT_MATRIX_JOBS[job_id]
            job["status"] = "failed"
            job["error"] = f"{error.__class__.__name__}: {error}"
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            failure = job["error"]
        _try_publish_component_event(
            job_id,
            workflow="agent_matrix",
            phase="job_failed",
            status="failed",
            provisional=False,
            payload={"error": failure},
            total=int(snapshot["limit"]),
        )


def _is_active_dataset(dataset: dict) -> bool:
    status = str(dataset.get("status") or "").lower()
    return "diagnostic" not in status and "superseded" not in status


def _active_question_datasets(question_bank: dict) -> list[dict]:
    datasets = [dict(dataset) for dataset in question_bank.get("datasets") or []]
    active_candidates = [dataset for dataset in datasets if _is_active_dataset(dataset)]
    superseded_paths = {
        str(dataset.get("supersedes"))
        for dataset in active_candidates
        if dataset.get("supersedes")
    }
    return [
        dataset
        for dataset in active_candidates
        if str(dataset.get("path")) not in superseded_paths
    ]


def _case_key(case: dict) -> str:
    return str(case.get("case_id") or case.get("id") or case.get("source_chunk_id") or case.get("query") or "")


def _matrix_row_key(dataset: str, case: dict | str) -> str:
    case_key = case if isinstance(case, str) else _case_key(case)
    return f"{dataset}::{case_key}"


def _result_run_id(path: Path) -> str:
    name = path.name
    return name.removeprefix("retrieval_eval_results_").removesuffix(".jsonl")


def _external_eval_suffix(path: Path) -> str:
    return path.name.removeprefix("retrieval_eval_dataset_").removesuffix(".jsonl")


def _external_eval_artifacts(suffix: str) -> dict[str, Path]:
    return {
        "dataset": TEST_REPORTS_DIR / f"retrieval_eval_dataset_{suffix}.jsonl",
        "partial": TEST_REPORTS_DIR / f"retrieval_eval_results_{suffix}.partial.jsonl",
        "results": TEST_REPORTS_DIR / f"retrieval_eval_results_{suffix}.jsonl",
        "summary": TEST_REPORTS_DIR / f"retrieval_eval_summary_{suffix}.json",
        "manifest": TEST_REPORTS_DIR / f"retrieval_eval_manifest_{suffix}.json",
    }


def _jsonl_record_count(path: Path) -> int:
    stat = path.stat()
    cache_key = str(path)
    cached = EXTERNAL_EVAL_FILE_COUNTS.get(cache_key)
    signature = (stat.st_mtime_ns, stat.st_size)
    if cached and cached[:2] == signature:
        return cached[2]
    with path.open("rb") as handle:
        count = sum(1 for line in handle if line.strip())
    EXTERNAL_EVAL_FILE_COUNTS[cache_key] = (*signature, count)
    return count


def _legacy_external_eval_run(run_id: str | None = None) -> dict | None:
    requested_suffix = (
        str(run_id).removeprefix(EXTERNAL_EVAL_RUN_PREFIX)
        if run_id and str(run_id).startswith(EXTERNAL_EVAL_RUN_PREFIX)
        else None
    )
    candidates = sorted(
        TEST_REPORTS_DIR.glob("retrieval_eval_dataset_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for dataset_path in candidates:
        suffix = _external_eval_suffix(dataset_path)
        if requested_suffix is not None and suffix != requested_suffix:
            continue
        paths = _external_eval_artifacts(suffix)
        if not paths["partial"].exists() and not paths["results"].exists():
            continue
        try:
            result_path = paths["results"] if paths["results"].exists() else paths["partial"]
            total = _jsonl_record_count(dataset_path)
            completed = _jsonl_record_count(result_path)
        except OSError:
            continue
        terminal = paths["results"].exists() and paths["summary"].exists() and completed >= total
        updated_at = max(dataset_path.stat().st_mtime, result_path.stat().st_mtime)
        return {
            "id": f"{EXTERNAL_EVAL_RUN_PREFIX}{suffix}",
            "artifact_run_id": suffix,
            "run_type": "external_retrieval_eval",
            "workflow": "evaluation",
            "status": "completed" if terminal else "running",
            "total": total,
            "completed": min(completed, total),
            "dataset_path": str(dataset_path.relative_to(MANUALS_ROOT)),
            "results_path": str(result_path.relative_to(MANUALS_ROOT)),
            "summary_path": str(paths["summary"].relative_to(MANUALS_ROOT)) if paths["summary"].exists() else None,
            "manifest_path": str(paths["manifest"].relative_to(MANUALS_ROOT)) if paths["manifest"].exists() else None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated_at)),
        }
    return None


AGENT_MATRIX_ARTIFACT_SCHEMA = "agentic-retrieval-matrix-v2"


def _agent_matrix_artifact_json(path: Path) -> dict | None:
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _agent_matrix_lock_is_live(path: Path, run_id: str) -> bool:
    """A persisted lock is active while a writer owns its cross-namespace flock."""
    try:
        with path.open("r+", encoding="utf-8") as handle:
            lock = json.loads(handle.readline())
            pid = int(lock.get("pid") or 0)
            if lock.get("run_id") != run_id or pid <= 0:
                return False
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _agent_matrix_dataset_contract(launch: dict) -> dict | None:
    dataset = launch.get("dataset") if isinstance(launch.get("dataset"), dict) else {}
    run_id = str(launch.get("run_id") or "")
    try:
        dataset_path = Path(str(dataset["path"])).resolve()
        dataset_path.relative_to(MANUALS_ROOT.resolve())
        offset = int(dataset.get("offset") or 0)
        expected = int(dataset.get("limit") or 0)
        ordered_keys = [str(value) for value in dataset.get("ordered_case_keys") or []]
        cases = _read_jsonl(dataset_path)
        digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        launch.get("artifact_schema_version") != AGENT_MATRIX_ARTIFACT_SCHEMA
        or launch.get("state") != "launched"
        or not run_id
        or offset < 0
        or expected <= 0
        or digest != dataset.get("sha256")
    ):
        return None
    full_keys = [f"{digest}:{_case_key(record.get('case') if isinstance(record.get('case'), dict) else record)}" for record in cases]
    sliced_keys = full_keys[offset : offset + expected]
    if not ordered_keys or ordered_keys != sliced_keys:
        return None
    return {
        "run_id": run_id,
        "path": dataset_path,
        "relative_path": str(dataset_path.relative_to(MANUALS_ROOT.resolve())),
        "sha256": digest,
        "cases": cases,
        "full_keys": full_keys,
        "offset": offset,
        "expected": len(ordered_keys),
        "ordered_keys": ordered_keys,
    }


def _validated_agent_matrix_report(path: Path, contract: dict, *, complete: bool) -> dict | None:
    report = _agent_matrix_artifact_json(path)
    if not report or report.get("artifact_schema_version") != AGENT_MATRIX_ARTIFACT_SCHEMA:
        return None
    provenance = report.get("provenance") if isinstance(report.get("provenance"), dict) else {}
    dataset = provenance.get("dataset") if isinstance(provenance.get("dataset"), dict) else {}
    keys = [str(value) for value in dataset.get("ordered_case_keys") or []]
    try:
        offset = int(dataset.get("offset") or 0)
    except (TypeError, ValueError):
        return None
    items = report.get("items") if isinstance(report.get("items"), list) else []
    expected_keys = contract["full_keys"][offset : offset + len(keys)]
    item_keys = [f"{contract['sha256']}:{str(item.get('case_id') or '')}" for item in items if isinstance(item, dict)]
    if (
        bool(report.get("complete")) is not complete
        or dataset.get("sha256") != contract["sha256"]
        or not keys
        or keys != expected_keys
        or item_keys != keys[: len(items)]
    ):
        return None
    if complete:
        try:
            if int(report.get("case_count")) != len(keys) or len(items) != len(keys):
                return None
        except (TypeError, ValueError):
            return None
    else:
        try:
            completed = int(report.get("completed_cases"))
            expected = int(report.get("expected_cases"))
        except (TypeError, ValueError):
            return None
        completed_keys = [str(value) for value in report.get("completed_case_keys") or []]
        if expected != len(keys) or completed != len(items) or completed_keys != keys[:completed]:
            return None
    return report


def _external_agent_matrix_candidate(launch_path: Path) -> dict | None:
    launch = _agent_matrix_artifact_json(launch_path)
    contract = _agent_matrix_dataset_contract(launch or {})
    if contract is None or launch_path.name != f"{contract['run_id']}.launch.json":
        return None
    base = launch_path.with_name(contract["run_id"])
    partial_path = base.with_suffix(".partial.json")
    final_path = base.with_suffix(".json")
    exit_path = base.with_suffix(".exit")
    lock_path = base.with_suffix(".lock")
    partial = _validated_agent_matrix_report(partial_path, contract, complete=False) if partial_path.exists() else None
    final = _validated_agent_matrix_report(final_path, contract, complete=True) if final_path.exists() else None
    if partial is not None and partial.get("run_id") != contract["run_id"]:
        return None
    if final is not None and final.get("run_id") != contract["run_id"]:
        return None
    live = _agent_matrix_lock_is_live(lock_path, contract["run_id"])
    exit_code: int | None = None
    if exit_path.exists():
        try:
            exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
    if live:
        if partial_path.exists() and partial is None:
            return None
        status = "running"
        current = partial
    elif exit_code == 0 and final is not None:
        status = "completed"
        current = final
    elif exit_code is not None and exit_code != 0:
        status = "failed"
        current = partial
    elif final is not None and int(final.get("process_exit_status") or 0) == 0:
        status = "completed"
        current = final
    else:
        return None

    artifact_dir = launch_path.parent
    completed_items: dict[str, dict] = {}
    report_paths: list[Path] = []
    # Segmented matrix runs share a stable cohort prefix such as
    # ``agent_matrix_v58-``.  Restrict reconciliation to that cohort instead
    # of parsing every JSON artifact in the report directory; production
    # directories contain hundreds of unrelated historical reports.
    cohort_prefix = contract["run_id"].split("-", 1)[0]
    report_candidates = list(artifact_dir.glob(f"{cohort_prefix}-*.json")) if cohort_prefix else []
    if final_path not in report_candidates:
        report_candidates.append(final_path)
    for report_path in report_candidates:
        if not report_path.exists():
            continue
        if report_path.name.endswith((".launch.json", ".partial.json")):
            continue
        report = _validated_agent_matrix_report(report_path, contract, complete=True)
        if report is None:
            continue
        provenance = report["provenance"]["dataset"]
        report_offset = int(provenance.get("offset") or 0)
        report_keys = [str(value) for value in provenance.get("ordered_case_keys") or []]
        if report_path != final_path and report_offset + len(report_keys) > contract["offset"]:
            continue
        report_paths.append(report_path)
        completed_items.update({key: item for key, item in zip(report_keys, report["items"])})
    current_keys: list[str] = []
    if current is not None:
        current_keys = [str(value) for value in current["provenance"]["dataset"]["ordered_case_keys"]][: len(current["items"])]
        completed_items.update({key: item for key, item in zip(current_keys, current["items"])})
        current_path = final_path if current is final else partial_path
        if current_path not in report_paths:
            report_paths.append(current_path)
    completed = len(set(contract["full_keys"]) & set(completed_items))
    result_path = final_path if final is not None and status == "completed" else partial_path
    updated_paths = [launch_path, *(path for path in (partial_path, final_path, exit_path, lock_path) if path.exists())]
    latest_item = current["items"][-1] if current and current.get("items") else None
    return {
        "id": f"{EXTERNAL_EVAL_RUN_PREFIX}{contract['run_id']}",
        "artifact_run_id": contract["run_id"],
        "run_type": "external_agent_matrix",
        "workflow": "agent_matrix",
        "status": status,
        "total": len(contract["cases"]),
        "completed": min(completed, len(contract["cases"])),
        "completed_questions": min(completed, len(contract["cases"])),
        "limit": len(contract["cases"]),
        "current_case_id": str((latest_item or {}).get("case_id") or "") or None,
        "live_results": {},
        "dataset_path": contract["relative_path"],
        "results_path": str(result_path.relative_to(MANUALS_ROOT)),
        "report_paths": [str(path.relative_to(MANUALS_ROOT)) for path in sorted(set(report_paths))],
        "provisional_case_ids": [str(item.get("case_id") or "") for item in (current or {}).get("items", [])] if status == "running" else [],
        "segment_offset": contract["offset"],
        "segment_completed": len((current or {}).get("items", [])),
        "segment_total": contract["expected"],
        "exit_code": exit_code,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(path.stat().st_mtime for path in updated_paths))),
    }


def _external_agent_matrix_run(run_id: str | None = None) -> dict | None:
    requested = str(run_id).removeprefix(EXTERNAL_EVAL_RUN_PREFIX) if run_id else None
    artifact_dir = TEST_REPORTS_DIR / "retrieval_improvement"
    if requested:
        launch_path = artifact_dir / f"{requested}.launch.json"
        if not launch_path.exists():
            return None
        candidate = _external_agent_matrix_candidate(launch_path)
        return candidate if candidate and candidate["artifact_run_id"] == requested else None
    try:
        launches = sorted(artifact_dir.glob("*.launch.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return None

    # A live writer wins even when a newer completed launch exists.  Checking
    # lock ownership is cheap and avoids fully parsing every historical run.
    for launch_path in launches:
        artifact_run_id = launch_path.name.removesuffix(".launch.json")
        lock_path = launch_path.with_name(f"{artifact_run_id}.lock")
        if not lock_path.exists() or not _agent_matrix_lock_is_live(lock_path, artifact_run_id):
            continue
        candidate = _external_agent_matrix_candidate(launch_path)
        if candidate is not None and candidate["status"] == "running":
            return candidate

    # Launches are newest-first, so the first validated terminal artifact is
    # the current reconciled result.  Do not parse the entire history.
    for launch_path in launches:
        candidate = _external_agent_matrix_candidate(launch_path)
        if candidate is not None:
            return candidate
    return None


def _external_eval_run(run_id: str | None = None) -> dict | None:
    legacy = _legacy_external_eval_run(run_id)
    if legacy is not None:
        return legacy
    return _external_agent_matrix_run(run_id)


def _sync_external_eval_event(run_id: str) -> dict | None:
    snapshot = _external_eval_run(run_id)
    if snapshot is None:
        return None
    observation = (int(snapshot["completed"]), str(snapshot["status"]))
    with EXTERNAL_EVAL_LOCK:
        if EXTERNAL_EVAL_OBSERVATIONS.get(run_id) == observation:
            return snapshot
        EXTERNAL_EVAL_OBSERVATIONS[run_id] = observation
    terminal = snapshot["status"] in TERMINAL_RUN_STATUSES
    workflow = str(snapshot.get("workflow") or "evaluation")
    if terminal:
        phase = "job_completed" if snapshot["status"] == "completed" else "job_failed"
    else:
        phase = "agent_matrix_progress" if workflow == "agent_matrix" else "evaluation_progress"
    event_name = (
        "external_agent_matrix_completed"
        if workflow == "agent_matrix" and snapshot["status"] == "completed"
        else "external_agent_matrix_failed"
        if workflow == "agent_matrix" and terminal
        else "external_agent_matrix_progress"
        if workflow == "agent_matrix" and not terminal
        else "external_evaluation_completed"
        if snapshot["status"] == "completed"
        else "external_evaluation_failed"
        if terminal
        else "external_evaluation_progress"
    )
    _try_publish_component_event(
        run_id,
        workflow=workflow,
        phase=phase,
        status=str(snapshot["status"]),
        provisional=not terminal,
        total=int(snapshot["total"]),
        artifact_ref=str(snapshot["results_path"]) if snapshot["status"] == "completed" else None,
        payload={
            "event": event_name,
            "completed_questions": int(snapshot["completed"]),
            "total_questions": int(snapshot["total"]),
            "dataset_path": snapshot["dataset_path"],
            "results_path": snapshot["results_path"],
            "current_case_id": snapshot.get("current_case_id"),
        },
    )
    return snapshot


def _completed_result_run_ids() -> set[str]:
    completed = {
        path.name.removeprefix("retrieval_eval_summary_").removesuffix(".json")
        for path in TEST_REPORTS_DIR.glob("retrieval_eval_summary_*.json")
    }
    for event_path in TEST_REPORTS_DIR.glob("question_matrix_job_*_events.jsonl"):
        try:
            events = _read_jsonl(event_path)
        except (OSError, json.JSONDecodeError):
            continue
        for event in events:
            if event.get("event") != "dataset_completed" or not event.get("results_path"):
                continue
            completed.add(_result_run_id(Path(str(event["results_path"]))))
    return completed


def _load_result_index(excluded_run_ids: set[str]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    completed_run_ids = _completed_result_run_ids()
    result_paths = sorted(TEST_REPORTS_DIR.glob("retrieval_eval_results_*.jsonl"), key=lambda path: path.stat().st_mtime)
    for path in result_paths:
        run_id = _result_run_id(path)
        if run_id in excluded_run_ids or run_id not in completed_run_ids:
            continue
        try:
            rows = _read_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            case = row.get("case") or {}
            dataset = str(row.get("dataset") or "")
            key = _matrix_row_key(dataset, case) if dataset else _case_key(case)
            if not key:
                continue
            query_debug_result = row.get("query_debug_result") or {}
            indexed[key] = {
                "run_id": run_id,
                "path": str(path.relative_to(MANUALS_ROOT)),
                "evaluation": row.get("retrieval_evaluation") or row.get("evaluation") or {},
                "answer": row.get("answer") or {},
                "answer_evaluation": row.get("answer_evaluation") or {},
                "query_debug_result": {
                    "completed_steps": query_debug_result.get("completed_steps") or [],
                    "step_timings_ms": query_debug_result.get("step_timings_ms") or {},
                    "answer": query_debug_result.get("answer") or row.get("answer") or {},
                    "stages": [
                        {
                            "name": stage.get("name"),
                            "label": stage.get("label"),
                            "sample_count": len(stage.get("samples") or []),
                            "samples": [
                                _compact_stage_sample(sample)
                                for sample in (stage.get("samples") or [])[:100]
                                if isinstance(sample, dict)
                            ],
                        }
                        for stage in query_debug_result.get("stages") or []
                    ],
                },
            }
    return indexed


def _matrix_cell(status: str, detail: str = "", label: str | None = None) -> dict[str, str]:
    cell = {"status": status, "detail": detail}
    if label is not None:
        cell["label"] = label
    return cell


def _compact_stage_sample(sample: dict) -> dict:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return {
        "chunk_id": sample.get("chunk_id") or sample.get("id"),
        "source_document_id": sample.get("source_document_id") or metadata.get("source_document_id"),
        "document_version_id": sample.get("document_version_id") or metadata.get("document_version_id"),
        "title": sample.get("title"),
        "pages": sample.get("pages") or metadata.get("pages") or [],
        "chunk_type": sample.get("chunk_type") or metadata.get("chunk_type"),
        "retrieval_stage": sample.get("retrieval_stage") or metadata.get("retrieval_stage"),
        "score": sample.get("score"),
    }


def _expected_targets(case: dict) -> dict[str, set[str]]:
    expected_evidence = case.get("expected_evidence") if isinstance(case.get("expected_evidence"), list) else []
    document_ids = {str(case.get("source_document_id") or "")}
    chunk_ids = {str(case.get("source_chunk_id") or "")}
    page_numbers: set[int] = set()
    try:
        page_from = int(case.get("page_from") or 0)
        page_to = int(case.get("page_to") or 0)
    except (TypeError, ValueError):
        page_from = 0
        page_to = 0
    if page_from > 0:
        if page_to < page_from:
            page_to = page_from
        page_numbers.update(range(page_from, page_to + 1))
    for item in expected_evidence:
        if not isinstance(item, dict):
            continue
        document_ids.add(str(item.get("source_document_id") or case.get("source_document_id") or ""))
        chunk_ids.add(str(item.get("chunk_id") or ""))
        try:
            item_page_from = int(item.get("page_from") or item.get("page") or 0)
            item_page_to = int(item.get("page_to") or item_page_from or 0)
        except (TypeError, ValueError):
            item_page_from = 0
            item_page_to = 0
        if item_page_from > 0:
            if item_page_to < item_page_from:
                item_page_to = item_page_from
            page_numbers.update(range(item_page_from, item_page_to + 1))
    document_ids.discard("")
    chunk_ids.discard("")
    return {"document_ids": document_ids, "page_numbers": page_numbers, "chunk_ids": chunk_ids}


def _stage_samples_by_step(debug_result: dict) -> dict[str, list[dict]]:
    return {
        str(stage.get("name") or ""): list(stage.get("samples") or [])
        for stage in debug_result.get("stages") or []
        if isinstance(stage, dict)
    }


def _sample_pages(sample: dict, metadata: dict) -> set[int]:
    raw_pages = sample.get("pages") or metadata.get("pages") or []
    if not raw_pages:
        raw_page_from = sample.get("page_from") or metadata.get("page_from")
        raw_page_to = sample.get("page_to") or metadata.get("page_to") or raw_page_from
        if raw_page_from is not None:
            raw_pages = [raw_page_from, raw_page_to]
    pages: set[int] = set()
    for page in raw_pages:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            pages.add(page_number)
    return pages


def _stage_target_counts(samples: list[dict], targets: dict[str, set[str]]) -> tuple[int, int, int]:
    matched_docs: set[str] = set()
    matched_pages: set[int] = set()
    matched_chunks: set[str] = set()
    document_ids = targets["document_ids"]
    page_numbers = targets["page_numbers"]
    chunk_ids = targets["chunk_ids"]
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        document_id = str(sample.get("source_document_id") or metadata.get("source_document_id") or "")
        chunk_id = str(sample.get("chunk_id") or sample.get("id") or "")
        if document_id in document_ids:
            matched_docs.add(document_id)
            matched_pages.update(_sample_pages(sample, metadata).intersection(page_numbers))
        if chunk_id in chunk_ids:
            matched_chunks.add(chunk_id)
    return len(matched_docs), len(matched_pages), len(matched_chunks)


def _result_stage_cell(
    key: str,
    samples_by_step: dict[str, list[dict]],
    targets: dict[str, set[str]],
    *,
    previously_found: bool,
) -> tuple[dict[str, str], bool]:
    step_name = next((step for step, mapped_key in DEBUG_STEP_TO_MATRIX_KEY.items() if mapped_key == key), "")
    samples = samples_by_step.get(step_name) or []
    expected_doc_count = max(1, len(targets["document_ids"]))
    expected_page_count = len(targets["page_numbers"])
    expected_chunk_count = len(targets["chunk_ids"])
    matched_doc_count, matched_page_count, matched_chunk_count = _stage_target_counts(samples, targets)
    found = bool(matched_doc_count or matched_page_count or matched_chunk_count)
    detail = (
        f"{matched_doc_count}/{expected_doc_count} expected document(s), "
        f"{matched_page_count}/{expected_page_count or 1} expected page(s), "
        f"{matched_chunk_count}/{expected_chunk_count or 1} expected chunk(s) in stage sample window"
    )
    if expected_chunk_count and matched_chunk_count >= expected_chunk_count:
        return _matrix_cell("pass", detail, "YES"), True
    if expected_page_count and matched_page_count >= expected_page_count:
        return _matrix_cell("pass", f"Expected page evidence found; {detail}", "YES"), True
    if expected_page_count and matched_page_count:
        return _matrix_cell("fail", detail, "PARTIAL"), True
    if matched_doc_count:
        return _matrix_cell("fail", f"Expected document found, but expected chunk evidence is absent; {detail}", "DOC_ONLY"), True
    if expected_chunk_count and matched_chunk_count:
        return _matrix_cell("fail", detail, "PARTIAL"), True
    if found:
        return _matrix_cell("fail", detail, "PARTIAL"), True
    if previously_found:
        return _matrix_cell("fail", f"Expected target was present earlier, then absent here; {detail}", "DROPPED"), False
    return _matrix_cell("fail", detail, "NO"), False


def _retrieval_retention_cell(cells: dict[str, dict[str, str]], retrieval: dict) -> dict[str, str]:
    context_cell = cells.get("assemble") or {}
    context_label = context_cell.get("label") or ""
    context_detail = context_cell.get("detail") or "final context was not recorded"
    if retrieval.get("passed") and context_cell.get("status") == "pass":
        return _matrix_cell("pass", f"Expected evidence retained in final context; {context_detail}", "PASS")
    match_reason = str(retrieval.get("match_reason") or "")
    equivalent_match_reasons = {
        "same_section_term_overlap",
        "same_document_term_overlap",
        "same_document_answerable_evidence",
        "same_document_snippet_evidence",
        "applicable_equivalent_answer_evidence",
        "cross_document_semantic_evidence",
    }
    if retrieval.get("passed") and match_reason in equivalent_match_reasons:
        return _matrix_cell(
            "pass",
            f"Answer-bearing equivalent evidence retained in final context ({match_reason}); exact anchor retention: {context_detail}",
            "EQUIV",
        )
    if context_cell.get("status") == "pass":
        return _matrix_cell("pass", f"Expected evidence retained in final context; {context_detail}", "PASS")
    if context_label in {"DOC_ONLY"}:
        return _matrix_cell("fail", f"Expected document retained, but expected chunk evidence is missing; {context_detail}", "FAIL")
    if context_label in {"NO", "PARTIAL", "DROPPED"} or context_cell.get("status") == "fail":
        return _matrix_cell("fail", f"Expected evidence missing from final context; {context_detail}", "FAIL")
    if retrieval:
        if retrieval.get("passed") and "candidate_recall" not in retrieval:
            return _matrix_cell("pass", f"rank {retrieval.get('rank')}" if retrieval.get("rank") else "retrieval passed", "PASS")
        return _matrix_cell("fail", retrieval.get("failure_category") or "expected document missing from retrieval", "FAIL")
    return _matrix_cell("blank", "not scored yet")


def _matrix_retrieval_evaluation(evaluation: dict, retrieval_cell: dict[str, str]) -> dict:
    normalized = dict(evaluation)
    evidence_passed = bool(evaluation.get("passed"))
    retention_passed = retrieval_cell.get("status") == "pass"
    normalized["evidence_passed"] = evidence_passed
    normalized["retention_passed"] = retention_passed
    normalized["retrieval_retention_label"] = retrieval_cell.get("label")
    normalized["retrieval_retention_detail"] = retrieval_cell.get("detail")
    if retention_passed and evidence_passed:
        normalized["passed"] = True
        normalized.pop("failure_category", None)
        normalized.pop("failure_reasons", None)
    elif retention_passed:
        if not evidence_passed and evaluation.get("failure_category"):
            normalized["evidence_failure_category"] = evaluation.get("failure_category")
        if not evidence_passed and evaluation.get("failure_reasons"):
            normalized["evidence_failure_reasons"] = evaluation.get("failure_reasons")
        normalized["passed"] = False
        normalized["failure_category"] = normalized.get("failure_category") or "expected_evidence_missing"
    else:
        normalized["passed"] = False
        normalized["failure_category"] = normalized.get("failure_category") or "retrieval_context_missing"
        normalized.setdefault("failure_reasons", [])
        if isinstance(normalized["failure_reasons"], list):
            normalized["failure_reasons"].append(retrieval_cell.get("detail") or "Expected document missing from final context.")
    return normalized


def _block_answer_cells(cells: dict[str, dict[str, str]], detail: str = "blocked by failed retrieval/context") -> dict[str, dict[str, str]]:
    for key in ANSWER_STAGE_KEYS:
        cells[key] = _matrix_cell("blank", detail)
    return cells


def _question_type(case: dict) -> dict[str, object]:
    expected_evidence = case.get("expected_evidence") if isinstance(case.get("expected_evidence"), list) else []
    expected_document_ids = {
        str(item.get("source_document_id") or case.get("source_document_id") or "")
        for item in expected_evidence
        if isinstance(item, dict)
    }
    expected_document_ids.discard("")
    multi_document = (
        len(expected_document_ids) > 1
        or str(case.get("generation_method") or "") == "cross_document_same_field_evidence"
    )
    multi_step = (
        str(case.get("retrieval_task") or "") == "multi_step_retrieval"
        or len(expected_evidence) > 1
        or multi_document
    )
    return {
        "label": "Multi-doc" if multi_document else ("Multi-step" if multi_step else "Single"),
        "multi_step": multi_step,
        "multi_document": multi_document,
        "retrieval_task": case.get("retrieval_task") or "single_step_retrieval",
        "generation_method": case.get("generation_method"),
        "expected_evidence_count": len(expected_evidence) or 1,
        "expected_document_count": len(expected_document_ids) or (1 if case.get("source_document_id") else 0),
    }


def _generation_review_cell(case: dict) -> dict[str, str]:
    generation_method = str(case.get("generation_method") or "")
    benchmark_quality = str(case.get("benchmark_quality") or "")
    if generation_method.startswith("reviewed_llm:") or benchmark_quality == "model_reviewed":
        return _matrix_cell("pass", "accepted by generation reviewer", "ACCEPT")
    if generation_method:
        return _matrix_cell("blank", "generated without model review", "")
    return _matrix_cell("blank", "not generated in the current live job", "")


def _build_row_cells(item: dict | None, case: dict | None = None) -> dict[str, dict[str, str]]:
    cells = {
        key: _matrix_cell("blank", "not scored yet")
        for key in MATRIX_STAGE_KEYS
    }
    if not item:
        return cells

    case = case or item.get("case") or {}
    debug_result = item.get("query_debug_result") or {}
    completed_steps = set(debug_result.get("completed_steps") or [])
    retrieval_step_to_key = {
        "classify_query": "query_classify",
        "build_filters": "filters",
        "run_dense_search": "dense",
        "run_sparse_search": "sparse",
        "run_special_search": "special",
        "fuse_results": "fuse",
        "rerank_results": "rerank",
        "assemble_context": "assemble",
    }
    answer_step_to_key = {
        "judge_answer_inputs": "relevance",
        "summarize_answer_inputs": "summaries",
        "generate_answer": "generation",
    }
    samples_by_step = _stage_samples_by_step(debug_result)
    targets = _expected_targets(case)
    previously_found = False
    for step, key in retrieval_step_to_key.items():
        if step in completed_steps:
            if step == "classify_query":
                cells[key] = _matrix_cell("pass", "Query classification completed; this stage does not return documents.", "DONE")
            elif step == "build_filters":
                cells[key] = _matrix_cell("pass", "Filters were built; document loss is measured by the following search stages.", "SET")
            elif step in RESULT_STAGE_STEPS:
                if step not in samples_by_step:
                    cells[key] = _matrix_cell("blank", "debug step completed, but stage samples were not recorded")
                else:
                    cells[key], found = _result_stage_cell(key, samples_by_step, targets, previously_found=previously_found)
                    previously_found = found or previously_found and cells[key].get("label") != "DROPPED"
            else:
                cells[key] = _matrix_cell("pass", "debug step completed", "DONE")

    retrieval = item.get("retrieval_evaluation") or item.get("evaluation") or {}
    metadata = retrieval.get("metadata_document_selection") or {}
    if metadata.get("attempted"):
        cells["metadata"] = _matrix_cell(
            "pass" if metadata.get("passed") else "fail",
            f"rank {metadata.get('rank')}" if metadata.get("passed") else metadata.get("failure_category") or "expected document not selected",
        )
    else:
        cells["metadata"] = _matrix_cell("blank", "not attempted")

    if retrieval and not completed_steps:
        cells["retrieval"] = _matrix_cell(
            "pass" if retrieval.get("passed") else "fail",
            (
                f"completed saved evaluation; rank {retrieval.get('rank')}"
                if retrieval.get("passed")
                else retrieval.get("failure_category") or "retrieval evaluation failed"
            ),
            "PASS" if retrieval.get("passed") else "FAIL",
        )
    else:
        cells["retrieval"] = _retrieval_retention_cell(cells, retrieval)
    if cells["retrieval"]["status"] != "pass":
        return _block_answer_cells(cells)

    for step, key in answer_step_to_key.items():
        if step in completed_steps:
            cells[key] = _matrix_cell("pass", "debug step completed")

    answer_eval = item.get("answer_evaluation") or {}
    if not answer_eval:
        return cells

    missing_docs = answer_eval.get("missing_document_ids") or []
    cells["answer_docs"] = _matrix_cell(
        "fail" if answer_eval.get("expected_document_used") is False or missing_docs else "pass",
        f"missing {len(missing_docs)} expected document(s)" if missing_docs else "expected document used",
    )

    citation = answer_eval.get("citation_fidelity") or {}
    if citation.get("checked"):
        cells["citations"] = _matrix_cell(
            "pass" if citation.get("passed") else "fail",
            f"{citation.get('checked_quote_count') or 0} quote(s) checked" if citation.get("passed") else "unsupported or missing citation evidence",
        )

    term_check = answer_eval.get("term_check") or {}
    if term_check:
        llm_info = term_check.get("llm_required_information") or answer_eval.get("llm_required_information") or {}
        term_detail = (
            f"model judge: {llm_info.get('reason') or 'required information judged'}"
            if term_check.get("llm_judged")
            else ("required terms present" if term_check.get("passed") else "expected terms/actions missing")
        )
        cells["terms"] = _matrix_cell(
            "pass" if term_check.get("passed") else "fail",
            term_detail,
        )

    cells["answer"] = _matrix_cell(
        "pass" if answer_eval.get("passed") else "fail",
        ", ".join(answer_eval.get("failure_reasons") or []) or "overall answer score",
    )
    return cells


def _mark_cells_provisional(cells: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    marked = deepcopy(cells)
    for cell in marked.values():
        final_status = str(cell.get("status") or "blank")
        if final_status not in {"pass", "fail"}:
            continue
        cell["final_status"] = final_status
        cell["status"] = "provisional"
        cell["label"] = f"LIVE {final_status.upper()}"
        cell["detail"] = f"{cell.get('detail') or final_status} · provisional until terminal reconciliation"
    return marked


def _build_external_eval_matrix(current_run: dict) -> dict:
    dataset_path = MANUALS_ROOT / str(current_run["dataset_path"])
    results_path = MANUALS_ROOT / str(current_run["results_path"])
    cases = _read_jsonl(dataset_path)
    results = _read_jsonl(results_path) if results_path.exists() else []
    result_index = {
        _case_key(result.get("case") or {}): result
        for result in results
        if _case_key(result.get("case") or {})
    }
    provisional = current_run["status"] != "completed"
    dataset_rel = str(current_run["dataset_path"])
    rows: list[dict] = []
    for row_index, case in enumerate(cases, start=1):
        case_key = _case_key(case)
        result = result_index.get(case_key)
        item = None
        latest_result = None
        if result:
            item = {
                "evaluation": result.get("retrieval_evaluation") or result.get("evaluation") or {},
                "answer": result.get("answer") or {},
                "answer_evaluation": result.get("answer_evaluation") or {},
                "query_debug_result": result.get("query_debug_result") or {},
            }
            latest_result = {
                "run_id": current_run["artifact_run_id"],
                "path": current_run["results_path"],
                **item,
            }
        cells = _build_row_cells(item, case)
        if provisional and result:
            cells = _mark_cells_provisional(cells)
        elif provisional:
            for cell in cells.values():
                cell["detail"] = "waiting for the current evaluation run"
        rows.append(
            {
                "key": _matrix_row_key(dataset_rel, case_key),
                "dataset": dataset_rel,
                "dataset_status": "current_run" if provisional else "latest_completed_run",
                "question_number": row_index,
                "case": case,
                "question_type": _question_type(case),
                "generation_review": _generation_review_cell(case),
                "latest_result": latest_result,
                "provisional": provisional,
                "cells": cells,
            }
        )
    single_step = sum(1 for case in cases if not _question_type(case)["multi_step"])
    return {
        "manifest_path": current_run.get("manifest_path") or dataset_rel,
        "official_total_questions": len(cases),
        "official_single_step_questions": single_step,
        "official_multi_step_questions": len(cases) - single_step,
        "loaded_questions": len(rows),
        "datasets": [
            {
                "path": dataset_rel,
                "status": "current_run" if provisional else "latest_completed_run",
                "total_questions": len(cases),
            }
        ],
        "question_view": {"hide_bank_questions": True, "generated_datasets": []},
        "active_job": None,
        "current_run": current_run,
        "rows": rows,
    }


def _build_question_matrix() -> dict:
    current_run = _external_eval_run()
    if (
        current_run is not None
        and current_run.get("run_type") == "external_retrieval_eval"
        and _active_question_matrix_job_id() is None
    ):
        return _build_external_eval_matrix(current_run)
    manifest_path = TEST_REPORTS_DIR / "retrieval_accuracy_question_bank_manifest.json"
    manifest = _read_json(manifest_path)
    question_bank = manifest.get("question_bank") or {}
    question_view, active_datasets = _visible_question_matrix_datasets(question_bank)
    excluded_run_ids = {
        str(exclusion.get("run_id"))
        for exclusion in (question_bank.get("run_exclusions") or manifest.get("run_exclusions") or [])
        if exclusion.get("run_id")
    }
    result_index = _load_result_index(excluded_run_ids)
    rows: list[dict] = []
    for dataset in active_datasets:
        dataset_path = MANUALS_ROOT / str(dataset.get("path") or "")
        if not dataset_path.exists():
            continue
        for row_index, case in enumerate(_read_jsonl(dataset_path), start=1):
            case_key = _case_key(case)
            key = _matrix_row_key(str(dataset_path.relative_to(MANUALS_ROOT)), case_key)
            result = result_index.get(key) or result_index.get(case_key)
            item = result if result else None
            rows.append(
                {
                    "key": key,
                    "dataset": str(dataset_path.relative_to(MANUALS_ROOT)),
                    "dataset_status": dataset.get("status"),
                    "question_number": row_index,
                    "case": case,
                    "question_type": _question_type(case),
                    "generation_review": _generation_review_cell(case),
                    "latest_result": {
                        "run_id": result.get("run_id"),
                        "path": result.get("path"),
                        "evaluation": result.get("evaluation"),
                        "answer": result.get("answer"),
                        "answer_evaluation": result.get("answer_evaluation"),
                        "query_debug_result": result.get("query_debug_result"),
                    } if result else None,
                    "cells": _build_row_cells(item, case),
                }
            )
    active_job = None
    active_job_id = _active_question_matrix_job_id()
    if active_job_id:
        try:
            active_job = _question_matrix_job_snapshot(active_job_id)
        except KeyError:
            active_job = None
    return {
        "manifest_path": str(manifest_path.relative_to(MANUALS_ROOT)),
        "official_total_questions": question_bank.get("total_questions"),
        "official_single_step_questions": question_bank.get("single_step_questions"),
        "official_multi_step_questions": question_bank.get("multi_step_questions"),
        "loaded_questions": len(rows),
        "datasets": active_datasets,
        "question_view": question_view,
        "active_job": active_job,
        "rows": rows,
    }


def _visible_question_matrix_datasets(question_bank: dict) -> tuple[dict, list[dict]]:
    question_view = _load_question_matrix_question_view()
    active_datasets = [] if question_view.get("hide_bank_questions") else _active_question_datasets(question_bank)
    active_datasets.extend(question_view.get("generated_datasets") or [])
    return question_view, active_datasets


def _question_matrix_job_snapshot(job_id: str) -> dict:
    _load_question_matrix_jobs_if_needed()
    with MATRIX_JOBS_LOCK:
        return dict(MATRIX_JOBS[job_id])


def _update_question_matrix_job(job_id: str, **updates: object) -> None:
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        job.update(updates)
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _persist_question_matrix_jobs_locked()


def _update_question_matrix_live_cell(job_id: str, case_id: str, key: str, status: str, detail: str = "", label: str | None = None) -> None:
    if key not in MATRIX_STAGE_KEYS:
        return
    cell = _matrix_cell(status, detail, label)
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        live_cells = dict(job.get("live_cells") or {})
        row_cells = dict(live_cells.get(case_id) or {})
        row_cells[key] = cell
        live_cells[case_id] = row_cells
        job["live_cells"] = live_cells
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _persist_question_matrix_jobs_locked()
    _try_publish_component_event(
        job_id,
        workflow="question_matrix",
        phase="matrix_cell",
        status="running",
        entity_type="matrix_cell",
        entity_id=f"{case_id}:{key}",
        provisional=True,
        payload={"case_id": case_id, "matrix_key": key, "cell": cell},
    )


def _replace_question_matrix_live_cells(job_id: str, case_id: str, cells: dict[str, dict[str, str]]) -> None:
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        live_cells = dict(job.get("live_cells") or {})
        live_cells[case_id] = {key: dict(value) for key, value in cells.items()}
        job["live_cells"] = live_cells
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _persist_question_matrix_jobs_locked()
    for key, cell in cells.items():
        if key in MATRIX_STAGE_KEYS:
            _try_publish_component_event(
                job_id,
                workflow="question_matrix",
                phase="matrix_cell",
                status="running",
                entity_type="matrix_cell",
                entity_id=f"{case_id}:{key}",
                provisional=True,
                payload={"case_id": case_id, "matrix_key": key, "cell": dict(cell)},
            )


def _update_question_matrix_live_result(job_id: str, case_id: str, record: dict) -> None:
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS[job_id]
        live_results = dict(job.get("live_results") or {})
        live_results[case_id] = {
            "evaluation": record.get("evaluation") or {},
            "answer": record.get("answer") or {},
            "answer_evaluation": record.get("answer_evaluation") or {},
            "query_debug_result": record.get("query_debug_result") or {},
        }
        job["live_results"] = live_results
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _persist_question_matrix_jobs_locked()
    _try_publish_component_event(
        job_id,
        workflow="question_matrix",
        phase="metrics",
        status="running",
        entity_type="question",
        entity_id=case_id,
        provisional=True,
        payload={
            "case_id": case_id,
            "evaluation": deepcopy(record.get("evaluation") or {}),
            "answer_evaluation": deepcopy(record.get("answer_evaluation") or {}),
        },
    )


def _active_question_matrix_job_id() -> str | None:
    _load_question_matrix_jobs_if_needed()
    for job_id, job in MATRIX_JOBS.items():
        if job.get("status") in {"queued", "running", "stopping"}:
            return job_id
    return None


def _question_matrix_jobs_state_path() -> Path:
    return TEST_REPORTS_DIR / ".question_matrix_jobs.json"


def _question_matrix_job_events_path(job_id: str) -> Path:
    safe_job_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", job_id)
    return TEST_REPORTS_DIR / f"question_matrix_job_{safe_job_id}_events.jsonl"


def _question_matrix_question_view_path() -> Path:
    return TEST_REPORTS_DIR / ".question_matrix_question_view.json"


def _load_question_matrix_question_view() -> dict:
    path = _question_matrix_question_view_path()
    if not path.exists():
        return {"hide_bank_questions": False, "generated_datasets": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"hide_bank_questions": False, "generated_datasets": []}
    generated = payload.get("generated_datasets") if isinstance(payload, dict) else []
    return {
        "hide_bank_questions": bool(payload.get("hide_bank_questions")) if isinstance(payload, dict) else False,
        "generated_datasets": [dict(item) for item in generated if isinstance(item, dict)],
        "updated_at": payload.get("updated_at") if isinstance(payload, dict) else None,
    }


def _save_question_matrix_question_view(payload: dict) -> None:
    payload = {
        **payload,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _question_matrix_question_view_path().write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _add_generated_question_dataset_to_view(dataset_path: str, question_count: int | None = None) -> None:
    if not dataset_path:
        return
    rel = dataset_path
    try:
        rel = str(Path(dataset_path).resolve().relative_to(MANUALS_ROOT.resolve()))
    except (OSError, ValueError):
        rel = dataset_path
    view = _load_question_matrix_question_view()
    generated = [item for item in (view.get("generated_datasets") or []) if item.get("path") != rel]
    generated.append(
        {
            "path": rel,
            "status": "generated_candidate",
            "total_questions": question_count,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _save_question_matrix_question_view({"hide_bank_questions": bool(view.get("hide_bank_questions", True)), "generated_datasets": generated})


def _record_question_matrix_job_event(job_id: str, event_type: str, **fields: object) -> None:
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": job_id,
        "event": event_type,
        **fields,
    }
    try:
        path = _question_matrix_job_events_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
    except OSError:
        pass
    with MATRIX_JOBS_LOCK:
        job = MATRIX_JOBS.get(job_id)
        if not job:
            return
        events = list(job.get("events") or [])
        events.append(event)
        job["events"] = events[-MATRIX_JOB_EVENT_TAIL_LIMIT:]
        if "path" in locals():
            try:
                job["event_log_path"] = str(path.relative_to(MANUALS_ROOT))
            except ValueError:
                job["event_log_path"] = str(path)
        job["updated_at"] = event["timestamp"]
        _persist_question_matrix_jobs_locked()
        job_status = str(job.get("status") or "running")
    _try_publish_component_event(
        job_id,
        workflow="question_matrix",
        phase=event_type,
        status=job_status,
        entity_type="question" if fields.get("case_id") else "run",
        entity_id=str(fields.get("case_id") or job_id),
        provisional=job_status.lower() not in TERMINAL_RUN_STATUSES,
        payload=deepcopy(event),
        artifact_ref=str(fields.get("results_path")) if fields.get("results_path") else None,
    )


def _persist_question_matrix_jobs_locked() -> None:
    state_path = _question_matrix_jobs_state_path()
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_jobs: dict[str, dict] = {}
        for job_id, job in MATRIX_JOBS.items():
            persisted = dict(job)
            current_row_key = str(job.get("current_row_key") or "")
            live_cells = job.get("live_cells") or {}
            live_results = job.get("live_results") or {}
            persisted["live_cells"] = (
                {current_row_key: live_cells[current_row_key]}
                if current_row_key and current_row_key in live_cells
                else {}
            )
            persisted["live_results"] = (
                {current_row_key: live_results[current_row_key]}
                if current_row_key and current_row_key in live_results
                else {}
            )
            persisted["events"] = list(job.get("events") or [])[-MATRIX_JOB_PERSISTED_EVENT_TAIL_LIMIT:]
            persisted_jobs[job_id] = persisted
        state_path.write_text(json.dumps({"jobs": persisted_jobs}, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass


def _load_question_matrix_jobs_if_needed() -> None:
    global MATRIX_JOBS_LOADED
    if MATRIX_JOBS_LOADED:
        return
    with MATRIX_JOBS_LOCK:
        if MATRIX_JOBS_LOADED:
            return
        state_path = _question_matrix_jobs_state_path()
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                jobs = payload.get("jobs") if isinstance(payload, dict) else {}
                if isinstance(jobs, dict):
                    MATRIX_JOBS.update({str(job_id): dict(job) for job_id, job in jobs.items() if isinstance(job, dict)})
                    _refresh_recovered_question_matrix_jobs_locked()
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        MATRIX_JOBS_LOADED = True


def _refresh_recovered_question_matrix_jobs_locked() -> None:
    changed = False
    for job in MATRIX_JOBS.values():
        if job.get("status") not in {"queued", "running", "stopping"}:
            continue
        pid = _job_pid(job)
        if pid and _pid_is_running(pid):
            job["recovered"] = True
            changed = True
            continue
        if not pid and not job.get("started_at"):
            continue
        job["status"] = "failed"
        job["error"] = "UI server restarted before this matrix job finished; no active worker process was found."
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        job["updated_at"] = job["completed_at"]
        changed = True
    if changed:
        _persist_question_matrix_jobs_locked()


def _job_pid(job: dict) -> int | None:
    try:
        pid = int(job.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _question_matrix_job_cancel_requested(job_id: str) -> bool:
    _load_question_matrix_jobs_if_needed()
    with MATRIX_JOBS_LOCK:
        return bool((MATRIX_JOBS.get(job_id) or {}).get("cancel_requested"))


def _raise_if_question_matrix_job_cancelled(job_id: str) -> None:
    if _question_matrix_job_cancel_requested(job_id):
        raise MatrixJobCancelled("matrix job stopped")


def _stop_question_matrix_job(job_id: str) -> dict:
    _load_question_matrix_jobs_if_needed()
    with MATRIX_JOBS_LOCK:
        if job_id not in MATRIX_JOBS:
            raise KeyError(job_id)
        job = MATRIX_JOBS[job_id]
        if job.get("status") not in {"queued", "running", "stopping"}:
            raise ValueError(f"Job {job_id} is not running.")
        job["status"] = "stopping"
        job["cancel_requested"] = True
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        process = MATRIX_PROCESSES.get(job_id)
        pid = _job_pid(job)
        _persist_question_matrix_jobs_locked()
    if process and process.poll() is None:
        process.terminate()
    elif pid and _pid_is_running(pid):
        os.kill(pid, signal.SIGTERM)
    _record_question_matrix_job_event(job_id, "stop_requested", pid=pid)
    return _question_matrix_job_snapshot(job_id)


def _clear_question_matrix_results() -> dict:
    _load_question_matrix_jobs_if_needed()
    active_job_id = _active_question_matrix_job_id()
    if active_job_id:
        raise ValueError(f"Stop matrix job {active_job_id} before clearing results.")

    patterns = {
        "results": "retrieval_eval_results_*.jsonl",
        "manifests": "retrieval_eval_manifest_*.json",
        "summaries": "retrieval_eval_summary_*.json",
        "job_events": "question_matrix_job_*_events.jsonl",
    }
    deleted: dict[str, int] = {key: 0 for key in patterns}
    for key, pattern in patterns.items():
        for path in TEST_REPORTS_DIR.glob(pattern):
            if not path.is_file():
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            deleted[key] += 1

    state_path = _question_matrix_jobs_state_path()
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass

    with MATRIX_JOBS_LOCK:
        MATRIX_JOBS.clear()
        MATRIX_PROCESSES.clear()
        _persist_question_matrix_jobs_locked()
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass
    return {"deleted": deleted, "total_deleted": sum(deleted.values())}


def _clear_question_matrix_generated_questions() -> dict:
    _load_question_matrix_jobs_if_needed()
    active_job_id = _active_question_matrix_job_id()
    if active_job_id:
        raise ValueError(f"Stop matrix job {active_job_id} before clearing generated questions.")

    question_view = _load_question_matrix_question_view()
    manifest_dataset_paths = {
        (MANUALS_ROOT / str(dataset.get("path") or "")).resolve()
        for dataset in (_read_json(TEST_REPORTS_DIR / "retrieval_accuracy_question_bank_manifest.json").get("question_bank", {}).get("datasets") or [])
        if dataset.get("path")
    }
    candidate_paths = {
        path.resolve()
        for path in TEST_REPORTS_DIR.glob("retrieval_eval_dataset_*.jsonl")
        if path.is_file() and path.resolve() not in manifest_dataset_paths
    }
    for dataset in question_view.get("generated_datasets") or []:
        rel = str(dataset.get("path") or "")
        if not rel:
            continue
        path = (MANUALS_ROOT / rel).resolve()
        if path.is_file() and path not in manifest_dataset_paths:
            candidate_paths.add(path)

    deleted = 0
    for path in sorted(candidate_paths):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        deleted += 1
    _save_question_matrix_question_view({"hide_bank_questions": False, "generated_datasets": []})
    cleared_jobs = 0
    with MATRIX_JOBS_LOCK:
        for job in MATRIX_JOBS.values():
            if job.get("mode") != "generate_questions":
                continue
            if job.get("generated_questions") or job.get("generated_question_count"):
                cleared_jobs += 1
            job["generated_questions"] = []
            job["generated_question_count"] = 0
            job["current_dataset"] = None
            job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if cleared_jobs:
            _persist_question_matrix_jobs_locked()
    return {
        "deleted": {"generated_question_datasets": deleted, "generated_question_job_snapshots": cleared_jobs},
        "total_deleted": deleted,
        "hide_bank_questions": False,
    }


def _reset_question_matrix_bank() -> dict:
    _load_question_matrix_jobs_if_needed()
    active_job_id = _active_question_matrix_job_id()
    if active_job_id:
        raise ValueError(f"Stop matrix job {active_job_id} before resetting the question bank.")

    manifest_path = TEST_REPORTS_DIR / "retrieval_accuracy_question_bank_manifest.json"
    manifest = _read_json(manifest_path)
    question_bank = manifest.get("question_bank")
    if not isinstance(question_bank, dict):
        question_bank = {}
        manifest["question_bank"] = question_bank
    datasets = [dict(dataset) for dataset in question_bank.get("datasets") or [] if isinstance(dataset, dict)]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    backup_dir = TEST_REPORTS_DIR / f"question_bank_reset_{timestamp}_{uuid.uuid4().hex[:8]}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest_backup = backup_dir / manifest_path.name
    shutil.copy2(manifest_path, manifest_backup)

    moved: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    root = MANUALS_ROOT.resolve()
    for dataset in datasets:
        rel = str(dataset.get("path") or "")
        if not rel:
            skipped.append({"path": rel, "reason": "missing path"})
            continue
        source_path = (MANUALS_ROOT / rel).resolve()
        if source_path in seen_paths:
            skipped.append({"path": rel, "reason": "duplicate manifest entry"})
            continue
        seen_paths.add(source_path)
        try:
            relative_source = source_path.relative_to(root)
        except ValueError:
            skipped.append({"path": rel, "reason": "outside manuals root"})
            continue
        if not source_path.is_file():
            skipped.append({"path": rel, "reason": "file not found"})
            continue
        target_path = backup_dir / relative_source
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(target_path)
        moved.append({"from": str(relative_source), "to": str(target_path.relative_to(MANUALS_ROOT))})

    question_bank["datasets"] = []
    question_bank["total_questions"] = 0
    question_bank["single_step_questions"] = 0
    question_bank["multi_step_questions"] = 0
    question_bank["run_exclusions"] = []
    manifest["run_exclusions"] = []
    manifest["reset_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["reset_backup"] = str(backup_dir.relative_to(MANUALS_ROOT))
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    _save_question_matrix_question_view({"hide_bank_questions": False, "generated_datasets": []})
    return {
        "backup_dir": str(backup_dir.relative_to(MANUALS_ROOT)),
        "manifest_backup": str(manifest_backup.relative_to(MANUALS_ROOT)),
        "moved": moved,
        "skipped": skipped,
        "total_moved": len(moved),
    }


def _as_positive_int(value: object, default: int, *, minimum: int = 1, maximum: int = 100000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _as_nonnegative_int(value: object, default: int, *, maximum: int = 1000000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(parsed, maximum))


def _previous_question_dataset_paths(include_generated: bool = True) -> list[Path]:
    paths: list[Path] = []
    try:
        manifest = _read_json(TEST_REPORTS_DIR / "retrieval_accuracy_question_bank_manifest.json")
        for dataset in manifest.get("question_bank", {}).get("datasets") or []:
            rel = str(dataset.get("path") or "")
            if rel:
                paths.append(MANUALS_ROOT / rel)
    except (OSError, json.JSONDecodeError):
        pass
    if include_generated:
        paths.extend(sorted(TEST_REPORTS_DIR.glob("retrieval_eval_dataset_*.jsonl"), key=lambda path: path.stat().st_mtime))
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _start_question_generation_job(payload: dict) -> dict:
    _load_question_matrix_jobs_if_needed()
    with MATRIX_JOBS_LOCK:
        active_job_id = next(
            (existing_id for existing_id, existing_job in MATRIX_JOBS.items() if existing_job.get("status") in {"queued", "running", "stopping"}),
            None,
        )
        if active_job_id:
            raise ValueError(f"Matrix job {active_job_id} is already running.")

    retrieval_task = str(payload.get("retrieval_task") or "single_step_retrieval")
    if retrieval_task not in {"single_step_retrieval", "multi_step_retrieval"}:
        raise ValueError("retrieval_task must be single_step_retrieval or multi_step_retrieval")
    case_family = str(payload.get("multi_step_case_family") or "all")
    if case_family not in {"all", "sibling_table_rows", "contextual_section", "warning_step", "cross_document"}:
        raise ValueError("Unsupported multi-step case family.")

    max_questions = _as_positive_int(payload.get("max_questions"), 20, maximum=10000)
    chunk_offset = _as_nonnegative_int(payload.get("chunk_offset"), 0)
    chunk_window = _as_nonnegative_int(payload.get("chunk_window"), 0)
    questions_per_window = _as_positive_int(payload.get("questions_per_window"), 1, maximum=20)
    num_ctx = _as_nonnegative_int(payload.get("num_ctx"), 4096, maximum=262144)
    per_query_timeout = _as_positive_int(payload.get("per_query_timeout_seconds"), 60, maximum=3600)
    question_generation_timeout = _as_positive_int(
        payload.get("question_generation_timeout_seconds"),
        DEFAULT_QUESTION_GENERATION_TIMEOUT_SECONDS,
        maximum=3600,
    )
    warmup_queries = _as_nonnegative_int(payload.get("warmup_queries"), 0, maximum=20)
    prompt_guidance = str(payload.get("prompt_guidance") or "").strip()[:4000]
    use_llm_generation = retrieval_task == "single_step_retrieval"
    resume_previous_questions = bool(payload.get("resume_previous_questions", True))
    clear_existing_generated = bool(payload.get("clear_existing_generated", False))
    if retrieval_task != "single_step_retrieval":
        resume_previous_questions = False
        prompt_guidance = ""
        num_ctx = 0

    if clear_existing_generated:
        _clear_question_matrix_generated_questions()

    previous_paths = _previous_question_dataset_paths(include_generated=resume_previous_questions) if resume_previous_questions else []
    job_id = f"matrix-generate-{uuid.uuid4().hex[:12]}"
    cmd = [
        sys.executable,
        str(MANUALS_ROOT / "scripts" / "benchmark" / "run_large_retrieval_eval.py"),
        "--existing-corpus-id",
        DEFAULT_CORPUS_ID,
        "--max-queries",
        str(max_questions),
        "--retrieval-task",
        retrieval_task,
        "--search-mode",
        "direct",
        "--response-mode",
        "retrieval_only",
        "--per-query-timeout-seconds",
        str(per_query_timeout),
        "--question-generation-timeout-seconds",
        str(question_generation_timeout),
        "--warmup-queries",
        str(warmup_queries),
        "--generation-chunk-offset",
        str(chunk_offset),
        "--generation-chunk-window",
        str(chunk_window),
        "--questions-per-window",
        str(questions_per_window),
        "--skip-evaluation",
    ]
    if retrieval_task == "multi_step_retrieval":
        cmd.extend(["--multi-step-case-family", case_family])
    if num_ctx:
        cmd.extend(["--question-generation-num-ctx", str(num_ctx)])
    if prompt_guidance:
        cmd.extend(["--question-generation-guidance", prompt_guidance])
    if not use_llm_generation:
        cmd.append("--disable-llm-query-generation")
    for path in previous_paths:
        cmd.extend(["--previous-questions-dataset", str(path)])

    job = {
        "id": job_id,
        "status": "queued",
        "mode": "generate_questions",
        "column": "questions",
        "response_mode": "retrieval_only",
        "use_model_judge": False,
        "generation": {
            "retrieval_task": retrieval_task,
            "multi_step_case_family": case_family,
            "max_questions": max_questions,
            "chunk_offset": chunk_offset,
            "chunk_window": chunk_window,
            "questions_per_window": questions_per_window,
            "num_ctx": num_ctx,
            "question_generation_timeout_seconds": question_generation_timeout,
            "prompt_guidance": prompt_guidance,
            "use_llm_generation": use_llm_generation,
            "resume_previous_questions": resume_previous_questions,
            "previous_question_dataset_count": len(previous_paths),
            "clear_existing_generated": clear_existing_generated,
        },
        "started_at": None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_at": None,
        "dataset_count": 1,
        "completed_datasets": 0,
        "current_dataset": None,
        "current_case_id": None,
        "current_row_key": None,
        "current_question_number": None,
        "current_stage_key": "questions",
        "returncode": None,
        "error": None,
        "cancel_requested": False,
        "live_cells": {},
        "live_results": {},
        "events": [],
        "generated_questions": [],
        "generated_question_count": 0,
        "event_log_path": str(_question_matrix_job_events_path(job_id).relative_to(MANUALS_ROOT)),
        "commands": [{"dataset": "generated_questions", "command": cmd}],
        "outputs": [],
    }
    with MATRIX_JOBS_LOCK:
        MATRIX_JOBS[job_id] = job
        _persist_question_matrix_jobs_locked()
    _record_question_matrix_job_event(job_id, "job_queued", mode="generate_questions", generation=job["generation"])
    thread = Thread(target=_run_question_generation_job, args=(job_id, cmd), daemon=True)
    thread.start()
    return _question_matrix_job_snapshot(job_id)


def _run_question_generation_job(job_id: str, cmd: list[str]) -> None:
    _update_question_matrix_job(job_id, status="running", started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _record_question_matrix_job_event(job_id, "job_started", response_mode="retrieval_only")
    output_lines: list[str] = []
    generated_dataset_path = ""
    try:
        process = subprocess.Popen(
            cmd,
            cwd=MANUALS_ROOT,
            env={
                **os.environ,
                "API_BASE": API_BASE,
                "LOCAL_ADMIN_TOKEN": os.getenv("LOCAL_ADMIN_TOKEN", "admin-token"),
                "LOCAL_END_USER_TOKEN": os.getenv("LOCAL_END_USER_TOKEN", "user-token"),
                "MANUALS_RAG_EVAL_QUESTION_TRACE": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES[job_id] = process
            MATRIX_JOBS[job_id]["pid"] = getattr(process, "pid", None)
            MATRIX_JOBS[job_id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _persist_question_matrix_jobs_locked()
        _record_question_matrix_job_event(job_id, "subprocess_started", pid=getattr(process, "pid", None), command=cmd)
        assert process.stdout is not None
        for line in process.stdout:
            _raise_if_question_matrix_job_cancelled(job_id)
            output_lines.append(line)
            if len(output_lines) > 400:
                output_lines = output_lines[-400:]
            try:
                event_payload = json.loads(line)
            except json.JSONDecodeError:
                event_payload = {}
            if isinstance(event_payload, dict) and str(event_payload.get("event") or "").startswith("question_"):
                _record_question_matrix_job_event(
                    job_id,
                    str(event_payload.get("event")),
                    chunk_id=event_payload.get("chunk_id"),
                    source_filename=event_payload.get("source_filename"),
                    document_title=event_payload.get("document_title"),
                    chunk_title=event_payload.get("chunk_title"),
                    section_path=event_payload.get("section_path"),
                    page_from=event_payload.get("page_from"),
                    page_to=event_payload.get("page_to"),
                    manufacturer=event_payload.get("manufacturer"),
                    product_family=event_payload.get("product_family"),
                    product_model=event_payload.get("product_model"),
                    document_kind=event_payload.get("document_kind"),
                    snippet_chars=event_payload.get("snippet_chars"),
                    parent_context_chars=event_payload.get("parent_context_chars"),
                    context_window_chars=event_payload.get("context_window_chars"),
                    section_context_chars=event_payload.get("section_context_chars"),
                    question=event_payload.get("question"),
                    intent=event_payload.get("intent"),
                    approved=event_payload.get("approved"),
                    category=event_payload.get("category"),
                    feedback=event_payload.get("feedback"),
                    reason=event_payload.get("reason"),
                    answer_in_snippet=event_payload.get("answer_in_snippet"),
                    false_rejection_check=event_payload.get("false_rejection_check"),
                    error=event_payload.get("error"),
                    generated_count=event_payload.get("generated_count"),
                    accepted_count=event_payload.get("accepted_count"),
                    preview=event_payload.get("preview"),
                    fragment=event_payload.get("fragment"),
                    done=event_payload.get("done"),
                    num_ctx=event_payload.get("num_ctx"),
                )
                if event_payload.get("event") == "question_generation_accepted" and event_payload.get("question"):
                    accepted_question = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "question": event_payload.get("question"),
                        "chunk_id": event_payload.get("chunk_id"),
                        "source_filename": event_payload.get("source_filename"),
                        "document_title": event_payload.get("document_title"),
                        "chunk_title": event_payload.get("chunk_title"),
                        "section_path": event_payload.get("section_path"),
                        "page_from": event_payload.get("page_from"),
                        "page_to": event_payload.get("page_to"),
                        "manufacturer": event_payload.get("manufacturer"),
                        "product_family": event_payload.get("product_family"),
                        "product_model": event_payload.get("product_model"),
                        "document_kind": event_payload.get("document_kind"),
                        "intent": event_payload.get("intent"),
                        "review_status": "accepted",
                        "review_label": "ACCEPT",
                        "review_detail": "accepted by generation reviewer",
                    }
                    with MATRIX_JOBS_LOCK:
                        if job_id in MATRIX_JOBS:
                            generated_questions = list(MATRIX_JOBS[job_id].get("generated_questions") or [])
                            generated_questions.append(accepted_question)
                            generated_questions = generated_questions[-500:]
                            MATRIX_JOBS[job_id]["generated_questions"] = generated_questions
                            MATRIX_JOBS[job_id]["generated_question_count"] = len(generated_questions)
                            MATRIX_JOBS[job_id]["updated_at"] = accepted_question["timestamp"]
                            _persist_question_matrix_jobs_locked()
            match = re.search(r'"case_id"\s*:\s*"([^"]+)"', line)
            if match:
                current_case_id = match.group(1)
                _update_question_matrix_job(job_id, current_case_id=current_case_id)
                _record_question_matrix_job_event(job_id, "generated_case_scored", case_id=current_case_id)
            dataset_match = re.search(r'"dataset_path"\s*:\s*"([^"]+)"', line)
            if dataset_match:
                generated_dataset_path = dataset_match.group(1)
                dataset_file = MANUALS_ROOT / generated_dataset_path
                question_count = None
                if dataset_file.exists():
                    try:
                        question_count = len(_read_jsonl(dataset_file))
                    except (OSError, json.JSONDecodeError):
                        question_count = None
                _add_generated_question_dataset_to_view(generated_dataset_path, question_count)
                _update_question_matrix_job(job_id, current_dataset=generated_dataset_path)
                _record_question_matrix_job_event(job_id, "dataset_written", dataset=generated_dataset_path)
        returncode = process.wait(timeout=MATRIX_JOB_TIMEOUT_SECONDS)
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
            if job_id in MATRIX_JOBS:
                MATRIX_JOBS[job_id]["pid"] = None
                _persist_question_matrix_jobs_locked()
        _raise_if_question_matrix_job_cancelled(job_id)
        stdout_tail = "".join(output_lines)[-8000:]
        output = {
            "dataset": generated_dataset_path or "generated_questions",
            "returncode": returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": "",
        }
        if returncode != 0:
            raise RuntimeError(f"Question generation failed with exit code {returncode}")
        _update_question_matrix_job(
            job_id,
            status="completed",
            outputs=[output],
            completed_datasets=1,
            returncode=returncode,
            current_dataset=generated_dataset_path or None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _record_question_matrix_job_event(job_id, "job_completed", dataset=generated_dataset_path)
    except MatrixJobCancelled as error:
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
        _update_question_matrix_job(
            job_id,
            status="cancelled",
            error=str(error),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _record_question_matrix_job_event(job_id, "job_cancelled", error=str(error))
    except Exception as error:
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
        _update_question_matrix_job(
            job_id,
            status="failed",
            outputs=[{"dataset": generated_dataset_path or "generated_questions", "returncode": 1, "stdout_tail": "".join(output_lines)[-8000:], "stderr_tail": ""}],
            error=f"{error.__class__.__name__}: {error}",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _record_question_matrix_job_event(job_id, "job_failed", error=f"{error.__class__.__name__}: {error}")


def _start_question_matrix_job(payload: dict) -> dict:
    _load_question_matrix_jobs_if_needed()
    mode = str(payload.get("mode") or "all_bank")
    column = str(payload.get("column") or "retrieval")
    use_model_judge = bool(payload.get("use_model_judge"))
    stop_on_answer_failure = payload.get("stop_on_answer_failure", True)
    if not isinstance(stop_on_answer_failure, bool):
        raise ValueError("stop_on_answer_failure must be a boolean")
    question_offset = payload.get("question_offset", 0)
    if isinstance(question_offset, bool) or not isinstance(question_offset, int) or question_offset < 0:
        raise ValueError("question_offset must be a non-negative integer")
    if mode not in {"all_bank", "column"}:
        raise ValueError("mode must be all_bank or column")
    valid_columns = {
        "query_classify",
        "filters",
        "dense",
        "sparse",
        "special",
        "fuse",
        "rerank",
        "assemble",
        "metadata",
        "retrieval",
        "relevance",
        "summaries",
        "generation",
        "answer_docs",
        "citations",
        "terms",
        "answer",
    }
    if mode == "column" and column not in valid_columns:
        raise ValueError(f"Unsupported matrix column: {column}")

    manifest = _read_json(TEST_REPORTS_DIR / "retrieval_accuracy_question_bank_manifest.json")
    _, datasets = _visible_question_matrix_datasets(manifest.get("question_bank") or {})
    if not datasets:
        raise ValueError("No visible question-bank datasets were found.")

    job_id = f"matrix-{uuid.uuid4().hex[:12]}"
    response_mode = "answer_with_citations" if mode == "all_bank" or column in ANSWER_STAGE_KEYS else "retrieval_only"
    job = {
        "id": job_id,
        "status": "queued",
        "mode": mode,
        "column": column if mode == "column" else "all",
        "response_mode": response_mode,
        "use_model_judge": use_model_judge and response_mode == "answer_with_citations",
        "stop_on_answer_failure": stop_on_answer_failure and response_mode == "answer_with_citations",
        "question_offset": question_offset,
        "started_at": None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_at": None,
        "dataset_count": len(datasets),
        "current_dataset": None,
        "current_case_id": None,
        "current_row_key": None,
        "current_question_number": None,
        "current_stage_key": "answer" if response_mode == "answer_with_citations" else "retrieval",
        "completed_datasets": 0,
        "returncode": None,
        "error": None,
        "cancel_requested": False,
        "live_cells": {},
        "live_results": {},
        "events": [],
        "event_log_path": str(_question_matrix_job_events_path(job_id).relative_to(MANUALS_ROOT)),
        "commands": [],
        "outputs": [],
    }
    with MATRIX_JOBS_LOCK:
        active_job_id = next(
            (existing_id for existing_id, existing_job in MATRIX_JOBS.items() if existing_job.get("status") in {"queued", "running", "stopping"}),
            None,
        )
        if active_job_id:
            raise ValueError(f"Matrix job {active_job_id} is already running.")
        MATRIX_JOBS[job_id] = job
        _persist_question_matrix_jobs_locked()
    _record_question_matrix_job_event(
        job_id,
        "job_queued",
        mode=mode,
        column=job["column"],
        response_mode=response_mode,
        dataset_count=len(datasets),
        stop_on_answer_failure=job["stop_on_answer_failure"],
    )
    thread = Thread(target=_run_question_matrix_job, args=(job_id, datasets), daemon=True)
    thread.start()
    return _question_matrix_job_snapshot(job_id)


def _run_question_matrix_job(job_id: str, datasets: list[dict]) -> None:
    job = _question_matrix_job_snapshot(job_id)
    _update_question_matrix_job(job_id, status="running", started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _record_question_matrix_job_event(
        job_id,
        "job_started",
        response_mode=job.get("response_mode"),
        stop_on_answer_failure=job.get("stop_on_answer_failure"),
    )
    try:
        for index, dataset in enumerate(datasets, start=1):
            _raise_if_question_matrix_job_cancelled(job_id)
            dataset_rel = str(dataset.get("path") or "")
            dataset_path = MANUALS_ROOT / dataset_rel
            if not dataset_path.exists():
                raise FileNotFoundError(f"Question-bank dataset not found: {dataset_rel}")
            case_numbers = {
                _case_key(case): case_index
                for case_index, case in enumerate(_read_jsonl(dataset_path), start=1)
            }
            _record_question_matrix_job_event(job_id, "dataset_started", dataset=dataset_rel, dataset_index=index, question_count=len(case_numbers))
            if job["response_mode"] == "answer_with_citations":
                _run_answer_matrix_dataset(job_id, dataset_rel, dataset_path, case_numbers, index)
                continue
            cmd = [
                sys.executable,
                str(MANUALS_ROOT / "scripts" / "benchmark" / "run_large_retrieval_eval.py"),
                "--existing-corpus-id",
                DEFAULT_CORPUS_ID,
                "--dataset-path",
                str(dataset_path),
                "--max-queries",
                str(max(1, int(dataset.get("total_questions") or 100000))),
                "--search-mode",
                "http",
                "--response-mode",
                str(job["response_mode"]),
                "--per-query-timeout-seconds",
                "180",
                "--warmup-queries",
                "0",
            ]
            if job.get("use_model_judge"):
                cmd.append("--use-llm-answer-judge")
            command_record = {"dataset": dataset_rel, "command": cmd}
            _update_question_matrix_job(
                job_id,
                current_dataset=dataset_rel,
                current_case_id=None,
                current_question_number=None,
                current_stage_key=str(job["column"]) if job["mode"] == "column" else str(job["current_stage_key"]),
                commands=[*(_question_matrix_job_snapshot(job_id).get("commands") or []), command_record],
            )
            process = subprocess.Popen(
                cmd,
                cwd=MANUALS_ROOT,
                env={
                    **os.environ,
                    "API_BASE": API_BASE,
                    "LOCAL_ADMIN_TOKEN": os.getenv("LOCAL_ADMIN_TOKEN", "admin-token"),
                    "LOCAL_END_USER_TOKEN": os.getenv("LOCAL_END_USER_TOKEN", "user-token"),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with MATRIX_JOBS_LOCK:
                MATRIX_PROCESSES[job_id] = process
                MATRIX_JOBS[job_id]["pid"] = getattr(process, "pid", None)
                MATRIX_JOBS[job_id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _persist_question_matrix_jobs_locked()
            _record_question_matrix_job_event(job_id, "subprocess_started", dataset=dataset_rel, pid=getattr(process, "pid", None), command=cmd)
            output_lines: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                _raise_if_question_matrix_job_cancelled(job_id)
                output_lines.append(line)
                if len(output_lines) > 400:
                    output_lines = output_lines[-400:]
                match = re.search(r'"case_id"\s*:\s*"([^"]+)"', line)
                if match:
                    current_case_id = match.group(1)
                    _update_question_matrix_job(
                        job_id,
                        current_case_id=current_case_id,
                        current_row_key=_matrix_row_key(dataset_rel, current_case_id),
                        current_question_number=case_numbers.get(current_case_id),
                    )
                    _record_question_matrix_job_event(job_id, "case_progress", dataset=dataset_rel, case_id=current_case_id, question_number=case_numbers.get(current_case_id))
            returncode = process.wait(timeout=MATRIX_JOB_TIMEOUT_SECONDS)
            with MATRIX_JOBS_LOCK:
                MATRIX_PROCESSES.pop(job_id, None)
                if job_id in MATRIX_JOBS:
                    MATRIX_JOBS[job_id]["pid"] = None
                    _persist_question_matrix_jobs_locked()
            _raise_if_question_matrix_job_cancelled(job_id)
            stdout_tail = "".join(output_lines)[-8000:]
            output = {
                "dataset": dataset_rel,
                "returncode": returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": "",
            }
            _update_question_matrix_job(
                job_id,
                outputs=[*(_question_matrix_job_snapshot(job_id).get("outputs") or []), output],
                completed_datasets=index,
                returncode=returncode,
            )
            _record_question_matrix_job_event(job_id, "dataset_completed", dataset=dataset_rel, dataset_index=index, returncode=returncode)
            if returncode != 0:
                raise RuntimeError(f"Benchmark failed for {dataset_rel} with exit code {returncode}")
        _update_question_matrix_job(
            job_id,
            status="completed",
            current_dataset=None,
            current_case_id=None,
            current_row_key=None,
            current_question_number=None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _record_question_matrix_job_event(job_id, "job_completed")
    except MatrixJobCancelled as error:
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
        _update_question_matrix_job(
            job_id,
            status="cancelled",
            error=str(error),
            current_dataset=None,
            current_case_id=None,
            current_row_key=None,
            current_question_number=None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _record_question_matrix_job_event(job_id, "job_cancelled", error=str(error))
    except MatrixAnswerFailure as error:
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
        _update_question_matrix_job(
            job_id,
            status="failed",
            error=str(error),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _record_question_matrix_job_event(job_id, "job_stopped_on_answer_failure", error=str(error))
    except Exception as error:
        with MATRIX_JOBS_LOCK:
            MATRIX_PROCESSES.pop(job_id, None)
        _update_question_matrix_job(
            job_id,
            status="failed",
            error=f"{error.__class__.__name__}: {error}",
            current_row_key=None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _record_question_matrix_job_event(job_id, "job_failed", error=f"{error.__class__.__name__}: {error}")


def _run_answer_matrix_dataset(
    job_id: str,
    dataset_rel: str,
    dataset_path: Path,
    case_numbers: dict[str, int],
    dataset_index: int,
) -> None:
    from manuals_rag_evals.retrieval_eval import RetrievalEvalCase
    from manuals_rag_evals.retrieval_eval import score_answer_response
    from manuals_rag_evals.retrieval_eval import score_search_results

    job = _question_matrix_job_snapshot(job_id)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = TEST_REPORTS_DIR / f"retrieval_eval_results_{timestamp}_{uuid.uuid4().hex[:6]}.jsonl"
    manifest_path = TEST_REPORTS_DIR / f"retrieval_eval_manifest_{timestamp}_{uuid.uuid4().hex[:6]}.json"
    all_cases = _read_jsonl(dataset_path)
    question_offset = min(int(job.get("question_offset") or 0), len(all_cases))
    cases = all_cases[question_offset:]
    results: list[dict] = []

    def write_manifest(*, status: str, completed: bool, failure: dict | None = None) -> None:
        manifest = {
            "corpus_id": DEFAULT_CORPUS_ID,
            "input_dataset_path": str(dataset_path),
            "search_mode": "http_debug_stream",
            "response_mode": "answer_with_citations",
            "use_llm_answer_judge": bool(job.get("use_model_judge")),
            "stop_on_answer_failure": bool(job.get("stop_on_answer_failure")),
            "question_offset": question_offset,
            "status": status,
            "completed": completed,
            "processed_questions": len(results),
            "total_questions": len(all_cases),
            "results_path": str(results_path),
        }
        if failure:
            manifest["failure"] = failure
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    _update_question_matrix_job(
        job_id,
        current_dataset=dataset_rel,
        current_case_id=None,
        current_row_key=None,
        current_question_number=1 if cases else None,
        current_stage_key="query_classify",
    )
    with results_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            _raise_if_question_matrix_job_cancelled(job_id)
            case_id = _case_key(case)
            matrix_row_key = _matrix_row_key(dataset_rel, case_id)
            eval_case = RetrievalEvalCase(**case)
            _record_question_matrix_job_event(
                job_id,
                "case_started",
                dataset=dataset_rel,
                case_id=case_id,
                question_number=case_numbers.get(case_id),
                query=case.get("query"),
            )
            _update_question_matrix_job(
                job_id,
                current_case_id=case_id,
                current_row_key=matrix_row_key,
                current_question_number=case_numbers.get(case_id),
                current_stage_key="query_classify",
            )
            debug_result, early_evaluation = _run_query_debug_stream(
                job_id,
                case_id,
                case_numbers.get(case_id),
                case["query"],
                eval_case=eval_case,
                matrix_row_key=matrix_row_key,
            )
            _raise_if_question_matrix_job_cancelled(job_id)
            top_results = _debug_top_results(debug_result)
            evaluation = early_evaluation or debug_result.get("matrix_retrieval_evaluation") or score_search_results(eval_case, top_results)
            answer: dict = {}
            answer_evaluation: dict = {}
            if evaluation.get("passed"):
                answer = dict(debug_result.get("answer") or {})
                for stage_key in ("answer_docs", "citations", "terms", "answer"):
                    _update_question_matrix_job(job_id, current_stage_key=stage_key)
                answer_evaluation = score_answer_response(
                    eval_case,
                    answer,
                    evaluation,
                    _debug_answer_evidence_results(debug_result, top_results),
                    use_llm_required_info_judge=bool(job.get("use_model_judge")),
                )
                _record_question_matrix_job_event(
                    job_id,
                    "answer_scored",
                    dataset=dataset_rel,
                    case_id=case_id,
                    question_number=case_numbers.get(case_id),
                    answer_passed=bool(answer_evaluation.get("passed")),
                    failure_reasons=answer_evaluation.get("failure_reasons") or [],
                    answer=answer,
                    answer_evaluation=answer_evaluation,
                )
            else:
                _record_question_matrix_job_event(
                    job_id,
                    "answer_blocked",
                    dataset=dataset_rel,
                    case_id=case_id,
                    question_number=case_numbers.get(case_id),
                    failure_category=evaluation.get("failure_category"),
                    failure_reasons=evaluation.get("failure_reasons") or [],
                )
            record = {
                "dataset": dataset_rel,
                "case": case,
                "evaluation": evaluation,
                "top_results": top_results[:5],
                "answer": answer,
                "answer_evaluation": answer_evaluation,
                "query_debug_result": debug_result,
            }
            _update_question_matrix_live_result(job_id, matrix_row_key, record)
            _replace_question_matrix_live_cells(job_id, matrix_row_key, _build_row_cells(record))
            results.append(record)
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            _record_question_matrix_job_event(
                job_id,
                "case_completed",
                dataset=dataset_rel,
                case_id=case_id,
                question_number=case_numbers.get(case_id),
                retrieval_passed=bool(evaluation.get("passed")),
                answer_passed=bool(answer_evaluation.get("passed")) if answer_evaluation else None,
            )
            retrieval_failed = not evaluation.get("passed")
            answer_failed = bool(answer_evaluation and not answer_evaluation.get("passed"))
            if job.get("stop_on_answer_failure") and (retrieval_failed or answer_failed):
                failure_reasons = (
                    (answer_evaluation or {}).get("failure_reasons")
                    or evaluation.get("failure_reasons")
                    or [evaluation.get("failure_category") or "retrieval_failed_before_answer"]
                )
                failure = {
                    "case_id": case_id,
                    "question_number": case_numbers.get(case_id),
                    "query": case.get("query"),
                    "failure_reasons": failure_reasons,
                }
                write_manifest(status="stopped_on_answer_failure", completed=False, failure=failure)
                output = {
                    "dataset": dataset_rel,
                    "returncode": 1,
                    "stdout_tail": json.dumps(
                        {
                            "results": len(results),
                            "results_path": str(results_path),
                            "manifest_path": str(manifest_path),
                            "stopped_on_answer_failure": failure,
                        }
                    ),
                    "stderr_tail": "",
                    "partial": True,
                }
                _update_question_matrix_job(
                    job_id,
                    outputs=[*(_question_matrix_job_snapshot(job_id).get("outputs") or []), output],
                    returncode=1,
                )
                _record_question_matrix_job_event(
                    job_id,
                    "answer_failure_stop",
                    dataset=dataset_rel,
                    results_path=str(results_path),
                    manifest_path=str(manifest_path),
                    **failure,
                )
                reason_text = ", ".join(failure["failure_reasons"]) or "answer evaluation failed"
                raise MatrixAnswerFailure(
                    f"Stopped on answer failure at question {failure['question_number']} "
                    f"({case_id}): {reason_text}"
                )
    write_manifest(status="completed", completed=True)
    _update_question_matrix_job(
        job_id,
        outputs=[
            *(_question_matrix_job_snapshot(job_id).get("outputs") or []),
            {"dataset": dataset_rel, "returncode": 0, "stdout_tail": json.dumps({"results": len(results), "results_path": str(results_path)}), "stderr_tail": ""},
        ],
        completed_datasets=dataset_index,
        returncode=0,
    )
    _record_question_matrix_job_event(job_id, "dataset_completed", dataset=dataset_rel, dataset_index=dataset_index, returncode=0, results_path=str(results_path))


def _run_query_debug_stream(
    job_id: str,
    case_id: str,
    question_number: int | None,
    query: str,
    *,
    eval_case: object,
    matrix_row_key: str | None = None,
) -> tuple[dict, dict | None]:
    from manuals_rag_evals.retrieval_eval import score_search_results

    eval_case_dict = eval_case.to_dict() if hasattr(eval_case, "to_dict") else dict(getattr(eval_case, "__dict__", {}))
    request_payload = {
        "query": query,
        "corpus_ids": [DEFAULT_CORPUS_ID],
        "filters": {},
        "response_mode": "answer_with_citations",
    }
    request = Request(
        f"{API_BASE}/debug/query-stream?sample_limit=100",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.getenv('LOCAL_ADMIN_TOKEN', 'admin-token')}", "Content-Type": "application/json"},
        method="POST",
    )
    completed_steps: list[str] = []
    step_timings_ms: dict = {}
    stages: list[dict] = []
    live_key = matrix_row_key or case_id
    with urlopen(request, timeout=900) as response:
        for raw_line in response:
            _raise_if_question_matrix_job_cancelled(job_id)
            if not raw_line.strip():
                continue
            event = json.loads(raw_line.decode("utf-8"))
            step = event.get("step")
            if event.get("event") == "step_started" and step in DEBUG_STEP_TO_MATRIX_KEY:
                _update_question_matrix_job(
                    job_id,
                    current_case_id=case_id,
                    current_question_number=question_number,
                    current_stage_key=DEBUG_STEP_TO_MATRIX_KEY[step],
                )
                _record_question_matrix_job_event(
                    job_id,
                    "step_started",
                    case_id=case_id,
                    question_number=question_number,
                    step=step,
                    matrix_key=DEBUG_STEP_TO_MATRIX_KEY[step],
                )
            if event.get("event") == "step_completed" and step in DEBUG_STEP_TO_MATRIX_KEY:
                completed_steps = list(event.get("completed_steps") or completed_steps)
                step_timings_ms = dict(event.get("step_timings_ms") or step_timings_ms)
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                stages.append(
                    {
                        "name": step,
                        "label": event.get("label"),
                        "duration_ms": event.get("duration_ms"),
                        "samples": payload.get("samples") or payload.get("summaries") or [],
                    }
                )
                live_cells = _build_row_cells(
                    {
                        "case": eval_case_dict,
                        "query_debug_result": {
                            "completed_steps": completed_steps,
                            "step_timings_ms": step_timings_ms,
                            "stages": stages,
                        },
                    }
                )
                live_cell = live_cells.get(DEBUG_STEP_TO_MATRIX_KEY[step], _matrix_cell("pass", "debug step completed", "DONE"))
                _update_question_matrix_live_cell(
                    job_id,
                    live_key,
                    DEBUG_STEP_TO_MATRIX_KEY[step],
                    live_cell.get("status", "pass"),
                    live_cell.get("detail", "debug step completed"),
                    live_cell.get("label"),
                )
                _record_question_matrix_job_event(
                    job_id,
                    "step_completed",
                    case_id=case_id,
                    question_number=question_number,
                    step=step,
                    matrix_key=DEBUG_STEP_TO_MATRIX_KEY[step],
                    status=live_cell.get("status", "pass"),
                    label=live_cell.get("label"),
                    detail=live_cell.get("detail"),
                    sample_count=len((payload.get("samples") or payload.get("summaries") or []) if isinstance(payload, dict) else []),
                    duration_ms=event.get("duration_ms"),
                )
                if step == "assemble_context":
                    top_results = _top_results_from_assemble_payload(payload)
                    evaluation = score_search_results(eval_case, top_results)
                    _update_question_matrix_job(job_id, current_stage_key="retrieval")
                    partial_debug_result = {
                        "query": query,
                        "corpus_ids": [DEFAULT_CORPUS_ID],
                        "completed_steps": completed_steps,
                        "step_timings_ms": step_timings_ms,
                        "stages": stages,
                        "answer": {},
                    }
                    retrieval_cell = _build_row_cells(
                        {
                            "case": eval_case_dict,
                            "query_debug_result": partial_debug_result,
                            "evaluation": evaluation,
                        }
                    )["retrieval"]
                    _update_question_matrix_live_cell(
                        job_id,
                        live_key,
                        "retrieval",
                        retrieval_cell.get("status", "blank"),
                        retrieval_cell.get("detail", "retrieval scored"),
                        retrieval_cell.get("label"),
                    )
                    _record_question_matrix_job_event(
                        job_id,
                        "retrieval_scored",
                        case_id=case_id,
                        question_number=question_number,
                        status=retrieval_cell.get("status"),
                        label=retrieval_cell.get("label"),
                        detail=retrieval_cell.get("detail"),
                        evaluation_passed=bool(evaluation.get("passed")),
                        failure_category=evaluation.get("failure_category"),
                    )
                    matrix_evaluation = _matrix_retrieval_evaluation(evaluation, retrieval_cell)
                    retrieval_passed = bool(matrix_evaluation.get("passed"))
                    partial_debug_result["matrix_retrieval_evaluation"] = matrix_evaluation
                    partial_debug_result["early_stopped"] = not retrieval_passed
                    partial_debug_result["early_stop_reason"] = (
                        None
                        if retrieval_passed
                        else matrix_evaluation.get("failure_category") or retrieval_cell.get("detail") or "retrieval_failed"
                    )
                    if not retrieval_passed:
                        return partial_debug_result, matrix_evaluation
            if event.get("event") == "run_completed":
                result = dict(event.get("result") or {})
                if completed_steps:
                    result["completed_steps"] = completed_steps
                if step_timings_ms:
                    result["step_timings_ms"] = step_timings_ms
                if stages:
                    result["stages"] = stages
                if "matrix_retrieval_evaluation" not in result and "matrix_evaluation" in locals():
                    result["matrix_retrieval_evaluation"] = matrix_evaluation
                _record_question_matrix_job_event(
                    job_id,
                    "query_stream_completed",
                    case_id=case_id,
                    question_number=question_number,
                    completed_steps=completed_steps,
                    stage_count=len(stages),
                )
                return result, None
            if event.get("event") == "run_failed":
                _record_question_matrix_job_event(
                    job_id,
                    "query_stream_failed",
                    case_id=case_id,
                    question_number=question_number,
                    error=str(event.get("error") or "debug query run failed"),
                )
                raise RuntimeError(str(event.get("error") or "debug query run failed"))
    raise RuntimeError("debug query stream ended without run_completed")


def _debug_top_results(debug_result: dict) -> list[dict]:
    for stage in debug_result.get("stages") or []:
        if stage.get("name") == "retrieval_results":
            return list(stage.get("samples") or [])
        if stage.get("name") == "assemble_context":
            return list(stage.get("samples") or [])
    return []


def _debug_answer_evidence_results(debug_result: dict, top_results: list[dict]) -> list[dict]:
    """Return every concrete chunk the answer generator was allowed to cite.

    Matrix retrieval scoring intentionally uses the bounded final-context preview,
    but answer scoring must also resolve citations from the serialized answer-input
    set. The answer workflow can expand a table row after context assembly, so a
    valid citation may not be one of the few samples displayed for that stage.
    """
    merged: list[dict] = []
    seen: set[str] = set()
    answer_inputs = debug_result.get("answer_generation_inputs") or {}
    for item in [*top_results, *(answer_inputs.get("samples") or [])]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("chunk_id") or item.get("id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _top_results_from_assemble_payload(payload: dict) -> list[dict]:
    return list(payload.get("samples") or [])


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8601), ManualsRagUiHandler)
    print(f"Serving Manuals RAG UI on http://0.0.0.0:8601 with API proxy {API_BASE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
