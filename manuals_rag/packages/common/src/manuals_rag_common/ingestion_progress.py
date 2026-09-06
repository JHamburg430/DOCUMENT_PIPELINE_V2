from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from manuals_rag_common.db import execute, json_dumps


INGESTION_STEPS: tuple[tuple[str, str], ...] = (
    ("upload", "Upload source"),
    ("load_source", "Load source file"),
    ("parse", "Parse document"),
    ("normalize", "Normalize content"),
    ("metadata", "Extract metadata"),
    ("chunk", "Build retrieval chunks"),
    ("assets", "Render page and table assets"),
    ("persist", "Persist nodes and chunks"),
    ("index_chunks", "Index retrieval chunks"),
    ("index_metadata", "Index document metadata"),
    ("complete", "Finalize ingestion"),
)


def ensure_ingestion_step_table() -> None:
    execute(
        """
        create table if not exists ingestion_run_steps (
            run_id uuid not null references ingestion_runs(id) on delete cascade,
            step_key text not null,
            sequence integer not null,
            label text not null,
            status text not null default 'queued',
            started_at timestamptz,
            completed_at timestamptz,
            duration_ms double precision,
            detail_json jsonb not null default '{}'::jsonb,
            error text,
            primary key (run_id, step_key)
        )
        """
    )


def initialize_ingestion_steps(run_id: str, *, upload_details: dict[str, Any] | None = None) -> None:
    ensure_ingestion_step_table()
    for sequence, (step_key, label) in enumerate(INGESTION_STEPS, start=1):
        completed = step_key == "upload"
        execute(
            """
            insert into ingestion_run_steps (
                run_id, step_key, sequence, label, status, started_at, completed_at,
                duration_ms, detail_json, error
            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, null)
            on conflict (run_id, step_key) do nothing
            """,
            (
                run_id,
                step_key,
                sequence,
                label,
                "completed" if completed else "queued",
                datetime.now(timezone.utc) if completed else None,
                datetime.now(timezone.utc) if completed else None,
                0.0 if completed else None,
                json_dumps(upload_details or {}) if completed else "{}",
            ),
        )


def start_ingestion_step(run_id: str, step_key: str, *, details: dict[str, Any] | None = None) -> None:
    execute(
        """
        update ingestion_run_steps
        set status = 'running', started_at = now(), completed_at = null,
            duration_ms = null, detail_json = %s::jsonb, error = null
        where run_id = %s and step_key = %s
        """,
        (json_dumps(details or {}), run_id, step_key),
    )


def complete_ingestion_step(run_id: str, step_key: str, *, details: dict[str, Any] | None = None) -> None:
    execute(
        """
        update ingestion_run_steps
        set status = 'completed', completed_at = now(),
            duration_ms = greatest(0, extract(epoch from (now() - coalesce(started_at, now()))) * 1000),
            detail_json = %s::jsonb, error = null
        where run_id = %s and step_key = %s
        """,
        (json_dumps(details or {}), run_id, step_key),
    )


def fail_ingestion_step(run_id: str, step_key: str, error: str) -> None:
    execute(
        """
        update ingestion_run_steps
        set status = 'failed', completed_at = now(),
            duration_ms = greatest(0, extract(epoch from (now() - coalesce(started_at, now()))) * 1000),
            error = %s
        where run_id = %s and step_key = %s
        """,
        (error, run_id, step_key),
    )
    execute(
        """
        update ingestion_run_steps
        set status = 'skipped', completed_at = now(),
            detail_json = jsonb_build_object('reason', 'Blocked by failed step', 'failed_step', %s)
        where run_id = %s
          and sequence > (select sequence from ingestion_run_steps where run_id = %s and step_key = %s)
          and status = 'queued'
        """,
        (step_key, run_id, run_id, step_key),
    )
