"""PostgreSQL source of truth for Agent repair cases and regression evidence."""

from __future__ import annotations

import re
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


RUN_ID_PATTERN = re.compile(r"agent-run-[a-f0-9]{12}\Z")
REGRESSION_KEYS = {
    "required_answer_terms", "required_citation_document_ids",
    "required_event_types", "expect_abstention", "replay_query",
}


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("A valid Agent run ID is required.")
    return run_id


def _public_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "source_run_id": row["source_run_id"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat(),
        "latest_fix": row.get("latest_fix"),
        "latest_regression": row.get("latest_regression"),
    }


def get_case(dsn: str, run_id: str, *, include_snapshot: bool = False) -> dict[str, Any] | None:
    validate_run_id(run_id)
    snapshot_column = ", c.run_snapshot" if include_snapshot else ""
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        row = conn.execute(
            f"""select c.request_id, c.source_run_id, c.status, c.created_at{snapshot_column},
                (select jsonb_build_object('id', f.id, 'commit_sha', f.commit_sha,
                    'fix_summary', f.fix_summary, 'created_at', f.created_at)
                 from agent_repair_fixes f where f.request_id = c.request_id
                 order by f.created_at desc, f.id desc limit 1) as latest_fix,
                (select jsonb_build_object('status', r.status, 'source_revision', r.source_revision,
                    'created_at', r.created_at)
                 from agent_repair_regression_runs r
                 join agent_repair_fixes f on f.id = r.fix_id
                 where f.request_id = c.request_id
                 order by r.created_at desc, r.id desc limit 1) as latest_regression
                from agent_repair_cases c where c.source_run_id = %s""",
            (run_id,),
        ).fetchone()
    if not row:
        return None
    result = _public_case(row)
    if include_snapshot:
        result["run"] = row["run_snapshot"]
    return result


def save_case(dsn: str, job: dict[str, Any]) -> dict[str, Any]:
    run_id = validate_run_id(str(job.get("id") or ""))
    if job.get("surface") != "chat":
        raise ValueError("Only Agent page runs can be sent for repair.")
    if job.get("status") in {"queued", "running"}:
        raise ValueError("Wait for the Agent run to finish before sending it for repair.")
    request_id = f"repair-{run_id}"
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        conn.execute(
            """insert into agent_repair_cases (request_id, source_run_id, run_snapshot)
               values (%s, %s, %s) on conflict (source_run_id) do nothing""",
            (request_id, run_id, Jsonb(job)),
        )
    saved = get_case(dsn, run_id)
    if saved is None:
        raise RuntimeError("Inserted Agent repair case was not readable.")
    return saved


def validate_regression_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict) or not spec or set(spec) - REGRESSION_KEYS:
        raise ValueError("Regression spec needs supported, measurable assertions.")
    if "expect_abstention" not in spec or not any(key in spec for key in
            ("required_answer_terms", "required_citation_document_ids", "required_event_types")):
        raise ValueError("Regression spec needs expect_abstention and at least one outcome assertion.")
    for key in ("required_answer_terms", "required_citation_document_ids", "required_event_types"):
        if key in spec and (not isinstance(spec[key], list) or not spec[key] or
                            any(not isinstance(item, str) or not item.strip() for item in spec[key])):
            raise ValueError(f"{key} must be a nonempty list of strings.")
    if "expect_abstention" in spec and not isinstance(spec["expect_abstention"], bool):
        raise ValueError("expect_abstention must be boolean.")
    if "replay_query" in spec and (not isinstance(spec["replay_query"], str) or not spec["replay_query"].strip()):
        raise ValueError("replay_query must be a nonempty string.")
    return spec


def record_fix(
    dsn: str, *, request_id: str, root_cause: str, fix_summary: str,
    commit_sha: str, regression_spec: dict[str, Any], verification: dict[str, Any],
) -> str:
    if not all(str(value).strip() for value in (request_id, root_cause, fix_summary, commit_sha)):
        raise ValueError("Case, root cause, fix summary, and commit SHA are required.")
    validate_regression_spec(regression_spec)
    if not isinstance(verification, dict) or not verification:
        raise ValueError("Verification evidence is required before recording a fix.")
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        row = conn.execute(
            """insert into agent_repair_fixes
               (request_id, root_cause, fix_summary, commit_sha, regression_spec, verification_json)
               values (%s, %s, %s, %s, %s, %s) returning id""",
            (request_id, root_cause, fix_summary, commit_sha,
             Jsonb(regression_spec), Jsonb(verification)),
        ).fetchone()
        conn.execute(
            "update agent_repair_cases set status = 'fixed', updated_at = now() where request_id = %s",
            (request_id,),
        )
    return str(row["id"])


def regression_cases(dsn: str) -> list[dict[str, Any]]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        return list(conn.execute(
            """select distinct on (c.request_id) c.request_id, c.source_run_id,
                c.run_snapshot, f.id as fix_id, f.regression_spec, f.commit_sha
                from agent_repair_cases c join agent_repair_fixes f on f.request_id = c.request_id
                where c.status <> 'closed'
                order by c.request_id, f.created_at desc, f.id desc"""
        ).fetchall())


def list_cases(dsn: str, *, limit: int = 100) -> list[dict[str, Any]]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        rows = conn.execute(
            """select c.request_id, c.source_run_id, c.status, c.created_at,
                (select f.commit_sha from agent_repair_fixes f where f.request_id = c.request_id
                 order by f.created_at desc, f.id desc limit 1) as latest_fix_commit
                from agent_repair_cases c order by c.created_at desc limit %s""",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [{**row, "created_at": row["created_at"].isoformat()} for row in rows]


def record_regression(dsn: str, *, fix_id: str, source_revision: str,
                      status: str, result: dict[str, Any], error: str | None = None) -> None:
    if status not in {"passed", "failed", "blocked"}:
        raise ValueError("Invalid regression status.")
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """insert into agent_repair_regression_runs
               (fix_id, source_revision, status, result_json, error)
               values (%s, %s, %s, %s, %s) returning fix_id""",
            (fix_id, source_revision, status, Jsonb(result), error),
        ).fetchone()
        if row:
            conn.execute(
                """update agent_repair_cases c set status = %s, updated_at = now()
                   from agent_repair_fixes f where f.id = %s and f.request_id = c.request_id
                     and f.id = (select newest.id from agent_repair_fixes newest
                                 where newest.request_id = c.request_id
                                 order by newest.created_at desc, newest.id desc limit 1)""",
                ("verified" if status == "passed" else "fixed", fix_id),
            )
