from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
import psycopg.rows

from manuals_rag_common.config import settings


@contextmanager
def get_db() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(settings.postgres_dsn, row_factory=psycopg.rows.dict_row)
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None


def execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()


def execute_many(query: str, params: list[tuple[Any, ...]]) -> None:
    if not params:
        return
    with get_db() as conn, conn.cursor() as cur:
        cur.executemany(query, params)
        conn.commit()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)
