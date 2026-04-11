from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from manuals_rag_common.config import settings
from manuals_rag_common.db import execute, fetch_all, json_dumps
from manuals_rag_common.queue import enqueue
from manuals_rag_parsers.metadata import infer_document_metadata


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "test_reports"


@dataclass
class BackfillResult:
    document_id: str
    version_id: str
    source_filename: str
    status: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    chunk_count: int = 0
    embed_enqueued: bool = False


def _ensure_metadata_table() -> None:
    execute(
        """
        create table if not exists document_metadata_extractions (
            source_document_id uuid primary key references source_documents(id) on delete cascade,
            document_version_id uuid not null references document_versions(id) on delete cascade,
            model text not null,
            metadata_json jsonb not null default '{}'::jsonb,
            extracted_at timestamptz not null default now()
        )
        """
    )


def _documents(limit: int | None = None) -> list[dict[str, Any]]:
    query = """
        select
            sd.id as document_id,
            sd.source_filename,
            sd.current_version_id as version_id
        from source_documents sd
        join document_versions dv on dv.id = sd.current_version_id
        where exists (
            select 1
            from logical_nodes ln
            where ln.document_version_id = dv.id
        )
        order by sd.updated_at desc, sd.id
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " limit %s"
        params = (limit,)
    return fetch_all(query, params)


def _document_excerpt(version_id: str, *, node_limit: int) -> str:
    rows = fetch_all(
        """
        select text_normalized, text_raw
        from logical_nodes
        where document_version_id = %s
          and coalesce(text_normalized, text_raw, '') <> ''
        order by ordinal
        limit %s
        """,
        (version_id, node_limit),
    )
    return "\n\n".join(str(row["text_normalized"] or row["text_raw"] or "") for row in rows)


def _metadata_payload(metadata: Any) -> dict[str, Any]:
    payload = metadata.__dict__.copy()
    payload["document_kind"] = metadata.document_kind.value
    payload["revision_date"] = metadata.revision_date.isoformat() if metadata.revision_date else None
    payload["effective_date"] = metadata.effective_date.isoformat() if metadata.effective_date else None
    return payload


def _chunk_metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_kind": metadata["document_kind"],
        "manufacturer": metadata["manufacturer"],
        "companies": metadata["companies"],
        "product_family": metadata["product_family"],
        "product_model": metadata["product_model"],
        "product_families": metadata["product_families"],
        "product_models": metadata["product_models"],
        "devices": metadata["devices"],
        "part_numbers": metadata["part_numbers"],
        "document_protocol_terms": metadata["protocol_terms"],
        "settings": metadata["settings"],
        "parameters": metadata["parameters"],
        "document_menu_labels": metadata["menu_labels"],
        "document_topics": metadata["document_topics"],
        "revision_date": metadata["revision_date"],
    }


def _apply_metadata(document: dict[str, Any], metadata: dict[str, Any], *, enqueue_embed: bool) -> tuple[int, bool]:
    execute(
        """
        insert into document_metadata_extractions (
            source_document_id, document_version_id, model, metadata_json, extracted_at
        ) values (%s, %s, %s, %s::jsonb, now())
        on conflict (source_document_id) do update
        set document_version_id = excluded.document_version_id,
            model = excluded.model,
            metadata_json = excluded.metadata_json,
            extracted_at = excluded.extracted_at
        """,
        (
            document["document_id"],
            document["version_id"],
            settings.ollama_metadata_model,
            json_dumps(metadata),
        ),
    )
    execute(
        """
        update source_documents
        set manufacturer = %s,
            product_family = %s,
            product_model = %s,
            document_kind = %s,
            title = coalesce(nullif(%s, ''), title),
            updated_at = now()
        where id = %s
        """,
        (
            metadata["manufacturer"],
            metadata["product_family"],
            metadata["product_model"],
            metadata["document_kind"],
            metadata["title"],
            document["document_id"],
        ),
    )
    chunk_metadata = _chunk_metadata_payload(metadata)
    execute(
        """
        update retrieval_chunks
        set metadata_json = metadata_json || %s::jsonb
        where document_version_id = %s
        """,
        (json_dumps(chunk_metadata), document["version_id"]),
    )
    chunk_count_row = fetch_all(
        "select count(*) as count from retrieval_chunks where document_version_id = %s",
        (document["version_id"],),
    )
    chunk_count = int(chunk_count_row[0]["count"]) if chunk_count_row else 0
    if enqueue_embed:
        run_id = str(uuid4())
        execute(
            """
            insert into ingestion_runs (id, source_document_id, document_version_id, status, failure_class, created_at, updated_at)
            values (%s, %s, %s, 'parsed', null, now(), now())
            """,
            (run_id, document["document_id"], document["version_id"]),
        )
        enqueue(
            "embed_jobs",
            {"run_id": run_id, "document_id": str(document["document_id"]), "version_id": str(document["version_id"])},
        )
        return chunk_count, True
    return chunk_count, False


def _write_report(results: list[BackfillResult]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"document_metadata_backfill_{timestamp}.json"
    path.write_text(json.dumps([asdict(result) for result in results], indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill model-extracted document metadata.")
    parser.add_argument("--apply", action="store_true", help="Persist metadata changes. Without this, only reports extracted metadata.")
    parser.add_argument("--no-enqueue-embed", action="store_true", help="Do not enqueue embed jobs after updating chunk metadata.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents.")
    parser.add_argument("--node-limit", type=int, default=20, help="Logical nodes to include in each metadata excerpt.")
    args = parser.parse_args()

    _ensure_metadata_table()
    results: list[BackfillResult] = []
    for document in _documents(limit=args.limit):
        try:
            excerpt = _document_excerpt(str(document["version_id"]), node_limit=args.node_limit)
            metadata = _metadata_payload(infer_document_metadata(str(document["source_filename"]), excerpt))
            chunk_count = 0
            embed_enqueued = False
            if args.apply:
                chunk_count, embed_enqueued = _apply_metadata(
                    document,
                    metadata,
                    enqueue_embed=not args.no_enqueue_embed,
                )
            results.append(
                BackfillResult(
                    document_id=str(document["document_id"]),
                    version_id=str(document["version_id"]),
                    source_filename=str(document["source_filename"]),
                    status="applied" if args.apply else "planned",
                    metadata=metadata,
                    chunk_count=chunk_count,
                    embed_enqueued=embed_enqueued,
                )
            )
            print(json.dumps({"document_id": str(document["document_id"]), "status": results[-1].status, "metadata": metadata}, default=str))
        except Exception as exc:
            results.append(
                BackfillResult(
                    document_id=str(document["document_id"]),
                    version_id=str(document["version_id"]),
                    source_filename=str(document["source_filename"]),
                    status="failed",
                    error=str(exc),
                )
            )
            print(json.dumps({"document_id": str(document["document_id"]), "status": "failed", "error": str(exc)}))

    report = _write_report(results)
    summary = {
        "report": str(report),
        "total": len(results),
        "applied": sum(1 for result in results if result.status == "applied"),
        "planned": sum(1 for result in results if result.status == "planned"),
        "failed": sum(1 for result in results if result.status == "failed"),
        "embed_enqueued": sum(1 for result in results if result.embed_enqueued),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
