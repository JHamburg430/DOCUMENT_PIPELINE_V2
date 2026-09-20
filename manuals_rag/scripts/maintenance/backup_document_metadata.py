#!/usr/bin/env python3
"""Create an immutable, targeted PostgreSQL + Qdrant metadata rollback bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from manuals_rag_common.config import settings
from manuals_rag_common.db import fetch_all
from manuals_rag_retrieval.qdrant_store import collection_name, document_metadata_collection_name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point_json(point: Any) -> dict[str, Any]:
    if hasattr(point, "model_dump"):
        return point.model_dump(mode="json")
    if hasattr(point, "dict"):
        return point.dict()
    raise TypeError(f"unsupported Qdrant point type: {type(point)!r}")


def _scroll_document_points(
    client: QdrantClient,
    *,
    collection: str,
    document_id: str,
) -> list[dict[str, Any]]:
    if not client.collection_exists(collection):
        return []
    points: list[dict[str, Any]] = []
    offset: Any | None = None
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source_document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )
    while True:
        page, offset = client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(_point_json(point) for point in page)
        if offset is None:
            break
    return points


def build_backup(document_ids: list[str], *, source_reports: list[Path]) -> dict[str, Any]:
    placeholders = ",".join(["%s"] * len(document_ids))
    source_rows = fetch_all(
        f"""
        select id, corpus_id, current_version_id, source_filename, title, manufacturer,
               product_family, product_model, document_kind, ingest_status, updated_at
        from source_documents
        where id in ({placeholders})
        order by id
        """,
        tuple(document_ids),
    )
    found = {str(row["id"]) for row in source_rows}
    missing = sorted(set(document_ids) - found)
    if missing:
        raise ValueError(f"source document(s) not found: {', '.join(missing)}")

    client = QdrantClient(url=settings.qdrant_url, timeout=60)
    documents: list[dict[str, Any]] = []
    for source in source_rows:
        document_id = str(source["id"])
        version_id = str(source["current_version_id"])
        corpus_id = str(source["corpus_id"])
        extraction = fetch_all(
            """
            select source_document_id, document_version_id, model, metadata_json, extracted_at
            from document_metadata_extractions where source_document_id = %s
            """,
            (document_id,),
        )
        chunks = fetch_all(
            """
            select id, document_version_id, metadata_json
            from retrieval_chunks where document_version_id = %s order by id
            """,
            (version_id,),
        )
        qdrant_chunks = _scroll_document_points(
            client,
            collection=collection_name(corpus_id),
            document_id=document_id,
        )
        qdrant_selector = _scroll_document_points(
            client,
            collection=document_metadata_collection_name(corpus_id),
            document_id=document_id,
        )
        active_chunk_ids = {
            str(point.get("payload", {}).get("chunk_id"))
            for point in qdrant_chunks
            if point.get("payload", {}).get("chunk_id")
        }
        if len(active_chunk_ids) != len(qdrant_chunks):
            raise ValueError(f"Qdrant chunk backup contains duplicate or missing chunk ids for {document_id}")
        documents.append(
            {
                "source_document": source,
                "extraction": extraction,
                "chunks": chunks,
                "qdrant_chunks": qdrant_chunks,
                "qdrant_selector": qdrant_selector,
                "verification": {
                    "postgres_chunk_count": len(chunks),
                    "qdrant_active_chunk_count": len(qdrant_chunks),
                    "qdrant_selector_count": len(qdrant_selector),
                },
            }
        )
    return {
        "schema": "manuals_rag_targeted_metadata_backup_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_reports
        ],
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", action="append", required=True, dest="document_ids")
    parser.add_argument("--source-report", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing backup: {args.output}")
    for report in args.source_report:
        if not report.is_file():
            parser.error(f"source report does not exist: {report}")
    try:
        payload = build_backup(args.document_ids, source_reports=args.source_report)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": _sha256(args.output),
        "document_count": len(payload["documents"]),
        "verification": [item["verification"] for item in payload["documents"]],
    }, indent=2))


if __name__ == "__main__":
    main()
