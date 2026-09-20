#!/usr/bin/env python3
"""Verify one document's PostgreSQL chunk set and metadata match Qdrant payloads."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

from manuals_rag_common.config import settings
from manuals_rag_common.db import fetch_all
from manuals_rag_parsers.metadata import METADATA_PIPELINE_VERSION
from manuals_rag_retrieval.qdrant_store import collection_name, document_metadata_collection_name


def _scroll(client: QdrantClient, collection: str, document_id: str) -> list[object]:
    points: list[object] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="source_document_id", match=models.MatchValue(value=document_id)
            )]),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            return points


def audit(document_id: str) -> dict[str, object]:
    rows = fetch_all(
        """
        select sd.corpus_id, sd.current_version_id, rc.id as chunk_id, rc.metadata_json
        from source_documents sd
        left join retrieval_chunks rc on rc.document_version_id = sd.current_version_id and rc.is_active = true
        where sd.id = %s order by rc.id
        """,
        (document_id,),
    )
    if not rows:
        raise ValueError(f"source document not found: {document_id}")
    corpus_id = str(rows[0]["corpus_id"])
    expected = {
        str(row["chunk_id"]): row.get("metadata_json") or {}
        for row in rows if row.get("chunk_id") is not None
    }
    client = QdrantClient(url=settings.qdrant_url, timeout=60)
    points = _scroll(client, collection_name(corpus_id), document_id)
    selectors = _scroll(client, document_metadata_collection_name(corpus_id), document_id)
    actual = {str(point.payload.get("chunk_id")): point.payload for point in points}
    failures: list[str] = []
    if set(actual) != set(expected):
        failures.append("postgres_qdrant_chunk_id_mismatch")
    if any(payload.get("metadata_pipeline_version") != METADATA_PIPELINE_VERSION for payload in actual.values()):
        failures.append("qdrant_chunk_pipeline_mismatch")
    if len(selectors) != 1:
        failures.append("qdrant_selector_count_mismatch")
    elif selectors[0].payload.get("metadata_pipeline_version") != METADATA_PIPELINE_VERSION:
        failures.append("qdrant_selector_pipeline_mismatch")
    scope_fields = (
        "routing_product_models", "routing_part_numbers", "routing_protocol_terms",
        "firmware_applicability", "software_applicability",
    )
    for chunk_id in set(actual) & set(expected):
        if any(actual[chunk_id].get(field) != expected[chunk_id].get(field) for field in scope_fields):
            failures.append("postgres_qdrant_scope_mismatch")
            break
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "document_id": document_id,
        "pipeline": METADATA_PIPELINE_VERSION,
        "postgres_active_chunks": len(expected),
        "qdrant_chunks": len(actual),
        "qdrant_selectors": len(selectors),
        "checks_passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite audit artifact: {args.output}")
    try:
        result = audit(args.document_id)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result["checks_passed"] else 1)


if __name__ == "__main__":
    main()
