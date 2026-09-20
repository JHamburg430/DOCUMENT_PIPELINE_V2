#!/usr/bin/env python3
"""Build or resume the isolated Qdrant BM25 index from active PostgreSQL chunks."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from manuals_rag_common.db import fetch_all
from manuals_rag_retrieval.qdrant_store import QdrantStore, bm25_collection_name
from manuals_rag_schemas.documents import RetrievalChunk


def _parsed_chunk(row: dict[str, object]) -> RetrievalChunk:
    return RetrievalChunk.model_validate({
        **row,
        "document_version_id": str(row["document_version_id"]),
        "source_document_id": str(row["source_document_id"]),
        "logical_node_ids_json": row["logical_node_ids_json"],
        "metadata_json": row["metadata_json"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite report: {args.output}")

    store = QdrantStore(timeout=120)
    store.ensure_bm25_collection(args.corpus_id)
    started = time.perf_counter()
    processed = 0
    last_id: str | None = None
    while args.limit is None or processed < args.limit:
        take = min(args.batch_size, args.limit - processed) if args.limit is not None else args.batch_size
        rows = fetch_all(
            """
            select rc.* from retrieval_chunks rc
            join source_documents sd on sd.id = rc.source_document_id
            where sd.corpus_id = %s and rc.is_active = true
              and (%s::text is null or rc.id > %s::text)
            order by rc.id limit %s
            """,
            (args.corpus_id, last_id, last_id, take),
        )
        if not rows:
            break
        chunks = [_parsed_chunk(row) for row in rows]
        store.upsert_bm25_chunks(args.corpus_id, chunks)
        processed += len(chunks)
        last_id = str(chunks[-1].id)
        print(json.dumps({"processed": processed, "last_id": last_id}), flush=True)

    elapsed = time.perf_counter() - started
    qdrant_count = store.client.count(
        collection_name=bm25_collection_name(args.corpus_id), exact=True
    ).count
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_id": args.corpus_id,
        "collection": bm25_collection_name(args.corpus_id),
        "processed": processed,
        "qdrant_count": int(qdrant_count),
        "elapsed_seconds": elapsed,
        "chunks_per_second": processed / elapsed if elapsed else None,
        "limited": args.limit is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
