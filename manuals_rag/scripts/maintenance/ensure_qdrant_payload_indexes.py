#!/usr/bin/env python3
"""Create and verify retrieval payload indexes on one Qdrant collection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import monotonic, sleep

from qdrant_client import QdrantClient, models


INDEX_FIELDS = (
    "source_document_id",
    "chunk_type",
    "document_version_id",
    "version_signal",
    "keywords",
)


def _collection_name(corpus_id: str) -> str:
    return f"manuals_{corpus_id}"


def _payload_schema_fields(collection_info: object) -> set[str]:
    payload_schema = getattr(collection_info, "payload_schema", None) or {}
    return {str(field_name) for field_name in payload_schema}


def ensure_indexes(
    client: QdrantClient,
    collection_name: str,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    before = _payload_schema_fields(client.get_collection(collection_name))
    created: list[str] = []
    for field_name in INDEX_FIELDS:
        if field_name in before:
            continue
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        created.append(field_name)

    deadline = monotonic() + timeout_seconds
    while True:
        info = client.get_collection(collection_name)
        indexed = _payload_schema_fields(info)
        missing = [field_name for field_name in INDEX_FIELDS if field_name not in indexed]
        status = str(getattr(info, "status", "")).casefold()
        optimizer_status = str(getattr(info, "optimizer_status", "")).casefold()
        healthy = status.endswith("green") and "error" not in optimizer_status
        if not missing and healthy:
            return {
                "collection": collection_name,
                "created": created,
                "already_present": sorted(before.intersection(INDEX_FIELDS)),
                "verified": list(INDEX_FIELDS),
                "status": status,
                "optimizer_status": optimizer_status,
            }
        if monotonic() >= deadline:
            raise TimeoutError(
                f"Qdrant index verification timed out: missing={missing}, "
                f"status={status}, optimizer_status={optimizer_status}"
            )
        sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    client = QdrantClient(url=args.qdrant_url, timeout=args.timeout_seconds)
    try:
        result = ensure_indexes(
            client,
            _collection_name(args.corpus_id),
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
