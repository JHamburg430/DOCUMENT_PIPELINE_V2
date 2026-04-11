from __future__ import annotations

import json
import time
from collections import defaultdict

from manuals_rag_common.db import fetch_all
from manuals_rag_common.storage import ObjectStore


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri.removeprefix("s3://")
    bucket, _, object_name = without_scheme.partition("/")
    return bucket, object_name


def main() -> None:
    store = ObjectStore()
    referenced: dict[str, set[str]] = defaultdict(set)
    for row in fetch_all("select storage_uri from source_documents where storage_uri is not null", ()):
        bucket, object_name = _parse_s3_uri(str(row["storage_uri"]))
        referenced[bucket].add(object_name)
    for row in fetch_all("select docling_artifact_uri from document_versions where docling_artifact_uri is not null", ()):
        bucket, object_name = _parse_s3_uri(str(row["docling_artifact_uri"]))
        referenced[bucket].add(object_name)

    report: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "buckets": {},
    }
    for bucket in ("manuals-originals", "manuals-artifacts"):
        removed: list[str] = []
        kept = 0
        total = 0
        for obj in store.client.list_objects(bucket, recursive=True):
            total += 1
            if obj.object_name in referenced.get(bucket, set()):
                kept += 1
                continue
            store.remove_object(bucket, obj.object_name)
            removed.append(obj.object_name)
        report["buckets"][bucket] = {
            "total_before": total,
            "kept_referenced": kept,
            "removed_unreferenced": len(removed),
            "removed_examples": removed[:50],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
