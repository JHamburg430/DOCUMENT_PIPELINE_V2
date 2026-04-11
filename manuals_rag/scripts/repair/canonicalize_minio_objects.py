from __future__ import annotations

import json
import os
import time
from collections import defaultdict

from manuals_rag_common.db import execute, fetch_all
from manuals_rag_common.storage import ObjectStore


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri.removeprefix("s3://")
    bucket, _, object_name = without_scheme.partition("/")
    return bucket, object_name


def _source_canonical_name(tenant_id: str, sha256: str, source_filename: str) -> str:
    extension = os.path.splitext(source_filename)[1].lower() or ".bin"
    return f"{tenant_id}/sha256/{sha256}{extension}"


def _artifact_canonical_name(tenant_id: str, etag: str, object_name: str) -> str:
    extension = os.path.splitext(object_name)[1].lower() or ".bin"
    clean_etag = etag.strip('"')
    return f"{tenant_id}/artifacts-etag/{clean_etag}{extension}"


def _bucket_inventory(store: ObjectStore, bucket: str) -> dict[str, tuple[int, str]]:
    inventory: dict[str, tuple[int, str]] = {}
    for obj in store.client.list_objects(bucket, recursive=True):
        inventory[obj.object_name] = (obj.size, obj.etag or "")
    return inventory


def main() -> None:
    store = ObjectStore()
    originals_inventory = _bucket_inventory(store, "manuals-originals")
    artifacts_inventory = _bucket_inventory(store, "manuals-artifacts")

    source_rows = fetch_all(
        """
        select id, tenant_id, source_filename, sha256, storage_uri
        from source_documents
        where storage_uri is not null
        """,
        (),
    )
    artifact_rows = fetch_all(
        """
        select dv.id, sd.tenant_id, dv.docling_artifact_uri
        from document_versions dv
        join source_documents sd on sd.id = dv.source_document_id
        where dv.docling_artifact_uri is not null
        """,
        (),
    )

    report: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "originals": {
            "rows_seen": len(source_rows),
            "copied_to_canonical": 0,
            "updated_rows": 0,
            "examples": [],
        },
        "artifacts": {
            "rows_seen": len(artifact_rows),
            "copied_to_canonical": 0,
            "updated_rows": 0,
            "examples": [],
        },
    }

    for row in source_rows:
        current_uri = str(row["storage_uri"])
        bucket, current_object_name = _parse_s3_uri(current_uri)
        if bucket != "manuals-originals":
            continue
        canonical_object_name = _source_canonical_name(
            str(row["tenant_id"]),
            str(row["sha256"]),
            str(row["source_filename"]),
        )
        if current_object_name == canonical_object_name:
            continue
        if not store.object_exists(bucket, canonical_object_name):
            store.copy_object(bucket, current_object_name, canonical_object_name)
            originals_inventory[canonical_object_name] = originals_inventory[current_object_name]
            report["originals"]["copied_to_canonical"] += 1
        new_uri = f"s3://{bucket}/{canonical_object_name}"
        execute("update source_documents set storage_uri = %s, updated_at = now() where id = %s", (new_uri, row["id"]))
        report["originals"]["updated_rows"] += 1
        if len(report["originals"]["examples"]) < 25:
            report["originals"]["examples"].append(
                {"id": str(row["id"]), "from": current_uri, "to": new_uri}
            )

    artifact_groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in artifact_rows:
        current_uri = str(row["docling_artifact_uri"])
        bucket, current_object_name = _parse_s3_uri(current_uri)
        if bucket != "manuals-artifacts":
            continue
        details = artifacts_inventory.get(current_object_name)
        if not details:
            continue
        size, etag = details
        artifact_groups[(str(row["tenant_id"]), size, etag)].append(row)

    for (tenant_id, _, etag), rows in artifact_groups.items():
        canonical_object_name = None
        for row in rows:
            current_uri = str(row["docling_artifact_uri"])
            _, current_object_name = _parse_s3_uri(current_uri)
            candidate = _artifact_canonical_name(tenant_id, etag, current_object_name)
            canonical_object_name = candidate
            if store.object_exists("manuals-artifacts", canonical_object_name):
                break
        if canonical_object_name is None:
            continue
        for row in rows:
            current_uri = str(row["docling_artifact_uri"])
            bucket, current_object_name = _parse_s3_uri(current_uri)
            if current_object_name == canonical_object_name:
                continue
            if not store.object_exists(bucket, canonical_object_name):
                store.copy_object(bucket, current_object_name, canonical_object_name)
                artifacts_inventory[canonical_object_name] = artifacts_inventory[current_object_name]
                report["artifacts"]["copied_to_canonical"] += 1
            new_uri = f"s3://{bucket}/{canonical_object_name}"
            execute("update document_versions set docling_artifact_uri = %s where id = %s", (new_uri, row["id"]))
            report["artifacts"]["updated_rows"] += 1
            if len(report["artifacts"]["examples"]) < 25:
                report["artifacts"]["examples"].append(
                    {"id": str(row["id"]), "from": current_uri, "to": new_uri}
                )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
