#!/usr/bin/env python3
"""Build the isolated visual-page sidecar index.

Dry-run is the default. Passing --apply performs Qdrant writes but still does
not enable query routing or visual answer generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from manuals_rag_common.config import settings
from manuals_rag_common.db import fetch_all
from manuals_rag_common.storage import ObjectStore
from manuals_rag_retrieval.visual_sidecar import VisualPage, VisualRetrievalSidecar, visual_collection_name


def _documents(corpus_id: str, limit: int | None) -> list[dict[str, Any]]:
    suffix = " limit %s" if limit else ""
    params: tuple[Any, ...] = (corpus_id, limit) if limit else (corpus_id,)
    return fetch_all(
        f"""
        select sd.id as source_document_id, sd.tenant_id, sd.title,
               dv.id as document_version_id, dv.page_count
        from source_documents sd
        join document_versions dv on dv.id = sd.current_version_id
        where sd.corpus_id = %s
          and sd.ingest_status in ('indexed', 'ready')
          and dv.status in ('parsed', 'active')
          and dv.page_count > 0
        order by sd.id
        {suffix}
        """,
        params,
    )


def _page_uri(document: dict[str, Any], page: int) -> str:
    return (
        f"s3://{settings.minio_bucket_artifacts}/{document['tenant_id']}/document-assets/"
        f"{document['source_document_id']}/{document['document_version_id']}/"
        f"page-images/page-{page:04d}.png"
    )


def _read_image(store: ObjectStore, uri: str) -> Image.Image:
    path = uri.removeprefix("s3://")
    bucket, object_name = path.split("/", 1)
    response = store.client.get_object(bucket, object_name)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    image = Image.open(BytesIO(data)).convert("RGB")
    image.load()
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-id", default=settings.default_corpus_id)
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite artifact: {args.output}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    documents = _documents(args.corpus_id, args.document_limit)
    page_manifest = [
        {
            "source_document_id": str(document["source_document_id"]),
            "document_version_id": str(document["document_version_id"]),
            "page": page,
            "image_uri": _page_uri(document, page),
        }
        for document in documents
        for page in range(1, int(document["page_count"] or 0) + 1)
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(page_manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "production_enabled": False,
        "corpus_id": args.corpus_id,
        "collection": visual_collection_name(args.corpus_id),
        "model": settings.visual_retrieval_model,
        "document_count": len(documents),
        "page_count": len(page_manifest),
        "page_manifest_sha256": manifest_sha256,
        "indexed_pages": 0,
        "missing_page_images": [],
    }
    if args.apply:
        store = ObjectStore()
        sidecar = VisualRetrievalSidecar()
        document_by_id = {str(document["source_document_id"]): document for document in documents}
        batch: list[VisualPage] = []
        for item in page_manifest:
            uri = str(item["image_uri"])
            path = uri.removeprefix("s3://")
            bucket, object_name = path.split("/", 1)
            if not store.object_exists(bucket, object_name):
                artifact["missing_page_images"].append(uri)
                continue
            document = document_by_id[str(item["source_document_id"])]
            batch.append(
                VisualPage(
                    source_document_id=str(item["source_document_id"]),
                    document_version_id=str(item["document_version_id"]),
                    page=int(item["page"]),
                    image_uri=uri,
                    title=str(document["title"]),
                    image=_read_image(store, uri),
                )
            )
            if len(batch) >= args.batch_size:
                artifact["indexed_pages"] += sidecar.upsert_pages(args.corpus_id, batch)
                batch.clear()
        if batch:
            artifact["indexed_pages"] += sidecar.upsert_pages(args.corpus_id, batch)
    artifact["complete"] = not args.apply or (
        artifact["indexed_pages"] + len(artifact["missing_page_images"]) == artifact["page_count"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
