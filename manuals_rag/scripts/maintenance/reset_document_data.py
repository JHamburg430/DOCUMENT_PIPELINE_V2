from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from minio.error import S3Error

from manuals_rag_common.config import settings
from manuals_rag_common.db import execute, fetch_all
from manuals_rag_common.queue import redis_client
from manuals_rag_common.storage import ObjectStore
from manuals_rag_retrieval.qdrant_store import collection_name, document_metadata_collection_name, QdrantStore


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "test_reports"
DOCUMENT_QUEUES = ("ingest_jobs", "embed_jobs")


@dataclass(frozen=True)
class ResetPlan:
    generated_at: str
    apply: bool
    corpus_ids: list[str] | None
    drop_corpora: bool
    include_minio: bool
    include_qdrant: bool
    include_redis: bool
    postgres: dict[str, Any]
    qdrant: dict[str, Any]
    minio: dict[str, Any]
    redis: dict[str, Any]


def _where_clause(corpus_ids: list[str] | None, alias: str = "sd") -> tuple[str, tuple[Any, ...]]:
    if not corpus_ids:
        return "", ()
    placeholders = ",".join(["%s"] * len(corpus_ids))
    return f" where {alias}.corpus_id in ({placeholders})", tuple(corpus_ids)


def _documents(corpus_ids: list[str] | None) -> list[dict[str, Any]]:
    where, params = _where_clause(corpus_ids)
    return fetch_all(
        f"""
        select
            sd.id as source_document_id,
            sd.tenant_id,
            sd.corpus_id,
            sd.storage_uri,
            sd.current_version_id,
            dv.docling_artifact_uri
        from source_documents sd
        left join document_versions dv on dv.id = sd.current_version_id
        {where}
        order by sd.corpus_id, sd.id
        """,
        params,
    )


def _postgres_counts(corpus_ids: list[str] | None) -> dict[str, Any]:
    where, params = _where_clause(corpus_ids)
    source_count = fetch_all(f"select count(*) as count from source_documents sd{where}", params)[0]["count"]
    version_count = fetch_all(
        f"""
        select count(*) as count
        from document_versions dv
        join source_documents sd on sd.id = dv.source_document_id
        {where}
        """,
        params,
    )[0]["count"]
    node_count = fetch_all(
        f"""
        select count(*) as count
        from logical_nodes ln
        join document_versions dv on dv.id = ln.document_version_id
        join source_documents sd on sd.id = dv.source_document_id
        {where}
        """,
        params,
    )[0]["count"]
    chunk_count = fetch_all(
        f"""
        select count(*) as count
        from retrieval_chunks rc
        join source_documents sd on sd.id = rc.source_document_id
        {where}
        """,
        params,
    )[0]["count"]
    metadata_count = fetch_all(
        f"""
        select count(*) as count
        from document_metadata_extractions dme
        join source_documents sd on sd.id = dme.source_document_id
        {where}
        """,
        params,
    )[0]["count"]
    run_count = fetch_all(
        f"""
        select count(*) as count
        from ingestion_runs ir
        join source_documents sd on sd.id = ir.source_document_id
        {where}
        """,
        params,
    )[0]["count"]
    corpora_count = 0
    if corpus_ids:
        placeholders = ",".join(["%s"] * len(corpus_ids))
        corpora_count = fetch_all(f"select count(*) as count from corpora where id in ({placeholders})", tuple(corpus_ids))[0]["count"]
    else:
        corpora_count = fetch_all("select count(*) as count from corpora")[0]["count"]
    by_corpus = fetch_all(
        f"""
        select sd.corpus_id, sd.ingest_status, count(*) as documents
        from source_documents sd
        {where}
        group by sd.corpus_id, sd.ingest_status
        order by sd.corpus_id, sd.ingest_status
        """,
        params,
    )
    return {
        "source_documents": int(source_count),
        "document_versions": int(version_count),
        "logical_nodes": int(node_count),
        "retrieval_chunks": int(chunk_count),
        "document_metadata_extractions": int(metadata_count),
        "ingestion_runs": int(run_count),
        "corpora": int(corpora_count),
        "documents_by_corpus": [dict(row) for row in by_corpus],
    }


def _qdrant_collections(corpus_ids: list[str] | None) -> list[str]:
    store = QdrantStore()
    existing = {collection.name for collection in store.client.get_collections().collections}
    if corpus_ids:
        wanted = []
        for corpus_id in corpus_ids:
            wanted.extend([collection_name(corpus_id), document_metadata_collection_name(corpus_id)])
        return sorted(name for name in wanted if name in existing)
    return sorted(name for name in existing if name.startswith("manuals_"))


def _qdrant_plan(corpus_ids: list[str] | None) -> dict[str, Any]:
    store = QdrantStore()
    collections = []
    for name in _qdrant_collections(corpus_ids):
        try:
            info = store.client.get_collection(name)
            points_count = info.points_count
        except Exception:
            points_count = None
        collections.append({"name": name, "points_count": points_count})
    return {"collection_count": len(collections), "collections": collections}


def _s3_parts(uri: str | None) -> tuple[str, str] | None:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        return None
    return parsed.netloc, parsed.path.lstrip("/")


def _object_exists(store: ObjectStore, bucket: str, object_name: str) -> bool:
    try:
        store.client.stat_object(bucket, object_name)
        return True
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            return False
        raise


def _list_bucket_objects(store: ObjectStore, bucket: str) -> set[str]:
    try:
        return {obj.object_name for obj in store.client.list_objects(bucket, recursive=True)}
    except S3Error as exc:
        if exc.code == "NoSuchBucket":
            return set()
        raise


def _list_prefix_objects(store: ObjectStore, bucket: str, prefix: str) -> set[str]:
    try:
        return {obj.object_name for obj in store.client.list_objects(bucket, prefix=prefix, recursive=True)}
    except S3Error as exc:
        if exc.code == "NoSuchBucket":
            return set()
        raise


def _minio_objects(corpus_ids: list[str] | None, documents: list[dict[str, Any]]) -> dict[str, set[str]]:
    store = ObjectStore()
    buckets = (settings.minio_bucket_originals, settings.minio_bucket_artifacts)
    objects: dict[str, set[str]] = {bucket: set() for bucket in buckets}
    if not corpus_ids:
        for bucket in buckets:
            objects[bucket] = _list_bucket_objects(store, bucket)
        return objects
    for document in documents:
        for uri_key in ("storage_uri", "docling_artifact_uri"):
            parts = _s3_parts(str(document.get(uri_key) or ""))
            if not parts:
                continue
            bucket, object_name = parts
            if bucket not in objects:
                objects[bucket] = set()
            if _object_exists(store, bucket, object_name):
                objects[bucket].add(object_name)
        tenant_id = str(document["tenant_id"])
        document_id = str(document["source_document_id"])
        version_id = str(document.get("current_version_id") or "")
        if version_id:
            prefix = f"{tenant_id}/document-assets/{document_id}/{version_id}/"
            objects.setdefault(settings.minio_bucket_artifacts, set()).update(
                _list_prefix_objects(store, settings.minio_bucket_artifacts, prefix)
            )
    return objects


def _minio_plan(corpus_ids: list[str] | None, documents: list[dict[str, Any]]) -> dict[str, Any]:
    objects = _minio_objects(corpus_ids, documents)
    return {
        "object_count": sum(len(names) for names in objects.values()),
        "buckets": {
            bucket: {
                "object_count": len(names),
                "objects": sorted(names),
                "sample": sorted(names)[:20],
            }
            for bucket, names in sorted(objects.items())
        },
    }


def _redis_plan() -> dict[str, Any]:
    client = redis_client()
    return {"queues": {queue: int(client.llen(queue)) for queue in DOCUMENT_QUEUES}}


def _build_plan(
    *,
    corpus_ids: list[str] | None,
    apply: bool,
    drop_corpora: bool,
    include_minio: bool,
    include_qdrant: bool,
    include_redis: bool,
) -> ResetPlan:
    documents = _documents(corpus_ids)
    return ResetPlan(
        generated_at=datetime.now(UTC).isoformat(),
        apply=apply,
        corpus_ids=corpus_ids,
        drop_corpora=drop_corpora,
        include_minio=include_minio,
        include_qdrant=include_qdrant,
        include_redis=include_redis,
        postgres=_postgres_counts(corpus_ids),
        qdrant=_qdrant_plan(corpus_ids) if include_qdrant else {"skipped": True},
        minio=_minio_plan(corpus_ids, documents) if include_minio else {"skipped": True},
        redis=_redis_plan() if include_redis else {"skipped": True},
    )


def _apply_postgres_reset(corpus_ids: list[str] | None, *, drop_corpora: bool) -> None:
    if corpus_ids:
        execute("delete from source_documents where corpus_id = any(%s)", (corpus_ids,))
        if drop_corpora:
            execute("delete from corpora where id = any(%s)", (corpus_ids,))
    else:
        execute("delete from source_documents")
        if drop_corpora:
            execute("delete from corpora")


def _apply_qdrant_reset(collections: list[dict[str, Any]]) -> None:
    store = QdrantStore()
    for collection in collections:
        name = str(collection["name"])
        if store.client.collection_exists(name):
            store.client.delete_collection(name)


def _apply_minio_reset(plan: dict[str, Any]) -> None:
    store = ObjectStore()
    for bucket, bucket_plan in plan["buckets"].items():
        for object_name in bucket_plan.get("objects", []):
            if _object_exists(store, bucket, object_name):
                store.remove_object(bucket, object_name)


def _apply_redis_reset() -> None:
    redis_client().delete(*DOCUMENT_QUEUES)


def _write_report(plan: ResetPlan) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = "applied" if plan.apply else "dry_run"
    path = REPORT_DIR / f"document_data_reset_{suffix}_{timestamp}.json"
    path.write_text(json.dumps(asdict(plan), indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset document data across Postgres, Qdrant, MinIO, and Redis.")
    parser.add_argument("--apply", action="store_true", help="Apply the reset. Default is dry-run only.")
    parser.add_argument("--corpus-id", action="append", help="Corpus id to reset. Repeat for multiple corpora. Omit to reset all document data.")
    parser.add_argument("--drop-corpora", action="store_true", help="Also delete corpus rows from Postgres.")
    parser.add_argument("--skip-minio", action="store_true", help="Do not inspect or remove MinIO objects.")
    parser.add_argument("--skip-qdrant", action="store_true", help="Do not inspect or remove Qdrant collections.")
    parser.add_argument("--skip-redis", action="store_true", help="Do not inspect or remove Redis document queues.")
    args = parser.parse_args()

    plan = _build_plan(
        corpus_ids=args.corpus_id,
        apply=args.apply,
        drop_corpora=args.drop_corpora,
        include_minio=not args.skip_minio,
        include_qdrant=not args.skip_qdrant,
        include_redis=not args.skip_redis,
    )
    report_path = _write_report(plan)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "report": str(report_path),
                "postgres": plan.postgres,
                "qdrant_collection_count": plan.qdrant.get("collection_count"),
                "minio_object_count": plan.minio.get("object_count"),
                "redis": plan.redis,
            },
            indent=2,
            default=str,
        )
    )
    if not args.apply:
        return

    if not args.skip_qdrant:
        _apply_qdrant_reset(plan.qdrant.get("collections", []))
    if not args.skip_minio:
        _apply_minio_reset(plan.minio)
    if not args.skip_redis:
        _apply_redis_reset()
    _apply_postgres_reset(args.corpus_id, drop_corpora=args.drop_corpora)


if __name__ == "__main__":
    main()
