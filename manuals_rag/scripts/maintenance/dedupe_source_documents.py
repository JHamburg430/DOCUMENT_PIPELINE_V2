from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manuals_rag_common.config import settings
from manuals_rag_common.db import execute, fetch_all
from manuals_rag_retrieval.qdrant_store import QdrantStore


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "test_reports"


@dataclass(frozen=True)
class DuplicateDocument:
    id: str
    corpus_id: str
    source_filename: str
    title: str
    ingest_status: str
    updated_at: Any
    current_version_id: str | None
    version_count: int
    chunk_count: int
    run_count: int


def _load_duplicate_groups() -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        with duplicates as (
            select sha256
            from source_documents
            group by sha256
            having count(*) > 1
        )
        select
            sd.sha256,
            sd.id,
            sd.corpus_id,
            sd.source_filename,
            sd.title,
            sd.ingest_status,
            sd.updated_at,
            sd.current_version_id,
            coalesce(version_stats.version_count, 0) as version_count,
            coalesce(chunk_stats.chunk_count, 0) as chunk_count,
            coalesce(run_stats.run_count, 0) as run_count
        from source_documents sd
        join duplicates d on d.sha256 = sd.sha256
        left join (
            select source_document_id, count(*) as version_count
            from document_versions
            group by source_document_id
        ) version_stats on version_stats.source_document_id = sd.id
        left join (
            select source_document_id, count(*) as chunk_count
            from retrieval_chunks
            group by source_document_id
        ) chunk_stats on chunk_stats.source_document_id = sd.id
        left join (
            select source_document_id, count(*) as run_count
            from ingestion_runs
            group by source_document_id
        ) run_stats on run_stats.source_document_id = sd.id
        order by sd.sha256, sd.updated_at desc, sd.id desc
        """
    )

    grouped: dict[str, list[DuplicateDocument]] = {}
    for row in rows:
        grouped.setdefault(row["sha256"], []).append(
            DuplicateDocument(
                id=str(row["id"]),
                corpus_id=str(row["corpus_id"]),
                source_filename=str(row["source_filename"]),
                title=str(row["title"]),
                ingest_status=str(row["ingest_status"]),
                updated_at=row["updated_at"],
                current_version_id=str(row["current_version_id"]) if row["current_version_id"] else None,
                version_count=int(row["version_count"]),
                chunk_count=int(row["chunk_count"]),
                run_count=int(row["run_count"]),
            )
        )
    return [{"sha256": sha256, "documents": documents} for sha256, documents in grouped.items()]


def _canonical_sort_key(document: DuplicateDocument) -> tuple[int, int, str, str]:
    return (
        1 if document.corpus_id == settings.default_corpus_id else 0,
        1 if document.ingest_status == "indexed" else 0,
        document.chunk_count,
        document.version_count,
        str(document.updated_at),
        document.id,
    )


def _build_plan() -> dict[str, Any]:
    groups = _load_duplicate_groups()
    plan_groups: list[dict[str, Any]] = []
    duplicate_documents = 0

    for group in groups:
        documents = list(group["documents"])
        canonical = max(documents, key=_canonical_sort_key)
        duplicates = [document for document in documents if document.id != canonical.id]
        duplicate_documents += len(duplicates)
        plan_groups.append(
            {
                "sha256": group["sha256"],
                "canonical": canonical.__dict__,
                "duplicates": [document.__dict__ for document in duplicates],
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "default_corpus_id": settings.default_corpus_id,
        "group_count": len(plan_groups),
        "duplicate_document_count": duplicate_documents,
        "groups": plan_groups,
    }


def _delete_duplicate_document(store: QdrantStore, document: dict[str, Any]) -> None:
    store.delete_document_chunks(
        document["corpus_id"],
        source_document_id=document["id"],
        document_version_id=document.get("current_version_id"),
    )
    execute("delete from source_documents where id = %s", (document["id"],))


def _write_report(plan: dict[str, Any], *, suffix: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"source_document_dedupe_{suffix}_{timestamp}.json"
    path.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove duplicate source documents by sha256.")
    parser.add_argument("--apply", action="store_true", help="Delete duplicates instead of printing a dry-run plan.")
    args = parser.parse_args()

    plan = _build_plan()
    report_path = _write_report(plan, suffix="plan")
    print(json.dumps({"report": str(report_path), "group_count": plan["group_count"], "duplicate_document_count": plan["duplicate_document_count"]}, indent=2))

    if not args.apply:
        return

    store = QdrantStore()
    deleted_documents = 0
    for group in plan["groups"]:
        for document in group["duplicates"]:
            _delete_duplicate_document(store, document)
            deleted_documents += 1

    result = {
        "applied_at": datetime.now(UTC).isoformat(),
        "deleted_document_count": deleted_documents,
        "group_count": plan["group_count"],
        "default_corpus_id": settings.default_corpus_id,
    }
    result_path = _write_report(result, suffix="applied")
    print(json.dumps({"result": str(result_path), **result}, indent=2))


if __name__ == "__main__":
    main()
