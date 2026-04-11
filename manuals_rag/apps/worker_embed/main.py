from __future__ import annotations

import logging

from manuals_rag_common.db import execute, fetch_all
from manuals_rag_common.logging import configure_logging
from manuals_rag_common.queue import dequeue
from manuals_rag_retrieval.qdrant_store import QdrantStore
from manuals_rag_schemas.documents import RetrievalChunk

log = logging.getLogger(__name__)


def process_job(job: dict[str, str]) -> None:
    log.info(
        "worker_embed processing run_id=%s document_id=%s version_id=%s",
        job["run_id"],
        job["document_id"],
        job["version_id"],
    )
    chunks = fetch_all("select * from retrieval_chunks where document_version_id = %s", (job["version_id"],))
    document = fetch_all("select corpus_id from source_documents where id = %s", (job["document_id"],))
    if not document:
        raise ValueError("Document missing for embed job.")
    store = QdrantStore()
    store.delete_document_chunks(
        document[0]["corpus_id"],
        source_document_id=job["document_id"],
        document_version_id=job["version_id"],
    )
    parsed_chunks = [
        RetrievalChunk.model_validate(
            {
                **chunk,
                "document_version_id": str(chunk["document_version_id"]),
                "source_document_id": str(chunk["source_document_id"]),
                "logical_node_ids_json": chunk["logical_node_ids_json"],
                "metadata_json": chunk["metadata_json"],
            }
        )
        for chunk in chunks
    ]
    store.upsert_chunks(document[0]["corpus_id"], parsed_chunks)
    execute("update ingestion_runs set status = 'completed', updated_at = now() where id = %s", (job["run_id"],))
    execute("update source_documents set ingest_status = 'indexed', updated_at = now() where id = %s", (job["document_id"],))
    log.info(
        "worker_embed completed run_id=%s document_id=%s version_id=%s chunks=%s",
        job["run_id"],
        job["document_id"],
        job["version_id"],
        len(parsed_chunks),
    )


def main() -> None:
    configure_logging()
    log.info("worker_embed started")
    while True:
        job = dequeue("embed_jobs", timeout=5)
        if not job:
            continue
        try:
            process_job(job)
        except Exception:
            log.exception(
                "worker_embed failed run_id=%s document_id=%s version_id=%s",
                job.get("run_id"),
                job.get("document_id"),
                job.get("version_id"),
            )
            execute(
                "update ingestion_runs set status = 'failed', error_message = %s, updated_at = now() where id = %s",
                ("embed worker failure; check worker logs", job.get("run_id")),
            )
            execute(
                "update source_documents set ingest_status = 'failed', updated_at = now() where id = %s",
                (job.get("document_id"),),
            )


if __name__ == "__main__":
    main()
