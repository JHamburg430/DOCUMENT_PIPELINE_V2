from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from manuals_rag_common.db import fetch_all
from manuals_rag_retrieval.document_metadata import enrich_document_metadata_with_chunk_signals
from manuals_rag_retrieval.qdrant_store import QdrantStore


def _metadata_documents(corpus_id: str | None = None) -> list[dict[str, Any]]:
    query = """
        select
            sd.id as source_document_id,
            sd.current_version_id as document_version_id,
            sd.corpus_id,
            sd.title,
            sd.source_filename,
            sd.manufacturer,
            sd.product_family,
            sd.product_model,
            sd.document_kind,
            coalesce(dme.metadata_json, '{}'::jsonb) as metadata_json,
            true as is_active
        from source_documents sd
        left join document_metadata_extractions dme on dme.source_document_id = sd.id
        where sd.current_version_id is not null
    """
    params: tuple[Any, ...] = ()
    if corpus_id:
        query += " and sd.corpus_id = %s"
        params = (corpus_id,)
    query += " order by sd.corpus_id, sd.updated_at desc, sd.id"
    documents = fetch_all(query, params)
    chunk_rows_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if documents:
        document_ids = [str(document["source_document_id"]) for document in documents]
        placeholders = ",".join(["%s"] * len(document_ids))
        chunk_rows = fetch_all(
            f"""
            select source_document_id, metadata_json
            from retrieval_chunks
            where source_document_id in ({placeholders})
            """,
            tuple(document_ids),
        )
        for row in chunk_rows:
            chunk_rows_by_document[str(row["source_document_id"])].append(row)
    return [
        enrich_document_metadata_with_chunk_signals(
            document,
            chunk_rows_by_document.get(str(document["source_document_id"]), []),
        )
        for document in documents
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Qdrant document metadata embeddings from Postgres metadata.")
    parser.add_argument("--corpus-id", help="Optional corpus id to rebuild.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    documents_by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in _metadata_documents(args.corpus_id):
        documents_by_corpus[str(document["corpus_id"])].append(document)

    store = QdrantStore()
    total = 0
    for corpus_id, documents in documents_by_corpus.items():
        for index in range(0, len(documents), args.batch_size):
            batch = documents[index : index + args.batch_size]
            store.upsert_document_metadata(corpus_id, batch)
            total += len(batch)
        print(f"indexed {len(documents)} document metadata records for corpus {corpus_id}")
    print(f"indexed {total} document metadata records")


if __name__ == "__main__":
    main()
