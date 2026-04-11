from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from manuals_rag_common.db import fetch_all
from manuals_rag_retrieval.embeddings import embed_dense
from manuals_rag_retrieval.qdrant_store import QdrantStore, collection_name
from manuals_rag_schemas.documents import RetrievalChunk


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "test_reports"


def _expected_query_dim() -> int:
    return len(embed_dense(["dimension probe"])[0])


def _mismatched_corpora(store: QdrantStore) -> list[dict[str, object]]:
    expected_dim = _expected_query_dim()
    rows = fetch_all("select id from corpora order by id")
    mismatches: list[dict[str, object]] = []
    for row in rows:
        corpus_id = str(row["id"])
        name = collection_name(corpus_id)
        if not store.client.collection_exists(name):
            continue
        info = store.client.get_collection(name)
        vectors = info.config.params.vectors
        dense_config = vectors.get("dense") if isinstance(vectors, dict) else None
        collection_dim = dense_config.size if dense_config is not None else None
        if collection_dim != expected_dim:
            mismatches.append(
                {
                    "corpus_id": corpus_id,
                    "collection_dim": collection_dim,
                    "expected_dim": expected_dim,
                }
            )
    return mismatches


def _load_chunks_for_corpus(corpus_id: str) -> list[RetrievalChunk]:
    rows = fetch_all(
        """
        select rc.*
        from retrieval_chunks rc
        join source_documents sd on sd.id = rc.source_document_id
        where sd.corpus_id = %s
        order by rc.page_from asc, rc.page_to asc, rc.chunk_level asc, rc.id asc
        """,
        (corpus_id,),
    )
    return [
        RetrievalChunk.model_validate(
            {
                **row,
                "document_version_id": str(row["document_version_id"]),
                "source_document_id": str(row["source_document_id"]),
                "logical_node_ids_json": row["logical_node_ids_json"],
                "metadata_json": row["metadata_json"],
            }
        )
        for row in rows
    ]


def _write_report(payload: dict[str, object], *, suffix: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"dense_collection_rebuild_{suffix}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Qdrant dense collections with mismatched vector dimensions.")
    parser.add_argument("--apply", action="store_true", help="Delete and rebuild mismatched collections.")
    args = parser.parse_args()

    store = QdrantStore()
    mismatches = _mismatched_corpora(store)
    plan = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mismatch_count": len(mismatches),
        "mismatches": [
            {
                **item,
                "chunk_count": len(_load_chunks_for_corpus(str(item["corpus_id"]))),
            }
            for item in mismatches
        ],
    }
    plan_path = _write_report(plan, suffix="plan")
    print(json.dumps({"report": str(plan_path), "mismatch_count": len(mismatches)}, indent=2))

    if not args.apply:
        return

    rebuilt: list[dict[str, object]] = []
    for item in mismatches:
        corpus_id = str(item["corpus_id"])
        chunks = _load_chunks_for_corpus(corpus_id)
        name = collection_name(corpus_id)
        if store.client.collection_exists(name):
            store.client.delete_collection(name)
        if chunks:
            store.upsert_chunks(corpus_id, chunks)
        rebuilt.append(
            {
                "corpus_id": corpus_id,
                "chunk_count": len(chunks),
                "previous_dim": item["collection_dim"],
                "new_dim": item["expected_dim"],
            }
        )

    result = {
        "applied_at": datetime.now(UTC).isoformat(),
        "rebuilt_count": len(rebuilt),
        "rebuilt": rebuilt,
    }
    result_path = _write_report(result, suffix="applied")
    print(json.dumps({"result": str(result_path), "rebuilt_count": len(rebuilt)}, indent=2))


if __name__ == "__main__":
    main()
