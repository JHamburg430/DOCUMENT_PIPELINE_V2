from __future__ import annotations

import json

from manuals_rag_common.db import fetch_all
from manuals_rag_retrieval.embeddings import embed_dense
from manuals_rag_retrieval.qdrant_store import QdrantStore, collection_name


def main() -> None:
    store = QdrantStore()
    query_dim = len(embed_dense(["dimension probe"])[0])
    rows = fetch_all("select id from corpora order by id")
    report: list[dict[str, object]] = []
    for row in rows:
        corpus_id = str(row["id"])
        name = collection_name(corpus_id)
        if not store.client.collection_exists(name):
            report.append(
                {
                    "corpus_id": corpus_id,
                    "collection_exists": False,
                    "query_dim": query_dim,
                    "collection_dim": None,
                    "dimension_match": None,
                }
            )
            continue
        info = store.client.get_collection(name)
        vectors = info.config.params.vectors
        dense_config = vectors.get("dense") if isinstance(vectors, dict) else None
        collection_dim = dense_config.size if dense_config is not None else None
        report.append(
            {
                "corpus_id": corpus_id,
                "collection_exists": True,
                "query_dim": query_dim,
                "collection_dim": collection_dim,
                "dimension_match": collection_dim == query_dim,
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
