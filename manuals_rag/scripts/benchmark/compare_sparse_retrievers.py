#!/usr/bin/env python3
"""Compare the legacy hashed sparse index with the isolated native BM25 index."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manuals_rag_retrieval.qdrant_store import QdrantStore


def _required_chunk_ids(case: dict[str, Any]) -> set[str]:
    graph = case.get("expected_evidence_graph") or {}
    required_nodes = set(graph.get("answer_requires") or [])
    ids = {
        str(chunk_id)
        for node in graph.get("nodes") or []
        if node.get("required") is True or node.get("node_id") in required_nodes
        for chunk_id in node.get("expected_chunk_ids") or []
    }
    if ids:
        return ids
    return {
        str(chunk_id)
        for chunk_id in (case.get("expected_source_chunk_ids") or [case.get("source_chunk_id")])
        if chunk_id
    }


def _summary(rows: list[dict[str, Any]], cutoffs: list[int]) -> dict[str, Any]:
    answerable = [row for row in rows if row["required_chunk_ids"]]
    return {
        "case_count": len(rows),
        "answerable_case_count": len(answerable),
        "mean_latency_ms": statistics.mean(row["latency_ms"] for row in rows),
        "p95_latency_ms": sorted(row["latency_ms"] for row in rows)[max(0, int(len(rows) * 0.95) - 1)],
        "recall": {
            f"required_any_at_{cutoff}": sum(row["metrics"][str(cutoff)]["any"] for row in answerable) / max(len(answerable), 1)
            for cutoff in cutoffs
        } | {
            f"required_all_at_{cutoff}": sum(row["metrics"][str(cutoff)]["all"] for row in answerable) / max(len(answerable), 1)
            for cutoff in cutoffs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite benchmark artifact: {args.output}")
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    store = QdrantStore(timeout=60)
    cutoffs = [5, 10, 20, args.limit]
    methods = {
        "legacy_hashed_sparse": store.search_sparse,
        "indexed_qdrant_bm25": store.search_bm25,
    }
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
    for index, case in enumerate(cases, start=1):
        required = _required_chunk_ids(case)
        for name, search in methods.items():
            started = time.perf_counter()
            hits = search(args.corpus_id, str(case["query"]), {"is_active": True}, limit=args.limit)
            latency_ms = (time.perf_counter() - started) * 1000
            hit_ids = [str(hit.chunk_id) for hit in hits]
            results[name].append({
                "case_id": case.get("case_id"),
                "query": case.get("query"),
                "required_chunk_ids": sorted(required),
                "hit_chunk_ids": hit_ids,
                "latency_ms": latency_ms,
                "metrics": {
                    str(cutoff): {
                        "any": bool(required & set(hit_ids[:cutoff])) if required else None,
                        "all": required <= set(hit_ids[:cutoff]) if required else None,
                    }
                    for cutoff in cutoffs
                },
            })
        print(json.dumps({"completed": index, "total": len(cases)}), flush=True)
    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "corpus_id": args.corpus_id,
        "cutoffs": cutoffs,
        "methods": {
            name: {"summary": _summary(rows, cutoffs), "cases": rows}
            for name, rows in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["summary"] for name, value in artifact["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
