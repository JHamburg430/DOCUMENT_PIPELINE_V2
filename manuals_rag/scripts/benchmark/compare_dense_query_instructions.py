#!/usr/bin/env python3
"""Compare current dense-query embeddings with Qwen3's instructed query format."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manuals_rag_retrieval.qdrant_store import QdrantStore


INSTRUCTION = "Retrieve technical-manual evidence that completely answers the query."


def _required(case: dict[str, Any]) -> set[str]:
    graph = case.get("expected_evidence_graph") or {}
    required_nodes = set(graph.get("answer_requires") or [])
    ids = {
        str(chunk_id)
        for node in graph.get("nodes") or []
        if node.get("required") is True or node.get("node_id") in required_nodes
        for chunk_id in node.get("expected_chunk_ids") or []
    }
    return ids or {str(value) for value in case.get("expected_source_chunk_ids") or []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite artifact: {args.output}")
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    store = QdrantStore(timeout=60)
    methods = {"current_uninstructed": None, "qwen3_instructed": INSTRUCTION}
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in methods}
    for index, case in enumerate(cases, start=1):
        required = _required(case)
        for name, instruction in methods.items():
            started = time.perf_counter()
            hits = store.search_dense(
                args.corpus_id, case["query"], {"is_active": True}, limit=40,
                query_instruction=instruction,
            )
            latency = (time.perf_counter() - started) * 1000
            ids = [hit.chunk_id for hit in hits]
            rows[name].append({
                "case_id": case["case_id"], "required_chunk_ids": sorted(required),
                "hit_chunk_ids": ids, "latency_ms": latency,
                "any_at_10": bool(required & set(ids[:10])) if required else None,
                "all_at_40": required <= set(ids[:40]) if required else None,
            })
        print(json.dumps({"completed": index, "total": len(cases)}), flush=True)
    artifact = {
        "generated_at": datetime.now(UTC).isoformat(), "instruction": INSTRUCTION,
        "methods": {},
    }
    for name, method_rows in rows.items():
        answerable = [row for row in method_rows if row["required_chunk_ids"]]
        artifact["methods"][name] = {
            "summary": {
                "answerable_cases": len(answerable),
                "required_any_at_10": sum(row["any_at_10"] for row in answerable) / max(len(answerable), 1),
                "required_all_at_40": sum(row["all_at_40"] for row in answerable) / max(len(answerable), 1),
                "mean_latency_ms": statistics.mean(row["latency_ms"] for row in method_rows),
            },
            "cases": method_rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["summary"] for key, value in artifact["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
