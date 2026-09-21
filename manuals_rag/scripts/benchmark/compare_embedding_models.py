#!/usr/bin/env python3
"""Compare embedding models on identical frozen dense candidate pools.

This is a component-ranking benchmark, not a full-corpus retrieval benchmark.
It answers whether a model merits the much more expensive isolated reindex and
downstream answer-gate evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _required_chunk_ids(case: dict[str, Any]) -> set[str]:
    graph = case.get("expected_evidence_graph") or {}
    required_nodes = set(graph.get("answer_requires") or [])
    ids = {
        str(chunk_id)
        for node in graph.get("nodes") or []
        if node.get("required") is True or node.get("node_id") in required_nodes
        for chunk_id in node.get("expected_chunk_ids") or []
    }
    return ids or {
        str(chunk_id)
        for chunk_id in case.get("expected_source_chunk_ids") or []
        if chunk_id
    }


def _dense_candidates(case: dict[str, Any]) -> list[dict[str, str]]:
    snapshots = (case.get("baseline") or {}).get("stage_snapshots") or []
    dense = next((snapshot for snapshot in snapshots if snapshot.get("stage") == "dense"), None)
    candidates: list[dict[str, str]] = []
    for result in (dense or {}).get("results") or []:
        chunk_id = str(result.get("chunk_id") or "").strip()
        text = str(result.get("evidence_text") or "").strip()
        if not chunk_id:
            raise ValueError(f"dense candidate is missing chunk_id for case {case.get('case_id')}")
        if not text:
            raise ValueError(f"dense candidate {chunk_id} has empty evidence_text")
        candidates.append({"chunk_id": chunk_id, "text": text})
    if not candidates:
        raise ValueError(f"case {case.get('case_id')} has no persisted dense candidates")
    return candidates


def _pool_hash(candidates: list[dict[str, str]]) -> str:
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimensions differ: {len(left)} != {len(right)}")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _embed(base_url: str, model: str, texts: list[str], batch_size: int) -> tuple[list[list[float]], float]:
    vectors: list[list[float]] = []
    started = time.perf_counter()
    for offset in range(0, len(texts), batch_size):
        request = Request(
            f"{base_url.rstrip('/')}/api/embed",
            data=json.dumps({"model": model, "input": texts[offset : offset + batch_size]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=300) as response:  # noqa: S310 - operator-supplied local endpoint
            batch = json.load(response).get("embeddings") or []
        if len(batch) != len(texts[offset : offset + batch_size]):
            raise ValueError(f"model {model} returned {len(batch)} vectors for a mismatched batch")
        vectors.extend(batch)
    return vectors, (time.perf_counter() - started) * 1000


def _summary(rows: list[dict[str, Any]], cutoffs: list[int]) -> dict[str, Any]:
    eligible = [row for row in rows if row["all_required_present_in_pool"]]
    return {
        "case_count": len(rows),
        "all_required_present_case_count": len(eligible),
        "pool_required_any_coverage": sum(row["any_required_present_in_pool"] for row in rows) / max(len(rows), 1),
        "pool_required_all_coverage": len(eligible) / max(len(rows), 1),
        "recall": {
            f"required_any_at_{cutoff}": sum(row["metrics"][str(cutoff)]["any"] for row in eligible) / max(len(eligible), 1)
            for cutoff in cutoffs
        } | {
            f"required_all_at_{cutoff}": sum(row["metrics"][str(cutoff)]["all"] for row in eligible) / max(len(eligible), 1)
            for cutoff in cutoffs
        },
        "mean_case_ranking_ms": statistics.mean(row["ranking_ms"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11436")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite benchmark artifact: {args.output}")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = matrix.get("items") or []
    frozen: list[dict[str, Any]] = []
    excluded_cases: list[dict[str, str]] = []
    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    for case in cases:
        snapshots = (case.get("baseline") or {}).get("stage_snapshots") or []
        dense = next((snapshot for snapshot in snapshots if snapshot.get("stage") == "dense"), None)
        if not (dense or {}).get("results"):
            excluded_cases.append({
                "case_id": str(case.get("case_id")),
                "reason": "baseline route did not execute dense retrieval",
            })
            continue
        candidates = _dense_candidates(case)
        query = str(case.get("query") or "").strip()
        if not query:
            raise ValueError(f"case {case.get('case_id')} has an empty query")
        for text in [query, *(candidate["text"] for candidate in candidates)]:
            if text not in text_index:
                text_index[text] = len(unique_texts)
                unique_texts.append(text)
        frozen.append({
            "case_id": str(case.get("case_id")),
            "query": query,
            "required_chunk_ids": sorted(_required_chunk_ids(case)),
            "candidates": candidates,
            "candidate_pool_hash": _pool_hash(candidates),
        })

    cutoffs = [1, 5, 10, 20, 40]
    artifact: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark_type": "frozen_dense_candidate_pool_ranking",
        "production_adoption_evidence": False,
        "matrix": str(args.matrix),
        "matrix_dataset_sha256": matrix.get("dataset_sha256"),
        "matrix_source": (matrix.get("provenance") or {}).get("source"),
        "candidate_text_contract": "persisted nonempty baseline dense stage evidence_text",
        "candidate_pool_count": len(frozen),
        "excluded_cases": excluded_cases,
        "unique_input_text_count": len(unique_texts),
        "exact_input_text_sha256": hashlib.sha256(
            json.dumps(unique_texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "models": {},
    }
    for model in args.models:
        vectors, embed_ms = _embed(args.ollama_url, model, unique_texts, args.batch_size)
        by_text = {text: vectors[index] for text, index in text_index.items()}
        rows: list[dict[str, Any]] = []
        for case in frozen:
            started = time.perf_counter()
            query_vector = by_text[case["query"]]
            ranked = sorted(
                (
                    {
                        "chunk_id": candidate["chunk_id"],
                        "score": _cosine(query_vector, by_text[candidate["text"]]),
                    }
                    for candidate in case["candidates"]
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
            ranking_ms = (time.perf_counter() - started) * 1000
            ranked_ids = [item["chunk_id"] for item in ranked]
            required = set(case["required_chunk_ids"])
            pool_ids = {candidate["chunk_id"] for candidate in case["candidates"]}
            all_present = bool(required) and required <= pool_ids
            rows.append({
                "case_id": case["case_id"],
                "candidate_pool_hash": case["candidate_pool_hash"],
                "required_chunk_ids": sorted(required),
                "any_required_present_in_pool": bool(required & pool_ids),
                "all_required_present_in_pool": all_present,
                "ranked_chunk_ids": ranked_ids,
                "ranking_ms": ranking_ms,
                "metrics": {
                    str(cutoff): {
                        "any": bool(required & set(ranked_ids[:cutoff])) if all_present else None,
                        "all": required <= set(ranked_ids[:cutoff]) if all_present else None,
                    }
                    for cutoff in cutoffs
                },
            })
        artifact["models"][model] = {
            "embedding_dimension": len(vectors[0]),
            "total_embedding_ms": embed_ms,
            "summary": _summary(rows, cutoffs),
            "cases": rows,
        }
        print(json.dumps({"model": model, "summary": artifact["models"][model]["summary"]}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
