#!/usr/bin/env python3
"""Benchmark rerankers on identical persisted fusion candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenCausalReranker:
    """Official Qwen3 reranker yes/no log-probability scoring contract."""

    def __init__(self, model_name: str, *, device: str, max_length: int) -> None:
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device).eval()
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
            "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    @torch.no_grad()
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int, show_progress_bar: bool = False) -> list[float]:
        del show_progress_bar
        scores: list[float] = []
        instruction = "Retrieve technical-manual evidence that completely answers the query."
        for offset in range(0, len(pairs), batch_size):
            texts = [
                f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
                for query, document in pairs[offset : offset + batch_size]
            ]
            encoded = self.tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )
            encoded["input_ids"] = [
                self.prefix_tokens + token_ids + self.suffix_tokens
                for token_ids in encoded["input_ids"]
            ]
            batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            logits = self.model(**batch).logits[:, -1, :]
            binary = torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1)
            scores.extend(torch.nn.functional.log_softmax(binary, dim=1)[:, 1].exp().float().cpu().tolist())
        return scores


def _required_chunk_ids(item: dict[str, Any]) -> set[str]:
    graph = item.get("expected_evidence_graph") or {}
    required_nodes = set(graph.get("answer_requires") or [])
    return {
        str(chunk_id)
        for node in graph.get("nodes") or []
        if node.get("required") is True or node.get("node_id") in required_nodes
        for chunk_id in node.get("expected_chunk_ids") or []
    }


def _fusion_candidates(item: dict[str, Any]) -> list[dict[str, Any]]:
    for snapshot in item["baseline"]["stage_snapshots"]:
        if snapshot.get("stage") == "fusion" and int(snapshot.get("result_count") or 0) >= 20:
            return list(snapshot.get("results") or [])
    return []


def _candidate_text(candidate: dict[str, Any], max_chars: int) -> str:
    metadata = candidate.get("metadata") or {}
    text = str(
        metadata.get("rerank_document")
        or metadata.get("content_for_rerank")
        or candidate.get("content")
        or candidate.get("evidence_text")
        or ""
    )
    return text[:max_chars]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-chars", type=int, required=True)
    parser.add_argument("--pool-sizes", default="10,20,30")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite benchmark artifact: {args.output}")
    pool_sizes = [int(value) for value in args.pool_sizes.split(",")]
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    items = matrix["items"][: args.case_limit]
    pool_contract = [
        {
            "case_id": item["case_id"],
            "query": item["query"],
            "candidates": [
                {
                    "chunk_id": candidate.get("chunk_id"),
                    "text_sha256": hashlib.sha256(
                        _candidate_text(candidate, args.max_chars).encode("utf-8")
                    ).hexdigest(),
                }
                for candidate in _fusion_candidates(item)
            ],
        }
        for item in items
    ]
    empty_candidates = [
        (item["case_id"], candidate.get("chunk_id"))
        for item in items
        for candidate in _fusion_candidates(item)
        if not _candidate_text(candidate, args.max_chars).strip()
    ]
    if empty_candidates:
        parser.error(
            "persisted candidate pool contains empty reranker text; "
            f"first empty candidate={empty_candidates[0]!r}, count={len(empty_candidates)}"
        )
    pool_sha256 = hashlib.sha256(
        json.dumps(pool_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    load_started = time.perf_counter()
    if "Qwen3-Reranker" in args.model:
        model = QwenCausalReranker(args.model, device=args.device, max_length=args.max_length)
    else:
        model = CrossEncoder(args.model, device=args.device, max_length=args.max_length)
    load_seconds = time.perf_counter() - load_started
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        candidates = _fusion_candidates(item)
        required = _required_chunk_ids(item)
        case_results = {}
        for pool_size in pool_sizes:
            pool = candidates[:pool_size]
            texts = [_candidate_text(candidate, args.max_chars) for candidate in pool]
            started = time.perf_counter()
            scores = model.predict([(item["query"], text) for text in texts], batch_size=8, show_progress_bar=False)
            elapsed_ms = (time.perf_counter() - started) * 1000
            ranked = sorted(
                zip(pool, [float(score) for score in scores], strict=True),
                key=lambda value: value[1],
                reverse=True,
            )
            ranked_ids = [str(candidate["chunk_id"]) for candidate, _score in ranked]
            case_results[str(pool_size)] = {
                "latency_ms": elapsed_ms,
                "candidate_required_ids": sorted(required & {str(candidate["chunk_id"]) for candidate in pool}),
                "ranked_chunk_ids": ranked_ids,
                "scores": [score for _candidate, score in ranked],
                "required_any_at_5": bool(required & set(ranked_ids[:5])) if required else None,
                "required_all_at_12": required <= set(ranked_ids[:12]) if required else None,
                "reranker_loss_at_12": bool(required & set(ranked_ids)) and not bool(required & set(ranked_ids[:12])),
            }
        rows.append({
            "case_id": item["case_id"],
            "query": item["query"],
            "required_chunk_ids": sorted(required),
            "candidate_count": len(candidates),
            "pools": case_results,
        })
        print(json.dumps({"completed": index, "total": len(items)}), flush=True)
    answerable = [row for row in rows if row["required_chunk_ids"]]
    summary = {}
    for pool_size in pool_sizes:
        key = str(pool_size)
        latencies = [row["pools"][key]["latency_ms"] for row in rows]
        summary[key] = {
            "required_any_at_5": sum(row["pools"][key]["required_any_at_5"] for row in answerable) / max(len(answerable), 1),
            "required_all_at_12": sum(row["pools"][key]["required_all_at_12"] for row in answerable) / max(len(answerable), 1),
            "reranker_loss_at_12_count": sum(row["pools"][key]["reranker_loss_at_12"] for row in answerable),
            "mean_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        }
    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "matrix": str(args.matrix),
        "matrix_dataset_sha256": matrix.get("dataset_sha256"),
        "candidate_pool_sha256": pool_sha256,
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "max_chars": args.max_chars,
        "load_seconds": load_seconds,
        "case_count": len(rows),
        "summary": summary,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"load_seconds": load_seconds, "candidate_pool_sha256": pool_sha256, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
