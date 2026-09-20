#!/usr/bin/env python3
"""Summarize reproducible local-inference evidence from a completed agent matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return round(ordered[index], 2)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 2) if values else None,
    }


def _backend_summary(items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    records = [item[backend] for item in items]
    elapsed = [float(record.get("elapsed_ms") or 0.0) for record in records]
    model_duration = [
        float((record.get("trace") or {}).get("cost", {}).get("total_duration_ms") or 0.0)
        for record in records
    ]
    tokens = [
        float((record.get("trace") or {}).get("cost", {}).get("total_tokens") or 0.0)
        for record in records
    ]
    stop_reasons: dict[str, int] = {}
    for record in records:
        reason = str(record.get("stop_reason") or "unknown")
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    return {
        "end_to_end_ms": _summary(elapsed),
        "measured_model_duration_ms": _summary(model_duration),
        "measured_model_tokens": _summary(tokens),
        "stop_reasons": dict(sorted(stop_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite artifact: {args.output}")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not matrix.get("complete") or int(matrix.get("process_exit_status", -1)) != 0:
        parser.error("performance audit requires a completed matrix with exit status 0")
    items = list(matrix.get("items") or [])
    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "matrix": str(args.matrix),
        "run_id": matrix.get("run_id"),
        "dataset_sha256": matrix.get("dataset_sha256"),
        "source": (matrix.get("provenance") or {}).get("source"),
        "configuration": (matrix.get("provenance") or {}).get("configuration"),
        "case_count": len(items),
        "backends": {
            backend: _backend_summary(items, backend)
            for backend in ("baseline", "langgraph", "llamaindex")
        },
        "telemetry_contract": {
            "available": [
                "end_to_end_case_latency",
                "measured_model_duration",
                "model_token_usage",
                "stop_reason",
                "configured_model_and_reranker",
            ],
            "missing": [
                "per_stage_latency",
                "queue_wait_latency",
                "worker_route",
                "worker_model_residency",
                "host_gpu_telemetry",
                "verifier_attempt_latency",
            ],
            "performance_acceptance_eligible": False,
        },
        "decision": {
            "serving_migration": "do_not_migrate",
            "reason": (
                "The run identifies end-to-end latency and model cost, but lacks the route, queue, "
                "residency, per-stage, and GPU evidence required to attribute the bottleneck. "
                "A vLLM migration is not justified as an accuracy fix."
            ),
            "rollback": "Keep the current Ollama and MiniLM serving path with agentic retrieval disabled.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
