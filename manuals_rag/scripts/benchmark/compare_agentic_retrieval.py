#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from threading import Lock
from time import perf_counter
from typing import Any

from manuals_rag_answering.agentic_retrieval import compare_agentic_backends, insufficient_agent_answer
from manuals_rag_answering.generator import generate_answer
from manuals_rag_common.ollama import capture_ollama_usage, summarize_ollama_usage
from manuals_rag_common.config import settings
from manuals_rag_evals.agent_eval_schema import build_expected_evidence_graph
from manuals_rag_evals.agent_eval import score_agent_run
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase, score_search_results
from manuals_rag_retrieval.retriever import (
    assess_evidence_sufficiency,
    build_filters,
    capture_retrieval_stages,
    retrieve,
)
from manuals_rag_schemas.documents import SearchResult


ARTIFACT_SCHEMA_VERSION = "agentic-retrieval-matrix-v2"
_PROGRESS_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _build_provenance(args: argparse.Namespace, raw_cases: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_bytes = args.dataset.read_bytes()
    dirty_status = _git_output("status", "--short", "--untracked-files=no")
    untracked_status = _git_output("status", "--short", "--untracked-files=all")
    untracked_paths = [line[3:] for line in untracked_status.splitlines() if line.startswith("?? ")]
    try:
        dirty_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            check=False,
            capture_output=True,
        ).stdout
    except FileNotFoundError:
        dirty_diff = b""
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    git_revision = _git_output("rev-parse", "HEAD")
    source_revision = str(getattr(args, "source_revision", None) or os.getenv("SOURCE_REVISION") or git_revision)
    source_branch = str(getattr(args, "source_branch", None) or os.getenv("SOURCE_BRANCH") or _git_output("branch", "--show-current"))
    dirty_override = getattr(args, "source_dirty", None)
    dirty_state_known = bool(git_revision) or dirty_override is not None
    source_dirty = bool(dirty_status) if dirty_override is None else bool(dirty_override)
    dirty_diff_sha256 = str(
        getattr(args, "source_dirty_diff_sha256", None)
        or hashlib.sha256(dirty_diff).hexdigest()
    )
    missing_provenance = [
        field
        for field, value in (
            ("source.revision", source_revision),
            ("source.branch", source_branch),
            ("source.dirty", dirty_state_known),
        )
        if not value
    ]
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": str(args.run_id),
        "started_at": str(args.started_at),
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": dataset_sha256,
            "offset": args.offset,
            "limit": args.limit,
            "ordered_case_keys": [
                f"{dataset_sha256}:{str(case.get('case_id') or '')}" for case in raw_cases
            ],
        },
        "source": {
            "revision": source_revision,
            "branch": source_branch,
            "dirty": source_dirty,
            "dirty_status": dirty_status.splitlines(),
            "dirty_diff_sha256": dirty_diff_sha256,
            "untracked_paths": untracked_paths,
        },
        "provenance_complete": not missing_provenance,
        "missing_provenance": missing_provenance,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
        },
        "configuration": {
            "corpus_ids": list(args.corpus_id),
            "backends": ["baseline", "langgraph", "llamaindex"],
            "max_hops": args.max_hops,
            "planner": "heuristic" if args.no_llm else "ollama",
            "ollama_url": settings.ollama_url,
            "ollama_embed_url": settings.ollama_embed_url,
            "ollama_embed_model": settings.ollama_embed_model,
            "ollama_fast_model": settings.ollama_fast_model,
            "ollama_retrieval_verifier_model": settings.ollama_retrieval_verifier_model,
            "ollama_answer_model": settings.ollama_answer_model,
            "qdrant_url": settings.qdrant_url,
            "rerank_model": settings.haystack_rerank_model,
            "rerank_device": settings.haystack_rerank_device,
            "result_limit": settings.agentic_retrieval_result_limit,
            "retrieval_branch_max_workers": settings.retrieval_branch_max_workers,
            "retrieval_qdrant_max_concurrency": settings.retrieval_qdrant_max_concurrency,
            "case_concurrency": int(getattr(args, "case_concurrency", 1)),
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _exclusive_output_lock(path: Path, run_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"matrix output is already owned by another writer: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"run_id": run_id, "pid": os.getpid(), "locked_at": _utc_now()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_cases(path: Path, *, limit: int, offset: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            case = record.get("case") if isinstance(record.get("case"), dict) else record
            if isinstance(case, dict):
                cases.append(case)
    return cases[max(0, offset) : max(0, offset) + max(1, limit)]


def _summary(items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    backend_items = [item[backend] for item in items]
    passed = sum(1 for item in backend_items if item["evaluation"]["passed"])
    sufficient = sum(1 for item in backend_items if item["sufficient"])
    layer_rates: dict[str, float] = {}
    if backend != "baseline" and backend_items:
        layer_names = list((backend_items[0].get("agent_evaluation") or {}).get("cells", {}))
        layer_rates = {
            layer: round(
                sum(
                    1
                    for item in backend_items
                    if ((item.get("agent_evaluation") or {}).get("cells", {}).get(layer) or {}).get("status") == "pass"
                )
                / len(backend_items),
                4,
            )
            for layer in layer_names
        }
    recovery_attempts = sum(
        1
        for item in backend_items
        for entry in (item.get("trace", {}).get("evidence_ledger") or {}).values()
        if entry.get("recovery_for")
    )
    successful_recoveries = sum(
        1
        for item in backend_items
        for entry in (item.get("trace", {}).get("evidence_ledger") or {}).values()
        if entry.get("recovery_for") and entry.get("sufficient")
    )
    return {
        "cases": len(backend_items),
        "retrieval_passed": passed,
        "retrieval_rate": round(passed / len(backend_items), 4) if backend_items else 0.0,
        "agent_sufficient": sufficient,
        "agent_sufficiency_rate": round(sufficient / len(backend_items), 4) if backend_items else 0.0,
        "mean_elapsed_ms": round(mean(item["elapsed_ms"] for item in backend_items), 2) if backend_items else 0.0,
        "mean_completed_hops": round(
            mean(len(item["trace"].get("completed_hops", [])) for item in backend_items), 2
        )
        if backend_items
        else 0.0,
        "agent_matrix_passed": sum(
            1 for item in backend_items if (item.get("agent_evaluation") or {}).get("passed")
        ),
        "layer_pass_rates": layer_rates,
        "recovery_attempts": recovery_attempts,
        "successful_recoveries": successful_recoveries,
        "recovery_success_rate": round(successful_recoveries / recovery_attempts, 4) if recovery_attempts else 0.0,
    }


def _category_summary(items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("agent_case_category") or "unknown")].append(item)
    return {category: _summary(category_items, backend) for category, category_items in grouped.items()}


def _emit_progress(enabled: bool, payload: dict[str, Any]) -> None:
    if not enabled:
        return
    with _PROGRESS_LOCK:
        print(json.dumps(payload), flush=True)


def _evaluate_case(
    args: argparse.Namespace,
    raw_case: dict[str, Any],
    question_number: int,
) -> dict[str, Any]:
    case = RetrievalEvalCase(**raw_case)
    _emit_progress(
        getattr(args, "progress_jsonl", False),
        {"event": "agent_case_started", "case_id": case.case_id, "question_number": question_number},
    )
    evidence_graph = build_expected_evidence_graph(raw_case)
    filters = build_filters(case.query, {})
    baseline_started = perf_counter()
    with capture_retrieval_stages() as baseline_stage_snapshots:
        baseline_results = retrieve(case.query, args.corpus_id, filters)
    baseline_elapsed_ms = round((perf_counter() - baseline_started) * 1000, 2)
    baseline_evaluation = score_search_results(
        case,
        [result.model_dump() for result in baseline_results],
        top_k=10,
    )
    baseline_sufficiency = assess_evidence_sufficiency(case.query, baseline_results)
    comparison = compare_agentic_backends(
        case.query,
        args.corpus_id,
        filters,
        max_hops=args.max_hops,
        use_llm=not args.no_llm,
    )
    item: dict[str, Any] = {
        "case_id": case.case_id,
        "query": case.query,
        "retrieval_task": case.retrieval_task,
        "agent_case_category": evidence_graph.category,
        "expected_evidence_graph": evidence_graph.model_dump(),
        "expected_document_ids": sorted(
            {
                case.source_document_id,
                *[
                    str(evidence.get("source_document_id") or "")
                    for evidence in case.expected_evidence or []
                    if evidence.get("source_document_id")
                ],
            }
        ),
        "equivalent_result_chunks": comparison["equivalent_result_chunks"],
        "baseline": {
            "elapsed_ms": baseline_elapsed_ms,
            "sufficient": baseline_sufficiency.sufficient,
            "stop_reason": "single_pass",
            "result_chunk_ids": [result.chunk_id for result in baseline_results],
            "result_document_ids": sorted({result.source_document_id for result in baseline_results}),
            "results": [result.model_dump() for result in baseline_results],
            "stage_snapshots": baseline_stage_snapshots,
            "trace": {"completed_hops": ["baseline"]},
            "evaluation": baseline_evaluation,
        },
    }
    for backend in ("langgraph", "llamaindex"):
        _emit_progress(
            getattr(args, "progress_jsonl", False),
            {"event": "answer_started", "case_id": case.case_id, "backend": backend},
        )
        output = comparison[backend]
        evaluation = score_search_results(case, output["results"], top_k=10)
        with capture_ollama_usage() as answer_usage_events:
            answer = (
                generate_answer(
                    case.query,
                    [SearchResult.model_validate(result) for result in output["results"]],
                )
                if output["sufficient"]
                else insufficient_agent_answer(case.query, output["trace"])
            ).model_dump()
        answer_usage = summarize_ollama_usage(answer_usage_events)
        retrieval_cost = dict(output["trace"].get("cost") or {})
        retrieval_by_purpose = dict(retrieval_cost.get("by_purpose") or {})
        answer_by_purpose = dict(answer_usage.get("by_purpose") or {})
        output["trace"]["cost"] = {
            **retrieval_cost,
            "retrieval_tokens": int(retrieval_cost.get("total_tokens") or 0),
            "answer_tokens": int(answer_usage.get("total_tokens") or 0),
            "answer_generation": answer_usage,
            "model_calls": int(retrieval_cost.get("model_calls") or 0)
            + int(answer_usage.get("model_calls") or 0),
            "prompt_tokens": int(retrieval_cost.get("prompt_tokens") or 0)
            + int(answer_usage.get("prompt_tokens") or 0),
            "completion_tokens": int(retrieval_cost.get("completion_tokens") or 0)
            + int(answer_usage.get("completion_tokens") or 0),
            "total_tokens": int(retrieval_cost.get("total_tokens") or 0)
            + int(answer_usage.get("total_tokens") or 0),
            "total_duration_ms": round(
                float(retrieval_cost.get("total_duration_ms") or 0.0)
                + float(answer_usage.get("total_duration_ms") or 0.0),
                2,
            ),
            "by_purpose": {**retrieval_by_purpose, **answer_by_purpose},
        }
        agent_evaluation = score_agent_run(
            raw_case,
            trace=output["trace"],
            results=output["results"],
            answer=answer,
            elapsed_ms=output["elapsed_ms"],
        )
        item[backend] = dict(output) | {
            "evaluation": evaluation,
            "answer": answer,
            "agent_evaluation": agent_evaluation,
        }
    return item


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_cases = _read_cases(args.dataset, limit=args.limit, offset=args.offset)
    provenance = getattr(args, "provenance", None) or _build_provenance(args, raw_cases)
    items: list[dict[str, Any]] = []
    case_concurrency = max(1, int(getattr(args, "case_concurrency", 1)))
    max_workers = min(case_concurrency, len(raw_cases) or 1)
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="matrix-case")
    active: dict[int, Future[dict[str, Any]]] = {}
    next_to_submit = 0
    try:
        while next_to_submit < min(max_workers, len(raw_cases)):
            active[next_to_submit] = executor.submit(
                _evaluate_case,
                args,
                raw_cases[next_to_submit],
                next_to_submit + 1,
            )
            next_to_submit += 1
        for case_index, _raw_case in enumerate(raw_cases):
            future = active.pop(case_index)
            item = future.result()
            items.append(item)
            if getattr(args, "output", None):
                _atomic_write_json(
                    args.output.with_suffix(".partial.json"),
                    {
                        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                        "run_id": provenance["run_id"],
                        "complete": False,
                        "completed_cases": len(items),
                        "expected_cases": len(raw_cases),
                        "completed_case_keys": provenance["dataset"]["ordered_case_keys"][: len(items)],
                        "provenance": provenance,
                        "items": items,
                    },
                )
            _emit_progress(
                getattr(args, "progress_jsonl", False),
                {
                    "event": "agent_case_completed",
                    "case_id": item["case_id"],
                    "question_number": case_index + 1,
                    "langgraph": item["langgraph"]["agent_evaluation"],
                    "llamaindex": item["llamaindex"]["agent_evaluation"],
                },
            )
            if next_to_submit < len(raw_cases):
                active[next_to_submit] = executor.submit(
                    _evaluate_case,
                    args,
                    raw_cases[next_to_submit],
                    next_to_submit + 1,
                )
                next_to_submit += 1
    except BaseException:
        for future in active.values():
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    dataset_bytes = args.dataset.read_bytes()
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": provenance["run_id"],
        "complete": True,
        "provenance": provenance,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "case_count": len(raw_cases),
        "category_counts": dict(Counter(item["agent_case_category"] for item in items)),
        "offset": args.offset,
        "limit": args.limit,
        "max_hops": args.max_hops,
        "planner": "heuristic" if args.no_llm else "ollama",
        "summary": {
            backend: _summary(items, backend) for backend in ("baseline", "langgraph", "llamaindex")
        },
        "category_summary": {
            backend: _category_summary(items, backend) for backend in ("baseline", "langgraph", "llamaindex")
        },
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LangGraph and LlamaIndex bounded multi-hop retrieval.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-id", action="append", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument(
        "--case-concurrency",
        type=int,
        default=int(os.getenv("AGENT_MATRIX_CASE_CONCURRENCY", "2")),
        help="Maximum cases evaluated concurrently; artifacts are still written in dataset order.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Use the deterministic fallback planner/refiner.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-jsonl", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--run-id", help="Immutable identifier shared by every artifact in this run.")
    parser.add_argument("--lock-file", type=Path, help="Exclusive writer lock path (defaults beside --output).")
    parser.add_argument("--source-revision", help="Source commit when Git metadata is unavailable at runtime.")
    parser.add_argument("--source-branch", help="Source branch when Git metadata is unavailable at runtime.")
    parser.add_argument("--source-dirty-diff-sha256", help="Hash of the launch-time dirty diff.")
    parser.add_argument("--source-dirty", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exit-file", type=Path, help="Write process exit status, including graceful timeout termination.")
    args = parser.parse_args()
    args.run_id = args.run_id or uuid.uuid4().hex
    args.started_at = _utc_now()
    lock_path = args.lock_file or (args.output.with_suffix(".lock") if args.output else None)
    exit_code = 1
    def terminated(signum, frame):
        raise SystemExit(128 + signum)
    previous_handler = signal.signal(signal.SIGTERM, terminated)
    lock_context = _exclusive_output_lock(lock_path, args.run_id) if lock_path else nullcontext()
    with lock_context:
        artifact_set_reserved = False
        try:
            artifact_paths = (
                [
                    args.output,
                    args.output.with_suffix(".launch.json"),
                    args.output.with_suffix(".partial.json"),
                ]
                if args.output
                else []
            )
            artifact_paths.extend([args.exit_file] if args.exit_file else [])
            collisions = [path for path in artifact_paths if path.exists()]
            if collisions:
                raise FileExistsError(
                    "refusing to overwrite immutable matrix artifact set: "
                    + ", ".join(str(path) for path in collisions)
                )
            artifact_set_reserved = True
            raw_cases = _read_cases(args.dataset, limit=args.limit, offset=args.offset)
            args.provenance = _build_provenance(args, raw_cases)
            if args.output:
                _atomic_write_json(
                    args.output.with_suffix(".launch.json"),
                    {**args.provenance, "state": "launched"},
                )
            report = run(args)
            report["completed_at"] = _utc_now()
            report["process_exit_status"] = 0
            rendered = json.dumps(report, indent=2)
            if args.output:
                _atomic_write_json(args.output, report)
            if not args.quiet:
                print(rendered)
            exit_code = 0
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            if args.exit_file and artifact_set_reserved:
                args.exit_file.parent.mkdir(parents=True, exist_ok=True)
                temporary_exit = args.exit_file.with_suffix(args.exit_file.suffix + ".tmp")
                temporary_exit.write_text(str(exit_code) + "\n", encoding="utf-8")
                temporary_exit.replace(args.exit_file)


if __name__ == "__main__":
    main()
