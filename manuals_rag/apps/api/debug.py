from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from manuals_rag_answering.generator import (
    ANSWER_SCHEMA,
    RECURSIVE_SUMMARY_PROMPT,
    RELEVANCE_PROMPT,
    RELEVANCE_SCHEMA,
    SUMMARY_PROMPT,
    SUMMARY_SCHEMA,
    SYSTEM_PROMPT,
    _evidence_text,
    _extract_json_summary,
    _fallback_answer,
    _fallback_relevance_judgments,
    _fallback_summary,
    _parse_relevance_response,
    _relevance_prompt,
    _summary_source_documents,
    generate_answer_with_trace,
    prioritize_results_for_answer,
    summarize_results_for_answer,
    validate_answer,
)
from manuals_rag_common.config import settings
from manuals_rag_common.db import fetch_all, fetch_one
from manuals_rag_common.ollama import chat_json_stream
from manuals_rag_answering.workflow import assemble, build_query_filters, classify_query, fuse, rerank, run_dense, run_sparse, run_special
from manuals_rag_schemas.documents import QueryRequest
from manuals_rag_schemas.documents import SearchResult

logger = logging.getLogger(__name__)


def list_recent_documents(*, limit: int = 50) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        select
            sd.id as document_id,
            sd.corpus_id,
            sd.title,
            sd.source_filename,
            sd.manufacturer,
            sd.product_family,
            sd.product_model,
            sd.document_kind,
            sd.ingest_status,
            sd.updated_at,
            dv.id as version_id,
            dv.page_count,
            dv.parse_profile,
            dv.quality_score
        from source_documents sd
        left join document_versions dv on dv.id = sd.current_version_id
        order by sd.updated_at desc
        limit %s
        """,
        (limit,),
    )
    return [dict(row) for row in rows]


def _serialize_search_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    content = str(result.get("content") or "")
    return {
        "chunk_id": result.get("chunk_id"),
        "score": result.get("score"),
        "title": result.get("title"),
        "document_version_id": result.get("document_version_id"),
        "source_document_id": result.get("source_document_id"),
        "pages": result.get("pages") or [],
        "section_path": result.get("section_path") or [],
        "chunk_type": metadata.get("chunk_type"),
        "retrieval_stage": metadata.get("retrieval_stage"),
        "content_preview": content[:280],
        "content": content,
        "context_window": metadata.get("context_window"),
        "parent_context": metadata.get("parent_context"),
        "metadata": metadata,
    }


def _attach_relevance_judgments(results: list[dict[str, Any]], judgments: list[dict[str, str]]) -> list[dict[str, Any]]:
    judgment_by_chunk_id = {item["chunk_id"]: item for item in judgments if item.get("chunk_id")}
    enriched: list[dict[str, Any]] = []
    for result in results:
        enriched_result = _serialize_search_result(result)
        judgment = judgment_by_chunk_id.get(str(enriched_result.get("chunk_id")), {})
        enriched_result["relevance_verdict"] = judgment.get("verdict")
        enriched_result["relevance_reason"] = judgment.get("reason")
        enriched.append(enriched_result)
    return enriched


def _attach_result_metadata(
    results: list[SearchResult],
    judgments: list[dict[str, str]],
) -> list[dict[str, Any]]:
    serialized = [result.model_dump() for result in results]
    return _attach_relevance_judgments(serialized, judgments)


def _serialize_stage(name: str, results: list[dict[str, Any]], *, sample_limit: int) -> dict[str, Any]:
    return {
        "name": name,
        "count": len(results),
        "samples": [_serialize_search_result(result) for result in results[:sample_limit]],
    }


def _serialize_step_result_sample(result: dict[str, Any]) -> dict[str, Any]:
    serialized = _serialize_search_result(result)
    return {
        "chunk_id": serialized.get("chunk_id"),
        "score": serialized.get("score"),
        "title": serialized.get("title"),
        "pages": serialized.get("pages"),
        "section_path": serialized.get("section_path"),
        "chunk_type": serialized.get("chunk_type"),
        "retrieval_stage": serialized.get("retrieval_stage"),
        "content_preview": serialized.get("content_preview"),
    }


def _step_results_payload(results: list[dict[str, Any]], *, sample_limit: int = 3) -> dict[str, Any]:
    return {
        "count": len(results),
        "samples": [_serialize_step_result_sample(result) for result in results[:sample_limit]],
    }


def _stream_step_payload(step_name: str, state: dict[str, Any], *, sample_limit: int = 3) -> dict[str, Any]:
    if step_name == "classify_query":
        return {"analysis": state.get("analysis", {})}
    if step_name == "build_filters":
        return {
            "request_filters": state.get("request_filters", {}),
            "filters": state.get("filters", {}),
            "applied_filters": state.get("filters", {}),
        }
    if step_name == "run_dense_search":
        return _step_results_payload([dict(result) for result in state.get("dense_results", [])], sample_limit=sample_limit)
    if step_name == "run_sparse_search":
        return _step_results_payload([dict(result) for result in state.get("sparse_results", [])], sample_limit=sample_limit)
    if step_name == "run_special_search":
        return _step_results_payload([dict(result) for result in state.get("special_results", [])], sample_limit=sample_limit)
    if step_name == "fuse_results":
        return {
            "input_counts": {
                "dense": len(state.get("dense_results", [])),
                "sparse": len(state.get("sparse_results", [])),
                "special": len(state.get("special_results", [])),
            },
            **_step_results_payload([dict(result) for result in state.get("fused_results", [])], sample_limit=sample_limit),
        }
    if step_name == "rerank_results":
        return _step_results_payload([dict(result) for result in state.get("retrieval_results", [])], sample_limit=sample_limit)
    if step_name == "assemble_context":
        results = [dict(result) for result in state.get("retrieval_results", [])]
        return {
            "count": len(results),
            "total_content_chars": sum(len(str(result.get("content") or "")) for result in results),
            "samples": [_serialize_step_result_sample(result) for result in results[:sample_limit]],
        }
    if step_name == "judge_answer_inputs":
        prioritized = state.get("prioritized") or {}
        judgments = prioritized.get("judgments") or []
        prioritized_results = prioritized.get("prioritized_results") or []
        return {
            "candidate_count": len(state.get("candidate_results", [])),
            "prioritized_count": len(prioritized_results),
            "judgments": judgments[:sample_limit],
            "samples": _attach_relevance_judgments(
                [
                    result.model_dump() if hasattr(result, "model_dump") else dict(result)
                    for result in prioritized_results[:sample_limit]
                ],
                judgments,
            ),
        }
    if step_name == "summarize_answer_inputs":
        summaries = state.get("summaries") or []
        return {
            "summary_count": len(summaries),
            "summaries": summaries[:sample_limit],
        }
    if step_name == "generate_answer":
        answer = state.get("answer")
        answer_payload = answer.model_dump() if hasattr(answer, "model_dump") else dict(answer or {})
        return {
            "answer": answer_payload,
            "answer_generation_trace": state.get("answer_trace") or {},
        }
    return {}


def build_query_debug_snapshot(
    request: QueryRequest,
    *,
    workflow_runner: Any,
    sample_limit: int = 10,
) -> dict[str, Any]:
    state = workflow_runner.invoke(
        {
            "query": request.query,
            "corpus_ids": request.corpus_ids,
            "request_filters": request.filters,
            "filters": request.filters,
        }
    )
    step_timings_ms = dict(state.get("step_timings_ms", {}))
    retrieval_results = [dict(result) for result in state.get("retrieval_results", [])]
    validated_results = [SearchResult.model_validate(result) for result in retrieval_results]
    candidate_results = validated_results[:8]
    relevance_started = perf_counter()
    prioritized = prioritize_results_for_answer(request.query, candidate_results)
    relevance_duration_ms = round((perf_counter() - relevance_started) * 1000, 2)
    summarization_started = perf_counter()
    summaries = summarize_results_for_answer(request.query, prioritized["prioritized_results"])
    summarization_duration_ms = round((perf_counter() - summarization_started) * 1000, 2)
    step_timings_ms["judge_answer_inputs"] = relevance_duration_ms
    step_timings_ms["summarize_answer_inputs"] = summarization_duration_ms
    answer_started = perf_counter()
    answer, answer_trace = generate_answer_with_trace(
        request.query,
        validated_results,
        prioritized_results=prioritized["prioritized_results"],
        summarized_evidence=summaries,
    )
    answer_payload = answer.model_dump()
    answer_duration_ms = round((perf_counter() - answer_started) * 1000, 2)
    step_timings_ms["generate_answer"] = answer_duration_ms
    retrieval_duration_ms = None
    if step_timings_ms.get("rerank_results") is not None or step_timings_ms.get("assemble_context") is not None:
        retrieval_duration_ms = round(
            float(step_timings_ms.get("rerank_results", 0.0)) + float(step_timings_ms.get("assemble_context", 0.0)),
            2,
        )
    stages = [
        {
            **_serialize_stage("dense_results", [dict(result) for result in state.get("dense_results", [])], sample_limit=sample_limit),
            "duration_ms": step_timings_ms.get("run_dense_search"),
        },
        {
            **_serialize_stage("sparse_results", [dict(result) for result in state.get("sparse_results", [])], sample_limit=sample_limit),
            "duration_ms": step_timings_ms.get("run_sparse_search"),
        },
        {
            **_serialize_stage("special_results", [dict(result) for result in state.get("special_results", [])], sample_limit=sample_limit),
            "duration_ms": step_timings_ms.get("run_special_search"),
        },
        {
            **_serialize_stage("fused_results", [dict(result) for result in state.get("fused_results", [])], sample_limit=sample_limit),
            "duration_ms": step_timings_ms.get("fuse_results"),
        },
        {
            **_serialize_stage("retrieval_results", retrieval_results, sample_limit=sample_limit),
            "duration_ms": retrieval_duration_ms,
        },
    ]
    return {
        "query": request.query,
        "corpus_ids": request.corpus_ids,
        "request_filters": request.filters,
        "filters": state.get("filters", {}),
        "applied_filters": state.get("filters", {}),
        "analysis": state.get("analysis", {}),
        "step_timings_ms": step_timings_ms,
        "stages": stages,
        "answer_generation_inputs": {
            "count": len(prioritized["prioritized_results"]),
            "samples": _attach_result_metadata(prioritized["prioritized_results"][:sample_limit], prioritized["judgments"]),
            "duration_ms": relevance_duration_ms,
        },
        "answer_summaries": {
            "count": len(summaries),
            "samples": summaries[:sample_limit],
            "duration_ms": summarization_duration_ms,
        },
        "answer": answer_payload,
        "answer_generation_trace": answer_trace,
    }


DEBUG_QUERY_STEP_SEQUENCE = [
    ("classify_query", "Analyzing query"),
    ("build_filters", "Building filters"),
    ("run_dense_search", "Running dense retrieval"),
    ("run_sparse_search", "Running sparse retrieval"),
    ("run_special_search", "Running special retrieval"),
    ("fuse_results", "Fusing retrieval results"),
    ("rerank_results", "Reranking candidates"),
    ("assemble_context", "Assembling answer context"),
    ("judge_answer_inputs", "Running relevance review"),
    ("summarize_answer_inputs", "Summarizing answer inputs"),
    ("generate_answer", "Generating final answer"),
]

DEBUG_QUERY_STEP_MODELS = {
    "judge_answer_inputs": settings.ollama_fast_model,
    "summarize_answer_inputs": settings.ollama_fast_model,
    "generate_answer": settings.ollama_answer_model,
}


def _query_debug_payload(
    request: QueryRequest,
    *,
    state: dict[str, Any],
    prioritized: dict[str, Any],
    summaries: list[dict[str, Any]],
    answer: Any,
    answer_trace: dict[str, Any],
    relevance_duration_ms: float,
    summarization_duration_ms: float,
    completed_steps: list[str],
    sample_limit: int,
) -> dict[str, Any]:
    retrieval_results = [dict(result) for result in state.get("retrieval_results", [])]
    retrieval_duration_ms = None
    if state["step_timings_ms"].get("rerank_results") is not None or state["step_timings_ms"].get("assemble_context") is not None:
        retrieval_duration_ms = round(
            float(state["step_timings_ms"].get("rerank_results", 0.0)) + float(state["step_timings_ms"].get("assemble_context", 0.0)),
            2,
        )
    stages = [
        {
            **_serialize_stage("dense_results", [dict(result) for result in state.get("dense_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("run_dense_search"),
        },
        {
            **_serialize_stage("sparse_results", [dict(result) for result in state.get("sparse_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("run_sparse_search"),
        },
        {
            **_serialize_stage("special_results", [dict(result) for result in state.get("special_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("run_special_search"),
        },
        {
            **_serialize_stage("fused_results", [dict(result) for result in state.get("fused_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("fuse_results"),
        },
        {
            **_serialize_stage("retrieval_results", retrieval_results, sample_limit=sample_limit),
            "duration_ms": retrieval_duration_ms,
        },
    ]
    return {
        "query": request.query,
        "corpus_ids": request.corpus_ids,
        "request_filters": request.filters,
        "filters": state.get("filters", {}),
        "applied_filters": state.get("filters", {}),
        "analysis": state.get("analysis", {}),
        "step_timings_ms": state.get("step_timings_ms", {}),
        "stages": stages,
        "answer_generation_inputs": {
            "count": len(prioritized["prioritized_results"]),
            "samples": _attach_result_metadata(prioritized["prioritized_results"][:sample_limit], prioritized["judgments"]),
            "duration_ms": relevance_duration_ms,
        },
        "answer_summaries": {
            "count": len(summaries),
            "samples": summaries[:sample_limit],
            "duration_ms": summarization_duration_ms,
        },
        "answer": answer.model_dump(),
        "answer_generation_trace": answer_trace,
        "progress": {
            "current_step": "generate_answer",
            "current_label": "Generating final answer",
            "current_model": DEBUG_QUERY_STEP_MODELS.get("generate_answer"),
            "completed_steps": completed_steps,
            "total_steps": len(DEBUG_QUERY_STEP_SEQUENCE),
            "step_sequence": [
                {"name": name, "label": step_label, "done": True, "model": DEBUG_QUERY_STEP_MODELS.get(name)}
                for name, step_label in DEBUG_QUERY_STEP_SEQUENCE
            ],
        },
    }


def execute_query_debug_run(
    request: QueryRequest,
    *,
    sample_limit: int = 10,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "query": request.query,
        "corpus_ids": request.corpus_ids,
        "request_filters": request.filters,
        "filters": request.filters,
        "step_timings_ms": {},
    }
    completed_steps: list[str] = []
    total_steps = len(DEBUG_QUERY_STEP_SEQUENCE)

    def report(step_name: str, label: str, *, status: str = "running") -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "status": status,
                "current_step": step_name,
                "current_label": label,
                "current_model": DEBUG_QUERY_STEP_MODELS.get(step_name),
                "completed_steps": completed_steps[:],
                "total_steps": total_steps,
                "step_sequence": [
                    {
                        "name": name,
                        "label": step_label,
                        "done": name in completed_steps,
                        "model": DEBUG_QUERY_STEP_MODELS.get(name),
                    }
                    for name, step_label in DEBUG_QUERY_STEP_SEQUENCE
                ],
                "step_timings_ms": dict(state.get("step_timings_ms", {})),
            }
        )

    def run_step(step_name: str, label: str, fn: Any) -> None:
        report(step_name, label)
        started = perf_counter()
        next_state = fn(state)
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        timings = dict(next_state.get("step_timings_ms", state.get("step_timings_ms", {})))
        timings[step_name] = elapsed_ms
        state.update(next_state)
        state["step_timings_ms"] = timings
        completed_steps.append(step_name)
        report(step_name, label, status="completed")

    run_step("classify_query", "Analyzing query", classify_query)
    run_step("build_filters", "Building filters", build_query_filters)
    run_step("run_dense_search", "Running dense retrieval", run_dense)
    run_step("run_sparse_search", "Running sparse retrieval", run_sparse)
    run_step("run_special_search", "Running special retrieval", run_special)
    run_step("fuse_results", "Fusing retrieval results", fuse)
    run_step("rerank_results", "Reranking candidates", rerank)
    run_step("assemble_context", "Assembling answer context", assemble)

    retrieval_results = [dict(result) for result in state.get("retrieval_results", [])]
    validated_results = [SearchResult.model_validate(result) for result in retrieval_results]
    candidate_results = validated_results[:8]

    report("judge_answer_inputs", "Running relevance review")
    relevance_started = perf_counter()
    prioritized = prioritize_results_for_answer(request.query, candidate_results)
    relevance_duration_ms = round((perf_counter() - relevance_started) * 1000, 2)
    state["step_timings_ms"]["judge_answer_inputs"] = relevance_duration_ms
    completed_steps.append("judge_answer_inputs")
    report("judge_answer_inputs", "Running relevance review", status="completed")

    report("summarize_answer_inputs", "Summarizing answer inputs")
    summarization_started = perf_counter()
    summaries = summarize_results_for_answer(request.query, prioritized["prioritized_results"])
    summarization_duration_ms = round((perf_counter() - summarization_started) * 1000, 2)
    state["step_timings_ms"]["summarize_answer_inputs"] = summarization_duration_ms
    completed_steps.append("summarize_answer_inputs")
    report("summarize_answer_inputs", "Summarizing answer inputs", status="completed")

    report("generate_answer", "Generating final answer")
    answer_started = perf_counter()
    answer, answer_trace = generate_answer_with_trace(
        request.query,
        validated_results,
        prioritized_results=prioritized["prioritized_results"],
        summarized_evidence=summaries,
    )
    answer_duration_ms = round((perf_counter() - answer_started) * 1000, 2)
    state["step_timings_ms"]["generate_answer"] = answer_duration_ms
    completed_steps.append("generate_answer")
    report("generate_answer", "Generating final answer", status="completed")

    retrieval_duration_ms = None
    if state["step_timings_ms"].get("rerank_results") is not None or state["step_timings_ms"].get("assemble_context") is not None:
        retrieval_duration_ms = round(
            float(state["step_timings_ms"].get("rerank_results", 0.0)) + float(state["step_timings_ms"].get("assemble_context", 0.0)),
            2,
        )
    stages = [
        {
            **_serialize_stage("dense_results", [dict(result) for result in state.get("dense_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("run_dense_search"),
        },
        {
            **_serialize_stage("sparse_results", [dict(result) for result in state.get("sparse_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("run_sparse_search"),
        },
        {
            **_serialize_stage("special_results", [dict(result) for result in state.get("special_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("run_special_search"),
        },
        {
            **_serialize_stage("fused_results", [dict(result) for result in state.get("fused_results", [])], sample_limit=sample_limit),
            "duration_ms": state["step_timings_ms"].get("fuse_results"),
        },
        {
            **_serialize_stage("retrieval_results", retrieval_results, sample_limit=sample_limit),
            "duration_ms": retrieval_duration_ms,
        },
    ]
    return {
        "query": request.query,
        "corpus_ids": request.corpus_ids,
        "request_filters": request.filters,
        "filters": state.get("filters", {}),
        "applied_filters": state.get("filters", {}),
        "analysis": state.get("analysis", {}),
        "step_timings_ms": state.get("step_timings_ms", {}),
        "stages": stages,
        "answer_generation_inputs": {
            "count": len(prioritized["prioritized_results"]),
            "samples": _attach_result_metadata(prioritized["prioritized_results"][:sample_limit], prioritized["judgments"]),
            "duration_ms": relevance_duration_ms,
        },
        "answer_summaries": {
            "count": len(summaries),
            "samples": summaries[:sample_limit],
            "duration_ms": summarization_duration_ms,
        },
        "answer": answer.model_dump(),
        "answer_generation_trace": answer_trace,
        "progress": {
            "current_step": "generate_answer",
            "current_label": "Generating final answer",
            "current_model": DEBUG_QUERY_STEP_MODELS.get("generate_answer"),
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "step_sequence": [
                {"name": name, "label": step_label, "done": True, "model": DEBUG_QUERY_STEP_MODELS.get(name)}
                for name, step_label in DEBUG_QUERY_STEP_SEQUENCE
            ],
        },
    }


def _stream_llm_json(
    *,
    emit: Any,
    step_name: str,
    call_id: str,
    label: str,
    model: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any],
    think: bool | None,
    timeout: float,
    purpose: str,
    num_predict: int | None = None,
) -> tuple[dict[str, Any], str]:
    emit(
        {
            "event": "llm_call_started",
            "step": step_name,
            "call_id": call_id,
            "label": label,
            "model": model,
            "purpose": purpose,
        }
    )

    def on_token(token: str) -> None:
        emit(
            {
                "event": "llm_token",
                "step": step_name,
                "call_id": call_id,
                "token": token,
            }
        )

    try:
        parsed, raw = chat_json_stream(
            model=model,
            messages=messages,
            json_schema=json_schema,
            on_token=on_token,
            think=think,
            timeout=timeout,
            purpose=purpose,
            num_predict=num_predict,
        )
    except Exception as exc:
        emit(
            {
                "event": "llm_call_failed",
                "step": step_name,
                "call_id": call_id,
                "label": label,
                "model": model,
                "purpose": purpose,
                "error": str(exc),
            }
        )
        raise
    emit(
        {
            "event": "llm_call_completed",
            "step": step_name,
            "call_id": call_id,
            "label": label,
            "model": model,
            "purpose": purpose,
            "raw_response": raw,
        }
    )
    return parsed, raw


def _stream_prioritize_results_for_answer(
    query: str,
    candidate_results: list[SearchResult],
    *,
    emit: Any,
) -> dict[str, Any]:
    if not candidate_results:
        return {"judgments": [], "prioritized_results": []}
    evidence = [
        {
            "chunk_id": result.chunk_id,
            "title": result.title,
            "pages": result.pages,
            "section_path": result.section_path,
            "content": _evidence_text(result)[:2000],
            "document_version_id": result.document_version_id,
        }
        for result in candidate_results
    ]
    fallback = _fallback_relevance_judgments(query, candidate_results)
    try:
        for attempt, strict in enumerate((False, True), start=1):
            _parsed, raw_response = _stream_llm_json(
                emit=emit,
                step_name="judge_answer_inputs",
                call_id=f"relevance_review_attempt_{attempt}",
                label=f"Relevance review attempt {attempt}",
                model=settings.ollama_fast_model,
                messages=[
                    {"role": "system", "content": RELEVANCE_PROMPT},
                    {"role": "user", "content": _relevance_prompt(query, evidence, strict=strict)},
                ],
                json_schema=RELEVANCE_SCHEMA,
                think=False,
                timeout=90.0,
                purpose="relevance_review",
            )
            try:
                judgments, diagnostics = _parse_relevance_response(raw_response, query, candidate_results)
            except Exception as exc:
                diagnostics = {
                    "missing_chunk_ids": [item["chunk_id"] for item in fallback],
                    "invalid_items": [{"error": str(exc)}],
                }
                judgments = fallback
            emit(
                {
                    "event": "llm_call_diagnostics",
                    "step": "judge_answer_inputs",
                    "call_id": f"relevance_review_attempt_{attempt}",
                    "diagnostics": diagnostics,
                }
            )
            if not diagnostics["missing_chunk_ids"] and not diagnostics["invalid_items"]:
                break
            logger.warning(
                "Streaming relevance judgment response was incomplete on attempt %s; retrying=%s missing_chunk_ids=%s invalid_items=%s raw_response=%s",
                attempt,
                attempt == 1,
                diagnostics["missing_chunk_ids"],
                diagnostics["invalid_items"],
                raw_response[:4000],
            )
            if attempt == 2:
                break
    except Exception as exc:
        logger.warning("Streaming relevance judgment failed; using fallback judgments: %s", exc)
        judgments = fallback

    judgment_by_chunk_id = {item["chunk_id"]: item for item in judgments}
    prioritized_results = [
        result
        for result in candidate_results
        if judgment_by_chunk_id.get(result.chunk_id, {}).get("verdict") == "relevant"
    ]
    prioritized_results.extend(
        result
        for result in candidate_results
        if judgment_by_chunk_id.get(result.chunk_id, {}).get("verdict") == "potentially_relevant"
        and result.chunk_id not in {item.chunk_id for item in prioritized_results}
    )
    if not prioritized_results:
        prioritized_results = candidate_results
    return {
        "judgments": judgments,
        "prioritized_results": prioritized_results,
    }


def _stream_summarize_chunk(query: str, result: SearchResult, *, emit: Any) -> dict[str, Any]:
    payload = {
        "chunk_id": result.chunk_id,
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "content": _evidence_text(result)[:2500],
        "parent_context": str(result.metadata.get("parent_context") or "")[:1000],
    }
    try:
        _parsed, raw = _stream_llm_json(
            emit=emit,
            step_name="summarize_answer_inputs",
            call_id=f"chunk_summary_{result.chunk_id}",
            label=f"Summarize chunk {result.chunk_id}",
            model=settings.ollama_fast_model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"Question: {query}\nEvidence item: {json.dumps(payload)}"},
            ],
            json_schema=SUMMARY_SCHEMA,
            think=False,
            timeout=60.0,
            purpose="chunk_summary",
        )
        summary = _extract_json_summary(raw)
    except Exception as exc:
        logger.warning("Streaming chunk summary failed for %s; using fallback summary: %s", result.chunk_id, exc)
        summary = _fallback_summary(query, result)
    return {
        "chunk_id": result.chunk_id,
        "title": result.title,
        "pages": result.pages,
        "section_path": result.section_path,
        "summary": summary,
        "source_document_id": result.source_document_id,
        "document_version_id": result.document_version_id,
        "source_documents": [
            {
                "chunk_id": result.chunk_id,
                "title": result.title,
                "pages": result.pages,
                "section_path": result.section_path,
                "source_document_id": result.source_document_id,
                "document_version_id": result.document_version_id,
            }
        ],
    }


def _stream_merge_summary_batch(query: str, batch: list[dict[str, Any]], *, emit: Any) -> dict[str, Any]:
    batch_chunk_ids = [item["chunk_id"] for item in batch]
    try:
        _parsed, raw = _stream_llm_json(
            emit=emit,
            step_name="summarize_answer_inputs",
            call_id=f"recursive_summary_{'_'.join(batch_chunk_ids)[:80]}",
            label=f"Merge summaries for {', '.join(batch_chunk_ids)}",
            model=settings.ollama_fast_model,
            messages=[
                {"role": "system", "content": RECURSIVE_SUMMARY_PROMPT},
                {"role": "user", "content": f"Question: {query}\nSummaries: {json.dumps(batch)}"},
            ],
            json_schema=SUMMARY_SCHEMA,
            think=False,
            timeout=60.0,
            purpose="recursive_summary",
        )
        summary = _extract_json_summary(raw)
    except Exception as exc:
        logger.warning("Streaming recursive summary failed for chunk_ids=%s; using concatenated fallback: %s", batch_chunk_ids, exc)
        summary = " ".join(item["summary"] for item in batch)
    return {
        "chunk_id": ",".join(batch_chunk_ids),
        "title": batch[0]["title"],
        "pages": sorted({page for item in batch for page in item.get("pages", [])}),
        "section_path": batch[0].get("section_path", []),
        "summary": summary[:2000],
        "source_document_id": batch[0]["source_document_id"],
        "document_version_id": batch[0]["document_version_id"],
        "source_documents": _summary_source_documents(batch),
    }


def _stream_summarize_results_for_answer(
    query: str,
    results: list[SearchResult],
    *,
    emit: Any,
) -> list[dict[str, Any]]:
    summaries = [_stream_summarize_chunk(query, result, emit=emit) for result in results]
    while len(summaries) > 4:
        merged: list[dict[str, Any]] = []
        for index in range(0, len(summaries), 3):
            batch = summaries[index:index + 3]
            if len(batch) == 1:
                merged.append(batch[0])
            else:
                merged.append(_stream_merge_summary_batch(query, batch, emit=emit))
        summaries = merged
    return summaries


def _stream_generate_answer_with_trace(
    query: str,
    results: list[SearchResult],
    *,
    prioritized_results: list[SearchResult],
    summarized_evidence: list[dict[str, Any]],
    emit: Any,
) -> tuple[Any, dict[str, Any]]:
    trace: dict[str, Any] = {
        "relevance_review": {
            "provider": "ollama",
            "model": settings.ollama_fast_model,
            "prompt_kind": "relevance_judgment",
            "think": False,
        },
        "summarization": {
            "provider": "ollama",
            "model": settings.ollama_fast_model,
            "prompt_kind": "evidence_summary",
            "think": False,
            "summary_count": len(summarized_evidence),
        },
        "final_answer": {
            "provider": "ollama",
            "model": settings.ollama_answer_model,
            "prompt_kind": "final_answer",
            "think": False,
            "num_predict": -1,
            "used_fallback": False,
            "answer_source": "model",
            "fallback_reason": None,
            "summarized_evidence": summarized_evidence,
        },
    }
    if not results:
        answer = _fallback_answer(query, results)
        trace["final_answer"].update(
            {
                "used_fallback": True,
                "answer_source": "fallback_no_results",
                "fallback_reason": "No retrieval results were available.",
                "summarized_evidence": [],
            }
        )
        return answer, trace
    try:
        generated, _raw = _stream_llm_json(
            emit=emit,
            step_name="generate_answer",
            call_id="final_answer",
            label="Generate final answer",
            model=settings.ollama_answer_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\nEvidence summaries: {json.dumps(summarized_evidence)}"},
            ],
            json_schema=ANSWER_SCHEMA,
            think=False,
            timeout=90.0,
            purpose="final_answer",
            num_predict=-1,
        )
        from manuals_rag_schemas.documents import AnswerResponse

        generated_answer = AnswerResponse.model_validate(generated)
        validated_answer = validate_answer(generated_answer, prioritized_results)
        if validated_answer.answer != generated_answer.answer and any(
            "not sufficiently supported" in warning for warning in validated_answer.warnings
        ):
            trace["final_answer"].update(
                {
                    "used_fallback": True,
                    "answer_source": "fallback_validation",
                    "fallback_reason": "Generated answer was replaced by retrieval-grounded fallback during validation.",
                }
            )
        return validated_answer, trace
    except Exception as exc:
        logger.warning("Streaming final answer generation failed for model=%s; using fallback answer: %s", settings.ollama_answer_model, exc)
        fallback_answer = validate_answer(_fallback_answer(query, prioritized_results), prioritized_results)
        trace["final_answer"].update(
            {
                "used_fallback": True,
                "answer_source": "fallback_exception",
                "fallback_reason": str(exc),
            }
        )
        return fallback_answer, trace


def stream_query_debug_events(
    request: QueryRequest,
    *,
    sample_limit: int = 10,
) -> Any:
    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        events.append(event)

    def flush() -> Any:
        while events:
            yield events.pop(0)

    state: dict[str, Any] = {
        "query": request.query,
        "corpus_ids": request.corpus_ids,
        "request_filters": request.filters,
        "filters": request.filters,
        "step_timings_ms": {},
    }
    completed_steps: list[str] = []

    def step_event(step_name: str, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "event": f"step_{status}",
            "step": step_name,
            "label": dict(DEBUG_QUERY_STEP_SEQUENCE).get(step_name, step_name),
            "model": DEBUG_QUERY_STEP_MODELS.get(step_name),
            "completed_steps": completed_steps[:],
            "step_timings_ms": dict(state.get("step_timings_ms", {})),
            **extra,
        }

    def run_step(step_name: str, fn: Any) -> Any:
        emit(step_event(step_name, "started"))
        yield from flush()
        started = perf_counter()
        next_state = fn(state)
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        timings = dict(next_state.get("step_timings_ms", state.get("step_timings_ms", {})))
        timings[step_name] = elapsed_ms
        state.update(next_state)
        state["step_timings_ms"] = timings
        completed_steps.append(step_name)
        emit(step_event(step_name, "completed", duration_ms=elapsed_ms, payload=_stream_step_payload(step_name, state)))
        yield from flush()

    try:
        yield {"event": "run_started", "step_sequence": [{"name": name, "label": label, "model": DEBUG_QUERY_STEP_MODELS.get(name)} for name, label in DEBUG_QUERY_STEP_SEQUENCE]}
        for step_name, fn in (
            ("classify_query", classify_query),
            ("build_filters", build_query_filters),
            ("run_dense_search", run_dense),
            ("run_sparse_search", run_sparse),
            ("run_special_search", run_special),
            ("fuse_results", fuse),
            ("rerank_results", rerank),
            ("assemble_context", assemble),
        ):
            yield from run_step(step_name, fn)

        retrieval_results = [dict(result) for result in state.get("retrieval_results", [])]
        validated_results = [SearchResult.model_validate(result) for result in retrieval_results]
        candidate_results = validated_results[:8]

        emit(step_event("judge_answer_inputs", "started"))
        yield from flush()
        relevance_started = perf_counter()
        prioritized = _stream_prioritize_results_for_answer(request.query, candidate_results, emit=emit)
        yield from flush()
        relevance_duration_ms = round((perf_counter() - relevance_started) * 1000, 2)
        state["step_timings_ms"]["judge_answer_inputs"] = relevance_duration_ms
        state["candidate_results"] = candidate_results
        state["prioritized"] = prioritized
        completed_steps.append("judge_answer_inputs")
        emit(
            step_event(
                "judge_answer_inputs",
                "completed",
                duration_ms=relevance_duration_ms,
                payload=_stream_step_payload("judge_answer_inputs", state, sample_limit=sample_limit),
            )
        )
        yield from flush()

        emit(step_event("summarize_answer_inputs", "started"))
        yield from flush()
        summarization_started = perf_counter()
        summaries = _stream_summarize_results_for_answer(request.query, prioritized["prioritized_results"], emit=emit)
        yield from flush()
        summarization_duration_ms = round((perf_counter() - summarization_started) * 1000, 2)
        state["step_timings_ms"]["summarize_answer_inputs"] = summarization_duration_ms
        state["summaries"] = summaries
        completed_steps.append("summarize_answer_inputs")
        emit(
            step_event(
                "summarize_answer_inputs",
                "completed",
                duration_ms=summarization_duration_ms,
                payload=_stream_step_payload("summarize_answer_inputs", state, sample_limit=sample_limit),
            )
        )
        yield from flush()

        emit(step_event("generate_answer", "started"))
        yield from flush()
        answer_started = perf_counter()
        answer, answer_trace = _stream_generate_answer_with_trace(
            request.query,
            validated_results,
            prioritized_results=prioritized["prioritized_results"],
            summarized_evidence=summaries,
            emit=emit,
        )
        yield from flush()
        answer_duration_ms = round((perf_counter() - answer_started) * 1000, 2)
        state["step_timings_ms"]["generate_answer"] = answer_duration_ms
        state["answer"] = answer
        state["answer_trace"] = answer_trace
        completed_steps.append("generate_answer")
        emit(
            step_event(
                "generate_answer",
                "completed",
                duration_ms=answer_duration_ms,
                payload=_stream_step_payload("generate_answer", state, sample_limit=sample_limit),
            )
        )
        yield from flush()

        result = _query_debug_payload(
            request,
            state=state,
            prioritized=prioritized,
            summaries=summaries,
            answer=answer,
            answer_trace=answer_trace,
            relevance_duration_ms=relevance_duration_ms,
            summarization_duration_ms=summarization_duration_ms,
            completed_steps=completed_steps,
            sample_limit=sample_limit,
        )
        yield {"event": "run_completed", "result": result}
    except Exception as exc:
        logger.exception("Streaming query debug run failed")
        yield {"event": "run_failed", "error": str(exc)}


def build_document_debug_snapshot(document_id: str, *, sample_limit: int = 25) -> dict[str, Any] | None:
    document = fetch_one(
        """
        select
            sd.id as document_id,
            sd.corpus_id,
            sd.title,
            sd.source_filename,
            sd.manufacturer,
            sd.product_family,
            sd.product_model,
            sd.document_kind,
            sd.ingest_status,
            sd.current_version_id as version_id,
            dv.page_count,
            dv.parse_profile,
            dv.quality_score,
            dv.parse_warnings,
            dv.docling_artifact_uri,
            dv.ingested_at
        from source_documents sd
        join document_versions dv on dv.id = sd.current_version_id
        where sd.id = %s
        """,
        (document_id,),
    )
    if not document:
        return None

    node_type_counts = [
        dict(row)
        for row in fetch_all(
            """
            select node_type, count(*) as count
            from logical_nodes
            where document_version_id = %s
            group by node_type
            order by node_type
            """,
            (document["version_id"],),
        )
    ]
    logical_nodes = [
        {
            "id": row["id"],
            "ordinal": row["ordinal"],
            "node_type": row["node_type"],
            "heading_text": row["heading_text"],
            "section_path": row["section_path_json"] or [],
            "page_from": row["page_from"],
            "page_to": row["page_to"],
            "warning_level": row["warning_level"],
            "procedure_step_number": row["procedure_step_number"],
            "spec_name": row["spec_name"],
            "spec_value": row["spec_value"],
            "spec_unit": row["spec_unit"],
            "keywords": row["keywords_json"] or [],
            "citability_score": row["citability_score"],
            "token_count": row["token_count"],
            "table_json": row["table_json"],
            "text_preview": str(row["text_normalized"] or row["text_raw"] or "")[:280],
            "text": str(row["text_normalized"] or row["text_raw"] or ""),
        }
        for row in fetch_all(
            """
            select *
            from logical_nodes
            where document_version_id = %s
            order by ordinal
            limit %s
            """,
            (document["version_id"], sample_limit),
        )
    ]
    chunk_type_counts = [
        dict(row)
        for row in fetch_all(
            """
            select chunk_type, count(*) as count
            from retrieval_chunks
            where document_version_id = %s
            group by chunk_type
            order by chunk_type
            """,
            (document["version_id"],),
        )
    ]
    retrieval_chunks = [
        {
            "id": row["id"],
            "chunk_type": row["chunk_type"],
            "chunk_level": row["chunk_level"],
            "title": row["title"],
            "section_path_text": row["section_path_text"],
            "page_from": row["page_from"],
            "page_to": row["page_to"],
            "priority_score": row["priority_score"],
            "logical_node_ids": row["logical_node_ids_json"] or [],
            "content_preview": str(row["content"] or "")[:280],
            "content": str(row["content"] or ""),
            "content_for_rerank": row["content_for_rerank"],
            "metadata": row["metadata_json"] or {},
        }
        for row in fetch_all(
            """
            select *
            from retrieval_chunks
            where document_version_id = %s
            order by priority_score desc, page_from asc, page_to asc, chunk_level asc
            limit %s
            """,
            (document["version_id"], sample_limit),
        )
    ]
    return {
        "document": dict(document),
        "logical_nodes": {
            "counts_by_type": node_type_counts,
            "sample_count": len(logical_nodes),
            "samples": logical_nodes,
        },
        "retrieval_chunks": {
            "counts_by_type": chunk_type_counts,
            "sample_count": len(retrieval_chunks),
            "samples": retrieval_chunks,
        },
    }


def build_document_metadata_snapshot(document_id: str, *, page: int | None = None) -> dict[str, Any] | None:
    document = fetch_one(
        """
        select
            sd.id as document_id,
            sd.corpus_id,
            sd.title,
            sd.source_filename,
            sd.manufacturer,
            sd.product_family,
            sd.product_model,
            sd.document_kind,
            sd.ingest_status,
            sd.current_version_id as version_id,
            dv.page_count,
            dme.model as metadata_model,
            dme.metadata_json as extracted_metadata,
            dme.extracted_at as metadata_extracted_at
        from source_documents sd
        join document_versions dv on dv.id = sd.current_version_id
        left join document_metadata_extractions dme on dme.source_document_id = sd.id
        where sd.id = %s
        """,
        (document_id,),
    )
    if not document:
        return None

    page_rows = fetch_all(
        """
        select generate_series(1, greatest(dv.page_count, 1)) as page
        from document_versions dv
        where dv.id = %s
        """,
        (document["version_id"],),
    )
    pages = [int(row["page"]) for row in page_rows]
    selected_page = page or (pages[0] if pages else 1)

    chunk_rows = fetch_all(
        """
        select
            id,
            chunk_type,
            chunk_level,
            title,
            section_path_text,
            page_from,
            page_to,
            priority_score,
            metadata_json,
            content
        from retrieval_chunks
        where document_version_id = %s
          and page_from <= %s
          and page_to >= %s
        order by priority_score desc, chunk_level asc, page_from asc, id asc
        limit 200
        """,
        (document["version_id"], selected_page, selected_page),
    )
    metadata_keys = [
        "manufacturer",
        "companies",
        "product_family",
        "product_model",
        "product_families",
        "product_models",
        "devices",
        "part_numbers",
        "document_protocol_terms",
        "settings",
        "parameters",
        "document_menu_labels",
        "document_topics",
        "document_kind",
        "revision_date",
        "version_label",
    ]
    chunks = []
    for row in chunk_rows:
        metadata = dict(row["metadata_json"] or {})
        chunks.append(
            {
                "id": row["id"],
                "chunk_type": row["chunk_type"],
                "chunk_level": row["chunk_level"],
                "title": row["title"],
                "section_path_text": row["section_path_text"],
                "page_from": row["page_from"],
                "page_to": row["page_to"],
                "priority_score": row["priority_score"],
                "metadata": {key: metadata.get(key) for key in metadata_keys if key in metadata},
                "content_preview": str(row["content"] or "")[:300],
            }
        )

    return {
        "document": dict(document),
        "pages": pages,
        "selected_page": selected_page,
        "page_chunks": chunks,
    }
