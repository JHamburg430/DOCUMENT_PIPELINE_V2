from __future__ import annotations

from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from manuals_rag_answering.generator import generate_answer
from manuals_rag_retrieval.qdrant_store import QdrantStore
from manuals_rag_retrieval.retriever import (
    _annotate_completeness,
    _apply_family_scoring,
    _apply_query_alignment,
    _contextual_lexical_limit,
    _dedupe_results,
    _preserve_identifier_dense_candidates,
    _promote_identifier_contextual_candidates,
    _promote_measurement_candidates,
    _promote_named_operation_candidates,
    _promote_named_setting_candidates,
    _promote_comparison_table_candidates,
    _promote_structured_table_candidates,
    _promote_troubleshooting_table_candidates,
    _promote_wiring_terminal_candidates,
    _select_family_candidates,
    _should_run_table_lexical_search,
    _troubleshooting_table_siblings,
    assemble_context,
    build_filters,
    enrich_candidates_for_rerank,
    fuse_results,
    rerank_results,
    retrieve,
    run_contextual_lexical_search,
    run_dense_search,
    run_sparse_search,
    run_special_search,
    run_table_search,
    run_table_lexical_search,
)
from manuals_rag_retrieval.query_analysis import QueryAnalysis, analyze_query


class QueryState(TypedDict, total=False):
    query: str
    corpus_ids: list[str]
    filters: dict[str, object]
    analysis: dict
    dense_results: list[dict]
    sparse_results: list[dict]
    special_results: list[dict]
    table_results: list[dict]
    contextual_lexical_results: list[dict]
    table_lexical_results: list[dict]
    fused_results: list[dict]
    retrieval_results: list[dict]
    answer: dict
    retrieval_trace: dict
    step_timings_ms: dict[str, float]


def classify_query(state: QueryState) -> QueryState:
    return {**state, "analysis": analyze_query(state["query"]).__dict__}


def build_query_filters(state: QueryState) -> QueryState:
    return {**state, "filters": build_filters(state["query"], state.get("filters", {}))}


def run_dense(state: QueryState) -> QueryState:
    store = QdrantStore()
    results = run_dense_search(store, state["query"], state["corpus_ids"], state["filters"])
    return {**state, "dense_results": [result.model_dump() for result in results]}


def run_sparse(state: QueryState) -> QueryState:
    store = QdrantStore()
    results = run_sparse_search(store, state["query"], state["corpus_ids"], state["filters"])
    return {**state, "sparse_results": [result.model_dump() for result in results]}


def run_special(state: QueryState) -> QueryState:
    from manuals_rag_retrieval.query_analysis import QueryAnalysis
    from manuals_rag_schemas.documents import SearchResult

    store = QdrantStore()
    analysis = QueryAnalysis(**state["analysis"])
    dense_results = [SearchResult.model_validate(item) for item in state.get("dense_results", [])]
    special = run_special_search(store, state["query"], state["corpus_ids"], state["filters"], analysis)
    table = run_table_search(store, state["query"], state["corpus_ids"], state["filters"])
    contextual = run_contextual_lexical_search(
        state["query"],
        state["corpus_ids"],
        state.get("request_filters", state["filters"]),
        analysis,
        limit=_contextual_lexical_limit(state["query"]),
    )
    return {
        **state,
        "special_results": [result.model_dump() for result in special],
        "table_results": [result.model_dump() for result in table],
        "contextual_lexical_results": [result.model_dump() for result in contextual],
        "dense_results": [r.model_dump() for r in dense_results],
    }


def fuse(state: QueryState) -> QueryState:
    from manuals_rag_schemas.documents import SearchResult

    store = QdrantStore()
    analysis = QueryAnalysis(**state["analysis"])
    table_lexical_results = (
        run_table_lexical_search(state["query"], state["corpus_ids"], state.get("request_filters", state["filters"]), analysis)
        if _should_run_table_lexical_search(analysis)
        else []
    )
    result_sets = [
        [SearchResult.model_validate(item) for item in state.get("dense_results", [])],
        [SearchResult.model_validate(item) for item in state.get("sparse_results", [])],
        [SearchResult.model_validate(item) for item in state.get("table_results", [])],
        table_lexical_results,
        [SearchResult.model_validate(item) for item in state.get("contextual_lexical_results", [])],
        [SearchResult.model_validate(item) for item in state.get("special_results", [])],
    ]
    fused = fuse_results(store, result_sets, limit=30)
    fused = _preserve_identifier_dense_candidates(fused, result_sets[0], analysis, limit=30)
    return {
        **state,
        "table_lexical_results": [result.model_dump() for result in table_lexical_results],
        "fused_results": [result.model_dump() for result in fused],
    }


def rerank(state: QueryState) -> QueryState:
    from manuals_rag_schemas.documents import SearchResult

    analysis = QueryAnalysis(**state["analysis"])
    fused = [SearchResult.model_validate(item) for item in state.get("fused_results", [])]
    table_lexical_results = [SearchResult.model_validate(item) for item in state.get("table_lexical_results", [])]
    contextual_results = [
        SearchResult.model_validate(item) for item in state.get("contextual_lexical_results", [])
    ]
    dense_results = [SearchResult.model_validate(item) for item in state.get("dense_results", [])]
    sparse_results = [SearchResult.model_validate(item) for item in state.get("sparse_results", [])]
    special_results = [SearchResult.model_validate(item) for item in state.get("special_results", [])]
    rescored = _apply_family_scoring(fused, analysis, stage="family_scored")[:30]
    completed = _annotate_completeness(rescored)
    aligned = _apply_query_alignment(completed, analysis, stage="query_aligned")
    family_selected = _select_family_candidates(aligned, analysis, filters=state["filters"], limit=12)
    enriched = enrich_candidates_for_rerank(family_selected, analysis, limit=12)
    reranked = rerank_results(enriched, state["query"], limit=12)
    troubleshooting_siblings = _troubleshooting_table_siblings(
        [*reranked, *table_lexical_results],
        analysis,
    )
    troubleshooting_supplemental = [*table_lexical_results, *troubleshooting_siblings]
    reranked = _promote_structured_table_candidates(reranked, table_lexical_results, analysis, limit=12)
    reranked = _promote_comparison_table_candidates(reranked, table_lexical_results, analysis, limit=12)
    reranked = _promote_troubleshooting_table_candidates(
        reranked,
        troubleshooting_supplemental,
        analysis,
        limit=12,
    )
    reranked = _promote_identifier_contextual_candidates(
        reranked,
        contextual_results,
        analysis,
        limit=12,
    )
    reranked = _promote_named_setting_candidates(
        reranked,
        [*contextual_results, *table_lexical_results, *fused],
        state["query"],
        limit=12,
    )
    reranked = _promote_named_operation_candidates(
        reranked,
        [
            *contextual_results,
            *table_lexical_results,
            *dense_results,
            *sparse_results,
            *special_results,
            *fused,
        ],
        state["query"],
        limit=12,
    )
    reranked = _promote_wiring_terminal_candidates(
        reranked,
        [*contextual_results, *table_lexical_results, *dense_results, *fused],
        state["query"],
        limit=12,
    )
    reranked = _promote_measurement_candidates(
        reranked,
        [*contextual_results, *dense_results, *sparse_results, *special_results, *fused],
        state["query"],
        analysis=analysis,
        limit=12,
    )
    return {**state, "retrieval_results": [result.model_dump() for result in reranked]}


def assemble(state: QueryState) -> QueryState:
    from manuals_rag_schemas.documents import SearchResult

    reranked = [SearchResult.model_validate(item) for item in state.get("retrieval_results", [])]
    analysis = QueryAnalysis(**state["analysis"])
    assembled = assemble_context(_dedupe_results(reranked, analysis))
    return {**state, "retrieval_results": [result.model_dump() for result in assembled]}


def retrieve_documents(state: QueryState) -> QueryState:
    results = retrieve(state["query"], state["corpus_ids"], state["filters"])
    retrieval_trace = {}
    if results:
        retrieval_trace = dict(results[0].metadata.get("corrective_retrieval") or {})
    return {
        **state,
        "retrieval_results": [result.model_dump() for result in results],
        "retrieval_trace": retrieval_trace,
    }


def validate_or_answer(state: QueryState) -> QueryState:
    from manuals_rag_schemas.documents import SearchResult

    results = [SearchResult.model_validate(item) for item in state.get("retrieval_results", [])]
    answer = generate_answer(state["query"], results)
    return {**state, "answer": answer.model_dump()}


def _timed_node(name: str, fn):
    def wrapped(state: QueryState) -> QueryState:
        started = perf_counter()
        next_state = fn(state)
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        timings = dict(next_state.get("step_timings_ms", state.get("step_timings_ms", {})))
        timings[name] = elapsed_ms
        return {**next_state, "step_timings_ms": timings}

    return wrapped


def build_workflow(*, include_answer: bool = True):
    graph = StateGraph(QueryState)
    graph.add_node("classify_query", _timed_node("classify_query", classify_query))
    graph.add_node("build_filters", _timed_node("build_filters", build_query_filters))
    if include_answer:
        graph.add_node("retrieve_documents", _timed_node("retrieve_documents", retrieve_documents))
        graph.add_node("generate_answer", _timed_node("generate_answer", validate_or_answer))
        graph.add_edge(START, "classify_query")
        graph.add_edge("classify_query", "build_filters")
        graph.add_edge("build_filters", "retrieve_documents")
        graph.add_edge("retrieve_documents", "generate_answer")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    graph.add_node("run_dense_search", _timed_node("run_dense_search", run_dense))
    graph.add_node("run_sparse_search", _timed_node("run_sparse_search", run_sparse))
    graph.add_node("run_special_search", _timed_node("run_special_search", run_special))
    graph.add_node("fuse_results", _timed_node("fuse_results", fuse))
    graph.add_node("rerank_results", _timed_node("rerank_results", rerank))
    graph.add_node("assemble_context", _timed_node("assemble_context", assemble))
    graph.add_node("generate_answer", _timed_node("generate_answer", validate_or_answer))
    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "build_filters")
    graph.add_edge("build_filters", "run_dense_search")
    graph.add_edge("run_dense_search", "run_sparse_search")
    graph.add_edge("run_sparse_search", "run_special_search")
    graph.add_edge("run_special_search", "fuse_results")
    graph.add_edge("fuse_results", "rerank_results")
    graph.add_edge("rerank_results", "assemble_context")
    graph.add_edge("assemble_context", END)
    return graph.compile()
