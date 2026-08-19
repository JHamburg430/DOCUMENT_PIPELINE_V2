from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import streamlit as st


API_BASE = os.getenv("MANUALS_RAG_API_BASE", "http://127.0.0.1:8600")
AUTH_TOKEN = os.getenv("MANUALS_RAG_AUTH_TOKEN", "admin-token")
DEFAULT_CORPUS = os.getenv("MANUALS_RAG_DEFAULT_CORPUS", "manuals_vendor_keyence")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def _get(path: str, **params: Any) -> Any:
    with httpx.Client(base_url=API_BASE, timeout=60.0) as client:
        response = client.get(path, headers=_headers(), params=params or None)
        response.raise_for_status()
        return response.json()


def _post(path: str, payload: dict[str, Any], **params: Any) -> Any:
    with httpx.Client(base_url=API_BASE, timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        response = client.post(path, headers=_headers(), json=payload, params=params or None)
        response.raise_for_status()
        return response.json()


def _stream_events(path: str, payload: dict[str, Any], **params: Any) -> Any:
    timeout = httpx.Timeout(900.0, connect=10.0, read=900.0)
    with httpx.Client(base_url=API_BASE, timeout=timeout) as client:
        with client.stream("POST", path, headers=_headers(), json=payload, params=params or None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                yield json.loads(line)


def _run_query_debug(payload: dict[str, Any], *, sample_limit: int) -> dict[str, Any]:
    if hasattr(st, "status"):
        with st.status("Submitting debug request...", expanded=True) as status:
            st.write("1. Creating debug run")
            run = _post("/debug/query-runs", payload, sample_limit=sample_limit)
            run_id = run["run_id"]
            progress_placeholder = st.empty()
            while True:
                polled = _get(f"/debug/query-runs/{run_id}")
                progress = polled.get("progress", {})
                current_label = progress.get("current_label") or polled.get("status", "Running")
                current_model = progress.get("current_model")
                step_sequence = progress.get("step_sequence", [])
                if current_model:
                    status.update(label=f"{current_label} [{current_model}]", state="running")
                else:
                    status.update(label=current_label, state="running")
                progress_lines = []
                for index, step in enumerate(step_sequence, start=1):
                    marker = "x" if step.get("done") else "…"
                    model_suffix = f" [{step.get('model')}]" if step.get("model") else ""
                    progress_lines.append(f"{index}. [{marker}] {step.get('label')}{model_suffix}")
                progress_placeholder.markdown("\n".join(progress_lines))
                if polled.get("status") == "completed":
                    status.update(label="Debug run completed", state="complete")
                    return polled["result"]
                if polled.get("status") == "failed":
                    status.update(label="Debug run failed", state="error")
                    raise RuntimeError(polled.get("error") or "Debug run failed.")
                time.sleep(0.5)
    with st.spinner("Submitting debug request..."):
        run = _post("/debug/query-runs", payload, sample_limit=sample_limit)
        run_id = run["run_id"]
        while True:
            polled = _get(f"/debug/query-runs/{run_id}")
            if polled.get("status") == "completed":
                return polled["result"]
            if polled.get("status") == "failed":
                raise RuntimeError(polled.get("error") or "Debug run failed.")
            time.sleep(0.5)


def _sample_selector(label: str, items: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not items:
        st.info(f"No samples for {label}.")
        return None
    options = list(range(len(items)))
    selected = st.selectbox(
        label,
        options=options,
        format_func=lambda index: f"{index + 1}. {items[index].get('chunk_type') or items[index].get('node_type') or items[index].get('id')}",
        key=key,
    )
    return items[selected]


def _render_json_block(label: str, payload: Any) -> None:
    with st.expander(label, expanded=False):
        st.json(payload, expanded=False)


def _format_duration(duration_ms: Any) -> str:
    if duration_ms is None:
        return "n/a"
    return f"{float(duration_ms):,.2f} ms"


def _document_label(document: dict[str, Any]) -> str:
    return f"{document.get('title')} | {document.get('source_filename')} | {document.get('document_id')}"


def _metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _render_step_table(step_sequence: list[dict[str, Any]], step_state: dict[str, dict[str, Any]]) -> None:
    rows = []
    for index, step in enumerate(step_sequence, start=1):
        name = step["name"]
        state = step_state.get(name, {})
        rows.append(
            {
                "#": index,
                "Step": step["label"],
                "Model": step.get("model") or "",
                "Status": state.get("status", "pending"),
                "Duration": _format_duration(state.get("duration_ms")) if state.get("duration_ms") is not None else "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_llm_streams(llm_outputs: dict[str, dict[str, Any]]) -> None:
    if not llm_outputs:
        st.info("Model output appears here while relevance review, evidence summaries, and final answer generation run.")
        return
    for call_id, call in llm_outputs.items():
        st.markdown(f"**{call.get('label') or call_id}**")
        st.caption(f"{call.get('model') or ''} | {call.get('status', 'running')}")
        text = str(call.get("text", ""))
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            if parsed.get("answer"):
                st.write(str(parsed.get("answer")))
            if parsed.get("citations"):
                st.dataframe(parsed.get("citations"), hide_index=True, use_container_width=True)
            with st.expander("Raw model payload", expanded=False):
                st.json(parsed, expanded=False)
        else:
            st.code(text, language="text")


def _latest_completed_eval() -> dict[str, Any] | None:
    try:
        runs = _get("/runs", run_type="end_to_end_eval", limit=10)
    except Exception:
        return None
    for run in runs:
        if run.get("status") == "completed" and run.get("result_json"):
            return run
    return None


def _short_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _pass_label(value: Any) -> str:
    return "pass" if value else "fail"


def _render_failure_reasons(label: str, reasons: list[Any]) -> None:
    if reasons:
        st.error(f"{label}: {', '.join(str(reason) for reason in reasons)}")
    else:
        st.success(f"{label}: passed")


def _render_final_query_result(payload: dict[str, Any]) -> None:
    meta_left, meta_mid, meta_right = st.columns(3)
    with meta_left:
        _render_json_block("Query Analysis", payload.get("analysis", {}))
    with meta_mid:
        _render_json_block("Applied Filters", payload.get("applied_filters", {}))
    with meta_right:
        _render_json_block("Step Timings", payload.get("step_timings_ms", {}))
    st.markdown("### Retrieval Stages")
    for stage in payload.get("stages", []):
        with st.expander(f"{stage.get('name')} | {stage.get('count', 0)} results | {_format_duration(stage.get('duration_ms'))}", expanded=False):
            samples = stage.get("samples", [])
            if samples:
                _render_json_block("First Sample", samples[0])
    st.markdown("### Final Answer")
    _render_json_block("Answer Payload", payload.get("answer", {}))
    _render_json_block("Answer Trace", payload.get("answer_generation_trace", {}))


def _consume_streaming_query(payload: dict[str, Any], *, sample_limit: int) -> None:
    step_sequence: list[dict[str, Any]] = []
    step_state: dict[str, dict[str, Any]] = {}
    llm_outputs: dict[str, dict[str, Any]] = {}
    final_result: dict[str, Any] | None = None
    progress_placeholder = st.empty()
    llm_placeholder = st.empty()
    result_placeholder = st.empty()
    for event in _stream_events("/debug/query-stream", payload, sample_limit=sample_limit):
        event_type = event.get("event")
        if event_type == "run_started":
            step_sequence = event.get("step_sequence", [])
        elif event_type == "step_started":
            step_state[event["step"]] = {"status": "running"}
        elif event_type == "step_completed":
            step_state[event["step"]] = {"status": "completed", "duration_ms": event.get("duration_ms")}
        elif event_type == "llm_call_started":
            llm_outputs[event["call_id"]] = {"label": event.get("label"), "model": event.get("model"), "status": "running", "text": ""}
        elif event_type == "llm_token":
            call = llm_outputs.setdefault(event["call_id"], {"status": "running", "text": ""})
            call["text"] = f"{call.get('text', '')}{event.get('token', '')}"
        elif event_type == "llm_call_completed":
            call = llm_outputs.setdefault(event["call_id"], {"text": ""})
            call.update({"label": event.get("label") or call.get("label"), "model": event.get("model") or call.get("model"), "status": "completed", "text": event.get("raw_response") or call.get("text", "")})
        elif event_type == "run_completed":
            final_result = event.get("result", {})
        elif event_type == "run_failed":
            st.error(event.get("error") or "Streaming query failed.")
        with progress_placeholder.container():
            st.markdown("### Progress")
            _render_step_table(step_sequence, step_state)
        with llm_placeholder.container():
            st.markdown("### Streaming Model Output")
            _render_llm_streams(llm_outputs)
        if final_result:
            with result_placeholder.container():
                _render_final_query_result(final_result)
    if final_result:
        st.session_state["streaming_query_result"] = final_result


def _consume_streaming_eval(payload: dict[str, Any], *, sample_limit: int) -> None:
    llm_outputs: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    step_sequence: list[dict[str, Any]] = []
    step_state: dict[str, dict[str, Any]] = {}
    final_result: dict[str, Any] | None = None
    run_id: str | None = None
    terminal_error: str | None = None
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    llm_placeholder = st.empty()
    result_placeholder = st.empty()
    try:
        for event in _stream_events("/eval/end-to-end-stream", payload, sample_limit=sample_limit):
            run_id = event.get("run_id") or run_id
            event_type = event.get("event")
            if event_type == "eval_started":
                status_placeholder.info(f"Started run {run_id} with {event.get('total_questions', 0)} generated questions.")
            elif event_type == "eval_question_started":
                case = event.get("case", {})
                status_placeholder.info(f"Question {event.get('question_index')}/{event.get('total_questions')}: {case.get('query')}")
            elif event_type == "eval_query_event":
                query_event = event.get("query_event", {})
                nested_type = query_event.get("event")
                if nested_type == "run_started":
                    step_sequence = query_event.get("step_sequence", [])
                    step_state = {}
                elif nested_type == "step_started":
                    step_state[query_event["step"]] = {"status": "running"}
                elif nested_type == "step_completed":
                    step_state[query_event["step"]] = {
                        "status": "completed",
                        "duration_ms": query_event.get("duration_ms"),
                    }
                if nested_type == "llm_call_started":
                    call_id = f"q{event.get('question_index')}:{query_event.get('call_id')}"
                    llm_outputs[call_id] = {"label": query_event.get("label"), "model": query_event.get("model"), "status": "running", "text": ""}
                elif nested_type == "llm_token":
                    call_id = f"q{event.get('question_index')}:{query_event.get('call_id')}"
                    call = llm_outputs.setdefault(call_id, {"status": "running", "text": ""})
                    call["text"] = f"{call.get('text', '')}{query_event.get('token', '')}"
                elif nested_type == "llm_call_completed":
                    call_id = f"q{event.get('question_index')}:{query_event.get('call_id')}"
                    call = llm_outputs.setdefault(call_id, {"text": ""})
                    call.update({"label": query_event.get("label") or call.get("label"), "model": query_event.get("model") or call.get("model"), "status": "completed", "text": query_event.get("raw_response") or call.get("text", "")})
            elif event_type == "eval_question_completed":
                items.append(event.get("item", {}))
                partial = {"summary": event.get("summary", {}), "items": items, "warnings": []}
                with result_placeholder.container():
                    _render_end_to_end_eval(partial, key_prefix="streaming_eval_partial")
            elif event_type == "eval_completed":
                final_result = event.get("result", {})
                status_placeholder.success(f"Completed run {run_id}")
            elif event_type == "eval_failed":
                terminal_error = event.get("error") or "Evaluation failed."
                status_placeholder.error(terminal_error)
            with progress_placeholder.container():
                st.markdown("### Current Question Progress")
                _render_step_table(step_sequence, step_state)
            with llm_placeholder.container():
                st.markdown("### Streaming Model Output")
                _render_llm_streams(llm_outputs)
    except httpx.HTTPError as exc:
        terminal_error = str(exc)
        status_placeholder.warning(f"Live stream ended early: {terminal_error}")
    if not final_result and run_id:
        persisted = _get(f"/runs/{run_id}")
        if persisted.get("status") == "completed" and persisted.get("result_json"):
            final_result = persisted["result_json"]
            status_placeholder.success(f"Completed run {run_id}")
        elif persisted.get("status") == "failed":
            status_placeholder.error(persisted.get("error") or terminal_error or "Evaluation failed.")
        else:
            status_placeholder.info(f"Run {run_id} is still running. Open Run History to inspect progress.")
    if final_result:
        st.session_state["end_to_end_eval_payload"] = final_result
        st.session_state["end_to_end_eval_run_meta"] = {"id": run_id, "source": "current streamed run"}
        st.rerun()


def _render_stage(stage: dict[str, Any], key_prefix: str) -> None:
    st.markdown(f"### {stage['name']}")
    st.caption(f"{stage['count']} results | {_format_duration(stage.get('duration_ms'))}")
    sample = _sample_selector(f"{stage['name']} sample", stage.get("samples", []), f"{key_prefix}_{stage['name']}")
    if sample:
        left, right = st.columns(2)
        with left:
            _render_json_block("Result Summary", {k: sample.get(k) for k in ("chunk_id", "score", "chunk_type", "title", "pages", "section_path", "retrieval_stage")})
            _render_json_block("Metadata", sample.get("metadata", {}))
        with right:
            _render_json_block("Content", {"content": sample.get("content", ""), "context_window": sample.get("context_window"), "parent_context": sample.get("parent_context")})


def _default_corpus_ids(documents: list[dict[str, Any]]) -> str:
    corpus_ids = sorted({str(item.get("corpus_id")) for item in documents if item.get("corpus_id")})
    return ",".join(corpus_ids) if corpus_ids else DEFAULT_CORPUS


def _normalize_search_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if isinstance(payload, dict):
        results = payload.get("results", [])
        return results if isinstance(results, list) else [], payload.get("source_assets", {})
    return [], {}


def _selected_document_hit(metadata: dict[str, Any]) -> dict[str, Any]:
    hits = metadata.get("selected_document_metadata_hits") or []
    return hits[0] if hits and isinstance(hits[0], dict) else {}


def _render_consolidated_search_results(payload: Any, *, key_prefix: str = "search") -> None:
    results, source_assets = _normalize_search_payload(payload)
    if not results:
        st.info("No search results returned.")
        return

    st.markdown("### Results")
    rows = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        selected_hit = _selected_document_hit(metadata)
        rows.append(
            {
                "Rank": index,
                "Score": result.get("score"),
                "Chunk Type": metadata.get("chunk_type"),
                "Stage": metadata.get("retrieval_stage"),
                "Document Selection": metadata.get("document_selection_stage"),
                "Selected Document": selected_hit.get("source_document_id") or result.get("source_document_id"),
                "Title": result.get("title"),
                "Pages": result.get("pages"),
                "Preview": str(result.get("content") or "")[:260],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)

    selected_index = st.selectbox(
        "Inspect search result",
        options=list(range(len(results))),
        format_func=lambda index: f"{index + 1}. {results[index].get('score', 0):.4f} | {results[index].get('title')} | p.{results[index].get('pages')}",
        key=f"{key_prefix}_inspect_search_result",
    )
    result = results[selected_index]
    metadata = result.get("metadata") or {}
    selected_hit = _selected_document_hit(metadata)
    left, right = st.columns(2)
    with left:
        _render_json_block(
            "Result",
            {
                key: result.get(key)
                for key in ("chunk_id", "score", "title", "source_document_id", "document_version_id", "pages", "section_path")
            },
        )
        _render_json_block("Selected Document Metadata Hit", selected_hit)
        _render_json_block("Metadata", metadata)
    with right:
        st.markdown("**Content**")
        st.code(str(result.get("content") or ""), language="text")
        if source_assets:
            _render_json_block("Source Assets", source_assets)


def _render_end_to_end_eval(payload: dict[str, Any], *, key_prefix: str = "eval") -> None:
    summary = payload.get("summary", {})
    metric_cols = st.columns(4)
    metric_cols[0].metric("Questions", summary.get("total_questions", 0))
    metric_cols[1].metric("Retrieval Correct", f"{summary.get('retrieval_correct_percent', 0.0):.2f}%")
    metric_cols[2].metric("Answers Correct", f"{summary.get('answers_correct_percent', 0.0):.2f}%")
    metric_cols[3].metric("Answer Passes", summary.get("answers_correct", 0))

    for warning in payload.get("warnings", []):
        st.warning(warning)

    items = payload.get("items", [])
    if not items:
        st.info("No evaluation items returned.")
        return

    rows = []
    for index, item in enumerate(items, start=1):
        case = item.get("case", {})
        retrieval = item.get("retrieval_evaluation", {})
        answer_eval = item.get("answer_evaluation", {})
        rows.append(
            {
                "Item": index,
                "Question": case.get("query"),
                "Expected Document": case.get("source_filename"),
                "Chunk Type": case.get("chunk_type"),
                "Retrieval": "pass" if retrieval.get("passed") else "fail",
                "Rank": retrieval.get("rank"),
                "Answer": "pass" if answer_eval.get("passed") else "fail",
                "Answer Failures": ", ".join(answer_eval.get("failure_reasons", [])),
                "Answer Preview": _short_text((item.get("answer") or {}).get("answer"), 180),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)

    selected_index = st.selectbox(
        "Review eval item",
        options=list(range(len(items))),
        format_func=lambda index: f"{index + 1}. {items[index].get('case', {}).get('query')}",
        key=f"{key_prefix}_review_eval_item",
    )
    item = items[selected_index]
    case = item.get("case", {})
    answer = item.get("answer", {})
    retrieval = item.get("retrieval_evaluation", {})
    answer_eval = item.get("answer_evaluation", {})

    st.markdown("### Review")
    result_cols = st.columns(4)
    result_cols[0].metric("Retrieval", _pass_label(retrieval.get("passed")))
    result_cols[1].metric("Rank", retrieval.get("rank") or "not found")
    result_cols[2].metric("Answer", _pass_label(answer_eval.get("passed")))
    result_cols[3].metric("Expected Terms", _pass_label((answer_eval.get("term_check") or {}).get("passed")))
    _render_failure_reasons("Retrieval", retrieval.get("failure_reasons", []))
    _render_failure_reasons("Answer", answer_eval.get("failure_reasons", []))

    st.markdown("### Question")
    st.write(case.get("query") or "")

    left, right = st.columns(2)
    with left:
        st.markdown("### Expected")
        st.dataframe(
            [
                {
                    "Document": case.get("source_filename"),
                    "Title": case.get("source_title"),
                    "Chunk Type": case.get("chunk_type"),
                    "Pages": f"{case.get('page_from') or ''}-{case.get('page_to') or ''}".strip("-"),
                    "Expected Terms": ", ".join(str(term) for term in case.get("expected_terms", [])),
                    "Generated By": case.get("generation_method"),
                }
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Expected Snippet**")
        st.code(str(case.get("expected_snippet") or ""), language="text")
    with right:
        st.markdown("### Generated Answer")
        st.write(str(answer.get("answer") or ""))
        citations = answer.get("citations", [])
        if citations:
            st.markdown("**Citations**")
            st.dataframe(citations, hide_index=True, use_container_width=True)
        used_documents = answer.get("used_documents", [])
        if used_documents:
            st.markdown("**Used Documents**")
            st.dataframe(used_documents, hide_index=True, use_container_width=True)
        for warning in answer.get("warnings", []):
            st.warning(warning)

    with st.expander("Detailed Scoring JSON", expanded=False):
        st.json({"retrieval_evaluation": retrieval, "answer_evaluation": answer_eval}, expanded=False)

    st.markdown("### Top Search Results")
    top_results = item.get("top_results", [])
    if not top_results:
        st.info("No search results recorded.")
        return
    st.dataframe(
        [
            {
                "Rank": index,
                "Score": result.get("score"),
                "Title": result.get("title"),
                "Pages": result.get("pages"),
                "Chunk": result.get("chunk_id"),
                "Preview": _short_text(result.get("content"), 240),
            }
            for index, result in enumerate(top_results, start=1)
        ],
        hide_index=True,
        use_container_width=True,
    )
    result_index = st.selectbox(
        "Inspect top result",
        options=list(range(len(top_results))),
        format_func=lambda index: f"{index + 1}. {top_results[index].get('score', 0):.4f} | {top_results[index].get('title')}",
        key=f"{key_prefix}_inspect_eval_top_result",
    )
    result = top_results[result_index]
    left, right = st.columns(2)
    with left:
        st.dataframe(
            [
                {
                    key: result.get(key)
                    for key in ("chunk_id", "score", "title", "source_document_id", "document_version_id", "pages", "section_path")
                }
            ],
            hide_index=True,
            use_container_width=True,
        )
        _render_json_block("Metadata", result.get("metadata", {}))
    with right:
        st.markdown("**Content**")
        st.code(str(result.get("content") or ""), language="text")


def _render_search_tab(documents: list[dict[str, Any]]) -> None:
    st.caption("Production retrieval: metadata document selection, dense retrieval, sparse retrieval, table retrieval, fusion, reranking, and assembly.")
    doc_options = {_document_label(item): item["document_id"] for item in documents}
    query_col, result_col = st.columns([1, 2])
    with query_col:
        query = st.text_area("Search Query", value="LJ-X8080 z axis repeatability", height=120, key="search_query")
        if "query_corpus_ids" not in st.session_state:
            st.session_state["query_corpus_ids"] = _default_corpus_ids(documents)
        corpus_ids_text = st.text_input("Corpus IDs", key="query_corpus_ids")
        selected_document = st.selectbox("Optional Document Filter", options=[""] + list(doc_options.keys()), key="search_document_filter")
        include_page_images = st.checkbox("Return page images", value=False, key="search_page_images")
        include_table_images = st.checkbox("Return table images", value=False, key="search_table_images")
        st.caption("No document filter is applied here; document selection is performed by metadata embeddings inside the API.")
        if st.button("Run Consolidated Search", use_container_width=True):
            corpus_ids = [item.strip() for item in corpus_ids_text.split(",") if item.strip()]
            if not corpus_ids:
                st.error("Corpus IDs cannot be empty.")
                st.stop()
            filters = {}
            if selected_document:
                filters["source_document_id"] = [doc_options[selected_document]]
            st.session_state["consolidated_search_payload"] = _post(
                "/search",
                {
                    "query": query,
                    "corpus_ids": corpus_ids,
                    "filters": filters,
                    "response_mode": "answer_with_citations",
                    "include_source_assets": include_page_images or include_table_images,
                    "include_page_images": include_page_images,
                    "include_table_images": include_table_images,
                },
            )
    with result_col:
        payload = st.session_state.get("consolidated_search_payload")
        if payload:
            _render_consolidated_search_results(payload, key_prefix="search")
        else:
            st.info("Run a search to inspect the consolidated retrieval results.")


def _render_streaming_query_tab(documents: list[dict[str, Any]], *, sample_limit: int) -> None:
    st.caption("Streams every model output in the query pipeline and shows progress for each retrieval and answer step.")
    doc_options = {_document_label(item): item["document_id"] for item in documents}
    documents_by_id = {item["document_id"]: item for item in documents}
    left, right = st.columns([1, 2])
    with left:
        query = st.text_area("Query", value="What product is described in this datasheet?", height=120, key="streaming_query_text")
        filter_document = st.selectbox("Filter to Document", options=[""] + list(doc_options.keys()), key="streaming_document_filter")
        selected_document_id = doc_options.get(filter_document) if filter_document else None
        selected_document = documents_by_id.get(selected_document_id) if selected_document_id else None
        default_corpus = str(selected_document["corpus_id"]) if selected_document else _default_corpus_ids(documents)
        corpus_ids_text = st.text_input("Corpus IDs", value=default_corpus, key="streaming_corpus_ids")
        filters = {"source_document_id": [selected_document_id]} if selected_document_id else {}
        if st.button("Run Streaming Query", use_container_width=True):
            corpus_ids = [item.strip() for item in corpus_ids_text.split(",") if item.strip()]
            if not corpus_ids:
                st.error("Corpus IDs cannot be empty.")
                st.stop()
            st.session_state.pop("streaming_query_result", None)
            _consume_streaming_query(
                {"query": query, "corpus_ids": corpus_ids, "filters": filters, "response_mode": "answer_with_citations"},
                sample_limit=sample_limit,
            )
    with right:
        if st.session_state.get("streaming_query_result"):
            _render_final_query_result(st.session_state["streaming_query_result"])
        else:
            st.info("Start a run to watch step progress and token streams.")


def _render_eval_tab(documents: list[dict[str, Any]], *, sample_limit: int) -> None:
    st.caption("Generates document-grounded questions, runs search and answer generation, scores correctness, streams model output, and stores the run.")
    latest_eval = _latest_completed_eval()
    if latest_eval and "end_to_end_eval_payload" not in st.session_state:
        st.session_state["end_to_end_eval_payload"] = latest_eval["result_json"]
        st.session_state["end_to_end_eval_run_meta"] = {
            "id": latest_eval.get("id"),
            "updated_at": latest_eval.get("updated_at"),
            "source": "latest persisted completed run",
        }
    doc_options = {_document_label(item): item["document_id"] for item in documents}
    eval_control_col, eval_result_col = st.columns([1, 2])
    with eval_control_col:
        if latest_eval:
            latest_summary = latest_eval.get("result_json", {}).get("summary", {})
            st.markdown("### Latest Completed Run")
            st.caption(f"`{latest_eval.get('id')}` updated {latest_eval.get('updated_at')}")
            latest_cols = st.columns(2)
            latest_cols[0].metric("Retrieval", f"{latest_summary.get('retrieval_correct_percent', 0.0):.2f}%")
            latest_cols[1].metric("Answers", f"{latest_summary.get('answers_correct_percent', 0.0):.2f}%")
            if st.button("Load Latest Completed Results", use_container_width=True):
                st.session_state["end_to_end_eval_payload"] = latest_eval["result_json"]
                st.session_state["end_to_end_eval_run_meta"] = {
                    "id": latest_eval.get("id"),
                    "updated_at": latest_eval.get("updated_at"),
                    "source": "latest persisted completed run",
                }
                st.rerun()
            st.divider()
        eval_scope = st.radio("Scope", options=["All indexed docs in corpora", "Single document"], horizontal=False, key="eval_scope")
        eval_corpus_ids_text = st.text_input("Eval Corpus IDs", value=st.session_state.get("query_corpus_ids", DEFAULT_CORPUS), key="eval_corpus_ids")
        selected_eval_document = ""
        if eval_scope == "Single document":
            selected_eval_document = st.selectbox("Eval Document", options=[""] + list(doc_options.keys()), key="eval_document_select")
        max_questions = st.slider("Max questions", min_value=1, max_value=50, value=8, key="eval_max_questions")
        use_llm_generation = st.checkbox("Use LLM question generation", value=True, key="eval_llm_generation")
        stream_eval = st.checkbox("Stream progress and model output", value=True, key="eval_stream")
        if st.button("Run End-to-End Eval", use_container_width=True):
            st.session_state.pop("end_to_end_eval_payload", None)
            st.session_state.pop("end_to_end_eval_run_meta", None)
            eval_corpus_ids = [item.strip() for item in eval_corpus_ids_text.split(",") if item.strip()]
            eval_document_id = doc_options.get(selected_eval_document) if selected_eval_document else None
            if eval_scope == "Single document" and not eval_document_id:
                st.error("Choose a document for single-document evaluation.")
                st.stop()
            if eval_scope != "Single document" and not eval_corpus_ids:
                st.error("Corpus IDs cannot be empty for all-doc evaluation.")
                st.stop()
            request_payload = {
                "corpus_ids": eval_corpus_ids,
                "document_id": eval_document_id,
                "max_questions": max_questions,
                "use_llm_generation": use_llm_generation,
            }
            if stream_eval:
                _consume_streaming_eval(request_payload, sample_limit=sample_limit)
            else:
                with st.spinner("Running generated questions through search and answer generation..."):
                    st.session_state["end_to_end_eval_payload"] = _post("/eval/end-to-end", request_payload)
                    st.session_state["end_to_end_eval_run_meta"] = {"source": "current run"}
    with eval_result_col:
        eval_payload = st.session_state.get("end_to_end_eval_payload")
        if eval_payload:
            run_meta = st.session_state.get("end_to_end_eval_run_meta") or {}
            if run_meta:
                st.caption(
                    f"Showing {run_meta.get('source', 'saved result')}"
                    + (f" `{run_meta.get('id')}`" if run_meta.get("id") else "")
                    + (f" updated {run_meta.get('updated_at')}" if run_meta.get("updated_at") else "")
                )
            _render_end_to_end_eval(eval_payload, key_prefix="eval")
        else:
            st.info("Run an evaluation to review generated questions, answers, scoring, citations, and top retrieved chunks.")


def _render_model_tab() -> None:
    st.caption("Direct prompt testing and recent model call log.")
    left, right = st.columns([1, 2])
    with left:
        model = st.selectbox("Model", options=["qwen3.5:4b", "qwen3.5:9b"], index=0, key="model_prompt_model")
        think_mode = st.selectbox("Thinking", options=["disabled", "provider default", "enabled"], index=0, key="model_prompt_think")
        json_mode = st.checkbox("JSON mode", value=False, key="model_prompt_json")
        system_prompt = st.text_area("System Prompt", value="Reply concisely. Do not include thinking or reasoning traces.", height=100, key="model_prompt_system")
        user_prompt = st.text_area("User Prompt", value='Return exactly: {"ok": true}', height=180, key="model_prompt_user")
        if st.button("Run Model Prompt", use_container_width=True):
            think_value = {"disabled": False, "provider default": None, "enabled": True}[think_mode]
            st.session_state["model_prompt_payload"] = _post(
                "/debug/model-prompt",
                {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt, "think": think_value, "json_mode": json_mode},
            )
        if st.button("Refresh Ollama Call Log", use_container_width=True):
            st.session_state["ollama_call_log_payload"] = _get("/debug/ollama-calls", limit=100)
    with right:
        payload = st.session_state.get("model_prompt_payload")
        if payload:
            summary_cols = st.columns(4)
            summary_cols[0].metric("Requested", payload.get("model_requested", ""))
            summary_cols[1].metric("Response", payload.get("model_response", ""))
            summary_cols[2].metric("Duration", _format_duration(payload.get("duration_ms")))
            summary_cols[3].metric("Think Tags", "yes" if payload.get("raw_contains_think_tag") else "no")
            _render_json_block("Raw Output", {"raw_content": payload.get("raw_content", ""), "raw_contains_think_tag": payload.get("raw_contains_think_tag")})
            _render_json_block("Cleaned Output", {"cleaned_content": payload.get("cleaned_content", ""), "cleaned_contains_think_tag": payload.get("cleaned_contains_think_tag")})
            _render_json_block("Request Payload", payload.get("request_payload", {}))
    ollama_calls = st.session_state.get("ollama_call_log_payload")
    if ollama_calls:
        call_sample = _sample_selector("Ollama call", ollama_calls, "ollama_call_log_select")
        if call_sample:
            _render_json_block("Ollama Call", call_sample)
    else:
        st.info("Use `Refresh Ollama Call Log` to inspect actual model requests sent to Ollama.")


def _render_documents_tab(documents: list[dict[str, Any]], *, sample_limit: int) -> None:
    st.caption("Inspect parsed document structure, logical nodes, chunks, and metadata.")
    doc_options = {_document_label(item): item["document_id"] for item in documents}
    selected_document = st.selectbox("Document", options=[""] + list(doc_options.keys()), key="doc_snapshot_select")
    if selected_document and st.button("Load Document Snapshot", use_container_width=True):
        st.session_state["document_snapshot_payload"] = _get(
            f"/debug/documents/{doc_options[selected_document]}",
            sample_limit=sample_limit,
        )

    snapshot = st.session_state.get("document_snapshot_payload")
    if snapshot:
        overview_left, overview_right = st.columns(2)
        with overview_left:
            _render_json_block("Document", snapshot.get("document", {}))
        with overview_right:
            _render_json_block("Logical Node Counts", snapshot.get("logical_nodes", {}).get("counts_by_type", []))
            _render_json_block("Chunk Counts", snapshot.get("retrieval_chunks", {}).get("counts_by_type", []))

        st.markdown("### Logical Nodes")
        node_sample = _sample_selector("Logical node sample", snapshot.get("logical_nodes", {}).get("samples", []), "logical_nodes")
        if node_sample:
            left, right = st.columns(2)
            with left:
                _render_json_block("Node Summary", {k: node_sample.get(k) for k in ("id", "ordinal", "node_type", "heading_text", "section_path", "page_from", "page_to", "warning_level", "procedure_step_number", "citability_score", "token_count")})
                _render_json_block("Node Signals", {"spec_name": node_sample.get("spec_name"), "spec_value": node_sample.get("spec_value"), "spec_unit": node_sample.get("spec_unit"), "keywords": node_sample.get("keywords", [])})
            with right:
                _render_json_block("Node Content", {"text": node_sample.get("text", ""), "table_json": node_sample.get("table_json")})

        st.markdown("### Retrieval Chunks")
        chunk_sample = _sample_selector("Retrieval chunk sample", snapshot.get("retrieval_chunks", {}).get("samples", []), "retrieval_chunks")
        if chunk_sample:
            left, right = st.columns(2)
            with left:
                _render_json_block("Chunk Summary", {k: chunk_sample.get(k) for k in ("id", "chunk_type", "chunk_level", "title", "section_path_text", "page_from", "page_to", "priority_score", "logical_node_ids")})
                _render_json_block("Chunk Metadata", chunk_sample.get("metadata", {}))
            with right:
                _render_json_block("Chunk Content", {"content": chunk_sample.get("content", ""), "content_for_rerank": chunk_sample.get("content_for_rerank")})
    else:
        st.info("Load a document snapshot to inspect extraction and chunking outputs.")


def _render_metadata_tab(documents: list[dict[str, Any]]) -> None:
    st.caption("Inspect document metadata and test metadata-led document selection.")
    if not documents:
        st.info("No documents are available.")
        return
    doc_options = {_document_label(item): item["document_id"] for item in documents}
    query = st.text_area("Selection Test Query", value="Which document discusses setup?", height=90, key="metadata_query_text")
    corpus_ids_text = st.text_input("Metadata Test Corpus IDs", value=_default_corpus_ids(documents), key="metadata_query_corpus_ids")
    expected_label = st.selectbox("Expected Document", options=[""] + list(doc_options.keys()), key="metadata_expected_document")
    if st.button("Test Document Selection", use_container_width=True):
        results = _post("/search", {"query": query, "corpus_ids": [item.strip() for item in corpus_ids_text.split(",") if item.strip()], "filters": {}, "response_mode": "answer_with_citations"})
        st.session_state["metadata_query_payload"] = results
    payload = st.session_state.get("metadata_query_payload")
    if payload:
        _render_consolidated_search_results(payload, key_prefix="metadata")
    st.markdown("### Document Metadata")
    selected_label = st.selectbox("Metadata Document", options=list(doc_options.keys()), key="metadata_document")
    if selected_label:
        snapshot = _get(f"/debug/documents/{doc_options[selected_label]}/metadata")
        document = snapshot.get("document", {})
        st.dataframe([{k: document.get(k) for k in ("title", "source_filename", "document_kind", "manufacturer", "product_family", "product_model", "ingest_status", "metadata_model", "metadata_extracted_at")}], hide_index=True, use_container_width=True)
        _render_json_block("Extracted Metadata", document.get("extracted_metadata") or {})


def _render_tables_tab(documents: list[dict[str, Any]]) -> None:
    st.caption("Search table-specific or table-like evidence and inspect source assets.")
    doc_options = {_document_label(item): item["document_id"] for item in documents}
    left, right = st.columns([1, 2])
    with left:
        query = st.text_area("Table Query", value="What are the listed specifications in the table?", height=120, key="table_query_text")
        filter_document = st.selectbox("Filter to Document", options=[""] + list(doc_options.keys()), key="table_document_filter")
        corpus_ids_text = st.text_input("Table Corpus IDs", value=_default_corpus_ids(documents), key="table_query_corpus_ids")
        retrieval_mode = st.selectbox("Retrieval Mode", options=["table-like evidence", "strict table chunks"], key="table_retrieval_mode")
        filters: dict[str, Any] = {"chunk_type": ["table_record"]} if retrieval_mode == "strict table chunks" else {"chunk_type": ["table_record", "spec_record", "datasheet_record", "section_window", "parent_section", "atomic_text"]}
        if filter_document:
            filters["source_document_id"] = [doc_options[filter_document]]
        _render_json_block("Search Filters", filters)
        if st.button("Search Evidence", use_container_width=True):
            st.session_state["table_retrieval_payload"] = _post(
                "/search",
                {
                    "query": query,
                    "corpus_ids": [item.strip() for item in corpus_ids_text.split(",") if item.strip()],
                    "filters": filters,
                    "response_mode": "answer_with_citations",
                    "include_source_assets": True,
                    "include_page_images": True,
                    "include_table_images": True,
                },
            )
    with right:
        payload = st.session_state.get("table_retrieval_payload")
        if payload:
            _render_consolidated_search_results(payload, key_prefix="tables")
        else:
            st.info("Run a table search to inspect table-specific retrieval evidence.")


def _render_history_tab() -> None:
    st.caption("Persistent run outputs and event logs saved in Postgres.")
    run_type = st.selectbox("Run Type", options=["", "query_debug", "end_to_end_eval"], key="history_run_type")
    runs = _get("/runs", run_type=run_type or None, limit=50)
    if not runs:
        st.info("No persisted runs yet.")
        return
    st.dataframe(
        [
            {
                "Run ID": row.get("id"),
                "Type": row.get("run_type"),
                "Status": row.get("status"),
                "Created": row.get("created_at"),
                "Updated": row.get("updated_at"),
                "Error": row.get("error"),
            }
            for row in runs
        ],
        hide_index=True,
        use_container_width=True,
    )
    selected = st.selectbox("Inspect Run", options=[str(row["id"]) for row in runs], key="history_selected_run")
    run = _get(f"/runs/{selected}")
    _render_json_block("Request", run.get("request_json", {}))
    _render_json_block("Progress", run.get("progress_json", {}))
    if run.get("status") == "failed":
        st.error(run.get("error") or "Run failed.")
    if run.get("result_json"):
        if run.get("run_type") == "end_to_end_eval":
            _render_end_to_end_eval(run["result_json"], key_prefix="history_eval")
        else:
            _render_final_query_result(run["result_json"])
    events = _get(f"/runs/{selected}/events", limit=200, tail=True)
    _render_json_block("Latest Events", events)


def main() -> None:
    st.set_page_config(page_title="Manuals RAG Debug", layout="wide")
    st.title("Manuals RAG Debug Console")
    st.caption("Tabbed operator workspace for search, streaming generation, evaluation, extraction, metadata, tables, models, and run history.")

    with st.sidebar:
        st.markdown("### Connection")
        api_base = st.text_input("API Base", value=API_BASE, disabled=True)
        token = st.text_input("Auth Token", value=AUTH_TOKEN, type="password", disabled=True)
        st.caption(f"Using `{api_base}` with token `{token[:4]}...`.")
        sample_limit = st.slider("Samples per stage", min_value=3, max_value=50, value=10)

    try:
        documents = _get("/debug/documents", limit=200)
    except Exception as exc:
        st.error(f"Failed to load documents from API: {exc}")
        documents = []

    tabs = st.tabs(["Search", "Streaming Query", "End-to-End Eval", "Documents", "Metadata", "Tables", "Models", "Run History"])
    with tabs[0]:
        _render_search_tab(documents)
    with tabs[1]:
        _render_streaming_query_tab(documents, sample_limit=sample_limit)
    with tabs[2]:
        _render_eval_tab(documents, sample_limit=sample_limit)
    with tabs[3]:
        _render_documents_tab(documents, sample_limit=sample_limit)
    with tabs[4]:
        _render_metadata_tab(documents)
    with tabs[5]:
        _render_tables_tab(documents)
    with tabs[6]:
        _render_model_tab()
    with tabs[7]:
        _render_history_tab()


if __name__ == "__main__":
    main()
