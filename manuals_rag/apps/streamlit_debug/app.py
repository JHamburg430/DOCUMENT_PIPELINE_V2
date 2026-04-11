from __future__ import annotations

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
    st.markdown(f"**{label}**")
    st.json(payload, expanded=False)


def _format_duration(duration_ms: Any) -> str:
    if duration_ms is None:
        return "n/a"
    return f"{float(duration_ms):,.2f} ms"


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


def main() -> None:
    st.set_page_config(page_title="Manuals RAG Debug", layout="wide")
    st.title("Manuals RAG Debug Console")
    st.caption("Central inspection view for extraction, retrieval, and answer generation.")

    with st.sidebar:
        st.markdown("### Connection")
        api_base = st.text_input("API Base", value=API_BASE, disabled=True)
        token = st.text_input("Auth Token", value=AUTH_TOKEN, type="password", disabled=True)
        st.caption(f"Using `{api_base}` with token `{token[:4]}...`.")
        sample_limit = st.slider("Samples per stage", min_value=3, max_value=50, value=10)
        if st.button("Refresh Ollama Call Log", use_container_width=True):
            st.session_state["ollama_call_log_payload"] = _get("/debug/ollama-calls", limit=100)

    documents = []
    try:
        documents = _get("/debug/documents", limit=100)
    except Exception as exc:
        st.error(f"Failed to load documents from API: {exc}")

    doc_options = {
        f"{item['title']} | {item['source_filename']} | {item['document_id']}": item["document_id"]
        for item in documents
    }
    documents_by_id = {item["document_id"]: item for item in documents}

    st.markdown("## Query Pipeline")
    query_col, result_col = st.columns([1, 2])
    with query_col:
        query = st.text_area("Query", value="What product is described in this datasheet?", height=120)
        filter_document = st.selectbox("Filter to Document", options=[""] + list(doc_options.keys()))
        selected_document_id = doc_options.get(filter_document) if filter_document else None
        selected_document = documents_by_id.get(selected_document_id) if selected_document_id else None
        selected_corpus_id = str(selected_document["corpus_id"]) if selected_document else DEFAULT_CORPUS
        previous_selected_document_id = st.session_state.get("query_filter_document_id")
        if "query_corpus_ids" not in st.session_state:
            st.session_state["query_corpus_ids"] = selected_corpus_id
        elif selected_document_id and selected_document_id != previous_selected_document_id:
            st.session_state["query_corpus_ids"] = selected_corpus_id
        elif not selected_document_id and previous_selected_document_id:
            st.session_state["query_corpus_ids"] = DEFAULT_CORPUS
        st.session_state["query_filter_document_id"] = selected_document_id
        corpus_ids_text = st.text_input("Corpus IDs", key="query_corpus_ids")
        if selected_document:
            st.caption(f"Using selected document corpus `{selected_corpus_id}` by default.")
        filters: dict[str, Any] = {}
        if selected_document_id:
            filters["source_document_id"] = selected_document_id
        if st.button("Run Query Debug", use_container_width=True):
            corpus_ids = [item.strip() for item in corpus_ids_text.split(",") if item.strip()]
            if not corpus_ids:
                st.error("Corpus IDs cannot be empty. Select a document or enter at least one corpus ID.")
                st.stop()
            st.session_state["query_debug_payload"] = _run_query_debug(
                {
                    "query": query,
                    "corpus_ids": corpus_ids,
                    "filters": filters,
                    "response_mode": "answer_with_citations",
                },
                sample_limit=sample_limit,
            )
    with result_col:
        payload = st.session_state.get("query_debug_payload")
        if payload:
            meta_left, meta_mid, meta_right = st.columns(3)
            with meta_left:
                _render_json_block("Query Analysis", payload.get("analysis", {}))
            with meta_mid:
                _render_json_block("Applied Filters", payload.get("applied_filters", {}))
            with meta_right:
                _render_json_block("Step Timings (ms)", payload.get("step_timings_ms", {}))
            for stage in payload.get("stages", []):
                _render_stage(stage, "query_stage")
            st.markdown("### LLM Relevance Review")
            answer_inputs = payload.get("answer_generation_inputs", {})
            st.caption(f"{answer_inputs.get('count', 0)} results | {_format_duration(answer_inputs.get('duration_ms'))}")
            answer_sample = _sample_selector("Answer input sample", answer_inputs.get("samples", []), "answer_inputs")
            if answer_sample:
                left, right = st.columns(2)
                with left:
                    _render_json_block(
                        "Input Summary",
                        {
                            k: answer_sample.get(k)
                            for k in ("chunk_id", "score", "chunk_type", "title", "pages", "section_path", "retrieval_stage")
                        },
                    )
                    _render_json_block(
                        "Model Judgment",
                        {
                            "verdict": answer_sample.get("relevance_verdict"),
                            "reason": answer_sample.get("relevance_reason"),
                        },
                    )
                with right:
                    _render_json_block(
                        "Input Content",
                        {
                            "content": answer_sample.get("content", ""),
                            "context_window": answer_sample.get("context_window"),
                            "parent_context": answer_sample.get("parent_context"),
                        },
                    )
            st.markdown("### Chunk Summaries")
            answer_summaries = payload.get("answer_summaries", {})
            st.caption(f"{answer_summaries.get('count', 0)} summaries | {_format_duration(answer_summaries.get('duration_ms'))}")
            summary_sample = _sample_selector("Summary sample", answer_summaries.get("samples", []), "answer_summaries")
            if summary_sample:
                left, right = st.columns(2)
                with left:
                    _render_json_block(
                        "Summary Source",
                        {
                            k: summary_sample.get(k)
                            for k in ("chunk_id", "title", "pages", "section_path", "source_document_id", "document_version_id")
                        },
                    )
                with right:
                    _render_json_block("Summary", {"summary": summary_sample.get("summary", "")})
            st.markdown("### Final Answer")
            st.caption(_format_duration(payload.get("step_timings_ms", {}).get("generate_answer")))
            _render_json_block("Answer Payload", payload.get("answer", {}))
            st.markdown("### Answer Generation Trace")
            answer_trace = payload.get("answer_generation_trace", {})
            trace_left, trace_mid, trace_right = st.columns(3)
            with trace_left:
                _render_json_block("Relevance Stage", answer_trace.get("relevance_review", {}))
            with trace_mid:
                _render_json_block("Summary Stage", answer_trace.get("summarization", {}))
            with trace_right:
                _render_json_block("Final Answer Stage", {k: v for k, v in answer_trace.get("final_answer", {}).items() if k != "summarized_evidence"})
            final_answer_trace = answer_trace.get("final_answer", {})
            _render_json_block("Summaries Sent To Final Answer", final_answer_trace.get("summarized_evidence", []))
        else:
            st.info("Run a query to inspect the retrieval and answer pipeline.")

    st.markdown("## Ollama Call Log")
    ollama_calls = st.session_state.get("ollama_call_log_payload")
    if ollama_calls:
        call_sample = _sample_selector("Ollama call", ollama_calls, "ollama_call_log_select")
        if call_sample:
            _render_json_block("Ollama Call", call_sample)
    else:
        st.info("Use `Refresh Ollama Call Log` to inspect actual model requests sent to Ollama.")

    st.markdown("## PDF Extraction")
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


if __name__ == "__main__":
    main()
