from __future__ import annotations

import json
import os
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


def _stream_events(path: str, payload: dict[str, Any], **params: Any) -> Any:
    timeout = httpx.Timeout(600.0, connect=10.0, read=600.0)
    with httpx.Client(base_url=API_BASE, timeout=timeout) as client:
        with client.stream("POST", path, headers=_headers(), json=payload, params=params or None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                yield json.loads(line)


def _format_duration(duration_ms: Any) -> str:
    if duration_ms is None:
        return "n/a"
    return f"{float(duration_ms):,.2f} ms"


def _render_json_block(label: str, payload: Any) -> None:
    st.markdown(f"**{label}**")
    st.json(payload, expanded=False)


def _render_step_table(step_sequence: list[dict[str, Any]], step_state: dict[str, dict[str, Any]]) -> None:
    rows = []
    for index, step in enumerate(step_sequence, start=1):
        name = step["name"]
        state = step_state.get(name, {})
        rows.append(
            {
                "#": index,
                "step": step["label"],
                "model": step.get("model") or "",
                "status": state.get("status", "pending"),
                "duration": _format_duration(state.get("duration_ms")) if state.get("duration_ms") is not None else "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_llm_streams(llm_outputs: dict[str, dict[str, Any]]) -> None:
    if not llm_outputs:
        st.info("LLM output will appear when relevance review, evidence summaries, and final answer generation run.")
        return
    for call_id, call in llm_outputs.items():
        st.markdown(f"**{call.get('label') or call_id}**")
        st.caption(f"{call.get('model') or ''} | {call.get('status', 'running')}")
        st.code(call.get("text", ""), language="json")


def _render_final_result(payload: dict[str, Any]) -> None:
    st.markdown("## Result")
    meta_left, meta_mid, meta_right = st.columns(3)
    with meta_left:
        _render_json_block("Query Analysis", payload.get("analysis", {}))
    with meta_mid:
        _render_json_block("Applied Filters", payload.get("applied_filters", {}))
    with meta_right:
        _render_json_block("Step Timings", payload.get("step_timings_ms", {}))

    st.markdown("### Retrieval Stages")
    for stage in payload.get("stages", []):
        st.markdown(f"**{stage.get('name')}**")
        st.caption(f"{stage.get('count', 0)} results | {_format_duration(stage.get('duration_ms'))}")
        samples = stage.get("samples", [])
        if samples:
            _render_json_block("First Sample", samples[0])

    st.markdown("### LLM Review")
    _render_json_block("Answer Inputs", payload.get("answer_generation_inputs", {}))
    _render_json_block("Answer Summaries", payload.get("answer_summaries", {}))

    st.markdown("### Final Answer")
    _render_json_block("Answer Payload", payload.get("answer", {}))
    _render_json_block("Answer Generation Trace", payload.get("answer_generation_trace", {}))


def _run_streaming_query(payload: dict[str, Any], *, sample_limit: int) -> None:
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
            step_state[event["step"]] = {
                "status": "completed",
                "duration_ms": event.get("duration_ms"),
            }
        elif event_type == "llm_call_started":
            llm_outputs[event["call_id"]] = {
                "label": event.get("label"),
                "model": event.get("model"),
                "status": "running",
                "text": "",
            }
        elif event_type == "llm_token":
            call = llm_outputs.setdefault(event["call_id"], {"status": "running", "text": ""})
            call["text"] = f"{call.get('text', '')}{event.get('token', '')}"
        elif event_type == "llm_call_completed":
            call = llm_outputs.setdefault(event["call_id"], {"text": ""})
            call.update(
                {
                    "label": event.get("label") or call.get("label"),
                    "model": event.get("model") or call.get("model"),
                    "status": "completed",
                    "text": event.get("raw_response") or call.get("text", ""),
                }
            )
        elif event_type == "llm_call_failed":
            call = llm_outputs.setdefault(event["call_id"], {"text": ""})
            call.update(
                {
                    "label": event.get("label") or call.get("label"),
                    "model": event.get("model") or call.get("model"),
                    "status": f"failed: {event.get('error')}",
                }
            )
        elif event_type == "run_completed":
            final_result = event.get("result", {})
        elif event_type == "run_failed":
            st.error(event.get("error") or "Streaming query failed.")

        with progress_placeholder.container():
            st.markdown("## Steps")
            _render_step_table(step_sequence, step_state)
        with llm_placeholder.container():
            st.markdown("## Streaming LLM Output")
            _render_llm_streams(llm_outputs)
        if final_result:
            with result_placeholder.container():
                _render_final_result(final_result)

    if final_result:
        st.session_state["streaming_query_result"] = final_result


def main() -> None:
    st.set_page_config(page_title="Streaming Query Debug", layout="wide")
    st.title("Streaming Query Debug")
    st.caption("Run the query pipeline and stream model output from each LLM step.")

    with st.sidebar:
        st.markdown("### Connection")
        api_base = st.text_input("API Base", value=API_BASE, disabled=True)
        token = st.text_input("Auth Token", value=AUTH_TOKEN, type="password", disabled=True)
        st.caption(f"Using `{api_base}` with token `{token[:4]}...`.")
        sample_limit = st.slider("Samples per stage", min_value=3, max_value=50, value=10)

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

    left, right = st.columns([1, 2])
    with left:
        query = st.text_area("Query", value="What product is described in this datasheet?", height=120)
        filter_document = st.selectbox("Filter to Document", options=[""] + list(doc_options.keys()))
        selected_document_id = doc_options.get(filter_document) if filter_document else None
        selected_document = documents_by_id.get(selected_document_id) if selected_document_id else None
        selected_corpus_id = str(selected_document["corpus_id"]) if selected_document else DEFAULT_CORPUS
        previous_selected_document_id = st.session_state.get("streaming_query_filter_document_id")
        if "streaming_query_corpus_ids" not in st.session_state:
            st.session_state["streaming_query_corpus_ids"] = selected_corpus_id
        elif selected_document_id and selected_document_id != previous_selected_document_id:
            st.session_state["streaming_query_corpus_ids"] = selected_corpus_id
        elif not selected_document_id and previous_selected_document_id:
            st.session_state["streaming_query_corpus_ids"] = DEFAULT_CORPUS
        st.session_state["streaming_query_filter_document_id"] = selected_document_id
        corpus_ids_text = st.text_input("Corpus IDs", key="streaming_query_corpus_ids")
        if selected_document:
            st.caption(f"Using selected document corpus `{selected_corpus_id}` by default.")
        filters: dict[str, Any] = {}
        if selected_document_id:
            filters["source_document_id"] = selected_document_id
        run_query = st.button("Run Streaming Query Debug", use_container_width=True)

    with right:
        st.info("Start a run to watch retrieval progress and streamed model output.")
        if st.session_state.get("streaming_query_result"):
            _render_final_result(st.session_state["streaming_query_result"])

    if run_query:
        corpus_ids = [item.strip() for item in corpus_ids_text.split(",") if item.strip()]
        if not corpus_ids:
            st.error("Corpus IDs cannot be empty. Select a document or enter at least one corpus ID.")
            st.stop()
        st.session_state.pop("streaming_query_result", None)
        _run_streaming_query(
            {
                "query": query,
                "corpus_ids": corpus_ids,
                "filters": filters,
                "response_mode": "answer_with_citations",
            },
            sample_limit=sample_limit,
        )


if __name__ == "__main__":
    main()
