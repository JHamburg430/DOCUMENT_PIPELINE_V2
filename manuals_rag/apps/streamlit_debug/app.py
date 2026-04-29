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


def _render_consolidated_search_results(payload: Any) -> None:
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

    st.markdown("## Consolidated Search")
    st.caption("Runs the production `/search` endpoint: metadata document selection, dense retrieval, sparse retrieval, table retrieval, fusion, reranking, and assembly.")
    query_col, result_col = st.columns([1, 2])
    with query_col:
        query = st.text_area("Search Query", value="LJ-X8080 z axis repeatability", height=120)
        if "query_corpus_ids" not in st.session_state:
            st.session_state["query_corpus_ids"] = _default_corpus_ids(documents)
        corpus_ids_text = st.text_input("Corpus IDs", key="query_corpus_ids")
        include_page_images = st.checkbox("Return page images", value=False)
        include_table_images = st.checkbox("Return table images", value=False)
        st.caption("No document filter is applied here; document selection is performed by metadata embeddings inside the API.")
        if st.button("Run Consolidated Search", use_container_width=True):
            corpus_ids = [item.strip() for item in corpus_ids_text.split(",") if item.strip()]
            if not corpus_ids:
                st.error("Corpus IDs cannot be empty.")
                st.stop()
            st.session_state["consolidated_search_payload"] = _post(
                "/search",
                {
                    "query": query,
                    "corpus_ids": corpus_ids,
                    "filters": {},
                    "response_mode": "answer_with_citations",
                    "include_source_assets": include_page_images or include_table_images,
                    "include_page_images": include_page_images,
                    "include_table_images": include_table_images,
                },
            )
    with result_col:
        payload = st.session_state.get("consolidated_search_payload")
        if payload:
            _render_consolidated_search_results(payload)
        else:
            st.info("Run a search to inspect the consolidated retrieval results.")

    with st.expander("Legacy debug pipeline", expanded=False):
        st.caption("This diagnostic view uses the debug run endpoint. Use consolidated search above for production retrieval behavior.")
        debug_query = st.text_area("Debug Query", value="What product is described in this datasheet?", height=100, key="debug_query_text")
        debug_corpus_ids_text = st.text_input("Debug Corpus IDs", value=st.session_state.get("query_corpus_ids", DEFAULT_CORPUS), key="debug_query_corpus_ids")
        if st.button("Run Legacy Debug Pipeline", use_container_width=True):
            debug_corpus_ids = [item.strip() for item in debug_corpus_ids_text.split(",") if item.strip()]
            if not debug_corpus_ids:
                st.error("Debug corpus IDs cannot be empty.")
                st.stop()
            st.session_state["query_debug_payload"] = _run_query_debug(
                {
                    "query": debug_query,
                    "corpus_ids": debug_corpus_ids,
                    "filters": {},
                    "response_mode": "answer_with_citations",
                },
                sample_limit=sample_limit,
            )
        debug_payload = st.session_state.get("query_debug_payload")
        if debug_payload:
            _render_json_block("Query Analysis", debug_payload.get("analysis", {}))
            _render_json_block("Applied Filters", debug_payload.get("applied_filters", {}))
            for stage in debug_payload.get("stages", []):
                _render_stage(stage, "legacy_query_stage")
            _render_json_block("Answer Payload", debug_payload.get("answer", {}))

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
