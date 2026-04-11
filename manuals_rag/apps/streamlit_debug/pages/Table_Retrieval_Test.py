from __future__ import annotations

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


def _post(path: str, payload: dict[str, Any], **params: Any) -> Any:
    with httpx.Client(base_url=API_BASE, timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        response = client.post(path, headers=_headers(), json=payload, params=params or None)
        response.raise_for_status()
        return response.json()


def _render_json_block(label: str, payload: Any) -> None:
    st.markdown(f"**{label}**")
    st.json(payload, expanded=False)


def _result_label(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    flags = [
        name
        for name in ("table_key_value", "table_row_group", "table_summary")
        if metadata.get(name)
    ]
    suffix = f" | {', '.join(flags)}" if flags else ""
    return f"{result.get('score', 0):.4f} | p.{','.join(str(page) for page in result.get('pages', []))} | {result.get('chunk_id')}{suffix}"


def _assets_for_chunk(source_assets: dict[str, Any], chunk_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pages = [
        item
        for item in source_assets.get("citation_pages", [])
        if str(item.get("chunk_id")) == str(chunk_id)
    ]
    tables = [
        item
        for item in source_assets.get("tables", [])
        if str(item.get("chunk_id")) == str(chunk_id)
    ]
    return pages, tables


def _render_source_assets(source_assets: dict[str, Any], chunk_id: str) -> None:
    page_assets, table_assets = _assets_for_chunk(source_assets, chunk_id)
    if not page_assets and not table_assets:
        st.info("No image assets were returned for this table result. Reingest the source document if it predates image asset generation.")
        return
    if page_assets:
        st.markdown("### Page Images")
        for asset in page_assets:
            st.caption(f"Page {asset.get('page')}")
            if asset.get("page_image_url"):
                st.image(asset["page_image_url"], use_container_width=True)
            if asset.get("pdf_download_url"):
                st.link_button("Open PDF", asset["pdf_download_url"])
    if table_assets:
        st.markdown("### Table Images")
        for asset in table_assets:
            st.caption(f"Table {asset.get('table_index')} | Page {asset.get('page')}")
            if asset.get("table_image_url"):
                st.image(asset["table_image_url"], use_container_width=True)
            _render_json_block("Table Asset", asset)


def _chunk_counts(document_detail: dict[str, Any]) -> dict[str, int]:
    retrieval_chunks = document_detail.get("retrieval_chunks", {})
    counts = retrieval_chunks.get("counts_by_type", {})
    if isinstance(counts, dict):
        return {str(key): int(value or 0) for key, value in counts.items()}
    if isinstance(counts, list):
        normalized: dict[str, int] = {}
        for item in counts:
            if isinstance(item, dict) and item.get("chunk_type"):
                normalized[str(item["chunk_type"])] = int(item.get("count") or 0)
        return normalized
    return {}


def _render_document_diagnostics(document_detail: dict[str, Any] | None) -> int:
    if not document_detail:
        return 0

    chunk_counts = _chunk_counts(document_detail)
    table_count = int(chunk_counts.get("table_record", 0) or 0)
    if table_count:
        st.success(f"Selected document has {table_count} table_record chunks.")
    else:
        st.warning(
            "Selected document has no table_record chunks. Table-only search will return no results; "
            "use table-like evidence mode to inspect sections, specs, and text extracted from table-shaped content."
        )
    _render_json_block("Selected Document Chunk Counts", chunk_counts)
    return table_count


def _render_results(payload: dict[str, Any], retrieval_mode: str | None = None) -> None:
    results = payload.get("results", payload if isinstance(payload, list) else [])
    source_assets = payload.get("source_assets", {}) if isinstance(payload, dict) else {}
    if not results:
        if retrieval_mode == "table-like evidence":
            st.info(
                "No table-like evidence returned. Try the product/model string used by the selected document "
                "or a narrower term from the table, such as repeatability."
            )
        else:
            st.info("No table results returned.")
        return

    st.markdown("## Results")
    st.dataframe(
        [
            {
                "rank": index,
                "score": result.get("score"),
                "pages": result.get("pages"),
                "chunk_id": result.get("chunk_id"),
                "flags": ", ".join(
                    name
                    for name in ("table_key_value", "table_row_group", "table_summary")
                    if result.get("metadata", {}).get(name)
                ),
                "content": str(result.get("content", ""))[:240],
            }
            for index, result in enumerate(results, start=1)
        ],
        use_container_width=True,
        hide_index=True,
    )
    selected_index = st.selectbox(
        "Inspect result",
        options=list(range(len(results))),
        format_func=lambda index: _result_label(results[index]),
    )
    result = results[selected_index]
    left, right = st.columns(2)
    with left:
        _render_json_block(
            "Result Summary",
            {
                key: result.get(key)
                for key in ("chunk_id", "score", "title", "document_version_id", "source_document_id", "pages", "section_path")
            },
        )
        _render_json_block("Metadata", result.get("metadata", {}))
        _render_json_block("Source Assets", source_assets)
    with right:
        st.markdown("### Content")
        st.code(result.get("content", ""), language="text")
        _render_source_assets(source_assets, str(result.get("chunk_id")))


def main() -> None:
    st.set_page_config(page_title="Table Retrieval Test", layout="wide")
    st.title("Table Retrieval Test")
    st.caption("Search only table retrieval chunks and inspect returned table evidence.")

    with st.sidebar:
        st.markdown("### Connection")
        api_base = st.text_input("API Base", value=API_BASE, disabled=True)
        token = st.text_input("Auth Token", value=AUTH_TOKEN, type="password", disabled=True)
        st.caption(f"Using `{api_base}` with token `{token[:4]}...`.")

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

    query_col, result_col = st.columns([1, 2])
    with query_col:
        query = st.text_area("Table Query", value="What are the listed specifications in the table?", height=120)
        filter_document = st.selectbox("Filter to Document", options=[""] + list(doc_options.keys()))
        selected_document_id = doc_options.get(filter_document) if filter_document else None
        selected_document = documents_by_id.get(selected_document_id) if selected_document_id else None
        selected_corpus_id = str(selected_document["corpus_id"]) if selected_document else DEFAULT_CORPUS
        previous_selected_document_id = st.session_state.get("table_query_filter_document_id")
        if "table_query_corpus_ids" not in st.session_state:
            st.session_state["table_query_corpus_ids"] = selected_corpus_id
        elif selected_document_id and selected_document_id != previous_selected_document_id:
            st.session_state["table_query_corpus_ids"] = selected_corpus_id
        elif not selected_document_id and previous_selected_document_id:
            st.session_state["table_query_corpus_ids"] = DEFAULT_CORPUS
        st.session_state["table_query_filter_document_id"] = selected_document_id
        corpus_ids_text = st.text_input("Corpus IDs", key="table_query_corpus_ids")
        if selected_document:
            st.caption(f"Using selected document corpus `{selected_corpus_id}` by default.")
        include_page_images = st.checkbox("Return page images", value=True)
        include_table_images = st.checkbox("Return table images", value=True)
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=["table-like evidence", "strict table chunks"],
            help=(
                "Strict mode only searches table_record chunks. Table-like evidence also searches spec, section, "
                "and atomic text chunks for documents where Docling did not emit table chunks."
            ),
        )
        table_variant = st.selectbox(
            "Table Variant",
            options=["any table chunk", "key/value rows", "row groups", "table summaries"],
            disabled=retrieval_mode != "strict table chunks",
        )

        selected_document_detail = None
        if selected_document_id:
            try:
                selected_document_detail = _get(f"/debug/documents/{selected_document_id}", sample_limit=1)
            except Exception as exc:
                st.error(f"Failed to load selected document diagnostics: {exc}")
        table_record_count = _render_document_diagnostics(selected_document_detail)

        if retrieval_mode == "strict table chunks":
            filters: dict[str, Any] = {"chunk_type": ["table_record"]}
        else:
            filters = {
                "chunk_type": [
                    "table_record",
                    "spec_record",
                    "datasheet_record",
                    "section_window",
                    "parent_section",
                    "atomic_text",
                ]
            }
        if selected_document_id:
            filters["source_document_id"] = selected_document_id
        if retrieval_mode == "strict table chunks":
            if table_variant == "key/value rows":
                filters["table_key_value"] = "true"
            elif table_variant == "row groups":
                filters["table_row_group"] = "true"
            elif table_variant == "table summaries":
                filters["table_summary"] = "true"
        elif selected_document_id and table_record_count == 0:
            st.caption("Table variant flags are skipped because this document has no table_record chunks.")
        _render_json_block("Search Filters", filters)
        if st.button("Search Evidence", use_container_width=True):
            corpus_ids = [item.strip() for item in corpus_ids_text.split(",") if item.strip()]
            if not corpus_ids:
                st.error("Corpus IDs cannot be empty. Select a document or enter at least one corpus ID.")
                st.stop()
            st.session_state["table_retrieval_payload"] = _post(
                "/search",
                {
                    "query": query,
                    "corpus_ids": corpus_ids,
                    "filters": filters,
                    "response_mode": "answer_with_citations",
                    "include_source_assets": True,
                    "include_page_images": include_page_images,
                    "include_table_images": include_table_images,
                },
            )
            st.session_state["table_retrieval_mode"] = retrieval_mode

    with result_col:
        payload = st.session_state.get("table_retrieval_payload")
        if payload:
            _render_results(payload, st.session_state.get("table_retrieval_mode"))
        else:
            st.info("Run a table search to inspect table-specific retrieval evidence.")


if __name__ == "__main__":
    main()
