from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st


API_BASE = os.getenv("MANUALS_RAG_API_BASE", "http://127.0.0.1:8600")
AUTH_TOKEN = os.getenv("MANUALS_RAG_AUTH_TOKEN", "admin-token")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def _get(path: str, **params: Any) -> Any:
    with httpx.Client(base_url=API_BASE, timeout=60.0) as client:
        response = client.get(path, headers=_headers(), params=params or None)
        response.raise_for_status()
        return response.json()


def _post(path: str, payload: dict[str, Any], **params: Any) -> Any:
    with httpx.Client(base_url=API_BASE, timeout=httpx.Timeout(180.0, connect=10.0)) as client:
        response = client.post(path, headers=_headers(), json=payload, params=params or None)
        response.raise_for_status()
        return response.json()


def _metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _render_query_selection_test(document: dict[str, Any]) -> None:
    st.markdown("### Query Document Selection Test")
    st.caption(
        "Run retrieval without a document filter to check whether the current document is selected from the user query."
    )

    document_id = str(document.get("document_id") or "")
    default_corpus_id = str(document.get("corpus_id") or "")
    corpus_key = f"metadata_query_corpus_ids_{document_id}"
    if corpus_key not in st.session_state:
        st.session_state[corpus_key] = default_corpus_id

    query = st.text_area(
        "Test Query",
        value=f"What does {document.get('product_model') or document.get('title') or 'this document'} say about setup?",
        height=100,
        key=f"metadata_query_text_{document_id}",
    )
    corpus_ids_text = st.text_input("Corpus IDs", key=corpus_key)

    if st.button("Test Document Selection", use_container_width=True):
        corpus_ids = [item.strip() for item in corpus_ids_text.split(",") if item.strip()]
        if not query.strip():
            st.error("Test query cannot be empty.")
            st.stop()
        if not corpus_ids:
            st.error("Corpus IDs cannot be empty.")
            st.stop()

        try:
            results_payload = _post(
                "/search",
                {
                    "query": query,
                    "corpus_ids": corpus_ids,
                    "filters": {},
                    "response_mode": "answer_with_citations",
                },
            )
        except Exception as exc:
            st.error(f"Query test failed: {exc}")
            return

        results = results_payload.get("results", []) if isinstance(results_payload, dict) else results_payload
        st.session_state[f"metadata_query_results_{document_id}"] = results

    results = st.session_state.get(f"metadata_query_results_{document_id}", [])
    if not results:
        st.info("Run a query to inspect which document retrieval selects.")
        return

    matching_ranks = [
        index + 1
        for index, result in enumerate(results)
        if str(result.get("source_document_id")) == document_id
    ]
    top_result = results[0]
    if str(top_result.get("source_document_id")) == document_id:
        st.success("Selected document matched the top retrieval result.")
    elif matching_ranks:
        st.warning(f"Selected document was returned at rank {matching_ranks[0]}, but not as the top result.")
    else:
        st.error("Selected document was not returned in the retrieval results.")

    rows = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        rows.append(
            {
                "Rank": index,
                "Expected Document": str(result.get("source_document_id")) == document_id,
                "Score": result.get("score"),
                "Title": result.get("title"),
                "Source Document": result.get("source_document_id"),
                "Pages": _metadata_value(result.get("pages") or []),
                "Chunk Type": metadata.get("chunk_type"),
                "Stage": metadata.get("retrieval_stage"),
                "Preview": str(result.get("content") or "")[:240],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Document Metadata", layout="wide")
    st.title("Document Metadata")
    st.caption("Inspect model-extracted metadata by document and source page.")

    try:
        documents = _get("/debug/documents", limit=200)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not documents:
        st.info("No documents are available.")
        return

    options = {
        f"{item['title']} | {item['source_filename']} | {item['document_id']}": item["document_id"]
        for item in documents
    }
    selected_label = st.selectbox("Document", options=list(options.keys()))
    document_id = options[selected_label]

    try:
        initial = _get(f"/debug/documents/{document_id}/metadata")
    except Exception as exc:
        st.error(f"Failed to load metadata: {exc}")
        return

    pages = initial.get("pages") or [1]
    selected_page = st.selectbox("Page", options=pages, index=0)
    snapshot = initial if selected_page == initial.get("selected_page") else _get(
        f"/debug/documents/{document_id}/metadata",
        page=selected_page,
    )

    document = snapshot.get("document", {})
    extracted = document.get("extracted_metadata") or {}

    st.markdown("### Document")
    st.dataframe(
        [
            {
                "Title": document.get("title"),
                "Filename": document.get("source_filename"),
                "Kind": document.get("document_kind"),
                "Manufacturer": document.get("manufacturer"),
                "Family": document.get("product_family"),
                "Model": document.get("product_model"),
                "Status": document.get("ingest_status"),
                "Metadata Model": document.get("metadata_model"),
                "Extracted At": document.get("metadata_extracted_at"),
            }
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Extracted Metadata")
    if extracted:
        st.json(extracted, expanded=True)
    else:
        st.warning("No document-level metadata extraction row is saved yet.")

    _render_query_selection_test(document)

    st.markdown(f"### Page {snapshot.get('selected_page')} Chunk Metadata")
    page_chunks = snapshot.get("page_chunks", [])
    if not page_chunks:
        st.info("No retrieval chunks overlap this page.")
        return

    table_rows = []
    for chunk in page_chunks:
        metadata = chunk.get("metadata", {})
        table_rows.append(
            {
                "Chunk": chunk.get("id"),
                "Type": chunk.get("chunk_type"),
                "Level": chunk.get("chunk_level"),
                "Section": chunk.get("section_path_text"),
                "Pages": f"{chunk.get('page_from')}-{chunk.get('page_to')}",
                "Models": _metadata_value(metadata.get("product_models") or metadata.get("product_model")),
                "Parts": _metadata_value(metadata.get("part_numbers")),
                "Protocols": _metadata_value(metadata.get("document_protocol_terms")),
                "Settings": _metadata_value(metadata.get("settings")),
                "Parameters": _metadata_value(metadata.get("parameters")),
                "Topics": _metadata_value(metadata.get("document_topics")),
                "Preview": chunk.get("content_preview"),
            }
        )
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    selected_chunk_index = st.selectbox(
        "Chunk detail",
        options=list(range(len(page_chunks))),
        format_func=lambda index: f"{index + 1}. {page_chunks[index].get('chunk_type')} | {page_chunks[index].get('section_path_text')}",
    )
    st.json(page_chunks[selected_chunk_index], expanded=True)


if __name__ == "__main__":
    main()
