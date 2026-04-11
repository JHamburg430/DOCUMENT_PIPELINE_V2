# Manuals RAG

Production-oriented document ingestion, retrieval, and grounded answering for technical manuals and related engineering documents.

## Stack

- FastAPI API gateway and admin/query APIs
- Postgres control plane
- Qdrant retrieval plane
- MinIO object storage
- Redis queueing and coordination
- Docling-first parsing with a PyMuPDF fallback for local bootstrap
- Custom normalization and hierarchical chunking
- TinyLlama-backed structured metadata extraction with Pydantic validation
- Hybrid dense+sparse retrieval with reranking
- LangGraph deterministic query workflow
- Static web UI served from a dedicated UI container

## Quick start

1. Copy `.env.example` to `.env`.
2. Install Python dependencies with `/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python -m pip install -r manuals_rag/requirements.txt`.
3. Start the stack with `docker compose -f manuals_rag/infra/compose/docker-compose.yml up -d --build`.
4. Run bootstrap with `manuals_rag/scripts/bootstrap/bootstrap_local.sh`.

## Repo layout

This repository follows the target layout from `.AGENT.md` under [`manuals_rag`](./manuals_rag).

## Metadata extraction

Document metadata extraction is model-backed and should not be replaced with filename or text-pattern heuristics. The current local model is `tinyllama:1.1b` through Ollama, configured by `OLLAMA_METADATA_MODEL`.

Saved metadata lives in Postgres in `document_metadata_extractions` and is copied into retrieval chunk metadata for filtering and ranking. Existing corpora can be updated with `manuals_rag/scripts/maintenance/backfill_document_metadata.py --apply`.

Operator inspection is available in Streamlit at `http://127.0.0.1:8601/Document_Metadata`. More detail is in [`docs/architecture/metadata_extraction.md`](docs/architecture/metadata_extraction.md).
