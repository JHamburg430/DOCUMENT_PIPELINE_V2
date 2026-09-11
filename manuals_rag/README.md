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
- Qwen3.5-backed structured metadata extraction with Pydantic validation and page-aware evidence grounding
- Hybrid dense+sparse retrieval with reranking
- Baseline deterministic query workflow plus bounded LangGraph and LlamaIndex multi-hop retrieval
- Static web UI served from a dedicated UI container

## Quick start

1. Copy `.env.example` to `.env`.
2. Install Python dependencies with `/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python -m pip install -r manuals_rag/requirements.txt`.
3. Start the stack with `docker compose -f manuals_rag/infra/compose/docker-compose.yml up -d --build`.
4. Run bootstrap with `manuals_rag/scripts/bootstrap/bootstrap_local.sh`.

## Repo layout

This repository follows the target layout from `.AGENT.md` under [`manuals_rag`](./manuals_rag).

## Metadata extraction

Document metadata extraction is model-backed and should not be replaced with filename or text-pattern heuristics. The current local model is `qwen3.5:9b` through Ollama, configured by `OLLAMA_METADATA_MODEL`.

Saved metadata lives in Postgres in `document_metadata_extractions` and is copied into retrieval chunk metadata for filtering and ranking. Extraction covers the full document in page-aware batches and records grounded quotes, page/section provenance, scoped firmware/software applicability, and normalized identifier aliases. Existing corpora can be updated independently of parsing with `manuals_rag/scripts/maintenance/backfill_document_metadata.py --apply --all`; production deployments should follow the staged metadata MRV rollout runbook.

Operator inspection is available in Streamlit at `http://127.0.0.1:8601/Document_Metadata`. More detail is in [`docs/architecture/metadata_extraction.md`](docs/architecture/metadata_extraction.md).

## Agentic retrieval comparison

The query API accepts `retrieval_orchestrator=baseline`, `langgraph_agent`, or
`llamaindex_agent`, plus a bounded `max_retrieval_hops`. Both agent backends use
the same retrieval tools and control policy so framework comparisons measure
orchestration overhead rather than different retrieval logic. See
[`docs/architecture/agentic_retrieval.md`](docs/architecture/agentic_retrieval.md).
