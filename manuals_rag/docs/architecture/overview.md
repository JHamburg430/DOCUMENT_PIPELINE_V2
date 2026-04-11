# Architecture

The system separates:

- Postgres control-plane metadata
- Qdrant retrieval-plane search points
- MinIO object storage artifacts
- Redis queueing
- FastAPI API
- Ingest, embed, and reindex workers
- LangGraph deterministic query flow

Retrieval uses dense + sparse search with RRF-style fusion and result reranking. Answers are required to carry citations and may abstain when evidence is weak.

Document-level metadata extraction is a model-backed pipeline, not a filename or text-pattern heuristic pass. The current extractor uses Ollama with `tinyllama:1.1b`, Pydantic response validation, source grounding, and persistence in Postgres. See [metadata_extraction.md](metadata_extraction.md) for the source-of-truth implementation notes, backfill command, and Streamlit inspection path.
