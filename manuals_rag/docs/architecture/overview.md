# Architecture

The system separates:

- Postgres control-plane metadata
- Qdrant retrieval-plane search points
- MinIO object storage artifacts
- Redis queueing
- FastAPI API
- Ingest, embed, and reindex workers
- LangGraph deterministic query flow

Retrieval uses a metadata-first document selection stage before chunk search:

1. Embed the user query and search the per-corpus document metadata index with dense and sparse retrieval.
2. Use the selected `source_document_id` candidates as the document scope.
3. Run the existing dense, sparse, and special-route chunk retrieval pipeline inside that scope.
4. Apply family scoring, semantic completeness, reranking, and answer validation.

Document selection must stay general: use metadata embeddings, sparse metadata signals, reranking, and LLM evaluation. Do not add document-specific heuristics, filename-specific routing, or query rules that only work for a particular manual.

Answers are required to carry citations and may abstain when evidence is weak.

Document-level metadata extraction is a model-backed pipeline, not a filename or text-pattern heuristic pass. The current extractor uses Ollama with `tinyllama:1.1b`, Pydantic response validation, source grounding, and persistence in Postgres. See [metadata_extraction.md](metadata_extraction.md) for the source-of-truth implementation notes, backfill command, and Streamlit inspection path.
