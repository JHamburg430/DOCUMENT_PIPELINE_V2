# Local Runbook

1. Install dependencies into the shared bootstrap environment:
   `/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python -m pip install -r manuals_rag/requirements.txt`
2. Start Ollama if it is not already running.
3. Start the stack:
   `docker compose -f manuals_rag/infra/compose/docker-compose.yml up -d --build`
4. Open the UI at `http://127.0.0.1:8601`.
5. Use `Authorization: Bearer admin-token` for admin operations and `Bearer user-token` for end-user queries.

## Metadata Extraction

The local metadata extractor expects Ollama to have `tinyllama:1.1b` available. The stack sets `OLLAMA_METADATA_MODEL=tinyllama:1.1b` for the API and workers.

To backfill extracted metadata for existing parsed documents and refresh retrieval payloads:

```bash
PYTHONPATH=manuals_rag/packages/parsers/src:manuals_rag/packages/schemas/src:manuals_rag/packages/retrieval/src:manuals_rag/packages/chunking/src:manuals_rag/packages/common/src:manuals_rag/packages/normalizers/src:manuals_rag/packages/observability/src:manuals_rag/packages/answering/src:manuals_rag/packages/evals/src:manuals_rag:manuals_rag/apps \
POSTGRES_DSN=postgresql://manuals:manuals@127.0.0.1:5433/manuals_rag \
REDIS_URL=redis://127.0.0.1:6379/0 \
OLLAMA_URL=http://127.0.0.1:11434 \
OLLAMA_METADATA_MODEL=tinyllama:1.1b \
/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python \
manuals_rag/scripts/maintenance/backfill_document_metadata.py --apply
```

The backfill writes `document_metadata_extractions`, merges metadata into `retrieval_chunks.metadata_json`, and queues embed jobs unless `--no-enqueue-embed` is passed.

Inspect document and page metadata in Streamlit at `http://127.0.0.1:8601/Document_Metadata`.
