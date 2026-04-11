# Metadata Extraction

Document-level metadata is extracted by a small local Ollama model, not by filename or text-pattern heuristics. The current default model is `tinyllama:1.1b`, configured with `OLLAMA_METADATA_MODEL`.

The extraction implementation lives in `packages/parsers/src/manuals_rag_parsers/metadata.py`. It uses Pydantic models to constrain the response shape and then applies source-grounding and validation before the values are persisted or copied onto chunk metadata.

## Extracted Fields

The extractor produces document-level fields for:

- Manufacturer and company names
- Product family and product model values
- Device names
- Part numbers
- Protocol terms
- Settings, parameters, menu labels, and document topics
- Title, document kind, revision date, and effective date

TinyLlama is intentionally treated as unreliable output infrastructure. Invalid JSON or invalid list-field responses are non-fatal and are downgraded to empty values for that field. Grounding and validation then remove common hallucinations, copied filenames, protocol mistakes, and generic non-entity terms.

## Storage

Postgres remains the source of truth for extracted metadata. The `document_metadata_extractions` table stores one current extraction row per source document:

- `source_document_id`
- `document_version_id`
- `model`
- `metadata_json`
- `extracted_at`

The backfill path also updates selected source-document columns and merges the extracted metadata into `retrieval_chunks.metadata_json` so retrieval filters and lexical metadata text can use it.

Qdrant is not the source of truth for this metadata. After chunk metadata changes, embedding jobs should be queued so Qdrant payloads are refreshed from Postgres.

## Retrieval Use

Retrieval uses the extracted metadata for filtering and scoring. Supported metadata-aware filters include product models, product families, part numbers, devices, settings, parameters, and protocol terms. The retriever also preserves scalar aliases such as `product_model` and `product_family` against list payloads such as `product_models` and `product_families`.

## Backfill

Use the maintenance backfill when adding metadata extraction to an existing corpus:

```bash
PYTHONPATH=manuals_rag/packages/parsers/src:manuals_rag/packages/schemas/src:manuals_rag/packages/retrieval/src:manuals_rag/packages/chunking/src:manuals_rag/packages/common/src:manuals_rag/packages/normalizers/src:manuals_rag/packages/observability/src:manuals_rag/packages/answering/src:manuals_rag/packages/evals/src:manuals_rag:manuals_rag/apps \
POSTGRES_DSN=postgresql://manuals:manuals@127.0.0.1:5433/manuals_rag \
REDIS_URL=redis://127.0.0.1:6379/0 \
OLLAMA_URL=http://127.0.0.1:11434 \
OLLAMA_METADATA_MODEL=tinyllama:1.1b \
/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python \
manuals_rag/scripts/maintenance/backfill_document_metadata.py --apply
```

Useful options:

- `--limit N` restricts the number of documents processed.
- `--node-limit N` controls how many leading logical nodes are sent to the metadata model.
- `--no-enqueue-embed` updates Postgres without queueing embedding refresh jobs.

Backfill reports are written to `manuals_rag/test_reports/document_metadata_backfill_*.json`.

## Inspection

Operators can inspect saved metadata and page-level chunk metadata through the Streamlit debug page:

```text
http://127.0.0.1:8601/Document_Metadata
```

The page calls:

```text
GET /debug/documents/{document_id}/metadata?page={page}
```

This endpoint is intended for operator/admin/auditor diagnostics and returns the document metadata extraction row plus the retrieval chunks overlapping the selected page.
