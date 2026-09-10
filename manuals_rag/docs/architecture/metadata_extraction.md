# Metadata Extraction

Document-level metadata is extracted by a local Ollama model, not by filename or text-pattern heuristics. The current default model is `qwen3.5:9b`, configured with `OLLAMA_METADATA_MODEL`.

The extraction implementation lives in `packages/parsers/src/manuals_rag_parsers/metadata.py`. Framework-neutral stage functions are orchestrated by a LangGraph Map–Reduce–Verify workflow:

1. **Map:** process every page-aware batch with bounded parallelism, supplying a deterministic high-recall candidate list to Qwen.
2. **Normalize/reduce:** merge grounded observations into a document-level `MetadataClaim` ledger and detect contradictory scope.
3. **Verify:** give an independent model call only the proposed typed claims and bounded local evidence windows.
4. **Publish:** materialize the existing schema-v2 fields according to claim trust status.

Ingestion and the standalone backfill invoke the same compiled graph. The harvester, reducer, verifier, and publisher remain ordinary Python functions so they can be tested or rerun independently. Ingestion processes the complete normalized document in page-aware batches; it no longer limits document metadata to the first 20 logical nodes.

## Extracted Fields

The extractor produces document-level fields for:

- Manufacturer and company names
- Product family and product model values
- Device names
- Part numbers
- Protocol terms
- Settings, parameters, menu labels, and document topics
- Title, document kind, revision date, and effective date
- Normalized punctuation-insensitive identifier aliases
- Grounded firmware and software applicability records
- A page/section/quote evidence ledger with entity relationship and subject scope
- A claim ledger with normalized values, source method, independent-verification status, and evidence-derived confidence

The stored schema is versioned with `metadata_schema_version`. Version 2 retains the original flat fields for compatibility and adds:

- `metadata_evidence`: exact source quote, page range, section path, relation, subject, confidence, and grounding status
- `routing_product_models`, `routing_part_numbers`, and `routing_protocol_terms`: scoped values suitable for routing
- `normalized_identifier_aliases`: canonical and punctuation-insensitive forms such as `CV-X482` and `CVX482`
- `firmware_applicability` and `software_applicability`: subject-bound records; external PLC/controller examples are retained in evidence but excluded from applicability
- `metadata_claims`: the source-of-truth intermediate claim ledger
- `metadata_pipeline_version`: identifies the orchestration and trust-policy version independently of the materialized schema version

Claim trust states control retrieval use:

- `confirmed`: eligible for exact routing and applicability checks
- `probable`: searchable/ranking-only evidence
- `unresolved`: retained for audit and lexical discovery, but never filtering
- `conflicting`: retained with neutral retrieval treatment and an applicability warning path
- `rejected`: retained for audit only

Model output is intentionally treated as unreliable infrastructure. Common response-shape, enum, null, and date variants are normalized, then malformed/truncated calls are retried. Large batches that repeatedly truncate are bisected and retried recursively. If a scoped batch still cannot be recovered, the extraction fails and the backfill reports the document as failed instead of persisting partial routing metadata.

Version-bearing batches have two completeness gates. If the general scoped pass misses explicit firmware or software-version signals, a focused applicability pass runs; a batch that still lacks grounded version evidence fails safely. After independent verification, the document also fails safely if every grounded claim for a detected version kind was rejected or remained unresolved. Scoped values are accepted only when the model supplies an exact quote found in the same page-aware source batch, version records require an explicit subject, and external PLC/controller versions remain evidence rather than product applicability.

Schema-v2 routing keys require a `confirmed` claim, grounded scoped evidence, and evidence-derived confidence of at least `0.8`. Model self-confidence is not used for publication. Corpus-general structural signals reject repeated short lines, compound code strings, footer/document codes, underscored identifiers, and filename-like values; production logic contains no product- or vendor-name exceptions. A filename identifier may create only an unresolved `mentioned` claim when the exact identifier is independently present on an opening page; filename prefixes and substring matches cannot establish identity. Legacy flat fields remain searchable for compatibility but are not promoted into schema-v2 hard-routing keys without independent verification.

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
OLLAMA_METADATA_MODEL=qwen3.5:9b \
/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python \
manuals_rag/scripts/maintenance/backfill_document_metadata.py --apply
```

Useful options:

- `--limit N` restricts the number of documents processed.
- The default processes every logical node in page-aware batches.
- `--segment-chars N` controls the maximum source characters in each scoped extraction call (default `3000`, matching ingestion).
- `--node-limit N` is a diagnostic-only cap and reduces metadata recall.
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
