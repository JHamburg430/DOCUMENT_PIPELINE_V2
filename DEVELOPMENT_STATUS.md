# Development Status

This file tracks implementation progress for this workspace against the source-of-truth requirements in [.AGENT.md](/home/john/Desktop/Programming/Document_Pipeline_V2/.AGENT.md).

## Current State

The workspace now contains a working production-style baseline application under [manuals_rag](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag). The stack is running locally with:

- FastAPI API
- Postgres control plane
- Qdrant retrieval plane
- MinIO object storage
- Redis queueing
- Ingest, embed, and reindex workers
- Basic UI
- Prometheus and Grafana containers

Real document ingestion and live query execution have been verified against PDFs in [Technical_Documents](/home/john/Desktop/Programming/Document_Pipeline_V2/Technical_Documents).

As of March 31, 2026, the codebase also includes:

- Heuristic structured-node extraction for warnings, cautions, procedures, tables, and spec-like records in the parser path
- Structured chunk families for `table_record`, `datasheet_record`, `spec_record`, `procedure_record`, `warning_record`, and `brochure_fact`
- Query-class-aware retrieval routing that prefers structured chunk families for safety/how-to/spec lookups
- Answer post-validation that reconstructs missing citations and warns when retrieval spans multiple document versions
- Expanded unit coverage around parser classification, normalization, chunk typing, retrieval analysis, and answer validation

As of April 11, 2026, the codebase also includes:

- TinyLlama-backed document metadata extraction using `tinyllama:1.1b`, Ollama, Pydantic response validation, source grounding, and non-fatal per-field fallbacks for invalid model JSON
- A Postgres `document_metadata_extractions` table for saved document-level metadata
- Chunk metadata propagation for companies, product families/models, devices, part numbers, protocol terms, settings, parameters, document menu labels, and document topics
- Retrieval filter and scoring support for the extracted metadata fields
- A maintenance backfill script at [backfill_document_metadata.py](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/scripts/maintenance/backfill_document_metadata.py)
- A Streamlit debug page at [Document_Metadata.py](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/apps/streamlit_debug/pages/Document_Metadata.py) and API endpoint `GET /debug/documents/{document_id}/metadata?page=...`
- Architecture notes in [metadata_extraction.md](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/docs/architecture/metadata_extraction.md)

## Status Against `.AGENT.md`

### Phase 1: Bootstrap

Status: Mostly complete

Implemented:

- Monorepo layout under [manuals_rag](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag)
- Docker Compose stack in [docker-compose.yml](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/infra/compose/docker-compose.yml)
- FastAPI service in [main.py](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/apps/api/main.py)
- Postgres, Qdrant, MinIO, Redis
- Health endpoint
- Local auth skeleton
- Initial schema migration in [init.sql](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/infra/migrations/init.sql)

Still missing:

- Real migration system beyond bootstrap SQL
- Signed internal service auth
- Production-grade secrets handling

### Phase 2: Ingestion Core

Status: Partially complete

Implemented:

- File upload API
- Object storage integration
- Document registry and version rows
- Ingestion queue via Redis
- Ingest worker
- Raw parse artifact persistence to MinIO
- Parse quality scoring
- Replayable stage separation between ingest and embed
- Model-backed document metadata extraction through Ollama `tinyllama:1.1b` with Pydantic validation and source grounding
- Persistence of extracted metadata in `document_metadata_extractions`

Still missing:

- Full failure class coverage beyond current parse path
- Replay from all stage boundaries
- More complete run diagnostics

### Phase 3: Normalization

Status: Improved, still partial

Implemented:

- Logical node model
- Text normalization
- Section path handling
- Keyword inference
- Provenance fields
- Structured normalization for spec nodes
- Structured normalization for table nodes
- Procedure-step normalization with stable step prefixes

Still missing:

- Strong table extraction from true document structure rather than text heuristics
- Rich multi-step procedure grouping beyond single-step promotion
- Warning/caution hierarchy beyond text-pattern promotion
- Spec extraction from layout-aware source semantics rather than line heuristics

### Phase 4: Chunking

Status: Improved, still partial

Implemented:

- L1/L2/L3 hierarchical chunking in [hierarchical.py](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/packages/chunking/src/manuals_rag_chunking/hierarchical.py)
- Priority scoring for warnings/procedures/spec-like nodes
- Deterministic chunk ids
- Explicit `table_record` chunks
- Explicit datasheet/spec chunk families
- Explicit brochure fact chunks for short brochure-style claims

Still missing:

- Lineage tables beyond current logical-node linkage
- Higher-confidence brochure fact extraction than the current short-paragraph heuristic
- More granular lineage for row-level table provenance

### Phase 5: Indexing

Status: Partially complete

Implemented:

- Dense embedding pipeline via Ollama
- Sparse vector generation
- Qdrant collection creation
- Named dense and sparse vectors
- Qdrant upsert logic
- Metadata backfill tooling that updates Postgres chunk metadata and enqueues embed refresh jobs

Still missing:

- Delete/rebuild tooling
- Index migration tooling
- Late interaction / multivector phase 2 work

### Phase 6: Retrieval

Status: Improved, still partial

Implemented:

- Query analysis
- Filter building
- Dense retrieval
- Sparse retrieval
- RRF-style fusion
- Local reranking
- Scroll-based lexical fallback when vector search is empty
- Raw `/search` and `/explain-retrieval` endpoints
- Query-class-aware chunk routing for safety/how-to/spec requests
- Better dedupe at the section/chunk-family level
- Product model and manufacturer auto-filters when the query names them
- Metadata-aware filtering and scoring for extracted product families, product models, part numbers, devices, settings, parameters, and protocol terms

Still missing:

- Parent expansion logic that is more selective
- More exact version-aware metadata filters
- True version-aware retrieval policy for latest/superseded conflicts
- Retrieval strategies specialized for troubleshooting/error-code lookups beyond baseline keyword routing

### Phase 7: Answering

Status: Improved, still partial

Implemented:

- LangGraph deterministic workflow
- JSON-first answer schema
- Citation-bearing responses
- Live answer generation via Ollama
- Abstention fallback when retrieval fails
- Answer post-validation for missing citations
- Conflict disclosure when retrieval evidence spans multiple versions

Still missing:

- Strong confidence calibration
- Explicit second-pass retrieval retry behavior
- Validator logic that checks answer claims against citations instead of only checking response structure

### Phase 8: Evaluation

Status: Started

Implemented:

- Unit tests
- Live integration test
- Smoke benchmark script
- Small smoke eval fixture set

Still missing:

- Benchmark corpus at the scale required by `.AGENT.md`
- Recall/MRR/NDCG reporting
- Citation coverage reporting
- Regression set for brochure/manual conflicts
- Acceptance gates and release thresholds

### Phase 9: Admin/UI

Status: Started

Implemented:

- Minimal web UI for upload and query
- Streamlit debug page for inspecting extracted document metadata and selected-page chunk metadata

Still missing:

- Real admin diagnostics
- Parse quality dashboard
- Chunk distribution view
- Retrieval trace viewer
- Corpus/permission management UI
- Evaluation dashboard

### Phase 10: Hardening

Status: Early

Implemented:

- Basic metrics endpoint
- Prometheus and Grafana containers
- Basic logs through service stdout

Still missing:

- Rate limits
- Poison queue handling
- Backup and restore drills
- Immutable audit logs
- Runbooks for failure recovery
- Strong RBAC and enterprise auth

## What Has Been Verified

Verified with real execution:

- Stack boot on Docker
- API health
- Real upload of `CA-EN100U_Datasheet.pdf`
- Real ingestion into Postgres/MinIO/Qdrant
- Real query returning grounded answer with citation
- Full local test run
- Live integration test
- Live smoke benchmark

Additionally verified in the current editing session:

- Python syntax compilation for all modified modules and new tests via `python3 -m py_compile`
- Focused metadata, filter, retriever, and chunking tests via the shared bootstrap virtualenv: `33 passed`
- Live metadata backfill completed for 9 documents with 0 failures, with report output under [test_reports](/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag/test_reports)

## Important Deviations From `.AGENT.md`

These are current deviations, not final design decisions:

- Auth is intentionally minimal to avoid blocking core development progress.
- Docling is coded as the preferred parser path, but the current verified runtime is using the PyMuPDF fallback path unless Docling is installed and stable in the service image.
- Retrieval quality is baseline-grade, not yet production-optimized for large manual corpora.
- Eval coverage is far below the required production target.

## Recommended Next Steps

Priority order:

1. Upgrade parsing to production-grade Docling runtime in containers and verify it on real manuals, not only datasheets, so the current heuristic structured extraction can be replaced or backed by layout-aware parsing.
2. Strengthen retrieval for latest/superseded-version questions with explicit version ranking and conflict handling at retrieval time, not only answer-time warnings.
3. Build a larger real evaluation corpus and automated metrics reporting.
4. Expand admin diagnostics and operational observability.
5. Restore a usable Python packaging toolchain in this workspace so local tests can run without relying on prebuilt images.
6. Return to enterprise auth/RBAC hardening after the retrieval and eval baseline is strong.

## Working Definition Of “Done”

Relative to `.AGENT.md`, this project is not done until all of the following are true:

- Real Docling-backed ingestion is stable on representative manuals
- Structured chunk families for tables/specs/procedures/warnings are implemented
- Retrieval is robust across major query classes
- Answers are version-aware, citation-grounded, and validator-checked
- Eval coverage is broad and automated
- Operational hardening and runbooks are in place
- Permission handling is production-grade
