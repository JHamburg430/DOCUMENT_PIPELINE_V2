# Test And Eval Alignment Todo

Use this checklist to bring tests, evals, and reports back in line with the current retrieval architecture:

`query -> metadata document selection using dense+sparse metadata index -> existing dense/sparse/special chunk retrieval -> rerank/LLM evaluation`

- [x] Decide and document upload dedupe semantics.
  - Corpus-scoped dedupe is the intended behavior.
  - `apps/api/main.py` now includes `corpus_id` in the duplicate lookup.
  - `infra/migrations/init.sql` now scopes the duplicate constraint to `(tenant_id, corpus_id, sha256)`.

- [x] Fix live pipeline-health eval for duplicate uploads.
  - Corpus-scoped dedupe prevents the live stage from receiving a document from another corpus.
  - Live pipeline-health passed against the local stack after the schema update.

- [x] Update retrieval debug eval to include metadata document selection.
  - `packages/evals/src/manuals_rag_evals/retrieval_debug.py` calls `select_documents_from_metadata()` before chunk search.
  - Reports include `metadata_document_selection`.
  - `tests/unit/test_retrieval_debug.py` covers the stage.

- [x] Add eval coverage for metadata-driven document selection.
  - `tests/unit/test_retriever.py` covers metadata selection before chunk search and no-hit fallback.
  - The regenerated retrieval debug report includes `LJ-X8080 z axis repeatability`.
  - Review note: this verifies the requested regression query, but the stronger long-term eval is a corpus-level assertion set where every query has an expected document id and the metadata-selection stage is scored before chunk retrieval.

- [x] Regenerate current retrieval eval reports after the eval code is updated.
  - Ran `scripts/benchmark/run_large_retrieval_eval.py`.
  - Ran `scripts/benchmark/run_tmp_eval_suite.py`.
  - Ran `scripts/benchmark/run_retrieval_debug_report.py`.
  - Ran `scripts/health/run_pipeline_health_report.py`.

- [x] Update `test_reports/README.md`.
  - Replaced stale report links with regenerated April 13 report links.
  - Added a note that reports validate metadata-first retrieval.

- [x] Fix `Makefile`.
  - Removed the old `Document_Pipeline/.venv` Python path.
  - Removed the extra `cd manuals_rag`.
  - Made `PYTHON` overridable.

- [x] Decide whether `run_smoke_eval.py` remains useful.
  - It remains as a legacy/simple answer smoke eval.
  - The local runbook points current retrieval-quality work to the retrieval eval runners.

- [x] Add a CI-style split for unit vs live tests.
  - Added the `live` pytest marker.
  - Default `pytest` excludes live tests.
  - Added a `make live` target and documented the explicit live test command.

- [x] Add cleanup or stronger isolation for live integration tests.
  - Corpus-scoped dedupe avoids cross-corpus duplicate detection.
  - Live integration tests passed against the local stack after applying the schema update.
  - Review note: the running local database was patched manually to match the migration; other environments still need the migration applied normally.

- [x] Update pipeline-health local retrieval coverage.
  - Renamed direct search coverage to `local_chunk_store_retrieval`.
  - Added `production_retrieval` using `retrieve()` with metadata document selection.

- [x] Add document metadata index health checks.
  - Added `document_metadata_index` to pipeline health.
  - Added test coverage for metadata hits and no-hit fallback.

- [x] Remove generated `__pycache__` noise from review outputs.
  - Confirmed no `__pycache__` or `.pyc` files are tracked.
  - Existing `.gitignore` covers Python bytecode.

- [x] Document the intended retrieval architecture.
  - Updated `docs/architecture/overview.md` with the metadata-first retrieval flow.
  - Explicitly documented that document selection must not use document-specific heuristics.

## Post-Implementation Review

- [x] Confirm production document selection is metadata-index driven.
  - `retrieve()` calls `select_documents_from_metadata()` before dense/sparse/special chunk search.
  - `build_filters()` no longer derives document-selection filters such as `product_models`, `product_families`, `part_numbers`, or `manufacturer` from query parsing.
  - No production retrieval code was found hardcoding the requested `LJ-X8080` document id, filename, or path.

- [x] Confirm the metadata selection path uses both dense and sparse signals.
  - `QdrantStore.search_document_metadata()` fuses metadata dense and sparse hits.
  - Metadata records are enriched with generic chunk metadata signals before indexing.
  - Retrieval debug reports expose the selected metadata documents and the applied `source_document_id` scope.

- [x] Classify fixture-specific references.
  - The `LJ-X8080 z axis repeatability` query is retained only as a regression/eval query.
  - The pipeline-health query still uses the checked-in fixture document family; that is acceptable for a fixture health check, not production routing logic.

- [x] Add document-selection eval assertions with expected document ids.
  - `source_document_id` is the expected document id for each eval case.
  - Completed by scoring whether the metadata selection stage returns that id at rank 1 and within top K before chunk retrieval.
  - Reports now include metadata-selection recall/rank-1 rates in `run_large_retrieval_eval.py` and the tmp eval suite aggregate.

- [x] Generalize or remove single-vendor query-analysis leftovers.
  - Completed by removing the query-analysis `manufacturer` field and the `keyence` contribution to `filter_strictness`.
  - Query analysis no longer contains a single-vendor detector for document selection.

- [x] Generalize metadata extraction prompt examples.
  - `packages/parsers/src/manuals_rag_parsers/metadata.py` previously included Keyence-style examples for companies, product models, product families, and part numbers.
  - Those examples were not direct document-routing logic, but they could bias metadata extraction toward the fixture/vendor set.
  - Completed by replacing the single-vendor examples with neutral/fictitious example values and adding a unit guard.

- [x] Add an eval guard for query-alignment scoring.
  - The term normalization and query-alignment changes are generic and are not document-specific, but they were motivated by the axis-repeatability failure.
  - Add mixed-manufacturer/mixed-domain cases to prove the scoring does not overfit technical table lookups or reduce performance for narrative/procedure searches.
  - Completed with mixed-vendor spec lookup and procedure-vs-spec query-alignment unit tests.
