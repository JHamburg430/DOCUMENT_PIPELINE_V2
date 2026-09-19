# Disabled agentic retrieval production release — 2026-09-19

## Revision

- Release branch: `release/manuals-rag-agentic-disabled-20260919`
- Deployed revision: `05b8ca0`
- Agentic retrieval: `AGENTIC_RETRIEVAL_ENABLED=false`
- Ollama gateway: `http://host.docker.internal:11437`
- Metadata and verifier batch size: `64`

## Pre-deploy recovery points

- PostgreSQL custom-format dump:
  `/home/john/Desktop/Programming/Document_Pipeline_V2_release_backups/20260919T143225Z/manuals_rag_pre_release_05b8ca0.dump`
- PostgreSQL dump size: 149 MiB
- PostgreSQL dump SHA-256:
  `7325b072b00778a46c76bba7cc53220128678327c64dca601fe1dbfbbf787e01`
- `pg_restore -l` completed successfully.

Qdrant snapshots:

- `manuals_manuals_prod_smoke_document_metadata-1603704880337785-2026-09-19-14-32-37.snapshot`
- `manuals_manuals_mrv_validation_20260910_document_metadata-1603704880337785-2026-09-19-14-32-37.snapshot`
- `manuals_manuals_prod_smoke-1603704880337785-2026-09-19-14-32-38.snapshot`
- `manuals_manuals_mrv_validation_20260910-1603704880337785-2026-09-19-14-32-38.snapshot`
- `manuals_manuals_vendor_keyence-1603704880337785-2026-09-19-14-32-38.snapshot`
- `manuals_manuals_vendor_keyence_document_metadata-1603704880337785-2026-09-19-14-32-54.snapshot`

## Verification

- Compose configuration validation: passed.
- Full unit suite: `1104 passed`.
- Non-live integration suite: `1 passed`, `4 deselected` live tests.
- Real Ollama/Qdrant pipeline checks: chunk retrieval, document-metadata
  retrieval, and production retrieval all passed.
- Post-deploy health: API, UI, PostgreSQL, Qdrant, Redis, MinIO, Prometheus,
  Grafana, ingestion worker, embedding worker, and reindex worker running.
- Deployed retrieval and agentic-controller file hashes match the release
  worktree.
- Metrics endpoint exposes the agentic run, claim, and hop metric families.
- Direct agentic API request returned HTTP 503 with the production-switch
  disabled message.
- Baseline API query returned a grounded medium-confidence answer with one page
  2 citation for the CA-EN100H to CA-EN100U connection procedure.

## Browser smoke

- Search tab returned HTTP 200 and rendered an answer plus citation table.
- Agent tab failed closed with no generated answer.
- Failed agent state survived reload.
- Ingestion tab loaded 57 indexed documents; filtering, empty state, and clear
  behavior worked.
- Run History loaded persisted runs.
- No browser console errors or uncaught page errors were observed.

## Integration repair

The post-merge live pipeline exposed that `qdrant-client==1.17.1` removed
`QdrantClient.search()`. The retrieval store now uses `query_points()` for the
current client and retains a tested legacy fallback.

## Remaining enablement gates

This release deliberately does not enable agentic retrieval. Corpus-wide
metadata waves, the clean two-backend matrix, independent held-out evaluation,
the 5,000-case relation stress gate, enabled UI proof, latency SLO, and the
user-facing canary remain required before enablement.
