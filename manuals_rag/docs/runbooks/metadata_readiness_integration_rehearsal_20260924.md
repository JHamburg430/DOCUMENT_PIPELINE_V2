# Metadata readiness integration rehearsal — 2026-09-24

This runbook is a no-write rehearsal for corpus
`manuals_vendor_keyence`. It does not authorize extraction, persistence,
embedding, collection replacement, or feature activation. The authoritative
inventory is
`test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_manifest.json`.
Every command below must use that exact file and a clean release revision.

## Hard guards

- Keep `AGENTIC_RETRIEVAL_ENABLED=false`.
- The first two metadata runs omit `--apply`, `--enqueue-current`, and
  `--force`.
- Never restore into the production DSN or Qdrant endpoint. The restore
  examples use new, disposable targets named `manuals_rag_restore_rehearsal`
  and `qdrant-restore-rehearsal`.
- Stop if the manifest SHA-256, count, document IDs, version IDs, or source
  hashes differ from the accepted gate.

## Exact no-write gate

```bash
cd manuals_rag
MANIFEST=test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_manifest.json
test "$(jq -r .document_count "$MANIFEST")" = 53
test "$(jq -r '.documents | map(.document_id) | unique | length' "$MANIFEST")" = 53
test "$(jq -r '.documents | map(.version_id) | unique | length' "$MANIFEST")" = 53
sha256sum "$MANIFEST"

mapfile -t DOCUMENT_ARGS < <(
  jq -r '.documents[].document_id | "--document-id", .' "$MANIFEST"
)

PYTHONPATH=packages/common/src:packages/parsers/src \
  python3 scripts/maintenance/backfill_document_metadata.py \
  "${DOCUMENT_ARGS[@]}" \
  --output test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_run1.json

PYTHONPATH=packages/common/src:packages/parsers/src \
  python3 scripts/maintenance/backfill_document_metadata.py \
  "${DOCUMENT_ARGS[@]}" \
  --output test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_run2.json
```

The two outputs are candidates only. Compare their ordered document/version
sets and canonical claim/evidence ledgers with
`audit_lr_t_metadata_determinism.py`-equivalent canonicalization. A timestamp,
file hash, or summary count is not enough. Any difference, extraction error,
missing document, source hash drift, or nonliteral evidence blocks persistence.

## Targeted backup command

The repository's backup utility reads the 53 PostgreSQL rows and their current
metadata/chunks plus the matching Qdrant chunk and selector points. It does not
modify either store.

```bash
MANIFEST=test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_manifest.json
mapfile -t DOCUMENT_ARGS < <(
  jq -r '.documents[].document_id | "--document-id", .' "$MANIFEST"
)
PYTHONPATH=packages/common/src:packages/retrieval/src \
  python3 scripts/maintenance/backup_document_metadata.py \
  "${DOCUMENT_ARGS[@]}" \
  --source-report test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_run1.json \
  --source-report test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_run2.json \
  --output test_reports/retrieval_improvement/readiness_integration_20260924/targeted_metadata_backup.json
sha256sum test_reports/retrieval_improvement/readiness_integration_20260924/targeted_metadata_backup.json
```

Before any future production apply, also take collection snapshots and a
PostgreSQL custom-format dump using the environment's approved backup system.
The targeted JSON is not a substitute for those independently restorable
backups.

## Disposable PostgreSQL restore rehearsal

These commands intentionally name a new database. They must not be changed to
the production database.

```bash
export RESTORE_DB=manuals_rag_restore_rehearsal
test "$RESTORE_DB" = manuals_rag_restore_rehearsal
createdb --host STAGING_POSTGRES_HOST --username STAGING_OPERATOR "$RESTORE_DB"
pg_restore --exit-on-error --single-transaction \
  --host STAGING_POSTGRES_HOST --username STAGING_OPERATOR \
  --dbname "$RESTORE_DB" POSTGRES_CUSTOM_DUMP
psql --host STAGING_POSTGRES_HOST --username STAGING_OPERATOR --dbname "$RESTORE_DB" \
  --set ON_ERROR_STOP=1 --command \
  "select count(*) = 53 as exact_documents
     from source_documents
    where corpus_id = 'manuals_vendor_keyence'
      and current_version_id is not null;"
```

Restore verification must additionally compare every document ID, current
version ID, source SHA-256, extraction pipeline, and active chunk ID against
the manifest and targeted bundle. A row count alone fails closed.

## Disposable Qdrant restore rehearsal

Use an isolated Qdrant instance named `qdrant-restore-rehearsal`. Recover each
production collection snapshot under its original collection name only inside
that isolated instance:

```bash
export RESTORE_QDRANT_URL=http://qdrant-restore-rehearsal:6333
case "$RESTORE_QDRANT_URL" in
  *qdrant-restore-rehearsal*) ;;
  *) echo "refusing non-rehearsal Qdrant target" >&2; exit 64 ;;
esac

curl --fail-with-body --silent --show-error \
  --request PUT "$RESTORE_QDRANT_URL/collections/manuals_manuals_vendor_keyence/snapshots/recover?wait=true" \
  --header 'content-type: application/json' \
  --data '{"location":"SNAPSHOT_URL_DENSE","priority":"snapshot"}'
curl --fail-with-body --silent --show-error \
  --request PUT "$RESTORE_QDRANT_URL/collections/manuals_manuals_vendor_keyence_bm25/snapshots/recover?wait=true" \
  --header 'content-type: application/json' \
  --data '{"location":"SNAPSHOT_URL_BM25","priority":"snapshot"}'
curl --fail-with-body --silent --show-error \
  --request PUT "$RESTORE_QDRANT_URL/collections/manuals_manuals_vendor_keyence_document_metadata/snapshots/recover?wait=true" \
  --header 'content-type: application/json' \
  --data '{"location":"SNAPSHOT_URL_SELECTORS","priority":"snapshot"}'
```

After recovery, run the persisted metadata audit against the restored
PostgreSQL database and one Qdrant audit per manifest document against the
restored endpoint. Require exact dense/BM25/PostgreSQL chunk-ID equality,
exactly one current selector per document, matching document/version/pipeline
fields, and green collection status.

Rehearsal status for this integration lane: **commands prepared, not executed**.
Execution is deliberately deferred to disposable/staging targets with real
backup identifiers.

## Ordered next-step checklist

1. **No-write dry run:** verify the manifest hash/count, source revision, and
   clean tree; produce two new immutable dry-run reports with no apply flags.
2. **Source audit:** compare all 53 ordered IDs/versions/source hashes and
   canonical ledgers; visually adjudicate the required representative cohort;
   stop on any drift or unverifiable evidence.
3. **Backup proof:** create and hash the targeted bundle plus independently
   restorable PostgreSQL and Qdrant backups; complete the disposable restore
   rehearsal above.
4. **Persistence:** only after approval, apply the exact accepted dry-run hash
   with `--apply-planned-report --no-enqueue-embed`; record exit status and
   affected IDs.
5. **Reconciliation:** require a zero-failure persisted audit, exact
   PostgreSQL/dense/BM25 chunk-ID equality, selector equality, and indexed
   ingestion runs before promotion.
6. **Zero-write re-entry:** rerun the same scope without `--force`; require
   every document to be reported current/skipped, zero changed rows, zero new
   ingestion runs, and unchanged PostgreSQL/Qdrant counts and payload hashes.

Any missing artifact, hash mismatch, unverified source, failed restore,
unexpected write, or reconciliation difference keeps rollout blocked.
