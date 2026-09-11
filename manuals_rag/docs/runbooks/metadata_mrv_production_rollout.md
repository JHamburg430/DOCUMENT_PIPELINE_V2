# Evidence-first metadata and agentic retrieval production rollout

## Purpose

This runbook promotes the evidence Map–Reduce–Verify metadata pipeline and the
bounded agentic retrieval controllers from the isolated three-document pilot to
the production corpus. It separates metadata persistence, metadata audit, vector
refresh, shadow retrieval evaluation, and user-facing activation.

The pilot established feasibility, not corpus-wide readiness: three documents,
9,345 active chunks, three successful persisted metadata audits, and ten
successful retrieval scenarios are too small to justify a one-step rollout.

## Safety controls

- `AGENTIC_RETRIEVAL_ENABLED=false` disables both agentic API orchestrators while
  leaving baseline retrieval available.
- `AGENTIC_RETRIEVAL_MAX_SECONDS` supplies a bounded controller budget; planner
  and verifier calls also retain independent transport timeouts.
- Retrieval-tool exceptions become unresolved claim evidence and produce a safe
  insufficient-evidence response instead of permitting unsupported synthesis.
- Backfill mutations require document IDs, a limit, or an explicit `--all`.
- Backfill writes for one document are one PostgreSQL transaction.
- Current document versions already processed by the active metadata pipeline
  are skipped unless `--force` is supplied.
- A checkpoint report is rewritten after every document by default.
- Three extraction failures stop a run by default; `--max-failures` changes that
  budget and zero disables it.
- Metadata persistence and embedding refresh can be staged separately.

The runtime budget is checked between controller stages. Transport timeouts and
the deployment proxy/request timeout remain the hard upper bounds for a model or
retrieval call already in progress.

## Required release state

Before changing production data:

1. Merge and deploy one commit to API, ingestion worker, embedding worker, and
   operator tooling.
2. Set `AGENTIC_RETRIEVAL_ENABLED=false` for the initial deployment.
3. Confirm Qwen3.5 9B is the configured metadata and verification model.
4. Take a PostgreSQL backup and Qdrant snapshot using the environment's normal
   backup procedure. Record their identifiers in the change ticket.
5. Confirm the API, PostgreSQL, Redis, Qdrant, object storage, Ollama endpoints,
   and workers are healthy.
6. Run the focused and full unit gates from the deployed revision.

Do not begin a mutating backfill without restorable database and vector-store
snapshots. The backfill intentionally has no automatic destructive rollback.

## Phase 1: dry-run extraction canary

Select 5–10 documents that cover different vendors, document kinds, lengths,
tables, protocols, part numbers, firmware, software versions, and documents with
external controller references. Use immutable source-document UUIDs.

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/maintenance/backfill_document_metadata.py \
  --document-id DOCUMENT_UUID_1 \
  --document-id DOCUMENT_UUID_2
```

Review the generated checkpoint report. Reject the canary if routing identifiers
are ungrounded, external firmware/software is published as applicability, primary
product scope is wrong, or critical version-bearing evidence is omitted.

## Phase 2: persist and audit the canary

Persist metadata and chunk payloads without changing Qdrant:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/maintenance/backfill_document_metadata.py \
  --apply --no-enqueue-embed \
  --document-id DOCUMENT_UUID_1 \
  --document-id DOCUMENT_UUID_2
```

Then audit exactly those documents:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/benchmark/audit_persisted_metadata_mrv.py \
  DOCUMENT_UUID_1 DOCUMENT_UUID_2 \
  --output test_reports/metadata_mrv_canary_audit.json
```

**Go/no-go:** every canary document must pass. Confirmed claims must be grounded
on cited pages with confidence at least 0.8, routing identifiers must be backed
by confirmed claims, external versions must not become applicability, and every
active chunk must carry the current pipeline version.

## Phase 3: promote canary metadata into Qdrant

After the persisted audit passes, enqueue refreshes without repeating extraction:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/maintenance/backfill_document_metadata.py \
  --enqueue-current \
  --document-id DOCUMENT_UUID_1 \
  --document-id DOCUMENT_UUID_2
```

Wait for every created ingestion run to reach `indexed`. Re-run the persisted
audit, rebuild the document-level metadata index if the deployment process does
not already do so, and verify the canary collection/points contain the expected
metadata pipeline version.

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/maintenance/rebuild_document_metadata_index.py \
  --corpus-id CANARY_CORPUS_ID
```

## Phase 4: shadow retrieval acceptance

Run LangGraph and LlamaIndex against the same frozen, reviewed evaluation bank.
Include exact identifiers, aliases, ordinary broad questions, multi-claim
questions, dependent questions, two-product comparisons, version conflicts,
external firmware references, and expected abstentions.

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/benchmark/compare_agentic_retrieval.py \
  --dataset tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl \
  --corpus-id CANARY_CORPUS_ID \
  --max-hops 6 \
  --output test_reports/agentic_retrieval_canary_comparison.json
```

Minimum acceptance criteria:

- zero unsupported answers;
- zero invalid, cross-scope, or out-of-corpus citations;
- 100% expected-abstention correctness;
- 100% expected-document coverage for safety/version conflict cases;
- no regression from the accepted baseline on exact-identifier and ordinary
  broad-query slices;
- no runtime-budget or controller-error outcome on the normal canary set;
- p95 latency and model-token use fit the deployment's agreed service budget.

Do not average away a failure in a safety, version, comparison, or abstention
slice. Those are hard gates.

## Phase 5: expand metadata in bounded waves

Increase coverage in waves such as 10%, 25%, 50%, then 100%. For each wave:

1. Run dry extraction and inspect the checkpoint report.
2. Persist with `--apply --no-enqueue-embed`.
3. Audit the selected document IDs or the completed corpus scope.
4. Enqueue only current, audited metadata with `--enqueue-current`.
5. Wait for indexing and verify counts.
6. Re-run the relevant retrieval slices.
7. Pause on the first hard-gate failure or when the failure budget aborts.

The full-corpus mutation requires an explicit acknowledgement:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/maintenance/backfill_document_metadata.py \
  --apply --no-enqueue-embed --all
```

Audit the full corpus before vector promotion:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/benchmark/audit_persisted_metadata_mrv.py \
  --corpus-id PRODUCTION_CORPUS_ID \
  --output test_reports/metadata_mrv_production_audit.json
```

Only after a zero-failure audit:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python scripts/maintenance/backfill_document_metadata.py \
  --enqueue-current --all
```

## Phase 6: user-facing canary

Enable `AGENTIC_RETRIEVAL_ENABLED=true` for an internal or percentage-based
canary while baseline retrieval remains the immediate fallback. Monitor:

- `manuals_agentic_retrieval_runs_total` by policy and terminal outcome;
- `manuals_agentic_retrieval_claims_total` by verifier trust state;
- `manuals_agentic_retrieval_hops`;
- existing query-duration, abstention, model latency, and error metrics;
- answer/citation audit samples and user-reported regressions.

Roll forward only when online behavior matches the accepted shadow gates.

## Stop and rollback

Immediately set `AGENTIC_RETRIEVAL_ENABLED=false` if unsupported synthesis,
cross-scope citations, a sharp controller-error increase, or unacceptable latency
appears. This returns agentic API requests to an explicit 503 while baseline
retrieval remains available; callers should switch to `baseline`.

For metadata failures:

1. Stop new backfill and embedding jobs.
2. Preserve checkpoint/audit reports and affected document/run IDs.
3. Restore PostgreSQL and Qdrant from the recorded pre-rollout snapshots, or
   restore only the affected corpus using the environment's tested procedure.
4. Rebuild the document metadata index.
5. Re-run persisted metadata and baseline retrieval audits before reopening.

Never use `--force` as a recovery shortcut until the extraction defect is fixed
and reproduced on a dry-run canary.

## Production completion criteria

The rollout is complete only when all current production documents have passing
schema-v2 metadata, all active chunks and document-level metadata points are
refreshed, frozen and held-out retrieval gates pass, metrics are visible, the
kill switch has been exercised, backups/restores are verified, and operator
ownership for alerts and rollback is recorded.
