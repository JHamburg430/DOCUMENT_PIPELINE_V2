# Agentic retrieval production rollout gates

This is the fail-closed release checklist for agentic retrieval. It defines
evidence contracts; it does not assert that any gate has passed. Keep
`AGENTIC_RETRIEVAL_ENABLED=false` until every gate below is accepted from the
same release revision and production corpus generation.

## Rules

- Evidence is immutable: use a new run ID and output path for every attempt,
  retain the launch/final/exit artifacts, and record SHA-256 values in the
  change ticket. Never replace a failed artifact.
- A missing, partial, malformed, dirty-source, stale-revision, or
  irreconcilable artifact blocks. File existence, mtime, a summary total, or a
  prose claim is never proof.
- Record the release commit, dataset/corpus identity, model/configuration,
  operator, UTC start/end, commands, exit statuses, and artifact hashes.
- Run latency/load gates only in an isolated window. Do not overlap them with
  an acceptance matrix or another material GPU/database workload.
- A diagnostic audit is not production acceptance. A pilot or canary does not
  substitute for the frozen matrix, full-corpus reconciliation, or rollback
  drill.

## 1. Frozen matrix acceptance

**Prerequisites.** The evaluator process has exited; its `.exit` file is `0`;
the final JSON (not `.partial.json`) and the exact frozen held-out and tuning
JSONL bytes are retained; source is clean and equals the release commit. The
held-out bank is document-disjoint from every document reference in the tuning
bank. Do not inspect the active 207-case matrix as complete evidence while its
writer is running.

Run the independent structural audit, then the acceptance gate:

```bash
cd manuals_rag
PYTHONPATH=packages/answering/src:packages/common/src:packages/evals/src:packages/retrieval/src:packages/schemas/src \
  python3 scripts/benchmark/audit_agent_matrix.py \
  --artifact ARTIFACT.json --dataset HELD_OUT.jsonl \
  --exit-file ARTIFACT.exit --json-output AUDIT.json \
  --markdown-output AUDIT.md

python3 scripts/benchmark/check_production_acceptance.py \
  --artifact ARTIFACT.json --dataset HELD_OUT.jsonl \
  --tuning-dataset TUNING.jsonl --expected-source-revision RELEASE_COMMIT \
  --min-cases APPROVED_MINIMUM --max-p95-latency-ms APPROVED_P95_MS \
  --output ACCEPTANCE.json
```

Accept only when `AUDIT.json.accepted_for_diagnostic_adjudication` and
`ACCEPTANCE.json.accepted` are true, both commands exit zero, and all blockers
are empty. The checker recomputes the dataset hash, exact ordered
dataset-qualified keys, unique case coverage, both backend records, required
cell outcomes, citation/abstention gates, p95 latency, source/config
provenance, and document disjointness. Archive all inputs and outputs together.

## 2. Metadata determinism and reconciliation

Follow [metadata_mrv_production_rollout.md](runbooks/metadata_mrv_production_rollout.md).
Before any apply, retain at least two independent dry-run reports for the same
ordered document/version set, release commit, source bytes, pipeline version,
model endpoints/models, prompts, seeds, and sampling configuration. Compare the
canonical claim/evidence ledgers, routing fields, applicability, title, and
scope—not report timestamps or file hashes alone. Any unexplained difference
blocks.

After persistence, run:

```bash
cd manuals_rag
PYTHONPATH=packages/common/src:packages/parsers/src \
  python3 scripts/benchmark/audit_persisted_metadata_mrv.py \
  --corpus-id PRODUCTION_CORPUS_ID --output METADATA_AUDIT.json
python3 scripts/benchmark/audit_qdrant_document_metadata.py \
  DOCUMENT_UUID --output QDRANT_AUDIT_DOCUMENT_UUID.json
```

Accept only when the persisted audit has the current `pipeline`, the exact
expected `document_count`, `checks_passed=true`, `failed=0`, empty
`scope_failures`, and every document has `checks_passed=true`; run the Qdrant
audit for every expected current document and independently require
PostgreSQL/Qdrant document, version, chunk-ID, routing-scope, pipeline, and
selector reconciliation. Retain the exact dry-run
report hash that was applied and prove idempotent re-entry. The current
`test_reports/retrieval_improvement/metadata_rollout_gate_v6_20260924.md`
explicitly blocks this gate due to nondeterministic evidence ledgers.

## 3. Enabled-mode UI proof

Use a disposable isolated stack at the release commit and a non-production
corpus. Capture the feature-disabled baseline and enabled mode through the real
browser UI. Retain the browser test command/exit status, screenshots or trace,
network request/response, and the corresponding API retrieval trace. Proof must
show the selected agent backend, bounded completion, citations rendered from
the returned chunk/document IDs, explicit insufficient-evidence behavior, and
no silent fallback presented as agentic success. A screenshot without request
and trace correlation blocks. No repository tool currently emits a standalone
UI-proof acceptance artifact, so this gate requires reviewed browser evidence.

## 4. Isolated latency/SLO

Run only after the matrix writer and competing inference workloads have exited.
Record worker/model residency and host GPU telemetry before and after the run.
Use the accepted matrix artifact's per-backend `elapsed_ms` records and the
configured threshold passed to `check_production_acceptance.py`; use
`scripts/benchmark/audit_local_inference_performance.py` only for component
diagnosis, not as end-to-end acceptance. Accept only if every correctness gate
still passes and the recomputed p95 for each backend is within the approved SLO.
Missing samples, mixed topology, stale worker state, overlap with another load,
or an undocumented threshold blocks.

## 5. Recovery and fail-closed drill

On an isolated deployment, force planner, retriever, and verifier failures one
at a time. Retain API responses and retrieval traces proving unresolved claims
cannot reach grounded synthesis and that enabled agentic requests return the
documented explicit failure/insufficient-evidence response. Then disable
`AGENTIC_RETRIEVAL_ENABLED` and prove agentic requests fail closed while the
baseline path remains available. Record exact commands, injected fault,
release/config identity, expected and observed status/body/trace, exit status,
and cleanup verification. Every scenario must match the contract; missing raw
responses or traces block.

## 6. Canary and rollback

Before traffic, retain identifiers and restore verification for the PostgreSQL
backup and every affected Qdrant snapshot, the exact pre/post configuration,
canary cohort rule, owner, duration, stop thresholds, and rollback command.
Perform the rollback drill in an isolated environment and reconcile metadata
and baseline retrieval afterward. Cross-scope/invalid citations, unsupported
synthesis, expected-abstention failure, controller-error growth, or SLO breach
is an immediate stop. A written procedure without an executed, timestamped,
reconciled drill blocks.

## 7. Final corpus rollout

Use the bounded waves in the metadata rollout runbook. Each wave requires the
exact applied report hash, document/version IDs, persistence exit status,
ingestion-run IDs all at `indexed`, a passing persisted metadata audit, passing
PostgreSQL/Qdrant reconciliation, and the applicable frozen retrieval slices.
Stop at the first hard-gate failure. The 100% wave requires all current
production documents and active chunks/selectors; a row count without exact ID,
version, pipeline, and routing-scope reconciliation blocks.

## 8. User-facing canary and completion

After gates 1–7 pass for one release, enable only the approved internal or
percentage cohort with baseline retrieval as the immediate rollback path.
Retain the cohort/config revision, start/end, request count, terminal outcomes,
hop/claim metrics, latency distribution, sampled answer/citation audits,
expected abstentions, errors, and user-reported regressions. Roll back on any
hard-gate failure. Expand only after the full observation window reconciles to
the approved shadow evidence.

Production completion requires a single change record linking every immutable
artifact and hash above, zero open blockers, successful rollback proof, and
post-rollout corpus reconciliation. Until then the status is **blocked**.
