# Readiness integration evidence

## Integration

- Base: `190e61f2c3878daa1badaa716f3bf4fc6a79338c`
- Metadata determinism source: `049faad12436671dcf4e5b5e391694feff4a308c`
- Rollout readiness source: `da56236f76b6e049765c41c15168617c6d0f713f`
- Replayed commits: `a3c20ba`, `db5835c`
- Conflicts: none; both cherry-picks applied without manual resolution.

The two accepted patch path sets are disjoint. The resulting changes do not
touch retrieval, planner, verifier, evaluator, frozen banks, UI/app/server,
services, production flags, embeddings, or live artifacts.

## Tests

- Focused:
  `python -m pytest -q tests/unit/test_metadata.py tests/unit/test_production_acceptance.py`
  — 98 passed.
- Broader relevant:
  `python -m pytest -q tests/unit/test_metadata.py tests/unit/test_metadata_backfill.py tests/unit/test_metadata_persisted_audit.py tests/unit/test_metadata_pilot_roles.py tests/unit/test_production_acceptance.py tests/unit/test_agent_matrix_audit.py tests/unit/test_qdrant_store.py`
  — 131 passed.
- Both successful runs used the existing `compose-api` image with the
  integration worktree mounted read-only and `--network none`.
- One preliminary host invocation did not run because `pytest` was absent
  from PATH; it produced no test result and triggered no dependency install.

## Exact no-write manifest

- Path:
  `test_reports/retrieval_improvement/readiness_integration_20260924/metadata_no_write_manifest.json`
- SHA-256:
  `0e82724dd9d13761808e532560b45cce3cb8e26227b158a723054b7495df4b12`
- Document count: 53
- Unique document IDs: 53
- Unique current version IDs: 53
- Every source SHA-256 is 64 hexadecimal characters.
- The file records each document ID, current version ID, source filename,
  source SHA-256, source byte count, storage URI, Docling artifact URI, page
  count, report key, and expected Qdrant audit path.

## Fail-closed and overlap proof

- `real_schema_fail_closed.json` used the retained
  `agentic-retrieval-matrix-v2` artifact and its real frozen pilot dataset.
  Exit status 1 and `accepted=false`; blockers include stale revision,
  incomplete runtime provenance, and failed correctness cells.
- `missing_evidence_fail_closed.json` deliberately referenced a nonexistent
  artifact. Exit status 1 and `accepted=false` with an unverifiable-input
  blocker.
- The checker computed no held-out/tuning document overlap for the supplied
  real frozen pilot and tuning datasets: `[]`.

## Backup/restore and mutation status

The staging/disposable-only targeted backup and PostgreSQL/Qdrant restore
commands are documented in
`docs/runbooks/metadata_readiness_integration_rehearsal_20260924.md`.
They were not executed because this lane prohibits live mutation and no
disposable backup identifiers were supplied.

No metadata extraction, persistence, embedding enqueue, Qdrant write,
PostgreSQL write, feature-flag change, load test, push, or merge was performed.
The sole corpus access was a read-only PostgreSQL inventory query used to
materialize the 53-document manifest.
