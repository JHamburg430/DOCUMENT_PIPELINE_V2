# Metadata lane checkpoint

## Current evidence

- Adopted existing bounded five-document run instead of duplicating GPU work.
  `/tmp/manuals-pilot-v4-validated.exit` is **1**. Report
  `../document_metadata_backfill_20260915_023227.json` records Lua failure:
  `Verification lost explicit software versions: ['4.2.0020']`; no documents applied.
- Raw `structured_budget_probe.json`: done_reason length, eval_count8192,
  content length0, total110.22seconds. This is exhausted output, not GPU inactivity.
  Existing non-thinking/client-validation workaround retained.
- Pre-workaround report020238 is not acceptable: duplicate Lua5.1.4 firmware claim,
  compatibility bullet selected as bracket title. Previous source audit remains
  authoritative for known omissions; process success is not source accuracy.

## Changes owned by metadata lane

- Deterministic extraction now supports a subject's explicit parenthesized version
  list. Each value is only a **mention**, never inferred applicability. Mixed-subject
  parentheses are rejected; separate source/model verification remains for stronger
  relationships. Addresses second-version disappearance without overriding a
  rejected applicability assertion.
- Persisted audit now compares every chunk's actual routing, aliases and version
  applicability values with its document metadata, and verifies document-version
  identity. A matching pipeline stamp alone cannot mask stale constraints.

## Validation

- `docker exec -e CUDA_VISIBLE_DEVICES= compose-api-1 python -m pytest
  tests/unit/test_metadata.py tests/unit/test_metadata_persisted_audit.py -q`:
  **76 passed**,0.73seconds. CPU-only applies to this test process, not model inference.
- Existing pre-workaround020238 report:52/52 ledger quotes match saved source page segments, but known firmware-kind/title defects still fail semantic accuracy.
- Real read-only persisted audit SQL executed successfully; Lua0/1 due old pipeline
  and incomplete propagation, report `team_persisted_audit.json`.
- Current Lua-only dry-run `/tmp/manuals-team-lua.log`,300s timeout, noapply,
  session76427. Source audit will follow completion; do not duplicate its GPU lease.

## Next action / remaining gates

Inspect explicit exit/report; audit quoted page spans, correct software kinds and
all three known version/subject pairs, then remaining four representative documents.
Do not persist until every sample passes source audit and parent integration agrees.
PostgreSQL/Qdrant synchronization and downstream answer improvement remain unproven.
No production enablement, full corpus apply, commits or unrelated workload interruption.

## 2026-09-17 resume

- The prior two-document guarded dry-run finished with exit 0 in
  `../document_metadata_backfill_20260915_025252.json`. It retained all four LJ-S
  heads and routed SZ-FB31 as an accessory, but the bracket catalog's 36 explicit
  compatible GL-R models still existed only in legacy flat fields, not the verified
  relationship ledger. The sample therefore remained incomplete and was not applied.
- Added deterministic parsing for `Recommended compatible models` table columns.
  It preserves continuation-row ownership and emits `compatible_with` claims keyed
  to GL-FB1000/1400/1900/2400; compatible entries are never promoted to primary IDs.
  The completeness gate now requires every compatible-column identifier to survive.
- Offline execution against the real saved page-2 source segment recovered all 36
  compatibility rows under the four expected GL-FB subjects. Focused metadata,
  backfill, pilot-role and persisted-audit verification: **84 passed**.
- A bracket-only no-write live attempt (`/tmp/manuals-pilot-compatible-rows.log`)
  was stopped after repeated Ollama HTTP 500 load failures and reload thrashing.
  Its client-side exit marker was invalid and was preserved as
  `/tmp/manuals-pilot-compatible-rows.exit.invalid-client-status`. No report or
  corpus mutation was produced. Ollama logged GPU discovery timeout/context-cancelled
  load failures while other GPU workloads remained untouched.

Next: rerun this one-document fail-closed dry-run only after Ollama can load the
metadata model reliably; audit all 36 relationships in the generated ledger before
sample-only persistence. Production remains off.

## Completed targeted live diagnostic

- `/tmp/manuals-team-lua.exit` **0**; output
  `../document_metadata_backfill_20260915_023512.json`,1planned,0applied,0failures.
- Assistant source-text audit:20/20 quotes match their saved cited page segments.
  Confirmed Lua5.1.4 and XG VisionEditor4.2.0020/5.1.0020 survive; editor versions
  remain external references, not controller applicability. No firmware claim.
  XG-7000/XG-8000 routing and Ethernet/RS232C are source-supported.
- The same audit found one extra RS232C mention assigned the filename as subject,
  absent from its quote. Added universal subject grounding for all relations,
  including mentions, so a quoted value cannot launder an invented subject.
  This final guard has offline validation only; report023512 predates that guard.
- Final focused run across metadata, persisted audit and backfill: **81 passed
  in0.78seconds**. An earlier misnamed backfill test path failed collection (exit4);
  corrected `tests/unit/test_metadata_backfill.py` passes4/4 within the81.
- GPU lease released; parent notified. Next bounded run should cover the four
  remaining documents plus guarded Lua when scheduling permits. All-five acceptance,
  sample persistence and index checks remain open. No source audit is being
  represented as human or visual PDF adjudication.

## 2026-09-17 completed sample gate

- Main Ollama Qwen3.5 9B loaded at 16K context and the bracket-only no-write run
  completed successfully in `../document_metadata_backfill_20260917_231629.json`.
- The audit found and fixed a provenance bug: continuation relationships had
  combined the table header, subject row, and later row while omitting intervening
  rows. Compatible-column claims now retain the literal contiguous table prefix.
- Bracket result: 36/36 compatible GL-R models, subject distribution 12/15/6/3
  across GL-FB1000/1400/1900/2400, no missing/extra/duplicate relationships,
  no compatible model promoted to a primary claim, and no nonliteral quote span.
- Exact audited LJ-S and bracket reports were persisted after an 8.2 MiB targeted
  PostgreSQL/Qdrant backup. PostgreSQL audit passed 2/2 before vector refresh.
- Final five-document state passes PostgreSQL 5/5 and Qdrant 5/5: 383 active
  chunks, 383 matching vector points, five matching document selectors, and no
  pipeline/version/scope payload mismatch. All three refresh jobs completed.
- Re-entry skipped both newly applied documents as current with zero writes or
  enqueues. Focused tests: 84 passed. Full unit suite: 1019 passed in 392.94s.
- Production remains disabled. Interrupted-resume proof and all downstream live
  answer-quality gates remain open; these results are metadata/index consistency,
  not end-to-end answer accuracy.

## 2026-09-17 interrupted-resume gate

- Added explicit `--resume-report` support with fail-closed report parsing,
  duplicate rejection, document-version matching, and mode-aware completion rules.
- Deterministic CLI coverage checkpoints the first document, interrupts during the
  second, then proves resume retains the first result and retries only the second.
- Database-backed no-write continuation resumed CA-EN100U and added VJ-H500CX as
  current without writes or embedding enqueues. Artifact:
  `pilot_interrupted_resume_20260917_235305.json`.
- Focused metadata/backfill/pilot-role/persisted-audit suite: **95 passed**. U3 and
  T3 are complete; broader non-live unit suite: **1017 passed, 58 warnings in
  459.79 seconds**, with live pipeline health excluded and CUDA hidden. T2 remains
  open pending human visual-PDF adjudication.
- A live Ollama 9B probe terminated in the runner. No extraction retry or unrelated
  GPU workload disruption followed. Production remains disabled.
