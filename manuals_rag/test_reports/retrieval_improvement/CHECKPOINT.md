# Active checklist execution — resumed checkpoint

User authorized checklist + implementation/testing until complete. Production OFF.
Checklist: docs/runbooks/retrieval_improvement_checklist.md. Preserve unrelated dirty
work/report churn on codex/agentic-default-off-20260911; no commits yet.

Implemented: complete-chunk 12KB evidence packets, shown-ID verification, conservative
subject/version checks and conflict demotion; verified document claim ledger projected
only onto matching document-version candidates, scope-only verifier input; dependency
DAG validation and content-grounded anchors; independent named-family comparison scopes;
VS uppercase parsing/row matching; multi-part fallback no longer claims completeness;
metadata claim-ID independent verdicts with natural assertions and complete-decision
validation, protocol/external reconciliation, garbage text fail-closed; pipeline now v3.
Model calls: no reload-on-timeout, bounded relevance/summary tokens; warmup now uses
requested num_ctx (previous warmup used model default32K before16K extraction).
Matrix CLI now writes atomic partial report after each case, --exit-file including SIGTERM.

393 focused tests passed before latest multipart/warmup changes. Current rerun log
/tmp/manuals-improvement-focused-final.log, exec30377: two failures so far; inspect summary.
Warmup tests exec12226. Need final full suite, UI, delivery.

Live probes: comparison now returns both source facts/citations. Cable oracle returned
only model, not orientation. Source says serial cable straight, not explicitly connector
orientation: benchmark wording is ambiguous; do not force an unsupported interpretation.
Artifacts oracle_probe_prefix_baseline.json and oracle_probe_postfix_fast9b.json.
Six of seven dependent-labelled frozen questions are malformed warning questions;
retain frozen baseline, construct independent clean held-out set.

Canary corrected v2 dry-run completed5, no application: LJ-S8000 and VJ-H500CX route;
Lua empty before new assertion verifier (one-claim v3 live probe succeeded); XGX source
is58nodes59chars mostlyq, original1322pagePDF image-only, Docling OCRdisabled. Must
reparse/OCR or quarantine; replaced with CA-EN100U for next five-valid-source canary.
Safety manual routes GL-FB1000/1400,GL-R,SZ-V: audit roles before persistence.

New v3 canary /tmp/manuals-metadata-canary-v3.log exec78589 CANCELLED deliberately
via exact timeout PID26778 after confirmed CUDA OOM repeated load failures. No data
applied. Ollama11434 GPU3090 is shared with voice,embedding,otherwork; no unrelated
models unloaded. 16K/512batch health probe still failed500 OOM. Now testing direct
16K/64batch tiny request exec98019; inspect output. If succeeds add optional configured
batch size consistently to warmup+chat (default unchanged), rerun canary serially.
If fails, actual live validation blocked on GPU headroom; no blind retries.

Remaining: resolve focused failures; new5doc v3 audit then persistence/index consistency;
controlled live oracle/dependency tests; frozen48matrix with exit and cleanheldout;
UI query/trace/warnings/reload; finalchecklist/results, scopedcommit/push. Do not claim
all complete or improved live accuracy before evidence.

FINAL OFFLINE CHECKPOINT: broad977passed455.60s excluding livepipelinehealth;
3CLItests and4fallbacktests passed; browser smoke actualdisabledquery+reload+
fixtures passed noJSerrors. Offline48allreturned; cable nowinsufficientTrue.
Serial16K health load afterparserGPUrelease STILL500. NoGPUworkloads paused;
request_user_input_async GPUcapacity permission pending. No v3 metadata applied.
Commit/push feature branch next; then blocked livegates remain explicit, notdone.

DELIVERED: commit986b2dd39dc6b91cfbd341933a5785b38fb354f9 pushed to origin/codex/agentic-default-off-20260911. All feature files clean; unrelated report churn remains. Pending GPU-workload question unanswered. No live v3 corpus application or production enablement.

GPU RETRY 2026-09-15 ~00:53 UTC: John requested retry, not interruption of other
workloads. Qwen3.5:9b16K inference succeeded in11.46s, returned {"ok":true};
Ollama size_vram=size=9306972800, context_length16384. GPU3090 utilization91%.
Resumed serial five-valid-document v3 dry-run, NO --apply, --force --max-failures1,
3600s timeout. Exec session58114. Log /tmp/manuals-metadata-gpu-retry-20260915.log;
exit /tmp/manuals-metadata-gpu-retry-20260915.exit. Source IDs same as corrected
pilot except XGX replaced by14879385-0608-44ee-85fc-0b79ac5c1d96 (CA-EN100U).
Still running at checkpoint, no completed-document result yet. Inspect report/log
and exit before claiming pilot success. No other workloads stopped, production OFF.

SOURCE AUDIT RESUME: original five-doc report 20260915_005340 audited against
logical-node text; 37/37 quote/page match, but Lua versions, LJ-S heads and safety
bracket identities incomplete. Two datasheets pass identity only. No application.
New fixes and tests in metadata.py/test_metadata.py; 71 focused passed.
Targeted first live run /tmp/manuals-pilot-audit-repair.log exited0 but fails
quality: Lua version subject is invalid prose, bracket routing only GL-R.
New stricter run /tmp/manuals-pilot-version-gate.log running exec63704, timeout1200;
check outcome before duplicate launch. It includes per-version checks and rejects
prose fallback. Small follow-up allows a grounded firmware value in mixed source
lines to count for completeness (unit fixture), irrelevant to Lua-only runtime.
Other workloads untouched. Continue source completeness remediation, not corpus
application or production enabling. Full checklist remains incomplete.

LATEST: committed+pushed e8fedc3 and e585163;71 metadata/backfill/audit tests pass.
Full Lua rerun /tmp/manuals-pilot-version-shape-full.log exits0,
report document_metadata_backfill_20260915_015102.json preserves Lua5.1.4 and
VisionEditor4.2.0020/5.1.0020, correct subjects+quotes; Ethernet/RS232C route.
Versions remain mentioned (not applicability) conservatively. No apply.
Model coverage rerun exec32052 /tmp/manuals-pilot-model-coverage.log running;
LJ-S completed with four expected heads now verified/routed. Bracket still pending.
All outputs still v3. Need finish bracket audit and current representative sample
before applying; preserve original failing artifacts, do not count unit passes
as retrieval accuracy. Other GPU workloads untouched.

RUNTIME ROOT CAUSE: local Ollama0.22.0 Qwen3.5 thinkfalse ignores JSON schema;
thinktrue obeys same schema/prompt. Probe /tmp/manuals-schema-enforcement-probe.json.
Metadata-only workaround added with8192token budget; no daemon/runtime changes.
Current all5 dryrun exec98579 /tmp/manuals-pilot-v4-structured.log,1800s timeout,
noapply. Earlier all5 report020238 completed but before thinking/title fixes.
Broad987passed372.10s excluding livehealth; final75focusedpassed. Quote truncation
fixed, focusedmodelcolumn pass plus coveragegate, pipelinev4, bullet-title rejection.
Need currentdryrun audit before anything applied; no corpus changes so far.

SUPERSEDING 02:26 UTC: old all5 thinking run stopped intentionally after ~18min
without document checkpoint, exit143 verified from native98579. Host kill lacked
permission; exact container timeout PID29670 terminated successfully. Other jobs
untouched. Not a successful extraction. Report020725 remains[].
Empty model content was accepted as{} in both common structured paths; now rejected
without reload. Scoped responses must explicitly include entities/decisions.
Bare runtime/software version cannot be called firmware without device/firmware
support.90focused tests pass. Broadtest native21077 log/tmp/manuals-final-empty-output-unit.log
stillrunning; no final result yet. Single-call raw budget probe native65362,
/tmp/manuals-structured-budget-probe.log; body stored in retrieval_improvement/
structured_budget_probe.json whencomplete. Inspect done_reason/eval_count/content
length before next sample run. No watcher exists (OpenClaw tool inference failed).
Current new scoped changes not yet committed; prior commit3c9f9e8 pushed.

02:34 UTC supersedes: two thinking probes both8192tokens/0content; retired that
workaround. Nonthinking firstbatch14s10candidates9grounded. All5nonthinking run
023227 failedLua coverage4.2.0020,exit1. Fixed prefixnormalization incoveragegate
(Ver./Version vsbarevalues),69metadata tests pass. New tracedall5run native76688
/tmp/manuals-pilot-v4-traced.log with/tmp/manuals-pilot-v4-traced.exit; persists
verifiercandidate+decision diagnostic pilot_verification_trace.json. Noapply.
Broadunit991passedexit0. Lastpushed97693f9; policyreversion/prefixfix pendingcommit.

2026-09-17 RESUME: prior two withheld documents completed dry-run exit0 in report
025252, but bracket acceptance remains blocked:36 compatible GL-R table rows were
legacy-only and absent from verified claims. Added deterministic compatible-column
claims with GL-FB row subjects, continuation handling, literal table-contract
confirmation, and fail-closed completeness. Real saved source yields36/36 across
GL-FB1000/1400/1900/2400; metadata-focused84pass, adjacent retrieval/index264pass.
Bracket-only live retry was stopped after repeated Ollama500 load failures/reload
thrashing; no report/apply. Invalid client exit0 renamed
/tmp/manuals-pilot-compatible-rows.exit.invalid-client-status. Log preserved at
/tmp/manuals-pilot-compatible-rows.log. Do not retry until metadata model loads
reliably; then run only bracket dry-run, audit ledger, and persist only if complete.
Production OFF; other GPU workloads untouched.

2026-09-17 23:20 UTC SUPERSEDES: Ollama9B/16K schema probe passed in10.5s and
bracket-only dry-run completed exit0. Source audit caught synthesized noncontiguous
quotes for continuation rows; fixed compatible-column evidence to preserve the
literal contiguous table prefix. Corrected report231629 passes36/36 relationships
(GL-FB1000/1400/1900/2400 =12/15/6/3), no missing/extra/duplicates, no compatible
model promoted to a primary claim, and49/49 ledger quotes are literal page spans.
Focused84pass; full unit1019pass in392.94s with CUDA hidden from tests.

Restorable pre-apply PostgreSQL/Qdrant backup:
retrieval_improvement/pilot_preapply_backup_20260917_231629.json (2 withheld docs,
225 chunks,225 vector points,2 selectors). Applied exact audited LJ-S report025252
and bracket report231629 without re-extraction or enqueue:153+72 chunks. Persisted
audit passed2/2. Enqueued those2 plus CA-EN100U stale-status recovery; all3 embed
runs completed. Final all5 PostgreSQL audit passed5/5; Qdrant audit passed383/383
active chunks and5/5 selectors with zero payload/version mismatches. Second ordinary
backfill skipped both newly applied docs as current with zero writes/enqueues.
Artifacts: pilot_acceptance_20260917_231629.json,
pilot_postgres_audit_all5_20260917_231629.json,
pilot_qdrant_audit_all5_20260917_231629.json. Qdrant client1.15.1/server1.17.1
compatibility warning remains non-blocking. Production and agentic retrieval OFF.

NEXT: verify an intentionally interrupted sample-resume path before checking T3;
then controlled live oracle/dependency experiments, frozen48 matrix with explicit
exit, clean held-out source audit, UI live-success flow, and rollout decision.

2026-09-17 23:53 UTC SUPERSEDES: backfill checkpoints can now be reopened with
--resume-report. Resume acceptance is mode- and database-aware: no-write accepts
planned/skipped entries; apply accepts only a matching current persisted version;
enqueue-current accepts only a confirmed prior enqueue. Duplicate document/version
entries and malformed reports fail closed. A deterministic CLI regression wrote
document A, interrupted while B began, then resumed without re-extracting A and
completed B. A real database-backed no-write continuation also resumed CA-EN100U
and added VJ-H500CX with zero writes/enqueues; artifact
retrieval_improvement/pilot_interrupted_resume_20260917_235305.json. Focused
metadata/backfill/audit tests:95pass. U3/T3 complete. Ollama live probe failed
with llama runner termination, so no GPU extraction was retried and downstream
live oracle/matrix gates remain open. Broad non-live unit suite:1017pass,
58warnings,459.79s, with live pipeline health excluded and CUDA hidden. Production OFF.
