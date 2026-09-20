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

2026-09-18 00:06 UTC: Ollama logs confirmed the failed metadata load used16K
context/batch512, requested8.7GiB, then hit CUDA OOM after stale GPU discovery.
Batch64 loaded and answered in10.35s without unloading other workloads. Added
OLLAMA_METADATA_NUM_BATCH default64 consistently to metadata warmup and chat;
other Ollama consumers remain unchanged. Focused common+metadata suite108pass.
Forced CA-EN100U no-write extraction then completed planned/exit0 in
retrieval_improvement/metadata_batch64_probe_20260918_000619.json with no corpus
mutation. Production OFF. Next gate: controlled live oracle/dependency experiment.

2026-09-17 LOCAL CONTROLLED RETRIEVAL GATE: direct-source and scoped live probes
are complete in `team_oracle_probe.json` and `team_retrieval_planner_probe.json`.
General repairs cover legacy-family scope, duplicate summaries, named-setting
table retrieval, scoped comparison planning, exact compatibility/power-source
verification, dependent demonstrative references, LlamaIndex structural routing,
and complete multipart synthesis. Frozen case6's unqualified LJ-X8000 `Standard
Angle` anchor is ambiguous: live retrieval found distinct blob-numbering and
proximity-exclusion definitions, so both planners correctly abstain. Clean
CA-EN100U→CA-EN100H→power-source dependency passes both backends with exact hop-2
binding, both required claims retained, complete grounded answer, and runtimes
127.15s/119.13s. Affected suites:501pass/36warnings. U6/T6/T7 complete.
Production/agentic retrieval OFF. NEXT: unchanged frozen48 matrix with explicit
exit artifact, then source-audited failure review; do not relabel the frozen set.

2026-09-18: Frozen-matrix diagnostic coverage is complete across three durable
segments (1 + 22 + 25 cases); IDs exactly match all 48 frozen cases and the final
segment exited 0. Raw matrix passes: LangGraph 12/48, LlamaIndex 10/48. The run is
not an acceptance result: 43 verifier calls timed out while NVIDIA discovery
reported an unknown GPU error and the 9B verifier loaded with zero VRAM. This
blocked later parallel/dependent hops and contaminated candidate/document scores.
Added `OLLAMA_RETRIEVAL_VERIFIER_NUM_BATCH=64`; affected suites 501 pass/37
warnings; full unit suite 1044 pass/77 warnings. T8 remains open pending
healthy-GPU affected-case and full-matrix reruns. Audit:
`agent_matrix_controlled_20260917_audit.md`. Production remains OFF.

2026-09-18 BALANCED-GATEWAY FOLLOW-UP: compose defaults now route API/shared/UI
Ollama traffic to the maintained 11437 gateway and pass verifier batch64 through
API/shared workers; the running stack was not recreated. The final targeted
RS-232C cable replay `agent_matrix_cable_gateway_batch64_v8_20260918.json`
completed exit0. Both backends planned the explicit `then` dependency, retained
2/2 evidence, confirmed both claims, and grounded `OP-26487` + `straight` with
two citations. All correctness layers passed. Only latency/token-cost failed:
4606/4571 measured tokens versus the frozen4000 ceiling. This is now classified
as a performance gate; T8 and production remain OFF pending cost work and a
healthy full48 rerun. Modified modules324pass; full unit1046pass/77warnings;
compose config and diff checks pass.

2026-09-20 SIX-RECOMMENDATION REVIEW: the clean committed-source 48-case matrix
`agent_matrix_full48_b20f9a3_20260920_1200.json` completed exit0 and independently
reconciles 48/48 exact ordered cases; both agent backends have 30/48 full passes.
It is accepted for diagnostic adjudication only. Production blockers remain a
separate held-out bank, source-backed failed-answer adjudication, and human visual
PDF review. V5 metadata rollout was safely proven for VJ-3302 only: exact audited
report apply, targeted backup, 5 PostgreSQL chunks, matching Qdrant chunks/selectors,
and skipped-current idempotent re-entry all passed. Full corpus v5 audit is 1 pass /
52 fail; no broad metadata apply is authorized by this checkpoint.

Indexed Qdrant BM25 improved large-cutoff recall and latency but regressed
required-all@5, so `INDEXED_BM25_ENABLED=false` remains the default. The initial
reranker benchmark was invalid because it scored empty persisted snapshot text;
the benchmark now consumes `evidence_text`, rejects empty candidates, and hashes
candidate text. Corrected equal-pool evidence keeps MiniLM: at pool12 Qwen improves
required-any@5 by2.2points but is about6x slower with identical required-all@12
and loss count. Instructed dense queries remain experimental pending downstream
held-out answer gates. Planning selectivity recheck passes48/48 for both backends,
but cross-document full pass stays0/10 and agentic latency remains high, so
agentic retrieval stays OFF. The performance audit lacks per-stage, queue, route,
residency, GPU, and verifier-attempt telemetry; no vLLM migration is approved.

Verification: changed-boundary549pass/2 fixture-dependent deselected; broad unit
execution1138pass with15 container failures rerun on host as82/82pass and three
temporary-fixture/directory failures explicitly environmental; non-live integration
1pass/4live deselected; compileall, compose config, diff check, metadata/Qdrant
reconciliation, and reranker pool-hash reconciliation pass. Summary:
`six_recommendation_review_20260920.md`. Production and agentic retrieval remain OFF.
