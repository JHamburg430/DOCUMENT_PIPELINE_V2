# Retrieval improvement delivery checklist

## Objective and baseline

Deliver source-supported complete answers, not merely successful retrieval or
passing unit tests. Preserve the original 48-question evaluation unchanged.
Baseline: `test_reports/live_validation_20260914/agent_matrix.json`: LangGraph
14/48 all-gate passes, LlamaIndex 12/48; retrieval 43/48 and 45/48 respectively.
Persisted metadata pilot: 0/3 current audit passes. Production stays disabled
until the rollout gates below pass. No claim of 99.9% accuracy is justified by
this sample. Existing LangGraph, Qdrant, and MRV mechanisms are retained.

## Updates

- [x] U1: Preserve baseline and define separate implementation/test checklists.
- [x] U2: Inventory metadata fields end-to-end: extraction, PostgreSQL document,
  chunk payload, Qdrant document selector, retrieval, verifier; expose gaps.
- [x] U3: Re-extract 5–10 representative documents with current MRV schema;
  audit grounded identity, aliases, roles, versions, and completeness before
  applying the sample. Synchronize every chunk and both retrieval indexes;
  verify idempotent resume and pipeline-version consistency.
- [ ] U4: Use verified subject-scoped metadata for explicit model/protocol/version
  constraints. Check version ranges and conflicts; unknowns stay eligible and
  are reported. External-controller facts must not establish primary-product
  applicability. Comparisons use separate entity branches and union evidence.
- [x] U5: Replace fixed verifier truncation with bounded claim-specific evidence
  packing, complete evidence units, source IDs, and explicit omission tracking.
- [x] U6: Preserve real dependency links in planning; resolve follow-up queries
  using supported intermediate facts and target recovery at missing claims.
- [ ] U7: Retain structured conditions, table-row bindings, and complete procedure
  evidence through synthesis and verification; never weaken support checks to
  increase the pass count.
- [x] U8: Make test runs diagnosable: per-case progress, bounded execution,
  exit-status artifact, final backend coverage, immutable run/source/config
  provenance, stage snapshots, raw judge outcomes, and actionable failure groups.
- [ ] U9: Present outcome/abstention and metadata applicability clearly in the
  app; verify existing Agent Lab controls, traces, citations and reload behavior.
- [ ] U10: Commit and push scoped changes, document measured results and rollback;
  do not include unrelated report churn or silently enable production.

## Tests and acceptance

- [x] T1: Field-consumer inventory identifies all unused/stale fields with evidence.
- [ ] T2: Sample audit passes for each selected document; quotes/pages grounded,
  no external-to-primary leakage, no footer routing, no omitted critical batches.
- [x] T3: PostgreSQL/Qdrant sample consistency: all active chunks and selector
  payloads current; second backfill is idempotent; interrupted resume verified.
- [ ] T4: Structured retrieval regressions cover normalized aliases, exact/unknown
  models, multi-entity comparisons, version boundaries/conflicts/unknowns,
  external controllers, and unscoped queries without over-filtering.
- [x] T5: Verifier evidence tests include relevant material beyond character 900,
  supporting evidence beyond rank four, full table rows/procedures, competing
  products, bounded payloads, and explicit omissions with no invented citations.
- [x] T6: Source-audited oracle-evidence experiment separates synthesis/verifier
  failures from retrieval failures. Correct-scope/dependency experiment isolates
  planning. Record incorrect answers, correct complete answers, abstentions.
- [x] T7: Cross-document and dependent retrieval tests prove intermediate facts
  drive subsequent queries; unresolved dependencies cannot silently pass.
- [ ] T8: Frozen 48-case matrix completes both backends with explicit exit status;
  compare each gate and category to baseline, adjudicate failures against source
  material, fix general causes and rerun affected cases before the full matrix.
- [ ] T9: Separate clean held-out source-backed set, not used for implementation
  tuning; report precision and coverage separately, with uncertainty. Label
  assistant audits honestly rather than calling them human adjudication.
- [ ] T10: Focused and full unit checks pass; running UI covers query submission,
  backend/scope controls, progress, errors, empty states, citations, traces,
  refresh/reload, and correct counts; GPU execution and runtime limits verified.
- [ ] T11: Rollout remains blocked on incorrect supported answers, ungrounded
  metadata, incomplete coverage, or unverified operational recovery. Report any
  unmet gate explicitly; successful abstention does not count as a correct answer.

## Execution record

### Agent matrix artifact contract v2 — 2026-09-19

- `compare_agentic_retrieval.py` now owns its exclusive output lock and immutable
  run ID. It refuses to overwrite a completed artifact and writes a launch record,
  atomic partial/final JSON, and atomic exit status.
- Every artifact records the frozen dataset SHA-256 and ordered dataset-qualified
  case keys, source revision/branch/dirty-diff hash, backend/model/endpoints,
  runtime configuration, start/completion timestamps, and process exit status.
- Baseline, LangGraph, and LlamaIndex records retain full final results plus
  bounded dense, fusion, rerank, final-context, and corrective-stage snapshots.
  Verifier records retain the raw response, parsed response, normalized verdict,
  retry errors, and checked/unchecked status.
- Disposable one-case contract smoke `evaluator-contract-smoke-20260919-01`
  completed with exit 0 and exact frozen dataset hash. The artifact contract
  reconciled, but the semantic case failed evidence, answer, latency, and token
  gates; this is diagnostic proof only and does not advance T8 or enable production.
- A production matrix must be launched from a committed clean source revision,
  with a fresh run ID and output path. Inspect its launch and one-case contract
  before starting all 48 frozen cases.

### First source-backed repair from the v2 trace — 2026-09-19

- Clean one-case artifact `evaluator-contract-smoke-clean-20260919-01` exposed a
  fail-open controller state: the model planner marked its only primary hop
  `required=false`, so an empty required-claim set was treated as sufficient even
  though the independent verifier rejected the evidence. All planner-created
  primary hops are now mandatory; only controller-created recovery hops may be
  optional.
- Single-hop model plans now preserve the original user query and objective so
  product, row, value, bit/column, and output-area qualifiers cannot be lost in a
  lossy planner rewrite.
- Exact structured table verification now binds numeric row coordinates, bit
  columns, cell values, and product scope directly. An exact coordinate match may
  override the generic term-overlap heuristic; scope and coordinate checks remain
  fail-closed.
- Focused changed-path gate: **270 passed**. In clean live artifact
  `agent-matrix-case0-postfix-20260919-02`, LlamaIndex passed all seven layers for
  frozen case `a7ed6202-8a02-566c-a845-55acfe8c07bb::curated1`. The later
  LangGraph artifact retrieved the exact anchor during recovery but was blocked by
  the old preliminary-overlap dependency; replaying that exact persisted record on
  revision `e75bc05` now returns `confirmed` with supporting chunk
  `a7ed6202-8a02-566c-a845-55acfe8c07bb`.
- Do not launch the 48-case matrix yet. First rerun this exact case end-to-end for
  LangGraph on `e75bc05` or later, then run a small representative slice to expose
  the next recurring failure class. Production remains disabled.

### Structured coordinate replay — 2026-09-20

- The first clean current-source replay
  `agent_matrix_case0_current_20260919T201031.json` preserved two truncated raw
  verifier responses and failed closed. Its one-hop LlamaIndex plan was mislabeled
  `parallel`, so normalization did not preserve the authoritative 6-bit qualifier.
- Revision `bcd8074` preserves the original objective/query for every one-primary-hop
  plan regardless of a model's `single`/`parallel` label. Revision `c824c15` also
  routes numeric row/bit/address/command coordinate lookups through hybrid retrieval
  instead of allowing sparse-only first-pass loss. The agentic retrieval unit module
  passes **81/81** after both changes.
- Immutable replay `agent_matrix_case0_hybrid_20260920_0116.json` is complete with
  exit 0, frozen dataset SHA
  `51028f6b5ea6b09515ac361d2b747feb998aaa5098dca710c7e6f4a972871a65`,
  clean source revision `c824c15`, and the exact offset-0 case key. LangGraph and
  LlamaIndex each pass all seven matrix layers, retain the exact table cell, cite
  chunk `a7ed6202-8a02-566c-a845-55acfe8c07bb`, and use one retrieval call with no
  recovery. LangGraph: **61.9 s / 467 measured tokens**. LlamaIndex:
  **55.4 s / 571 measured tokens**.
- A preceding replay was invalidated before retrieval by an embedding-route timeout.
  The aggregate AI gateway had two stale busy slots while both direct backends had
  no loaded model and the local assigned GPU was idle. Its response backpressure
  wait now rejects on client close, allowing `finally` cleanup to clear scheduler
  state. A post-fix embedding probe returned a 1,024-dimensional vector, the live
  replay completed, and both gateway workers returned to idle with an empty queue.
  The gateway service directory is not a Git repository; this operational repair
  has no repository commit.
- This closes only the repaired case boundary. Next run a small representative
  slice covering neighboring categories, then the clean full frozen 48-case matrix.
  T8 and production remain disabled until those broader artifacts reconcile.

- Initial inspection: API bind-mounts this working tree; verifier still passes
  four chunks capped at 900 characters. Existing unrelated report deletions and
  generated artifacts will be preserved, not staged wholesale.

### Field-consumer inventory

| Field group | Persistence / consumer | Remaining limitation |
| --- | --- | --- |
| Models, families, parts, aliases | Ingest and backfill chunk payload; Qdrant selector; scope/ranking | Stored pilot predates current pipeline |
| Protocols, settings, parameters, topics | Chunk and document-selector lexical signals | Protocol mention is not a hard applicability constraint |
| Firmware/software records | Chunk and selector payload; verifier; new query applicability checks | Explicit subject required; ambiguous multi-version requests need branch binding |
| Claim ledger and provenance | Document selector payload; matching-version query-time claim projection; verifier scope-only evidence | No full ledger replication across hundreds of thousands of chunks; action facts still require source content |
| Pipeline/schema version | Document and chunk payload, persisted audit | Must synchronize selector/index after sample passes |

### Current implementation checks

- Evidence packing, version checks, and dependency validation: 80 focused tests
  passed. Metadata reconciliation: 55 tests passed. Broader answering tests are
  running; these figures are not live-answer accuracy.
- Five-document dry-run is running against Qwen3.5 9B. First completed source
  (Lua manual) exposed empty routing and protocol reconciliation defects; no
  metadata has been applied. The run began before the latest fixes and will be
  treated as a diagnostic baseline, not proof of the corrected extractor.
- Exported all 48 benchmark-labelled source sets with zero missing expected
  chunks to `test_reports/retrieval_improvement/oracle_candidates.json`.
  These are candidates for audit, not automatically trusted gold labels.
- Complete-source comparison probe exposed lost named-family scope, uppercase
  family/operator collision, and duplicated model-token matching requirements.
  General fixes implemented; live post-fix verification remains pending.
- Six of seven labelled dependent questions are malformed generated warning
  questions. Preserve them in the frozen benchmark and distinguish genuine
  dependency testing in a separate clean set.

### Document-disjoint held-out pilot — 2026-09-23

- Question generation now accepts repeatable tuning datasets and excludes every
  referenced source document before deterministic holdout selection. The current
  corpus has 53 indexed manuals; 10 are used by the tuning matrix and 43 remain
  eligible for held-out generation.
- A fail-closed freezer verifies active persisted source chunks, exact expected
  snippets, document/version identity, expected terms, unique case IDs, and zero
  tuning-document overlap. It records immutable dataset/source hashes and labels
  assistant source verification with `human_reviewed=false`.
- The first frozen pilot is intentionally one case. It proves the generation,
  freeze, provenance, and gate contracts; it does **not** satisfy T9 or the
  200-case production minimum.
- Clean smoke `heldout-pilot-contract-20260923-02` reconciled on revision
  `f795303` with exit 0. Both agents retrieved and retained the exact answer
  evidence but invented an unasked minimum-clearance requirement, rejected the
  valid source, and abstained without citations. LangGraph used 5,983 tokens at
  54.4 s; LlamaIndex used 16,404 tokens at 95.3 s. Both exceed the 4,000-token
  policy for this simple single-hop control.
- The acceptance gate correctly rejected the pilot for size, category,
  evidence-sufficiency, and claim/citation failures. Production remains disabled.
- Qualitative explanation repair on revision `561a7e9` preserves the original
  question as one hybrid lookup and permits deterministic confirmation only for
  a scoped passage with sufficient term overlap and explicit causal/risk text.
  Focused answering tests pass: 110/110.
- Final immutable smoke `heldout-pilot-contract-20260923-04` is complete with
  clean provenance and exact dataset order. Both LangGraph and LlamaIndex pass
  all seven cells, cite the answer-bearing equivalent source window, use one
  retrieval call and zero model calls, and finish in 27.8 s and 25.9 s.
- The strict acceptance gate now rejects only because the frozen bank has 1 case
  instead of the required 200. This validates the pilot contract, not production
  readiness; deterministic hybrid/structural retrieval remains the default.
- Bank-expansion smoke accepted 25 questions after reviewing 114 source chunks.
  Revision after `ce91c8e` re-anchors generated cases to answer-bearing model
  rows, input/output qualifiers, categorical values, code causes, and complete
  short procedures before freezing. Retrieval-only validation passed 24/25; the
  one miss was an ambiguous generated question that dropped an X-axis qualifier
  while another manual carried a different valid unqualified value. That
  immutable bank was quarantined. The retained 24-case bank is source verified,
  document-disjoint across 9 held-out documents, and still explicitly
  non-human-reviewed. The production minimum remains 200 cases.

### Full 200-case matrix repair and bank revalidation — 2026-09-25

- Immutable v102 matrix artifact `agent_matrix_v102-full-200-1d4b87a` reconciled
  cleanly at revision `1d4b87a` and exact dataset SHA-256
  `841825c4f2a48f9dbd6e822ec0788e7c8bbca857b878614c98b5b7d28ea6f978`.
  LangGraph passed 193/200 and LlamaIndex passed 190/200. The independent
  audit accepted the artifact for diagnostic adjudication only.
- Source-backed routing, deterministic verification, and guarded equivalent-
  evidence scoring repairs were committed as `7890691`, `d5be261`, and
  `337d87f`. The affected answering/evaluation suite passed 454/454. Focused
  immutable replays recovered every repairable failed case on both backends;
  the final IV4 dual-unit temperature replay passed 1/1 on both backends and
  was independently accepted for diagnostic adjudication.
- Re-freezing v102 through the supported bank validator produced additive v103
  with 198 retained cases and two explicit rejections. The known unqualified
  LJ-S8000 movable-range question is ambiguous between stage and non-stage
  ranges. Revalidation also rejected an MU-N11 analog-output question that
  omits the selection criterion. Frozen v102 was not modified.
- v103 is document-disjoint and source-verified but remains
  `human_reviewed=false` and below the 200-case minimum. A new full acceptance
  matrix is intentionally deferred until two source-reviewed replacement cases
  are added. Production remains disabled, with the separate frozen-bank and
  human visual-PDF gates still open.
- Additive v104 restores the immutable bank to 200 cases without modifying
  v102 or v103. The two replacement queries retain the original source anchors
  while explicitly disambiguating the LJ-S8000 non-stage movable range and
  asking for the MU-N11's documented analog-output range inventory instead of
  an unsupported recommendation. The supported freezer accepted 200/200 cases
  with dataset SHA-256
  `1e4b8a22eaa913c810e001c5cab34e8891c07808bcbdd1b263ac10fae75cdc6d`;
  its focused unit suite passed 173/173. Assistant visual review of the two
  source PDF pages is recorded separately and remains `human_reviewed=false`.
  Production stays disabled pending a clean full v104 matrix, independent
  audit, and explicit human visual-PDF sign-off.

### Latest checkpoint

- Pipeline v3 adds explicit claim-ID verification decisions and rejects source text
  too sparse for trustworthy extraction. A one-claim live probe recovered the Lua
  manual identity; a complete v3 pilot has **not** passed yet.
- Latest focused run: **395 passed**. Three regressions from compound-query handling
  were fixed and rerun. A broader unit run excludes `test_pipeline_health.py`
  explicitly because it calls the unavailable live model; it is not a full-suite pass.
- The five-document v3 attempt was cancelled after repeated CUDA OOM failures.
  Warmup now honors the requested context instead of loading default32K first.
  Even a serial16K load failed after parser tests released their GPU memory.
  Other GPU workloads were left untouched; permission to pause them is pending.
- Browser smoke: actual disabled-production submission returns a clear unavailable
  message; error state survives reload; backend/max-hop/trace controls and empty
  states work; incomplete-answer and applicability labels tested with explicitly
  synthetic display fixtures. No successful live answer flow is claimed.
- Comparison oracle now retains both supplied sources. Cable oracle exposed a
  partial-answer fallback, now marked incomplete; wording `connector orientation`
  is not independently supported by the source label `straight serial cable`.
- CLI partial/exit artifacts added. Frozen48-case rerun, valid pilot persistence,
  index consistency, clean held-out evaluation and final release remain open.

- The offline48-case diagnostic exposed a second partial-answer route through
  best-summary recovery. That route now declines compound/comparison recovery;
  source-backed cause/remedy and location/purpose shortcuts remain supported.
- Matrix CLI tests verify exit0 on success, exit1 on failure, and exit143 on
  graceful termination, with no final report written on failure.
- Operational rule from this run: Docling parser tests themselves allocate GPU
  memory. Do not overlap them with live9B validation on these shared GPUs.

### Delivery checkpoint — live validation blocked

- Broad regression run: **977 passed in455.60s**, explicitly excluding live
  `test_pipeline_health.py`. Parser tests ran with CUDA disabled in that test
  process only; no CPU-only inference fallback was enabled for the app.
- Additional new matrix CLI tests: **3 passed**. Follow-up compound-summary,
  location and cause/remedy regressions: **4 passed**.
- Offline oracle diagnostic completed48/48 cases without exceptions. This is
  model-unavailability containment, **not48 correct answers**. The cable case
  now reports insufficient evidence instead of silently claiming completeness.
- Frozen dataset SHA256 still matches the original baseline:
  `51028f6b5ea6b09515ac361d2b747feb998aaa5098dca710c7e6f4a972871a65`.
- Live v3 extraction, sample persistence/index synchronization, the full live
  matrix and held-out answer-quality evaluation are **NOT complete**.
- Other GPU/voice jobs remain running. John was asked whether they may be paused;
  no approval has been received. Do not silently unload their models.
- Feature-branch delivery carries forward the related pending guardrail work;
  unrelated report deletions and other workspace changes are not included.

Resume order: obtain available GPU capacity → verify9B16K-context load and inference
→ rerun five-valid-document v3 dry-run → audit every routing/applicability claim
→ apply only passing sample and verify PostgreSQL/Qdrant consistency → controlled
live oracle/dependency experiment → frozen48-case matrix with explicit exit file
→ source-audited held-out evaluation → update measured outcomes and rollout gate.
No full-corpus backfill, production enabling, or broad accuracy claim is authorized
by the current test results.

### Source audit follow-up

- Completed assistant audit of the five-document dry-run against persisted source
  text. All 37 ledger quotations match their cited pages, but three documents
  fail completeness: Lua software versions, LJ-S head models, and bracket models/
  compatibility rows. The two datasheets pass identity-only checks; visual
  dimensional evidence remains unaudited. This does not satisfy T2.
- Fixed named-editor/runtime version-signal detection, per-value version loss
  checks, an invalid prose-subject fallback, missing plain-Ethernet harvesting,
  and pre-splitting candidate-dense batches before the ten-entity response cap.
- Focused metadata, backfill and persisted-audit tests: 71 passed. A targeted live
  diagnostic still omitted important metadata; neither it nor the original
  pilot was applied. Stronger completeness checks are being verified live.
- Audit artifacts: test_reports/retrieval_improvement/pilot_source_audit.md and
  pilot_quote_audit.json. These are source-text/assistant checks, not human or
  visual PDF adjudication. No corpus changes or production enablement.

- Follow-up live Lua run now rejects missing software versions (exit 1), proving
  fail-closed behavior, not successful extraction. Raw focused probe found an
  unexpected `extracted_versions` response shape and omitted version subjects.
  Explicit output-shape instructions + required nullable subject recovered both
  VisionEditor versions on the exact page-3 passage. Full verification is pending.
- Clarified accessory-catalog roles: the accessory being specified may itself be
  the primary product; compatible-model columns are distinct. No model-specific
  routing values or benchmark answers were hard-coded.

### Version 4 and structured-output diagnosis

- Preserve full grounded quotes: the prior 500-character truncation could remove
  a late table identifier after grounding. Focused live verification now confirms
  all four model-column identifiers on the audited bracket table.
- Add dedicated model-column extraction and fail closed if a model-column value
  is lost during independent verification. Compatibility columns remain distinct.
- Pipeline version is now `evidence_map_reduce_verify_v4`; old v3 outputs must not
  satisfy current-pipeline checks. Reject compatibility-list bullets as titles.
- Local Ollama 0.22.0/Qwen3.5:9b A/B: with the same contradictory prompt and schema,
  think=false returned the prompt's wrong shape; think=true returned the schema's
  required shape. This reproduces https://github.com/ollama/ollama/issues/14645.
  Metadata-only Qwen3.5 calls now enable thinking and reserve up to 8192 generated
  tokens for reasoning plus complete output. Other models retain prior budgets.
  No Ollama/gateway changes or restarts. Other structured agent calls have not
  yet been changed; this potential downstream issue needs its own measured test.
- Broad regression run: 987 passed in 372.10 seconds, excluding live pipeline
  health and with CUDA disabled for that test process. Final focused run after
  compatibility changes: 75 passed. These are not answer-accuracy results.
- Pre-workaround v4 five-document run completed 5/5 (report ...020238.json), with
  expected routing recovered, but it still selected a compatibility bullet as
  title and relied on unreliable schema enforcement. New all-five dry-run with
  the runtime workaround and title fix: /tmp/manuals-pilot-v4-structured.log.
  Source audit, sample persistence/index synchronization and live matrix remain
  open. No corpus changes or production enabling.

### Empty-response and runtime-kind follow-up

- Common structured clients now reject empty content instead of accepting it as
  `{}`; scoped extraction rejects an omitted collection. Neither condition causes
  an unrelated model reload. Regression covers ordinary and streaming clients.
- A pre-workaround report mislabelled Lua's runtime version as firmware. A bare
  non-device version without firmware wording no longer qualifies as firmware.
- 90 focused tests passed; broad unit suite completed **991 passed**, excluding
  live pipeline-health tests. No measured end-to-end accuracy gain is claimed.
- Stopped the outdated thinking-enabled all-five dry-run with exit143 after no
  document checkpoint in18minutes. No sample applied. A bounded raw response
  probe is inspecting runtime behavior before another complete extraction.

- Thinking workaround **retired** after two real-source probes each exhausted8192
  tokens with zero content. Adjusted sampling did not fix it. Non-thinking probe
  returned10 candidates in14s;9 grounded. Metadata again uses bounded non-thinking
  calls with client/schema/grounding checks, not reliance on Ollama formatting.
- Subsequent pilot correctly failed closed on Lua version coverage. Completeness
  matching now normalizes printed `Ver.`/`Version` prefixes without collapsing
  distinct version numbers. A new dry-run records independent verifier inputs
  and decisions so missing values can be diagnosed rather than guessed.

### Five-document sample persistence completed

- Corrected bracket dry-run `document_metadata_backfill_20260917_231629.json`
  passed a source-text audit for all 36 compatibility relationships. Continuation
  claims now retain literal contiguous table spans rather than synthesized quotes
  that omit intervening rows.
- Persisted the two previously withheld documents from their exact audited reports
  after backing up their PostgreSQL and Qdrant state. The other three sample
  documents were already on pipeline v4.
- Final PostgreSQL/chunk audit passes 5/5. Final Qdrant audit passes all 383 active
  chunks and five document selectors with no scope/version payload mismatches;
  all refresh jobs completed. A second ordinary backfill skipped the two newly
  applied documents as current with zero writes or enqueues.
- Focused metadata/backfill/audit tests: 84 passed. Full unit suite: 1019 passed.
- Explicit checkpoint recovery is now implemented with `--resume-report`. A
  deterministic CLI test interrupts during the second document, then proves the
  first completed document is not re-extracted and the unfinished document is
  retried. A database-backed two-document no-write run also resumed the first
  current document and added the second with zero writes or embedding enqueues;
  artifact: `test_reports/retrieval_improvement/pilot_interrupted_resume_20260917_235305.json`.
  Focused verification passed 95 tests; the broader non-live unit suite passed
  1017 tests with 58 warnings in 459.79 seconds. U3/T3 are complete. T2 remains
  conservatively unchecked because this is an assistant source-text audit, not
  human visual-PDF adjudication. Production remains disabled.

- Ollama diagnostics showed the 9B/16K metadata runner failing at the default
  batch 512 with CUDA OOM and stale GPU discovery while other approved workloads
  remained resident. A bounded batch-64 probe loaded and answered in 10.35 seconds.
  Metadata warmup and every metadata chat now use the configurable
  `OLLAMA_METADATA_NUM_BATCH` (default 64); unrelated Ollama calls are unchanged.
  A forced CA-EN100U no-write extraction completed successfully with no corpus
  mutation in `test_reports/retrieval_improvement/metadata_batch64_probe_20260918_000619.json`.

### Controlled oracle and dependency gate — 2026-09-17

- Direct-source case `curated_cross_document_v2::6` confirmed the CV-X482
  `Condition list` fact after repairing legacy-family scope and duplicate direct
  summaries. Live scoped retrieval found two distinct LJ-X8000 settings named
  `Standard Angle` in different tool contexts. The unqualified frozen question
  is therefore an ambiguous oracle anchor, not a retrieval failure; both planners
  now fail closed instead of choosing one definition.
- The combined comparison is deterministically decomposed into independently
  scoped setting branches before model planning. Exact named-setting retrieval,
  structured compatibility/power-source verification, and multipart synthesis
  are covered without product-specific answer hard-coding.
- Clean dependent query: identify the encoder head supported by `CA-EN100U`, then
  determine how that discovered head is powered. Both LangGraph and LlamaIndex
  bound `CA-EN100H` into hop two, confirmed the exact structured rows, retained
  both required claims, and answered that CA-EN100H is powered by CA-EN100U.
  Runtimes were 127.15s and 119.13s respectively.
- Controlled artifact:
  `test_reports/retrieval_improvement/team_retrieval_planner_probe.json`.
  This is an assistant audit of extracted text, not human/PDF or held-out
  adjudication. Full affected suites: **501 passed, 36 warnings**.
- Production and agentic retrieval remain OFF. Next gate: unchanged frozen
  48-case matrix with explicit exit status and source-audited failure review.

### Frozen matrix diagnostic run — 2026-09-17/18

- The frozen dataset SHA remained
  `51028f6b5ea6b09515ac361d2b747feb998aaa5098dca710c7e6f4a972871a65`.
  Three checkpointed segments cover all 48 unique case IDs in exact dataset
  order; the final segment exited 0.
- Raw passes were 12/48 LangGraph and 10/48 LlamaIndex, but this is not an
  acceptance result: 43 verifier calls timed out after NVIDIA discovery failed
  and the 9B verifier loaded with zero VRAM.
- The timeout fails closed before later branches/hops can be certified, so many
  apparent cross-document and dependency misses are downstream artifacts.
- Added verifier-only `OLLAMA_RETRIEVAL_VERIFIER_NUM_BATCH=64`, matching the
  proven metadata batch stabilization. Affected unit suites: 501 passed; full
  unit suite: 1044 passed with 77 warnings.
- Full adjudication: `test_reports/retrieval_improvement/agent_matrix_controlled_20260917_audit.md`.
- T8 remains open until GPU discovery is healthy, affected cases are rerun, and
  a clean full-matrix comparison completes. Production remains OFF.

### Balanced-gateway cable dependency replay — 2026-09-18

- Corrected compose defaults for API/shared workers/UI from local Ollama 11434
  to the maintained balanced gateway on 11437; verifier batch64 is now passed to
  API/shared workers. No running-service recreation was performed during the
  controlled gate.
- Final affected-case artifact:
  `test_reports/retrieval_improvement/agent_matrix_cable_gateway_batch64_v8_20260918.json`
  with explicit exit 0.
- LangGraph and LlamaIndex each passed candidate recall (2/2), document
  retention, dependent-hop structure, evidence sufficiency, and grounded answer.
  Both returned OP-26487 / straight with two valid citations.
- Both still failed the frozen cost cell at 4,606 and 4,571 measured tokens
  versus the 4,000 ceiling. Treat this as performance work; do not relabel it as
  a retrieval miss or close T8 before the clean full-matrix rerun.
- Verification: modified modules **324 passed**; full unit suite **1,046 passed,
  77 warnings**; compose configuration and diff checks passed.

## September 20 evidence-and-retrieval review checklist

This checklist converts the six external review recommendations into release
gates. A checked implementation subtask does not imply production readiness;
each gate closes only with the named persisted evidence. Agentic retrieval and
production rollout remain disabled until all release-blocking gates pass.

### R1 — Trustworthy acceptance evidence

- [x] Persist immutable matrix provenance, exact ordered dataset-qualified case
  keys, per-stage ranked evidence, raw/parsed/normalized judge records, final
  answers, citations, and an explicit process exit status.
- [x] Complete and reconcile the clean 48-case v2 artifact
  `agent_matrix_full48_b20f9a3_20260920_1200.json` (exit 0, 48 unique cases,
  clean source revision `b20f9a3`).
- [x] Generate a deterministic independent audit that recomputes coverage,
  totals, stage retention, unchecked judges, and failure-stage classes from
  individual records rather than trusting the summary.
- [ ] Produce a separate source-backed held-out bank covering tables,
  revisions/version conflicts, comparisons, complete procedures, and
  unanswerable requests. Freeze its hash before execution.
- [ ] Record correct complete answers, incorrect supported answers, and safe
  abstentions separately. Human visual-PDF adjudication remains a distinct
  required sign-off; assistant/source-text review must not be relabelled human.

Independent audit artifact:
`test_reports/retrieval_improvement/agent_matrix_full48_b20f9a3_20260920_1200_independent_recheck_20260920.md`.
It accepts the run for diagnostic adjudication, not production. It independently
recomputes 30/48 full passes for each backend and exposes one unchecked LLM judge
for LangGraph and two for LlamaIndex. Observed dense/fusion/rerank/final-context
snapshots cover 14 required evidence groups per backend; six survive reranking
and nine are present in final context after corrective retrieval. Failure classes
remain separated instead of being flattened into retrieval misses.

### R2 — Corpus-wide metadata and evidence completeness

- [x] Complete the audited five-document metadata pilot, targeted backup,
  PostgreSQL/Qdrant reconciliation (383/383 active chunks and five selectors),
  vector refresh, idempotent re-entry, and interrupted-resume proof.
- [x] Inventory the full active corpus by pipeline/schema version and persisted
  audit state; identify every stale, missing, or mismatched document/chunk/point.
- [ ] Backfill only audited/stale documents with restorable targeted backups;
  verify subject/model/version applicability and literal provenance spans.
- [ ] Prove table headers, units, conditions, footnotes, and complete procedure
  steps survive parsing, retrieval stages, verifier packing, and final context.
- [ ] Complete human visual-PDF review for the difficult source sample.

Read-only full-corpus artifact:
`test_reports/retrieval_improvement/manuals_rag_full_corpus_metadata_audit_20260920.json`.
It audited 53 active documents: 3 pass and 50 fail. Failure counts are
`wrong_schema_version=49`, `wrong_pipeline_version=50`,
`metadata_not_propagated_to_every_chunk=50`, and
`opening_title_identifier_not_routable=24`. This inventory performed no writes;
the next rollout unit is the smallest failing document under a no-write dry-run,
followed by literal-source audit and a targeted backup before any apply.

### R3 — Genuine indexed BM25 experiment

- [x] Add an opt-in Qdrant/FastEmbed BM25 sparse-vector path with collection
  IDF enabled; preserve exact identifiers and existing RRF semantics.
- [x] Migrate/backfill a disposable benchmark collection or named sparse vector
  without modifying the production index until acceptance.
- [x] Compare current hashed term-frequency sparse retrieval against indexed
  BM25 on identical cases, measuring stage recall, evidence recall, latency,
  and downstream answer gates.
- [x] Make an adoption decision from the measured evidence. Keep
  `INDEXED_BM25_ENABLED=false`: BM25 improved required-any recall at 10/20/40
  and required-all recall at 20/40 while reducing mean/p95 latency, but
  required-all@5 fell from 38.3% to 23.4% and no downstream answer-gate run was
  completed. The isolated index remains available for a future fused/downstream
  acceptance run without changing production behavior.

Artifacts: `bm25_index_full_20260920.json` and
`sparse_retriever_comparison_48_20260920.json`.

### R4 — Reranker benchmark and calibration

- [x] Benchmark the current MiniLM reranker against local
  `Qwen3-Reranker-0.6B` on identical persisted candidate pools.
- [x] Sweep candidate-pool size and truncation length; record reranker losses
  where answer-bearing evidence existed before reranking but disappeared after.
- [x] Make the model/calibration decision without comparing raw scores across
  models. Keep MiniLM and its existing fusion path; Qwen is not admitted, so no
  Qwen score calibration is applied to production.
- [x] Verify embedding query instructions and the current 6,000-character
  clipping behavior with focused regressions.

The first reranker artifacts were invalid because the benchmark ignored the
persisted snapshot's `evidence_text` field and scored empty documents. The
benchmark now rejects empty candidates, hashes candidate text as well as IDs,
and the corrected artifacts share candidate-pool SHA-256
`6b4688f9571286e53cd0d54e7141d96922f0d1df33626a48326f363828c1a7fc`.
At the product's 12-candidate pool, Qwen improved required-any@5 from 51.1% to
53.3% but was about 6x slower (122.8 ms mean vs 20.3 ms), with identical
required-all@12 and zero losses. It is not adopted. The matrix snapshots bound
evidence text to 1,200 characters, so 512- and 2,048-token Qwen results were
identical and the 6,000-character clip was not exercised by live candidates;
the clip itself now has an exact focused regression. Instructed Qwen3 query
embeddings improved required-any@10 from 53.3% to 60.0% and required-all@40
from 48.9% to 57.8% at +7.3 ms mean latency, but remain experimental until an
end-to-end held-out answer gate passes.

Authoritative corrected artifacts:
`reranker_minilm_48_pools_evidence_v2_20260920.json`,
`reranker_qwen06_48_pools_evidence_v2_20260920.json`, and
`dense_query_instruction_comparison_48_20260920.json`.

### R5 — Selective expansion and decomposition

- [x] Preserve original identifiers and all product/version/row/value qualifiers
  through expansion and every single-lookup plan.
- [x] Decompose only genuinely dependent or comparative requests into explicit,
  required evidence claims; keep recovery work optional and separate.
- [x] Cap retries and expose unresolved/missing claims in traces and answers.
- [x] Compare expansion/decomposition on/off for the affected source-backed bank
  and make a rollout decision. The 48-case audit passed all 96 backend planning
  contracts, and dependent, parallel, entity-resolution, exact-lookup, conflict,
  and unanswerable categories behaved selectively. Agentic mode remains off by
  default because cross-document full passes stayed 0/10 and p95 latency is
  materially worse than baseline.

Artifact: `planning_selectivity_audit_48_v2_20260920.json`.

### R6 — Measured local-inference optimization

- [ ] Capture per-stage p50/p95 latency, model-load/residency, queue wait,
  verifier timeout, token usage, and worker routing for clean benchmark runs.
- [ ] Detect and exclude stale scheduler/nonresident-worker runs from performance
  acceptance; retain them only as correctness diagnostics.
- [ ] Benchmark the current Ollama path before considering vLLM; test local
  reranking separately from answer/verifier serving.
- [ ] Publish the measured bottleneck, chosen configuration, rollback, and final
  production decision. Serving migration is optional and must not be described
  as an accuracy fix.

The current matrix supports end-to-end and measured model p50/p95, token usage,
and stop reasons, but does not persist per-stage latency, queue wait, worker
route/residency, host GPU telemetry, or verifier-attempt latency. Therefore it is
not performance-acceptance evidence. No vLLM migration is approved; the rollback
and current decision are to retain Ollama + MiniLM with agentic retrieval off.
See `local_inference_performance_audit_20260920.json`. R6 remains a production
observability gate, not an accuracy fix.

### Six-recommendation review status

- [x] R1 reviewed: diagnostic evidence accepted; held-out and human PDF gates open.
- [x] R2 reviewed: v5 rollout mechanism proven on one document; 52 documents remain.
- [x] R3 reviewed: indexed BM25 is useful but remains opt-in pending downstream gates.
- [x] R4 reviewed: corrected benchmark retains MiniLM; query instructions stay experimental.
- [x] R5 reviewed: planning contracts pass; selective agentic rollout remains disabled.
- [x] R6 reviewed: current telemetry is insufficient for a serving migration; no migration.

The review itself is complete. These checks do not close the remaining production
release gates or authorize agentic retrieval.
