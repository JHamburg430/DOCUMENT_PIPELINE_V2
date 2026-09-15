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
- [ ] U3: Re-extract 5–10 representative documents with current MRV schema;
  audit grounded identity, aliases, roles, versions, and completeness before
  applying the sample. Synchronize every chunk and both retrieval indexes;
  verify idempotent resume and pipeline-version consistency.
- [ ] U4: Use verified subject-scoped metadata for explicit model/protocol/version
  constraints. Check version ranges and conflicts; unknowns stay eligible and
  are reported. External-controller facts must not establish primary-product
  applicability. Comparisons use separate entity branches and union evidence.
- [x] U5: Replace fixed verifier truncation with bounded claim-specific evidence
  packing, complete evidence units, source IDs, and explicit omission tracking.
- [ ] U6: Preserve real dependency links in planning; resolve follow-up queries
  using supported intermediate facts and target recovery at missing claims.
- [ ] U7: Retain structured conditions, table-row bindings, and complete procedure
  evidence through synthesis and verification; never weaken support checks to
  increase the pass count.
- [x] U8: Make test runs diagnosable: per-case progress, bounded execution,
  exit-status artifact, final backend coverage, and actionable failure groups.
- [ ] U9: Present outcome/abstention and metadata applicability clearly in the
  app; verify existing Agent Lab controls, traces, citations and reload behavior.
- [ ] U10: Commit and push scoped changes, document measured results and rollback;
  do not include unrelated report churn or silently enable production.

## Tests and acceptance

- [x] T1: Field-consumer inventory identifies all unused/stale fields with evidence.
- [ ] T2: Sample audit passes for each selected document; quotes/pages grounded,
  no external-to-primary leakage, no footer routing, no omitted critical batches.
- [ ] T3: PostgreSQL/Qdrant sample consistency: all active chunks and selector
  payloads current; second backfill is idempotent; interrupted resume verified.
- [ ] T4: Structured retrieval regressions cover normalized aliases, exact/unknown
  models, multi-entity comparisons, version boundaries/conflicts/unknowns,
  external controllers, and unscoped queries without over-filtering.
- [x] T5: Verifier evidence tests include relevant material beyond character 900,
  supporting evidence beyond rank four, full table rows/procedures, competing
  products, bounded payloads, and explicit omissions with no invented citations.
- [ ] T6: Source-audited oracle-evidence experiment separates synthesis/verifier
  failures from retrieval failures. Correct-scope/dependency experiment isolates
  planning. Record incorrect answers, correct complete answers, abstentions.
- [ ] T7: Cross-document and dependent retrieval tests prove intermediate facts
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
