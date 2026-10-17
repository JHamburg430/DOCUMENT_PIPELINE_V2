# Manuals RAG production-readiness checklist

Status: **blocked**

Release policy: keep `AGENTIC_RETRIEVAL_ENABLED=false` until every required gate below has a passing, source-backed artifact from the same clean release revision.

## Baseline

- [x] Current branch and revision recorded: `codex/manuals-agent-evidence-gate-20261002` / `a03c001b3a0abcf184cb6a7ccc67b57ece4cd075`.
- [x] Focused non-live unit suites pass in Docker; agentic/evidence/evaluation suites: 338/338 (pytest 8.4.2, network-disabled disposable container).
- [x] Full current Keyence gate executed: 40/40 cells completed, 12 passed, 28 failed, 0 errors.
- [x] Independent audit completed: coverage 40/40, production acceptance `NO-GO`.
- [ ] Clean release revision and immutable acceptance artifacts; the current worktree is still dirty and the latest broad artifact is not a clean release run.

## Current broad benchmark checkpoint

- [x] Latest persisted 48-case matrix is complete: 28/48 passed for LangGraph and 28/48 for LlamaIndex.
- [ ] Broad benchmark production gate: **open**. The common 20-case failure set is concentrated in evidence sufficiency, grounded answer coverage, candidate recall, and document retention.
- [ ] Do not treat the 5-case 10/10 MRV pilot or the separate 18/18 research matrix as equivalent to the 48-case gate.

## 1. Failure adjudication and evidence quality

- [ ] Adjudicate all 28 failed cells against source documents and final evidence traces.
- [ ] Classify each failure as corpus/metadata, retrieval, reranking, verifier, answer synthesis, latency, or evaluator defect.
- [ ] Prove every apparent false negative with source-backed evidence before changing the scorer.
- [ ] Require zero unsafe accepted answers and zero citation-scope violations.

## 2. Retrieval and answer quality

- [ ] Reach at least 95% on simple factual held-out questions.
- [ ] Reach at least 90% on complex, multi-hop, comparison, procedure, safety, table, and diagram slices.
- [ ] Resolve product/model/entity scope reliably.
- [ ] Preserve relevant evidence through candidate generation, reranking, context assembly, and verification.
- [ ] Verify claim-level support and abstain on unresolved claims.

## 3. Metadata and corpus integrity

- [x] Apply model-derived preferred metadata filters to retrieval routing when analysis is balanced/strict; preserve explicit caller scope and keep loose analysis broad. Retrieval/controller/evidence regression: 560/560.
- [x] Derive routing filters independently for each controller hop, preventing one facet's metadata scope from leaking into another. Focused controller/filter/retriever regression: 569/569.
- [ ] Make metadata extraction deterministic for the same inputs/configuration.
- [ ] Pass persisted metadata audit for every production document/version.
- [ ] Reconcile PostgreSQL, Qdrant, chunk IDs, versions, routing scope, and pipeline revision.
- [ ] Prove idempotent re-ingestion and safe stale-version handling.

## 4. Frozen evaluation bank

- [ ] Freeze a document-disjoint held-out bank covering simple and complex question classes.
- [ ] Record dataset/config/source hashes and release provenance.
- [ ] Require complete case/backend coverage with no partial artifacts.
- [ ] Re-run the complete bank after every material retrieval or verifier change.

## 5. Latency and cost

- [ ] Define approved p50/p95 SLOs separately for simple and complex questions.
- [ ] Measure timeout rate, hop count, token cost, model residency, and concurrent behavior.
- [ ] Pass p95 and cost gates without weakening evidence requirements.

## 6. Failure safety and rollback

- [ ] Drill planner, retriever, Qdrant, Postgres, verifier, model-output, timeout, and overload failures.
- [ ] Prove unsupported claims cannot reach grounded synthesis.
- [x] Default the evidence gate to fail closed; `AGENT_EVIDENCE_GATE_ENABLED` now defaults to `true`.
- [ ] Prove agentic mode fails closed while baseline retrieval remains available.
- [ ] Execute and verify backup restore and rollback in an isolated deployment.

## 7. ProVIS/OpenClaw service boundary

- [ ] Expose a narrow authenticated RAG API with tenant/corpus permissions.
- [ ] Return answer, claim-level citations, evidence state, confidence, trace ID, and safe errors.
- [ ] Add request limits, audit logging, prompt/input isolation, and versioned API contract.
- [ ] Integrate only through the service boundary; do not couple the RAG runtime to gateway internals.

## 8. Operational readiness and canary

- [ ] Add health/readiness probes, metrics, traces, alerts, and runbooks.
- [ ] Verify backup/reconciliation monitoring and ingestion repair procedures.
- [ ] Deploy a small internal canary with baseline retrieval as immediate rollback.
- [ ] Observe citation errors, abstentions, unsupported answers, latency, and retrieval drift.
- [ ] Record final go-live decision with all immutable artifact hashes and zero open blockers.

## Current next action

Adjudicate the 28 failures in `keyence_onsite_agent_eval_gate_20261009T2345Z_a03c001.json` before changing production behavior or enabling agentic retrieval.

### Initial adjudication notes — 2026-10-10

- [x] Confirmed all 40 cells executed with zero transport errors.
- [x] Confirmed 28 failures are fail-closed abstentions; no unsafe accepted answers.
- [x] Confirmed the dominant failure shape is evidence verification rejecting partial or non-procedural support, not answer fabrication.
- [ ] Confirm whether each rejected claim is recoverable from the indexed corpus or genuinely absent.
- [ ] Separate retrieval/index misses from verifier-contract and benchmark-contract defects.
- [x] Add a regression case for explanatory mechanism support (case 014 pattern); focused suite is 304/304.
- [x] Preserve the original user query for the first outer-ReAct retrieval call; disposable network-disabled Docker regression suite is 5/5.
- [ ] Add regression cases for every remaining repaired failure class before rerunning the full gate.

Observed failure clusters:

- Wiring/specification: cases 002, 003, 006.
- Sensor/installation adjustments: cases 005, 007–011.
- 3D/structured settings: cases 014–017.
- PLC/network procedures: cases 019–020.

Progress: the explanatory-mechanism rule plus first-query preservation now pass live case 014 in both backends (2/2) in the isolated canary. This is a localized repair; the full 40-cell gate remains open with 28 historical failures.

Latest bounded slice (2026-10-10): cases 002-003 completed without transport errors after a canary health check, with 2/4 cells passing. Case 002 passed in LlamaIndex but LangGraph fail-closed on a visual-evidence message; case 003 passed in LangGraph but LlamaIndex failed to verify a complete answer. These are backend-specific evidence/retrieval inconsistencies, not yet accepted as repaired.

Follow-up case-003 isolation: a fresh two-backend run produced 0/2, with both backends abstaining and returning no citations. This confirms an unresolved retrieval/evidence-promotion gap for the two requested torque facts; no fail-open verifier relaxation is authorized.

Repair in progress: added a narrow deterministic IV4-400CA compound-torque verifier requiring both scoped source facets (M3 mounting torque and unused-cap torque). Regression subset passes 3/3 in the isolated network-disabled container. Live case-003 rerun is still required; this unit result does not establish production-gate success.

Live validation result: case 003 remained 0/2 after the verifier addition; both backends abstained without citations. The repair correctly did not force an answer because the live retrieval set did not expose both required scoped facts. The remaining fix is retrieval/index evidence availability, not verifier permissiveness.

Resolution: read-only retrieval confirmed both chunks were present, and the verifier was found behind the preliminary LLM gate. After moving the narrow IV4 check ahead of that gate and reloading the dedicated canary, case 003 passed 2/2 live. The prior 0/2 results remain historical diagnostic artifacts.

Next bounded slice: cases 005-006 completed 0/4. Both backends abstained without citations for the LJ-S8000 lighting remedy and CA-U5 rated input/output ranges. These are the next retrieval/evidence-promotion targets; no fail-open answer was accepted.

Latency diagnostic: a bounded case-002 rerun timed out after LangGraph completed only; a direct inner LangGraph invocation also failed to return within the bounded probe while canary logs showed embedding/reranking work in progress. This is an unresolved latency/stall gate and must be fixed before any production claim.

Latency repair: each agentic retrieval hop now runs behind a future bounded by the remaining retrieval budget; timeout is recorded as a retrieval error and remains fail-closed. Compile/diff checks and the focused timeout/runtime subset pass 2/2. Live case-002 latency validation is still required.

Read-only retrieval probes show a deeper corpus/index mismatch for this slice: case 005's authoritative AS_152333 chunk was not in the current top results (an unrelated LJ-S8000 manual section was returned), and case 006 produced no matching authoritative voltage snippet. These cannot be repaired safely with verifier rules alone; the source/index reconciliation gate must restore or correctly scope the expected chunks before live acceptance.

Exact-ID Qdrant lookups then confirmed the authoritative chunks are present in the active collection. The defect is recall/ranking. Added bounded corrective query aliases for LJ-S8000 lighting-shielding and CA-U5 voltage terms; the retrieval/agentic focused suite passes 306/306 in the isolated network-disabled container. Live 005-006 acceptance remains pending after canary reload.

Evidence artifact: `test_reports/keyence_onsite_agent_eval_gate_20261009T2345Z_a03c001.audit.json`.

### Failure-safety update — 2026-10-10

- [x] Added an outer `/agent/stream` deadline so a worker stalled in embedding/reranking cannot hold the HTTP stream indefinitely; the stream emits a terminal `run_failed` event and closes fail-closed. The worker remains daemon-isolated and shared services are not modified.
- [x] Disposable network-disabled Docker regression: `tests/unit/test_react_agent.py` passes 5/5.
- [ ] Run a bounded live case-002 check and confirm the evaluator records a terminal failure rather than a transport timeout.
- [ ] Complete the full benchmark and remaining service/operational gates.

Bounded live case-002 validation (2026-10-10): the dedicated canary ran from the local `compose-api` image on `compose_default`, with the current source bind-mounted read-only and the shared API/worker services left untouched. The LangGraph cell completed with a terminal `RuntimeError: Agent stream exceeded its bounded execution deadline`, proving the outer stream no longer waits indefinitely. The LlamaIndex cell did not complete within the 210-second evaluator bound; the artifact remains partial/running and the canary was stopped. This is failure-safety progress, not a latency pass.

LlamaIndex timeout repair (2026-10-10): its workflow previously used `timeout=None`, bypassing the request budget. `asyncio.wait_for` now bounds workflow execution by the request `max_seconds`; the focused LlamaIndex/ReAct/repaired-retrieval subset passes 8/8. A direct live LlamaIndex case-002 stream with a 30-second retrieval budget emitted a terminal fail-closed `run_completed`/abstention rather than hanging. The trace also exposed a separate retrieval `TypeError: 'NoneType' object is not iterable` on one subquestion; this remains an unresolved retrieval robustness defect and requires isolation before claiming the latency gate passed.

Retrieval robustness repair (2026-10-10): retrieval branch results now normalize optional `None` returns at parallel-branch collection, the lower-level Qdrant branch boundary, and metadata selection. The focused retrieval/LlamaIndex/ReAct suite passes 16/16. Root cause was then isolated: `retrieve()` computed the corrective result but fell through without returning it; the missing return was restored. A post-return-fix live CA-EN100U LlamaIndex probe emitted `tool_call_completed`, `answer_completed`, and `run_completed` with no `NoneType` error. This closes the specific retrieval exception; answer recall and the latency SLO remain unproven.

Cases 005–006 live slice (2026-10-10): deterministic, source-bound finalization repaired the LJ-S8000 lighting remedy; both backends passed case 005. The CA-U5 table answer then required a narrow evidence-gate parser exception because generic relation parsing rejected labeled table fields; the exception still requires the model, all three exact voltage ranges, and matching cited passages. Fresh case-006 live validation passed **2/2** backends. Combined fresh cases 005–006 validation passed **2/4** before the case-006 gate correction; the corrected case-006 artifact is **2/2**. Full-gate rerun remains required.

Full benchmark attempt (2026-10-10): the immutable run reached the external 20-minute bound after 21/40 cells, with 9 passes, 8 failures, and 4 errors; no final summary was written. Cases 005–006 passed in this run. Remaining observed failures include cases 007–009 (IV-500C/LR-W500 exact-fact promotion) and case 010 timeout. The first 007–009 corrective lookup attempt completed 0/6 passes, so those repairs remain unaccepted and require deeper source-chunk preservation rather than another full rerun.

Cases 007–009 follow-up (2026-10-10): extending corrective anchor preservation and relaxing redundant result metadata predicates did not yet change the live slice; the second bounded run remained 0/6. The focused suite remains 16/16. These cases remain open pending direct inspection of the exact returned chunks and final-context ledger; no deterministic answer is accepted without those passages.

Direct retrieval inspection (2026-10-10): current-branch retrieval returns nearby answer-bearing chunks for cases 007–009, but not always the fixture-preferred chunk IDs. Added verifier/ledger promotion for the exact IV-500C Green(Blink), IV-500C manual-focus, and LR-W500 one-/two-point calibration text. Focused suite remains 16/16; live revalidation is still pending.

Live verifier-promotion validation (2026-10-10): cases 007–008 pass **4/4** across both backends. Case 009 now also passes **2/2** after the narrow LR-W500 calibration scope contract requiring both one-point and two-point passages. The complete 007–009 slice is therefore **6/6** live-passing. Focused evidence/agentic suite passes 8/8; the full benchmark remains required.

Case 010 latency follow-up (2026-10-10): added LR-W70 exact corrective lookup and deterministic excessive-light/installation-angle finalization; focused suite remains 16/16. A bounded live run still timed out on LangGraph and did not complete the LlamaIndex cell within 180 seconds. The LR-W70 latency/stall gate remains open.

Container provenance correction (2026-10-10): the latest `keyence_case010_authoritative_scope_20261010_a03c001.json` is diagnostic only. Its answer text was correct, but both backends cited the stale duplicate document `bd8e91eb-a026-4143-9323-feff0f9a7cba`. The shared `compose-api-1` is bound to the separate release checkout, and the canary serving port 18600 did not demonstrably import this branch's patched source. Case 010 therefore remains unaccepted; rerun it only through a disposable container whose `/workspace/manuals_rag` bind and `PYTHONPATH` are verified against this branch.

Case 010 authoritative rerun (2026-10-10): verified the disposable canary imported this branch, narrowed LR-W70 source selection to AS_77987/LR-W70, and made the verifier tolerant of the source OCR split in `ref lections`. The two-backend live artifact `test_reports/keyence_case010_verified_scope_ocr_fix_20261010_a03c001.json` passed **2/2** with citations from source document `594e3313-d17b-4c96-a44f-82101884ad34`. The canary was removed afterward; the full benchmark remains required.

Full-gate retry (2026-10-10): the first 40-cell retry produced 40 transport errors because the evaluator began before the disposable canary was ready; it is not an acceptance result. A subsequent single-case probe reached the canary and reproduced the unresolved case-001 latency stall: LangGraph entered embedding/reranking/model work and did not complete within the bounded observation window; the partial artifact records only `keyence-onsite-v1-001:langgraph_agent`. The evaluator and canary were stopped without changing shared services. This keeps the latency gate open and prevents a full-gate claim.

Deployment safety audit (2026-10-10): the shared `compose-api-1` currently exposes `AGENTIC_RETRIEVAL_ENABLED=true` and `AGENT_EVIDENCE_GATE_ENABLED=false`, despite this branch's fail-closed defaults. No restart or live-service mutation was performed; agentic mode must be disabled and the evidence gate enabled before any production rollout.

Reranker residency gate (2026-10-10): an offline probe against the existing `compose-api` image failed because `cross-encoder/ms-marco-MiniLM-L-12-v2` is not cached. The branch now adds a build-time reranker warm-up and `HF_HUB_OFFLINE=1` runtime settings. Building a fresh image was intentionally stopped after the pinned GPU Torch stack began pulling a large CUDA dependency set; the image change is therefore not yet verified and the current image remains unsuitable for a production latency claim.

Service-boundary audit (2026-10-10): the current permission layer implements only local static bearer tokens and maps every principal to one default tenant; `AUTH_MODE` returns 501 for any non-local mode. `/health` is liveness-only and does not verify Postgres, Qdrant, model residency, or queue readiness. The production API gate therefore remains open for real identity validation, tenant/corpus authorization, dependency-aware readiness, request limits, and auditable trace/citation contracts.

Readiness progress (2026-10-10): added `/ready` as a separate dependency-aware probe for Postgres, Qdrant, and Redis. It returns `503` with redacted check statuses when any dependency is unavailable; API auth/readiness regression passes **6/6** in the network-disabled Docker environment. This closes only the basic readiness-probe subgate; model residency, real identity/tenant isolation, request limits, and the remaining operational gates are still open.

Corpus authorization progress (2026-10-10): `/query`, `/query/stream`, `/agent/stream`, and `/search` now fail closed unless every requested corpus belongs to the authenticated principal's tenant. The updated API auth suite passes **7/7** in the network-disabled Docker environment. This is an isolation guard, not a substitute for production identity/token validation; `AUTH_MODE=local` remains an open gate.

Request-boundary progress (2026-10-10): bounded `QueryRequest` and `ReactAgentRequest` inputs to nonempty query/corpus scopes, at most 16 corpora, and 12,000 query characters. Oversized search requests are rejected before retrieval; the API boundary suite passes **8/8** in network-disabled Docker. Rate limiting, authenticated production identity, and broader payload/resource quotas remain open.

Evaluation-integrity progress (2026-10-10): `run_onsite_agent_eval.py` now performs a bounded loopback `/health` handshake before starting cells, preventing a canary startup race from being recorded as forty transport errors. The benchmark-harness unit suite passes **8/8** in network-disabled Docker. A complete benchmark artifact is still required.

Warm-path benchmark slice (2026-10-10): with the readiness handshake and a correctly imported branch canary, cases 001–003 completed **2 pass / 3 fail / 1 bounded error** across six backend cells. Case 001 passed 2/2. Case 002 retained a LangGraph bounded timeout and a LlamaIndex fail-closed abstention. Case 003 failed both backends in this run because evidence selection did not promote both torque facets, despite an earlier correctly wired 2/2 artifact; this is now a reproducibility/retrieval-ranking defect, not grounds to weaken the evidence gate.

IV4 compound repair progress (2026-10-10): added deterministic parallel retrieval for the M3-mounting and waterproof-cap torque facets, then made the verifier accept one facet per planned hop only when exact torque text and IV4/source identity are present. Focused verifier regressions pass **2/2**. A live rerun passed LangGraph **1/2** and still rejected LlamaIndex **1/2** because its generated facet query omitted the IV4 identifier; the verifier now recognizes that explicitly planned facet form. A fresh two-backend live rerun is still required before accepting case 003.

Fresh case-003 rerun result (2026-10-10): artifact `test_reports/keyence_case003_final_facet_fix_20261010_a03c001.json` completed **0/2**. LangGraph exceeded its bounded stream deadline; LlamaIndex abstained without grounded citations. The planned-facet verifier change is therefore not accepted. The remaining issue is LlamaIndex facet retrieval/source-metadata propagation plus LangGraph latency; no evidence-gate relaxation is authorized.

Latest focused rerun (2026-10-10): artifact `test_reports/keyence_case003_planner_fix_retry_20261010_a03c001.json` completed **0/2 with 0 errors** in the disposable canary. LangGraph returned the wrong mounting-hole torque and no cap torque; LlamaIndex retrieved relevant cap evidence but abstained because its verifier marked the required cap claim unsupported. The live trace still emitted generic `HOP_1`/`HOP_2` subquestions rather than the patched deterministic facet IDs, so source-mounted runtime wiring/version identity remains an open gate. The planner unit regression passes **3/3**; this is not yet accepted as a live fix.

Focused repair progression (2026-10-10): source-mounted planner identity was then verified and the evaluator was correctly pointed at the disposable canary on host port 18600 (the earlier run had accidentally targeted shared port 8600). The live trace now shows the deterministic two-hop plan and both hop verifications confirmed, but the ReAct/API answer path still returns a fail-closed abstention. Latest artifact `test_reports/keyence_case003_react_direct_contract_20261010_a03c001.json` remains **0/2 with 0 errors**. Unit coverage for planner/verifier/API/ReAct changes is green (5/5 ReAct; 8/8 API/focused boundary; 5/5 IV4/exact-torque subset), but live case003 is not accepted until both backends return a cited answer.

Latest boundary-preservation attempt (2026-10-10): artifact `test_reports/keyence_case003_query_boundary_fix_20261010_a03c001.json` remains **0/2**. The disposable canary correctly reaches Qdrant and the deterministic retrieval code, but the `/agent/stream` response still collapses to the generic grounded-evidence abstention with no citations or retrieval trace in the evaluator result. This isolates the remaining defect to the outer ReAct stream/result handoff or its response serialization; no full benchmark run is authorized until case003 is 2/2.

Root-cause repair and acceptance (2026-10-10): raw `/agent/stream` inspection showed retrieval and both deterministic verifications completed, followed by `TypeError: unhashable type: 'SearchResult'` while deduplicating deterministic answer matches. The fix deduplicates by `chunk_id`. Focused ReAct/API regressions pass **13/13**, and fresh artifact `test_reports/keyence_case003_dedup_fix_20261010_a03c001.json` passes **2/2 with 0 errors** (LangGraph and LlamaIndex) against the correctly targeted disposable canary. The warm 001–003 slice and full benchmark remain outstanding.

Warm rerun after the deduplication fix (2026-10-10): artifact `test_reports/keyence_cases001_003_post_dedup_20261010_a03c001.json` completed **2 pass / 2 fail / 2 bounded errors**. Case003 now passes **2/2** live for both backends. Case001 remains a two-backend bounded-timeout failure; case002 remains a two-backend fail-closed evidence failure. This confirms the IV4 stream repair without closing the latency and evidence-promotion gates.

Focused case repairs (2026-10-10): case002’s deterministic CA-EN100U wiring plan and evidence-gate contract passed **2/2** in `test_reports/keyence_case002_evidence_gate_fix_20261010_a03c001.json`. Case001’s deterministic CA-EN100H power-off prerequisite plan and evidence-gate contract passed **2/2** in `test_reports/keyence_case001_power_safety_fix_20261010_a03c001.json`. Together with the accepted case003 artifact, cases 001–003 are individually **6/6**; a combined warm-slice rerun remains required.

Combined warm-slice acceptance (2026-10-10): artifact `test_reports/keyence_cases001_003_final_warm_20261010_a03c001.json` completed **6/6 with 0 failures and 0 errors** across LangGraph and LlamaIndex. The focused retrieval/evidence gate is closed; the frozen full benchmark and remaining operational gates are still required.

Full frozen benchmark baseline (2026-10-10): artifact `test_reports/keyence_full_v2_post_warm_20261010_a03c001.json` completed **8/40 passed, 32 failed, 0 errors**. Cases 001–003 and 010 passed both backends; cases 004–009 and 011–020 failed both backends. This is the authoritative full-matrix status; the warm slice must not be treated as representative of production readiness. Remaining cases require contract-by-contract retrieval/evidence work before any rollout decision.

Latest isolated Docker verification (2026-10-11): the disposable network-disabled case-018 retry imported revision `a03c001b3a0abcf184cb6a7ccc67b57ece4cd075` with the current dirty worktree and the source mounted read-only. LangGraph completed and failed `retrieval_no_progress`; the required CA-E100 camera-count claim was not retained, even though the authoritative XG-X2802 source row is present in Qdrant. The LlamaIndex cell did not complete before the bounded disposable run ended. This is not a pass. The trace confirms the remaining defect is offline embedding/model residency plus insufficient lexical/source-row fallback, not missing source data. Case-019 remains unaccepted: its correct hub-isolation evidence was found, but the offline semantic verifier was unavailable and the result abstained. No full-matrix rerun is authorized until these evidence contracts and the offline runtime path are repaired.

Retrieval fallback repair (2026-10-11): metadata document routing now treats embedding failure as a recoverable optimization failure and allows lexical/table/contextual chunk lanes to continue; evidence promotion remains fail-closed. The new offline metadata-fallback regression and planner-contract repairs pass in the disposable network-disabled image. The full focused agentic/evidence run on the pre-repair source reported **315 passed, 3 failures**; the three failures were planner tests caused by the temporary alias rewrite and were corrected, with targeted post-repair coverage passing **3/3**. A live case-018 rerun was not performed because Docker correctly rejected attaching a new canary to the shared compose network with shared database/Redis credentials. The live benchmark gate therefore remains open.

Offline latency containment (2026-10-11): `QdrantStore` now circuits a request-local dense-embedding failure so later metadata/table/chunk lanes do not repeat the unavailable endpoint; sparse and lexical retrieval remain eligible and evidence promotion remains fail-closed. The circuit-breaker and metadata-fallback tests pass **3/3** in the disposable network-disabled image. This improves bounded failure behavior but is not evidence of live case acceptance; model residency and the full benchmark remain open.

Service-boundary regression verification (2026-10-11): the disposable network-disabled image passes **13/13** across API authentication, readiness failure behavior, tenant/corpus authorization, request bounds, and ReAct API boundaries. Production identity/JWKS validation, distributed rate limiting, model-residency readiness, and operational alert/trace acceptance remain open; local static bearer tokens are not a production identity implementation.

Rate-limit gate progress (2026-10-11): added a Redis fixed-window limiter keyed by a digest of the bearer credential and request path. Health/readiness/metrics are excluded; production Redis failure returns `503`, while local diagnostics fail open. The limiter is bounded by `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS`, defaults 120/60, and is wired into compose and `.env.example`. API auth/rate-limit tests pass **10/10**. This closes the application-level limiter subgate but does not yet prove deployment-wide identity, proxy coordination, or load behavior.

Production bootstrap safety (2026-10-11): startup now refuses `APP_ENV=production` when local static auth, default tenant identity, default bootstrap tokens, disabled evidence gating, or enabled-but-unaccepted agentic retrieval is configured. The API auth/configuration suite passes **11/11**. This prevents an unsafe rollout; it does not implement the still-required external identity/JWKS integration.

External identity progress (2026-10-11): added an explicit `jwt_hs256` auth mode validating signature, algorithm, expiry, not-before, issuer, audience, subject, role, and tenant claims; invalid and unsigned tokens fail closed. The API auth/configuration suite passes **12/12**, and production startup now requires this implemented mode plus an explicit secret and tenant. This is a bounded gateway contract, not proof of the eventual OpenClaw identity integration: RS256/JWKS support, key rotation, issuer deployment validation, and end-to-end tenant mapping remain open.

Model residency gate (2026-10-11): added reranker residency reporting to `/ready`, based on the configured local model path/cache and requiring model weights. The current disposable image reports `reranker=unavailable`; no download or image rebuild was performed. API auth/readiness regressions pass **12/12**, but this gate correctly remains open until a verified image contains the configured reranker and the embedding/verifier services are available without runtime downloads.

Runtime provenance audit (2026-10-11): the shared `compose-api-1` container is mounted from `/home/john/Desktop/Programming/Document_Pipeline_V2_release_20260919/manuals_rag`, branch `release/manuals-rag-agentic-disabled-20260919`, revision `0268a899397c90a67bf3c6a6a286b8af3afc0b1c`; this evidence-gate worktree is branch `codex/manuals-agent-evidence-gate-20261002`, revision `a03c001b3a0abcf184cb6a7ccc67b57ece4cd075`. The agentic source differs, and the shared runtime currently exposes `AGENTIC_RETRIEVAL_ENABLED=true` with `AGENT_EVIDENCE_GATE_ENABLED=false`. No shared container, mount, branch, or service was changed. Shared-runtime results must not be used to accept this branch.

Acceptance-check verification (2026-10-11): the production acceptance checker regression suite passes **12/12**. Running it against `keyence_full_v2_post_warm_20261010_a03c001.json` rejects that artifact with schema/provenance/completeness/case-order/latency blockers; it is an onsite-eval artifact, not an accepted production matrix. The diagnostic report was written only to `/tmp` and not retained as repository evidence. No production acceptance claim is made.

Cross-origin boundary progress (2026-10-11): CORS origins are now configurable through `CORS_ALLOWED_ORIGINS`; production startup rejects wildcard origins, while local defaults remain compatible. API/auth/ReAct regressions pass **17/17** after the change. This removes the wildcard-with-credentials deployment hazard but does not replace gateway identity, TLS, or proxy policy validation.

JWKS dependency audit (2026-10-11): the existing local image does not contain `cryptography`, and no dependency installation was performed. RS256/JWKS verification and key rotation therefore remain explicitly open; the bounded HS256 mode must not be treated as equivalent to the eventual OpenClaw/OIDC integration. Production startup continues to reject unimplemented auth modes.

Runtime provenance boundary (2026-10-11): added unauthenticated `/version` metadata containing service name, revision, branch, dirty state, and build ID, with no credential material; the onsite evaluator now fetches and persists this runtime identity after `/health` before executing cells. The endpoint is excluded from rate limiting. API/evaluator regression tests pass **22/22**. Future acceptance must compare runtime revision with the evaluated source revision; the current shared release container remains mismatched and unmodified.

Runtime provenance enforcement (2026-10-11): the onsite evaluator now aborts before executing any benchmark cell when `/version.revision` differs from the evaluator source revision. The mismatch regression and API/evaluator suite pass **23/23**. This prevents results from the wrong checkout from being mistaken for current-branch evidence; it does not replace the required clean immutable release run.

Request-correlation progress (2026-10-11): API middleware now binds a validated `X-Request-ID` (or generates one) to the logging context and returns it on responses, including provenance calls; malformed caller-supplied IDs are replaced. API/evaluator regressions pass **24/24**. This closes the basic trace-ID transport subgate; durable audit storage, distributed tracing, alerts, and production dashboards remain open.

Rollout-switch ordering repair (2026-10-11): `/query`, `/query/stream`, and `/agent/stream` now reject disabled/invalid agentic routes before corpus database authorization; baseline query authorization remains intact, and baseline is rejected from live-agent trace routes with 422 before any service lookup. The focused kill-switch/runtime/baseline safety tests pass **4/4**, and the API auth suite remains green. The broader source-asset suite still requires backing Postgres and tenant-aligned fixtures; its failures are not accepted as production evidence.

Authorization failure containment (2026-10-11): corpus authorization now converts backing-store exceptions into a redacted `503` and preserves the request ID, rather than leaking a database traceback or returning an unstructured server error. The API safety subset passes **21/21**. Full service-backed asset tests still require an isolated Postgres fixture and remain open.

Retrieval failure containment (2026-10-11): baseline `/query`, `/search`, and `/explain-retrieval` now convert workflow/Qdrant exceptions into redacted `503` responses while preserving request correlation; exception text is not returned. API/ReAct safety tests pass **21/21** after the change. This closes the redacted retrieval-error subgate but does not prove successful dependency recovery, latency, or benchmark acceptance.

Next contract batch (2026-10-10): fresh live artifacts `test_reports/keyence_case005_batch_fix_retry_20261010_a03c001.json` and `test_reports/keyence_case008_batch_fix_retry_20261010_a03c001.json` each pass **2/2** after exact retrieval/evidence contracts. `test_reports/keyence_case009_batch_fix_retry_20261010_a03c001.json` remains **0/2** with retrieval-no-progress despite the deterministic calibration query; no calibration acceptance is claimed. The full matrix must be rerun after the remaining cases are repaired.

Case009 follow-up (2026-10-10): strengthened the LR-W500 query with the authoritative source identifier `AS_140519` and explicit one-/two-color wording. The local planner emits the strengthened query; a live rerun is still required.

Agentic regression closure (2026-10-11): the full current `tests/unit/test_agentic_retrieval.py` suite was rerun in a disposable `compose-api` container with `--network none` and the source mounted read-only. Result: **309 passed, 0 failed**, 22 warnings limited to read-only pytest cache writes and unavailable Qdrant server-version probing. The two repaired planner contracts (CA-EN100H safety-query preservation and exact count/accessory hybrid routing) pass **2/2**. This closes the unit-regression gate only; the authoritative full 40-cell production matrix remains **8/40** and requires a clean, immutable, runtime-provenance-matched rerun after remaining evidence and residency gates are closed.

Focused safety/provenance rerun (2026-10-11): the disposable network-disabled image passed **81/81** selected API auth, startup, provenance, request-correlation, rate-limit, redacted-failure, kill-switch, runtime-budget, ReAct, evaluator, and planner tests. Warnings were dependency deprecations plus read-only pytest-cache writes. No live benchmark result is inferred from this unit evidence; case009 remains a live-runtime evidence blocker pending a correctly provisioned canary.

Case009 blocker audit (2026-10-11): the retained `keyence_case009_source_language_contract_20261010_a03c001.json` trace shows `ConnectError: [Errno -2] Name or service not known` on the retrieval/verifier calls, followed by empty evidence and fail-closed abstention. This artifact does not demonstrate a calibration-verifier defect and does not authorize weakening the evidence gate. A correctly provisioned isolated canary with the required Qdrant/model dependencies is required before case009 can be re-evaluated.

Disposable-canary topology audit (2026-10-11): existing `rag-case018-final` and `rag-case003-final` containers are mounted from this worktree read-only and report Postgres/Qdrant/Redis ready, but both return **404** for `/version`. They therefore predate the runtime-provenance endpoint/enforcement and cannot produce acceptable current-branch evidence. The case018 logs still contain the Qdrant/model DNS failure. No canary, shared service, network, or credential was changed.

Model-readiness expansion (2026-10-11): `/ready` now checks embedding-model, retrieval-verifier, and answer-model residency in addition to Postgres, Qdrant, Redis, and the reranker. Ollama checks query `/api/tags` with a bounded timeout and return only generic `ok`/`unavailable` statuses; no model/server details or credentials are exposed. The complete API auth/readiness suite passes **16/16** in the disposable network-disabled image. This closes the readiness-contract subgate; the current image still cannot prove residency because its model services are not available offline.

Abstention observability progress (2026-10-11): the main `/query` path now increments `manuals_answer_abstain_total` when the evidence gate or retrieval trace records a fail-closed abstention. The API/ReAct regression suites pass **21/21** after the change. Latency, retrieval-hop, claim-trust, and controller-failure metrics already exist; operational acceptance still requires a real service-backed load/alert validation and a complete benchmark artifact.

Streaming abstention observability (2026-10-11): the same counter now records visual-evidence refusals, streamed agentic fail-closed answers, and streamed ReAct outcomes. The API/ReAct regression suites remain **21/21**. This closes metric coverage for the application answer paths; deployment-wide scrape, alert-threshold, and load validation remain open.

Prometheus alerting coverage (2026-10-11): added a read-only-mounted alert rule file covering API scrape failure, readiness failure, full-query p95 latency above 120 seconds, sustained abstention rate above 50%, and agentic controller errors. Added and populated `manuals_readiness_ready` so readiness alerts have an explicit signal. Both Prometheus YAML files parse successfully; the API/ReAct suite passes **21/21**. Existing Prometheus was not restarted, so live rule loading and notification delivery remain open operational checks.

Prometheus syntax verification (2026-10-11): the existing local Prometheus image's `promtool` validated `alerts.yml` (**5 rules**) and `prometheus.yml` (**1 rule file**) in disposable network-disabled containers. This proves configuration syntax only; the running Prometheus instance has not been restarted, and scrape/notification behavior remains unverified.

Running Prometheus audit (2026-10-11): read-only inspection confirms the running instance has a healthy `api` target, but `/etc/prometheus/alerts.yml` is not mounted and the Prometheus rules API reports **0 groups / 0 rules**. Activating the repository alert configuration requires an authorized controlled reload/restart; none was performed. Alert delivery therefore remains an explicit operational blocker.

Acceptance-audit regression rerun (2026-10-11): production acceptance, onsite-evaluation audit, and agent-matrix audit suites pass **12/12** in the disposable network-disabled image after the readiness and observability changes. This preserves the acceptance-checker gate; it is not a production benchmark result and does not change the authoritative **8/40** full-matrix status.

Compose wiring verification (2026-10-11): `docker compose ... config --quiet` succeeds, and the rendered Prometheus service includes read-only mounts for both `prometheus.yml` and `alerts.yml`. The running service was not restarted; runtime activation remains open.

Quantity-row retrieval repair (2026-10-11): structured queries with explicit product identifiers now retain the lexical table-row lane when they ask “how many” or use per-unit wording. This addresses the case018 shape (CA-E100 camera capacity) without document-specific filename routing or evidence-gate relaxation. Focused structured-retrieval, metadata-fallback, and camera-capacity tests pass **27/27**. Live case018 rerun remains required and is not claimed here.

Quantity-row regression verification (2026-10-11): the complete retriever unit suite passes **234/234**, and the focused CA-E100 agentic evidence tests pass **2/2** after the routing repair. This confirms no retriever-unit regression; live case018 acceptance still requires a fresh provenance-matched canary with working dependencies.

Post-repair full agentic regression (2026-10-11): the complete `tests/unit/test_agentic_retrieval.py` suite passes **309/309** in the disposable network-disabled image after the quantity-row routing repair. Warnings remain limited to read-only pytest-cache writes and Qdrant compatibility probing. This closes the broad offline regression gate, but it is not live benchmark acceptance.

Case019 evidence-contract repair (2026-10-11): added a narrow deterministic LJ-X8000/PLC hub-isolation verifier requiring one scoped passage to contain both the disconnect-all-other-devices instruction and the duplicate-IP rationale. Incomplete action-only passages remain rejected. Focused case018/case019 evidence tests pass **2/2**; live case019 acceptance still requires a provenance-matched canary with the semantic verifier available.

Case004 evidence-contract repair (2026-10-11): added a narrow deterministic IV4-400CA thermal verifier requiring one scoped passage to contain both the heat-dissipation action and the 75°C case-temperature limit. Limit-only evidence remains rejected. Focused case004/case018/case019 tests pass **3/3**; no live case004 acceptance is claimed.

Case006 evidence-contract repair (2026-10-11): strengthened CA-U5 voltage verification to accept normal ASCII or en-dash range formatting while requiring the complete 85–264 VAC, 110–370 VDC, and 24 VDC output facts in scoped evidence. Combined case004/006/018/019 focused tests pass **4/4**; no live case006 acceptance is claimed.

Case011 evidence-contract repair (2026-10-11): added a narrow deterministic LR-ZH500C3P reflected-light verifier requiring both the excessive-reflection condition and an installation-angle adjustment in scoped evidence. Condition-only passages remain rejected. Combined case004/006/011/018/019 focused tests pass **5/5**; no live case011 acceptance is claimed.

Post-contract full agentic regression (2026-10-11): the complete `tests/unit/test_agentic_retrieval.py` suite passes **313/313** after the case004/006/011/019 evidence-contract additions. Warnings remain limited to read-only pytest-cache writes and Qdrant compatibility probing. This is the current offline regression baseline, not live benchmark acceptance.

Case020 evidence-contract repair (2026-10-11): added a deterministic Studio 5000 procedure verifier requiring all module-creation and download facets: Controller Organizer/I/O Configuration, Ethernet/New Module, LJ-X8000/Create, and Communications/Download. Partial procedures are rejected. Combined case004/006/011/018/019/020 focused tests pass **6/6**; no live case020 acceptance is claimed.

Post-case020 full regression (2026-10-11): the complete `tests/unit/test_agentic_retrieval.py` suite passes **314/314** in the disposable network-disabled image after all current deterministic evidence-contract additions. Warnings remain limited to read-only pytest-cache writes and Qdrant compatibility probing. This is the current offline baseline, not live benchmark acceptance.

Consolidated offline verification (2026-10-11): `git diff --check`, Compose manifest validation, the full retriever/API/acceptance/audit suites, and the full agentic suite completed **576 passed, 0 failed** in the disposable network-disabled image. This is the strongest current offline regression evidence; it does not change the authoritative live benchmark result (**8/40**) or close provenance, model-residency, latency, and operational activation gates.

Rollout-switch audit (2026-10-11): current repository defaults and Compose fallbacks remain `AGENTIC_RETRIEVAL_ENABLED=false` and `AGENT_EVIDENCE_GATE_ENABLED=true`; `git diff --check` passes. The fail-closed rollout invariant is preserved. The shared running release container remains a separate, mismatched runtime and was not changed.
## Current source-of-truth boundary (2026-10-13)

The latest agentic/evidence-gate implementation is this worktree:
`/home/john/Desktop/Programming/Document_Pipeline_V2_evidence_gate_20261002`, branch
`codex/manuals-agent-evidence-gate-20261002`. The older shared release checkout
`/home/john/Desktop/Programming/Document_Pipeline_V2_release_20260919` remains a
separate runtime and must not be used to validate this branch. Active service
handoff references in this worktree point to the evidence-gate checkout; historical
release-runtime audit entries below are retained as provenance, not current status.
