# Manuals RAG six-recommendation review — 2026-09-20

## Outcome

The six recommendations have been converted into measured decisions. This is a
completed technical review, **not** a production-enablement decision. Agentic
retrieval remains disabled and the existing Ollama + MiniLM path remains the
rollback/default configuration.

1. **Acceptance evidence — diagnostic only.** The clean 48-case artifact has
   exit status 0, exact 48/48 ordered coverage, immutable source/dataset
   provenance, and independently reconciled totals. Both backends pass 30/48.
   Production remains blocked by a separate frozen held-out bank, source-backed
   failed-answer adjudication, and human visual-PDF review.
2. **Corpus metadata — safe mechanism proven, rollout incomplete.** The v5 exact
   audited-report workflow completed backup, apply, PostgreSQL/Qdrant
   reconciliation, and idempotent re-entry for VJ-3302. The current corpus audit
   is 1 pass / 52 fail, so broad production claims are prohibited.
3. **Indexed BM25 — keep opt-in.** Against the same 48 cases, native Qdrant BM25
   improved required-any@40 from 59.6% to 78.7%, required-all@40 from 53.2% to
   66.0%, and mean latency from 57.0 ms to 27.0 ms. Required-all@5 regressed from
   38.3% to 23.4%, and no downstream answer gate passed; default remains off.
4. **Reranking — keep MiniLM.** A defect in the initial benchmark was found and
   fixed: it had ignored `evidence_text` and scored empty documents. Corrected
   text-hashed candidate pools show Qwen3-Reranker-0.6B at pool 12 improves
   required-any@5 by 2.2 points but is about 6x slower, with identical
   required-all@12 and loss count. MiniLM remains selected.
5. **Selective planning — contracts pass, rollout stays off.** All 96 backend
   planning-contract rows pass identifier/claim/selectivity checks. Dependent
   and structured classes benefit, but cross-document full passes remain 0/10
   and latency is substantially higher, so agentic retrieval remains disabled.
6. **Local inference — no migration.** The clean matrix records end-to-end and
   model duration/token distributions, but lacks per-stage, queue, route,
   residency, GPU, and verifier-attempt telemetry. It cannot justify vLLM or
   another serving migration, and serving changes are not treated as accuracy
   fixes.

## Verification and artifacts

- Focused changed-boundary suite: **549 passed, 2 deselected**, with the two
  deselections limited to tests whose temporary 10 MB PDF fixture is absent from
  this release worktree. The same fixture exists in the original checkout; this
  is not a product-code failure.
- Broad unit execution: **1,138 passed**. The remaining 18 container failures
  were environment-only: 15 needed a `git` executable absent from the runtime
  image, two needed that temporary PDF, and one expected populated temporary
  evaluation directories. The 15 manifest tests were rerun on the host with
  `git` available and passed **82/82**.
- Non-live integration gate: **1 passed, 4 live tests deselected**.
- `git diff --check`, compose configuration validation, Python compileall, the
  VJ-3302 PostgreSQL/Qdrant reconciliation assertions, and corrected reranker
  candidate-pool hash reconciliation all passed.
- Matrix audit:
  `agent_matrix_full48_b20f9a3_20260920_1200_independent_recheck_20260920.md`.
- Metadata audit:
  `manuals_rag_full_corpus_metadata_audit_v5_20260920.json`.
- BM25 comparison: `sparse_retriever_comparison_48_20260920.json`.
- Corrected rerankers:
  `reranker_minilm_48_pools_evidence_v2_20260920.json` and
  `reranker_qwen06_48_pools_evidence_v2_20260920.json`.
- Query instruction comparison:
  `dense_query_instruction_comparison_48_20260920.json`.
- Planning audit: `planning_selectivity_audit_48_v2_20260920.json`.
- Reconciled planning recheck:
  `planning_selectivity_audit_48_recheck_20260920.json` (48/48 per backend).
- Performance audit: `local_inference_performance_audit_20260920.json`.

## Open production gates

- Freeze and pass an independent source-backed held-out bank.
- Complete source-backed and human visual-PDF adjudication.
- Roll v5 metadata through the remaining 52 active documents with per-wave
  literal-source audit, targeted backup, exact-report apply, vector refresh,
  PostgreSQL/Qdrant reconciliation, and idempotent re-entry.
- Add the missing routing/residency/queue/per-stage telemetry before performance
  acceptance or any serving migration.
- Rerun downstream answer gates before enabling BM25 or query instructions.
