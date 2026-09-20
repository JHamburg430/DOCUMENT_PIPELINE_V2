# Independent audit: `agent_matrix_production_clean_20260919T131138`

Audit date: 2026-09-19

## Decision

The run is **structurally valid and complete for diagnostic use**, but it is
**not acceptable as production-enablement evidence**.

Production must remain fail-closed with `AGENTIC_RETRIEVAL_ENABLED=false`.

The structural checks passed, but the artifact does not persist enough
provenance or retrieval-stage evidence to perform the full independent review
required by the retrieval evaluation procedure. Its performance measurements
also predate the GPU-routing repair and are not representative of the repaired
topology.

## Artifacts reviewed

- Final: `agent_matrix_production_clean_20260919T131138.json`
- Partial checkpoint: `agent_matrix_production_clean_20260919T131138.partial.json`
- Process log: `agent_matrix_production_clean_20260919T131138.log`
- Exit status: `agent_matrix_production_clean_20260919T131138.exit`
- Dataset: `tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl`
- Dataset SHA-256:
  `51028f6b5ea6b09515ac361d2b747feb998aaa5098dca710c7e6f4a972871a65`

The earlier duplicate-writer artifact under
`quarantine_duplicate_writers_20260919T124715/` was excluded.

## Structural integrity

- Evaluator process is no longer running.
- Explicit exit status is `0`.
- Final artifact exists and parses as JSON.
- Dataset SHA-256 matches the artifact and frozen manifest value.
- Requested slice is `offset=0`, `limit=48`.
- Final artifact contains exactly 48 unique case IDs in exact dataset order.
- No missing, duplicate, or extra case IDs were found.
- Every case contains one baseline, one LangGraph, and one LlamaIndex record.
- The log contains exactly 48 ordered case-start events, 48 ordered
  case-complete events, and one answer-start event for each of the 96
  case/backend pairs.
- The partial checkpoint reached `48/48`; its item array is identical to the
  final artifact's item array. Its `complete=false` marker correctly identifies
  it as a checkpoint rather than the final result.
- All top-level summaries and all per-category summaries recompute exactly from
  the 48 individual records.

## Recomputed results

| Metric | Baseline | LangGraph | LlamaIndex |
|---|---:|---:|---:|
| Retrieval passed | 45/48 (93.75%) | 44/48 (91.67%) | 42/48 (87.50%) |
| Agent sufficient | 37/48 (77.08%) | 30/48 (62.50%) | 28/48 (58.33%) |
| Full matrix pass | n/a | 10/48 (20.83%) | 7/48 (14.58%) |
| Candidate evidence retained | n/a | 29/48 (60.42%) | 27/48 (56.25%) |
| Expected documents retained | n/a | 46/48 (95.83%) | 45/48 (93.75%) |
| Evidence sufficiency passed | n/a | 15/48 (31.25%) | 12/48 (25.00%) |
| Grounded answer passed | n/a | 13/48 (27.08%) | 13/48 (27.08%) |
| Latency/token-cost passed | n/a | 16/48 (33.33%) | 14/48 (29.17%) |

The high document-retention rate does not establish answer-bearing evidence
retention. LangGraph had 18 cases where the expected document remained but the
required candidate evidence failed; LlamaIndex also had 18. LangGraph had two
expected-document retention failures and LlamaIndex had three.

Representative persisted failures:

- Candidate-evidence loss with document retained:
  `349bdece-a37e-58b0-8cf5-d2f37d527982::curated2` (both backends).
- Evidence/synthesis failure after candidate evidence was retained:
  `a7ed6202-8a02-566c-a845-55acfe8c07bb::curated1` (LangGraph).
- Cross-document retention failure:
  `curated_cross_document_v2::1` (LangGraph) and
  `curated_cross_document_v2::2` (LlamaIndex).
- Functional cells passed but performance alone failed:
  `lj-x8000-rs232c-dependent-cable-lookup` (both backends). This performance
  result is nonrepresentative because the run predates the routing repair.

No cross-document, dependent-multi-hop, entity-resolution,
exact-structured-lookup, or unanswerable category achieved a full LangGraph or
LlamaIndex matrix pass. Full passes were confined to single-hop control and
parallel multi-part cases.

## Verification and provenance gaps

The final artifact omits all of the following:

- immutable run ID;
- source revision;
- backend set and exact inference endpoints;
- effective runtime configuration;
- full retrieved result records and answer-bearing evidence text;
- dense, fusion, reranking, and final-context stage snapshots;
- raw model-judge payloads.

The normalized verifier records are internally typed: all 114 LangGraph and
108 LlamaIndex verification objects contain a boolean `claim_supported` and a
known trust state. However, without raw judge payloads, malformed/missing
verdict handling cannot be audited independently. Without result text and
stage snapshots, claimed false negatives, equivalent evidence, reranker loss,
and dense/fusion retention cannot be independently confirmed.

The release worktree currently points to `0af529a`, the deployed disabled
release evidence names `05b8ca0`, and `origin/main` contains merge revision
`6283eeb`; because the matrix artifact embeds no source revision, none of those
revisions can be proven as the exact evaluator source solely from the artifact.

## Required next gate

Fix evaluation observability before another acceptance run:

1. Persist an immutable launch record containing run ID, source revision,
   dataset hash/slice, corpus IDs, backend set, endpoints, model names, and
   effective configuration.
2. Persist per-stage ranked result records (IDs, document IDs, ranks, and
   bounded evidence text) for dense retrieval, fusion, reranking, and final
   context, plus raw and normalized judge outcomes.
3. Rerun the unchanged 48-case bank on the repaired GPU topology with one
   writer and a fresh run ID.
4. Reapply exact coverage/reconciliation checks, then classify anchor/scoring
   mismatches, reranker losses, genuine retrieval misses, and answer/citation
   failures from inspectable evidence.

Only after evaluation correctness is independently auditable should work move
to the held-out bank (at least 2,995 human-adjudicated cases), corpus metadata
rollout, enabled UI, canary, latency, recovery, and rollback gates.
