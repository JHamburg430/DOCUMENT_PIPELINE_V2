# Agentic Retrieval Production-Readiness Sprint — 2026-09-26

## Objective

Reach a defensible go/no-go decision by **18:00 America/Detroit** for an internal,
reversible Agent-tab production canary. The release remains fail-closed unless every
gate below passes on the same committed revision.

John has explicitly replaced human adjudication for this sprint with independent
secondary-agent review. Source verification is still mandatory: every accepted bank
case must remain bound to persisted source evidence, and a second agent must review
the final artifacts, failures, enabled-mode proof, and rollback evidence.

## Scope decisions

- Evaluate the existing 200-case frozen v112 bank with both LangGraph and LlamaIndex.
- Require zero failed terminal cells, exact 200/200 coverage, immutable provenance,
  a clean release revision, and the existing 120-second p95 latency ceiling.
- Keep the unaccepted v6 metadata extraction/backfill out of this release. The canary
  uses the currently reconciled PostgreSQL/Qdrant corpus; no corpus-wide metadata
  persistence or embedding refresh is authorized by this sprint.
- Align Agent-tab and evaluation defaults at six retrieval hops.
- Enable only an internal reversible canary after acceptance, not an unrestricted
  corpus or metadata rollout.

## Timeline and gates

### 11:00–12:00 — Repair and focused replay

1. Correct acceptance-bank disjointness calculation so diagnostic metadata hits do
   not masquerade as authoritative source references.
2. Trace the seven v112 failures to the earliest proven boundary.
3. Add focused regressions and replay every failed case under fresh immutable run IDs.
4. Gate: all seven cases pass both backends; focused unit tests pass.

### 12:00–12:30 — Representative failure-class slice

1. Run the seven repaired cases together plus adjacent exact-lookup and control cases.
2. Reconcile dataset-qualified case IDs, terminal cells, citations, latency, and
   persisted provenance.
3. Gate: zero failures and no malformed/unchecked verifier result.

### 12:30–16:30 — Full frozen matrix and deterministic stress gate

1. Commit the repair revision before launch.
2. Run the complete frozen 200-case matrix with six hops and both agent backends.
3. Run the 5,000-case claim/relation guard on the same revision.
4. Run the production acceptance checker against the frozen tuning-document set.
5. Gate: exact 200/200 coverage; 200/200 full pass per backend; category, citation,
   abstention, latency, provenance, disjointness, and stress checks all pass.

### 16:30–17:15 — Independent review

1. Give the secondary agent only the committed source diff, immutable launch/final/
   exit artifacts, acceptance report, stress report, and relevant source chunks.
2. Require an explicit approve/reject decision with artifact hashes and blockers.
3. Gate: secondary agent approves; unresolved findings fail closed.

### 17:15–17:45 — Enabled-mode and rollback proof

1. Start an isolated enabled-mode stack on the accepted revision.
2. Exercise the real Agent tab through UI → local job bridge → `/query/stream` for a
   supported question, an abstention case, and a repaired structured lookup.
3. Capture correlated browser/API evidence, citations, completion, and bounded timing.
4. Inject/verify disabled-mode rejection and immediate baseline availability.
5. Gate: enabled flow, fail-closed behavior, and rollback all pass.

### 17:45–18:00 — Release decision

- **GO:** commit/push the accepted revision and artifacts, enable only the internal
  canary cohort, verify live health, and retain the baseline rollback switch.
- **NO-GO:** leave `AGENTIC_RETRIEVAL_ENABLED=false`, publish exact failed gate and
  artifact evidence, and do not relabel partial results as production-ready.

## Non-negotiable acceptance contract

- No human review requirement for this sprint; independent secondary-agent review is
  required.
- No edits to a frozen bank in place.
- No acceptance from smoke tests, partial files, dirty full-run provenance, or summary
  totals that do not reconcile to individual records.
- No metadata v6 persistence, full-corpus re-embedding, or unrestricted rollout.
- Any missing, malformed, or ambiguous verifier output remains fail-closed.
