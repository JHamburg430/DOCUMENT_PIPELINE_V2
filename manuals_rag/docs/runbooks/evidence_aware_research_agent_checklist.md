# Evidence-aware research agent delivery checklist

## Objective

Replace the current bounded tool-caller behavior with a bounded research
controller that can interpret a request, maintain a required-claim ledger,
retrieve repeatedly, evaluate evidence, change retrieval strategy, and finish
only when the evidence contract is satisfied or an explicit terminal condition
is reached.

This checklist does not promise answers to every possible question. The target
is reliable handling of questions that are answerable from the connected corpus
and tools, with honest clarification or abstention otherwise.

The companion contract matrix is
`tests/fixtures/research_agent_validation_matrix_v1.jsonl`. Its manifest freezes
the exact bytes and category counts. Contract cases are deterministic controller
tests; live cases must additionally run against persisted source evidence.

## A. Request understanding

- [ ] A1. Resolve known products, models, aliases, menu labels, and document
  entities before applying ordinary-language intent heuristics.
- [ ] A2. Treat `VS` as a product/entity when corpus metadata or session context
  supports that interpretation; classify `vs` as comparison only when two valid
  operands are present.
- [ ] A3. Preserve the user's authoritative wording and qualifiers: product,
  model, version, operation, object, coordinate, units, warnings, and requested
  output.
- [ ] A4. Distinguish action/object pairs such as create versus view, configure
  versus inspect, enable versus verify, and install versus operate.
- [ ] A5. Retain multiple ranked interpretations when ambiguity remains and ask
  for clarification only when retrieval cannot safely resolve them.
- [ ] A6. Resolve follow-up references and corrections from bounded session
  context without treating conversation or memory as factual evidence.

## B. Claim and evidence planning

- [ ] B1. Convert the request into explicit primary claims before retrieval.
- [ ] B2. Mark every primary claim required unless the product contract defines
  it as optional; an empty required-claim set cannot complete.
- [ ] B3. Keep controller-created recovery work distinct from primary claims.
- [ ] B4. Record per-claim state: unresolved, supported, contradicted,
  ambiguous, or unsupported.
- [ ] B5. Require action/object alignment. Procedural-looking text for viewing a
  screen must not satisfy a claim about creating a screen.
- [ ] B6. For multi-part or dependent requests, preserve dependencies and never
  let one successful subanswer hide unresolved claims.

## C. RAG tool contract

- [ ] C1. `manuals_rag` is always the first factual tool and remains the
  authoritative fact source.
- [ ] C2. Every RAG observation returns the interpretation used, query strategy,
  retrieved section paths/headings, bounded snippets, citations, confidence,
  and sufficiency.
- [ ] C3. Every insufficient observation returns satisfied and missing claims,
  rejected evidence with reasons, contradictions, discovered terminology, and
  suggested follow-up searches.
- [ ] C4. Preserve per-hop retrieval snapshots and source identifiers needed to
  diagnose dense, fusion, rerank, final-context, verifier, and answer stages.
- [ ] C5. A malformed verifier result remains fail-closed and is classified as a
  verifier-contract failure when evidence is present.

## D. Recovery and query variation

- [ ] D1. Insufficient evidence triggers a different retrieval strategy rather
  than immediate finalization.
- [ ] D2. The bounded query portfolio includes original, entity-expanded,
  operation/object, evidence-terminology, and neighbor-section formulations.
- [ ] D3. Independent query variants may run concurrently within configured
  limits; dependent variants wait for discovered evidence.
- [ ] D4. Promising headings and terminology from one call influence the next
  query and are recorded in the trace.
- [ ] D5. A duplicate proposed query is rejected, disclosed to the planner, and
  replaced by a distinct strategy; it is not a terminal condition.
- [ ] D6. If the planner repeats after correction, the controller generates a
  deterministic reformulation from missing claims and discovered terms.
- [ ] D7. Contradictory evidence triggers authoritative-scope/version recovery,
  not arbitrary selection of one result.
- [ ] D8. Tool errors may be retried within budget without erasing prior
  evidence or duplicating completed work.

## E. Evidence ledger and finalization

- [ ] E1. Merge evidence by claim across calls, deduplicating citations by
  document/chunk identity.
- [ ] E2. Preserve provenance for every supported claim and contradiction.
- [ ] E3. Final answers use only supported claims and cite the exact evidence
  used for each factual part.
- [ ] E4. Do not expose citations from unused, rejected, out-of-scope, or
  contradictory evidence.
- [ ] E5. Calculator use is allowed only after RAG supplies the source numbers;
  calculation is transformation, not independent evidence.
- [ ] E6. Session memory may store durable user context and resolved entities,
  but never retrieved technical facts, secrets, or speculative conclusions.
- [ ] E7. Memory-disabled sessions do not retain context after the turn.

## F. Termination and user experience

- [ ] F1. Finish successfully only when every required primary claim has valid
  grounded evidence.
- [ ] F2. Ask a targeted clarification when unresolved ambiguity materially
  changes the answer and the corpus cannot resolve it.
- [ ] F3. Abstain only after the configured distinct recovery strategies are
  exhausted or the corpus conclusively lacks support.
- [ ] F4. On budget exhaustion, report the exact remaining claims and attempted
  strategies; do not imply research completed successfully.
- [ ] F5. Emit a trace showing interpretations, claim state transitions, tool
  calls, strategy tags, duplicate recovery, evidence decisions, and stop reason.
- [ ] F6. UI session controls, memory controls, new-session behavior, traces,
  citations, empty/error states, reload, and phone-width rendering all match the
  controller state.

## G. Validation gates

- [ ] G1. Validate the matrix schema, unique IDs, category counts, required
  fields, and immutable SHA-256 manifest.
- [ ] G2. Run all deterministic controller-contract cases on both selectable
  backends where backend behavior is in scope.
- [ ] G3. Require exact expected stop reason, tool ordering, distinct query
  count, required strategy tags, claim coverage, citation behavior, and memory
  behavior for every case.
- [ ] G4. Run live source-backed cases from a committed clean revision with
  immutable artifacts containing source/configuration provenance.
- [ ] G5. Make the VS custom-screen conversation a mandatory live regression:
  the controller must reject viewer-only evidence, use discovered authoring
  terminology, and retrieve the Adding Parts/Properties procedure.
- [ ] G6. Add adversarial live cases for acronym/ordinary-word collisions,
  action/object confusion, follow-up corrections, contradictory revisions,
  multi-part questions, poor first retrieval, and truly unsupported requests.
- [ ] G7. Recompute acceptance from individual records. Passing answer text
  alone is insufficient when the trace violates tool, claim, citation, retry,
  memory, or termination contracts.
- [ ] G8. A production candidate requires 100% deterministic contract coverage,
  100% mandatory live-regression coverage, exact backend coverage, exit zero,
  structural audit, and independent acceptance review.

## Required acceptance metrics

| Metric | Gate |
| --- | --- |
| Matrix structure | 100% valid, exact manifest hash |
| First factual tool | `manuals_rag` in every factual case |
| Required claim coverage | 100% before successful finish |
| Duplicate-query behavior | 0 premature terminations |
| Required query strategies | 100% observed where specified |
| Citation validity | 100% used, in-scope, claim-bound citations |
| Unsupported/ambiguous behavior | 100% correct clarification or abstention |
| Memory boundary | 100% context-only, no technical-fact persistence |
| Mandatory live regressions | 100% on LangGraph and LlamaIndex |
| Artifact integrity | committed clean revision, immutable final, exit 0 |

## Execution order

1. Validate the frozen matrix and manifest.
2. Implement or repair the earliest failed controller boundary.
3. Run deterministic cases and replay exact failed records.
4. Run the mandatory live-regression slice on both backends.
5. Commit the repair, rerun the focused slice from the clean revision, then run
   the full matrix with immutable launch/final/exit artifacts.
6. Audit exact coverage and obtain independent acceptance review.

## Matrix integrity commands

Run the schema and immutable-manifest validator before using the bank:

```bash
python scripts/benchmark/validate_research_agent_matrix.py
```

Run its focused enforcement tests in the project container:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api \
  python -m pytest -q tests/unit/test_research_agent_validation_matrix.py
```

Run the deterministic controller-contract runner and inspect its immutable
report:

```bash
python scripts/benchmark/run_research_agent_matrix.py \
  --output test_reports/research_agent_matrix_latest.json
```

The runner executes the real bounded ReAct loop with deterministic planner/RAG
observations. Live-corpus rows are explicitly `not_run` until both production
backends are exercised against the committed corpus; they are never counted as
passes. The UI exposes this report under **Evidence-Aware Research-Agent
Matrix**, separately from the older retrieval matrix.
