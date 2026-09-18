# Retrieval diagnostic — two-agent execution

## Scope and method

Assistant audit of frozen `live_validation_20260914/agent_matrix.json` and exported
`oracle_candidates.json` extracted source text. This is **not human/PDF visual
adjudication or held-out evaluation**. Frozen dataset/results are unchanged.
No GPU inference, corpus writes, production activation, or benchmark relabelling
was performed during the offline phase.

## Findings and repairs

1. Dependency refinement discarded the pending detailed query whenever an anchor
   was found, substituting a short objective. Hybrid execution repeated that
   substitution, even after a refiner supplied a precise query. Thus a query for
   calibration tolerance **at 25 degrees C** became merely “Find specifications”.
   Preserve the original pending query and append grounded identifiers in both
   paths. Regression verifies the condition survives actual graph execution.
2. Dependency evidence imported every retrieved candidate, including an original
   failed candidate set after successful recovery. An unrelated part identifier
   could become an exact constraint for the next hop. Only sufficient hops'
   independently selected `supporting_chunk_ids` now feed dependency refinement;
   rejected and distractor candidates remain in the trace, not binding evidence.

Focused verification: **69 tests passed**, `tests/unit/test_agentic_retrieval.py`,
9.96s. Three new regressions. This proves implementation behavior, not live answer
accuracy. No commits/staging performed by this agent.

## Frozen category audit

- Six generated `warning_step_multi_step` questions are malformed/truncated and
  their two labelled chunks are a caution heading plus an installation passage,
  not a discovery→dependent-lookup chain. One asks about an illustration absent
  from text. Their frozen failures must remain reported; do not treat passing
  these as evidence of real dependency ability.
- The seventh dependent case asks cable **connector orientation**, while the
  labelled source says `Serial connection cable (2.5 m, straight)`. It supports
  model OP-26487 and the literal “straight” description, but does not establish
  what straight describes. Do not force a spatial interpretation to match terms.
- Cross-document case 3 is useful for verification diagnosis: both expected chunks
  and documents survived baseline retrieval, yet both requirements failed
  verification. Source IV4-CU1 enclosure entry is `IP67 *1` (p451) and shock entry
  is `500 m/s², 6 different directions, 3 times for each` (p489). Footnote*1 is not
  in candidate text; it cannot establish all conditions for an unconditional IP67
  claim. Use a labelled-entry query or obtain surrounding source before gold use.
- Cross-document case 6 supplies complete textual setting definitions and is the
  preferred initial live oracle. CV-X p459, chunk
  `7361e08a-84bb-5f53-954c-819d2c179a3e`: Condition list permits up to16 reference
  conditions, combined read-data ranges/sorting; turning off Use in Tool's
  Judgment permits ignoring mismatch for sort use. LJ-X8000 p278, chunk
  `125b3908-de90-53b8-809e-b367bcc83548`: Standard Angle sets blob numbering start
  angle when label order is clockwise/counterclockwise. Correct comparison must
  preserve these product-setting bindings and conditional behavior.
- Cross-document cases1/4 reference IV-HG500CA while candidate source filename
  names IV-H500CA/IV-H500MA/IV-H2000MA. This is a potential scope/label issue;
  verify manual applicability before treating it as trusted model-specific gold.
- Cases8/9 ask corrective action for unsupported SD card, but supplied cell is a
  compatibility/data-loss warning, not an explicit replacement instruction. An
  answer must report that guidance without inventing a replacement action.
- Other cross-document candidates contain plausible table/setting facts, but
  no blanket clean-gold designation is made without page/applicability audit.

## Next live experiment (GPU lease required)

1. Supply the two case6 source chunks directly to synthesis and separate claim
   verifiers; retain exact answer, citations, warnings, model settings, runtime,
   explicit exception/exit status. No retrieval/index dependency.
2. Classify correct complete / incorrect / abstention by source; compare returned
   evidence versus independent verifier disposition. Partial answer is not pass.
3. Then run retrieval with the same two explicit setting/product scopes, followed
   by normal planner mode. This separates answer/verifier, retrieval and planning.
4. A true clean dependent experiment must use a newly source-audited discovery
   fact and follow-up property, not reinterpret the malformed frozen cases.

Parent owns GPU scheduling, full-suite integration, held-out set, UI/release and
final checklist state. Original all-gate14/48 and12/48 baseline is not superseded
by these offline checks.

Adjacent regression verification: **191 passed** in7.92s across
`test_agent_eval.py` and `test_retriever.py`; combined with69 agentic tests,
**260 focused/adjacent tests passed**. These runs did not allocate live models.
Task-specific runnable oracle: `team_oracle_probe.py`, writes
`team_oracle_probe.json` after each stage. It has not yet been run (GPU lease
reserved by the extraction pilot). Coordinator should use an external timeout.

## Controlled live result — 2026-09-17

The oracle and planner probe is complete. Direct-source CV-X and LJ-X evidence
was retained after scope/deduplication repairs. Scoped live retrieval confirmed
the CV-X482 `Condition list` definition but found two legitimate LJ-X8000
definitions for the unqualified `Standard Angle` label. That frozen question is
ambiguous; the exact-label conflict gate makes both backends abstain instead of
selecting a definition by rank.

A separate source-audited dependency uses the CA-EN100U table: hop one discovers
`CA-EN100H`; hop two uses that identifier to retrieve `Power-supply: Supply from
CA-EN100U`. Both backends execute the dependency, retain the two required claims,
and return the complete grounded answer. LangGraph completed in127.15s and
LlamaIndex in119.13s. Artifact: `team_retrieval_planner_probe.json`. Full affected
suites:501 passed. These are extracted-text/assistant results, not human/PDF
adjudication or held-out accuracy. Production remains OFF; next is the unchanged
frozen48-case matrix.

## Frozen matrix diagnostic — 2026-09-17/18

All 48 frozen IDs completed across three checkpointed segments with unchanged
dataset SHA; the final segment exited 0. Raw passes were 12/48 LangGraph and
10/48 LlamaIndex. Do not use those rates as acceptance: 43 verifier calls timed
out after GPU discovery failed and the 9B verifier loaded on CPU. Timeouts fail
closed before later branches or dependent hops can execute, contaminating many
candidate/document scores. The cable dependency itself now plans correctly in
both backends (dependent, two hops, one edge), but hop-1 verification timed out.
The separate CA-EN100U dependency remains the clean live proof.

Verifier inference now receives `num_batch=64`, matching metadata stabilization.
Affected tests: 501 passed, 37 warnings; full unit suite: 1044 passed, 77
warnings. T8 stays open until healthy-GPU affected cases and then the full matrix
are rerun. See
`agent_matrix_controlled_20260917_audit.md`. Production remains OFF.
