# Retrieval improvement: active team coordination

This is the execution contract for John's current retrieval-improvement task,
not evidence that the product is ready. Continue the acceptance checklist in
`retrieval_improvement_checklist.md`; preserve its frozen baseline.

## Ownership

| Owner | Exclusive implementation scope | Required deliverable |
|---|---|---|
| metadata_quality | Parser metadata, backfill, persisted audit, associated tests | Source-audited current sample and propagation evidence |
| retrieval_quality | Answering, retrieval, evaluation, associated tests | Adjudicated failures, general fixes, controlled answer-quality measurements |
| Parent coordinator | Common clients, UI, integration, GPU scheduling, release | Integrated checks, live comparison, operational evidence, honest release decision |

Agents must request coordination before crossing ownership boundaries. Existing
dirty changes belong to the ongoing task or other work: inspect, preserve, and
stage only reviewed task files. Parent performs commits and pushes.

## Parent instructions for this task

1. On every resume, read this contract, the main checklist, latest checkpoint,
   and both `test_reports/retrieval_improvement/team_*.md` reports. Check actual
   processes/artifacts before resuming a job or reporting status.
2. Give each agent a bounded next experiment with a falsifiable result. Require
   milestone messages and immediate failure/blocker messages, not activity lists.
   Each report records evidence path, changed files, test command/result, current
   failure, and next executable action.
3. Use completion messages to review and immediately assign the next unmet gate.
   Keep the progress card current. During active coordination, communicate
   meaningful progress at least once a minute; do not repeatedly poll agents.
4. Parent grants one live GPU experiment lease at a time. Inspect capacity and
   active requests first. Parser tests use CPU-only test-process settings when
   appropriate; app inference must not silently fall back to CPU. Never stop
   unrelated models, voice, or other sessions without authorization.
5. Live runs need a bounded timeout, incremental case/document checkpoints,
   explicit exit status, model/context identification, and preserved logs. On
   missing progress inspect requests and output before retrying. After two
   failures of the same hypothesis, change the experiment, not just the timeout.
6. If GPU work is unavailable, move immediately to source adjudication, offline
   reproduction, parser/retrieval regressions, UI verification, or recovery tests.
   If a child is stalled, inspect its last evidence and narrow/reassign work.
   Never bypass a quality gate to avoid admitting a genuine external blocker.
7. Review patches against source-backed counterexamples; preserve negative tests,
   unknown applicability, and abstention safeguards. No benchmark-specific runtime
   answers, weakened scoring, or relabeling abstentions as correct answers.
8. Integrate in small batches: focused tests, source audit, sample-only persistence
   after audit, PostgreSQL/chunk/Qdrant consistency, controlled live diagnostics,
   then frozen full matrix and untouched held-out set. Do not repeatedly run a
   costly full matrix while a smaller known failure remains unresolved.
9. Verify the running UI for every changed user-facing flow. Measure correct
   complete answers, incorrect answers, and abstentions separately; show coverage,
   latency, failures, dataset size, and audit limitations.
10. Commit/push reviewed delivery increments and update recovery checkpoints.
    Production stays off while acceptance gates remain unmet. Never equate unit
    tests, synthetic fixtures, or extraction exit zero with product accuracy.

## Completion and continuity

Completion requires every applicable checklist gate to have inspectable evidence,
no unresolved incorrect supported answers in the acceptance set, a completed
both-backend comparison, independent held-out evaluation, sample synchronization,
running-UI and recovery verification, and a documented rollback/release decision.
This does not prove universal 100% correctness.

While children run, parent handles independent integration work. If this turn
must yield, use the native child completion handoff so their results resume the
parent; do not leave an unmonitored shell job or ask John to say "continue".
Record genuine resource/credential blockers and the remaining independent work;
never promise that external dependencies cannot block progress.
