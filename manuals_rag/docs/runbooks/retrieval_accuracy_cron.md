# Retrieval Accuracy Cron Runbook

This runbook is the standing brief for the recurring retrieval and answer accuracy agent job.

## Mission

Advance Manuals RAG until it can answer every reasonable engineering question that can be answered from the indexed documentation.

This is a production-system goal, not a leaderboard goal. The system must retrieve and generate grounded answers for the kinds of questions an engineer, salesperson, manager, technician, or support person would ask about the documents Manuals RAG is designed to serve.

The job must improve three capabilities:

1. Optimal and efficient single-step retrieval for direct questions whose answer is in one chunk, row, table cell, warning, step, or short section.
2. Multi-step retrieval for questions that require gathering evidence from multiple chunks, sections, tables, pages, or documents before answering.
3. Final answer generation that accurately uses the retrieved evidence, cites/grounds claims, and says when the indexed documents do not contain enough information.

## Application Quality Standard

John expects this automation to analyze the application at the same level of detail he applies during live review. Do not stop at "tests pass" or "the command returned 0" when the workflow being improved is user-visible. Inspect the actual app behavior, API payloads, saved artifacts, and live job traces closely enough to catch missing status cells, stale table rows, hidden model switches, metadata loss, accepted-count drift, vague generated questions, and misleading progress displays.

Every run should ask:

- What would John see in the Eval Matrix or API response while this is running?
- Does the UI expose enough detail to understand progress, failures, accepted/rejected decisions, source metadata, and generated artifacts without tailing logs?
- Does the backend state agree with the frontend state after refresh, clear, restart, and completed-job reload?
- Does a requested count mean successful accepted outcomes, not attempted candidates or partial work?
- Does the model receive enough source context to produce grounded work without relying on filenames, pages, hidden chunk ids, or noisy ingestion metadata?
- Are `NONE`, rejected, failed, and accepted outcomes classified and displayed honestly?
- Would the behavior still be correct after a browser refresh, UI service restart, or a later cron run?

When a bug spans retrieval/eval logic and the Eval Matrix or local app workflow, analyze the full path: user control, frontend payload, local UI endpoint, subprocess command, eval/generation code, event stream, persisted dataset, matrix reload, and visible table/detail rendering. Record the observed payload or job id when it proves behavior.

## Hard Scope

- Use the same local models already configured in this application.
- Do not change model providers, model names, embedding model, reranker model, or external services unless John explicitly asks.
- Prefer retrieval, evaluation, and answering logic, but fix the actual bottleneck. UI, local eval-console endpoints, and ingestion metadata are in scope when they directly affect evaluation usability, source-grounding, question quality, app observability, or John's ability to verify the system. Keep those changes tightly scoped and document why they were necessary.
- Avoid auth, infrastructure, Docker, schema, deployment, model/provider, parser, or broad ingestion changes unless John explicitly asks or the run proves the current app cannot produce/verify grounded behavior without that scoped change.
- Never make document-specific routing rules, filename heuristics, vendor-specific shortcuts, or eval-only production behavior.
- Preserve explicit user/request filters, but do not turn query text into hard document filters unless the request explicitly provides filters.
- Do not optimize by deleting hard questions, narrowing to tiny green subsets, weakening expected evidence without a source-backed reason, or adding logic that only works for the current eval cases.
- Before committing, ask whether the change would still be valid for an unseen manual, unseen vendor, and a question from an engineer, salesperson, manager, technician, or support person.

## Required State Files

Maintain these files on every run:

- `docs/runbooks/retrieval_accuracy_cron.md`: standing instructions only. Update rarely.
- `test_reports/retrieval_accuracy_progress.md`: append a short dated run note every run.
- `test_reports/retrieval_accuracy_question_bank_manifest.json`: maintain current question-bank counts, datasets, run ids, pass rates, failure categories, and the next target.

If a file does not exist, create it before doing evaluation work.

## Run Loop

Each 30-minute run should do the smallest complete improvement cycle possible:

1. Read `git status --short`, this runbook, and the progress manifest.
2. Confirm the local stack/test environment is usable and identify whether the running services have loaded the code path under test.
3. Select one target from John's latest feedback, unresolved guardrail findings, current eval failures, replacement debt, UI observability gaps, or coverage gaps.
4. Reproduce the target through the same route John uses when practical: Eval Matrix UI/local endpoint, generation job endpoint, eval script, saved dataset, or search/answer API.
5. Expand or refine the question bank with realistic engineer questions.
6. Run focused evals that distinguish:
   - single-step retrieval
   - multi-step retrieval
   - final answer accuracy and citation grounding
7. Make only scoped code/test changes in the layer that actually owns the problem.
8. Run focused tests first, then broader tests when the change affects shared behavior.
9. Run a live smoke check through the actual app/API path when the change affects a user-visible workflow, generated artifacts, model calls, or long-running jobs. Inspect the response or event stream, not just the final exit code.
10. Update the progress note and manifest with:
   - question-bank size
   - datasets touched
   - pass/fail rates
   - failure categories fixed or introduced
   - commands run
   - live job ids or API payload checks when relevant
   - files changed
   - next target
11. Commit successful scoped changes with a clear message.
12. Attempt to push; if credentials are unavailable, record that the local commit is ready.

## Temporary Worktree Hygiene

When the primary checkout is dirty or stale, prefer one known clean job-owned worktree over creating a fresh throwaway path. Before reusing it, verify `git status --short --branch` is clean and move it to the current `origin/main` with a normal non-destructive detach/switch. If no clean job-owned worktree exists, create at most one for the run and record its path.

At the end of any run that used a job-owned temporary worktree, verify there are no uncommitted or untracked files before attempting cleanup. Use supported `git worktree remove <exact-path>` only on the worktree used by the current run when it is safe to remove, then verify it no longer appears in `git worktree list --porcelain`. If removal fails or ownership is uncertain, leave the path registered, record the exact blocker in the progress log and manifest, and reuse that known path on the next run rather than creating another serial throwaway worktree. Do not bulk-clean historical worktrees from this job.

## Question Bank Requirements

Build toward 10000+ questions across retrieval and answer coverage. Quality matters, but count is also a coverage requirement: do not treat a smaller green eval as progress when coverage was reduced.

Question-bank growth rules:

- Retiring a bad or stale question creates replacement debt. Replace it with a fresh validated engineer question in the same run when possible; otherwise record the exact debt in the progress log and manifest as the next target.
- Active single-step and multi-step bank counts must be monotonic unless John explicitly approves a reset. If queryworthiness cleaning drops active cases, add replacement cases before declaring the bank healthy.
- `--max-queries` is a runtime batch size, not the desired durable bank size. Small 20-case batches are acceptable for cron timing, but they should accumulate into the bank instead of replacing larger coverage.
- Preserve retired-question metadata and reason codes so bad patterns are measurable without counting them as active coverage.
- Locked regression banks should only grow or be superseded by larger equivalent-quality banks.

Question classes to include:

- Direct spec lookup: numeric values, units, ranges, tolerances, classes, environmental limits.
- Table lookup: row/column/cell questions, model comparisons, option/accessory tables.
- Procedure lookup: setup, configuration, wiring, calibration, connection checks, maintenance.
- Warning and safety lookup: hazards, cautions, prohibited operations, preconditions.
- Troubleshooting and error lookup: alarms, symptoms, causes, remedies.
- Cross-section lookup: questions requiring a procedure plus a spec, a warning plus a step, or multiple table rows.
- Cross-document lookup: questions requiring selecting the right document first, then retrieving supporting evidence.
- Natural engineer phrasing: terse, misspelled, partial model names, units omitted, and realistic shop-floor wording.

Question generation rules:

- Questions must be answerable from source documentation.
- Do not include filename-only artifacts or opaque long-filename fragments.
- Do not ask questions a user could only ask after seeing internal chunk ids, filenames, run ids, or eval metadata.
- Use a model review pass for generated questions when quality is uncertain. The reviewer should receive the source content and generated question and either approve or provide actionable feedback; do not rely on brittle word-count or banned-word validation to decide quality.
- If a generation request asks for `N` questions, keep going until `N` questions are accepted or the source candidates are exhausted. Rejections, parse failures, and `NONE` windows do not count toward the requested accepted total.
- Treat `NONE` as the right output for weak, duplicate, internal-format, or already-covered source windows. Track `NONE` separately from rejection and failure.
- Generated question rows must preserve model-facing source context such as parent article/section, product/device/family when grounded, source snippet, and expected evidence. If that information is unavailable or wrong because ingestion metadata is missing/noisy, investigate the ingestion metadata path rather than teaching the generator to guess.
- The Eval Matrix must show live generation progress: model output, reviewer status, accepted/rejected/NONE decisions, accumulated accepted questions, generated dataset path, and review status after refresh.
- Store expected evidence ids/snippets/terms so regressions are measurable.
- Separate train/dev-style exploration data from locked regression data.

## Non-Regression Gates

Before committing a change, at minimum run the focused tests for touched modules.

When retrieval, evaluation, or answering shared behavior changes, run:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q
```

For eval-pipeline changes, also run the smallest relevant benchmark script and record the report path.

For Eval Matrix or question-generation workflow changes, also exercise the local UI endpoint or browser-visible path and inspect the returned JSON/table state. Verify clear/generate/reload behavior when relevant, including that generated rows display review status and that requested accepted counts are honored.

A change is not done if it improves one query by hardcoding that query, filename, vendor, product, or document-specific phrase.

## Time Budget

The cron schedule is every 30 minutes. Keep each run within 30 minutes.

If the evaluation pipeline grows so a normal useful run cannot finish in 30 minutes, do not silently extend runtime. Record the evidence in the progress log and leave a recommendation to update the cron timeout/schedule.
