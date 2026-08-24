# Retrieval Accuracy Cron Runbook

This runbook is the standing brief for the recurring retrieval and answer accuracy agent job.

## Mission

Advance Manuals RAG until it can answer every reasonable engineering question that can be answered from the indexed documentation.

The job must improve two capabilities:

1. Optimal and efficient single-step retrieval for direct questions whose answer is in one chunk, row, table cell, warning, step, or short section.
2. Multi-step retrieval for questions that require gathering evidence from multiple chunks, sections, tables, pages, or documents before answering.

## Hard Scope

- Use the same local models already configured in this application.
- Do not change model providers, model names, embedding model, reranker model, or external services unless John explicitly asks.
- Only update retrieval, evaluation, and answering logic.
- Avoid ingestion, parser, UI, auth, infrastructure, Docker, schema, or deployment changes unless a test fixture or eval harness absolutely cannot run without a tiny scoped adjustment.
- Never make document-specific routing rules, filename heuristics, vendor-specific shortcuts, or eval-only production behavior.
- Preserve explicit user/request filters, but do not turn query text into hard document filters unless the request explicitly provides filters.

## Required State Files

Maintain these files on every run:

- `docs/runbooks/retrieval_accuracy_cron.md`: standing instructions only. Update rarely.
- `test_reports/retrieval_accuracy_progress.md`: append a short dated run note every run.
- `test_reports/retrieval_accuracy_question_bank_manifest.json`: maintain current question-bank counts, datasets, run ids, pass rates, failure categories, and the next target.

If a file does not exist, create it before doing evaluation work.

## Run Loop

Each 30-minute run should do the smallest complete improvement cycle possible:

1. Read `git status --short`, this runbook, and the progress manifest.
2. Confirm the local stack/test environment is usable.
3. Select one target from the current eval failures or coverage gaps.
4. Expand or refine the question bank with realistic engineer questions.
5. Run focused evals that distinguish:
   - single-step retrieval
   - multi-step retrieval
   - final answer accuracy and citation grounding
6. Make only scoped code/test changes in retrieval, evals, or answering.
7. Run focused tests first, then broader tests when the change affects shared behavior.
8. Update the progress note and manifest with:
   - question-bank size
   - datasets touched
   - pass/fail rates
   - failure categories fixed or introduced
   - commands run
   - files changed
   - next target
9. Commit successful scoped changes with a clear message.
10. Attempt to push; if credentials are unavailable, record that the local commit is ready.

## Question Bank Requirements

Build toward 10000+ questions if needed. Quality matters more than count.

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
- Store expected evidence ids/snippets/terms so regressions are measurable.
- Separate train/dev-style exploration data from locked regression data.

## Non-Regression Gates

Before committing a change, at minimum run the focused tests for touched modules.

When retrieval, evaluation, or answering shared behavior changes, run:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q
```

For eval-pipeline changes, also run the smallest relevant benchmark script and record the report path.

A change is not done if it improves one query by hardcoding that query, filename, vendor, product, or document-specific phrase.

## Time Budget

The cron schedule is every 30 minutes. Keep each run within 30 minutes.

If the evaluation pipeline grows so a normal useful run cannot finish in 30 minutes, do not silently extend runtime. Record the evidence in the progress log and leave a recommendation to update the cron timeout/schedule.
