# Retrieval Accuracy Guardrail Cron Runbook

This runbook is the standing brief for the guardrail job that audits the recurring Manuals RAG retrieval accuracy job.

## Production Goal

Manuals RAG is being built as a production system that can retrieve and generate grounded answers for any reasonable question about the indexed documentation. The expected users include engineers, salespeople, managers, technicians, support people, and other document consumers.

The guardrail job protects that goal. It should prevent the accuracy cron from drifting into benchmark gaming, narrow-case patching, eval bias, or unnecessary changes that make the system less generally useful.

It should also protect John's app-level quality expectations. If the accuracy job changes a workflow that John experiences through the Eval Matrix, local UI endpoints, generated datasets, or long-running jobs, audit the user-visible behavior and persisted state, not only the code diff and pytest result.

## Audited Job

- Accuracy job id: `39262386-1bb6-4571-98e1-13a30047ddb8`
- Accuracy job name: `Manuals RAG Retrieval Accuracy Improvement`
- Repository: `/home/john/Desktop/Programming/Document_Pipeline_V2/manuals_rag`
- Primary runbook: `docs/runbooks/retrieval_accuracy_cron.md`
- Progress log: `test_reports/retrieval_accuracy_progress.md`
- Question-bank manifest: `test_reports/retrieval_accuracy_question_bank_manifest.json`

## Red Lines

Flag a critical issue if an accuracy run does any of these:

- Adds document-specific, vendor-specific, filename-derived, query-specific, or eval-only production behavior.
- Turns inferred query text into hard document filters when the user did not explicitly provide filters.
- Uses opaque filename fragments, chunk ids, run ids, internal metadata, or copied source phrasing to make eval questions easier.
- Improves pass rate by deleting hard questions, shrinking active coverage, weakening expected evidence, or replacing broad coverage with a small green slice.
- Changes parser, ingestion, infrastructure, Docker, schema, auth, UI, model provider, model name, embedding model, or reranker settings without John's explicit request or a documented tiny eval-harness necessity.
- Commits a retrieval/answering change without meaningful regression gates for single-step retrieval, multi-step retrieval, and answer grounding when those surfaces are affected.
- Leaves a failed or regressing experiment in production code.

Flag at least `needs_fix` when an accuracy run claims a user-visible eval/generation workflow is fixed but does not prove the actual app path:

- Requested `N` generated questions but accepted fewer than `N` while unattempted ranked source candidates remain.
- Counts rejected candidates, parse failures, or `NONE` windows as generated/accepted questions.
- Does not show live model/reviewer progress, accepted/rejected/NONE decisions, accumulated accepted questions, generated dataset path, or review status in the Eval Matrix when those are relevant.
- Leaves generated rows without review status after refresh, clear, service restart, or matrix reload.
- Clears backend state without the UI table reflecting the clear.
- Uses mismatched models for generation/review without documenting and justifying the runtime-load cost.
- Feeds the model misleading or missing source metadata when parent article, section, grounded product/device/family, or source snippet context is available.
- Lets bad ingestion metadata force the question generator to guess instead of tracing and correcting the ingestion/source-context path.

## Review Checklist

For each recent accuracy cron run, review:

- The cron run summary and any diagnostics.
- The git commits created since the last guardrail review.
- Diffs in retrieval, evaluation, and answering code.
- Progress-log and manifest changes, especially question counts, replacement debt, dropped cases, pass/fail rates, and failure categories.
- Whether eval datasets grew toward broad production coverage or merely optimized a narrow current bank.
- Whether the change improves behavior for unseen manuals and realistic user phrasing, not only named examples in the current report.
- Whether answer generation and citation grounding are being measured, not only retrieval ranking.
- Whether any UI/local-endpoint workflow change was exercised through the actual route John uses, with job ids, API payloads, event traces, or visible table state recorded.
- Whether model-generated questions were reviewed by a model using the source content and candidate question, and whether rejection feedback was used constructively instead of replaced by brittle banned-word/length filters.
- Whether generated question batches meet accepted-count semantics: `max_questions` means accepted questions, while `NONE` and rejected candidates are tracked separately.
- Whether generated rows retain and display review status after reload.
- Whether source metadata supplied to the generation model is grounded and useful, and whether missing/noisy metadata was traced to ingestion or source-context construction.

## Allowed Findings

Use these severities:

- `ok`: no issue found.
- `watch`: acceptable but needs follow-up evidence, broader coverage, or answer-generation validation.
- `needs_fix`: questionable or incomplete change that should be corrected in the next accuracy run.
- `critical`: likely counterproductive, biased, hardcoded, or production-risky change.

## Actions

- If everything is `ok`, do not create noisy commits. Record only in the guardrail state file if one exists.
- For `watch` or `needs_fix`, append a concise note to `test_reports/retrieval_accuracy_guardrail.md` and set the next accuracy target in the manifest/progress log when appropriate.
- For `critical`, append a guardrail note, attempt to pause or disable the accuracy cron if the cron tool permits it, and leave clear instructions for John/main-session review. Do not silently revert user or cron work.
- Never edit production retrieval/answering logic from the guardrail job. Its job is audit, containment, and instruction correction.

When auditing app-level workflow fixes, require evidence proportionate to the claim:

- Code-only tests are enough for pure helpers and data transforms.
- UI/local endpoint fixes require inspecting the endpoint response or browser-visible state.
- Long-running matrix jobs require inspecting the job snapshot/event stream and the persisted dataset or matrix rows after completion.
- Runtime behavior claims require evidence that the running service or subprocess loaded the changed code.

## Guardrail State

Maintain `test_reports/retrieval_accuracy_guardrail.md` when findings exist. Include:

- review timestamp
- reviewed commit range and cron run ids
- severity
- findings with file/line or commit references when available
- required next action

If no findings exist and no state file exists, no repository change is required.
