# Retrieval Accuracy Guardrail Findings

## 2026-08-24T21:10:00Z Guardrail Review

- Reviewed accuracy job: `39262386-1bb6-4571-98e1-13a30047ddb8`
- Reviewed commits: `ce562a3..dc667d9` (`Pay down single-step retrieval question debt`)
- Reviewed run evidence: progress-log entries through `retrieval_eval_20260824_210311`; direct cron run history was unavailable because the cron tool was restricted to the current guardrail job.
- Severity: `needs_fix`
- Finding: the replacement-debt run correctly avoided production retrieval changes and increased active single-step coverage from 87 to 100, but the 13 promoted replacement questions are not all realistic engineer/user phrasing. Several still look source-derived or table/TOC-shaped, for example `test_reports/retrieval_eval_dataset_20260824_210121.jsonl` lines 1, 2, 3, 7, 9, 10, and 13. Line 7 in particular asks "What contents is described for IV4-G120?" against a contents/TOC-like snippet, despite the run note saying TOC/file-list prompts were filtered.
- Evidence: `packages/evals/src/manuals_rag_evals/retrieval_eval.py` added a mechanical-query filter only; no production retrieval, answering, parser, model, or provider logic changed. Regression gates reported green (`retrieval_eval_20260824_210219` at 20/20 single-step and `retrieval_eval_20260824_210311` at 15/15 warning-step), but those do not validate the realism of the new 13-question replacement slice.
- Required next action: do not treat the 13-question replacement debt as truly complete until the replacement slice is re-reviewed or regenerated with stronger realistic-user phrasing/queryworthiness checks. Keep active counts monotonic, but mark/replace weak replacement questions rather than tuning retrieval against them.

