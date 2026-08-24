# Retrieval Accuracy Guardrail Findings

## 2026-08-24T21:10:00Z Guardrail Review

- Reviewed accuracy job: `39262386-1bb6-4571-98e1-13a30047ddb8`
- Reviewed commits: `ce562a3..dc667d9` (`Pay down single-step retrieval question debt`)
- Reviewed run evidence: progress-log entries through `retrieval_eval_20260824_210311`; direct cron run history was unavailable because the cron tool was restricted to the current guardrail job.
- Severity: `needs_fix`
- Finding: the replacement-debt run correctly avoided production retrieval changes and increased active single-step coverage from 87 to 100, but the 13 promoted replacement questions are not all realistic engineer/user phrasing. Several still look source-derived or table/TOC-shaped, for example `test_reports/retrieval_eval_dataset_20260824_210121.jsonl` lines 1, 2, 3, 7, 9, 10, and 13. Line 7 in particular asks "What contents is described for IV4-G120?" against a contents/TOC-like snippet, despite the run note saying TOC/file-list prompts were filtered.
- Evidence: `packages/evals/src/manuals_rag_evals/retrieval_eval.py` added a mechanical-query filter only; no production retrieval, answering, parser, model, or provider logic changed. Regression gates reported green (`retrieval_eval_20260824_210219` at 20/20 single-step and `retrieval_eval_20260824_210311` at 15/15 warning-step), but those do not validate the realism of the new 13-question replacement slice.
- Required next action: do not treat the 13-question replacement debt as truly complete until the replacement slice is re-reviewed or regenerated with stronger realistic-user phrasing/queryworthiness checks. Keep active counts monotonic, but mark/replace weak replacement questions rather than tuning retrieval against them.

## 2026-08-24T21:41:28Z Guardrail Review

- Reviewed accuracy job: `39262386-1bb6-4571-98e1-13a30047ddb8`
- Reviewed commits: `905259e..2affd6f` (`Classify reverse table lookup queries`)
- Reviewed run evidence: progress-log entries through `retrieval_eval_20260824_212704`; direct cron run history was unavailable because the cron tool was restricted to the current guardrail job.
- Severity: `needs_fix`
- Finding: the latest run changed production retrieval classification to make the remaining failure in the questionable 13-case replacement-debt slice pass, before completing the prior guardrail action to re-review or replace weak generated questions in that slice. The code change is grammar-based rather than document/vendor-specific, and no parser/model/provider/infrastructure changes were found, but using the weak replacement bank as the optimization target risks treating a tiny green slice as production progress.
- Evidence: `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py` added reverse lookup phrasing such as `what setting item ... selects ...` to `structured_lookup`; `tests/unit/test_retriever.py` added one focused example matching the LJ-X8000 failure. The logged evals show `retrieval_eval_20260824_212529` improved the 13-case replacement-debt bank from 12/13 to 13/13 and preserved the 20-case single-step and 15-case warning-step regression gates, but no new realistic-user replacement review was recorded and answer grounding remained unmeasured.
- Required next action: stop optimizing against `test_reports/retrieval_eval_dataset_20260824_210121.jsonl` until its weak source-shaped questions are replaced or explicitly marked diagnostic-only. The next accuracy run should pay down that quality debt with realistic engineer/user phrasing, keep active counts monotonic, and only then use the replacement slice as a production retrieval target.
