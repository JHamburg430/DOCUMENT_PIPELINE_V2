# Retrieval Accuracy Progress

This log is maintained by the recurring retrieval accuracy cron job.

## 2026-08-23 Setup

- Created the standing runbook at `docs/runbooks/retrieval_accuracy_cron.md`.
- Cron target: run every 30 minutes with a 30-minute timeout.
- Scope: retrieval, evaluation, and answering logic only, using the existing local model configuration.
- Initial local commits awaiting push due missing GitHub HTTPS credentials:
  - `11239a1` Stop eval questions using filename artifacts
  - `9a4521a` Avoid inferred metadata filters from queries
- Latest verification before setup: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> `163 passed, 55 warnings`.

Next target:

- Start building a durable question-bank manifest and split coverage between direct single-step retrieval and multi-step retrieval.

## 2026-08-24 Cron 39262386

- Target: first durable question-bank baseline and single-step/multi-step tracking.
- Local stack: compose services were up; live indexed corpora available were `manuals_vendor_keyence` (53 indexed docs) and `manuals_prod_smoke` (1 indexed doc). Older eval corpus `manuals_eval_20260404_025707` was no longer present in the live DB.
- Changed eval logic to add `retrieval_task` to `RetrievalEvalCase` (default `single_step_retrieval`), summarize benchmark results by retrieval task, and generate table questions from row headers, column headers, and high-signal cell anchors instead of generic `table/header/column` terms.
- Tightened eval label handling so long slash-delimited product lists are not used as user-facing query labels.
- Added deterministic question-bank artifact: `test_reports/retrieval_accuracy_question_bank_20260824_024426.jsonl` with 40 exploratory single-step questions (`table_record`: 30, `atomic_text`: 10). Summary: `test_reports/retrieval_accuracy_question_bank_20260824_024426.summary.json`.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py tests/unit/test_api_debug.py -q` -> 35 passed, 9 warnings.
- Live eval evidence: full LLM-generated live eval was stopped after about 2 minutes with no completed cases; deterministic answer-with-citations live eval completed 20 queries in `test_reports/retrieval_eval_summary_20260824_024852.json`.
- Live eval result: 8/20 passed (40% pass rate), pass@1 25%, pass@3/pass@5 40%, all `single_step_retrieval`; failures were `candidate_miss` (9) and `wrong_document_or_filter_loss` (3). Dataset/results: `test_reports/retrieval_eval_dataset_20260824_024852.jsonl`, `test_reports/retrieval_eval_results_20260824_024852.jsonl`.

Next target:

- Add an efficient retrieval-only focused eval path for the saved question bank, then start generating true multi-step cases with expected evidence from multiple chunks/sections.
