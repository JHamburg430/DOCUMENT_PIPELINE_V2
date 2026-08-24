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

## 2026-08-24 Cron 39262386 Follow-up

- Target: add the efficient retrieval-only focused eval path for saved question-bank datasets.
- Local stack: compose services were up; API and Postgres were reachable through the existing compose environment.
- Changed eval harness logic in `scripts/benchmark/run_large_retrieval_eval.py` to accept `--dataset-path` with `--existing-corpus-id`, load saved `RetrievalEvalCase` JSONL records, cap them with `--max-queries`, and write normal dataset/results/summary/manifest artifacts. The runner now uses `response_mode: retrieval_only` for `/search` calls and can read corpus metadata from either host-side `docker exec` or in-container `POSTGRES_DSN`.
- Added focused test coverage in `tests/unit/test_retrieval_eval.py` for saved dataset loading, blank-line handling, wrapped `{"case": ...}` records, max-case limiting, and default `retrieval_task` preservation.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py -q` -> 23 passed.
- Live eval attempts:
  - `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_024426.jsonl --max-queries 40` -> stopped after about 90 seconds; 1/40 cases completed, failed with rank null.
  - Same command with `--max-queries 10` -> stopped after about 2 minutes after loading cases, before first completed case.
- Completed eval result: none this run. Failure category evidence is evaluation latency/timeout rather than a measured retrieval regression.
- Changed files: `scripts/benchmark/run_large_retrieval_eval.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`.

Next target:

- Make saved-bank retrieval eval fast enough for cron by bypassing HTTP/query overhead or adding per-query timing/timeout controls in the eval harness, then rerun the 40-case single-step bank and generate the first true multi-step cases.

## 2026-08-24 Cron 39262386 Timeout Controls

- Target: make saved-bank retrieval eval bounded enough for cron runs to produce artifacts instead of stalling.
- Local stack: compose services were up; API, Postgres, and Qdrant were reachable through the existing compose environment. Direct in-process retrieval of the first saved-bank case was manually stopped after more than 60 seconds, confirming latency is inside retrieval/model work rather than only HTTP overhead.
- Changed eval harness logic in `scripts/benchmark/run_large_retrieval_eval.py` to add `--search-mode direct|http`, `--per-query-timeout-seconds`, per-query elapsed timing, direct in-process retrieval using the app retriever, timeout scoring as `eval_timeout`, and JSON-safe manifest writing for UUID-valued document metadata.
- Added focused tests in `tests/unit/test_retrieval_eval.py` for timeout scoring and wrapped timeout exception detection.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py -q` -> 25 passed.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_024426.jsonl --max-queries 3 --search-mode direct --per-query-timeout-seconds 8`.
- Live eval result: completed 3 saved single-step table questions in bounded mode; 0/3 passed, pass@1/pass@3/pass@5 0%, failure categories `eval_timeout: 3`. Artifacts: `test_reports/retrieval_eval_dataset_20260824_032823.jsonl`, `test_reports/retrieval_eval_results_20260824_032823.jsonl`, `test_reports/retrieval_eval_summary_20260824_032823.json`, `test_reports/retrieval_eval_manifest_20260824_032823.json`.
- Prior useful eval artifact already present at start of this run: `test_reports/retrieval_eval_summary_20260824_025839.json` reported 7/10 passed (70%) on saved single-step table cases, with failures `candidate_miss: 1` and `wrong_document_or_filter_loss: 2`.
- Changed files: `scripts/benchmark/run_large_retrieval_eval.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`.

Next target:

- Profile and optimize single-query retrieval latency, especially the sparse/table paths that scroll large Qdrant payload sets for every query, then rerun the 40-case saved single-step bank with bounded per-query timing before generating true multi-step cases.

## 2026-08-24 Cron 39262386 Native Sparse Search

- Target: reduce single-query retrieval latency in the sparse/table path that previously scrolled Qdrant payloads and timed out direct saved-bank evals.
- Local stack: compose services were up; API, Postgres, and Qdrant were reachable. Live native sparse-vector smoke search against `manuals_vendor_keyence` returned 3 table hits for a measurement-range query.
- Changed retrieval logic in `packages/retrieval/src/manuals_rag_retrieval/qdrant_store.py` so `search_sparse` queries Qdrant's stored sparse vector directly with `NamedSparseVector` instead of scrolling all matching payloads and building local BM25 every query. The previous BM25 scroll remains as a compatibility fallback, and eval timeout exceptions are re-raised instead of being swallowed by fallback handling.
- Added focused tests in `tests/unit/test_retriever.py` covering native sparse-vector search, BM25 fallback, and timeout re-raise behavior.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py -q` -> 40 passed, 1 warning.
- Broader tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 170 passed, 58 warnings.
- Live eval commands:
  - `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_024426.jsonl --max-queries 3 --search-mode direct --per-query-timeout-seconds 8`
  - `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_024426.jsonl --max-queries 10 --search-mode direct --per-query-timeout-seconds 8`
- Live eval result: corrected bounded 3-case run `retrieval_eval_20260824_035739` completed at 2/3 passed (66.67%), pass@1/pass@3/pass@5 66.67%, failure categories `eval_timeout: 1`. The first query still pays reranker warmup and times out; the next two warmed queries completed in 2.510s and 1.092s at rank 1. Earlier same-run 10-case evidence `retrieval_eval_20260824_035613` completed at 8/10 passed (80%), pass@1 80%, failures `candidate_miss: 1`, `wrong_document_or_filter_loss: 1`; it also exposed that timeout exceptions were being swallowed before the final fix.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/qdrant_store.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for the 03:55-03:57 UTC runs.

Next target:

- Add explicit eval/runtime warmup accounting or a harness warmup phase so bounded saved-bank evals do not mark first-query model loading as retrieval failure, then rerun the full 40-case single-step bank and investigate the remaining AS_151292 metadata/candidate misses before generating true multi-step cases.

## 2026-08-24 Cron 39262386 Warmup Accounting

- Target: separate first-query model/retriever startup cost from scored retrieval failures in saved-bank direct evals.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus was usable with 53 indexed documents.
- Changed eval harness logic in `scripts/benchmark/run_large_retrieval_eval.py` to add `--warmup-queries` and `--warmup-timeout-seconds`, run unscored warmup searches before timed scoring, record warmup status/timing in the eval manifest, and share search dispatch between HTTP/direct modes.
- Added focused tests in `tests/unit/test_retrieval_eval.py` for completed warmup accounting and warmup timeout accounting.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py -q` -> 27 passed.
- Live eval commands:
  - `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_024426.jsonl --max-queries 10 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`
  - `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_024426.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`
- Live eval result: 10-case warmup smoke `retrieval_eval_20260824_042510` completed at 8/10 passed (80%) with no `eval_timeout` failures; the unscored warmup completed in 12.202s.
- Full saved-bank result: `retrieval_eval_20260824_042547` completed all 40 single-step questions at 28/40 passed (70%), pass@1 67.5%, pass@3/pass@5 70%, no `eval_timeout` failures; the unscored warmup completed in 10.198s. Table questions were 20/30, atomic text questions were 8/10. Failure categories: `candidate_miss: 7`, `wrong_document_or_filter_loss: 3`, `ranking_or_context_loss: 2`.
- Failure evidence: all 6 saved-bank cases from `AS_151292_VS_UM_J18GB_WW_GB_2035_7.pdf` failed in the full run, so the next target should focus on generic VS/manual table metadata selection and candidate recall rather than eval timing.
- Changed files: `scripts/benchmark/run_large_retrieval_eval.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval artifacts for `retrieval_eval_20260824_042510` and `retrieval_eval_20260824_042547`.

Next target:

- Investigate and improve the remaining VS/manual table retrieval failures, especially AS_151292 metadata-document selection and candidate misses, then generate the first true multi-step cases once single-step evidence improves.
