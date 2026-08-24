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

## 2026-08-24 Cron 39262386 Product-Family Labels

- Target: improve the remaining VS/manual table retrieval failures without filename routing or document-specific shortcuts.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus was usable with 53 indexed documents.
- Failure review: prior full saved-bank run `retrieval_eval_20260824_042547` was 28/40 (70%) and all 6 AS_151292 VS cases failed. Several failed prompts lacked a natural product/family scope because long product-model lists were dropped from query labels.
- Changed eval logic in `packages/evals/src/manuals_rag_evals/retrieval_eval.py` so deterministic and LLM-assisted question generation can use concise `product_family` as the user-facing label when `product_model` is an unwieldy slash-delimited list. The eval discriminator now accepts long named table concepts as source-specific anchors.
- Changed retrieval logic in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` so scalar `product_model` and `product_family` participate in generic identifier scoring, with a larger adjustment for multiple matching identifier terms instead of a flat one-term match.
- Changed eval harness in `scripts/benchmark/run_large_retrieval_eval.py` to add `--disable-llm-query-generation`, keeping cron-sized question-bank refreshes deterministic and bounded.
- Added focused unit coverage in `tests/unit/test_retrieval_eval.py` and `tests/unit/test_retriever.py` for product-family fallback labels and scalar product-family retrieval scoring.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py tests/unit/test_retriever.py -q` -> 69 passed, 1 warning.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 174 passed, 58 warnings.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45 --disable-llm-query-generation`.
- Live eval result: refreshed deterministic single-step bank completed at 32/40 passed (80%), pass@1 70%, pass@3 77.5%, pass@5 80%, candidate recall 97.5%, metadata-document recall 97.37%, no `eval_timeout` failures. AS_151292 VS cases improved to 10/11 passed. Failure categories: `ranking_or_context_loss: 7`, `wrong_document_or_filter_loss: 1`.
- New artifacts: `test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl`, `test_reports/retrieval_accuracy_question_bank_20260824_045817.summary.json`, `test_reports/retrieval_eval_dataset_20260824_045817.jsonl`, `test_reports/retrieval_eval_results_20260824_045817.jsonl`, `test_reports/retrieval_eval_summary_20260824_045817.json`, `test_reports/retrieval_eval_manifest_20260824_045817.json`.
- Question-bank manifest now tracks 80 exploratory single-step questions across two datasets; multi-step remains 0.

Next target:

- Improve ranking/context selection for high-recall misses, especially IV4 address/detection and protocol-symbol table cases, then add the first true multi-step retrieval cases.

## 2026-08-24 Cron 39262386 Structured Applies-To Lookups

- Target: improve high-recall ranking/context misses for terse engineering field lookups such as address, symbol, message, and error-message questions.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus was usable with 53 indexed documents.
- Failure review: prior refreshed bank run `retrieval_eval_20260824_045817` was 32/40 (80%) with candidate recall 97.5%, metadata-document recall 97.37%, and failures mostly `ranking_or_context_loss`. Several failed prompts used the generic form `What <field> ... applies to <model/family>?` and were analyzed as general prose or troubleshooting instead of structured table/spec lookups.
- Changed retrieval query analysis in `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py` to classify `applies to/for` field questions as `structured_lookup` and prefer `table_record`, `spec_record`, and `section_window` candidates.
- Changed retrieval family scoring/selection in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` so `structured_lookup` queries prefer table evidence first, then spec/context evidence, without adding any query-derived document filters.
- Added focused unit coverage in `tests/unit/test_retriever.py` for structured applies-to classification and table-family candidate selection.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py tests/unit/test_filters.py -q` -> 51 passed, 1 warning.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 176 passed, 58 warnings.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Live eval result: saved 40-question single-step bank improved to 34/40 passed (85%), pass@1 77.5%, pass@3/pass@5 85%, candidate recall 100%, metadata-document recall 100%, metadata-document rank-1 rate 68.42%. Failure categories: `ranking_or_context_loss: 6`; `wrong_document_or_filter_loss` dropped to 0. Table questions were 28/33, atomic text 5/6, spec 1/1.
- New artifacts: `test_reports/retrieval_eval_dataset_20260824_052609.jsonl`, `test_reports/retrieval_eval_results_20260824_052609.jsonl`, `test_reports/retrieval_eval_summary_20260824_052609.json`, `test_reports/retrieval_eval_manifest_20260824_052609.json`.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py`, `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for the 05:26 UTC run.

Next target:

- Improve context assembly/scoring for the remaining all-candidate-recall misses, especially IV4 address/detection table rows and short XG-X error-message table cells, then generate the first true multi-step retrieval cases.

## 2026-08-24 Cron 39262386 Table Question Specificity

- Target: reduce false single-step table failures from under-specified generated table-cell questions while making a small generic ranking improvement for structured table lookups.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus was usable with 53 indexed documents.
- Failure review: prior saved-bank run `retrieval_eval_20260824_052609` was 34/40 (85%) with all failures in `ranking_or_context_loss`. One short XG-X error-message cell failed because the question only asked for "Error Message value" for a product family, and IV4 address failures showed table header/model chunks outranking value rows.
- Changed eval question generation in `packages/evals/src/manuals_rag_evals/retrieval_eval.py` so table key-value rows generate questions that include the adjacent row-disambiguating value, and answer-only table cells without row context are not treated as single-step queryworthy. These should move error-message row/cause/action coverage toward future multi-step cases instead of ambiguous single-cell lookups.
- Changed retrieval scoring in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` to demote table-header chunks for structured field/value lookup queries, without adding document-specific routing or query-derived filters.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py tests/unit/test_retriever.py -q` -> 74 passed, 1 warning.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 179 passed, 58 warnings.
- Live eval command (refreshed deterministic generation): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45 --disable-llm-query-generation`.
- Refreshed eval result: `retrieval_eval_20260824_055851` completed 40 single-step questions at 29/40 passed (72.5%), pass@1 67.5%, pass@3 70%, pass@5 72.5%, candidate recall 85%, metadata-document recall 84.38%. Failures: `eval_timeout: 1`, `ranking_or_context_loss: 5`, `candidate_miss: 4`, `wrong_document_or_filter_loss: 1`. This was a harder regenerated dataset, not added as a durable bank file.
- Live eval command (saved-bank apples-to-apples): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved-bank result: `retrieval_eval_20260824_060124` improved from 34/40 (85%) to 35/40 (87.5%), pass@1 80%, pass@3/pass@5 87.5%, candidate recall 100%, metadata-document recall 100%, failure categories `ranking_or_context_loss: 5`. The previous XG-X error-message single-cell case now passed at rank 1.
- Changed files: `packages/evals/src/manuals_rag_evals/retrieval_eval.py`, `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `tests/unit/test_retrieval_eval.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for the 05:58 and 06:01 UTC runs.

Next target:

- Add the first true multi-step retrieval cases for table rows that require combining sibling cells, especially error-message/cause/corrective-action rows and IV4 address/stored-data rows, then improve context assembly so sibling row cells are available to the answerer.

## 2026-08-24 Cron 39262386 Multi-Step Sibling Evidence

- Target: add the first true multi-step retrieval cases for table rows requiring sibling evidence, starting with error-message/symptom, cause, and corrective-action rows.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus was usable with 53 indexed documents.
- Changed eval logic in `packages/evals/src/manuals_rag_evals/retrieval_eval.py` to add optional `expected_source_chunk_ids` and `expected_evidence` to `RetrievalEvalCase`, generate deterministic `multi_step_retrieval` cases from sibling table cells with error/symptom, cause, and corrective-action evidence, and score multi-step cases by requiring each expected evidence item in the final top-k context.
- Changed benchmark logic in `scripts/benchmark/run_large_retrieval_eval.py` to add `--retrieval-task single_step_retrieval|multi_step_retrieval` for deterministic generation when no saved dataset is provided.
- Added focused unit coverage in `tests/unit/test_retrieval_eval.py` for multi-step case generation and multi-step evidence scoring.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py -q` -> 32 passed.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 181 passed, 58 warnings in 151.13s. The full suite is now long enough that future cron runs should prefer focused tests unless shared behavior changed.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --retrieval-task multi_step_retrieval --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45 --disable-llm-query-generation`.
- Live eval result: first exploratory multi-step bank completed 20 table-row sibling-evidence questions at 10/20 passed (50%), pass@1 40%, pass@3/pass@5 50%, candidate recall 85%, metadata-document recall 100% across 18 metadata-selection attempts. Failure categories: `ranking_or_context_loss: 7`, `eval_timeout: 2`, `candidate_miss: 1`. Warmup completed in 8.981s.
- New artifacts: `test_reports/retrieval_eval_dataset_20260824_062714.jsonl`, `test_reports/retrieval_eval_results_20260824_062714.jsonl`, `test_reports/retrieval_eval_summary_20260824_062714.json`, `test_reports/retrieval_eval_manifest_20260824_062714.json`. The generated dataset is now tracked in the question-bank manifest as the first exploratory multi-step dataset.
- Changed files: `packages/evals/src/manuals_rag_evals/retrieval_eval.py`, `scripts/benchmark/run_large_retrieval_eval.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus the new 06:27 UTC eval summary/manifest artifacts.

Next target:

- Improve context assembly for multi-step sibling-row evidence so answer/citation context includes the matching cause and corrective-action cells or row-group chunk together, then rerun the 20-case multi-step bank with a slightly higher per-query timeout or record a timeout/schedule recommendation if 8 seconds is still too tight.

## 2026-08-24 Cron 39262386 Table Row-Group Context

- Target: improve multi-step context assembly for sibling table-row evidence so cause and corrective-action cells are available together for retrieval scoring and answer generation.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Changed query analysis in `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py` so natural cause/correction questions such as `What causes ... and how should it be corrected?` are treated as structured troubleshooting lookups with table/context preference instead of narrative-only troubleshooting.
- Changed retrieval context assembly in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` so table-record results receive the nearest same-section table row-group chunk as `context_window`/`table_row_group_context`, preserving section/parent context without adding document-specific routing or query-derived filters.
- Changed answering/eval evidence handling in `packages/answering/src/manuals_rag_answering/generator.py` and `packages/evals/src/manuals_rag_evals/retrieval_eval.py` so table row-group context can support multi-step cause/remedy answers and retrieval scoring.
- Added focused unit coverage in `tests/unit/test_retriever.py` and `tests/unit/test_retrieval_eval.py` for cause/correction intent detection, row-group context assembly, and multi-step evidence matching through context.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py tests/unit/test_retrieval_eval.py tests/unit/test_parser_and_answering.py -q` -> 100 passed, 24 warnings.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 184 passed, 58 warnings in 151.23s.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_eval_dataset_20260824_062714.jsonl --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Live eval result: saved 20-case multi-step sibling table-row bank improved from 10/20 (50%) to 14/20 (70%), pass@1 60%, pass@3 65%, pass@5 70%, candidate recall 90%, metadata-document recall 100%, metadata-document rank-1 rate 85%, and no scored `eval_timeout` failures. Failure categories: `ranking_or_context_loss: 4`, `candidate_miss: 2`.
- New artifacts: `test_reports/retrieval_eval_summary_20260824_065927.json`, `test_reports/retrieval_eval_manifest_20260824_065927.json`; generated dataset/results artifacts for the run are present under the same timestamp where tracked/available.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py`, `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `packages/evals/src/manuals_rag_evals/retrieval_eval.py`, `packages/answering/src/manuals_rag_answering/generator.py`, `tests/unit/test_retriever.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus the new 06:59 UTC eval summary/manifest artifacts.

Next target:

- Improve remaining multi-step candidate recall and ranking/context losses for cause/corrective-action rows, especially XG-X expansion/backup/firmware cases and CV-X light-controller/LJ-head misses, then add broader multi-step cases beyond sibling troubleshooting tables.

## 2026-08-24 Cron 39262386 Structured Troubleshooting Table Rows

- Target: fix the remaining saved multi-step sibling-row failures for cause/corrective-action prompts without document-specific routing or query-derived hard filters.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Failure review: prior saved multi-step run `retrieval_eval_20260824_065927` was 14/20 (70%) with `ranking_or_context_loss: 4` and `candidate_miss: 2`. The misses were natural troubleshooting questions such as XG-X line-scan/expansion/backup/firmware rows and CV-X light-controller/LJ-head rows.
- Root cause: structured cause/correction prompts also contain `how`, so family selection treated them as procedure/context lookups before `structured_lookup` and excluded table candidates before rerank. Metadata document selection also made some fallback searches too narrow for CV-X/LJ-head table rows.
- Changed retrieval logic in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` so `structured_lookup` takes precedence over `how_to`/`configuration` during family ordering and allowed-family selection.
- Added a bounded generic lexical table-row candidate source for structured table/troubleshooting lookups. It scores table rows by salient query-term overlap, table row-group evidence, troubleshooting field names, and exact prompted symptom/error phrase, while using the original request filters rather than metadata-selected document IDs so inferred document selection cannot become a hard filter.
- Added focused unit coverage in `tests/unit/test_retriever.py` for lexical row-group scoring, general-query skip behavior, and structured lookup family precedence when the query also says `how`.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py -q` -> 49 passed, 1 warning.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 187 passed, 58 warnings in 151.19s.
- Diagnostic eval attempts before the final family-precedence fix: `retrieval_eval_20260824_072809` and `retrieval_eval_20260824_073220` both remained 14/20 (70%), confirming lexical candidates alone were not enough while table candidates were excluded by family selection.
- Live eval command (saved multi-step bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_eval_dataset_20260824_062714.jsonl --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved multi-step result: `retrieval_eval_20260824_073510` improved from 14/20 (70%) to 20/20 (100%), pass@1/pass@3/pass@5 all 100%, candidate recall 100%, metadata-document recall 100%, failure categories `{}`. All 20 saved sibling troubleshooting rows passed at rank 1.
- Live eval command (saved single-step regression bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved single-step result: `retrieval_eval_20260824_073935` improved from 35/40 (87.5%) to 37/40 (92.5%), pass@1 80%, pass@3 90%, pass@5 92.5%, candidate recall 100%, metadata-document recall 100%, failure categories `ranking_or_context_loss: 3`.
- New artifacts tracked in manifest: `test_reports/retrieval_eval_summary_20260824_073510.json`, `test_reports/retrieval_eval_manifest_20260824_073510.json`, `test_reports/retrieval_eval_summary_20260824_073935.json`, `test_reports/retrieval_eval_manifest_20260824_073935.json`.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus the new successful eval summary/manifest artifacts.

Next target:

- Broaden multi-step coverage beyond sibling troubleshooting rows, including procedure-plus-spec, warning-plus-step, and cross-document engineering questions; continue reducing the remaining single-step ranking/context losses.

## 2026-08-24 Cron 39262386 Contextual Procedure Multi-Step

- Target: broaden multi-step retrieval coverage beyond sibling troubleshooting tables with procedure-plus-section evidence questions.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Changed eval logic in `packages/evals/src/manuals_rag_evals/retrieval_eval.py` to generate deterministic `contextual_procedure_plus_section_evidence` cases from a procedure chunk plus linked same-section support evidence such as setup constraints, ports, menu labels, or communication details. The generator now rejects numeric-only procedure subjects and requires local-context linkage before treating same-section support as multi-step evidence.
- Changed benchmark logic in `scripts/benchmark/run_large_retrieval_eval.py` to add `--multi-step-case-family all|sibling_table_rows|contextual_section`, so cron can measure the broader contextual family separately from the existing sibling table-row bank.
- Added focused unit coverage in `tests/unit/test_retrieval_eval.py` for contextual procedure-plus-section case generation.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py -q` -> 34 passed.
- Diagnostic live eval before tightening contextual linkage: `retrieval_eval_20260824_075859` completed 20 contextual cases at 18/20 passed (90%), but failure inspection showed one unrelated same-page camera table pair and one numeric-only procedure label, so the dataset was not added to the durable question-bank manifest.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --retrieval-task multi_step_retrieval --multi-step-case-family contextual_section --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45 --disable-llm-query-generation`.
- Live eval result: refined contextual multi-step bank `retrieval_eval_20260824_080232` completed 20 procedure-plus-section questions at 17/20 passed (85%), all passing cases at rank 1. Candidate recall was 85%, metadata-document recall 88.89%, metadata-document rank-1 rate 77.78%, and failures were `candidate_miss: 3`.
- Failure evidence: all 3 misses clustered in `AS_128241_LJ-X8000_SG_D48GB_WW_GB_2072_1.pdf` image-capture timing/setup-guide cases; top retrieval chose XG-X or CV-X evidence, so the next retrieval target is generic contextual multi-step document/candidate selection for terse product-family setup-guide queries.
- New artifacts: `test_reports/retrieval_eval_dataset_20260824_080232.jsonl`, `test_reports/retrieval_eval_results_20260824_080232.jsonl`, `test_reports/retrieval_eval_summary_20260824_080232.json`, `test_reports/retrieval_eval_manifest_20260824_080232.json`. The refined 20-case dataset is now tracked in the question-bank manifest.
- Question-bank manifest now tracks 120 exploratory questions: 80 single-step and 40 multi-step.
- Changed files: `packages/evals/src/manuals_rag_evals/retrieval_eval.py`, `scripts/benchmark/run_large_retrieval_eval.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus the new 08:02 UTC eval summary/manifest artifacts.

Next target:

- Improve contextual multi-step candidate recall for procedure-plus-section questions, especially LJ-X8000 setup-guide image-capture timing cases where metadata document selection chooses XG-X/CV-X evidence; then add warning-plus-step and cross-document multi-step cases.

## 2026-08-24 Cron 39262386 Contextual Lexical Candidates

- Target: improve contextual multi-step candidate recall for procedure-plus-section questions, especially LJ-X8000 setup-guide image-capture timing cases where retrieval drifted to XG-X/CV-X image-capture prose.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Changed query analysis in `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py` so letter-number family names such as `X8000 Series` are recognized as product families rather than error codes.
- Changed retrieval logic in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` to add a bounded contextual lexical candidate source for operational/procedure/configuration questions. It scores procedure, atomic, and section-window chunks using salient query terms plus `local_rerank_context`, honors explicit request filters, and only anchors the supplemental scan on product-like terms when the query includes them.
- Adjusted operational-flow family ordering to prefer section context and procedure evidence before short prose, and changed family selection so one primary family cannot consume the entire rerank candidate budget.
- Added focused unit coverage in `tests/unit/test_retriever.py` for letter-number family parsing, product-identifier scoring, and contextual lexical section-context scoring.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py -q` -> 52 passed, 1 warning.
- Live eval command (saved contextual multi-step bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_eval_dataset_20260824_080232.jsonl --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved contextual multi-step result: `retrieval_eval_20260824_083633` improved from 17/20 (85%) to 18/20 (90%), pass@1 85%, pass@5 90%, candidate recall 90%, metadata-document recall 88.89%, failures `candidate_miss: 2`. Two LJ-X8000 setup-guide cases that previously missed now passed; remaining misses are one LJ-X8000 high-speed image-capture case and one XG-X contextual case.
- Live eval command (saved single-step regression bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved single-step result: `retrieval_eval_20260824_083824` held at 37/40 (92.5%), pass@1 77.5%, pass@3 87.5%, pass@5 92.5%, candidate recall 97.5%, metadata-document recall 97.37%, failures `ranking_or_context_loss: 2`, `wrong_document_or_filter_loss: 1`. Pass rate did not regress, but candidate/document recall was lower than the previous 100% run and should be watched next.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py`, `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval artifacts for the 08:36 and 08:38 UTC runs.

Next target:

- Fix the remaining contextual candidate misses without harming the saved single-step bank, especially the LJ-X8000 high-speed image-capture support case and the single-step IV4 document-selection miss introduced in the regression run; then add warning-plus-step and cross-document multi-step cases.

## 2026-08-24 Cron 39262386 Inferred Filter Cleanup

- Target: fix the remaining contextual candidate misses without harming the saved single-step bank, especially the XG-X menu-label contextual case and IV4 numbered-prefix model parsing.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Failure review: prior saved contextual run `retrieval_eval_20260824_083633` was 18/20 (90%) with two `candidate_miss` failures. One failed XG-X trigger-settings prompt included `[Step 2/3 Trigger Settings]`; `build_filters` converted that inferred menu label into a hard filter and excluded the exact XG-X evidence. The saved single-step run `retrieval_eval_20260824_083824` remained 37/40 (92.5%) with an IV4-G600CA PDO table wrong-document miss, and query analysis did not parse numbered-prefix models such as `IV4-G600CA`.
- Changed retrieval logic in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` so `build_filters` preserves explicit request filters and `is_active` only; inferred query signals such as menu labels remain available to scoring/query analysis but no longer become hard filters. Contextual lexical scoring now gives modest generic credit for matching product-family terms and slightly stronger procedure/section/atomic context evidence.
- Changed query analysis in `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py` so product-model extraction recognizes numbered prefixes such as `IV4-G600CA` while still rejecting non-numeric family strings like `XG-X` as product models.
- Added focused unit coverage in `tests/unit/test_filters.py` and `tests/unit/test_retriever.py` for inferred menu-label filter avoidance, numbered-prefix model parsing, and product-family contextual lexical scoring.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py tests/unit/test_filters.py -q` -> 63 passed, 1 warning.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 194 passed, 58 warnings in 149.76s.
- Live eval command (saved contextual multi-step bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_eval_dataset_20260824_080232.jsonl --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved contextual multi-step result: `retrieval_eval_20260824_085757` improved from 18/20 (90%) to 19/20 (95%), pass@1/pass@3 90%, pass@5 95%, candidate recall 95%, metadata-document recall 90%, failure categories `candidate_miss: 1`. The XG-X `[Step 2/3 Trigger Settings]` case now passes at rank 1; the remaining miss is the LJ-X8000 high-speed image-capture support case.
- Live eval command (saved single-step regression bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved single-step result: `retrieval_eval_20260824_085932` held at 37/40 (92.5%), pass@1 77.5%, pass@3 87.5%, pass@5 92.5%, candidate recall 97.5%, metadata-document recall 97.5%, failure categories `ranking_or_context_loss: 2`, `wrong_document_or_filter_loss: 1`. The IV4-G600CA PDO row remains a wrong-document/filter-loss case despite the generic model parser fix.
- New artifacts tracked in manifest: `test_reports/retrieval_eval_summary_20260824_085757.json`, `test_reports/retrieval_eval_manifest_20260824_085757.json`, `test_reports/retrieval_eval_summary_20260824_085932.json`, `test_reports/retrieval_eval_manifest_20260824_085932.json`.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/query_analysis.py`, `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `tests/unit/test_filters.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for the 08:57 and 08:59 UTC runs.

Next target:

- Fix the remaining LJ-X8000 high-speed image-capture contextual evidence miss and the IV4-G600CA PDO wrong-document table miss with generic metadata/table ranking improvements, then add warning-plus-step and cross-document multi-step cases.

## 2026-08-24 Cron 39262386 Contextual Pre-Limit Ordering

- Target: fix the remaining LJ-X8000 high-speed image-capture contextual evidence miss and recheck the IV4-G600CA PDO single-step wrong-document miss without adding document-specific routing or query-derived hard filters.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Failure review: prior contextual run `retrieval_eval_20260824_085757` was 19/20 (95%) with one `candidate_miss`; the missed LJ-X8000 evidence was discoverable by contextual lexical scoring but could be lost before scoring because the SQL pre-limit scan had no deterministic relevance ordering. Prior single-step run `retrieval_eval_20260824_085932` was 37/40 (92.5%) with the IV4-G600CA PDO row still reported as `wrong_document_or_filter_loss`.
- Changed retrieval logic in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` so contextual lexical candidate scans order rows before `LIMIT` using generic product-term matches plus content/local-context query-term matches, then continue through the existing Python scorer/reranker. The ordering is bounded to a small term set to limit added SQL cost.
- Added focused unit coverage in `tests/unit/test_retriever.py` to assert contextual lexical scans include deterministic relevance ordering and product-term metadata matching.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py -q` -> 54 passed, 1 warning.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 194 passed, 58 warnings in 150.83s.
- Live eval command (saved contextual multi-step bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_eval_dataset_20260824_080232.jsonl --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved contextual multi-step result: `retrieval_eval_20260824_093050` held at 19/20 (95%) with pass@1/pass@3/pass@5 all 95%, candidate recall 95%, metadata-document recall 89.47%, and failure categories `eval_timeout: 1`. The prior LJ-X8000 high-speed image-capture candidate miss now passes at rank 1; the remaining failure is an XG-X contextual case brushing the 8-second timeout.
- Live eval command (saved single-step regression bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved single-step result: `retrieval_eval_20260824_093241` held at 37/40 (92.5%), pass@1 77.5%, pass@3 87.5%, pass@5 92.5%, candidate recall 97.5%, metadata-document recall improved to 100%, and failures were `eval_timeout: 1` plus `ranking_or_context_loss: 2`. The IV4-G600CA PDO row now passes at rank 1.
- Timeout evidence: two useful saved-bank evals now have one scored `eval_timeout` each under the 8-second per-query ceiling. Future runs should reduce contextual/query latency or consider raising the per-query eval timeout before treating these as accuracy misses.
- New artifacts tracked in manifest: `test_reports/retrieval_eval_summary_20260824_093050.json`, `test_reports/retrieval_eval_manifest_20260824_093050.json`, `test_reports/retrieval_eval_summary_20260824_093241.json`, `test_reports/retrieval_eval_manifest_20260824_093241.json`.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for the 09:30 and 09:32 UTC runs.

Next target:

- Reduce contextual retrieval latency for XG-X cases brushing the 8-second timeout, then add warning-plus-step and cross-document multi-step cases while continuing to reduce remaining single-step ranking/context losses.

## 2026-08-24 Cron 39262386 Warning-Step Multi-Step Coverage

- Target: add the first warning-plus-step multi-step retrieval cases while staying within retrieval/eval logic and avoiding document-specific routing or filename heuristics.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Changed eval logic in `packages/evals/src/manuals_rag_evals/retrieval_eval.py` to add deterministic `warning_plus_step_evidence` multi-step cases. The generator pairs warning/caution records with nearby same-document operational step evidence, preferring procedure records but allowing action-shaped atomic step chunks because this corpus stores many installation steps as `atomic_text`.
- Changed benchmark logic in `scripts/benchmark/run_large_retrieval_eval.py` to accept `--multi-step-case-family warning_step`.
- Added focused unit coverage in `tests/unit/test_retrieval_eval.py` for warning/step neighborhood case generation and expected multi-evidence chunk ids.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py -q` -> 35 passed.
- Diagnostic live evals before relaxing the warning-side gate produced zero cases: `retrieval_eval_20260824_095614` and `retrieval_eval_20260824_095745`. The corpus warning records are often short headings or status fields, so the final generator uses warning-specific subject validation plus TOC/legal filters instead of the general `chunk_is_queryworthy` gate.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --retrieval-task multi_step_retrieval --multi-step-case-family warning_step --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45 --disable-llm-query-generation`.
- Live eval result: first warning-plus-step bank `retrieval_eval_20260824_095925` generated 18 validated multi-step questions and passed 10/18 (55.56%), pass@1 44.44%, pass@3/pass@5 55.56%, candidate recall 55.56%, metadata-document recall 100%, metadata-document rank-1 rate 60%. All 8 failures were `eval_timeout` under the 8-second per-query limit.
- New artifacts tracked in manifest: `test_reports/retrieval_eval_dataset_20260824_095925.jsonl`, `test_reports/retrieval_eval_results_20260824_095925.jsonl`, `test_reports/retrieval_eval_summary_20260824_095925.json`, `test_reports/retrieval_eval_manifest_20260824_095925.json`.
- Question-bank manifest now tracks 138 exploratory questions: 80 single-step and 58 multi-step.
- Changed files: `packages/evals/src/manuals_rag_evals/retrieval_eval.py`, `scripts/benchmark/run_large_retrieval_eval.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for 09:59 UTC.

Next target:

- Reduce contextual/warning-step retrieval latency that causes 8-second eval timeouts, then rerun the warning-plus-step bank and broaden to cross-document multi-step cases.

## 2026-08-24 Cron 39262386 Safety Route Pruning

- Target: reduce warning-plus-step retrieval latency that caused 8-second eval timeouts without changing model settings, ingestion, UI, schema, or adding document-specific routing.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Failure review: prior warning-plus-step run `retrieval_eval_20260824_095925` was 10/18 (55.56%) with all 8 failures scored as `eval_timeout`. A stage-timing probe on a timeout-shaped warning query showed table-only search was avoidable for safety/procedure intent and overlapping special safety/how-to routes spent about 7.7s before rerank.
- Changed retrieval logic in `packages/retrieval/src/manuals_rag_retrieval/retriever.py` so table-only dense/sparse search only runs when query analysis says table/spec/structured evidence is plausible. Safety/procedure/context questions still use dense, sparse, contextual lexical, and special routes.
- Changed special route planning so safety questions keep the warning/procedure route but do not also run the broader how-to procedure/section route. Non-safety how-to/configuration questions keep the existing procedure/section route.
- Added focused unit coverage in `tests/unit/test_retriever.py` for skipping table search on safety procedure questions, preserving table search for structured value questions, and avoiding duplicate special how-to routes for safety questions.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retriever.py -q` -> 57 passed, 1 warning.
- Full unit tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit -q` -> 198 passed, 58 warnings in 150.92s.
- Diagnostic warning-step eval after only skipping table search stayed at 10/18 (55.56%), confirming the overlapping special route was the larger bottleneck.
- Live eval command (saved warning-step multi-step bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_eval_dataset_20260824_095925.jsonl --max-queries 18 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved warning-step result: `retrieval_eval_20260824_103005` improved from 10/18 (55.56%) to 11/18 (61.11%), pass@1 50%, pass@3/pass@5 61.11%, candidate recall 66.67%, metadata-document recall 100%, and timeout failures dropped from 8 to 6. One former timeout now completes as `ranking_or_context_loss`, giving a concrete next ranking target.
- Live eval command (saved single-step regression bank): `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --dataset-path test_reports/retrieval_accuracy_question_bank_20260824_045817.jsonl --max-queries 40 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45`.
- Saved single-step result: `retrieval_eval_20260824_103510` completed at 36/40 (90%), pass@1 75%, pass@3 85%, pass@5 90%, candidate recall 95%, metadata-document recall 97.44%, failures `eval_timeout: 1`, `ranking_or_context_loss: 2`, `wrong_document_or_filter_loss: 1`. This is below the prior 37/40 run, so keep it as a watch item; the failures are the existing VS/IV4 timeout/ranking/document-selection cluster and the changed safety routing should not apply to structured table questions.
- New artifacts tracked in manifest: `test_reports/retrieval_eval_summary_20260824_103005.json`, `test_reports/retrieval_eval_manifest_20260824_103005.json`, `test_reports/retrieval_eval_summary_20260824_103510.json`, `test_reports/retrieval_eval_manifest_20260824_103510.json`.
- Changed files: `packages/retrieval/src/manuals_rag_retrieval/retriever.py`, `tests/unit/test_retriever.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for 10:30 and 10:35 UTC.

Next target:

- Continue warning-step latency reduction by profiling the remaining safety special route and rerank inputs, then improve the newly exposed warning-step ranking/context loss and restore the saved single-step bank to at least 37/40 before broadening to cross-document multi-step cases.

## 2026-08-24 Cron 39262386 Refined Warning-Step Questions

- Target: refine warning-plus-step multi-step eval coverage so it uses realistic engineer wording before making further retrieval changes.
- Local stack: compose services were up; API, Postgres, Qdrant, Redis, workers, and UI were running. The existing `manuals_vendor_keyence` corpus remained usable with 53 indexed documents.
- Changed eval logic in `packages/evals/src/manuals_rag_evals/retrieval_eval.py` so warning-step generation skips prohibition/status snippets as operational steps, avoids generic labels such as `User's Manual (3D mode)` when a product family is available, and asks `What warning or caution ... applies when ...?` instead of pasting long step snippets after `When`.
- Added focused unit coverage in `tests/unit/test_retrieval_eval.py` for the refined question shape, generic manual-label fallback, and prohibition skipping.
- Focused tests: `docker compose -f infra/compose/docker-compose.yml exec -T api python -m pytest tests/unit/test_retrieval_eval.py -q` -> 36 passed.
- Live eval command: `docker compose -f infra/compose/docker-compose.yml exec -T api python scripts/benchmark/run_large_retrieval_eval.py --existing-corpus-id manuals_vendor_keyence --retrieval-task multi_step_retrieval --multi-step-case-family warning_step --max-queries 20 --search-mode direct --per-query-timeout-seconds 8 --warmup-queries 1 --warmup-timeout-seconds 45 --disable-llm-query-generation`.
- Live eval result: refined warning-step bank `retrieval_eval_20260824_105701` generated 15 validated multi-step questions and passed 8/15 (53.33%), pass@1 46.67%, pass@3/pass@5 53.33%, candidate recall 53.33%, metadata-document recall 100%, and failure categories `eval_timeout: 7`. The new dataset is tracked in the manifest, bringing the question bank to 153 exploratory questions: 80 single-step and 73 multi-step.
- Diagnostic retrieval experiment: a smaller safety/procedure rerank pool was tested and reverted after `retrieval_eval_20260824_110038` dropped to 7/15 (46.67%) with `eval_timeout: 8`; no retrieval logic from that experiment was kept.
- Changed files: `packages/evals/src/manuals_rag_evals/retrieval_eval.py`, `tests/unit/test_retrieval_eval.py`, `test_reports/retrieval_accuracy_progress.md`, `test_reports/retrieval_accuracy_question_bank_manifest.json`, plus new eval summary/manifest artifacts for 10:57 and 11:00 UTC.

Next target:

- Profile and reduce IV4 warning/status safety-route latency causing 8-second warning-step eval timeouts, then rerun the refined warning-step bank and the saved single-step regression bank.
