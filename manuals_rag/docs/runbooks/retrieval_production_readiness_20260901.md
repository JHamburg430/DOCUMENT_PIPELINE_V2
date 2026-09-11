# Retrieval Production Readiness Review — 2026-09-01

## Scope

This review covers the Eval Matrix answer-mode pipeline, retrieval-stage retention,
retrieval scoring, hybrid candidate generation, family selection, reranking, and
final context assembly for the two currently loaded 20-question datasets.

## Baseline

- Completed job: `matrix-17d54bbcc4d1`
- Artifacts:
  - `test_reports/retrieval_eval_results_20260901_202524_f68ec0.jsonl`
  - `test_reports/retrieval_eval_results_20260901_203404_8c0472.jsonl`
- Cases: 40 single-step questions from two generated datasets.
- Displayed retrieval pass: 9/40 (22.5%).
- Answer stages attempted: 9/40; answer pass: 7/9 attempted.
- Expected-document retention in recorded stage windows:
  - dense: 38/40
  - sparse: 34/40
  - special: 20/40
  - fusion: 37/40
  - rerank: 40/40
  - final context: 40/40

## Root causes

### 1. Eval transport/scorer contract mismatch (critical)

The streaming `assemble_context` event serialized only `content_preview` and
removed full content, context, and metadata. The retrieval scorer read `content`
but ignored `content_preview`. Answer-bearing results therefore scored as misses
even when the displayed preview visibly contained the expected answer.

Replaying the same artifacts after making preview-only records replayable raises
retrieval pass from 9/40 to 40/40. This is an evaluation correctness repair, not
a claim that ranking itself improved by that amount.

### 2. Exact-anchor bias (high)

Generated cases are anchored to one derived chunk. Manuals often contain the same
answer in an atomic chunk, section window, structured record, or duplicated model
manual. Exact chunk/type matching therefore produced false negatives despite
answer-equivalent evidence. Scoring now accepts:

- same-document material snippet evidence with query agreement; and
- strong cross-document duplicate evidence for queries that do not explicitly
  bind a conflicting product identifier.

The thresholds deliberately require material source-answer overlap so headings or
mere product mentions do not pass.

### 3. Generic title/heading dominance in reranking (high)

Query alignment counted question scaffolding and generic manual words such as
`when`, `series`, `vision`, and `system`. This over-rewarded title/TOC fragments.
For the VS archiving question, dense retrieval found archive material but final
context was dominated by generic VS overview/preparation headings.

The retrieval repair:

- removes generic/question-only words from alignment scoring while retaining a
  separate product-identifier score;
- adds lightweight inflection normalization (`start`/`starts`);
- applies semantic-completeness evidence in the final cross-encoder blend; and
- enables contextual lexical candidates for general answer-seeking prose queries,
  not only procedure/configuration queries.

### 4. Unbounded final-answer generation (high)

Final answer calls used `num_predict=-1`, so a model that continued emitting
tokens could keep an eval row in generation indefinitely despite the HTTP read
timeout (the timeout resets whenever another stream token arrives). A fresh
verification run exposed this on row 2. Final answers now default to a bounded
1,024-token budget through `OLLAMA_ANSWER_NUM_PREDICT`; the existing validation
fallback remains responsible for producing a grounded answer if model JSON is
invalid or incomplete.

## Verification gates

Completion requires all of the following:

1. Focused scorer, API-debug, UI-server, and retriever unit suites pass.
2. The full unit suite passes.
3. A loaded API/UI stack completes a new all-bank answer-mode matrix run.
4. Every new retrieval failure is inspected against expected and retrieved text;
   no preview/anchor false negative is counted as a genuine miss.
5. Stage totals, final retrieval totals, and answer-attempt totals reconcile.
6. Changed UI/API behavior is checked in the running Eval Matrix, including live
   cell updates and completed job persistence.

## Final verification

- A complete answer-mode run finished before the final retrieval repair with
  40/40 retrieval passes and 32/40 answer passes. The answer misses were kept
  separate from retrieval scoring; several were benchmark-term artifacts, and
  the broad VS archiving question was identified as ambiguous because its
  generated wording dropped the named Vision Dashboard condition.
- The affected API/debug, answer, eval, retrieval, and UI suites passed 366
  tests after the matrix row-key and scoring repairs.
- The final full unit gate passed 469 tests with 59 warnings.
- Final loaded-stack retrieval acceptance job: `matrix-9b6fc098063b`.
  - Status: completed; return code 0; two datasets finalized.
  - Retrieval: 40/40 passed, with no failure category emitted.
  - Matrix identity: 40 displayed rows and 40 unique dataset-qualified row
    keys, despite the datasets sharing the same 20 raw case IDs.
  - The IV-500C prohibited-installation-location case passed at rank 1 in both
    datasets after retaining exact-model, query-aligned dense evidence through
    fusion and family selection.
  - Results:
    `test_reports/retrieval_eval_results_20260902_024343.jsonl` and
    `test_reports/retrieval_eval_results_20260902_024600.jsonl`.
  - Summaries:
    `test_reports/retrieval_eval_summary_20260902_024343.json` and
    `test_reports/retrieval_eval_summary_20260902_024600.json`.

## Current limitation

The reset/currently loaded matrix bank contains only single-step questions. The
repository's multi-step, comparison, procedure, safety, structured-table, and
answer-grounding behavior remains covered by unit and historical curated gates,
but a future official production bank should restore curated multi-step slices as
active matrix rows rather than relying on generated single-step datasets alone.
