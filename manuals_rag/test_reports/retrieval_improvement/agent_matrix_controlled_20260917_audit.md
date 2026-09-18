# Controlled frozen 48-case matrix audit — 2026-09-17/18

## Coverage and integrity

- Frozen dataset: `tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl`
- SHA-256: `51028f6b5ea6b09515ac361d2b747feb998aaa5098dca710c7e6f4a972871a65`
- The three durable segments contain 48 unique case IDs in exactly the frozen
  dataset order; an ID-sequence diff is empty.
- Segment 1: offset 0, 1 completed case, exit 1 after an embedding HTTP timeout.
- Segment 2: offset 1, 22 completed cases, exit 1 after another embedding HTTP
  timeout.
- Segment 3: offset 23, 25 completed cases, exit 0.
- No completed case was rerun during recovery.

## Raw matrix result

| Backend | Full pass | Candidate recall | Document retention | Hop dependencies | Evidence sufficiency | Grounded answer | Latency/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LangGraph | 12/48 | 28/48 | 41/48 | 39/48 | 18/48 | 15/48 | 18/48 |
| LlamaIndex | 10/48 | 29/48 | 42/48 | 36/48 | 18/48 | 15/48 | 18/48 |

These raw pass rates are not an acceptance result. The run was materially
contaminated by inference infrastructure failure:

- 49 verifier exceptions were recorded: 43 `ReadTimeout`, four malformed JSON
  responses, and two schema-validation errors.
- NVIDIA discovery reported `GPU0: Unknown Error`; the verifier model was loaded
  with `size_vram=0`, so calls fell back to CPU and repeatedly exceeded the
  verifier and latency budgets.
- A verifier timeout fails closed before later parallel or dependent hops can be
  certified. Candidate/document misses after that point cannot be classified as
  raw retrieval misses from this run.

## Adjudication

1. **Frozen oracle artifacts remain artifacts.** The nine generated warning
   questions combine truncated source text with an unrelated controller-mounting
   warning and demand a synthetic dependency. Two of the final three warning
   cases retained both expected chunks in both backends but correctly did not
   invent the expected dependency. Keep these cases frozen and reported; do not
   tune retrieval to their malformed shape.
2. **The CV-X/LJ-X `Standard Angle` case is ambiguous.** Both backends retained
   both expected documents/chunks and failed closed because the same
   unqualified setting label has multiple definitions. This matches the separate
   source-audited oracle probe and is an expected oracle failure.
3. **Structured lookups expose verifier failure.** Four of five final structured
   lookup cases retained the exact expected chunk in both backends, then timed
   out in independent verification and abstained. The fifth retained the target
   document but needs an affected-case rerun before deciding whether its missing
   chunk is a retrieval defect.
4. **Cross-document failures are not yet clean retrieval evidence.** Most
   cross-document cases stopped on a verifier timeout in the first branch, so a
   missing second document/chunk is downstream of the timeout. Rerun after
   stable verifier inference before changing retrieval.
5. **Real dependency planning is repaired.** The RS-232C cable case planned as
   dependent with two hops and one dependency edge in both backends. Its hop-1
   verifier timed out, so hop 2 did not run. The separate CA-EN100U dependency
   probe completed both hops and grounded both required claims before the GPU
   failure.
6. **Unanswerable behavior stayed fail-closed.** Both backends returned
   insufficient evidence with no citations for the invented ZX-9999 value. Its
   remaining failures were expected-tool scoring and CPU-fallback latency.

## Action

- Added `OLLAMA_RETRIEVAL_VERIFIER_NUM_BATCH=64` and passed it to verifier
  inference, matching the already-proven metadata stabilization.
- Verification: 501 affected tests passed; the full unit suite passed 1044 tests
  with 77 warnings; compose configuration validated.
- Kept T8 open. Once GPU discovery is healthy, rerun the affected structured,
  cross-document, cable-dependency, and unanswerable cases first. Only then rerun
  the full frozen matrix and compare a clean report.
- Production and agentic retrieval remain off.
