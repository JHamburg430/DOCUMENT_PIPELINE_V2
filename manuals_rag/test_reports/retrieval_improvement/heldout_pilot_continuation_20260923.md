# Held-out evaluation continuation — 2026-09-23

## Outcome

The held-out evaluation boundary now excludes every source document referenced
by the 48-case tuning matrix before question generation. A fail-closed freezer
then verifies exact expected snippets against active persisted chunks, checks
document/version identity, rejects tuning overlap, labels assistant verification
honestly (`human_reviewed=false`), and records immutable hashes and case order.

The first frozen pilot contains one source-verified case from a document outside
all 10 tuning documents. It is a contract probe, not a production-quality bank;
the production policy still requires at least 200 cases.

## Clean contract smoke

- Final run: `heldout-pilot-contract-20260923-04`
- Source: clean revision `561a7e9`
- Dataset SHA-256: `43960bae5bc355cef6bf2897760292fce31ab5eebdd15ecd7c0e050ebc15fa25`
- Artifact: complete, exit 0, provenance complete, exact case order reconciled
- Baseline retrieval: passed; answer-bearing source was retained
- LangGraph: all seven required matrix cells passed at 27.8 s
- LlamaIndex: all seven required matrix cells passed at 25.9 s

The earlier `-02` smoke exposed a planner/verification alignment failure: both
agents retrieved the exact answer-bearing chunk but invented an unasked minimum
clearance requirement, rejected the valid source, and abstained without
citations. The repair now routes a simple qualitative `why` question as one
exact hybrid lookup and deterministically confirms only a scoped passage with
adequate query-term overlap plus explicit causal or risk language.

On the final clean run, both backends used one retrieval call, made zero model
calls, retained the equivalent source window `0b7f3363-f8ff-53f2-a8a6-dde3a71629b2`,
matched all four expected terms, and emitted one valid citation with no invalid
citations. An intermediate `-03` run had a 123.5 s LlamaIndex latency outlier;
the identical immutable repeat completed in 25.9 s, so the outlier is not used
as representative performance evidence.

## Acceptance result

**Rejected for bank size only.** The strict gate reports one blocker: 1 case is
below the 200-case production minimum. Both backend category, evidence,
citation, token, and latency gates pass for the pilot case.

## Bank expansion checkpoint

- A 25-case generation smoke reviewed 114 candidate chunks across 12 held-out
  manuals before accepting 25 questions. This confirms that quality filtering,
  rather than raw generation throughput, is the limiting step.
- Source re-anchoring now binds quantitative model rows, input/output qualifiers,
  categorical standard rows, display-code causes, and multi-sentence procedures
  before freezing. The focused generation/freezer suite passes **178/178**.
- Frozen bank `heldout_retrieval_eval_pilot_v3_25.jsonl` contains 25 unique,
  source-verified cases from 9 held-out documents, has zero overlap with the 10
  tuning documents, and has SHA-256
  `b5f4e7ed9a724e634eb464d06d4dffda10cda1e639f09995ee0a4843794f1c42`.
- An earlier 25-case freeze was quarantined after review found that one input
  voltage question had been anchored to an output-voltage row. It is not valid
  evaluation evidence and was not reused.
- This is still assistant source verification (`human_reviewed=false`) and only
  25/200 required cases. No 25-case agent matrix was launched because the bank
  is incomplete and the acceptance run would be knowingly non-final.

## Decision

Keep deterministic hybrid/structural retrieval as the default and keep agentic
production retrieval disabled. The pilot contract is now sound, but one case
cannot establish production accuracy or safety. The next bounded milestone is
to expand the same source-verified, document-disjoint bank to at least 200 cases
across required categories, then rerun the unchanged acceptance gate.
