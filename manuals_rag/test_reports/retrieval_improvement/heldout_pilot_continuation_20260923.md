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

- Run: `heldout-pilot-contract-20260923-02`
- Source: clean revision `f79530310e6dd8ed94cbb15af2fd0d9083edeb26`
- Dataset SHA-256: `43960bae5bc355cef6bf2897760292fce31ab5eebdd15ecd7c0e050ebc15fa25`
- Artifact: complete, exit 0, provenance complete, exact case order reconciled
- Baseline retrieval: passed; answer-bearing source was retained
- LangGraph: retrieval/document retention passed; full matrix failed
- LlamaIndex: retrieval/document retention passed; full matrix failed

Both agents retrieved the exact answer-bearing chunk. Their model-generated
requirements then changed the question from *why sensors should not be installed
near a moving robotic arm* into an unsupported request for a quantified minimum
clearance distance. Verification correctly rejected that invented requirement,
so both paths abstained and emitted no citations. This is an agent planning and
claim-verification alignment failure, not a document-discovery failure.

The token gate also failed: LangGraph used 5,983 measured tokens and LlamaIndex
used 16,404 against the 4,000-token policy. Latencies were 54.4 s and 95.3 s,
respectively, within the 120 s ceiling but too expensive for a simple single-hop
question.

## Acceptance result

**Rejected.** Blockers:

1. 1 case is below the 200-case production minimum.
2. Both backends fail the `single_hop_control` category gate.
3. Both backends fail evidence sufficiency and claim/citation correctness.

## Decision

Keep deterministic hybrid/structural retrieval as the default and keep agentic
production retrieval disabled. Before generating the full bank, improve candidate
chunk ranking (the model rejected seven thin/visual chunks before finding one
usable source) and repair planner requirement preservation so it cannot add an
unasked quantitative constraint. Then expand the same frozen, document-disjoint
bank and rerun the gate.
