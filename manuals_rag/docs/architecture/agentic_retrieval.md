# Bounded multi-hop retrieval

Manuals RAG exposes two agentic retrieval backends over the same retrieval-tool contract:

- LangGraph state graph
- LlamaIndex Workflow

The controllers are intentionally independent. LangGraph uses an explicit state
graph with explicit plan dependencies and gap-directed backtracking. LlamaIndex uses
subquestion decomposition, per-subquestion query transformation, query-engine
tool routing, query transformations, and alternate-query-engine recovery. They share only the underlying
hybrid, structural, sparse, dense, and broad retrieval tools; evidence/result
schemas; budgets; and evaluation contract.

Neither backend may synthesize merely because its retrieval budget ended. Each
hop writes a claim-level evidence ledger entry that names the supporting chunks,
documents, answer facets, dependency bindings, contradictions, and unresolved
gaps. A claim is supported only when one evidence item contains both its answer
facet and any dependency anchor. Aggregate keywords spread across unrelated
chunks do not satisfy the claim. The final context assembler reserves attributed
support for every required claim and rechecks coverage before synthesis; any
unresolved or dropped claim produces an explicit insufficient-evidence answer.

In both policies, dependent hops use evidence discovered by prior hops. Concrete identifiers not
already present in the dependency query become anchors for the next query and
switch generic hybrid retrieval to exact sparse retrieval. Sufficiency requires
the requested answer signal and the discovered anchor to occur in the same
evidence item. Successful recovery evidence is available to downstream hops.

## API

`POST /query` accepts:

```json
{
  "query": "Which cable connects the RS-232C port, then what is its connector orientation?",
  "corpus_ids": ["manuals_vendor_keyence"],
  "retrieval_orchestrator": "langgraph_agent",
  "max_retrieval_hops": 4
}
```

Set `retrieval_orchestrator` to `baseline`, `langgraph_agent`, or
`llamaindex_agent`. Agent responses include `retrieval_trace` with the plan,
completed hops, stop reason, per-hop strategy, queries, evidence chunk IDs, and
sufficiency assessments, context-retention decisions, recovery effectiveness,
and measured Ollama prompt/completion token and duration counters.

`POST /query/stream` accepts the same request for either agent backend and
returns newline-delimited JSON events as the run progresses. Events cover plan
creation, hop start/completion, evidence assessment, recovery scheduling,
tool selection, retrieval completion, answer generation, and terminal success
or failure. Every trace identifies its control policy.

## Live Agent Lab

Open the web console at `http://127.0.0.1:8601`, select **Agent Lab**, enter a
multi-hop question, and choose LangGraph, LlamaIndex, or **Compare both**. The
comparison view starts both backends concurrently and displays their plans,
executed queries, strategy changes, per-hop evidence, sufficiency decisions,
stop reasons, timings, citations, final answers, and event logs while they run.
Live comparisons are owned by the UI server rather than the browser tab. The
browser starts one background job and polls its retained event history, so a
reload or a second page reattaches to the same backend runs without starting
retrieval again. Inactive History and Ingestion views load lazily; opening the
console renders immediately and synchronizes only the active view.

The **Agent Evaluation Matrix** below the live comparison defaults to a frozen
48-question bank and runs it through
both policies and uses one row per question. Its columns score tool selection,
candidate recall, document retention, hop/dependency correctness, evidence
sufficiency, grounded-answer correctness, and latency/token cost. Candidate
recall and final-context retention are deliberately separate so context loss is
not misdiagnosed as a search failure.

Every case carries an expected-evidence graph. Its nodes define the claims,
expected chunks/documents/terms, and dependency edges; its outcome is either
answerable or an expected abstention. The bank covers single-hop controls,
parallel multi-part requests, dependent retrieval, cross-document requests,
entity resolution, exact structured lookups, and an unanswerable control.
Category summaries prevent a strong easy-case average from hiding weak
dependent or cross-document behavior.

The companion manifest freezes ten source documents as a **prospective**
document-held-out set: they may not drive question-specific patches after the
freeze. Some source cases existed before this contract, so the manifest does
not misrepresent them as historically unseen. A future bank version should add
newly ingested manuals for a strict never-before-seen document test.

## Comparison runner

Run both agent backends independently against the same questions and budgets:

```bash
python scripts/benchmark/compare_agentic_retrieval.py \
  --dataset tests/fixtures/agentic_retrieval_eval_matrix_v1.jsonl \
  --corpus-id manuals_vendor_keyence \
  --max-hops 6 \
  --output /tmp/agentic-retrieval-comparison.json
```

Use `--no-llm` for each backend's deterministic fallback policy. The report keeps
baseline, LangGraph, and LlamaIndex retrieval accuracy, sufficiency, latency,
completed-hop count, result-document coverage, answers, per-layer agent matrix
scores, recovery attempts/successes, per-category results, measured model cost,
the dataset SHA-256, and result equivalence separate. Rebuild the frozen bank
from its reviewed source datasets with
`python scripts/benchmark/build_agent_eval_bank.py`; the builder can read
deleted worktree sources from a Git ref without restoring or mutating them.
