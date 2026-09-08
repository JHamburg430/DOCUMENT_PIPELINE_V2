# Bounded multi-hop retrieval

Manuals RAG exposes two agentic retrieval backends over the same retrieval-tool contract:

- LangGraph state graph
- LlamaIndex Workflow

The controllers are intentionally independent. LangGraph uses an explicit state
graph with fixed plan dependencies and broad fallback recovery. LlamaIndex uses
subquestion decomposition, per-subquestion query transformation, query-engine
tool routing, and alternate-tool recovery. They share only the underlying
hybrid, structural, sparse, dense, and broad retrieval tools; evidence/result
schemas; budgets; and evaluation contract.

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
sufficiency assessments.

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

The **Agent Evaluation Matrix** below the live comparison runs a dataset through
both policies and uses one row per question. Its columns score tool selection,
candidate recall, document retention, hop/dependency correctness, evidence
sufficiency, grounded-answer correctness, and latency/token cost. Candidate
recall and final-context retention are deliberately separate so context loss is
not misdiagnosed as a search failure.

## Comparison runner

Run both agent backends independently against the same questions and budgets:

```bash
python scripts/benchmark/compare_agentic_retrieval.py \
  --dataset tests/fixtures/agentic_dependent_retrieval_eval.jsonl \
  --corpus-id manuals_vendor_keyence \
  --max-hops 4 \
  --output /tmp/agentic-retrieval-comparison.json
```

Use `--no-llm` for each backend's deterministic fallback policy. The report keeps
baseline, LangGraph, and LlamaIndex retrieval accuracy, sufficiency, latency,
completed-hop count, result-document coverage, answers, per-layer agent matrix
scores, and result equivalence separate.
