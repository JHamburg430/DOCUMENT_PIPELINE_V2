# Bounded multi-hop retrieval

Manuals RAG exposes two agentic retrieval backends behind the same controller:

- LangGraph state graph
- LlamaIndex Workflow

The shared controller makes their behavior directly comparable. It plans
single, parallel, or dependent hops; routes each hop to hybrid, structural,
sparse, dense, or broad retrieval; records an evidence ledger; judges evidence
sufficiency; and permits one bounded broad recovery per insufficient required
hop. The loop stops when all required evidence is sufficient, the hop budget is
exhausted, or no hop can run.

Dependent hops use evidence discovered by prior hops. Concrete identifiers not
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
retrieval completion, answer generation, and terminal success or failure.

## Live Agent Lab

Open the web console at `http://127.0.0.1:8601`, select **Agent Lab**, enter a
multi-hop question, and choose LangGraph, LlamaIndex, or **Compare both**. The
comparison view starts both backends concurrently and displays their plans,
executed queries, strategy changes, per-hop evidence, sufficiency decisions,
stop reasons, timings, citations, final answers, and event logs while they run.

## Comparison runner

Run both agent backends against the same frozen plan and query refinements:

```bash
python scripts/benchmark/compare_agentic_retrieval.py \
  --dataset tests/fixtures/agentic_dependent_retrieval_eval.jsonl \
  --corpus-id manuals_vendor_keyence \
  --max-hops 4 \
  --output /tmp/agentic-retrieval-comparison.json
```

Use `--no-llm` for the deterministic planner/refiner fallback. The report keeps
baseline, LangGraph, and LlamaIndex retrieval accuracy, sufficiency, latency,
completed-hop count, result-document coverage, and result equivalence separate.
