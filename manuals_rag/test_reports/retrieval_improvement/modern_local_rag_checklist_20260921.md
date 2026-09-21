# Modern local RAG checklist — 2026-09-21

## Outcome

All five recommendations were investigated in order. The architecture remains
sound, but production enablement is still blocked by the strict acceptance gate.
No model/index default was changed solely on component-level evidence.

- [x] **1. Benchmark real BM25 against hashed sparse retrieval.**
  - Qdrant BM25 improved required-any@40 from 59.6% to 78.7%,
    required-all@40 from 53.2% to 66.0%, and mean latency from 57.0 ms to
    27.1 ms. Required-all@5 regressed from 38.3% to 23.4%.
  - Decision: keep indexed BM25 available but opt-in pending downstream fusion
    tuning and an answer-quality gate.
  - Evidence: `sparse_retriever_comparison_48_20260920.json`.

- [x] **2. Benchmark modern reranker and embedding candidates.**
  - Qwen3-Reranker-0.6B produced a small shallow-pool gain but was about 6x
    slower than MiniLM; required-all@12 was unchanged. Keep MiniLM.
  - Qwen3-Embedding-4B improved required-any@5 (68.8% to 75.0%) and
    required-all@20 (81.3% to 87.5%), but regressed required-all@5
    (56.3% to 50.0%), used 2,560-dimensional vectors rather than 1,024, and
    lacked full-corpus/downstream proof. Keep Qwen3-Embedding-0.6B.
  - Evidence: `reranker_*_evidence_v2_20260920.json` and
    `embedding_model_frozen_pool_20260921.json`.

- [x] **3. Add deterministic contextual prefixes.**
  - Verified this is already implemented for both dense and lexical text:
    title, manufacturer, product family/model, document kind, and section.
    Structured table chunks additionally carry row/column coordinates and
    row/column headers.
  - Decision: no reindex-only rewrite. Version labeling remains a candidate
    refinement that must first win a frozen comparison.

- [x] **4. Add visual page retrieval as a sidecar.**
  - Added an opt-in ColSmol page-image sidecar in an isolated Qdrant collection.
    It uses pooled-vector prefetch followed by bounded MaxSim multivector
    reranking, deterministic page IDs, document filters, and a dry-run/apply
    indexer. Production routing remains disabled.
  - Prior local benchmark: ColSmol-256M reached R@3 91.7% and R@5 100% on hard
    visual paraphrases. The small visual answer model reached only 1/6 exact, so
    visual answering remains fail-closed.
  - Current dry run resolved 2 manuals / 534 page-image targets without writes.
    A temporary live Qdrant contract smoke indexed two multivector pages,
    returned the expected page first, and removed the temporary collection.
  - This follows the ColPali page-image retrieval design and Qdrant's guidance
    to use multivectors as a bounded reranking stage rather than full-corpus
    unindexed search:
    <https://arxiv.org/abs/2407.01449>,
    <https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/>.

- [x] **5. Enforce strict held-out and operational gates.**
  - Added a fail-closed executable gate covering immutable dataset hash/order,
    clean provenance, minimum held-out size, source/human adjudication,
    document disjointness from tuning data, every category, every matrix cell,
    claim-level citation correctness, abstention correctness, and p95 latency.
  - Current result: **rejected**. The 48-case set is not labeled held-out, is not
    source/human-adjudicated under the new contract, has no supplied tuning set
    for disjointness, and both agent backends pass only 30/48 full gates.
    Failures concentrate in `single_hop_control` and `cross_document`.
  - Latency itself passes the 120 s p95 policy: LangGraph 106.1 s and
    LlamaIndex 68.9 s.
  - Evidence: `production_acceptance_gate_20260921.json`.

## Release decision

Keep deterministic hybrid/structural retrieval as the production foundation.
Keep BM25 and visual retrieval shadow/opt-in, retain the current embedding and
reranker defaults, and do not enable agentic production retrieval until a new
document-disjoint held-out bank passes the executable gate.
