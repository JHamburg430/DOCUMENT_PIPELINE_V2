from __future__ import annotations

from prometheus_client import Counter, Histogram

INGEST_DURATION = Histogram("manuals_ingest_duration_seconds", "Ingestion duration", ["stage"])
QUERY_DURATION = Histogram("manuals_query_duration_seconds", "Query duration", ["stage"])
PARSE_FAILURES = Counter("manuals_parse_failures_total", "Parse failures", ["failure_class"])
ABSTAIN_COUNT = Counter("manuals_answer_abstain_total", "Abstained answers")
AGENTIC_RETRIEVAL_RUNS = Counter(
    "manuals_agentic_retrieval_runs_total",
    "Agentic retrieval runs by policy and terminal outcome",
    ["policy", "outcome"],
)
AGENTIC_RETRIEVAL_CLAIMS = Counter(
    "manuals_agentic_retrieval_claims_total",
    "Agentic retrieval claim-verification decisions",
    ["policy", "trust_state"],
)
AGENTIC_RETRIEVAL_HOPS = Histogram(
    "manuals_agentic_retrieval_hops",
    "Completed retrieval hops per agentic run",
    ["policy"],
    buckets=(1, 2, 3, 4, 6, 8),
)
