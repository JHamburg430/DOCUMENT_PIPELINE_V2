from __future__ import annotations

from prometheus_client import Counter, Histogram

INGEST_DURATION = Histogram("manuals_ingest_duration_seconds", "Ingestion duration", ["stage"])
QUERY_DURATION = Histogram("manuals_query_duration_seconds", "Query duration", ["stage"])
PARSE_FAILURES = Counter("manuals_parse_failures_total", "Parse failures", ["failure_class"])
ABSTAIN_COUNT = Counter("manuals_answer_abstain_total", "Abstained answers")
