"""Privacy-bounded hybrid improvement helpers.

Paid models are optional offline reviewers. Production retrieval remains local.
Only redacted, bounded evaluation summaries are sent to an external provider.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable


SECRET_KEY = re.compile(r"(?i)(api[_-]?key|token|password|secret|authorization|cookie)")


def redact_review_payload(value: Any, *, max_text: int = 1200) -> Any:
    """Remove secret-like fields and bound text before external review."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SECRET_KEY.search(str(key)) else redact_review_payload(item, max_text=max_text)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_review_payload(item, max_text=max_text) for item in value[:20]]
    if isinstance(value, str):
        return value[:max_text]
    return value


def build_external_review_request(improvement_record: dict[str, Any]) -> dict[str, Any]:
    """Create the minimum safe payload for an optional paid evaluator."""
    safe = redact_review_payload(improvement_record)
    return {
        "schema": "rag-external-review.v1",
        "purpose": "diagnose_failure_clusters_and_propose_regressions",
        "record_sha256": hashlib.sha256(json.dumps(safe, sort_keys=True).encode()).hexdigest(),
        "improvement_record": safe,
        "constraints": [
            "Do not invent evidence or firmware/software values.",
            "Do not propose disabling evidence gates.",
            "Return JSON with failure_diagnosis, regression_cases, and ranked_changes.",
        ],
    }


def normalize_external_review(review: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider output into durable, provider-neutral artifacts."""
    regressions = []
    for item in review.get("regression_cases") or []:
        if not isinstance(item, dict) or not item.get("name") or not item.get("assertion"):
            continue
        regressions.append({
            "name": str(item["name"])[:160],
            "question": str(item.get("question") or "")[:1200],
            "assertion": str(item["assertion"])[:1200],
            "source_failure_class": str(item.get("source_failure_class") or "unknown"),
        })
    ranked = []
    for item in review.get("ranked_changes") or []:
        if not isinstance(item, dict) or not item.get("change"):
            continue
        ranked.append({
            "rank": len(ranked) + 1,
            "change": str(item["change"])[:500],
            "rationale": str(item.get("rationale") or "")[:1000],
            "risk": str(item.get("risk") or "unknown")[:80],
            "validation": str(item.get("validation") or "")[:500],
        })
    return {
        "schema": "rag-external-review-result.v1",
        "failure_diagnosis": [str(item)[:1000] for item in (review.get("failure_diagnosis") or []) if item][:20],
        "regression_cases": regressions,
        "ranked_changes": ranked[:20],
        "review_status": "pending_human_review",
    }


def rank_local_change_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    """HALO-like deterministic prioritization when no paid reviewer is configured."""
    counts = (record.get("summary") or {}).get("failure_classes") or {}
    priorities = {
        "retrieval_recall": (1, "Improve candidate recall and verify expected evidence retention."),
        "grounded_answer": (2, "Strengthen answer evidence/citation validation and evaluator feedback."),
        "evidence_sufficiency": (3, "Improve ledger completeness and multi-hop evidence checks."),
        "document_selection": (4, "Audit metadata filters and document routing."),
        "controller_termination": (5, "Review stop conditions and remaining-claim handling."),
    }
    rows = []
    for category, count in counts.items():
        priority, change = priorities.get(category, (99, "Review uncategorized failures."))
        rows.append({"category": category, "count": int(count), "priority": priority, "change": change})
    return sorted(rows, key=lambda row: (-row["count"], row["priority"]))


def review_with_provider(record: dict[str, Any], provider: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run an injected paid-model adapter, or return a safe local fallback."""
    request = build_external_review_request(record)
    if provider is None:
        return {"schema": "rag-external-review-result.v1", "provider": "local-fallback", "review_status": "pending_human_review", "ranked_changes": rank_local_change_candidates(record), "regression_cases": [], "failure_diagnosis": []}
    return {"provider": "external", **normalize_external_review(provider(request))}
