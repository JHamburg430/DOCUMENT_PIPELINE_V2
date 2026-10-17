from manuals_rag_retrieval.hybrid_improvement import (
    build_external_review_request,
    rank_local_change_candidates,
    redact_review_payload,
    review_with_provider,
)


def test_external_payload_redacts_secrets_and_bounds_text():
    payload = build_external_review_request({"api_key": "do-not-send", "failure": "x" * 5000})
    assert payload["improvement_record"]["api_key"] == "[REDACTED]"
    assert len(payload["improvement_record"]["failure"]) == 1200


def test_local_fallback_ranks_failure_clusters():
    result = review_with_provider({"summary": {"failure_classes": {"grounded_answer": 2, "retrieval_recall": 5}}})
    assert result["provider"] == "local-fallback"
    assert result["ranked_changes"][0]["category"] == "retrieval_recall"


def test_provider_output_is_normalized_and_review_gated():
    result = review_with_provider({"summary": {}}, lambda _: {"regression_cases": [{"name": "x", "assertion": "y"}], "ranked_changes": [{"change": "z"}]})
    assert result["review_status"] == "pending_human_review"
    assert result["regression_cases"][0]["name"] == "x"
