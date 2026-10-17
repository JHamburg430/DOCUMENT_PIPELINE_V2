from manuals_rag_retrieval.knowledge_graph import rerank_metadata_hits


def test_graph_boosts_query_matching_model_alias_without_removing_hits():
    hits = [
        {"source_document_id": "wrong", "score": 0.9, "payload": {"product_model": "IV4"}},
        {"source_document_id": "right", "score": 0.8, "payload": {"normalized_identifier_aliases": ["IV-H2000"]}},
    ]
    ranked = rerank_metadata_hits("What does IV-H2000 support?", hits)
    assert [hit["source_document_id"] for hit in ranked] == ["right", "wrong"]
    assert all("knowledge_graph_score" in hit for hit in ranked)


def test_graph_is_safe_for_single_hit_and_preserves_fallback():
    hit = {"source_document_id": "only", "score": 0.4, "payload": {"product_family": "sensor"}}
    assert rerank_metadata_hits("sensor", [hit]) == [hit]
