from manuals_rag_answering.agentic_retrieval import _controlled_ledger_view
from manuals_rag_answering.agentic_retrieval import RetrievalPlan, RetrievalHop


def test_controlled_view_keeps_verified_claims_and_bounds_active_candidates():
    plan = RetrievalPlan(
        mode="parallel", rationale="test", hops=[
            RetrievalHop(hop_id="h1", objective="find voltage", query="voltage", required=True),
            RetrievalHop(hop_id="h2", objective="find current", query="current", required=True),
        ]
    )
    ledger = {
        "h1": {"sufficient": True, "assessment": {"supporting_chunk_ids": ["c1"]}},
        "h2": {"sufficient": False, "assessment": {"trust_state": "unconfirmed"}},
    }
    results = {"h2": [{"chunk_id": f"c{i}", "source_document_id": "d", "pages": [i], "content": "candidate"} for i in range(10)]}
    view = _controlled_ledger_view(plan, ledger, results)
    assert view["verified_claims"][0]["chunk_ids"] == ["c1"]
    assert view["unresolved_facets"][0]["facet_id"] == "h2"
    assert len(view["active_candidates"]) == 3
    assert view["rejected_candidates_included"] is False
    assert view["next_required_action"] == "verify_unresolved_facets"


def test_controlled_view_builds_provenance_bound_temporary_graph():
    plan = RetrievalPlan(
        mode="parallel", rationale="test", hops=[
            RetrievalHop(hop_id="h1", objective="find voltage", query="voltage", required=True),
        ]
    )
    view = _controlled_ledger_view(
        plan,
        {"h1": {"sufficient": True, "assessment": {"supporting_chunk_ids": ["c1"]}}},
        {"h1": [{
            "chunk_id": "c1",
            "source_document_id": "d1",
            "pages": [12],
            "content": "The device supports the requested voltage.",
        }]},
    )
    graph = view["temporary_evidence_graph"]
    assert graph["schema"] == "temporary-evidence-graph-v1"
    assert graph["scope"] == "single_question"
    assert graph["persistence"] == "discard_after_question"
    assert graph["model_edges_allowed"] is False
    assert {node["type"] for node in graph["nodes"]} == {"facet", "evidence", "document", "page"}
    assert any(edge["relation"] == "supported_by" for edge in graph["edges"])
    assert any(edge["relation"] == "located_on" for edge in graph["edges"])
