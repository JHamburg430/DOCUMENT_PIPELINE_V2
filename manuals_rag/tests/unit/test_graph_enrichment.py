from types import SimpleNamespace

from manuals_rag_retrieval.graph_enrichment import enrich_graph_with_model


def test_graph_enrichment_accepts_only_exact_source_quoted_edges():
    chunk = SimpleNamespace(id="c1", content="Before starting, position a workpiece on the stage.")
    graph = {"nodes": [{"id": "document", "type": "document"}, {"id": "evidence:c1", "type": "evidence", "chunk_id": "c1", "pages": [2, 2]}], "edges": []}

    def call(**_kwargs):
        return {"edges": [
            {"from": "document", "to": "evidence:c1", "relation": "requires", "evidence_id": "evidence:c1", "quote": "Before starting, position a workpiece on the stage."},
            {"from": "document", "to": "missing", "relation": "requires", "evidence_id": "evidence:c1", "quote": "Before starting, position a workpiece on the stage."},
            {"from": "document", "to": "evidence:c1", "relation": "invented", "evidence_id": "evidence:c1", "quote": "The manual supports wireless operation."},
        ]}, "{}"

    result = enrich_graph_with_model(graph=graph, chunks=[chunk], model="qwen3.5:9b", call=call)
    assert result["model_enrichment"] == {"status": "completed", "accepted": 1, "rejected": 2, "model": "qwen3.5:9b"}
    assert result["edges"][-1]["source"] == "model_verified_quote"


def test_graph_enrichment_does_not_call_model_without_grounded_graph():
    called = []
    result = enrich_graph_with_model(graph={"nodes": [], "edges": []}, chunks=[], model="qwen3.5:9b", call=lambda **_: called.append(True))
    assert result["model_enrichment"]["status"] == "skipped"
    assert not called
