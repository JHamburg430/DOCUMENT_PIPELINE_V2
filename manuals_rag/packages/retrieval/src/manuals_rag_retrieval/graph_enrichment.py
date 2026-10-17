"""Evidence-gated model enrichment for the deterministic ingestion graph."""
from __future__ import annotations

from typing import Any, Callable

from manuals_rag_common.ollama import chat_json

GRAPH_SCHEMA = {"type": "object", "properties": {"edges": {"type": "array", "items": {"type": "object"}}}, "required": ["edges"]}


def enrich_graph_with_model(*, graph: dict[str, Any], chunks: list[Any], model: str,
                            call: Callable[..., tuple[Any, str]] = chat_json,
                            timeout: float = 180.0) -> dict[str, Any]:
    """Add only source-quoted, page-bound edges proposed by the model."""
    if not graph.get("nodes") or not chunks:
        return {**graph, "model_enrichment": {"status": "skipped", "accepted": 0, "rejected": 0}}
    by_chunk = {str(getattr(chunk, "id", "")): chunk for chunk in chunks}
    evidence = [node for node in graph["nodes"] if node.get("type") == "evidence" and node.get("chunk_id") in by_chunk]
    context = [{"evidence_id": node["id"], "chunk_id": node["chunk_id"], "pages": node.get("pages"), "text": str(getattr(by_chunk[node["chunk_id"]], "content", ""))[:5000]} for node in evidence[:24]]
    node_ids = {str(node.get("id")) for node in graph["nodes"]}
    prompt = ("Enrich this source-grounded manual graph. Propose only relationships explicitly supported by supplied text. "
              "Do not invent entities, versions, compatibility, causes, or prerequisites. Every edge needs from, to, relation, evidence_id, quote. "
              "Use existing node IDs only and return an empty list when uncertain.\n\n" f"Existing node IDs: {sorted(node_ids)}\nEvidence: {context}")
    proposed, _ = call(model=model, messages=[{"role": "system", "content": "Return strict JSON only."}, {"role": "user", "content": prompt}], json_schema=GRAPH_SCHEMA, timeout=timeout, purpose="ingestion_graph_enrichment", temperature=0.0)
    accepted, rejected = [], 0
    existing = {(edge.get("from"), edge.get("to"), edge.get("relation"), edge.get("quote")) for edge in graph.get("edges", [])}
    for edge in (proposed or {}).get("edges", []) if isinstance(proposed, dict) else []:
        if not isinstance(edge, dict):
            rejected += 1; continue
        evidence_id, quote = str(edge.get("evidence_id") or ""), str(edge.get("quote") or "").strip()
        chunk = next((by_chunk[node["chunk_id"]] for node in evidence if node["id"] == evidence_id), None)
        key = (edge.get("from"), edge.get("to"), edge.get("relation"), quote)
        if edge.get("from") not in node_ids or edge.get("to") not in node_ids or chunk is None or len(quote) < 12 or quote not in str(getattr(chunk, "content", "")) or key in existing:
            rejected += 1; continue
        accepted.append({"from": edge["from"], "to": edge["to"], "relation": str(edge["relation"]), "evidence_id": evidence_id, "quote": quote, "source": "model_verified_quote"})
        existing.add(key)
    return {**graph, "model_edges_allowed": True, "edges": [*graph.get("edges", []), *accepted], "model_enrichment": {"status": "completed", "accepted": len(accepted), "rejected": rejected, "model": model}}
