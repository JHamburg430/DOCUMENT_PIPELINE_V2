"""Small deterministic metadata graph used to stabilize document routing.

This is intentionally not a second retrieval store.  It builds an ephemeral
graph from the metadata hits already returned by the metadata index and uses
explicit aliases, models, families, and document titles to score connected
documents.  Vector/lexical retrieval remains the fallback.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) > 1
    }


def _values(payload: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            values.update(str(item) for item in value)
        elif value not in (None, ""):
            values.add(str(value))
    return values


def rerank_metadata_hits(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank metadata hits using a bounded entity/relationship graph.

    Nodes are documents; edges connect documents sharing a model alias,
    product family, part number, or protocol term.  A query matching an
    entity boosts that node and its directly connected siblings.  No new
    entities are invented and no hit is removed, so routing can only improve
    ordering and cannot make retrieval fail closed.
    """
    if len(hits) < 2:
        return hits
    query_tokens = _tokens(query)
    nodes: list[dict[str, Any]] = []
    groups: dict[str, set[int]] = defaultdict(set)
    for index, hit in enumerate(hits):
        payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else hit
        entities = _values(
            payload,
            "routing_product_models",
            "normalized_identifier_aliases",
            "routing_part_numbers",
            "product_models",
            "product_model",
            "product_family",
            "product_families",
            "protocol_terms",
            "document_title",
            "source_filename",
        )
        entity_tokens = set().union(*(_tokens(value) for value in entities)) if entities else set()
        node = {"index": index, "tokens": entity_tokens, "entities": entities}
        nodes.append(node)
        for entity in entities:
            key = " ".join(sorted(_tokens(entity)))
            if key:
                groups[key].add(index)

    scores: dict[int, float] = {}
    for node in nodes:
        index = node["index"]
        direct = len(query_tokens & node["tokens"])
        connected = 0
        for key, members in groups.items():
            if index in members:
                connected += sum(1 for member in members if member != index and query_tokens & nodes[member]["tokens"])
        base = float(hits[index].get("score", 0.0) or 0.0)
        scores[index] = base + (0.08 * direct) + (0.02 * min(connected, 3))

    ranked: list[dict[str, Any]] = []
    for index in sorted(range(len(hits)), key=lambda item: scores[item], reverse=True):
        ranked.append(
            {
                **hits[index],
                "knowledge_graph_score": round(scores[index], 6),
                "retrieval_stage": "metadata_knowledge_graph",
            }
        )
    return ranked
