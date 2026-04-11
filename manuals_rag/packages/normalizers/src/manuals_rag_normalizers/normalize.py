from __future__ import annotations

import re

from manuals_rag_schemas.documents import LogicalNode
from manuals_rag_schemas.enums import NodeType


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_keywords(text: str) -> list[str]:
    exact_matches = re.findall(
        r"\b[A-Z]{1,5}(?:-[A-Z0-9]{1,8})+\b|\b[A-Z]\d{2,4}\b|\b\d+(?:\.\d+)?\s?(?:V|A|mA|W|kW|mm|cm|m|ms|s|Hz|kHz|MHz|fps|kg|g|N|MPa|deg|C|°C|%)\b",
        text,
    )
    normalized_matches = []
    for match in exact_matches:
        normalized_matches.extend(
            [
                match,
                match.lower(),
                match.replace("-", ""),
                match.replace("-", " ").lower(),
            ]
        )
    menu_like = re.findall(r"\[[^\]]+\]", text)
    normalized_matches.extend(menu_like)
    return sorted({value.strip() for value in normalized_matches if value.strip()})[:40]


def normalize_nodes(nodes: list[LogicalNode]) -> list[LogicalNode]:
    normalized: list[LogicalNode] = []
    section_path: list[str] = []
    for node in nodes:
        text_normalized = normalize_text(node.text_raw or node.text_normalized)
        if node.node_type == NodeType.table and node.table_json:
            header_line = " | ".join(node.table_json.get("headers", []))
            row_lines = [" | ".join(row) for row in node.table_json.get("rows", [])]
            text_normalized = normalize_text("\n".join([header_line, *row_lines]))
        elif node.node_type == NodeType.spec and node.spec_name and node.spec_value:
            unit = f" {node.spec_unit}" if node.spec_unit else ""
            text_normalized = normalize_text(f"{node.spec_name}: {node.spec_value}{unit}")
        elif node.node_type == NodeType.procedure_step and node.procedure_step_number and not text_normalized.startswith(
            f"{node.procedure_step_number}"
        ):
            text_normalized = normalize_text(f"{node.procedure_step_number}. {text_normalized}")
        if node.node_type == NodeType.section:
            heading = text_normalized.splitlines()[0][:160]
            section_path = [heading]
            normalized.append(
                node.model_copy(
                    update={
                        "heading_text": heading,
                        "section_path_json": section_path,
                        "text_normalized": text_normalized,
                        "keywords_json": infer_keywords(text_normalized),
                    }
                )
            )
            continue
        normalized.append(
            node.model_copy(
                update={
                    "section_path_json": section_path,
                    "text_normalized": text_normalized,
                    "keywords_json": infer_keywords(text_normalized),
                    "citability_score": 0.99
                    if node.node_type in {NodeType.warning, NodeType.caution, NodeType.procedure_step, NodeType.spec, NodeType.table}
                    else (0.98 if node.node_type != NodeType.header_footer else 0.05),
                    "token_count": len(text_normalized.split()),
                }
            )
        )
    return normalized
