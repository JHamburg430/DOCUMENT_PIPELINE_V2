from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from manuals_rag_common.ids import deterministic_uuid
from manuals_rag_schemas.documents import LogicalNode, RetrievalChunk
from manuals_rag_schemas.enums import ChunkType, NodeType


SECTION_WINDOW_TOKEN_LIMIT = 220
SECTION_WINDOW_STRIDE = 2
TABLE_KEY_VALUE_MAX_ROWS = 12
MERGEABLE_NODE_TYPES = {NodeType.paragraph, NodeType.note}
MERGEABLE_TOKEN_LIMIT = 14
MERGED_NODE_TOKEN_LIMIT = 48
LOW_INFORMATION_PATTERNS = [
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:toll free|phone|fax)\b", re.IGNORECASE),
    re.compile(r"\.(?:gif|png|jpe?g|svg|webp|bmp)\b", re.IGNORECASE),
]
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{6,}\d)")


def _priority_for(node: LogicalNode) -> float:
    if node.node_type in {NodeType.warning, NodeType.caution}:
        return 25.0
    if node.node_type == NodeType.procedure_step:
        return 15.0
    if node.node_type == NodeType.spec:
        return 20.0
    if node.node_type == NodeType.table:
        return 10.0
    return 0.0


def _has_retrieval_signal(node: LogicalNode) -> bool:
    text = node.text_normalized.lower()
    if _is_low_information_node(node):
        return False
    if node.node_type in {NodeType.warning, NodeType.caution, NodeType.procedure_step, NodeType.spec, NodeType.table}:
        return True
    if node.node_type == NodeType.section:
        return False
    if any(char.isdigit() for char in text):
        return True
    if node.keywords_json:
        return True
    if "[" in text and "]" in text:
        return True
    if any(unit in text for unit in (" v", " a", " ma", " w", " kw", " mm", " cm", " hz", " khz", " mhz", "%")):
        return True
    technical_terms = (
        "parameter",
        "setting",
        "protocol",
        "command",
        "version",
        "revision",
        "warning",
        "caution",
        "input",
        "output",
        "resolution",
        "accuracy",
        "tolerance",
        "procedure",
        "configuration",
    )
    return node.token_count >= 8 and any(signal in text for signal in technical_terms)


def _is_low_information_node(node: LogicalNode) -> bool:
    text = (node.text_normalized or "").strip()
    if not text:
        return True
    if node.node_type == NodeType.table:
        return False
    if any(pattern.search(text) for pattern in LOW_INFORMATION_PATTERNS):
        return True
    if PHONE_PATTERN.search(text):
        return True
    if node.node_type == NodeType.spec:
        return False
    tokens = text.split()
    if len(tokens) <= 2 and not any(char.isdigit() for char in text) and not node.keywords_json:
        return True
    if len(tokens) <= 3 and sum(1 for token in tokens if any(char.isalpha() for char in token)) <= 1:
        return True
    return False


def _metadata_prefix(title: str, section_key: str, metadata: dict[str, Any]) -> str:
    parts = [
        title,
        str(metadata.get("manufacturer", "") or ""),
        str(metadata.get("product_family", "") or ""),
        str(metadata.get("product_model", "") or ""),
        str(metadata.get("document_kind", "") or ""),
        section_key,
    ]
    return " | ".join(part for part in parts if part)


def _join_non_empty(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _window_text(nodes: list[LogicalNode], *, title: str, section_key: str) -> str:
    labels = [title]
    if section_key and section_key != "Document":
        labels.append(section_key)
    return _join_non_empty([" | ".join(labels), *(node.text_normalized for node in nodes)])


def _local_rerank_context(nodes: list[LogicalNode], index: int, *, title: str, section_key: str) -> str:
    start = max(0, index - 1)
    end = min(len(nodes), index + 2)
    return _window_text(nodes[start:end], title=title, section_key=section_key)


def _sliding_groups(nodes: list[LogicalNode], *, token_limit: int, stride: int) -> list[list[LogicalNode]]:
    groups: list[list[LogicalNode]] = []
    start = 0
    while start < len(nodes):
        token_total = 0
        end = start
        while end < len(nodes):
            candidate = max(nodes[end].token_count, len(nodes[end].text_normalized.split()))
            if end > start and token_total + candidate > token_limit:
                break
            token_total += candidate
            end += 1
        groups.append(nodes[start:end])
        if end >= len(nodes):
            break
        start = max(start + stride, end - 1)
    return [group for group in groups if group]


def _should_emit_section_context(nodes: list[LogicalNode]) -> bool:
    if len(nodes) >= 3:
        return True
    if not nodes:
        return False
    token_total = sum(node.token_count for node in nodes)
    if token_total >= 60:
        return True
    return any(node.node_type in {NodeType.table, NodeType.procedure_step, NodeType.warning, NodeType.caution} for node in nodes)


def _can_merge_nodes(current: LogicalNode, candidate: LogicalNode) -> bool:
    if current.node_type not in MERGEABLE_NODE_TYPES or candidate.node_type not in MERGEABLE_NODE_TYPES:
        return False
    if current.page_from != candidate.page_from or current.page_to != candidate.page_to:
        return False
    if current.section_path_json != candidate.section_path_json:
        return False
    if current.token_count > MERGEABLE_TOKEN_LIMIT or candidate.token_count > MERGEABLE_TOKEN_LIMIT:
        return False
    return current.token_count + candidate.token_count <= MERGED_NODE_TOKEN_LIMIT


def _merge_node_pair(document_version_id: str, current: LogicalNode, candidate: LogicalNode) -> LogicalNode:
    merged_text = f"{current.text_normalized.strip()} {candidate.text_normalized.strip()}".strip()
    merged_keywords = sorted(set((current.keywords_json or []) + (candidate.keywords_json or [])))
    return current.model_copy(
        update={
            "id": deterministic_uuid(document_version_id, "merged-node", current.id, candidate.id),
            "text_raw": f"{current.text_raw.strip()} {candidate.text_raw.strip()}".strip(),
            "text_normalized": merged_text,
            "token_count": len(merged_text.split()),
            "page_to": candidate.page_to,
            "keywords_json": merged_keywords,
            "citability_score": max(current.citability_score, candidate.citability_score),
        }
    )


def _consolidate_nodes_for_chunking(document_version_id: str, nodes: list[LogicalNode]) -> list[LogicalNode]:
    consolidated: list[LogicalNode] = []
    buffer: LogicalNode | None = None
    for node in nodes:
        if buffer is None:
            buffer = node
            continue
        if _can_merge_nodes(buffer, node):
            buffer = _merge_node_pair(document_version_id, buffer, node)
            continue
        consolidated.append(buffer)
        buffer = node
    if buffer is not None:
        consolidated.append(buffer)
    return consolidated


def _content_variants(node: LogicalNode, *, title: str, section_key: str, metadata: dict[str, Any], chunk_type: ChunkType, content: str) -> tuple[str, str]:
    prefix = _metadata_prefix(title, section_key, metadata)
    sparse_lines = [prefix] if prefix else []
    dense_lines = [prefix] if prefix else []
    if chunk_type in {ChunkType.spec_record, ChunkType.datasheet_record}:
        sparse_lines.append("specification record")
        dense_lines.append("specification record")
        if node.spec_name:
            sparse_lines.append(f"specification {node.spec_name}")
            dense_lines.append(f"specification {node.spec_name}")
    elif chunk_type == ChunkType.table_record:
        sparse_lines.append("table record specification values")
        dense_lines.append("table record specification values")
        if node.table_json:
            headers = [str(header).strip() for header in node.table_json.get("headers", []) if str(header).strip()]
            rows = [
                " ".join(str(cell).strip() for cell in row if str(cell).strip())
                for row in node.table_json.get("rows", [])
            ]
            if headers:
                header_text = " | ".join(headers)
                sparse_lines.append(f"table headers {header_text}")
                dense_lines.append(f"table headers {header_text}")
            if rows:
                preview_rows = " ; ".join(row for row in rows[:4] if row)
                if preview_rows:
                    sparse_lines.append(f"table rows {preview_rows}")
                    dense_lines.append(f"table rows {preview_rows}")
    elif chunk_type == ChunkType.procedure_record:
        sparse_lines.append("procedure step")
        dense_lines.append("procedure step")
    elif chunk_type == ChunkType.warning_record:
        sparse_lines.append("warning notice")
        dense_lines.append("warning notice")
    sparse_lines.append(content)
    dense_lines.append(content)
    return "\n".join(line for line in sparse_lines if line), "\n".join(line for line in dense_lines if line)


def _base_chunk_metadata(node: LogicalNode, metadata: dict[str, Any], chunk_type: ChunkType, section_path: list[str]) -> dict[str, Any]:
    menu_labels = [label.strip() for label in re.findall(r"\[[^\]]+\]", node.text_normalized) if label.strip()]
    unit_tokens = re.findall(r"\b\d+(?:\.\d+)?\s?(?:V|A|mA|W|kW|mm|cm|m|ms|s|Hz|kHz|MHz|fps|kg|g|N|MPa|deg|C|°C|%)\b", node.text_normalized)
    identifier_tokens = [
        token
        for token in node.keywords_json
        if re.fullmatch(r"[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)+", token) or re.fullmatch(r"[A-Za-z]{2,}\d+[A-Za-z0-9-]*", token)
    ]
    protocol_terms = sorted(
        {
            match.lower()
            for match in re.findall(
                r"\b(?:ethernet/?ip|ethercat|profinet|modbus|tcp/?ip|udp|rs-232|rs-485|usb|io-link|canopen)\b",
                node.text_normalized,
                flags=re.IGNORECASE,
            )
        }
    )
    return {
        **metadata,
        "node_type": node.node_type.value,
        "section_path": section_path,
        "chunk_family": chunk_type.value,
        "warning_level": node.warning_level,
        "keywords": node.keywords_json,
        "citability_score": node.citability_score,
        "safety_flag": node.node_type in {NodeType.warning, NodeType.caution},
        "spec_flag": node.node_type == NodeType.spec,
        "procedure_flag": node.node_type == NodeType.procedure_step,
        "table_flag": node.node_type == NodeType.table,
        "brochure_flag": chunk_type == ChunkType.brochure_fact,
        "version_signal": "true" if metadata.get("revision_date") or metadata.get("version_label") else "false",
        "menu_labels": menu_labels[:10],
        "unit_tokens": unit_tokens[:10],
        "identifier_tokens": sorted(set(identifier_tokens))[:20],
        "protocol_terms": protocol_terms[:10],
    }


def _table_cell_text(cell: dict[str, Any]) -> str:
    return str(cell.get("text") or "").strip()


def _cell_row_range(cell: dict[str, Any]) -> range:
    row = int(cell.get("row") or 0)
    row_span = max(1, int(cell.get("row_span") or 1))
    return range(row, row + row_span)


def _cell_column_range(cell: dict[str, Any]) -> range:
    column = int(cell.get("column") or 0)
    col_span = max(1, int(cell.get("col_span") or 1))
    return range(column, column + col_span)


def _ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


def _dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        fingerprint = normalized.lower()
        if not normalized or fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(normalized)
    return deduped


def _table_cells(table_json: dict[str, Any]) -> list[dict[str, Any]]:
    cells = [cell for cell in table_json.get("cells", []) if isinstance(cell, dict)]
    if cells:
        return cells
    synthetic_cells: list[dict[str, Any]] = []
    for column, header in enumerate(table_json.get("headers", []) or []):
        synthetic_cells.append(
            {
                "row": 0,
                "column": column,
                "row_span": 1,
                "col_span": 1,
                "text": str(header),
                "column_header": True,
                "row_header": False,
            }
        )
    for row_offset, row in enumerate(table_json.get("rows", []) or [], start=1):
        for column, value in enumerate(row):
            synthetic_cells.append(
                {
                    "row": row_offset,
                    "column": column,
                    "row_span": 1,
                    "col_span": 1,
                    "text": str(value),
                    "column_header": False,
                    "row_header": column == 0,
                }
            )
    return synthetic_cells


def _table_column_headers_for_cell(cell: dict[str, Any], cells: list[dict[str, Any]]) -> list[str]:
    cell_rows = _cell_row_range(cell)
    cell_columns = _cell_column_range(cell)
    headers = [
        _table_cell_text(candidate)
        for candidate in cells
        if candidate is not cell
        and bool(candidate.get("column_header"))
        and _cell_row_range(candidate).stop <= cell_rows.start
        and _ranges_overlap(_cell_column_range(candidate), cell_columns)
    ]
    return _dedupe_texts(headers)


def _table_row_headers_for_cell(cell: dict[str, Any], cells: list[dict[str, Any]]) -> list[str]:
    cell_rows = _cell_row_range(cell)
    cell_columns = _cell_column_range(cell)
    headers = [
        _table_cell_text(candidate)
        for candidate in cells
        if candidate is not cell
        and bool(candidate.get("row_header"))
        and _ranges_overlap(_cell_row_range(candidate), cell_rows)
        and _cell_column_range(candidate).stop <= cell_columns.start
    ]
    return _dedupe_texts(headers)


def _fallback_column_header(cell: dict[str, Any], table_json: dict[str, Any]) -> list[str]:
    headers = table_json.get("headers", []) or []
    column = int(cell.get("column") or 0)
    if 0 <= column < len(headers):
        return _dedupe_texts([str(headers[column])])
    return []


def _looks_like_table_value(text: str) -> bool:
    return bool(
        re.search(
            r"^\s*(?:[±+\-]?\d|from\s+[±+\-]?\d)|\b\d+(?:\.\d+)?\s?(?:v|a|ma|w|kw|µm|μm|um|mm|cm|m|ms|s|hz|khz|mhz|fps|kg|g|n|mpa|deg|c|°c|%)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _fallback_row_headers_for_cell(cell: dict[str, Any], cells: list[dict[str, Any]]) -> list[str]:
    cell_rows = _cell_row_range(cell)
    cell_columns = _cell_column_range(cell)
    headers: list[str] = []
    same_row_left = sorted(
        [
            candidate
            for candidate in cells
            if candidate is not cell
            and _ranges_overlap(_cell_row_range(candidate), cell_rows)
            and _cell_column_range(candidate).stop <= cell_columns.start
        ],
        key=lambda candidate: int(candidate.get("column") or 0),
    )
    for candidate in same_row_left:
        text = _table_cell_text(candidate)
        if not text:
            continue
        if _looks_like_table_value(text):
            break
        headers.append(text)
    return _dedupe_texts(headers)


def _pdf_row_group_labels_for_cell(cell: dict[str, Any], table_json: dict[str, Any]) -> list[str]:
    row = int(cell.get("row") or 0)
    labels: list[str] = []
    for item in table_json.get("pdf_row_group_labels", []) or []:
        if not isinstance(item, dict):
            continue
        rows = []
        for value in item.get("rows", []) or []:
            try:
                rows.append(int(value))
            except (TypeError, ValueError):
                continue
        if row in rows:
            labels.append(str(item.get("text") or ""))
    return _dedupe_texts(labels)


def _table_header_and_cell_chunks(
    *,
    node: LogicalNode,
    source_document_id: str,
    document_version_id: str,
    title: str,
    section_key: str,
    metadata: dict[str, Any],
) -> list[RetrievalChunk]:
    if not node.table_json:
        return []
    cells = _table_cells(node.table_json)
    if not cells:
        return []
    chunks: list[RetrievalChunk] = []
    base_metadata = _base_chunk_metadata(node, metadata, ChunkType.table_record, node.section_path_json)
    prefix = _metadata_prefix(title, section_key, metadata)
    for cell_index, cell in enumerate(cells, start=1):
        text = _table_cell_text(cell)
        if not text:
            continue
        row = int(cell.get("row") or 0)
        column = int(cell.get("column") or 0)
        row_span = max(1, int(cell.get("row_span") or 1))
        col_span = max(1, int(cell.get("col_span") or 1))
        if bool(cell.get("column_header")) or bool(cell.get("row_header")):
            header_role = "column" if bool(cell.get("column_header")) else "row"
            if bool(cell.get("column_header")) and bool(cell.get("row_header")):
                header_role = "row and column"
            content = f"Table header: {text}; Header role: {header_role}; Row: {row}; Column: {column}"
            searchable = _join_non_empty([prefix, "table header", content])
            chunks.append(
                RetrievalChunk(
                    id=deterministic_uuid(document_version_id, "table-header", node.id, cell_index),
                    document_version_id=document_version_id,
                    source_document_id=source_document_id,
                    logical_node_ids_json=[node.id],
                    chunk_type=ChunkType.table_record,
                    chunk_level=1,
                    title=title,
                    section_path_text=section_key,
                    page_from=node.page_from,
                    page_to=node.page_to,
                    content=content,
                    content_for_sparse=searchable,
                    content_for_dense=searchable,
                    content_for_rerank=content,
                    metadata_json={
                        **base_metadata,
                        "table_header": True,
                        "table_header_role": header_role,
                        "table_row": row,
                        "table_column": column,
                        "table_row_span": row_span,
                        "table_col_span": col_span,
                    },
                    priority_score=11.0,
                )
            )
            continue
        column_headers = _table_column_headers_for_cell(cell, cells) or _fallback_column_header(cell, node.table_json)
        row_headers = _dedupe_texts(
            [
                *_pdf_row_group_labels_for_cell(cell, node.table_json),
                *(_table_row_headers_for_cell(cell, cells) or _fallback_row_headers_for_cell(cell, cells)),
            ]
        )
        context_parts = []
        if column_headers:
            context_parts.append(f"Column headers: {' > '.join(column_headers)}")
        if row_headers:
            context_parts.append(f"Row headers: {' > '.join(row_headers)}")
        context = "; ".join(context_parts)
        content = "; ".join(part for part in [context, f"Cell value: {text}", f"Row: {row}", f"Column: {column}"] if part)
        searchable = _join_non_empty([prefix, "table cell", content])
        chunks.append(
            RetrievalChunk(
                id=deterministic_uuid(document_version_id, "table-cell", node.id, cell_index),
                document_version_id=document_version_id,
                source_document_id=source_document_id,
                logical_node_ids_json=[node.id],
                chunk_type=ChunkType.table_record,
                chunk_level=1,
                title=title,
                section_path_text=section_key,
                page_from=node.page_from,
                page_to=node.page_to,
                content=content,
                content_for_sparse=searchable,
                content_for_dense=searchable,
                content_for_rerank=content,
                metadata_json={
                    **base_metadata,
                    "table_cell": True,
                    "table_row": row,
                    "table_column": column,
                    "table_row_span": row_span,
                    "table_col_span": col_span,
                    "table_column_headers": column_headers,
                    "table_row_headers": row_headers,
                },
                priority_score=13.0,
            )
        )
    return chunks


def _table_row_group_chunks(
    *,
    node: LogicalNode,
    source_document_id: str,
    document_version_id: str,
    title: str,
    section_key: str,
    metadata: dict[str, Any],
) -> list[RetrievalChunk]:
    if not node.table_json:
        return []
    headers = [str(header).strip() for header in node.table_json.get("headers", []) if str(header).strip()]
    rows = [[str(cell).strip() for cell in row if str(cell).strip()] for row in node.table_json.get("rows", [])]
    rows = [row for row in rows if row]
    chunks: list[RetrievalChunk] = []
    if headers:
        summary_text = f"Table summary: {' | '.join(headers)}"
        sparse, dense = _content_variants(
            node,
            title=title,
            section_key=section_key,
            metadata=metadata,
            chunk_type=ChunkType.table_record,
            content=summary_text,
        )
        chunks.append(
            RetrievalChunk(
                id=deterministic_uuid(document_version_id, "table-summary", node.id),
                document_version_id=document_version_id,
                source_document_id=source_document_id,
                logical_node_ids_json=[node.id],
                chunk_type=ChunkType.table_record,
                chunk_level=1,
                title=title,
                section_path_text=section_key,
                page_from=node.page_from,
                page_to=node.page_to,
                content=summary_text,
                content_for_sparse=sparse,
                content_for_dense=dense,
                content_for_rerank=summary_text,
                metadata_json={**_base_chunk_metadata(node, metadata, ChunkType.table_record, node.section_path_json), "table_summary": True},
                priority_score=10.0,
            )
        )
    for group_index, start in enumerate(range(0, len(rows), 4), start=1):
        group = rows[start : start + 4]
        row_lines = []
        for row in group:
            if headers and len(headers) == len(row):
                row_lines.append("; ".join(f"{header}: {value}" for header, value in zip(headers, row, strict=True)))
            else:
                row_lines.append(" | ".join(row))
        content = "\n".join(row_lines)
        sparse, dense = _content_variants(
            node,
            title=title,
            section_key=section_key,
            metadata=metadata,
            chunk_type=ChunkType.table_record,
            content=content,
        )
        chunks.append(
            RetrievalChunk(
                id=deterministic_uuid(document_version_id, "table-group", node.id, group_index),
                document_version_id=document_version_id,
                source_document_id=source_document_id,
                logical_node_ids_json=[node.id],
                chunk_type=ChunkType.table_record,
                chunk_level=1,
                title=title,
                section_path_text=section_key,
                page_from=node.page_from,
                page_to=node.page_to,
                content=content,
                content_for_sparse=sparse,
                content_for_dense=dense,
                content_for_rerank=content,
                metadata_json={**_base_chunk_metadata(node, metadata, ChunkType.table_record, node.section_path_json), "table_row_group": True},
                priority_score=10.0,
            )
        )
    if headers and rows and len(rows) <= TABLE_KEY_VALUE_MAX_ROWS:
        for row_index, row in enumerate(rows, start=1):
            if len(headers) < 2 or len(row) < 2:
                continue
            content = "; ".join(f"{header}: {value}" for header, value in zip(headers, row, strict=False) if value)
            sparse, dense = _content_variants(
                node,
                title=title,
                section_key=section_key,
                metadata=metadata,
                chunk_type=ChunkType.table_record,
                content=content,
            )
            chunks.append(
                RetrievalChunk(
                    id=deterministic_uuid(document_version_id, "table-kv", node.id, row_index),
                    document_version_id=document_version_id,
                    source_document_id=source_document_id,
                    logical_node_ids_json=[node.id],
                    chunk_type=ChunkType.table_record,
                    chunk_level=1,
                    title=title,
                    section_path_text=section_key,
                    page_from=node.page_from,
                    page_to=node.page_to,
                    content=content,
                    content_for_sparse=sparse,
                    content_for_dense=dense,
                    content_for_rerank=content,
                    metadata_json={**_base_chunk_metadata(node, metadata, ChunkType.table_record, node.section_path_json), "table_key_value": True},
                    priority_score=12.0,
                )
            )
    chunks.extend(
        _table_header_and_cell_chunks(
            node=node,
            source_document_id=source_document_id,
            document_version_id=document_version_id,
            title=title,
            section_key=section_key,
            metadata=metadata,
        )
    )
    return chunks


def build_chunks(
    *,
    source_document_id: str,
    document_version_id: str,
    title: str,
    nodes: list[LogicalNode],
    metadata: dict[str, Any],
) -> list[RetrievalChunk]:
    source_document_id = str(source_document_id)
    document_version_id = str(document_version_id)
    document_kind = str(metadata.get("document_kind", ""))
    nodes = _consolidate_nodes_for_chunking(document_version_id, nodes)
    chunks: list[RetrievalChunk] = []
    section_groups: dict[str, list[LogicalNode]] = defaultdict(list)
    procedure_groups: dict[str, list[LogicalNode]] = defaultdict(list)
    for node in nodes:
        if node.node_type == NodeType.header_footer:
            continue
        if node.node_type != NodeType.section and _is_low_information_node(node):
            continue
        section_key = " / ".join(node.section_path_json) or "Document"
        section_groups[section_key].append(node)
    ordered_section_groups = {
        section_key: sorted(grouped, key=lambda item: (item.page_from, item.ordinal))
        for section_key, grouped in section_groups.items()
    }
    for section_key, grouped in ordered_section_groups.items():
        node_index = {node.id: index for index, node in enumerate(grouped)}
        for node in grouped:
            if not _has_retrieval_signal(node):
                continue
            local_rerank_context = _local_rerank_context(grouped, node_index[node.id], title=title, section_key=section_key)
            chunk_type = ChunkType.atomic_text
            content = node.text_normalized
            if node.node_type == NodeType.warning:
                chunk_type = ChunkType.warning_record
                content = f"Warning: {node.text_normalized}"
            elif node.node_type == NodeType.caution:
                chunk_type = ChunkType.warning_record
                content = f"Caution: {node.text_normalized}"
            elif node.node_type == NodeType.procedure_step:
                chunk_type = ChunkType.procedure_record
                content = f"Procedure step {node.procedure_step_number or ''}: {node.text_normalized}".strip()
                procedure_groups[section_key].append(node)
            elif node.node_type == NodeType.spec:
                chunk_type = ChunkType.datasheet_record if document_kind in {"datasheet", "spec_sheet"} else ChunkType.spec_record
                if node.spec_name and node.spec_value:
                    unit = f" {node.spec_unit}" if node.spec_unit else ""
                    content = f"{node.spec_name}: {node.spec_value}{unit}"
            elif node.node_type == NodeType.table:
                chunk_type = ChunkType.table_record
                content = node.text_normalized
            elif document_kind == "brochure" and node.node_type in {NodeType.paragraph, NodeType.note} and node.token_count <= 80:
                chunk_type = ChunkType.brochure_fact

            structured_table_chunks: list[RetrievalChunk] = []
            if node.node_type == NodeType.table:
                structured_table_chunks = _table_row_group_chunks(
                    node=node,
                    source_document_id=source_document_id,
                    document_version_id=document_version_id,
                    title=title,
                    section_key=section_key,
                    metadata=metadata,
                )

            content_for_sparse, content_for_dense = _content_variants(
                node,
                title=title,
                section_key=section_key,
                metadata=metadata,
                chunk_type=chunk_type,
                content=content,
            )
            if chunk_type in {ChunkType.atomic_text, ChunkType.procedure_record}:
                content_for_sparse, content_for_dense = _content_variants(
                    node,
                    title=title,
                    section_key=section_key,
                    metadata=metadata,
                    chunk_type=chunk_type,
                    content=local_rerank_context,
                )

            if not (node.node_type == NodeType.table and structured_table_chunks):
                chunks.append(
                    RetrievalChunk(
                        id=deterministic_uuid(document_version_id, "l1", node.id),
                        document_version_id=document_version_id,
                        source_document_id=source_document_id,
                        logical_node_ids_json=[node.id],
                        chunk_type=chunk_type,
                        chunk_level=1,
                        title=title,
                        section_path_text=section_key,
                        page_from=node.page_from,
                        page_to=node.page_to,
                        content=content,
                        content_for_sparse=content_for_sparse,
                        content_for_dense=content_for_dense,
                        content_for_rerank=local_rerank_context if chunk_type in {ChunkType.atomic_text, ChunkType.procedure_record} else content,
                        metadata_json={
                            **_base_chunk_metadata(node, metadata, chunk_type, node.section_path_json),
                            "local_rerank_context": local_rerank_context if chunk_type in {ChunkType.atomic_text, ChunkType.procedure_record} else None,
                        },
                        priority_score=_priority_for(node),
                    )
                )
            if structured_table_chunks:
                chunks.extend(structured_table_chunks)

    for section_key, grouped in ordered_section_groups.items():
        if not _should_emit_section_context(grouped):
            continue
        for index, group in enumerate(_sliding_groups(grouped, token_limit=SECTION_WINDOW_TOKEN_LIMIT, stride=SECTION_WINDOW_STRIDE), start=1):
            text = _window_text(group, title=title, section_key=section_key)
            chunks.append(
                RetrievalChunk(
                    id=deterministic_uuid(document_version_id, "l2", section_key, index),
                    document_version_id=document_version_id,
                    source_document_id=source_document_id,
                    logical_node_ids_json=[node.id for node in group],
                    chunk_type=ChunkType.section_window,
                    chunk_level=2,
                    title=title,
                    section_path_text=section_key,
                    page_from=min(node.page_from for node in group),
                    page_to=max(node.page_to for node in group),
                    content=text,
                    content_for_sparse=text,
                    content_for_dense=text,
                    content_for_rerank=text,
                    metadata_json={
                        **metadata,
                        "section_path": section_key.split(" / ") if section_key else [],
                        "section_window": True,
                        "window_index": index,
                    },
                    priority_score=max(_priority_for(node) for node in group),
                )
            )

        full_text = _window_text(grouped, title=title, section_key=section_key)
        chunks.append(
            RetrievalChunk(
                id=deterministic_uuid(document_version_id, "l3", section_key),
                document_version_id=document_version_id,
                source_document_id=source_document_id,
                logical_node_ids_json=[node.id for node in grouped],
                chunk_type=ChunkType.parent_section,
                chunk_level=3,
                title=title,
                section_path_text=section_key,
                page_from=min(node.page_from for node in grouped),
                page_to=max(node.page_to for node in grouped),
                content=full_text,
                content_for_sparse=full_text,
                content_for_dense=full_text,
                content_for_rerank=full_text,
                metadata_json={**metadata, "section_path": section_key.split(" / ") if section_key else []},
                priority_score=max(_priority_for(node) for node in grouped),
            )
        )
    for section_key, steps in procedure_groups.items():
        for index, start in enumerate(range(0, len(steps), 4), start=1):
            group = steps[start : start + 4]
            if len(group) < 2:
                continue
            text = "\n".join(
                f"Step {node.procedure_step_number or order}: {node.text_normalized}" for order, node in enumerate(group, start=1)
            )
            exemplar = group[0]
            sparse, dense = _content_variants(
                exemplar,
                title=title,
                section_key=section_key,
                metadata=metadata,
                chunk_type=ChunkType.procedure_record,
                content=text,
            )
            chunks.append(
                RetrievalChunk(
                    id=deterministic_uuid(document_version_id, "procedure-group", section_key, index),
                    document_version_id=document_version_id,
                    source_document_id=source_document_id,
                    logical_node_ids_json=[node.id for node in group],
                    chunk_type=ChunkType.procedure_record,
                    chunk_level=2,
                    title=title,
                    section_path_text=section_key,
                    page_from=min(node.page_from for node in group),
                    page_to=max(node.page_to for node in group),
                    content=text,
                    content_for_sparse=sparse,
                    content_for_dense=dense,
                    content_for_rerank=text,
                    metadata_json={**_base_chunk_metadata(exemplar, metadata, ChunkType.procedure_record, exemplar.section_path_json), "grouped_procedure": True},
                    priority_score=15.0,
                )
            )
    return chunks
