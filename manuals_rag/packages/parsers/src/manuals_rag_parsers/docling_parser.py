from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Any

import fitz

from manuals_rag_common.ids import deterministic_uuid
from manuals_rag_schemas.documents import LogicalNode, ParseResult
from manuals_rag_schemas.enums import DocumentKind, NodeType, ParseProfile

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.base_models import DocumentStream
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
except Exception:  # pragma: no cover - bootstrap fallback
    DocumentConverter = None
    DocumentStream = None
    InputFormat = None
    PdfPipelineOptions = None
    PdfFormatOption = None
    TableFormerMode = None


DOCLING_BATCH_SIZE_BY_PROFILE = {
    ParseProfile.fast_text: 12,
    ParseProfile.standard_manual: 6,
    ParseProfile.deep_manual: 1,
}

DOCLING_DEVICE = os.getenv("DOCLING_DEVICE", "cuda")
DOCLING_NUM_THREADS = int(os.getenv("DOCLING_NUM_THREADS", "4"))
DOCLING_LAYOUT_BATCH_SIZE = int(os.getenv("DOCLING_LAYOUT_BATCH_SIZE", "1"))
DOCLING_TABLE_BATCH_SIZE = int(os.getenv("DOCLING_TABLE_BATCH_SIZE", "1"))
DOCLING_OCR_BATCH_SIZE = int(os.getenv("DOCLING_OCR_BATCH_SIZE", "1"))
DOCLING_ENABLE_TABLE_STRUCTURE = os.getenv("DOCLING_ENABLE_TABLE_STRUCTURE", "true").lower() not in {"0", "false", "no"}
DOCLING_TABLEFORMER_MODE = os.getenv("DOCLING_TABLEFORMER_MODE", "accurate").lower()


def detect_document_kind(filename: str) -> DocumentKind:
    return DocumentKind.manual


def select_parse_profile(kind: DocumentKind) -> ParseProfile:
    if kind == DocumentKind.brochure:
        return ParseProfile.fast_text
    if kind in {DocumentKind.datasheet, DocumentKind.spec_sheet}:
        return ParseProfile.deep_manual
    return ParseProfile.standard_manual


SPEC_PATTERNS = [
    re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9 /().,%+-]{1,80}?)\s*[:\-]\s*(?P<value>.+?)(?:\s+(?P<unit>V|A|mA|W|kW|mm|cm|m|ms|s|Hz|kHz|MHz|fps|kg|g|N|MPa|deg|C|°C|%))?$",
        flags=re.I,
    ),
    re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9 /().,%+-]{1,80}?)\s{2,}(?P<value>.+?)(?:\s+(?P<unit>V|A|mA|W|kW|mm|cm|m|ms|s|Hz|kHz|MHz|fps|kg|g|N|MPa|deg|C|°C|%))?$",
        flags=re.I,
    ),
]

LOW_INFORMATION_SPEC_PATTERNS = [
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:toll free|phone|fax)\b", re.IGNORECASE),
]
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{6,}\d)")
LOW_SIGNAL_HEADING_PATTERN = re.compile(r"^[^A-Za-z0-9]*$|^[A-Za-z]$|^[A-Za-z0-9]{1,2}$")


def _is_ignorable_block(text: str) -> bool:
    stripped = text.strip().replace("\\_", "_")
    if not stripped:
        return True
    if stripped.startswith("<!--") and stripped.endswith("-->"):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_./\\:-]+\.(?:gif|png|jpe?g|svg|webp|bmp)", stripped, flags=re.I))


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip().replace("\\_", "_")
    if not stripped:
        return False
    if LOW_SIGNAL_HEADING_PATTERN.fullmatch(stripped):
        return False
    if len(re.findall(r"[A-Za-z0-9]", stripped)) < 3:
        return False
    if len(stripped) < 120 and stripped == stripped.upper():
        return True
    return bool(re.match(r"^(?:\d+(?:\.\d+)*)\s+[A-Z][^\n]{0,100}$", stripped))


def _extract_table(block: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    if any("|" in line for line in lines):
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines if "|" in line]
    else:
        split_rows = [re.split(r"\s{2,}|\t", line) for line in lines]
        widths = {len([cell for cell in row if cell.strip()]) for row in split_rows}
        if not widths or max(widths) < 2:
            return None
        if len(widths) > 2:
            return None
        rows = [[cell.strip() for cell in row if cell.strip()] for row in split_rows]
    rows = [row for row in rows if len(row) >= 2]
    if len(rows) < 2:
        return None
    return {
        "headers": rows[0],
        "rows": rows[1:],
        "row_count": max(0, len(rows) - 1),
        "column_count": len(rows[0]),
    }


def _extract_spec(text: str) -> tuple[str, str, str | None] | None:
    stripped = text.strip().replace("\\_", "_")
    if "\n" in stripped:
        return None
    for pattern in SPEC_PATTERNS:
        match = pattern.match(stripped)
        if not match:
            continue
        name = match.group("name").strip(" :.-")
        value = match.group("value").strip()
        unit = match.groupdict().get("unit")
        if len(name.split()) > 8 or len(value) > 120:
            continue
        if name.lower() in {"warning", "caution", "note", "notice"}:
            continue
        if any(pattern.search(name) or pattern.search(value) for pattern in LOW_INFORMATION_SPEC_PATTERNS):
            continue
        if PHONE_PATTERN.search(name) or PHONE_PATTERN.search(value):
            continue
        if "/" in name and not re.search(r"\d", name):
            continue
        if re.search(r"\.(?:gif|png|jpe?g|svg|webp|bmp)$", value, flags=re.I):
            continue
        return name, value, unit
    return None


def _classify_block(block: str) -> tuple[NodeType, dict[str, Any]]:
    stripped = block.strip()
    metadata: dict[str, Any] = {}
    if _looks_like_heading(stripped):
        return NodeType.section, {"heading_text": stripped.splitlines()[0][:160]}
    if re.match(r"^(warning|danger)\b", stripped, flags=re.I):
        return NodeType.warning, {"warning_level": "warning"}
    if re.match(r"^caution\b", stripped, flags=re.I):
        return NodeType.caution, {"warning_level": "caution"}
    if re.match(r"^(note|notice)\b", stripped, flags=re.I):
        return NodeType.note, metadata
    if re.match(r"^\d+[\.\)]\s+", stripped):
        step = re.match(r"^(?P<step>\d+)[\.\)]", stripped)
        metadata["procedure_step_number"] = int(step.group("step")) if step else None
        return NodeType.procedure_step, metadata
    table = _extract_table(stripped)
    if table:
        metadata["table_json"] = table
        return NodeType.table, metadata
    spec = _extract_spec(stripped)
    if spec:
        name, value, unit = spec
        metadata.update({"spec_name": name, "spec_value": value, "spec_unit": unit})
        return NodeType.spec, metadata
    return NodeType.paragraph, metadata


def _make_logical_node(
    *,
    document_version_id: str,
    ordinal: int,
    page_num: int,
    text: str,
    current_heading: str | None,
    seed_parts: tuple[Any, ...],
) -> LogicalNode:
    node_type, classified = _classify_block(text)
    heading_text = classified.get("heading_text")
    section_path = [heading_text] if node_type == NodeType.section and heading_text else ([current_heading] if current_heading else [])
    canonical_text = text.strip().replace("\\_", "_")
    if node_type == NodeType.table and classified.get("table_json"):
        table = classified["table_json"]
        header_line = " | ".join(table["headers"])
        row_lines = [" | ".join(row) for row in table["rows"]]
        canonical_text = "\n".join([header_line, *row_lines]).strip()
    elif node_type == NodeType.spec and classified.get("spec_name") and classified.get("spec_value"):
        unit = f" {classified['spec_unit']}" if classified.get("spec_unit") else ""
        canonical_text = f"{classified['spec_name']}: {classified['spec_value']}{unit}"
    return LogicalNode(
        id=deterministic_uuid(document_version_id, *seed_parts),
        document_version_id=document_version_id,
        node_type=node_type,
        ordinal=ordinal,
        depth=1 if node_type == NodeType.section else 2,
        heading_text=heading_text,
        section_path_json=section_path,
        page_from=page_num,
        page_to=page_num,
        text_raw=text.strip(),
        text_normalized=canonical_text,
        table_json=classified.get("table_json"),
        warning_level=classified.get("warning_level"),
        procedure_step_number=classified.get("procedure_step_number"),
        spec_name=classified.get("spec_name"),
        spec_value=classified.get("spec_value"),
        spec_unit=classified.get("spec_unit"),
        citability_score=0.9 if canonical_text else 0.1,
        token_count=len(canonical_text.split()),
    )


def _make_table_node(
    *,
    document_version_id: str,
    ordinal: int,
    page_num: int,
    table_json: dict[str, Any],
    current_heading: str | None,
    seed_parts: tuple[Any, ...],
) -> LogicalNode:
    headers = [str(header).strip() for header in table_json.get("headers", [])]
    rows = [[str(cell).strip() for cell in row] for row in table_json.get("rows", [])]
    lines = [" | ".join(headers), *[" | ".join(row) for row in rows]]
    canonical_text = "\n".join(line for line in lines if line.strip()).strip()
    return LogicalNode(
        id=deterministic_uuid(document_version_id, *seed_parts),
        document_version_id=document_version_id,
        node_type=NodeType.table,
        ordinal=ordinal,
        depth=2,
        heading_text=None,
        section_path_json=[current_heading] if current_heading else [],
        page_from=page_num,
        page_to=page_num,
        text_raw=canonical_text,
        text_normalized=canonical_text,
        table_json=table_json,
        citability_score=0.95 if canonical_text else 0.1,
        token_count=len(canonical_text.split()),
    )


def _docling_page_batches(page_count: int, batch_size: int) -> list[tuple[int, int]]:
    if page_count <= 0:
        return []
    return [(start, min(page_count, start + batch_size - 1)) for start in range(1, page_count + 1, batch_size)]


def _resolved_page_no(raw_page_no: int, *, batch_start: int, batch_end: int) -> int:
    batch_size = batch_end - batch_start + 1
    if batch_start > 1 and 1 <= raw_page_no <= batch_size:
        return batch_start + raw_page_no - 1
    return raw_page_no


def _normalize_for_page_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\\_", "_").strip().lower())


def _compact_for_page_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _page_texts_for_range(data: bytes, *, batch_start: int, batch_end: int) -> dict[int, str]:
    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        return {
            page_num: pdf.load_page(page_num - 1).get_text("text")
            for page_num in range(batch_start, batch_end + 1)
        }
    finally:
        pdf.close()


def _page_words_for_range(data: bytes, *, batch_start: int, batch_end: int) -> tuple[dict[int, list[tuple[Any, ...]]], dict[int, float]]:
    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        words_by_page: dict[int, list[tuple[Any, ...]]] = {}
        heights_by_page: dict[int, float] = {}
        for page_num in range(batch_start, batch_end + 1):
            page = pdf.load_page(page_num - 1)
            words_by_page[page_num] = page.get_text("words")
            heights_by_page[page_num] = float(page.rect.height)
        return words_by_page, heights_by_page
    finally:
        pdf.close()


def _match_block_page(text: str, page_texts: dict[int, str]) -> int | None:
    normalized_block = _normalize_for_page_match(text)
    if not normalized_block:
        return None
    normalized_pages = {page_num: _normalize_for_page_match(page_text) for page_num, page_text in page_texts.items()}
    direct_matches = [page_num for page_num, page_text in normalized_pages.items() if normalized_block in page_text]
    if len(direct_matches) == 1:
        return direct_matches[0]

    compact_block = _compact_for_page_match(normalized_block)
    if len(compact_block) < 24:
        return None
    compact_snippet = compact_block[: min(len(compact_block), 240)]
    compact_pages = {
        page_num: _compact_for_page_match(page_text)
        for page_num, page_text in normalized_pages.items()
    }
    compact_matches = [page_num for page_num, page_text in compact_pages.items() if compact_snippet in page_text]
    if len(compact_matches) == 1:
        return compact_matches[0]
    return None


def _docling_text_blocks(
    exported: dict[str, Any],
    *,
    batch_start: int,
    batch_end: int,
    page_texts: dict[int, str] | None = None,
    excluded_refs: set[str] | None = None,
) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    for index, item in enumerate(exported.get("texts", [])):
        if excluded_refs and f"#/texts/{index}" in excluded_refs:
            continue
        text = str(item.get("text") or item.get("orig") or "").strip()
        if _is_ignorable_block(text):
            continue
        prov = item.get("prov") or []
        page_no = batch_start
        if prov:
            raw_page_no = int(prov[0].get("page_no") or batch_start)
            page_no = _resolved_page_no(raw_page_no, batch_start=batch_start, batch_end=batch_end)
        normalized_text = text.replace("\\_", "_")
        matched_page_no = _match_block_page(normalized_text, page_texts or {}) if page_texts else None
        if matched_page_no is not None:
            page_no = matched_page_no
        blocks.append((page_no, normalized_text))
    return blocks


def _docling_table_child_refs(exported: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for table in exported.get("tables", []) or []:
        if not isinstance(table, dict):
            continue
        for item in table.get("children", []) or []:
            if isinstance(item, dict) and isinstance(item.get("$ref"), str):
                refs.add(item["$ref"])
    return refs


def _docling_table_json(table: dict[str, Any]) -> dict[str, Any] | None:
    data = table.get("data") or {}
    if not isinstance(data, dict):
        return None
    cells = [cell for cell in data.get("table_cells", []) or [] if isinstance(cell, dict)]
    if not cells:
        return None
    row_count = int(data.get("num_rows") or (max(int(cell.get("end_row_offset_idx") or 0) for cell in cells) if cells else 0))
    column_count = int(data.get("num_cols") or (max(int(cell.get("end_col_offset_idx") or 0) for cell in cells) if cells else 0))
    if row_count <= 0 or column_count <= 0:
        return None

    grid = [["" for _ in range(column_count)] for _ in range(row_count)]
    normalized_cells: list[dict[str, Any]] = []
    for cell in cells:
        text = str(cell.get("text") or "").strip().replace("\\_", "_")
        start_row = int(cell.get("start_row_offset_idx") or 0)
        start_col = int(cell.get("start_col_offset_idx") or 0)
        end_row = int(cell.get("end_row_offset_idx") or start_row + 1)
        end_col = int(cell.get("end_col_offset_idx") or start_col + 1)
        if 0 <= start_row < row_count and 0 <= start_col < column_count:
            grid[start_row][start_col] = text
        normalized_cells.append(
            {
                "row": start_row,
                "column": start_col,
                "row_span": max(1, end_row - start_row),
                "col_span": max(1, end_col - start_col),
                "text": text,
                "column_header": bool(cell.get("column_header")),
                "row_header": bool(cell.get("row_header")),
            }
        )

    headers = grid[0] if grid else []
    rows = grid[1:] if len(grid) > 1 else []
    return {
        "headers": headers,
        "rows": rows,
        "row_count": max(0, row_count - 1),
        "column_count": column_count,
        "cells": normalized_cells,
    }


def _normalize_label_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\\_", "_")).strip()


def _compact_label_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _table_bbox_top_left(table_json: dict[str, Any], page_height: float) -> tuple[float, float, float, float] | None:
    bbox = table_json.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        left = float(bbox["l"])
        right = float(bbox["r"])
        if str(bbox.get("coord_origin", "")).upper() == "BOTTOMLEFT":
            top = page_height - float(bbox["t"])
            bottom = page_height - float(bbox["b"])
        else:
            top = float(bbox["t"])
            bottom = float(bbox["b"])
    except (KeyError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _row_index_for_y(y_center: float, *, top: float, bottom: float, row_count: int) -> int:
    row_height = (bottom - top) / max(row_count, 1)
    return max(0, min(row_count - 1, int(round((y_center - top) / max(row_height, 1e-6)))))


def _enrich_table_row_group_labels(
    table_json: dict[str, Any],
    *,
    page_words: list[tuple[Any, ...]],
    page_height: float,
) -> dict[str, Any]:
    bbox = _table_bbox_top_left(table_json, page_height)
    if not bbox or not page_words:
        return table_json
    left, top, right, bottom = bbox
    table_text = _compact_label_text(
        " ".join(
            [
                *[str(item) for item in table_json.get("headers", []) or []],
                *[
                    str(cell)
                    for row in table_json.get("rows", []) or []
                    for cell in row
                ],
            ]
        )
    )
    row_count = int(table_json.get("row_count") or 0) + 1
    if row_count <= 1:
        return table_json
    row_height = (bottom - top) / max(row_count, 1)
    far_left_boundary = left + min(55.0, (right - left) * 0.14)
    secondary_label_boundary = left + min(150.0, (right - left) * 0.34)

    far_left_by_block: dict[int, list[tuple[Any, ...]]] = {}
    secondary_labels: list[tuple[float, float, int]] = []
    for word in page_words:
        x0, y0, x1, y1, text, block, line, word_no = word[:8]
        if not (left <= float(x0) <= right and top <= float(y0) <= bottom):
            continue
        x0_float = float(x0)
        if x0_float < far_left_boundary:
            far_left_by_block.setdefault(int(block), []).append(word)
        elif x0_float < secondary_label_boundary:
            label_text = _normalize_label_text(str(text))
            if not label_text or not re.search(r"[A-Za-z]", label_text) or re.match(r"^[±+\-]?\(?\d", label_text):
                continue
            y_center = (float(y0) + float(y1)) / 2.0
            secondary_labels.append(
                (
                    float(y0),
                    float(y1),
                    _row_index_for_y(y_center, top=top, bottom=bottom, row_count=row_count),
                )
            )

    inferred: list[dict[str, Any]] = []
    for block_words in far_left_by_block.values():
        block_words = sorted(block_words, key=lambda item: (int(item[6]), int(item[7]), float(item[0])))
        text = _normalize_label_text(" ".join(str(item[4]) for item in block_words))
        compact = _compact_label_text(text)
        if len(compact) < 3 or compact in table_text:
            continue
        y_min = min(float(item[1]) for item in block_words)
        y_max = max(float(item[3]) for item in block_words)
        y_center = (y_min + y_max) / 2.0
        row_label_margin = min(2.0, row_height * 0.25)
        nearby_rows = sorted(
            {
                row
                for label_y_min, label_y_max, row in secondary_labels
                if label_y_min <= y_max + row_label_margin and label_y_max >= y_min - row_label_margin
            }
        )
        if not nearby_rows:
            nearby_rows = [_row_index_for_y(y_center, top=top, bottom=bottom, row_count=row_count)]
        inferred.append(
            {
                "text": text,
                "rows": nearby_rows[:4],
                "source": "pdf_text_layer",
            }
        )
    if not inferred:
        return table_json
    existing = [item for item in table_json.get("pdf_row_group_labels", []) if isinstance(item, dict)]
    return {**table_json, "pdf_row_group_labels": [*existing, *inferred]}


def _docling_table_blocks(
    exported: dict[str, Any],
    *,
    batch_start: int,
    batch_end: int,
    page_words: dict[int, list[tuple[Any, ...]]] | None = None,
    page_heights: dict[int, float] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    blocks: list[tuple[int, dict[str, Any]]] = []
    for table in exported.get("tables", []) or []:
        if not isinstance(table, dict):
            continue
        table_json = _docling_table_json(table)
        if not table_json:
            continue
        prov = table.get("prov") or []
        page_no = batch_start
        if prov and isinstance(prov[0], dict):
            raw_page_no = int(prov[0].get("page_no") or batch_start)
            page_no = _resolved_page_no(raw_page_no, batch_start=batch_start, batch_end=batch_end)
            if isinstance(prov[0].get("bbox"), dict):
                table_json["bbox"] = prov[0]["bbox"]
        if page_words and page_heights and page_no in page_words and page_no in page_heights:
            table_json = _enrich_table_row_group_labels(
                table_json,
                page_words=page_words[page_no],
                page_height=page_heights[page_no],
            )
        blocks.append((page_no, table_json))
    return blocks


def _merge_docling_artifacts(filename: str, total_pages: int, batches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "parser": "docling_batched",
        "source_filename": filename,
        "original_page_count": total_pages,
        "batch_count": len(batches),
        "batches": batches,
    }


def _docling_pipeline_options(profile: ParseProfile, *, device: str) -> PdfPipelineOptions:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = DOCLING_ENABLE_TABLE_STRUCTURE and profile != ParseProfile.fast_text
    if pipeline_options.do_table_structure and TableFormerMode is not None:
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE if DOCLING_TABLEFORMER_MODE == "accurate" else TableFormerMode.FAST
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_table_images = False
    pipeline_options.force_backend_text = True
    pipeline_options.layout_batch_size = DOCLING_LAYOUT_BATCH_SIZE
    pipeline_options.table_batch_size = DOCLING_TABLE_BATCH_SIZE
    pipeline_options.ocr_batch_size = DOCLING_OCR_BATCH_SIZE
    pipeline_options.accelerator_options.device = device
    pipeline_options.accelerator_options.num_threads = DOCLING_NUM_THREADS
    return pipeline_options


def _is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _fallback_parse(document_version_id: str, data: bytes) -> ParseResult:
    document_version_id = str(document_version_id)
    pdf = fitz.open(stream=data, filetype="pdf")
    logical_nodes: list[LogicalNode] = []
    total_chars = 0
    current_heading: str | None = None
    for page_index in range(pdf.page_count):
        page = pdf.load_page(page_index)
        text = page.get_text("text")
        total_chars += len(text.strip())
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        for block_index, block in enumerate(blocks):
            logical_nodes.append(
                _make_logical_node(
                    document_version_id=document_version_id,
                    ordinal=len(logical_nodes) + 1,
                    page_num=page_index + 1,
                    text=block,
                    current_heading=current_heading,
                    seed_parts=(page_index + 1, block_index, block[:80]),
                )
            )
            if logical_nodes[-1].node_type == NodeType.section and logical_nodes[-1].heading_text:
                current_heading = logical_nodes[-1].heading_text
    quality_score = min(1.0, 0.2 + (total_chars / max(pdf.page_count, 1)) / 2000.0)
    return ParseResult(
        profile=ParseProfile.standard_manual,
        page_count=pdf.page_count,
        docling_artifact={"parser": "pymupdf_fallback", "pages": pdf.page_count},
        logical_nodes=logical_nodes,
        parse_warnings=["Docling unavailable; used PyMuPDF fallback parser."],
        quality_score=quality_score,
    )


def parse_document(document_version_id: str, filename: str, data: bytes) -> ParseResult:
    document_version_id = str(document_version_id)
    kind = detect_document_kind(filename)
    profile = select_parse_profile(kind)
    if DocumentConverter is None:
        result = _fallback_parse(document_version_id, data)
        return result.model_copy(update={"profile": profile})

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        total_pages = pdf.page_count
        pdf.close()
        batch_size = DOCLING_BATCH_SIZE_BY_PROFILE[profile]
        page_batches = _docling_page_batches(total_pages, batch_size)
        parse_warnings: list[str] = []
        for device in (DOCLING_DEVICE, "cpu"):
            try:
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=_docling_pipeline_options(profile, device=device)),
                    }
                )
                logical_nodes: list[LogicalNode] = []
                current_heading = "Document"
                artifacts: list[dict[str, Any]] = []
                ordinal = 0
                for batch_start, batch_end in page_batches:
                    source = DocumentStream(name=filename, stream=BytesIO(data))
                    converted = converter.convert(source=source, page_range=(batch_start, batch_end))
                    exported = converted.document.export_to_dict()
                    page_texts = _page_texts_for_range(data, batch_start=batch_start, batch_end=batch_end)
                    page_words, page_heights = _page_words_for_range(data, batch_start=batch_start, batch_end=batch_end)
                    artifacts.append({"page_range": [batch_start, batch_end], "document": json.loads(json.dumps(exported))})
                    table_child_refs = _docling_table_child_refs(exported)
                    for page_num, table_json in _docling_table_blocks(
                        exported,
                        batch_start=batch_start,
                        batch_end=batch_end,
                        page_words=page_words,
                        page_heights=page_heights,
                    ):
                        ordinal += 1
                        logical_nodes.append(
                            _make_table_node(
                                document_version_id=document_version_id,
                                ordinal=ordinal,
                                page_num=page_num,
                                table_json=table_json,
                                current_heading=current_heading,
                                seed_parts=(page_num, ordinal, "docling-table", table_json.get("row_count"), table_json.get("column_count")),
                            )
                        )
                    for page_num, block_text in _docling_text_blocks(
                        exported,
                        batch_start=batch_start,
                        batch_end=batch_end,
                        page_texts=page_texts,
                        excluded_refs=table_child_refs,
                    ):
                        ordinal += 1
                        node = _make_logical_node(
                            document_version_id=document_version_id,
                            ordinal=ordinal,
                            page_num=page_num,
                            text=block_text,
                            current_heading=current_heading,
                            seed_parts=(page_num, ordinal, block_text[:80]),
                        )
                        if node.node_type == NodeType.section and node.heading_text:
                            current_heading = node.heading_text
                        logical_nodes.append(node)
                return ParseResult(
                    profile=profile,
                    page_count=total_pages,
                    docling_artifact=_merge_docling_artifacts(filename, total_pages, artifacts),
                    logical_nodes=logical_nodes,
                    parse_warnings=parse_warnings,
                    quality_score=0.85 if logical_nodes else 0.2,
                )
            except Exception as exc:
                if device == DOCLING_DEVICE and device != "cpu" and _is_cuda_oom_error(exc):
                    parse_warnings.append(f"Docling GPU parse failed: {exc}")
                    continue
                raise
    except Exception as exc:  # pragma: no cover - integration path
        fallback = _fallback_parse(document_version_id, data)
        return fallback.model_copy(
            update={
                "profile": profile,
                "parse_warnings": [f"Docling parse failed: {exc}", *fallback.parse_warnings],
            }
        )
