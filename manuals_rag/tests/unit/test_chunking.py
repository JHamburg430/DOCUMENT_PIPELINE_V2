import re

import pytest

from manuals_rag_chunking.hierarchical import _is_low_information_node, build_chunks
from manuals_rag_normalizers.normalize import normalize_nodes
from manuals_rag_parsers.docling_parser import parse_document
from manuals_rag_schemas.documents import LogicalNode
from manuals_rag_schemas.enums import ChunkType, NodeType
from tests.helpers import tmp_eval_small_pdf_path


def test_chunking_builds_l1_l2_l3():
    nodes = [
        LogicalNode(
            id="n1",
            document_version_id="v1",
            node_type=NodeType.section,
            ordinal=1,
            depth=1,
            heading_text="Setup",
            section_path_json=["Setup"],
            page_from=1,
            page_to=1,
            text_raw="SETUP",
            text_normalized="SETUP",
            token_count=1,
        ),
        LogicalNode(
            id="n2",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=2,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="Configure DHCP parameter 1 from the network menu and confirm the update sequence remains synchronized with the controller status output.",
            text_normalized="Configure DHCP parameter 1 from the network menu and confirm the update sequence remains synchronized with the controller status output.",
            section_path_json=["Setup"],
            token_count=19,
        ),
        LogicalNode(
            id="n3",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=3,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="Confirm the controller reports the updated setting on the status display and records the same configuration value in the diagnostics page.",
            text_normalized="Confirm the controller reports the updated setting on the status display and records the same configuration value in the diagnostics page.",
            section_path_json=["Setup"],
            token_count=19,
        ),
    ]
    chunks = build_chunks(
        source_document_id="d1",
        document_version_id="v1",
        title="Doc",
        nodes=nodes,
        metadata={"manufacturer": "Keyence"},
    )
    levels = {chunk.chunk_level for chunk in chunks}
    assert levels == {1, 2, 3}


def test_chunking_emits_datasheet_and_table_chunks():
    nodes = [
        LogicalNode(
            id="spec1",
            document_version_id="v1",
            node_type=NodeType.spec,
            ordinal=1,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="Voltage: 24 V",
            text_normalized="Voltage: 24 V",
            section_path_json=["Specifications"],
            spec_name="Voltage",
            spec_value="24",
            spec_unit="V",
            token_count=3,
        ),
        LogicalNode(
            id="table1",
            document_version_id="v1",
            node_type=NodeType.table,
            ordinal=2,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="Item | Value\nCurrent | 1 A",
            text_normalized="Item | Value\nCurrent | 1 A",
            section_path_json=["Specifications"],
            table_json={"headers": ["Item", "Value"], "rows": [["Current", "1 A"]]},
            token_count=6,
        ),
    ]
    chunks = build_chunks(
        source_document_id="d1",
        document_version_id="v1",
        title="Datasheet",
        nodes=nodes,
        metadata={"manufacturer": "Keyence", "document_kind": "datasheet"},
    )
    chunk_types = {chunk.chunk_type for chunk in chunks if chunk.chunk_level == 1}
    assert ChunkType.datasheet_record in chunk_types
    assert ChunkType.table_record in chunk_types
    table_chunks = [chunk for chunk in chunks if chunk.chunk_level == 1 and chunk.chunk_type == ChunkType.table_record]
    assert len(table_chunks) >= 3
    table_chunk = table_chunks[0]
    assert "table headers Item | Value" in table_chunk.content_for_sparse
    assert "Current 1 A" in table_chunk.content_for_dense


def test_chunking_carries_document_level_filter_metadata():
    nodes = [
        LogicalNode(
            id="n1",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=1,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="Configure EtherNet/IP communication parameter 1 for LJ-X8080.",
            text_normalized="Configure EtherNet/IP communication parameter 1 for LJ-X8080.",
            section_path_json=["Setup"],
            token_count=7,
        )
    ]
    chunks = build_chunks(
        source_document_id="d1",
        document_version_id="v1",
        title="Manual",
        nodes=nodes,
        metadata={
            "manufacturer": "Keyence",
            "document_kind": "manual",
            "product_model": "LJ-X8000",
            "product_models": ["LJ-X8000", "LJ-X8080"],
            "product_families": ["LJ", "LJ-X"],
            "devices": ["laser profiler"],
            "part_numbers": ["OP-88310"],
            "document_protocol_terms": ["ethernet/ip"],
            "settings": ["Sensor setup"],
            "parameters": ["communication parameter 1"],
            "document_topics": ["configuration", "communications"],
        },
    )

    chunk = next(item for item in chunks if item.chunk_level == 1)
    assert chunk.metadata_json["product_models"] == ["LJ-X8000", "LJ-X8080"]
    assert chunk.metadata_json["product_families"] == ["LJ", "LJ-X"]
    assert chunk.metadata_json["devices"] == ["laser profiler"]
    assert chunk.metadata_json["part_numbers"] == ["OP-88310"]
    assert chunk.metadata_json["document_protocol_terms"] == ["ethernet/ip"]
    assert chunk.metadata_json["settings"] == ["Sensor setup"]
    assert chunk.metadata_json["parameters"] == ["communication parameter 1"]
    assert chunk.metadata_json["document_topics"] == ["configuration", "communications"]


def test_atomic_chunks_include_local_rerank_context_and_small_section_windows():
    nodes = [
        LogicalNode(
            id="p1",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=1,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="The controller updates the output data.",
            text_normalized="The controller updates the output data.",
            section_path_json=["3.3 Checking the Data Output Flow Chart and Timing Diagrams"],
            token_count=7,
        ),
        LogicalNode(
            id="p2",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=2,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="After updating the output data, the controller turns the Result ready flag ON.",
            text_normalized="After updating the output data, the controller turns the Result ready flag ON.",
            section_path_json=["3.3 Checking the Data Output Flow Chart and Timing Diagrams"],
            token_count=12,
        ),
        LogicalNode(
            id="p3",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=3,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="When the command is successful, this remains OFF and the cycle continues to the next update interval.",
            text_normalized="When the command is successful, this remains OFF and the cycle continues to the next update interval.",
            section_path_json=["3.3 Checking the Data Output Flow Chart and Timing Diagrams"],
            token_count=16,
        ),
        LogicalNode(
            id="p4",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=4,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="A timeout condition clears the flag and restarts the command sequence for the controller.",
            text_normalized="A timeout condition clears the flag and restarts the command sequence for the controller.",
            section_path_json=["3.3 Checking the Data Output Flow Chart and Timing Diagrams"],
            token_count=14,
        ),
    ]
    chunks = build_chunks(
        source_document_id="d1",
        document_version_id="v1",
        title="Manual",
        nodes=nodes,
        metadata={"manufacturer": "Keyence", "product_model": "LJ-X8000"},
    )
    atomic_chunk = next(
        chunk
        for chunk in chunks
        if chunk.content
        == "The controller updates the output data. After updating the output data, the controller turns the Result ready flag ON."
    )
    assert atomic_chunk.chunk_type == ChunkType.atomic_text
    assert "3.3 Checking the Data Output Flow Chart and Timing Diagrams" in atomic_chunk.content_for_rerank
    assert "3.3 Checking the Data Output Flow Chart and Timing Diagrams" in atomic_chunk.content_for_dense
    assert "The controller updates the output data. After updating the output data, the controller turns the Result ready flag ON." in atomic_chunk.content_for_rerank
    assert "When the command is successful, this remains OFF and the cycle continues to the next update interval." in atomic_chunk.content_for_rerank

    section_windows = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.section_window]
    assert section_windows
    assert all(chunk.content.startswith("Manual | 3.3 Checking the Data Output Flow Chart and Timing Diagrams") for chunk in section_windows)


def test_structured_tables_do_not_emit_redundant_raw_table_chunk():
    raw_text = "Item | Value\nCurrent | 1 A\nVoltage | 24 V\nResponse time | 2 ms"
    nodes = [
        LogicalNode(
            id="table1",
            document_version_id="v1",
            node_type=NodeType.table,
            ordinal=1,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw=raw_text,
            text_normalized=raw_text,
            section_path_json=["Specifications"],
            table_json={
                "headers": ["Item", "Value"],
                "rows": [["Current", "1 A"], ["Voltage", "24 V"], ["Response time", "2 ms"]],
            },
            token_count=11,
        ),
    ]
    chunks = build_chunks(
        source_document_id="d1",
        document_version_id="v1",
        title="Datasheet",
        nodes=nodes,
        metadata={"manufacturer": "Keyence", "document_kind": "datasheet"},
    )
    l1_tables = [chunk for chunk in chunks if chunk.chunk_level == 1 and chunk.chunk_type == ChunkType.table_record]

    assert l1_tables
    assert all(chunk.content != raw_text for chunk in l1_tables)
    assert any(chunk.metadata_json.get("table_summary") for chunk in l1_tables)
    assert any(chunk.metadata_json.get("table_key_value") for chunk in l1_tables)


def test_table_nodes_are_not_dropped_by_phone_like_numeric_content():
    node = LogicalNode(
        id="table-phone-like",
        document_version_id="v1",
        node_type=NodeType.table,
        ordinal=1,
        depth=2,
        page_from=1,
        page_to=1,
        text_raw="Port | Value\nRS-232C | 230,400 bps\nMemory | CA-SD16G",
        text_normalized="Port | Value\nRS-232C | 230,400 bps\nMemory | CA-SD16G",
        section_path_json=["Specifications"],
        table_json={"headers": ["Port", "Value"], "rows": [["RS-232C", "230,400 bps"], ["Memory", "CA-SD16G"]]},
        token_count=8,
    )

    assert _is_low_information_node(node) is False


def test_chunking_does_not_emit_structured_chunk_for_low_information_spec_like_node():
    nodes = [
        LogicalNode(
            id="spec-bad",
            document_version_id="v1",
            node_type=NodeType.spec,
            ordinal=1,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="https: //www.keyence.com",
            text_normalized="https: //www.keyence.com",
            section_path_json=["Contact"],
            spec_name="https",
            spec_value="//www.keyence.com",
            token_count=2,
        ),
        LogicalNode(
            id="p-good",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=2,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="Encoder relay unit",
            text_normalized="Encoder relay unit",
            section_path_json=["Overview"],
            token_count=3,
        ),
    ]
    chunks = build_chunks(
        source_document_id="d1",
        document_version_id="v1",
        title="Datasheet",
        nodes=nodes,
        metadata={"manufacturer": "Keyence", "document_kind": "datasheet"},
    )
    assert all(chunk.content != "https: //www.keyence.com" for chunk in chunks)
    assert all(not (chunk.chunk_level == 1 and chunk.chunk_type == ChunkType.datasheet_record) for chunk in chunks)


def test_procedure_chunks_include_local_context_for_retrieval():
    nodes = [
        LogicalNode(
            id="step-1",
            document_version_id="v1",
            node_type=NodeType.procedure_step,
            ordinal=1,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="1. Connecting over EtherNet/IP",
            text_normalized="1. Connecting over EtherNet/IP",
            section_path_json=["Connection setup"],
            procedure_step_number=1,
            token_count=4,
        ),
        LogicalNode(
            id="p-1",
            document_version_id="v1",
            node_type=NodeType.paragraph,
            ordinal=2,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="Connect the controller to the network and confirm the communication status before continuing.",
            text_normalized="Connect the controller to the network and confirm the communication status before continuing.",
            section_path_json=["Connection setup"],
            token_count=12,
        ),
    ]
    chunks = build_chunks(
        source_document_id="d1",
        document_version_id="v1",
        title="Manual",
        nodes=nodes,
        metadata={"manufacturer": "Keyence", "document_kind": "manual", "product_model": "LJ-X8000"},
    )
    procedure_chunk = next(chunk for chunk in chunks if chunk.chunk_type == ChunkType.procedure_record and chunk.chunk_level == 1)
    assert "Connect the controller to the network" in procedure_chunk.content_for_dense
    assert "Connect the controller to the network" in procedure_chunk.content_for_sparse
    assert "Connect the controller to the network" in procedure_chunk.content_for_rerank


def test_chunking_multi_page_fixture_preserves_page_coverage_without_runaway_density():
    pdf_path = tmp_eval_small_pdf_path("AS_151019_LJ-X8000_C_689092_KA_US_2055_2.pdf")
    parsed = parse_document("chunking-small-version", pdf_path.name, pdf_path.read_bytes())
    normalized = normalize_nodes(parsed.logical_nodes)
    chunks = build_chunks(
        source_document_id="small-doc",
        document_version_id="chunking-small-version",
        title=pdf_path.name,
        nodes=normalized,
        metadata={"manufacturer": "Keyence", "document_kind": "manual"},
    )

    distinct_chunk_pages = {chunk.page_from for chunk in chunks}
    non_structural_chunks = [
        chunk
        for chunk in chunks
        if not chunk.metadata_json.get("table_cell") and not chunk.metadata_json.get("table_header")
    ]
    counts_by_page: dict[int, int] = {}
    for chunk in non_structural_chunks:
        counts_by_page[chunk.page_from] = counts_by_page.get(chunk.page_from, 0) + 1

    assert parsed.page_count > 1
    assert len(distinct_chunk_pages) > 1
    assert max(distinct_chunk_pages) == parsed.page_count
    assert max(counts_by_page.values()) <= 130
    assert any(chunk.metadata_json.get("table_cell") for chunk in chunks)


def test_large_structured_table_limits_row_level_chunk_explosion():
    rows = [[f"Field {index}", f"Value {index}"] for index in range(1, 41)]
    table_text = "\n".join(["Item | Value", *[f"{name} | {value}" for name, value in rows]])
    normalized = normalize_nodes(
        [
            LogicalNode(
                id="table-large",
                document_version_id="v1",
                node_type=NodeType.table,
                ordinal=1,
                depth=2,
                page_from=1,
                page_to=1,
                text_raw=table_text,
                text_normalized=table_text,
                section_path_json=["Specifications"],
                table_json={"headers": ["Item", "Value"], "rows": rows},
                token_count=len(table_text.split()),
            )
        ]
    )
    chunks = build_chunks(
        source_document_id="table-doc",
        document_version_id="v1",
        title="Large table manual",
        nodes=normalized,
        metadata={"manufacturer": "Keyence", "document_kind": "manual"},
    )

    l1_tables = [chunk for chunk in chunks if chunk.chunk_level == 1 and chunk.chunk_type == ChunkType.table_record]
    grouped_l1_tables = [
        chunk
        for chunk in l1_tables
        if not chunk.metadata_json.get("table_cell") and not chunk.metadata_json.get("table_header")
    ]
    row_groups = [chunk for chunk in l1_tables if chunk.metadata_json.get("table_row_group")]
    row_level = [chunk for chunk in l1_tables if chunk.metadata_json.get("table_key_value")]
    cell_level = [chunk for chunk in l1_tables if chunk.metadata_json.get("table_cell")]
    header_level = [chunk for chunk in l1_tables if chunk.metadata_json.get("table_header")]

    assert len(grouped_l1_tables) == 11
    assert len(row_groups) == 10
    assert not row_level
    assert len(cell_level) == 40
    assert len(header_level) == 42


def _retrieval_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _table_cell_values(table_json: dict) -> list[str]:
    values: list[str] = []
    for cell in table_json.get("cells", []):
        text = str(cell.get("text") or "").strip()
        if text:
            values.append(text)
    if values:
        return values
    values.extend(str(header).strip() for header in table_json.get("headers", []) if str(header).strip())
    for row in table_json.get("rows", []):
        values.extend(str(cell).strip() for cell in row if str(cell).strip())
    return values


def _table_cell_count(table_json: dict) -> int:
    cells = table_json.get("cells", [])
    if cells:
        return len(cells)
    return len(table_json.get("headers", [])) + sum(len(row) for row in table_json.get("rows", []))


def test_lj_x8000_docling_table_cells_are_retrievable_from_table_chunks():
    pdf_path = tmp_eval_small_pdf_path("AS_151019_LJ-X8000_C_689092_KA_US_2055_2.pdf")
    parsed = parse_document("table-cell-retrieval-version", pdf_path.name, pdf_path.read_bytes())
    table_nodes = [node for node in parsed.logical_nodes if node.node_type == NodeType.table and node.table_json]
    if not table_nodes:
        pytest.skip("Docling did not emit table nodes in this environment.")
    extracted_cell_count = sum(_table_cell_count(node.table_json or {}) for node in table_nodes)
    pages_with_tables = {node.page_from for node in table_nodes}

    assert len(table_nodes) >= 10
    assert extracted_cell_count >= 600
    assert {34, 35, 45, 47}.issubset(pages_with_tables)

    normalized = normalize_nodes(parsed.logical_nodes)
    chunks = build_chunks(
        source_document_id="lj-x8000-doc",
        document_version_id="table-cell-retrieval-version",
        title=pdf_path.name,
        nodes=normalized,
        metadata={"manufacturer": "Keyence", "document_kind": "manual", "product_model": "LJ-X8000"},
    )
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.table_record]
    assert table_chunks

    searchable_chunks = [
        (
            chunk.id,
            _retrieval_text(
                "\n".join(
                    str(part or "")
                    for part in (
                        chunk.content,
                        chunk.content_for_sparse,
                        chunk.content_for_dense,
                        chunk.content_for_rerank,
                    )
                )
            ),
        )
        for chunk in table_chunks
    ]
    missing: list[tuple[int, str]] = []
    checked = 0
    for node in table_nodes:
        for cell_value in _table_cell_values(node.table_json or {}):
            checked += 1
            query = _retrieval_text(cell_value)
            if not any(query in chunk_text for _, chunk_text in searchable_chunks):
                missing.append((node.page_from, cell_value))

    assert checked == extracted_cell_count
    assert not missing, (
        f"Queried {checked} table cells from {len(table_nodes)} tables on pages {sorted(pages_with_tables)}; "
        f"missing cells from table retrieval chunks: {missing[:20]}"
    )
    assert any(
        all(
            expected in chunk_text
            for expected in (
                _retrieval_text("Column headers: LJ-X8080"),
                _retrieval_text("Row headers: Repeatability *2 > Z-axis (height) *3"),
                _retrieval_text('Cell value: 0.5 μm 0.000020"'),
            )
        )
        for _, chunk_text in searchable_chunks
    )
