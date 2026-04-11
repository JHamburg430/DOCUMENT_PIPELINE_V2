from manuals_rag_normalizers.normalize import infer_keywords, normalize_nodes, normalize_text
from manuals_rag_schemas.documents import LogicalNode
from manuals_rag_schemas.enums import NodeType


def test_normalize_text_preserves_identifiers():
    text = "  KV-8000   uses 24 V  \n\n  "
    assert normalize_text(text) == "KV-8000 uses 24 V"


def test_infer_keywords_extracts_models_and_units():
    keywords = infer_keywords("Use KV-8000 at 24 V and 10 ms delay.")
    assert "KV-8000" in keywords


def test_normalize_nodes_promotes_structured_text():
    nodes = [
        LogicalNode(
            id="s1",
            document_version_id="v1",
            node_type=NodeType.section,
            ordinal=1,
            depth=1,
            heading_text="Specs",
            section_path_json=["Specs"],
            page_from=1,
            page_to=1,
            text_raw="SPECS",
            text_normalized="SPECS",
        ),
        LogicalNode(
            id="n1",
            document_version_id="v1",
            node_type=NodeType.spec,
            ordinal=2,
            depth=2,
            page_from=1,
            page_to=1,
            text_raw="",
            text_normalized="",
            spec_name="Voltage",
            spec_value="24",
            spec_unit="V",
        ),
    ]
    normalized = normalize_nodes(nodes)
    assert normalized[1].text_normalized == "Voltage: 24 V"
    assert normalized[1].section_path_json == ["SPECS"]
