from __future__ import annotations

from enum import Enum


class DocumentKind(str, Enum):
    manual = "manual"
    setup_guide = "setup_guide"
    installation_guide = "installation_guide"
    service_guide = "service_guide"
    brochure = "brochure"
    datasheet = "datasheet"
    spec_sheet = "spec_sheet"
    troubleshooting_guide = "troubleshooting_guide"
    safety_bulletin = "safety_bulletin"
    release_note = "release_note"
    parts_catalog = "parts_catalog"


class ParseProfile(str, Enum):
    fast_text = "fast_text"
    standard_manual = "standard_manual"
    deep_manual = "deep_manual"


class NodeType(str, Enum):
    section = "section"
    paragraph = "paragraph"
    list = "list"
    table = "table"
    table_row_group = "table_row_group"
    figure = "figure"
    caption = "caption"
    procedure_step = "procedure_step"
    warning = "warning"
    caution = "caution"
    note = "note"
    spec = "spec"
    header_footer = "header_footer"
    appendix = "appendix"


class ChunkType(str, Enum):
    atomic_text = "atomic_text"
    section_window = "section_window"
    parent_section = "parent_section"
    table_record = "table_record"
    procedure_record = "procedure_record"
    spec_record = "spec_record"
    warning_record = "warning_record"
    brochure_fact = "brochure_fact"
    datasheet_record = "datasheet_record"
