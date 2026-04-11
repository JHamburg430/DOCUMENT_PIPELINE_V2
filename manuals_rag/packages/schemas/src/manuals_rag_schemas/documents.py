from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from manuals_rag_schemas.enums import ChunkType, DocumentKind, NodeType, ParseProfile


class SourceDocumentCreate(BaseModel):
    tenant_id: str
    corpus_id: str
    title: str
    manufacturer: str
    product_family: str | None = None
    product_model: str | None = None
    language: str = "en"
    document_kind: DocumentKind
    source_filename: str
    mime_type: str
    visibility_scope: str = "internal"
    permissions_tags: list[str] = Field(default_factory=list)


class LogicalNode(BaseModel):
    id: str
    document_version_id: str
    node_type: NodeType
    ordinal: int
    depth: int
    heading_text: str | None = None
    section_path_json: list[str] = Field(default_factory=list)
    page_from: int
    page_to: int
    text_raw: str = ""
    text_normalized: str = ""
    table_json: dict[str, Any] | None = None
    caption_text: str | None = None
    warning_level: str | None = None
    procedure_step_number: int | None = None
    spec_name: str | None = None
    spec_value: str | None = None
    spec_unit: str | None = None
    keywords_json: list[str] = Field(default_factory=list)
    citability_score: float = 0.5
    token_count: int = 0


class RetrievalChunk(BaseModel):
    id: str
    document_version_id: str
    source_document_id: str
    logical_node_ids_json: list[str]
    chunk_type: ChunkType
    chunk_level: int
    title: str
    section_path_text: str
    page_from: int
    page_to: int
    content: str
    content_for_sparse: str
    content_for_dense: str
    content_for_rerank: str
    metadata_json: dict[str, Any]
    is_active: bool = True
    priority_score: float = 0.0


class ParseResult(BaseModel):
    profile: ParseProfile
    page_count: int
    docling_artifact: dict[str, Any]
    logical_nodes: list[LogicalNode]
    parse_warnings: list[str] = Field(default_factory=list)
    quality_score: float


class QueryRequest(BaseModel):
    query: str
    corpus_ids: list[str]
    filters: dict[str, list[str] | bool | str | int] = Field(default_factory=dict)
    response_mode: str = "answer_with_citations"
    include_source_assets: bool = False
    include_page_images: bool = False
    include_table_images: bool = False


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    title: str
    document_version_id: str
    source_document_id: str
    pages: list[int]
    section_path: list[str]
    content: str
    metadata: dict[str, Any]


class AnswerResponse(BaseModel):
    answer: str
    confidence: str
    used_documents: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    warnings: list[str]
    followup_questions: list[str]
    insufficient_evidence: bool
