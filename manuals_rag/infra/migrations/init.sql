create extension if not exists pgcrypto;

create table if not exists corpora (
    id text primary key,
    tenant_id text not null,
    name text not null,
    permissions_json jsonb not null default '{}'::jsonb
);

create table if not exists source_documents (
    id uuid primary key,
    tenant_id text not null,
    corpus_id text not null,
    document_kind text not null,
    title text not null,
    manufacturer text,
    product_family text,
    product_model text,
    language text not null default 'en',
    mime_type text not null,
    source_filename text not null,
    storage_uri text not null,
    sha256 text not null,
    file_size_bytes bigint not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    ingest_status text not null default 'uploaded',
    current_version_id uuid,
    visibility_scope text not null default 'internal',
    permissions_tags jsonb not null default '[]'::jsonb,
    retention_policy text
);

create unique index if not exists source_documents_tenant_sha256_uidx
    on source_documents (tenant_id, sha256);

create table if not exists document_versions (
    id uuid primary key,
    source_document_id uuid not null references source_documents(id) on delete cascade,
    version_label text not null,
    revision_date date,
    effective_date date,
    supersedes_version_id uuid,
    docling_artifact_uri text,
    page_count integer not null default 0,
    parse_profile text,
    ocr_used boolean not null default false,
    table_extraction_used boolean not null default false,
    status text not null,
    ingested_at timestamptz,
    parse_warnings jsonb not null default '[]'::jsonb,
    quality_score double precision not null default 0
);

create table if not exists logical_nodes (
    id text primary key,
    document_version_id uuid not null references document_versions(id) on delete cascade,
    node_type text not null,
    ordinal integer not null,
    depth integer not null,
    heading_text text,
    section_path_json jsonb not null default '[]'::jsonb,
    page_from integer not null,
    page_to integer not null,
    text_raw text not null default '',
    text_normalized text not null default '',
    table_json jsonb,
    caption_text text,
    warning_level text,
    procedure_step_number integer,
    spec_name text,
    spec_value text,
    spec_unit text,
    keywords_json jsonb not null default '[]'::jsonb,
    citability_score double precision not null default 0,
    token_count integer not null default 0
);

create table if not exists retrieval_chunks (
    id text primary key,
    document_version_id uuid not null references document_versions(id) on delete cascade,
    source_document_id uuid not null references source_documents(id) on delete cascade,
    logical_node_ids_json jsonb not null default '[]'::jsonb,
    chunk_type text not null,
    chunk_level integer not null,
    title text not null,
    section_path_text text not null,
    page_from integer not null,
    page_to integer not null,
    content text not null,
    content_for_sparse text not null,
    content_for_dense text not null,
    content_for_rerank text not null,
    metadata_json jsonb not null default '{}'::jsonb,
    is_active boolean not null default true,
    priority_score double precision not null default 0
);

create table if not exists document_metadata_extractions (
    source_document_id uuid primary key references source_documents(id) on delete cascade,
    document_version_id uuid not null references document_versions(id) on delete cascade,
    model text not null,
    metadata_json jsonb not null default '{}'::jsonb,
    extracted_at timestamptz not null default now()
);

create table if not exists ingestion_runs (
    id uuid primary key,
    source_document_id uuid not null references source_documents(id) on delete cascade,
    document_version_id uuid not null references document_versions(id) on delete cascade,
    status text not null,
    failure_class text,
    failure_reason text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists feedback (
    id uuid primary key,
    payload_json jsonb not null,
    created_at timestamptz not null default now()
);

insert into corpora (id, tenant_id, name, permissions_json)
values ('manuals_vendor_keyence', 'local-tenant', 'Keyence Manuals', '{"roles":["admin","operator","end_user","auditor"]}'::jsonb)
on conflict (id) do nothing;
