-- Persist the normalized forms used by contextual/table lexical retrieval so
-- PostgreSQL does not recompute regexp_replace over the corpus for every query.
-- Generated columns keep ingestion and metadata backfills automatically in sync.

create extension if not exists pg_trgm;

alter table retrieval_chunks
    add column if not exists content_search_compact text generated always as (
        regexp_replace(lower(coalesce(content, '')), '[^a-z0-9]+', '', 'g')
    ) stored;

alter table retrieval_chunks
    add column if not exists local_context_search_compact text generated always as (
        regexp_replace(
            lower(coalesce(metadata_json->>'local_rerank_context', '')),
            '[^a-z0-9]+',
            '',
            'g'
        )
    ) stored;

alter table retrieval_chunks
    add column if not exists metadata_search_compact text generated always as (
        regexp_replace(lower(metadata_json::text), '[^a-z0-9]+', '', 'g')
    ) stored;

create index concurrently if not exists retrieval_chunks_active_content_search_trgm_idx
    on retrieval_chunks using gin (content_search_compact gin_trgm_ops)
    where is_active = true;

create index concurrently if not exists retrieval_chunks_active_local_context_search_trgm_idx
    on retrieval_chunks using gin (local_context_search_compact gin_trgm_ops)
    where is_active = true;

create index concurrently if not exists retrieval_chunks_active_metadata_search_trgm_idx
    on retrieval_chunks using gin (metadata_search_compact gin_trgm_ops)
    where is_active = true;

create index concurrently if not exists retrieval_chunks_active_corpus_type_idx
    on retrieval_chunks ((metadata_json->>'corpus_id'), chunk_type)
    where is_active = true;

create index concurrently if not exists retrieval_chunks_active_source_document_idx
    on retrieval_chunks (source_document_id)
    where is_active = true;

create index concurrently if not exists retrieval_chunks_active_document_version_idx
    on retrieval_chunks (document_version_id)
    where is_active = true;

analyze retrieval_chunks;
