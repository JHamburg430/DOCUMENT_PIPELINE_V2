from __future__ import annotations

import json
import os
import queue
from contextlib import asynccontextmanager
from datetime import timedelta
from datetime import datetime
from io import BytesIO
from threading import Lock, Thread
from time import perf_counter
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import fitz
import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from apps.api.debug import (
    DEBUG_QUERY_STEP_SEQUENCE,
    build_document_debug_snapshot,
    build_document_metadata_snapshot,
    build_query_debug_snapshot,
    execute_query_debug_run,
    list_recent_documents,
    stream_query_debug_events,
)
from manuals_rag_answering.workflow import build_workflow
from manuals_rag_common.config import settings
from manuals_rag_common.db import execute, fetch_all, fetch_one, json_dumps
from manuals_rag_common.ids import sha256_bytes
from manuals_rag_common.logging import configure_logging
from manuals_rag_common.ollama import build_chat_payload, ensure_model_loaded, extract_chat_content, recent_ollama_calls
from manuals_rag_common.queue import enqueue, redis_client
from manuals_rag_common.storage import ObjectStore
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase, build_eval_cases_from_chunks, score_search_results, tokenize
from manuals_rag_observability.metrics import QUERY_DURATION
from manuals_rag_parsers.metadata import infer_document_metadata
from manuals_rag_permissions.auth import Principal, require_role
from manuals_rag_retrieval.retriever import build_filters, retrieve
from manuals_rag_schemas.documents import QueryRequest, SourceDocumentCreate


def _storage_object_name(tenant_id: str, sha256: str, filename: str) -> str:
    extension = os.path.splitext(filename)[1].lower() or ".bin"
    return f"{tenant_id}/sha256/{sha256}{extension}"


def _default_corpus_name(corpus_id: str) -> str:
    return corpus_id.replace("_", " ").replace("-", " ").strip().title() or corpus_id


def _upsert_corpus(
    corpus_id: str,
    tenant_id: str,
    name: str,
    permissions: dict[str, Any] | None = None,
    *,
    update_on_conflict: bool = True,
) -> None:
    conflict_action = "do update set name = excluded.name, permissions_json = excluded.permissions_json" if update_on_conflict else "do nothing"
    execute(
        f"""
        insert into corpora (id, tenant_id, name, permissions_json)
        values (%s, %s, %s, %s)
        on conflict (id) {conflict_action}
        """,
        (corpus_id, tenant_id, name, json_dumps(permissions or {})),
    )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Unsupported storage URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _safe_filename(filename: str) -> str:
    stem, extension = os.path.splitext(filename)
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem).strip("_")
    safe_extension = extension.lower() if extension else ".pdf"
    return f"{safe_stem or 'document'}{safe_extension}"


def _fetch_eval_chunk_rows(
    *,
    corpus_ids: list[str],
    document_id: str | None,
    max_chunks: int,
) -> list[dict[str, Any]]:
    where = ["rc.is_active = true", "rc.chunk_level = 1", "length(rc.content) >= 40"]
    params: list[Any] = []
    if document_id:
        where.append("rc.source_document_id = %s")
        params.append(document_id)
    elif corpus_ids:
        where.append("sd.corpus_id = any(%s)")
        params.append(corpus_ids)
    else:
        where.append("sd.ingest_status = 'indexed'")
    params.append(max(1, min(max_chunks, 1000)))
    return fetch_all(
        f"""
        select
            rc.id,
            rc.source_document_id,
            rc.document_version_id,
            rc.chunk_type,
            rc.chunk_level,
            rc.title,
            rc.section_path_text,
            rc.page_from,
            rc.page_to,
            rc.content,
            rc.metadata_json,
            sd.source_filename,
            coalesce(sd.product_model, rc.metadata_json->>'product_model', '') as product_model
        from retrieval_chunks rc
        join source_documents sd on sd.id = rc.source_document_id
        where {' and '.join(where)}
          and rc.chunk_type in ('table_record','datasheet_record','spec_record','procedure_record','warning_record','atomic_text')
        order by
          case rc.chunk_type
            when 'datasheet_record' then 1
            when 'spec_record' then 2
            when 'procedure_record' then 3
            when 'warning_record' then 4
            when 'table_record' then 5
            else 6
          end,
          rc.priority_score desc,
          length(rc.content) desc
        limit %s
        """,
        tuple(params),
    )


def _answer_contains_expected_terms(answer: dict[str, Any], expected_terms: list[str]) -> dict[str, Any]:
    answer_text = str(answer.get("answer") or "")
    answer_text_lower = answer_text.lower()
    answer_tokens = set(tokenize(answer_text))
    expected = [term for term in expected_terms if term]
    matched = [
        term
        for term in expected
        if _expected_term_matches_text(term, answer_text_lower, answer_tokens)
    ]
    required = min(2, len(expected))
    return {
        "passed": len(matched) >= required if required else False,
        "matched_terms": matched,
        "expected_terms": expected,
        "required_terms": required,
    }


def _expected_term_matches_text(term: str, text_lower: str, text_tokens: set[str]) -> bool:
    term_lower = term.lower().strip()
    if not term_lower:
        return False
    if term_lower in text_lower or term_lower in text_tokens:
        return True
    if "/" in term_lower:
        parts = [part for part in term_lower.split("/") if part]
        return bool(parts) and all(part in text_tokens for part in parts)
    return False


def _score_answer(case: RetrievalEvalCase, answer: dict[str, Any], retrieval_evaluation: dict[str, Any]) -> dict[str, Any]:
    citation_document_ids = {
        str(citation.get("document_id") or citation.get("source_document_id") or "")
        for citation in answer.get("citations", [])
        if isinstance(citation, dict)
    }
    used_document_ids = {
        str(document.get("document_id") or document.get("source_document_id") or "")
        for document in answer.get("used_documents", [])
        if isinstance(document, dict)
    }
    expected_document_used = case.source_document_id in citation_document_ids or case.source_document_id in used_document_ids
    terms = _answer_contains_expected_terms(answer, case.expected_terms)
    answer_text = str(answer.get("answer") or "").strip()
    passed = bool(
        answer_text
        and not answer.get("insufficient_evidence")
        and expected_document_used
        and terms["passed"]
    )
    failure_reasons = []
    if not answer_text:
        failure_reasons.append("empty_answer")
    if answer.get("insufficient_evidence"):
        failure_reasons.append("insufficient_evidence")
    if not expected_document_used:
        failure_reasons.append("expected_document_not_cited_or_used")
    if not terms["passed"]:
        failure_reasons.append("expected_terms_missing")
    return {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "expected_document_used": expected_document_used,
        "term_check": terms,
    }


def _summarize_end_to_end_eval(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    retrieval_passed = sum(1 for item in items if item["retrieval_evaluation"]["passed"])
    answer_passed = sum(1 for item in items if item["answer_evaluation"]["passed"])
    return {
        "total_questions": total,
        "retrieval_correct": retrieval_passed,
        "retrieval_correct_percent": round((retrieval_passed / total) * 100, 2) if total else 0.0,
        "answers_correct": answer_passed,
        "answers_correct_percent": round((answer_passed / total) * 100, 2) if total else 0.0,
    }


def _ensure_run_tables() -> None:
    execute(
        """
        create table if not exists app_runs (
            id uuid primary key,
            run_type text not null,
            status text not null,
            request_json jsonb not null default '{}'::jsonb,
            progress_json jsonb not null default '{}'::jsonb,
            result_json jsonb,
            error text,
            created_at timestamptz not null default now(),
            updated_at timestamptz not null default now()
        )
        """
    )
    execute(
        """
        create table if not exists app_run_events (
            id bigserial primary key,
            run_id uuid not null references app_runs(id) on delete cascade,
            event_index integer not null,
            event_json jsonb not null,
            created_at timestamptz not null default now(),
            unique (run_id, event_index)
        )
        """
    )


def _create_persisted_run(run_id: str, run_type: str, request_payload: dict[str, Any]) -> None:
    execute(
        """
        insert into app_runs (id, run_type, status, request_json, progress_json, created_at, updated_at)
        values (%s, %s, 'queued', %s::jsonb, '{}'::jsonb, now(), now())
        on conflict (id) do update
        set run_type = excluded.run_type,
            status = excluded.status,
            request_json = excluded.request_json,
            progress_json = excluded.progress_json,
            result_json = null,
            error = null,
            updated_at = now()
        """,
        (run_id, run_type, json_dumps(request_payload)),
    )


def _update_persisted_run(
    run_id: str,
    *,
    status: str | None = None,
    progress: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    current = fetch_one("select progress_json from app_runs where id = %s", (run_id,)) or {}
    progress_payload = progress if progress is not None else dict(current.get("progress_json") or {})
    execute(
        """
        update app_runs
        set status = coalesce(%s, status),
            progress_json = %s::jsonb,
            result_json = coalesce(%s::jsonb, result_json),
            error = coalesce(%s, error),
            updated_at = now()
        where id = %s
        """,
        (
            status,
            json_dumps(progress_payload),
            json_dumps(result) if result is not None else None,
            error,
            run_id,
        ),
    )


def _append_run_event(run_id: str, event_index: int, event: dict[str, Any]) -> None:
    execute(
        """
        insert into app_run_events (run_id, event_index, event_json, created_at)
        values (%s, %s, %s::jsonb, now())
        on conflict (run_id, event_index) do nothing
        """,
        (run_id, event_index, json_dumps(event)),
    )


def _fail_stale_running_runs() -> None:
    execute(
        """
        update app_runs
        set status = 'failed',
            error = coalesce(error, 'Run was left running without progress and was marked failed.'),
            updated_at = now()
        where status = 'running'
          and updated_at < now() - interval '10 minutes'
        """
    )


def _finalize_abandoned_run(run_id: str, message: str) -> None:
    current = fetch_one("select status from app_runs where id = %s", (run_id,))
    if current and current.get("status") == "running":
        _update_persisted_run(run_id, status="failed", error=message)


def _load_eval_cases(payload: dict[str, Any]) -> tuple[list[str], str | None, list[RetrievalEvalCase], list[str]]:
    corpus_ids = [str(item).strip() for item in payload.get("corpus_ids", []) if str(item).strip()]
    document_id = str(payload.get("document_id") or "").strip() or None
    max_questions = max(1, min(int(payload.get("max_questions") or 10), 50))
    max_chunks = max(max_questions, min(int(payload.get("max_chunks") or max_questions * 8), 1000))
    use_llm_generation = bool(payload.get("use_llm_generation", True))
    if not document_id and not corpus_ids:
        raise HTTPException(status_code=400, detail="Provide corpus_ids for all-doc eval or document_id for a single-document eval.")
    chunk_rows = _fetch_eval_chunk_rows(corpus_ids=corpus_ids, document_id=document_id, max_chunks=max_chunks)
    cases = build_eval_cases_from_chunks(
        chunk_rows,
        max_cases=max_questions,
        use_llm_generation=use_llm_generation,
    )
    warnings = [] if cases else ["No query-worthy indexed chunks were found for the requested scope."]
    return corpus_ids, document_id, cases, warnings


def _eval_search_scope(corpus_ids: list[str], document_id: str | None) -> tuple[list[str], dict[str, Any]]:
    filters: dict[str, Any] = {}
    search_corpus_ids = corpus_ids
    if document_id:
        filters["source_document_id"] = [document_id]
        document = fetch_one("select corpus_id from source_documents where id = %s", (document_id,))
        if document and document.get("corpus_id"):
            search_corpus_ids = [str(document["corpus_id"])]
    return search_corpus_ids, filters


def _top_results_from_query_debug(result: dict[str, Any]) -> list[dict[str, Any]]:
    for stage in result.get("stages", []):
        if stage.get("name") == "retrieval_results":
            return [dict(item) for item in stage.get("samples", [])]
    return []


def _render_pdf_page_png(data: bytes, page_number: int) -> bytes:
    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        if page_number < 1 or page_number > pdf.page_count:
            raise ValueError(f"Page {page_number} is outside PDF page range 1-{pdf.page_count}.")
        page = pdf.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        return pixmap.tobytes("png")
    finally:
        pdf.close()


def _page_image_uri(
    *,
    store: ObjectStore,
    document: dict[str, Any],
    page_number: int,
) -> str:
    object_name = (
        f"{document['tenant_id']}/document-assets/"
        f"{document['id']}/{document['current_version_id']}/page-images/page-{page_number:04d}.png"
    )
    if not store.object_exists(settings.minio_bucket_artifacts, object_name):
        source_bucket, source_object = _parse_s3_uri(document["storage_uri"])
        response = store.client.get_object(source_bucket, source_object)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        image_bytes = _render_pdf_page_png(data, page_number)
        store.put_bytes(settings.minio_bucket_artifacts, object_name, image_bytes, "image/png")
    return f"s3://{settings.minio_bucket_artifacts}/{object_name}"


def _presigned_s3_url(store: ObjectStore, uri: str) -> str:
    bucket, object_name = _parse_s3_uri(uri)
    return store.presigned_get_url(bucket, object_name, expires=timedelta(hours=1))


def _read_json_s3_uri(store: ObjectStore, uri: str | None) -> dict[str, Any]:
    if not uri:
        return {}
    bucket, object_name = _parse_s3_uri(uri)
    response = store.client.get_object(bucket, object_name)
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()


def _artifact_image_assets(store: ObjectStore, document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    artifact = _read_json_s3_uri(store, document.get("docling_artifact_uri"))
    image_assets = artifact.get("image_assets") if isinstance(artifact, dict) else {}
    if not isinstance(image_assets, dict):
        return {"page_images": [], "table_images": []}
    return {
        "page_images": [dict(item) for item in image_assets.get("page_images", []) if isinstance(item, dict)],
        "table_images": [dict(item) for item in image_assets.get("table_images", []) if isinstance(item, dict)],
    }


def _source_assets_for_results(
    results: list[dict[str, Any]],
    *,
    include_page_images: bool,
    include_table_images: bool = False,
) -> dict[str, Any]:
    document_ids = sorted({str(result["source_document_id"]) for result in results if result.get("source_document_id")})
    if not document_ids:
        return {"documents": [], "citation_pages": [], "tables": []}
    placeholders = ",".join(["%s"] * len(document_ids))
    documents = fetch_all(
        f"""
        select
            sd.id,
            sd.tenant_id,
            sd.current_version_id,
            sd.title,
            sd.source_filename,
            sd.storage_uri,
            dv.docling_artifact_uri
        from source_documents sd
        left join document_versions dv on dv.id = sd.current_version_id
        where sd.id in ({placeholders})
        """,
        tuple(document_ids),
    )
    document_by_id = {str(document["id"]): dict(document) for document in documents}
    store = ObjectStore()
    document_assets: dict[str, dict[str, Any]] = {}
    image_assets_by_document: dict[str, dict[str, list[dict[str, Any]]]] = {}
    citation_pages: list[dict[str, Any]] = []
    table_assets: list[dict[str, Any]] = []
    for document_id, document in document_by_id.items():
        pdf_uri = str(document["storage_uri"])
        if include_page_images or include_table_images:
            image_assets_by_document[document_id] = _artifact_image_assets(store, document)
        document_assets[document_id] = {
            "document_id": document_id,
            "version_id": str(document["current_version_id"]),
            "title": document["title"],
            "source_filename": document["source_filename"],
            "pdf_uri": pdf_uri,
            "pdf_download_url": _presigned_s3_url(store, pdf_uri),
            "download_filename": _safe_filename(str(document["source_filename"])),
        }
    for result in results:
        document_id = str(result.get("source_document_id") or "")
        document = document_by_id.get(document_id)
        if not document:
            continue
        stored_page_images = {
            int(item["page"]): str(item["uri"])
            for item in image_assets_by_document.get(document_id, {}).get("page_images", [])
            if item.get("page") is not None and item.get("uri")
        }
        for page_number in sorted({int(page) for page in result.get("pages", [])}):
            page_asset: dict[str, Any] = {
                "chunk_id": result.get("chunk_id"),
                "document_id": document_id,
                "version_id": str(document["current_version_id"]),
                "page": page_number,
                "pdf_download_url": document_assets[document_id]["pdf_download_url"],
                "pdf_uri": document_assets[document_id]["pdf_uri"],
            }
            if include_page_images:
                image_uri = stored_page_images.get(page_number) or _page_image_uri(store=store, document=document, page_number=page_number)
                page_asset["page_image_uri"] = image_uri
                page_asset["page_image_url"] = _presigned_s3_url(store, image_uri)
            citation_pages.append(page_asset)
        if include_table_images:
            result_pages = {int(page) for page in result.get("pages", [])}
            for table_image in image_assets_by_document.get(document_id, {}).get("table_images", []):
                if int(table_image.get("page", -1)) not in result_pages:
                    continue
                table_uri = str(table_image.get("uri") or "")
                if not table_uri:
                    continue
                table_assets.append(
                    {
                        "chunk_id": result.get("chunk_id"),
                        "document_id": document_id,
                        "version_id": str(document["current_version_id"]),
                        "page": int(table_image.get("page")),
                        "table_index": table_image.get("table_index"),
                        "bbox": table_image.get("bbox"),
                        "table_image_uri": table_uri,
                        "table_image_url": _presigned_s3_url(store, table_uri),
                        "pdf_download_url": document_assets[document_id]["pdf_download_url"],
                    }
                )
    return {"documents": list(document_assets.values()), "citation_pages": citation_pages, "tables": table_assets}


def _attach_source_assets(
    answer: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    include_page_images: bool,
    include_table_images: bool = False,
) -> dict[str, Any]:
    assets = _source_assets_for_results(
        results,
        include_page_images=include_page_images,
        include_table_images=include_table_images,
    )
    page_assets_by_chunk_id: dict[str, list[dict[str, Any]]] = {}
    for page_asset in assets["citation_pages"]:
        page_assets_by_chunk_id.setdefault(str(page_asset.get("chunk_id")), []).append(page_asset)
    table_assets_by_chunk_id: dict[str, list[dict[str, Any]]] = {}
    for table_asset in assets["tables"]:
        table_assets_by_chunk_id.setdefault(str(table_asset.get("chunk_id")), []).append(table_asset)
    enriched_citations = []
    for citation in answer.get("citations", []):
        chunk_id = str(citation.get("chunk_id") or "")
        enriched_citations.append(
            {
                **citation,
                "source_assets": page_assets_by_chunk_id.get(chunk_id, []),
                "table_assets": table_assets_by_chunk_id.get(chunk_id, []),
            }
        )
    return {**answer, "citations": enriched_citations, "source_assets": assets}


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    ObjectStore().ensure_buckets()
    _ensure_run_tables()
    yield


app = FastAPI(title="Manuals RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow = build_workflow()
debug_workflow = build_workflow(include_answer=False)
debug_query_runs: dict[str, dict[str, Any]] = {}
debug_query_runs_lock = Lock()


def _set_debug_query_run(run_id: str, payload: dict[str, Any]) -> None:
    with debug_query_runs_lock:
        current = dict(debug_query_runs.get(run_id, {}))
        current.update(payload)
        current["updated_at"] = datetime.utcnow().isoformat() + "Z"
        debug_query_runs[run_id] = current
    _update_persisted_run(
        run_id,
        status=current.get("status"),
        progress=current.get("progress"),
        result=current.get("result"),
        error=current.get("error"),
    )


def _run_debug_query_job(run_id: str, request: QueryRequest, sample_limit: int) -> None:
    def progress_callback(progress: dict[str, Any]) -> None:
        _set_debug_query_run(run_id, {"status": progress.get("status", "running"), "progress": progress})

    try:
        _set_debug_query_run(
            run_id,
            {
                "status": "running",
                "progress": {
                    "current_step": DEBUG_QUERY_STEP_SEQUENCE[0][0],
                "current_label": DEBUG_QUERY_STEP_SEQUENCE[0][1],
                "current_model": None,
                "completed_steps": [],
                "total_steps": len(DEBUG_QUERY_STEP_SEQUENCE),
                "step_sequence": [
                    {"name": name, "label": label, "done": False, "model": None}
                    for name, label in DEBUG_QUERY_STEP_SEQUENCE
                ],
                "step_timings_ms": {},
                },
            },
        )
        result = execute_query_debug_run(request, sample_limit=sample_limit, progress_callback=progress_callback)
        _set_debug_query_run(
            run_id,
            {
                "status": "completed",
                "result": result,
                "progress": result.get("progress", {}),
            },
        )
    except Exception as exc:
        _set_debug_query_run(
            run_id,
            {
                "status": "failed",
                "error": str(exc),
            },
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/corpora")
def create_corpus(
    payload: dict[str, Any],
    principal: Principal = Depends(require_role("admin", "operator")),
) -> dict[str, Any]:
    corpus_id = payload.get("id") or str(uuid4())
    _upsert_corpus(corpus_id, principal.tenant_id, payload["name"], payload.get("permissions", {}))
    return {"id": corpus_id}


@app.post("/corpora/{corpus_id}/permissions")
def set_corpus_permissions(
    corpus_id: str,
    payload: dict[str, Any],
    _: Principal = Depends(require_role("admin")),
) -> dict[str, str]:
    execute("update corpora set permissions_json = %s where id = %s", (json_dumps(payload), corpus_id))
    return {"status": "updated"}


@app.post("/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    corpus_id: str | None = Form(None),
    principal: Principal = Depends(require_role("admin", "operator")),
) -> dict[str, Any]:
    store = ObjectStore()
    uploaded = []
    for upload in files:
        data = await upload.read()
        sha = sha256_bytes(data)
        target_corpus_id = corpus_id or settings.default_corpus_id
        _upsert_corpus(
            target_corpus_id,
            principal.tenant_id,
            _default_corpus_name(target_corpus_id),
            update_on_conflict=False,
        )
        existing = fetch_one(
            """
            select sd.id as document_id, sd.current_version_id as version_id, sd.source_filename, sd.corpus_id
            from source_documents sd
            where sd.tenant_id = %s and sd.sha256 = %s and sd.corpus_id = %s
            order by sd.updated_at desc
            limit 1
            """,
            (principal.tenant_id, sha, target_corpus_id),
        )
        if existing:
            uploaded.append(
                {
                    "document_id": str(existing["document_id"]),
                    "version_id": str(existing["version_id"]),
                    "filename": existing["source_filename"],
                    "corpus_id": existing["corpus_id"],
                    "duplicate": True,
                }
            )
            continue
        source_document_id = str(uuid4())
        version_id = str(uuid4())
        object_name = f"{principal.tenant_id}/{source_document_id}/{upload.filename}"
        object_name = _storage_object_name(principal.tenant_id, sha, upload.filename)
        storage_uri = store.put_bytes(
            settings.minio_bucket_originals,
            object_name,
            data,
            upload.content_type or "application/octet-stream",
        ) if not store.object_exists(settings.minio_bucket_originals, object_name) else f"s3://{settings.minio_bucket_originals}/{object_name}"
        metadata = infer_document_metadata(upload.filename, "")
        payload = SourceDocumentCreate(
            tenant_id=principal.tenant_id,
            corpus_id=target_corpus_id,
            title=metadata.title,
            manufacturer=metadata.manufacturer,
            product_family=metadata.product_family,
            product_model=metadata.product_model,
            document_kind=metadata.document_kind,
            source_filename=upload.filename,
            mime_type=upload.content_type or "application/pdf",
        )
        execute(
            """
            insert into source_documents (
                id, tenant_id, corpus_id, document_kind, title, manufacturer, product_family,
                product_model, language, mime_type, source_filename, storage_uri, sha256,
                file_size_bytes, ingest_status, visibility_scope, permissions_tags, created_at, updated_at
            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now(), now())
            """,
            (
                source_document_id,
                payload.tenant_id,
                payload.corpus_id,
                payload.document_kind.value,
                payload.title,
                payload.manufacturer,
                payload.product_family,
                payload.product_model,
                payload.language,
                payload.mime_type,
                payload.source_filename,
                storage_uri,
                sha,
                len(data),
                "uploaded",
                payload.visibility_scope,
                json_dumps(payload.permissions_tags),
            ),
        )
        execute(
            """
            insert into document_versions (
                id, source_document_id, version_label, revision_date, effective_date, status,
                docling_artifact_uri, page_count, parse_profile, ocr_used, table_extraction_used,
                parse_warnings, quality_score, ingested_at
            ) values (%s, %s, %s, null, null, 'uploaded', null, 0, null, false, false, '[]'::jsonb, 0, now())
            """,
            (version_id, source_document_id, "initial"),
        )
        execute(
            "update source_documents set current_version_id = %s where id = %s",
            (version_id, source_document_id),
        )
        uploaded.append({"document_id": source_document_id, "version_id": version_id, "filename": upload.filename, "duplicate": False})
    return {"uploaded": uploaded}


@app.post("/documents/{document_id}/ingest")
def ingest_document(document_id: str, _: Principal = Depends(require_role("admin", "operator"))) -> dict[str, str]:
    source = fetch_one("select current_version_id from source_documents where id = %s", (document_id,))
    if not source:
        raise HTTPException(status_code=404, detail="Document not found.")
    run_id = str(uuid4())
    execute(
        """
        insert into ingestion_runs (id, source_document_id, document_version_id, status, failure_class, created_at, updated_at)
        values (%s, %s, %s, 'queued', null, now(), now())
        """,
        (run_id, document_id, source["current_version_id"]),
    )
    enqueue("ingest_jobs", {"run_id": run_id, "document_id": document_id, "version_id": source["current_version_id"]})
    return {"run_id": run_id}


@app.post("/documents/{document_id}/reingest")
def reingest_document(document_id: str, _: Principal = Depends(require_role("admin", "operator"))) -> dict[str, str]:
    return ingest_document(document_id)


@app.get("/documents/{document_id}")
def get_document(document_id: str, _: Principal = Depends(require_role("end_user", "operator", "admin", "auditor"))) -> dict[str, Any]:
    document = fetch_one("select * from source_documents where id = %s", (document_id,))
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@app.get("/documents/{document_id}/versions")
def list_versions(document_id: str, _: Principal = Depends(require_role("end_user", "operator", "admin", "auditor"))) -> list[dict[str, Any]]:
    return fetch_all("select * from document_versions where source_document_id = %s order by ingested_at desc", (document_id,))


@app.get("/ingestion-runs/{run_id}")
def get_ingestion_run(run_id: str, _: Principal = Depends(require_role("operator", "admin", "auditor"))) -> dict[str, Any]:
    run = fetch_one("select * from ingestion_runs where id = %s", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@app.post("/query")
def query_documents(
    request: QueryRequest,
    _: Principal = Depends(require_role("end_user", "operator", "admin", "auditor")),
) -> JSONResponse:
    with QUERY_DURATION.labels("full").time():
        result = workflow.invoke(
            {"query": request.query, "corpus_ids": request.corpus_ids, "filters": request.filters}
        )
    answer = dict(result["answer"])
    if request.include_source_assets or request.include_page_images or request.include_table_images:
        answer = _attach_source_assets(
            answer,
            [dict(item) for item in result.get("retrieval_results", [])],
            include_page_images=request.include_page_images,
            include_table_images=request.include_table_images,
        )
    return JSONResponse(answer)


@app.post("/search")
def search_documents(
    request: QueryRequest,
    _: Principal = Depends(require_role("end_user", "operator", "admin", "auditor")),
) -> list[dict[str, Any]] | dict[str, Any]:
    filters = build_filters(request.query, request.filters)
    results = [item.model_dump() for item in retrieve(request.query, request.corpus_ids, filters)]
    if request.include_source_assets or request.include_page_images or request.include_table_images:
        return {
            "results": results,
            "source_assets": _source_assets_for_results(
                results,
                include_page_images=request.include_page_images,
                include_table_images=request.include_table_images,
            ),
        }
    return results


@app.post("/explain-retrieval")
def explain_retrieval(
    request: QueryRequest,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    filters = build_filters(request.query, request.filters)
    results = [item.model_dump() for item in retrieve(request.query, request.corpus_ids, filters)]
    return {
        "applied_filters": filters,
        "reranked_top_results": results,
    }


@app.post("/eval/end-to-end")
def run_end_to_end_eval(
    payload: dict[str, Any],
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    run_id = str(uuid4())
    _create_persisted_run(run_id, "end_to_end_eval", payload)
    corpus_ids, document_id, cases, warnings = _load_eval_cases(payload)
    if not cases:
        result = {
            "run_id": run_id,
            "scope": {"corpus_ids": corpus_ids, "document_id": document_id},
            "summary": _summarize_end_to_end_eval([]),
            "items": [],
            "warnings": warnings,
        }
        _update_persisted_run(run_id, status="completed", result=result)
        return result

    items: list[dict[str, Any]] = []
    for case in cases:
        search_corpus_ids, filters = _eval_search_scope(corpus_ids, document_id)
        search_results = [item.model_dump() for item in retrieve(case.query, search_corpus_ids, build_filters(case.query, filters))]
        retrieval_evaluation = score_search_results(case, search_results)
        answer_result = workflow.invoke({"query": case.query, "corpus_ids": search_corpus_ids, "filters": filters})
        answer = dict(answer_result["answer"])
        answer_evaluation = _score_answer(case, answer, retrieval_evaluation)
        items.append(
            {
                "case": case.to_dict(),
                "retrieval_evaluation": retrieval_evaluation,
                "answer": answer,
                "answer_evaluation": answer_evaluation,
                "top_results": search_results[:5],
            }
        )

    result = {
        "run_id": run_id,
        "scope": {"corpus_ids": corpus_ids, "document_id": document_id},
        "summary": _summarize_end_to_end_eval(items),
        "items": items,
        "warnings": warnings,
    }
    _update_persisted_run(run_id, status="completed", result=result)
    return result


@app.post("/eval/end-to-end-stream")
def stream_end_to_end_eval(
    payload: dict[str, Any],
    sample_limit: int = 10,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> StreamingResponse:
    run_id = str(uuid4())
    _create_persisted_run(run_id, "end_to_end_eval", payload)
    bounded_sample_limit = max(5, min(sample_limit, 100))
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=500)
    queued_event = {
        "run_id": run_id,
        "event": "eval_queued",
        "scope": {
            "corpus_ids": payload.get("corpus_ids") or [],
            "document_id": payload.get("document_id"),
        },
        "sample_limit": bounded_sample_limit,
    }
    _append_run_event(run_id, 1, queued_event)
    _update_persisted_run(run_id, status="queued", progress=queued_event)

    def publish(event: dict[str, Any]) -> None:
        try:
            event_queue.put_nowait(event)
        except queue.Full:
            pass
    publish(queued_event)

    def run_eval() -> None:
        event_index = 1

        def emit(event: dict[str, Any]) -> Any:
            nonlocal event_index
            event_index += 1
            event_with_run = {"run_id": run_id, **event}
            _append_run_event(run_id, event_index, event_with_run)
            publish(event_with_run)
            return event_with_run

        items: list[dict[str, Any]] = []
        try:
            corpus_ids, document_id, cases, warnings = _load_eval_cases(payload)
            start_event = emit(
                {
                    "event": "eval_started",
                    "scope": {"corpus_ids": corpus_ids, "document_id": document_id},
                    "total_questions": len(cases),
                    "warnings": warnings,
                }
            )
            _update_persisted_run(run_id, status="running", progress=start_event)
            if not cases:
                result = {
                    "run_id": run_id,
                    "scope": {"corpus_ids": corpus_ids, "document_id": document_id},
                    "summary": _summarize_end_to_end_eval([]),
                    "items": [],
                    "warnings": warnings,
                }
                completed = emit({"event": "eval_completed", "result": result})
                _update_persisted_run(run_id, status="completed", progress=completed, result=result)
                return

            for question_index, case in enumerate(cases, start=1):
                search_corpus_ids, filters = _eval_search_scope(corpus_ids, document_id)
                question_started = emit(
                    {
                        "event": "eval_question_started",
                        "question_index": question_index,
                        "total_questions": len(cases),
                        "case": case.to_dict(),
                    }
                )
                _update_persisted_run(run_id, status="running", progress=question_started)
                query_request = QueryRequest(
                    query=case.query,
                    corpus_ids=search_corpus_ids,
                    filters=filters,
                    response_mode="answer_with_citations",
                )
                final_query_result: dict[str, Any] = {}
                for event in stream_query_debug_events(query_request, sample_limit=bounded_sample_limit):
                    nested = emit(
                        {
                            "event": "eval_query_event",
                            "question_index": question_index,
                            "query_event": event,
                        }
                    )
                    if event.get("event") == "run_completed":
                        final_query_result = dict(event.get("result") or {})
                if not final_query_result:
                    raise RuntimeError(f"Question {question_index} query stream ended without a completed result.")
                top_results = _top_results_from_query_debug(final_query_result)
                retrieval_evaluation = score_search_results(case, top_results)
                answer = dict(final_query_result.get("answer") or {})
                answer_evaluation = _score_answer(case, answer, retrieval_evaluation)
                item = {
                    "case": case.to_dict(),
                    "retrieval_evaluation": retrieval_evaluation,
                    "answer": answer,
                    "answer_evaluation": answer_evaluation,
                    "top_results": top_results[:5],
                    "query_debug_result": final_query_result,
                }
                items.append(item)
                question_completed = emit(
                    {
                        "event": "eval_question_completed",
                        "question_index": question_index,
                        "summary": _summarize_end_to_end_eval(items),
                        "item": item,
                    }
                )
                _update_persisted_run(run_id, status="running", progress=question_completed)

            result = {
                "run_id": run_id,
                "scope": {"corpus_ids": corpus_ids, "document_id": document_id},
                "summary": _summarize_end_to_end_eval(items),
                "items": items,
                "warnings": warnings,
            }
            completed = emit({"event": "eval_completed", "result": result})
            _update_persisted_run(run_id, status="completed", progress=completed, result=result)
        except Exception as exc:
            failed = emit({"event": "eval_failed", "error": str(exc)})
            _update_persisted_run(run_id, status="failed", progress=failed, error=str(exc))
        finally:
            publish(None)

    Thread(target=run_eval, name=f"end-to-end-eval-{run_id}", daemon=True).start()

    def iter_events() -> Any:
        while True:
            event = event_queue.get()
            if event is None:
                return
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(iter_events(), media_type="application/x-ndjson")


@app.get("/runs")
def list_app_runs(
    run_type: str | None = None,
    limit: int = 25,
    include_result: bool = True,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> list[dict[str, Any]]:
    _fail_stale_running_runs()
    params: list[Any] = []
    where = ""
    if run_type:
        where = "where run_type = %s"
        params.append(run_type)
    params.append(max(1, min(limit, 100)))
    progress_select = (
        "progress_json"
        if include_result
        else """
            jsonb_strip_nulls(
                jsonb_build_object(
                    'summary',
                    coalesce(progress_json -> 'summary', progress_json #> '{result,summary}', result_json -> 'summary')
                )
            ) as progress_json
        """
    )
    result_select = "result_json" if include_result else "null::jsonb as result_json"
    return fetch_all(
        f"""
        select id, run_type, status, request_json, {progress_select}, {result_select}, error, created_at, updated_at
        from app_runs
        {where}
        order by updated_at desc
        limit %s
        """,
        tuple(params),
    )


@app.get("/runs/{run_id}")
def get_app_run(
    run_id: str,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    _fail_stale_running_runs()
    run = fetch_one("select * from app_runs where id = %s", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@app.get("/runs/{run_id}/events")
def list_app_run_events(
    run_id: str,
    after: int = 0,
    limit: int = 500,
    tail: bool = False,
    compact: bool = False,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(limit, 2000))
    compact_where = ""
    if compact:
        compact_where = """
            and coalesce(event_json #>> '{query_event,event}', '') not in ('llm_token')
        """
    if tail:
        rows = fetch_all(
            f"""
            select event_index, event_json, created_at
            from app_run_events
            where run_id = %s and event_index > %s
            {compact_where}
            order by event_index desc
            limit %s
            """,
            (run_id, after, bounded_limit),
        )
        return list(reversed(rows))
    return fetch_all(
        f"""
        select event_index, event_json, created_at
        from app_run_events
        where run_id = %s and event_index > %s
        {compact_where}
        order by event_index asc
        limit %s
        """,
        (run_id, after, bounded_limit),
    )


@app.get("/debug/documents")
def debug_documents(
    limit: int = 50,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> list[dict[str, Any]]:
    return list_recent_documents(limit=max(1, min(limit, 200)))


@app.get("/debug/ingestion-status")
def debug_ingestion_status(
    limit: int = 50,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    bounded_limit = max(1, min(limit, 200))
    document_status = fetch_all(
        """
        select corpus_id, ingest_status, count(*)::int as count
        from source_documents
        group by corpus_id, ingest_status
        order by corpus_id, ingest_status
        """
    )
    run_status = fetch_all(
        """
        select status, count(*)::int as count
        from ingestion_runs
        group by status
        order by status
        """
    )
    recent_runs = fetch_all(
        """
        select
            ir.id as run_id,
            ir.status,
            ir.failure_class,
            ir.failure_reason,
            ir.created_at,
            ir.updated_at,
            sd.id as document_id,
            sd.corpus_id,
            sd.source_filename,
            sd.ingest_status,
            dv.page_count,
            (
                select count(*)::int
                from retrieval_chunks rc
                where rc.source_document_id = sd.id
            ) as chunk_count
        from ingestion_runs ir
        join source_documents sd on sd.id = ir.source_document_id
        left join document_versions dv on dv.id = ir.document_version_id
        order by ir.updated_at desc
        limit %s
        """,
        (bounded_limit,),
    )
    recent_documents = fetch_all(
        """
        select
            sd.id as document_id,
            sd.corpus_id,
            sd.source_filename,
            sd.ingest_status,
            sd.updated_at,
            dv.page_count,
            (
                select count(*)::int
                from retrieval_chunks rc
                where rc.source_document_id = sd.id
            ) as chunk_count
        from source_documents sd
        left join document_versions dv on dv.id = sd.current_version_id
        order by sd.updated_at desc
        limit %s
        """,
        (bounded_limit,),
    )
    redis = redis_client()
    queues = {
        "ingest_jobs": redis.llen("ingest_jobs"),
        "embed_jobs": redis.llen("embed_jobs"),
        "reindex_jobs": redis.llen("reindex_jobs"),
    }
    totals = {
        "documents": sum(int(row["count"]) for row in document_status),
        "runs": sum(int(row["count"]) for row in run_status),
    }
    return {
        "document_status": document_status,
        "run_status": run_status,
        "queues": queues,
        "totals": totals,
        "recent_runs": recent_runs,
        "recent_documents": recent_documents,
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/debug/ollama-calls")
def debug_ollama_calls(
    limit: int = 50,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> list[dict[str, Any]]:
    return recent_ollama_calls(limit=max(1, min(limit, 200)))


@app.post("/debug/model-prompt")
def debug_model_prompt(
    payload: dict[str, Any],
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    model = str(payload.get("model") or settings.ollama_fast_model)
    system_prompt = str(payload.get("system_prompt") or "").strip()
    user_prompt = str(payload.get("user_prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="user_prompt is required.")
    raw_think = payload.get("think", None)
    think = raw_think if isinstance(raw_think, bool) else None
    json_mode = bool(payload.get("json_mode", False))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    json_schema = {"type": "object"} if json_mode else None
    request_payload = build_chat_payload(
        model=model,
        messages=messages,
        json_schema=json_schema,
        think=think,
    )
    started = perf_counter()
    with httpx.Client(base_url=settings.ollama_url, timeout=180.0) as client:
        ensure_model_loaded(client=client, model=model, purpose="model_prompt_test")
        response = client.post("/api/chat", json=request_payload)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
    duration_ms = round((perf_counter() - started) * 1000, 2)
    raw_content = str((body.get("message") or {}).get("content") or "")
    cleaned_content = extract_chat_content(body)
    return {
        "model_requested": model,
        "model_response": body.get("model"),
        "think_requested": think,
        "json_mode": json_mode,
        "duration_ms": duration_ms,
        "raw_content": raw_content,
        "cleaned_content": cleaned_content,
        "raw_contains_think_tag": "<think" in raw_content.lower() or "</think>" in raw_content.lower(),
        "cleaned_contains_think_tag": "<think" in cleaned_content.lower() or "</think>" in cleaned_content.lower(),
        "ollama_response": body,
        "request_payload": {
            **request_payload,
            "messages": [
                {"role": message.get("role"), "content": str(message.get("content") or "")[:1000]}
                for message in request_payload.get("messages", [])
            ],
        },
    }


@app.get("/debug/documents/{document_id}")
def debug_document_snapshot(
    document_id: str,
    sample_limit: int = 25,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    snapshot = build_document_debug_snapshot(document_id, sample_limit=max(1, min(sample_limit, 200)))
    if not snapshot:
        raise HTTPException(status_code=404, detail="Document not found.")
    return snapshot


@app.get("/debug/documents/{document_id}/metadata")
def debug_document_metadata_snapshot(
    document_id: str,
    page: int | None = None,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    snapshot = build_document_metadata_snapshot(document_id, page=page)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Document not found.")
    return snapshot


@app.post("/debug/query")
def debug_query_pipeline(
    request: QueryRequest,
    sample_limit: int = 10,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    return execute_query_debug_run(request, sample_limit=max(1, min(sample_limit, 100)))


@app.post("/debug/query-stream")
def debug_query_pipeline_stream(
    request: QueryRequest,
    sample_limit: int = 10,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> StreamingResponse:
    bounded_sample_limit = max(1, min(sample_limit, 100))

    def iter_events() -> Any:
        for event in stream_query_debug_events(request, sample_limit=bounded_sample_limit):
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(iter_events(), media_type="application/x-ndjson")


@app.post("/debug/query-runs")
def create_debug_query_run(
    request: QueryRequest,
    sample_limit: int = 10,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    bounded_sample_limit = max(1, min(sample_limit, 100))
    run_id = str(uuid4())
    _create_persisted_run(
        run_id,
        "query_debug",
        {
            "query": request.query,
            "corpus_ids": request.corpus_ids,
            "filters": request.filters,
            "sample_limit": bounded_sample_limit,
        },
    )
    _set_debug_query_run(
        run_id,
        {
            "run_id": run_id,
            "status": "queued",
            "progress": {
                "current_step": None,
                "current_label": "Queued",
                "current_model": None,
                "completed_steps": [],
                "total_steps": len(DEBUG_QUERY_STEP_SEQUENCE),
                "step_sequence": [
                    {"name": name, "label": label, "done": False, "model": None}
                    for name, label in DEBUG_QUERY_STEP_SEQUENCE
                ],
                "step_timings_ms": {},
            },
        },
    )
    Thread(target=_run_debug_query_job, args=(run_id, request, bounded_sample_limit), daemon=True).start()
    return {"run_id": run_id, "status": "queued"}


@app.get("/debug/query-runs/{run_id}")
def get_debug_query_run(
    run_id: str,
    _: Principal = Depends(require_role("operator", "admin", "auditor")),
) -> dict[str, Any]:
    with debug_query_runs_lock:
        payload = debug_query_runs.get(run_id)
    if not payload:
        persisted = fetch_one("select * from app_runs where id = %s and run_type = 'query_debug'", (run_id,))
        if not persisted:
            raise HTTPException(status_code=404, detail="Debug query run not found.")
        return {
            "run_id": str(persisted["id"]),
            "status": persisted["status"],
            "progress": persisted.get("progress_json") or {},
            "result": persisted.get("result_json"),
            "error": persisted.get("error"),
            "created_at": persisted.get("created_at"),
            "updated_at": persisted.get("updated_at"),
        }
    return payload


@app.post("/feedback")
def store_feedback(payload: dict[str, Any], _: Principal = Depends(require_role("end_user", "operator", "admin", "auditor"))) -> dict[str, str]:
    execute(
        "insert into feedback (id, payload_json, created_at) values (%s, %s::jsonb, now())",
        (str(uuid4()), json.dumps(payload)),
    )
    return {"status": "stored"}
