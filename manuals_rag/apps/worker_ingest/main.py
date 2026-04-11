from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

import fitz

from manuals_rag_chunking.hierarchical import build_chunks
from manuals_rag_common.db import execute, execute_many, fetch_one, json_dumps
from manuals_rag_common.ids import sha256_bytes
from manuals_rag_common.logging import configure_logging
from manuals_rag_common.queue import dequeue, enqueue
from manuals_rag_common.storage import ObjectStore
from manuals_rag_normalizers.normalize import normalize_nodes
from manuals_rag_observability.metrics import INGEST_DURATION, PARSE_FAILURES
from manuals_rag_parsers.docling_parser import parse_document
from manuals_rag_parsers.metadata import infer_document_metadata
from manuals_rag_schemas.enums import NodeType

log = logging.getLogger(__name__)
PAGE_IMAGE_SCALE = 1.5
TABLE_IMAGE_SCALE = 2.0


def _read_minio_uri(uri: str) -> bytes:
    parsed = urlparse(uri)
    store = ObjectStore()
    response = store.client.get_object(parsed.netloc, parsed.path.lstrip("/"))
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _artifact_object_name(tenant_id: str, artifact_bytes: bytes) -> str:
    return f"{tenant_id}/artifacts-sha256/{sha256_bytes(artifact_bytes)}.json"


def _asset_object_prefix(tenant_id: str, source_document_id: str, version_id: str) -> str:
    return f"{tenant_id}/document-assets/{source_document_id}/{version_id}"


def _put_once(store: ObjectStore, bucket: str, object_name: str, data: bytes, content_type: str) -> str:
    if not store.object_exists(bucket, object_name):
        return store.put_bytes(bucket, object_name, data, content_type)
    return f"s3://{bucket}/{object_name}"


def _render_page_image(page: fitz.Page, *, scale: float = PAGE_IMAGE_SCALE, clip: fitz.Rect | None = None) -> bytes:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, clip=clip)
    return pixmap.tobytes("png")


def _docling_page_no(raw_page_no: int, *, batch_start: int, batch_end: int) -> int:
    batch_size = batch_end - batch_start + 1
    if batch_start > 1 and 1 <= raw_page_no <= batch_size:
        return batch_start + raw_page_no - 1
    return raw_page_no


def _table_bbox_rect(page: fitz.Page, bbox: dict[str, object]) -> fitz.Rect | None:
    try:
        left = float(bbox.get("l", bbox.get("x0")))
        right = float(bbox.get("r", bbox.get("x1")))
        top = float(bbox.get("t", bbox.get("y0")))
        bottom = float(bbox.get("b", bbox.get("y1")))
    except (TypeError, ValueError):
        return None
    origin = str(bbox.get("coord_origin") or "").lower()
    if "bottom" in origin:
        y0 = page.rect.height - top
        y1 = page.rect.height - bottom
    else:
        y0 = top
        y1 = bottom
    x0, x1 = sorted((left, right))
    y0, y1 = sorted((y0, y1))
    rect = fitz.Rect(x0, y0, x1, y1) + (-4, -4, 4, 4)
    rect = rect & page.rect
    if rect.is_empty or rect.width <= 1 or rect.height <= 1:
        return None
    return rect


def _docling_table_specs(artifact: dict[str, object]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for batch in artifact.get("batches", []):
        if not isinstance(batch, dict):
            continue
        batch_start, batch_end = batch.get("page_range", [1, 1])
        document = batch.get("document") or {}
        if not isinstance(document, dict):
            continue
        for table_index, table in enumerate(document.get("tables", []) or [], start=1):
            if not isinstance(table, dict):
                continue
            prov = table.get("prov") or []
            if not prov or not isinstance(prov[0], dict):
                continue
            raw_page_no = int(prov[0].get("page_no") or batch_start)
            bbox = prov[0].get("bbox")
            if not isinstance(bbox, dict):
                continue
            specs.append(
                {
                    "table_index": len(specs) + 1,
                    "batch_table_index": table_index,
                    "page": _docling_page_no(raw_page_no, batch_start=int(batch_start), batch_end=int(batch_end)),
                    "bbox": bbox,
                }
            )
    return specs


def _store_document_images(
    *,
    store: ObjectStore,
    raw_pdf: bytes,
    artifact: dict[str, object],
    tenant_id: str,
    source_document_id: str,
    version_id: str,
) -> dict[str, list[dict[str, object]]]:
    bucket = "manuals-artifacts"
    prefix = _asset_object_prefix(tenant_id, source_document_id, version_id)
    page_images: list[dict[str, object]] = []
    table_images: list[dict[str, object]] = []
    table_specs_by_page: dict[int, list[dict[str, object]]] = {}
    for spec in _docling_table_specs(artifact):
        table_specs_by_page.setdefault(int(spec["page"]), []).append(spec)

    pdf = fitz.open(stream=raw_pdf, filetype="pdf")
    try:
        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            page = pdf.load_page(page_index)
            page_object_name = f"{prefix}/page-images/page-{page_number:04d}.png"
            page_uri = _put_once(store, bucket, page_object_name, _render_page_image(page), "image/png")
            page_images.append({"page": page_number, "uri": page_uri})

            for spec in table_specs_by_page.get(page_number, []):
                rect = _table_bbox_rect(page, spec["bbox"])
                if rect is None:
                    continue
                table_object_name = f"{prefix}/table-images/table-{int(spec['table_index']):04d}-page-{page_number:04d}.png"
                table_uri = _put_once(
                    store,
                    bucket,
                    table_object_name,
                    _render_page_image(page, scale=TABLE_IMAGE_SCALE, clip=rect),
                    "image/png",
                )
                table_images.append(
                    {
                        "table_index": int(spec["table_index"]),
                        "page": page_number,
                        "uri": table_uri,
                        "bbox": spec["bbox"],
                    }
                )
    finally:
        pdf.close()
    return {"page_images": page_images, "table_images": table_images}


def process_job(job: dict[str, str]) -> None:
    run_id = job["run_id"]
    document = fetch_one(
        """
        select sd.*, dv.id as version_id, dv.version_label
        from source_documents sd
        join document_versions dv on dv.id = sd.current_version_id
        where sd.id = %s
        """,
        (job["document_id"],),
    )
    if not document:
        raise ValueError("Source document not found.")
    execute("update ingestion_runs set status = 'running', updated_at = now() where id = %s", (run_id,))
    try:
        with INGEST_DURATION.labels("parse").time():
            raw = _read_minio_uri(document["storage_uri"])
            result = parse_document(document["version_id"], document["source_filename"], raw)
        normalized = normalize_nodes(result.logical_nodes)
        table_extraction_used = any(node.node_type == NodeType.table for node in normalized)
        combined_text = "\n\n".join(node.text_normalized or node.text_raw for node in normalized[:20])
        inferred_metadata = infer_document_metadata(document["source_filename"], combined_text)
        metadata = {
            "tenant_id": document["tenant_id"],
            "corpus_id": document["corpus_id"],
            "document_kind": inferred_metadata.document_kind.value,
            "manufacturer": inferred_metadata.manufacturer if inferred_metadata.manufacturer != "Unknown" else document["manufacturer"],
            "companies": inferred_metadata.companies,
            "product_family": inferred_metadata.product_family or document["product_family"],
            "product_model": inferred_metadata.product_model or document["product_model"],
            "product_families": inferred_metadata.product_families,
            "product_models": inferred_metadata.product_models or ([document["product_model"]] if document["product_model"] else []),
            "devices": inferred_metadata.devices,
            "part_numbers": inferred_metadata.part_numbers,
            "document_protocol_terms": inferred_metadata.protocol_terms,
            "settings": inferred_metadata.settings,
            "parameters": inferred_metadata.parameters,
            "document_menu_labels": inferred_metadata.menu_labels,
            "document_topics": inferred_metadata.document_topics,
            "language": document["language"],
            "visibility_scope": document["visibility_scope"],
            "permissions_tags": document["permissions_tags"] or [],
            "version_label": document["version_label"],
            "revision_date": inferred_metadata.revision_date.isoformat() if inferred_metadata.revision_date else None,
            "ingest_run_id": run_id,
            "parse_profile": result.profile.value,
            "ocr_used": False,
            "is_active": True,
        }
        chunks = build_chunks(
            source_document_id=document["id"],
            document_version_id=document["version_id"],
            title=document["title"],
            nodes=normalized,
            metadata=metadata,
        )
        store = ObjectStore()
        result.docling_artifact["image_assets"] = _store_document_images(
            store=store,
            raw_pdf=raw,
            artifact=result.docling_artifact,
            tenant_id=str(document["tenant_id"]),
            source_document_id=str(document["id"]),
            version_id=str(document["version_id"]),
        )
        artifact_bytes = json.dumps(result.docling_artifact, sort_keys=True).encode("utf-8")
        artifact_object_name = _artifact_object_name(str(document["tenant_id"]), artifact_bytes)
        artifact_uri = (
            store.put_bytes(
                "manuals-artifacts",
                artifact_object_name,
                artifact_bytes,
                "application/json",
            )
            if not store.object_exists("manuals-artifacts", artifact_object_name)
            else f"s3://manuals-artifacts/{artifact_object_name}"
        )
        execute("delete from logical_nodes where document_version_id = %s", (document["version_id"],))
        execute("delete from retrieval_chunks where document_version_id = %s", (document["version_id"],))
        execute_many(
            """
            insert into logical_nodes (
                id, document_version_id, node_type, ordinal, depth, heading_text, section_path_json,
                page_from, page_to, text_raw, text_normalized, table_json, caption_text, warning_level,
                procedure_step_number, spec_name, spec_value, spec_unit, keywords_json, citability_score, token_count
            ) values (
                %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s
            )
            """,
            [
                (
                    node.id,
                    node.document_version_id,
                    node.node_type.value,
                    node.ordinal,
                    node.depth,
                    node.heading_text,
                    json_dumps(node.section_path_json),
                    node.page_from,
                    node.page_to,
                    node.text_raw,
                    node.text_normalized,
                    json_dumps(node.table_json),
                    node.caption_text,
                    node.warning_level,
                    node.procedure_step_number,
                    node.spec_name,
                    node.spec_value,
                    node.spec_unit,
                    json_dumps(node.keywords_json),
                    node.citability_score,
                    node.token_count,
                )
                for node in normalized
            ],
        )
        execute_many(
            """
            insert into retrieval_chunks (
                id, document_version_id, source_document_id, logical_node_ids_json, chunk_type, chunk_level,
                title, section_path_text, page_from, page_to, content, content_for_sparse, content_for_dense,
                content_for_rerank, metadata_json, is_active, priority_score
            ) values (
                %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s
            )
            """,
            [
                (
                    chunk.id,
                    chunk.document_version_id,
                    chunk.source_document_id,
                    json_dumps(chunk.logical_node_ids_json),
                    chunk.chunk_type.value,
                    chunk.chunk_level,
                    chunk.title,
                    chunk.section_path_text,
                    chunk.page_from,
                    chunk.page_to,
                    chunk.content,
                    chunk.content_for_sparse,
                    chunk.content_for_dense,
                    chunk.content_for_rerank,
                    json_dumps(chunk.metadata_json),
                    chunk.is_active,
                    chunk.priority_score,
                )
                for chunk in chunks
            ],
        )
        execute(
            """
            update document_versions
            set status = 'parsed',
                docling_artifact_uri = %s,
                page_count = %s,
                parse_profile = %s,
                revision_date = %s,
                effective_date = %s,
                ocr_used = false,
                table_extraction_used = %s,
                parse_warnings = %s::jsonb,
                quality_score = %s,
                ingested_at = now()
            where id = %s
            """,
            (
                artifact_uri,
                result.page_count,
                result.profile.value,
                inferred_metadata.revision_date,
                inferred_metadata.effective_date,
                table_extraction_used,
                json.dumps(result.parse_warnings),
                result.quality_score,
                document["version_id"],
            ),
        )
        execute(
            """
            update source_documents
            set ingest_status = 'parsed',
                manufacturer = %s,
                product_family = %s,
                product_model = %s,
                document_kind = %s,
                title = %s,
                updated_at = now()
            where id = %s
            """,
            (
                metadata["manufacturer"],
                metadata["product_family"],
                metadata["product_model"],
                metadata["document_kind"],
                inferred_metadata.title,
                document["id"],
            ),
        )
        execute("update ingestion_runs set status = 'parsed', updated_at = now() where id = %s", (run_id,))
        enqueue("embed_jobs", {"run_id": run_id, "document_id": document["id"], "version_id": document["version_id"]})
    except Exception as exc:
        PARSE_FAILURES.labels("PARSE_FAILED").inc()
        execute(
            "update ingestion_runs set status = 'failed', failure_class = 'PARSE_FAILED', failure_reason = %s, updated_at = now() where id = %s",
            (str(exc), run_id),
        )
        raise


def main() -> None:
    configure_logging()
    log.info("worker_ingest started")
    while True:
        job = dequeue("ingest_jobs", timeout=5)
        if not job:
            continue
        process_job(job)


if __name__ == "__main__":
    main()
