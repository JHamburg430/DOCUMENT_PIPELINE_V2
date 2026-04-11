from __future__ import annotations

from collections import Counter
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from manuals_rag_chunking.hierarchical import build_chunks
from manuals_rag_evals.retrieval_quality import content_quality_flags
from manuals_rag_normalizers.normalize import normalize_nodes
from manuals_rag_parsers.docling_parser import parse_document
from manuals_rag_parsers.metadata import infer_document_metadata
from manuals_rag_retrieval.embeddings import build_sparse_vector, embed_dense
from manuals_rag_retrieval.qdrant_store import QdrantStore, collection_name
from manuals_rag_retrieval.query_analysis import analyze_query
from manuals_rag_retrieval.retriever import build_filters
from manuals_rag_schemas.enums import NodeType


@dataclass
class StageCheckResult:
    name: str
    status: str
    duration_ms: int
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineHealthReport:
    fixture_path: str
    generated_at: str
    overall_status: str
    passed: int
    failed: int
    skipped: int
    stage_results: list[StageCheckResult]


def _result(name: str, start: float, *, status: str, details: dict[str, Any] | None = None, error: str | None = None) -> StageCheckResult:
    return StageCheckResult(
        name=name,
        status=status,
        duration_ms=int((time.perf_counter() - start) * 1000),
        details=details or {},
        error=error,
    )


def check_fixture_path(pdf_path: Path) -> StageCheckResult:
    start = time.perf_counter()
    exists = pdf_path.exists()
    return _result(
        "fixture_path",
        start,
        status="pass" if exists else "fail",
        details={"path": str(pdf_path), "exists": exists, "size_bytes": pdf_path.stat().st_size if exists else 0},
        error=None if exists else "Fixture PDF is missing.",
    )


def check_parse_stage(pdf_path: Path) -> tuple[StageCheckResult, Any | None]:
    start = time.perf_counter()
    try:
        result = parse_document("pipeline-health-version", pdf_path.name, pdf_path.read_bytes())
        return _result(
            "parse_document",
            start,
            status="pass",
            details={
                "page_count": result.page_count,
                "logical_node_count": len(result.logical_nodes),
                "parse_profile": result.profile.value,
                "parse_warnings": result.parse_warnings,
                "quality_score": result.quality_score,
            },
        ), result
    except Exception as exc:
        return _result("parse_document", start, status="fail", error=str(exc)), None


def check_page_provenance_stage(parsed: Any) -> StageCheckResult:
    start = time.perf_counter()
    try:
        logical_pages = sorted({node.page_from for node in parsed.logical_nodes})
        page_count = int(parsed.page_count or 0)
        status = "pass"
        error = None
        if page_count > 1 and len(logical_pages) <= 1:
            status = "fail"
            error = "Parsed logical nodes collapse onto a single page."
        elif logical_pages and (min(logical_pages) < 1 or max(logical_pages) > page_count):
            status = "fail"
            error = "Parsed logical node pages fall outside the source page range."
        return _result(
            "page_provenance",
            start,
            status=status,
            details={
                "page_count": page_count,
                "distinct_logical_pages": len(logical_pages),
                "first_page": logical_pages[0] if logical_pages else None,
                "last_page": logical_pages[-1] if logical_pages else None,
            },
            error=error,
        )
    except Exception as exc:
        return _result("page_provenance", start, status="fail", error=str(exc))


def check_metadata_stage(pdf_path: Path, parsed: Any) -> tuple[StageCheckResult, Any | None]:
    start = time.perf_counter()
    try:
        combined_text = "\n\n".join(node.text_normalized or node.text_raw for node in parsed.logical_nodes[:20])
        metadata = infer_document_metadata(pdf_path.name, combined_text)
        return _result(
            "infer_document_metadata",
            start,
            status="pass",
            details={
                "title": metadata.title,
                "manufacturer": metadata.manufacturer,
                "product_family": metadata.product_family,
                "product_model": metadata.product_model,
                "document_kind": metadata.document_kind.value,
                "revision_date": metadata.revision_date.isoformat() if metadata.revision_date else None,
            },
        ), metadata
    except Exception as exc:
        return _result("infer_document_metadata", start, status="fail", error=str(exc)), None


def check_normalize_stage(parsed: Any) -> tuple[StageCheckResult, list[Any] | None]:
    start = time.perf_counter()
    try:
        normalized = normalize_nodes(parsed.logical_nodes)
        with_keywords = sum(1 for node in normalized if node.keywords_json)
        return _result(
            "normalize_nodes",
            start,
            status="pass",
            details={
                "normalized_nodes": len(normalized),
                "nodes_with_keywords": with_keywords,
                "avg_token_count": round(sum(node.token_count for node in normalized) / max(len(normalized), 1), 2),
            },
        ), normalized
    except Exception as exc:
        return _result("normalize_nodes", start, status="fail", error=str(exc)), None


def check_chunking_stage(metadata: Any, normalized: list[Any]) -> tuple[StageCheckResult, list[Any] | None]:
    start = time.perf_counter()
    try:
        chunk_metadata = {
            "tenant_id": "local-tenant",
            "corpus_id": "pipeline_health",
            "document_kind": metadata.document_kind.value,
            "manufacturer": metadata.manufacturer,
            "product_family": metadata.product_family,
            "product_model": metadata.product_model,
            "language": "en",
            "visibility_scope": "internal",
            "permissions_tags": [],
            "version_label": "health",
            "revision_date": metadata.revision_date.isoformat() if metadata.revision_date else None,
            "ingest_run_id": "pipeline-health",
            "parse_profile": "standard_manual",
            "ocr_used": False,
            "is_active": True,
        }
        chunks = build_chunks(
            source_document_id="pipeline-health-doc",
            document_version_id="pipeline-health-version",
            title=metadata.title,
            nodes=normalized,
            metadata=chunk_metadata,
        )
        level_counts: dict[int, int] = {}
        type_counts: dict[str, int] = {}
        for chunk in chunks:
            level_counts[chunk.chunk_level] = level_counts.get(chunk.chunk_level, 0) + 1
            type_counts[chunk.chunk_type.value] = type_counts.get(chunk.chunk_type.value, 0) + 1
        return _result(
            "build_chunks",
            start,
            status="pass",
            details={"chunk_count": len(chunks), "levels": level_counts, "chunk_types": type_counts},
        ), chunks
    except Exception as exc:
        return _result("build_chunks", start, status="fail", error=str(exc)), None


def check_fragmentation_stage(parsed: Any, normalized: list[Any], chunks: list[Any]) -> StageCheckResult:
    start = time.perf_counter()
    try:
        logical_counts_by_page = Counter(node.page_from for node in normalized)
        chunk_counts_by_page = Counter(chunk.page_from for chunk in chunks)
        distinct_chunk_pages = sorted(chunk_counts_by_page)
        prose_nodes = [node for node in normalized if node.node_type in {NodeType.paragraph, NodeType.note}]
        short_nodes = sum(1 for node in prose_nodes if node.token_count <= 3)
        normalized_texts = [node.text_normalized.strip().lower() for node in normalized if node.text_normalized.strip()]
        duplicate_texts = sum(count - 1 for count in Counter(normalized_texts).values() if count > 1)
        short_fragment_ratio = short_nodes / max(len(prose_nodes), 1)
        duplicate_text_ratio = duplicate_texts / max(len(normalized_texts), 1)
        avg_nodes_per_page = round(sum(logical_counts_by_page.values()) / max(len(logical_counts_by_page), 1), 2)
        avg_chunks_per_page = round(sum(chunk_counts_by_page.values()) / max(len(chunk_counts_by_page), 1), 2)
        max_nodes_per_page = max(logical_counts_by_page.values(), default=0)
        max_chunks_per_page = max(chunk_counts_by_page.values(), default=0)

        status = "pass"
        error = None
        if parsed.page_count > 1 and len(distinct_chunk_pages) <= 1:
            status = "fail"
            error = "Retrieval chunks collapse onto a single page."
        elif avg_chunks_per_page > 40 or max_chunks_per_page > 150:
            status = "fail"
            error = "Chunk density exceeds the structural health ceiling."
        elif parsed.page_count > 1 and short_fragment_ratio > 0.45:
            status = "fail"
            error = "Short-fragment ratio is too high."

        return _result(
            "fragmentation",
            start,
            status=status,
            details={
                "distinct_chunk_pages": len(distinct_chunk_pages),
                "avg_nodes_per_page": avg_nodes_per_page,
                "max_nodes_per_page": max_nodes_per_page,
                "avg_chunks_per_page": avg_chunks_per_page,
                "max_chunks_per_page": max_chunks_per_page,
                "short_fragment_ratio": round(short_fragment_ratio, 4),
                "duplicate_text_ratio": round(duplicate_text_ratio, 4),
            },
            error=error,
        )
    except Exception as exc:
        return _result("fragmentation", start, status="fail", error=str(exc))


def check_chunk_quality_stage(chunks: list[Any]) -> StageCheckResult:
    start = time.perf_counter()
    try:
        flags = [
            content_quality_flags(
                content=str(chunk.content),
                title=str(chunk.title),
                section_path=list(chunk.metadata_json.get("section_path", [])),
                chunk_type=str(chunk.chunk_type.value),
            )
            for chunk in chunks
        ]
        low_information_count = sum(1 for item in flags if item["low_information"])
        structured_low_information_count = sum(1 for item in flags if item["structured_low_information"])
        technical_signal_mean = round(sum(item["technical_signal_score"] for item in flags) / max(len(flags), 1), 4)
        low_information_ratio = low_information_count / max(len(flags), 1)
        structured_total = sum(1 for chunk in chunks if chunk.chunk_type.value in {"spec_record", "datasheet_record", "table_record", "procedure_record", "warning_record"})
        structured_low_information_ratio = structured_low_information_count / max(structured_total, 1)
        status = "pass"
        error = None
        if low_information_ratio > 0.35:
            status = "fail"
            error = "Too many retrieval chunks are low-information."
        elif structured_total and structured_low_information_ratio > 0.15:
            status = "fail"
            error = "Structured retrieval chunks contain too much low-information content."
        return _result(
            "chunk_quality",
            start,
            status=status,
            details={
                "chunk_count": len(chunks),
                "low_information_ratio": round(low_information_ratio, 4),
                "structured_low_information_ratio": round(structured_low_information_ratio, 4),
                "technical_signal_mean": technical_signal_mean,
            },
            error=error,
        )
    except Exception as exc:
        return _result("chunk_quality", start, status="fail", error=str(exc))


def check_embedding_stage(chunks: list[Any]) -> StageCheckResult:
    start = time.perf_counter()
    try:
        sample = chunks[: min(3, len(chunks))]
        dense = embed_dense([chunk.content_for_dense for chunk in sample])
        sparse = [build_sparse_vector(chunk.content_for_sparse) for chunk in sample]
        return _result(
            "embedding_generation",
            start,
            status="pass",
            details={
                "sampled_chunks": len(sample),
                "dense_vector_dim": len(dense[0]) if dense else 0,
                "sparse_nonzero": [len(indices) for indices, _ in sparse],
            },
        )
    except Exception as exc:
        return _result("embedding_generation", start, status="fail", error=str(exc))


def check_query_analysis_stage() -> StageCheckResult:
    start = time.perf_counter()
    try:
        query = "What does the LJ-X8000 manual say about command timing and handshake flags?"
        analysis = analyze_query(query)
        filters = build_filters(query, {})
        return _result(
            "query_analysis",
            start,
            status="pass",
            details={
                "query_types": analysis.query_types,
                "preferred_chunk_types": analysis.preferred_chunk_types,
                "product_model": analysis.product_model,
                "applied_filters": filters,
            },
        )
    except Exception as exc:
        return _result("query_analysis", start, status="fail", error=str(exc))


def check_local_retrieval_stage(chunks: list[Any], product_model: str | None) -> StageCheckResult:
    start = time.perf_counter()
    corpus_id = f"pipeline_health_{uuid4().hex[:12]}"
    store = QdrantStore()
    try:
        store.upsert_chunks(corpus_id, chunks)
        query = f"What product is described in {product_model or 'this'} datasheet?"
        filters = {"product_model": product_model} if product_model else {}
        results = store.search(corpus_id, query, filters, limit=5)
        return _result(
            "local_retrieval",
            start,
            status="pass" if results else "fail",
            details={
                "corpus_id": corpus_id,
                "result_count": len(results),
                "top_chunk_type": results[0].metadata.get("chunk_type") if results else None,
                "top_title": results[0].title if results else None,
            },
            error=None if results else "No local retrieval results returned.",
        )
    except Exception as exc:
        return _result("local_retrieval", start, status="fail", error=str(exc))
    finally:
        try:
            store.client.delete_collection(collection_name(corpus_id))
        except Exception:
            pass


def _api_available(api_base: str) -> bool:
    try:
        response = httpx.get(f"{api_base}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def check_live_api_stage(pdf_path: Path, *, api_base: str = "http://127.0.0.1:8600") -> StageCheckResult:
    start = time.perf_counter()
    if not _api_available(api_base):
        return _result("live_api_pipeline", start, status="skipped", error="Live API stack is not running.")
    admin_headers = {"Authorization": "Bearer admin-token"}
    user_headers = {"Authorization": "Bearer user-token"}
    corpus_id = f"pipeline_health_{uuid4().hex[:12]}"
    try:
        with httpx.Client(base_url=api_base, timeout=180.0) as client:
            client.post("/corpora", headers=admin_headers, json={"id": corpus_id, "name": corpus_id}).raise_for_status()
            with pdf_path.open("rb") as handle:
                upload = client.post(
                    "/documents/upload",
                    headers=admin_headers,
                    data={"corpus_id": corpus_id},
                    files={"files": (pdf_path.name, handle, "application/pdf")},
                )
            upload.raise_for_status()
            uploaded = upload.json()["uploaded"][0]
            document_id = uploaded["document_id"]
            ingest = client.post(f"/documents/{document_id}/ingest", headers=admin_headers)
            ingest.raise_for_status()
            run_id = ingest.json()["run_id"]
            run_status = "queued"
            for _ in range(240):
                run = client.get(f"/ingestion-runs/{run_id}", headers=admin_headers)
                run.raise_for_status()
                run_status = run.json()["status"]
                if run_status == "completed":
                    break
                if run_status == "failed":
                    return _result("live_api_pipeline", start, status="fail", error=run.json().get("failure_reason"))
                time.sleep(1)
            if run_status != "completed":
                return _result("live_api_pipeline", start, status="fail", error="Ingestion did not complete in time.")
            document = client.get(f"/documents/{document_id}", headers=admin_headers)
            document.raise_for_status()
            search = client.post(
                "/search",
                headers=user_headers,
                json={
                    "query": f"What product is described in the {pdf_path.stem.split('_')[0]} datasheet?",
                    "corpus_ids": [corpus_id],
                    "filters": {"source_document_id": document_id},
                },
            )
            search.raise_for_status()
            query = client.post(
                "/query",
                headers=user_headers,
                json={
                    "query": f"What product is described in the {pdf_path.stem.split('_')[0]} datasheet?",
                    "corpus_ids": [corpus_id],
                    "filters": {"source_document_id": document_id},
                    "response_mode": "answer_with_citations",
                },
            )
            query.raise_for_status()
            explain = client.post(
                "/explain-retrieval",
                headers=admin_headers,
                json={
                    "query": f"What product is described in the {pdf_path.stem.split('_')[0]} datasheet?",
                    "corpus_ids": [corpus_id],
                    "filters": {"source_document_id": document_id},
                },
            )
            explain.raise_for_status()
            answer_payload = query.json()
            search_payload = search.json()
            explain_payload = explain.json()
            return _result(
                "live_api_pipeline",
                start,
                status="pass" if search_payload and answer_payload.get("citations") else "fail",
                details={
                    "corpus_id": corpus_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "document_kind": document.json().get("document_kind"),
                    "search_results": len(search_payload),
                    "citations": len(answer_payload.get("citations", [])),
                    "explain_results": len(explain_payload.get("reranked_top_results", [])),
                },
                error=None if search_payload and answer_payload.get("citations") else "Live query returned incomplete result.",
            )
    except Exception as exc:
        return _result("live_api_pipeline", start, status="fail", error=str(exc))


def run_pipeline_health_checks(pdf_path: Path, *, api_base: str = "http://127.0.0.1:8600", include_live: bool = True) -> PipelineHealthReport:
    stage_results: list[StageCheckResult] = []

    fixture_result = check_fixture_path(pdf_path)
    stage_results.append(fixture_result)
    if fixture_result.status != "pass":
        return _build_report(pdf_path, stage_results)

    parse_result, parsed = check_parse_stage(pdf_path)
    stage_results.append(parse_result)
    if parsed is None:
        return _build_report(pdf_path, stage_results)
    stage_results.append(check_page_provenance_stage(parsed))

    metadata_result, metadata = check_metadata_stage(pdf_path, parsed)
    stage_results.append(metadata_result)
    if metadata is None:
        return _build_report(pdf_path, stage_results)

    normalize_result, normalized = check_normalize_stage(parsed)
    stage_results.append(normalize_result)
    if normalized is None:
        return _build_report(pdf_path, stage_results)

    chunk_result, chunks = check_chunking_stage(metadata, normalized)
    stage_results.append(chunk_result)
    if chunks is None:
        return _build_report(pdf_path, stage_results)
    stage_results.append(check_fragmentation_stage(parsed, normalized, chunks))
    stage_results.append(check_chunk_quality_stage(chunks))

    stage_results.append(check_embedding_stage(chunks))
    stage_results.append(check_query_analysis_stage())
    stage_results.append(check_local_retrieval_stage(chunks, metadata.product_model))
    if include_live:
        stage_results.append(check_live_api_stage(pdf_path, api_base=api_base))
    return _build_report(pdf_path, stage_results)


def _build_report(pdf_path: Path, stage_results: list[StageCheckResult]) -> PipelineHealthReport:
    passed = sum(1 for result in stage_results if result.status == "pass")
    failed = sum(1 for result in stage_results if result.status == "fail")
    skipped = sum(1 for result in stage_results if result.status == "skipped")
    overall_status = "fail" if failed else ("pass" if passed else "skipped")
    return PipelineHealthReport(
        fixture_path=str(pdf_path),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        overall_status=overall_status,
        passed=passed,
        failed=failed,
        skipped=skipped,
        stage_results=stage_results,
    )


def report_to_json(report: PipelineHealthReport) -> str:
    return json.dumps(
        {
            **asdict(report),
            "stage_results": [asdict(result) for result in report.stage_results],
        },
        ensure_ascii=True,
        indent=2,
    )


def report_to_markdown(report: PipelineHealthReport) -> str:
    lines = [
        "# Pipeline Health Report",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Fixture: `{report.fixture_path}`",
        f"- Overall status: `{report.overall_status}`",
        f"- Passed: `{report.passed}`",
        f"- Failed: `{report.failed}`",
        f"- Skipped: `{report.skipped}`",
        "",
        "## Stage Results",
        "",
    ]
    for result in report.stage_results:
        lines.append(f"### {result.name}")
        lines.append(f"- Status: `{result.status}`")
        lines.append(f"- Duration: `{result.duration_ms} ms`")
        if result.error:
            lines.append(f"- Error: `{result.error}`")
        if result.details:
            lines.append("- Details:")
            for key, value in result.details.items():
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines)
