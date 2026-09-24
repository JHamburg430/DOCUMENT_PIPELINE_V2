from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from manuals_rag_common.config import settings
from manuals_rag_common.db import execute, fetch_all, get_db, json_dumps
from manuals_rag_common.queue import enqueue
from manuals_rag_parsers.metadata import (
    METADATA_PIPELINE_VERSION,
    MetadataSourceSegment,
    infer_document_metadata_from_segments,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "test_reports"


@dataclass
class BackfillResult:
    document_id: str
    version_id: str
    source_filename: str
    status: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    chunk_count: int = 0
    embed_enqueued: bool = False
    warning: str | None = None


def _ensure_metadata_table() -> None:
    execute(
        """
        create table if not exists document_metadata_extractions (
            source_document_id uuid primary key references source_documents(id) on delete cascade,
            document_version_id uuid not null references document_versions(id) on delete cascade,
            model text not null,
            metadata_json jsonb not null default '{}'::jsonb,
            extracted_at timestamptz not null default now()
        )
        """
    )


def _documents(
    limit: int | None = None,
    *,
    document_ids: list[str] | None = None,
    corpus_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    query = """
        select
            sd.id as document_id,
            sd.corpus_id,
            sd.source_filename,
            sd.current_version_id as version_id,
            c.permissions_json #>> '{metadata_defaults,manufacturer}' as authoritative_manufacturer,
            dme.document_version_id as extracted_version_id,
            dme.metadata_json ->> 'metadata_pipeline_version' as extracted_pipeline_version
        from source_documents sd
        join corpora c on c.id = sd.corpus_id
        join document_versions dv on dv.id = sd.current_version_id
        left join document_metadata_extractions dme on dme.source_document_id = sd.id
        where exists (
            select 1
            from logical_nodes ln
            where ln.document_version_id = dv.id
        )
    """
    params: tuple[Any, ...] = ()
    if document_ids:
        placeholders = ",".join(["%s"] * len(document_ids))
        query += f" and sd.id in ({placeholders})"
        params = tuple(document_ids)
    if corpus_ids:
        placeholders = ",".join(["%s"] * len(corpus_ids))
        query += f" and sd.corpus_id in ({placeholders})"
        params = (*params, *corpus_ids)
    query += " order by sd.updated_at desc, sd.id"
    if limit is not None:
        query += " limit %s"
        params = (*params, limit)
    return fetch_all(query, params)


def _document_segments(version_id: str, *, node_limit: int | None = None) -> list[MetadataSourceSegment]:
    limit_clause = " limit %s" if node_limit is not None else ""
    params: tuple[Any, ...] = (version_id, node_limit) if node_limit is not None else (version_id,)
    rows = fetch_all(
        f"""
        select text_normalized, text_raw, page_from, page_to, section_path_json
        from logical_nodes
        where document_version_id = %s
          and coalesce(text_normalized, text_raw, '') <> ''
        order by ordinal
        {limit_clause}
        """,
        params,
    )
    return [
        MetadataSourceSegment(
            # Metadata provenance is a literal-source contract.  Normalized
            # parser text is useful for retrieval, but it may rewrite printed
            # punctuation (for example VJ-3302 -> VJ: 3302), which makes an
            # otherwise grounded source_quote nonliteral.  Feed the model and
            # grounding checks the raw extracted text whenever it exists.
            text=str(row["text_raw"] or row["text_normalized"] or ""),
            page_from=row.get("page_from"),
            page_to=row.get("page_to"),
            section_path=tuple(row.get("section_path_json") or []),
        )
        for row in rows
    ]


def _metadata_payload(metadata: Any) -> dict[str, Any]:
    payload = metadata.__dict__.copy()
    payload["document_kind"] = metadata.document_kind.value
    payload["revision_date"] = metadata.revision_date.isoformat() if metadata.revision_date else None
    payload["effective_date"] = metadata.effective_date.isoformat() if metadata.effective_date else None
    return payload


def _apply_authoritative_metadata_defaults(
    document: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Overlay explicit corpus-owned identity without trusting extracted company mentions."""
    manufacturer = str(document.get("authoritative_manufacturer") or "").strip()
    if not manufacturer:
        return metadata
    payload = dict(metadata)
    payload["manufacturer"] = manufacturer
    payload["companies"] = list(dict.fromkeys([manufacturer, *list(payload.get("companies") or [])]))
    authoritative_key = " ".join(manufacturer.casefold().split())
    for field in ("metadata_evidence", "metadata_claims"):
        normalized_claims = []
        for claim in payload.get(field) or []:
            normalized_claim = dict(claim)
            if (
                str(normalized_claim.get("kind") or "").casefold() == "company"
                and str(normalized_claim.get("relation") or "").casefold()
                == "primary_manufacturer"
                and " ".join(str(normalized_claim.get("value") or "").casefold().split())
                != authoritative_key
            ):
                # Keep regional affiliates, distributors, and example vendors
                # as grounded mentions without allowing them to override the
                # corpus-owned product manufacturer.
                normalized_claim["relation"] = "mentioned"
            normalized_claims.append(normalized_claim)
        payload[field] = normalized_claims
    payload["metadata_authoritative_defaults"] = {
        "manufacturer": manufacturer,
        "source": "corpus_configuration",
    }
    return payload


def _chunk_metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_kind": metadata["document_kind"],
        "manufacturer": metadata["manufacturer"],
        "companies": metadata["companies"],
        "product_family": metadata["product_family"],
        "product_model": metadata["product_model"],
        "product_families": metadata["product_families"],
        "product_models": metadata["product_models"],
        "devices": metadata["devices"],
        "part_numbers": metadata["part_numbers"],
        "document_protocol_terms": metadata["protocol_terms"],
        "settings": metadata["settings"],
        "parameters": metadata["parameters"],
        "document_menu_labels": metadata["menu_labels"],
        "document_topics": metadata["document_topics"],
        "revision_date": metadata["revision_date"],
        "metadata_schema_version": metadata["metadata_schema_version"],
        "metadata_pipeline_version": metadata.get("metadata_pipeline_version", "legacy"),
        "normalized_identifier_aliases": metadata["normalized_identifier_aliases"],
        "routing_product_models": metadata["routing_product_models"],
        "routing_part_numbers": metadata["routing_part_numbers"],
        "routing_protocol_terms": metadata["routing_protocol_terms"],
        "firmware_applicability": metadata["firmware_applicability"],
        "software_applicability": metadata["software_applicability"],
    }


def _apply_metadata(
    document: dict[str, Any], metadata: dict[str, Any], *, enqueue_embed: bool
) -> tuple[int, bool, str | None]:
    """Persist one document atomically, then enqueue its post-commit embedding refresh."""
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into document_metadata_extractions (
                source_document_id, document_version_id, model, metadata_json, extracted_at
            ) values (%s, %s, %s, %s::jsonb, now())
            on conflict (source_document_id) do update
            set document_version_id = excluded.document_version_id,
                model = excluded.model,
                metadata_json = excluded.metadata_json,
                extracted_at = excluded.extracted_at
            """,
            (
                document["document_id"],
                document["version_id"],
                settings.ollama_metadata_model,
                json_dumps(metadata),
            ),
        )
        cur.execute(
            """
            update source_documents
            set manufacturer = %s,
                product_family = %s,
                product_model = %s,
                document_kind = %s,
                title = coalesce(nullif(%s, ''), title),
                updated_at = now()
            where id = %s
            """,
            (
                metadata["manufacturer"],
                metadata["product_family"],
                metadata["product_model"],
                metadata["document_kind"],
                metadata["title"],
                document["document_id"],
            ),
        )
        cur.execute(
            """
            update retrieval_chunks
            set metadata_json = metadata_json || %s::jsonb
            where document_version_id = %s
            """,
            (json_dumps(_chunk_metadata_payload(metadata)), document["version_id"]),
        )
        cur.execute(
            "select count(*) as count from retrieval_chunks where document_version_id = %s",
            (document["version_id"],),
        )
        count_row = cur.fetchone()
        chunk_count = int(count_row["count"]) if count_row else 0
        conn.commit()

    if enqueue_embed:
        embed_enqueued, warning = _enqueue_embed_refresh(document)
        return chunk_count, embed_enqueued, warning
    return chunk_count, False, None


def _enqueue_embed_refresh(document: dict[str, Any]) -> tuple[bool, str | None]:
    run_id = str(uuid4())
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into ingestion_runs (
                id, source_document_id, document_version_id, status, failure_class, created_at, updated_at
            ) values (%s, %s, %s, 'parsed', null, now(), now())
            """,
            (run_id, document["document_id"], document["version_id"]),
        )
        conn.commit()
    try:
        enqueue(
            "embed_jobs",
            {"run_id": run_id, "document_id": str(document["document_id"]), "version_id": str(document["version_id"])},
        )
    except Exception as exc:
        return False, f"embedding refresh run {run_id} persisted but enqueue failed: {exc}"
    return True, None


def _write_report(results: list[BackfillResult], *, path: Path | None = None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"document_metadata_backfill_{timestamp}.json"
    path.write_text(json.dumps([asdict(result) for result in results], indent=2, default=str), encoding="utf-8")
    return path


def _load_report(path: Path) -> list[BackfillResult]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load resume report {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"resume report {path} must contain a JSON array")
    results: list[BackfillResult] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"resume report {path} entry {index} must be an object")
        try:
            results.append(BackfillResult(**item))
        except TypeError as exc:
            raise ValueError(f"resume report {path} entry {index} is invalid: {exc}") from exc
    keys = [(result.document_id, result.version_id) for result in results]
    if len(keys) != len(set(keys)):
        raise ValueError(f"resume report {path} contains duplicate document/version entries")
    return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _planned_metadata_for_exact_apply(
    result: BackfillResult | None,
    *,
    document: dict[str, Any],
) -> dict[str, Any]:
    if result is None:
        raise ValueError("exact-report apply requires a matching planned document/version entry")
    if result.status != "planned" or not isinstance(result.metadata, dict):
        raise ValueError("exact-report apply requires a planned entry with metadata")
    if result.metadata.get("metadata_pipeline_version") != METADATA_PIPELINE_VERSION:
        raise ValueError(
            "exact-report metadata pipeline does not match the running pipeline: "
            f"{result.metadata.get('metadata_pipeline_version')!r} != {METADATA_PIPELINE_VERSION!r}"
        )
    if result.document_id != str(document["document_id"]) or result.version_id != str(document["version_id"]):
        raise ValueError("exact-report entry does not match the current document/version")
    return dict(result.metadata)


def _can_resume_result(
    result: BackfillResult,
    *,
    apply: bool,
    enqueue_current: bool,
    already_current: bool,
) -> bool:
    if enqueue_current:
        return already_current and result.status == "embed_enqueued" and result.embed_enqueued
    if apply:
        return already_current and result.status in {"applied", "applied_with_warning", "skipped_current"}
    if result.status == "skipped_current":
        return already_current
    return (
        result.status == "planned"
        and isinstance(result.metadata, dict)
        and result.metadata.get("metadata_pipeline_version") == METADATA_PIPELINE_VERSION
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill model-extracted document metadata.")
    parser.add_argument("--apply", action="store_true", help="Persist metadata changes. Without this, only reports extracted metadata.")
    parser.add_argument("--no-enqueue-embed", action="store_true", help="Do not enqueue embed jobs after updating chunk metadata.")
    parser.add_argument(
        "--enqueue-current",
        action="store_true",
        help="Enqueue embedding refreshes for already-current metadata without re-running extraction.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly acknowledge that a mutating operation may target the entire corpus.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents.")
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help="Restrict processing to one source-document UUID. Repeat for multiple documents.",
    )
    parser.add_argument(
        "--corpus-id",
        action="append",
        dest="corpus_ids",
        help="Restrict processing to one corpus ID. Repeat for multiple corpora.",
    )
    parser.add_argument(
        "--node-limit",
        type=int,
        default=None,
        help="Optional diagnostic cap. By default the full document is enriched; limiting nodes reduces metadata recall.",
    )
    parser.add_argument("--segment-chars", type=int, default=3000, help="Maximum source characters per page-aware model call.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess documents already extracted with this pipeline and current document version.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Abort after this many document failures. Use 0 for no failure limit.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Rewrite the resumable JSON report after this many processed documents.",
    )
    parser.add_argument(
        "--resume-report",
        type=Path,
        default=None,
        help="Continue an interrupted run from an existing checkpoint report.",
    )
    parser.add_argument(
        "--apply-planned-report",
        action="store_true",
        help=(
            "With --apply and --resume-report, persist the exact planned metadata in that report "
            "without re-running model extraction. Requires --report-sha256."
        ),
    )
    parser.add_argument(
        "--report-sha256",
        help="Expected lowercase SHA-256 of --resume-report for an exact planned-report apply.",
    )
    parser.add_argument(
        "--result-report",
        type=Path,
        help="Write exact planned-report apply results to this new file, preserving the audited source report.",
    )
    args = parser.parse_args()

    if args.max_failures < 0:
        parser.error("--max-failures must be zero or greater")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least one")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least one")
    if args.node_limit is not None and args.node_limit < 1:
        parser.error("--node-limit must be at least one")
    if args.segment_chars < 500:
        parser.error("--segment-chars must be at least 500")
    if args.apply and args.enqueue_current:
        parser.error("--apply and --enqueue-current are separate stages and cannot be combined")
    if args.no_enqueue_embed and not args.apply:
        parser.error("--no-enqueue-embed requires --apply")
    if args.apply_planned_report and (not args.apply or args.resume_report is None):
        parser.error("--apply-planned-report requires --apply and --resume-report")
    if args.apply_planned_report and not args.report_sha256:
        parser.error("--apply-planned-report requires --report-sha256")
    if args.apply_planned_report and args.result_report is None:
        parser.error("--apply-planned-report requires --result-report")
    if args.result_report is not None and not args.apply_planned_report:
        parser.error("--result-report is only valid with --apply-planned-report")
    if args.report_sha256 and not args.apply_planned_report:
        parser.error("--report-sha256 is only valid with --apply-planned-report")
    if args.all and (args.document_ids or args.limit is not None):
        parser.error("--all cannot be combined with --document-id or --limit")
    if (args.apply or args.enqueue_current) and not (args.document_ids or args.limit is not None or args.all):
        parser.error("mutating the complete corpus requires --all; otherwise use --document-id or --limit")

    _ensure_metadata_table()
    if args.resume_report is not None:
        source_report_path = args.resume_report.resolve()
        if args.apply_planned_report:
            result_report_path = args.result_report.resolve()
            if result_report_path.exists():
                parser.error(f"refusing to overwrite existing result report: {result_report_path}")
            actual_sha256 = _sha256(source_report_path)
            if actual_sha256 != str(args.report_sha256).casefold():
                parser.error(
                    f"resume report SHA-256 mismatch: expected {args.report_sha256}, got {actual_sha256}"
                )
        try:
            results = _load_report(source_report_path)
        except ValueError as exc:
            parser.error(str(exc))
        report_path = result_report_path if args.apply_planned_report else source_report_path
    else:
        results = []
        report_path = _write_report(results)
    documents = _documents(
        limit=args.limit,
        document_ids=args.document_ids,
        corpus_ids=args.corpus_ids,
    )
    selected_keys = {
        (str(document["document_id"]), str(document["version_id"]))
        for document in documents
    }
    out_of_scope_results = [
        result
        for result in results
        if (result.document_id, result.version_id) not in selected_keys
    ]
    if out_of_scope_results:
        parser.error(
            "resume report contains document/version entries outside the selected scope: "
            + ", ".join(
                f"{result.document_id}/{result.version_id}"
                for result in out_of_scope_results[:5]
            )
        )

    failure_count = 0
    for document in documents:
        print(json.dumps({"document_id": str(document["document_id"]), "status": "started"}), flush=True)
        already_current = (
            str(document.get("extracted_version_id") or "") == str(document["version_id"])
            and document.get("extracted_pipeline_version") == METADATA_PIPELINE_VERSION
        )
        matching_result = next(
            (
                result
                for result in results
                if result.document_id == str(document["document_id"])
                and result.version_id == str(document["version_id"])
            ),
            None,
        )
        if matching_result is not None and _can_resume_result(
            matching_result,
            apply=args.apply,
            enqueue_current=args.enqueue_current,
            already_current=already_current,
        ):
            print(
                json.dumps(
                    {
                        "document_id": str(document["document_id"]),
                        "status": "resumed_completed",
                        "checkpoint_status": matching_result.status,
                    }
                )
            )
            continue
        if matching_result is not None:
            if not args.apply_planned_report:
                results.remove(matching_result)
        if args.enqueue_current:
            if not already_current:
                results.append(
                    BackfillResult(
                        document_id=str(document["document_id"]),
                        version_id=str(document["version_id"]),
                        source_filename=str(document["source_filename"]),
                        status="failed",
                        error="current pipeline metadata is not persisted for this document version",
                    )
                )
                failure_count += 1
            else:
                embed_enqueued, warning = _enqueue_embed_refresh(document)
                results.append(
                    BackfillResult(
                        document_id=str(document["document_id"]),
                        version_id=str(document["version_id"]),
                        source_filename=str(document["source_filename"]),
                        status="embed_enqueued" if embed_enqueued else "enqueue_failed",
                        embed_enqueued=embed_enqueued,
                        warning=warning,
                    )
                )
                if not embed_enqueued:
                    failure_count += 1
            print(json.dumps(asdict(results[-1]), default=str))
            if len(results) % args.checkpoint_every == 0:
                _write_report(results, path=report_path)
            if args.max_failures and failure_count >= args.max_failures:
                print(json.dumps({"status": "aborted", "reason": "failure_budget_exhausted", "failures": failure_count}))
                break
            continue
        if already_current and not args.force:
            results.append(
                BackfillResult(
                    document_id=str(document["document_id"]),
                    version_id=str(document["version_id"]),
                    source_filename=str(document["source_filename"]),
                    status="skipped_current",
                )
            )
            print(json.dumps({"document_id": str(document["document_id"]), "status": "skipped_current"}))
            if len(results) % args.checkpoint_every == 0:
                _write_report(results, path=report_path)
            continue
        try:
            if args.apply_planned_report:
                metadata = _planned_metadata_for_exact_apply(matching_result, document=document)
                results.remove(matching_result)
            else:
                segments = _document_segments(str(document["version_id"]), node_limit=args.node_limit)
                metadata = _apply_authoritative_metadata_defaults(
                    document,
                    _metadata_payload(
                        infer_document_metadata_from_segments(
                            str(document["source_filename"]),
                            segments,
                            max_segment_chars=args.segment_chars,
                        )
                    ),
                )
            chunk_count = 0
            embed_enqueued = False
            warning = None
            if args.apply:
                chunk_count, embed_enqueued, warning = _apply_metadata(
                    document,
                    metadata,
                    enqueue_embed=not args.no_enqueue_embed,
                )
            results.append(
                BackfillResult(
                    document_id=str(document["document_id"]),
                    version_id=str(document["version_id"]),
                    source_filename=str(document["source_filename"]),
                    status=("applied_with_warning" if warning else "applied") if args.apply else "planned",
                    metadata=metadata,
                    chunk_count=chunk_count,
                    embed_enqueued=embed_enqueued,
                    warning=warning,
                )
            )
            print(json.dumps({"document_id": str(document["document_id"]), "status": results[-1].status, "metadata": metadata}, default=str))
        except Exception as exc:
            failure_count += 1
            results.append(
                BackfillResult(
                    document_id=str(document["document_id"]),
                    version_id=str(document["version_id"]),
                    source_filename=str(document["source_filename"]),
                    status="failed",
                    error=str(exc),
                )
            )
            print(json.dumps({"document_id": str(document["document_id"]), "status": "failed", "error": str(exc)}))
        if len(results) % args.checkpoint_every == 0:
            _write_report(results, path=report_path)
        if args.max_failures and failure_count >= args.max_failures:
            print(json.dumps({"status": "aborted", "reason": "failure_budget_exhausted", "failures": failure_count}))
            break

    report = _write_report(results, path=report_path)
    summary = {
        "report": str(report),
        "total": len(results),
        "applied": sum(1 for result in results if result.status == "applied"),
        "applied_with_warning": sum(1 for result in results if result.status == "applied_with_warning"),
        "planned": sum(1 for result in results if result.status == "planned"),
        "failed": sum(1 for result in results if result.status == "failed"),
        "skipped_current": sum(1 for result in results if result.status == "skipped_current"),
        "warnings": sum(1 for result in results if result.warning),
        "enqueue_failed": sum(1 for result in results if result.status == "enqueue_failed"),
        "embed_enqueued": sum(1 for result in results if result.embed_enqueued),
    }
    print(json.dumps(summary, indent=2))
    raise SystemExit(1 if summary["failed"] or summary["enqueue_failed"] or summary["warnings"] else 0)


if __name__ == "__main__":
    main()
