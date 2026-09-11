#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manuals_rag_common.db import fetch_all


PIPELINE = "evidence_map_reduce_verify_v1"
TITLE_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{1,8}(?:[-:]\s*[A-Z0-9]{1,16})+|"
    r"[A-Z]{1,8}[A-Z-]*\d+[A-Z0-9-]*)(?![A-Za-z0-9])"
)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _audit_document(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata_json") or {})
    claims = [dict(item) for item in metadata.get("metadata_claims") or []]
    confirmed = [item for item in claims if item.get("verification_status") == "confirmed"]
    routing = [
        *list(metadata.get("routing_product_models") or []),
        *list(metadata.get("routing_part_numbers") or []),
    ]
    failures: list[str] = []
    if row.get("ingest_status") != "indexed":
        failures.append("document_not_indexed")
    if metadata.get("metadata_schema_version") != 2:
        failures.append("wrong_schema_version")
    if metadata.get("metadata_pipeline_version") != PIPELINE:
        failures.append("wrong_pipeline_version")
    title_evidence = [
        item for item in metadata.get("metadata_evidence") or []
        if item.get("kind") == "document_title" and item.get("source") == "opening_page_title"
    ]
    if title_evidence and min(int(item.get("page_from") or 10**9) for item in title_evidence) > 2:
        failures.append("title_not_grounded_on_opening_pages")
    for item in confirmed:
        if item.get("grounded") is not True or not item.get("source_quote") or item.get("page_from") is None:
            failures.append("confirmed_claim_missing_grounding")
            break
        if float(item.get("confidence") or 0.0) < 0.8:
            failures.append("confirmed_claim_below_trust_threshold")
            break
    if any(len(str(value)) > 80 or len(str(value).split()) > 6 for value in routing):
        failures.append("prose_shaped_routing_identifier")
    title = str(metadata.get("title") or row.get("title") or "")
    if TITLE_IDENTIFIER_PATTERN.search(title) and not metadata.get("routing_product_models"):
        failures.append("opening_title_identifier_not_routable")
    confirmed_keys = {
        _compact(str(item.get("value") or ""))
        for item in confirmed
        if item.get("kind") in {"product_model", "product_family", "part_number"}
    }
    if any(_compact(str(value)) not in confirmed_keys for value in routing):
        failures.append("routing_identifier_without_confirmed_claim")
    applicable = [
        *list(metadata.get("firmware_applicability") or []),
        *list(metadata.get("software_applicability") or []),
    ]
    if any(item.get("relation") == "external_reference" for item in applicable):
        failures.append("external_version_published_as_applicability")
    if int(row.get("chunk_count") or 0) == 0:
        failures.append("no_persisted_chunks")
    if int(row.get("mrv_chunk_count") or 0) != int(row.get("chunk_count") or 0):
        failures.append("metadata_not_propagated_to_every_chunk")
    return {
        "document_id": str(row["document_id"]),
        "source_filename": row.get("source_filename"),
        "title": metadata.get("title") or row.get("title"),
        "claim_counts": dict(Counter(str(item.get("verification_status") or "unknown") for item in claims)),
        "routing_product_models": metadata.get("routing_product_models") or [],
        "routing_part_numbers": metadata.get("routing_part_numbers") or [],
        "firmware_applicability": metadata.get("firmware_applicability") or [],
        "software_applicability": metadata.get("software_applicability") or [],
        "chunk_count": int(row.get("chunk_count") or 0),
        "checks_passed": not failures,
        "failures": failures,
    }


def run(document_ids: list[str]) -> dict[str, Any]:
    placeholders = ",".join(["%s"] * len(document_ids))
    rows = fetch_all(
        f"""
        select sd.id as document_id, sd.source_filename, sd.title, sd.ingest_status,
               dme.metadata_json,
               count(rc.id)::int as chunk_count,
               count(rc.id) filter (
                   where rc.metadata_json->>'metadata_pipeline_version' = %s
               )::int as mrv_chunk_count
        from source_documents sd
        left join document_metadata_extractions dme on dme.source_document_id = sd.id
        left join retrieval_chunks rc on rc.source_document_id = sd.id and rc.is_active = true
        where sd.id in ({placeholders})
        group by sd.id, sd.source_filename, sd.title, sd.ingest_status, dme.metadata_json
        order by sd.source_filename
        """,
        (PIPELINE, *document_ids),
    )
    documents = [_audit_document(row) for row in rows]
    found = {item["document_id"] for item in documents}
    for missing in sorted(set(document_ids) - found):
        documents.append({"document_id": missing, "checks_passed": False, "failures": ["document_not_found"]})
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline": PIPELINE,
        "document_count": len(documents),
        "passed": sum(1 for item in documents if item["checks_passed"]),
        "failed": sum(1 for item in documents if not item["checks_passed"]),
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit persisted evidence-first metadata and chunk propagation.")
    parser.add_argument("document_ids", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.document_ids)
    rendered = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
