from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuals_rag_common.config import settings
from manuals_rag_parsers.metadata import (
    METADATA_PIPELINE_VERSION,
    MetadataSourceSegment,
    _compact_identifier,
    _expected_version_kinds,
    _quote_location,
    infer_document_metadata_from_segments,
)

from scripts.maintenance.backfill_document_metadata import (
    _document_segments,
    _documents,
    _metadata_payload,
)


DEFAULT_FILENAMES = (
    "AS_126535_CV-X_C_611S71_KA_US_2124_5.pdf",
    "AS_130681_SR-1000_SR-2000_SR-PN1_CM_D65GB_WW_GB_2122_1.pdf",
    "AS_145895_KV-X_C_611X95_KA_US_2084_1.pdf",
    "AS_147320_LJ-S8000_SG_K47GB_WW_GB_2094_1.pdf",
    "VJ-H500CX_Datasheet.pdf",
)


@dataclass
class ComparisonResult:
    document_id: str
    version_id: str
    source_filename: str
    status: str
    elapsed_seconds: float
    segment_count: int
    page_count: int
    metadata: dict[str, Any] | None = None
    audit: dict[str, Any] | None = None
    error: str | None = None


def _qualified_routing_evidence(metadata: dict[str, Any], kind: str) -> set[str]:
    return {
        _compact_identifier(str(item.get("value") or ""))
        for item in metadata.get("metadata_claims") or []
        if item.get("kind") == kind
        and item.get("relation") in {"primary_product", "applies_to", "compatible_with", "accessory_for"}
        and item.get("grounded") is True
        and item.get("verification_status") == "confirmed"
        and float(item.get("confidence") or 0.0) >= 0.8
    }


def _audit(metadata: dict[str, Any], segments: list[MetadataSourceSegment]) -> dict[str, Any]:
    evidence = metadata.get("metadata_evidence") or []
    claims = metadata.get("metadata_claims") or []
    ungrounded = [
        item
        for item in evidence
        if not str(item.get("source_quote") or "").strip()
        or _quote_location(str(item.get("source_quote") or ""), segments) is None
    ]
    model_evidence = _qualified_routing_evidence(metadata, "product_model")
    part_evidence = _qualified_routing_evidence(metadata, "part_number")
    protocol_evidence = _qualified_routing_evidence(metadata, "protocol")
    routing_without_evidence = {
        "product_models": [
            value
            for value in metadata.get("routing_product_models") or []
            if _compact_identifier(value) not in model_evidence
        ],
        "part_numbers": [
            value
            for value in metadata.get("routing_part_numbers") or []
            if _compact_identifier(value) not in part_evidence
        ],
        "protocols": [
            value
            for value in metadata.get("routing_protocol_terms") or []
            if _compact_identifier(value) not in protocol_evidence
        ],
    }
    malformed_applicability = [
        item
        for key in ("firmware_applicability", "software_applicability")
        for item in metadata.get(key) or []
        if not item.get("version")
        or not item.get("subject")
        or item.get("relation") not in {"applies_to", "compatible_with"}
        or item.get("grounded") is not True
        or float(item.get("confidence") or 0.0) < 0.8
    ]
    expected_version_kinds = _expected_version_kinds(segments)
    confirmed_version_kinds = {
        str(item.get("kind") or "")
        for item in claims
        if item.get("kind") in {"firmware_version", "software_version"}
        and item.get("verification_status") == "confirmed"
        and item.get("grounded") is True
    }
    missing_version_kinds = sorted(expected_version_kinds - confirmed_version_kinds)
    checks = {
        "schema_v2": metadata.get("metadata_schema_version") == 2,
        "map_reduce_verify_pipeline": metadata.get("metadata_pipeline_version") == METADATA_PIPELINE_VERSION,
        "has_independently_confirmed_claims": any(
            item.get("verification_status") == "confirmed" for item in claims
        ),
        "all_evidence_grounded": not ungrounded,
        "all_routing_values_evidence_gated": not any(routing_without_evidence.values()),
        "no_conflicting_claim_controls_routing": not any(
            _compact_identifier(str(item.get("value") or ""))
            in {
                *map(_compact_identifier, metadata.get("routing_product_models") or []),
                *map(_compact_identifier, metadata.get("routing_part_numbers") or []),
                *map(_compact_identifier, metadata.get("routing_protocol_terms") or []),
            }
            and item.get("verification_status") == "conflicting"
            for item in claims
        ),
        "all_applicability_scoped": not malformed_applicability,
        "version_signals_have_confirmed_claims": not missing_version_kinds,
        "title_grounded_on_opening_pages": bool(
            any(
                item.get("kind") == "document_title"
                and item.get("grounded") is True
                and int(item.get("page_from") or 9999) <= 2
                for item in evidence
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence_count": len(evidence),
        "claim_count": len(claims),
        "claim_status_counts": {
            status: sum(item.get("verification_status") == status for item in claims)
            for status in ("confirmed", "probable", "unresolved", "conflicting", "rejected")
        },
        "ungrounded_evidence": ungrounded,
        "routing_without_evidence": routing_without_evidence,
        "malformed_applicability": malformed_applicability,
        "expected_version_kinds": sorted(expected_version_kinds),
        "confirmed_version_kinds": sorted(confirmed_version_kinds),
        "missing_version_kinds": missing_version_kinds,
        "routing_product_models": metadata.get("routing_product_models") or [],
        "routing_part_numbers": metadata.get("routing_part_numbers") or [],
        "routing_protocol_terms": metadata.get("routing_protocol_terms") or [],
        "firmware_applicability_count": len(metadata.get("firmware_applicability") or []),
        "software_applicability_count": len(metadata.get("software_applicability") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run metadata extraction on a fixed difficult-document set.")
    parser.add_argument("--filename", action="append", dest="filenames")
    parser.add_argument("--segment-chars", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    wanted = set(args.filenames or DEFAULT_FILENAMES)
    documents = [item for item in _documents() if str(item["source_filename"]) in wanted]
    missing = wanted - {str(item["source_filename"]) for item in documents}
    if missing:
        raise SystemExit(f"Missing requested documents: {sorted(missing)}")

    started = datetime.now(UTC)
    results: list[ComparisonResult] = []
    for index, document in enumerate(documents, start=1):
        filename = str(document["source_filename"])
        segments = _document_segments(str(document["version_id"]))
        page_count = len({segment.page_from for segment in segments if segment.page_from is not None})
        before = time.perf_counter()
        try:
            metadata = _metadata_payload(
                infer_document_metadata_from_segments(
                    filename,
                    segments,
                    max_segment_chars=args.segment_chars,
                )
            )
            result = ComparisonResult(
                document_id=str(document["document_id"]),
                version_id=str(document["version_id"]),
                source_filename=filename,
                status="ok",
                elapsed_seconds=round(time.perf_counter() - before, 3),
                segment_count=len(segments),
                page_count=page_count,
                metadata=metadata,
                audit=_audit(metadata, segments),
            )
        except Exception as exc:
            result = ComparisonResult(
                document_id=str(document["document_id"]),
                version_id=str(document["version_id"]),
                source_filename=filename,
                status="failed",
                elapsed_seconds=round(time.perf_counter() - before, 3),
                segment_count=len(segments),
                page_count=page_count,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(documents),
                    "source_filename": filename,
                    "status": result.status,
                    "elapsed_seconds": result.elapsed_seconds,
                    "audit_passed": bool(result.audit and result.audit["passed"]),
                    "error": result.error,
                }
            ),
            flush=True,
        )

    finished = datetime.now(UTC)
    report = {
        "summary": {
            "model": settings.ollama_metadata_model,
            "ollama_url": settings.ollama_url,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "documents": len(results),
            "completed": sum(item.status == "ok" for item in results),
            "failed": sum(item.status == "failed" for item in results),
            "audit_passed": sum(bool(item.audit and item.audit["passed"]) for item in results),
            "elapsed_seconds": round((finished - started).total_seconds(), 3),
            "applied": False,
        },
        "results": [asdict(item) for item in results],
    }
    output = args.output or ROOT / "test_reports" / (
        f"metadata_model_comparison_{settings.ollama_metadata_model.replace(':', '-')}_"
        f"{finished.strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"report": str(output), **report["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
