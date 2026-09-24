from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from manuals_rag_parsers.metadata import (
    _canonical_source_native_claims,
    _claim_fingerprint,
    _claim_scope_key,
    infer_document_metadata_from_segments,
)
from scripts.maintenance.backfill_document_metadata import (
    _apply_authoritative_metadata_defaults,
    _document_segments,
    _documents,
    _metadata_payload,
)


def _canonical_bytes(claims: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        claims,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one read-only LR-T extraction and audit its canonical claim ledger."
    )
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-chars", type=int, default=3000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    documents = _documents(document_ids=[args.document_id])
    if len(documents) != 1:
        parser.error(f"expected one document, found {len(documents)}")
    document = documents[0]
    segments = _document_segments(str(document["version_id"]))
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
    claims = list(metadata["metadata_claims"])
    expected = _canonical_source_native_claims(
        str(document["source_filename"]),
        str(metadata["title"]),
        segments,
    )
    fingerprints = [_claim_fingerprint(item) for item in claims]
    expected_fingerprints = {_claim_fingerprint(item) for item in expected}
    observed_fingerprints = set(fingerprints)
    duplicates = sorted(
        {fingerprint for fingerprint in fingerprints if fingerprints.count(fingerprint) > 1}
    )
    relations_by_scope: dict[tuple[str, str, str], set[str]] = {}
    for item in claims:
        relations_by_scope.setdefault(_claim_scope_key(item), set()).add(
            str(item.get("relation") or "")
        )
    role_overlap = sorted(
        scope
        for scope, relations in relations_by_scope.items()
        if "primary_product" in relations and "external_reference" in relations
    )
    nonliteral = []
    for item in claims:
        quote = str(item.get("source_quote") or "")
        page = item.get("page_from")
        if not quote or not any(
            segment.page_from == page and quote in segment.text for segment in segments
        ):
            nonliteral.append(_claim_fingerprint(item))

    ledger_bytes = _canonical_bytes(claims)
    audit = {
        "missing": sorted(expected_fingerprints - observed_fingerprints),
        "extra": sorted(observed_fingerprints - expected_fingerprints),
        "duplicate": duplicates,
        "role_overlap": role_overlap,
        "nonliteral_contiguous_provenance": sorted(nonliteral),
    }
    if any(audit.values()):
        raise SystemExit(f"FAIL CLOSED: {json.dumps(audit, ensure_ascii=False)}")
    result = {
        "document_id": str(document["document_id"]),
        "version_id": str(document["version_id"]),
        "source_filename": str(document["source_filename"]),
        "metadata_pipeline_version": metadata["metadata_pipeline_version"],
        "canonical_ledger_count": len(claims),
        "canonical_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "audit": audit,
        "canonical_ledger": claims,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: result[key] for key in (
                "document_id",
                "version_id",
                "source_filename",
                "metadata_pipeline_version",
                "canonical_ledger_count",
                "canonical_ledger_sha256",
                "audit",
            )},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
