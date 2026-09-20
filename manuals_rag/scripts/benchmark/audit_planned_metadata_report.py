#!/usr/bin/env python3
"""Fail-closed audit of planned metadata quotes against raw parsed page text."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manuals_rag_common.db import fetch_all
from manuals_rag_parsers.metadata import METADATA_PIPELINE_VERSION


def _quote_is_literal(version_id: str, item: dict[str, Any]) -> bool:
    quote = str(item.get("source_quote") or "")
    page_from = item.get("page_from")
    page_to = item.get("page_to") or page_from
    if not quote or page_from is None:
        return False
    rows = fetch_all(
        """
        select text_raw from logical_nodes
        where document_version_id = %s
          and page_from <= %s and page_to >= %s
        order by ordinal
        """,
        (version_id, page_to, page_from),
    )
    return any(quote in str(row.get("text_raw") or "") for row in rows)


def audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("planned report must contain a non-empty JSON array")
    documents = []
    for result in payload:
        failures: list[str] = []
        metadata = result.get("metadata") or {}
        if result.get("status") != "planned":
            failures.append("entry_not_planned")
        if metadata.get("metadata_pipeline_version") != METADATA_PIPELINE_VERSION:
            failures.append("pipeline_version_mismatch")
        claims = list(metadata.get("metadata_claims") or [])
        evidence = list(metadata.get("metadata_evidence") or [])
        if any(item.get("verification_status") == "unresolved" for item in claims):
            failures.append("unresolved_claims")
        audited_items = [
            item for item in [*claims, *evidence]
            if item.get("grounded") is True or item.get("verification_status") == "confirmed"
        ]
        nonliteral = [
            {
                "kind": item.get("kind"),
                "value": item.get("value"),
                "source_quote": item.get("source_quote"),
                "page_from": item.get("page_from"),
                "page_to": item.get("page_to"),
            }
            for item in audited_items
            if not _quote_is_literal(str(result.get("version_id")), item)
        ]
        if nonliteral:
            failures.append("nonliteral_or_unlocatable_source_quote")
        documents.append({
            "document_id": result.get("document_id"),
            "version_id": result.get("version_id"),
            "source_filename": result.get("source_filename"),
            "audited_grounded_items": len(audited_items),
            "nonliteral_items": nonliteral,
            "checks_passed": not failures,
            "failures": failures,
        })
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": str(path),
        "pipeline": METADATA_PIPELINE_VERSION,
        "checks_passed": all(item["checks_passed"] for item in documents),
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite audit artifact: {args.output}")
    try:
        result = audit(args.report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result["checks_passed"] else 1)


if __name__ == "__main__":
    main()
