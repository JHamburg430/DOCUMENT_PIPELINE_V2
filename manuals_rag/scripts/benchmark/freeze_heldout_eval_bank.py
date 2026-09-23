#!/usr/bin/env python3
"""Freeze generated eval cases only after persisted-source and split checks pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MANUALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "common" / "src"))

from manuals_rag_common.db import fetch_all


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {path} line {line_number}") from exc
        case = record.get("case") if isinstance(record.get("case"), dict) else record
        if not isinstance(case, dict):
            raise ValueError(f"record in {path} line {line_number} is not an eval case")
        records.append(case)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def referenced_document_ids(cases: list[dict[str, Any]]) -> set[str]:
    document_ids: set[str] = set()
    for case in cases:
        for key in ("source_document_id",):
            value = str(case.get(key) or "").strip()
            if value:
                document_ids.add(value)
        for item in case.get("expected_evidence") or []:
            if isinstance(item, dict) and item.get("source_document_id"):
                document_ids.add(str(item["source_document_id"]))
        graph = case.get("expected_evidence_graph") or {}
        for node in graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("source_document_id"):
                document_ids.add(str(node["source_document_id"]))
    return document_ids


def verify_and_freeze_cases(
    cases: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    tuning_document_ids: set[str],
    verified_at: str,
) -> list[dict[str, Any]]:
    if not cases:
        raise ValueError("cannot freeze an empty held-out bank")
    overlap = referenced_document_ids(cases) & tuning_document_ids
    if overlap:
        raise ValueError(f"held-out/tuning document overlap: {sorted(overlap)}")

    frozen: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id") or "").strip()
        chunk_id = str(case.get("source_chunk_id") or "").strip()
        if not case_id or case_id in seen_case_ids:
            raise ValueError(f"missing or duplicate case_id: {case_id!r}")
        seen_case_ids.add(case_id)
        chunk = chunks_by_id.get(chunk_id)
        if not chunk or chunk.get("is_active") is not True:
            raise ValueError(f"{case_id}: source chunk is missing or inactive")
        for field in ("source_document_id", "document_version_id"):
            if str(case.get(field) or "") != str(chunk.get(field) or ""):
                raise ValueError(f"{case_id}: {field} does not match persisted chunk")
        snippet = _normalized(case.get("expected_snippet"))
        content = _normalized(chunk.get("content"))
        if not snippet or snippet not in content:
            raise ValueError(f"{case_id}: expected snippet is not present in persisted chunk")
        missing_terms = [
            str(term)
            for term in case.get("expected_terms") or []
            if _normalized(term) not in snippet
        ]
        if missing_terms:
            raise ValueError(f"{case_id}: expected terms absent from snippet: {missing_terms}")

        source_hash = hashlib.sha256(str(chunk.get("content") or "").encode("utf-8")).hexdigest()
        frozen.append(
            {
                **case,
                "evaluation_split": "held_out",
                "adjudication": {
                    "status": "source_verified",
                    "method": "assistant_persisted_chunk_exact_snippet_v1",
                    "verified_at": verified_at,
                    "human_reviewed": False,
                    "source_chunk_sha256": source_hash,
                },
            }
        )
    return frozen


def _fetch_chunks(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = fetch_all(
        """
        select id, source_document_id, document_version_id, content, is_active
        from retrieval_chunks
        where id = any(%s)
        """,
        (chunk_ids,),
    )
    return {str(row["id"]): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tuning-dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest_output.exists():
        parser.error("refusing to overwrite a frozen dataset or manifest")

    cases = _load_jsonl(args.input)
    tuning_cases = [case for path in args.tuning_dataset for case in _load_jsonl(path)]
    verified_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    frozen = verify_and_freeze_cases(
        cases,
        _fetch_chunks([str(case.get("source_chunk_id") or "") for case in cases]),
        tuning_document_ids=referenced_document_ids(tuning_cases),
        verified_at=verified_at,
    )
    _write_jsonl(args.output, frozen)
    manifest = {
        "schema_version": 1,
        "frozen_at": verified_at,
        "input_path": str(args.input),
        "input_sha256": _sha256(args.input),
        "output_path": str(args.output),
        "output_sha256": _sha256(args.output),
        "ordered_case_ids": [str(case["case_id"]) for case in frozen],
        "case_count": len(frozen),
        "held_out_document_ids": sorted(referenced_document_ids(frozen)),
        "tuning_datasets": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.tuning_dataset
        ],
        "tuning_document_ids": sorted(referenced_document_ids(tuning_cases)),
        "document_disjoint": True,
        "adjudication_method": "assistant_persisted_chunk_exact_snippet_v1",
        "human_reviewed": False,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "case_count": len(frozen), "sha256": manifest["output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
