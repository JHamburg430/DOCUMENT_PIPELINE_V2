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
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "evals" / "src"))

from manuals_rag_common.db import fetch_all
from manuals_rag_evals.retrieval_eval import _query_aligned_expected_snippet, extract_anchor_terms


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
    punctuation_neutral = re.sub(r"[.;]+", " ", str(text or ""))
    return re.sub(r"\s+", " ", punctuation_neutral).strip().casefold()


def _field_context(content: str, expected_snippet: str = "") -> str:
    """Return the table/spec label that scopes a value-bearing source chunk."""

    if expected_snippet:
        snippet_index = content.casefold().find(expected_snippet.casefold())
        if snippet_index >= 0:
            scoped = content[max(0, snippet_index - 240) : snippet_index + len(expected_snippet)]
            return scoped.split(";", 1)[0].strip()
    prefix = re.split(r"\bCell value\s*:", content, maxsplit=1, flags=re.IGNORECASE)[0]
    return prefix.split(";", 1)[0].strip()


def missing_query_qualifiers(
    query: str,
    content: str,
    expected_snippet: str = "",
    source_context: str = "",
) -> list[str]:
    """Reject source-backed questions that drop decision-changing field qualifiers.

    Generated questions may be fluent and source-anchored while still becoming
    ambiguous when a table label such as ``X Reference distance`` is shortened to
    ``reference distance``.  Check only the value's field context so unrelated
    labels elsewhere in a row-group do not create false requirements.
    """

    field = _normalized(_field_context(content, expected_snippet))
    normalized_query = _normalized(query)
    rules = (
        ("x-axis", r"(?:^|\b)(?:x axis|x reference|x near|x far)(?:\b|$)", r"(?:^|\b)(?:x|x axis|horizontal)(?:\b|$)"),
        ("y-axis", r"(?:^|\b)(?:y axis|y reference|y near|y far)(?:\b|$)", r"(?:^|\b)(?:y|y axis|vertical)(?:\b|$)"),
        ("z-axis", r"(?:^|\b)(?:z axis|measurement range z|z measurement)(?:\b|$)", r"(?:^|\b)(?:z|z axis|height)(?:\b|$)"),
        ("input", r"\binput\b", r"\binput\b"),
        ("output", r"\boutput\b", r"\boutput\b"),
        ("near side", r"\bnear side\b", r"\bnear(?: side)?\b"),
        ("far side", r"\bfar side\b", r"\bfar(?: side)?\b"),
    )
    missing = [
        label
        for label, source_pattern, query_pattern in rules
        if re.search(source_pattern, field) and not re.search(query_pattern, normalized_query)
    ]
    # Row-group chunks can omit column headers even though the neighboring
    # structural context identifies a model family (for example VS-LxxxCX).
    # A family-wide question is ambiguous when sibling variants have different
    # values, so require at least one visible model-family prefix.
    header_context = " ".join(
        re.findall(
            r"(?:table header|column headers)\s*:\s*([^\n;]+)",
            source_context,
            flags=re.IGNORECASE,
        )
    )
    model_tokens = re.findall(
        r"\b[A-Z]{1,6}-[A-Z0-9]+(?:/[A-Z0-9-]+)*\b",
        header_context,
        flags=re.IGNORECASE,
    )
    model_prefixes = {
        re.sub(r"x+.*$", "", token, flags=re.IGNORECASE).rstrip("-/").casefold()
        for token in model_tokens
        if "x" in token.casefold()
    }
    compact_query = re.sub(r"[^a-z0-9]+", "", normalized_query)
    if model_prefixes and not any(
        re.sub(r"[^a-z0-9]+", "", prefix) in compact_query
        for prefix in model_prefixes
    ):
        missing.append("model variant")
    if re.search(
        r"\b(?:this|that|these|those)\s+"
        r"(?:camera|sensor|controller|device|product|unit|model|system|manual|series)\b",
        normalized_query,
    ):
        # Frozen evaluation questions are executed without conversational
        # context.  A demonstrative subject such as "this camera" therefore
        # cannot establish which source model the expected value applies to.
        missing.append("explicit subject")
    # Some manuals reuse the same metric label for distinct displayed
    # quantities.  The W500, for example, gives a ``Display range`` for both
    # workpiece conformity and received-light intensity.  A standalone eval
    # question that asks only for the display range cannot identify which
    # source row is authoritative even when the numeric bounds happen to be
    # equal.  Require the semantic quantity carried by the source evidence.
    if re.search(r"\bdisplay range\b", normalized_query):
        expected = _normalized(expected_snippet)
        display_quantity_rules = (
            ("workpiece conformity", r"\bworkpiece\b|\bconform(?:ity)?\b", r"\bworkpiece\b|\bconform(?:ity)?\b"),
            ("received light intensity", r"\breceived light intensity\b", r"\breceived light intensity\b"),
        )
        for label, source_pattern, query_pattern in display_quantity_rules:
            if re.search(source_pattern, expected) and not re.search(query_pattern, normalized_query):
                missing.append(label)
    return missing


def missing_answer_requirements(query: str, expected_snippet: str) -> list[str]:
    """Return answer-bearing requirements absent from the frozen evidence.

    Exact source anchoring is necessary but not sufficient: a generated question
    can ask for a duration while the selected snippet contains a neighboring
    current row, or enumerate I/O terminals that the snippet only partially
    covers.  Keep these checks narrow and deterministic so they reject malformed
    benchmark cases without attempting to judge arbitrary prose answers.
    """

    normalized_query = _normalized(query)
    normalized_snippet = _normalized(expected_snippet)
    missing: list[str] = []

    asks_duration = bool(
        re.search(r"\bhow long\b", normalized_query)
        or re.search(r"\b(?:charging|charge|response|cycle)\s+(?:time|duration)\b", normalized_query)
    )
    has_duration = bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|h|minutes?|mins?|seconds?|secs?|milliseconds?|msecs?|ms|s|days?)\b",
            normalized_snippet,
        )
    )
    if asks_duration and not has_duration:
        missing.append("duration value")

    compact_snippet = re.sub(r"[^a-z0-9]+", "", normalized_snippet)
    for match in re.finditer(r"\b(in|out)\s*(\d+)\s*[-–]\s*(\d+)\b", normalized_query):
        prefix, start_text, end_text = match.groups()
        start, end = int(start_text), int(end_text)
        if end < start or end - start > 32:
            continue
        absent = [
            f"{prefix.upper()}{number}"
            for number in range(start, end + 1)
            if f"{prefix}{number}" not in compact_snippet
        ]
        if absent:
            missing.append("I/O terminals " + ", ".join(absent))

    return missing


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
    reanchor_source_snippets: bool = False,
) -> list[dict[str, Any]]:
    if not cases:
        raise ValueError("cannot freeze an empty held-out bank")
    overlap = referenced_document_ids(cases) & tuning_document_ids
    if overlap:
        raise ValueError(f"held-out/tuning document overlap: {sorted(overlap)}")

    frozen: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for original_case in cases:
        case = dict(original_case)
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
        if reanchor_source_snippets:
            if case.get("expected_evidence"):
                raise ValueError(f"{case_id}: source re-anchoring is only supported for single-step cases")
            snippet_text = _query_aligned_expected_snippet(
                str(case.get("query") or ""),
                str(chunk.get("content") or ""),
            )
            terms = extract_anchor_terms(snippet_text)[:4]
            if not snippet_text or not terms:
                raise ValueError(f"{case_id}: source re-anchoring produced no usable evidence")
            case["expected_snippet"] = snippet_text
            case["expected_terms"] = terms
            case["anchor_terms"] = terms
        evidence_hashes: dict[str, str] = {}
        expected_evidence = case.get("expected_evidence") or []
        if not expected_evidence:
            missing_qualifiers = missing_query_qualifiers(
                str(case.get("query") or ""),
                str(chunk.get("content") or ""),
                str(case.get("expected_snippet") or ""),
                str((case.get("source_metadata") or {}).get("context_window") or ""),
            )
            if missing_qualifiers:
                raise ValueError(
                    f"{case_id}: query drops source qualifier(s): {', '.join(missing_qualifiers)}"
                )
            missing_requirements = missing_answer_requirements(
                str(case.get("query") or ""),
                str(case.get("expected_snippet") or ""),
            )
            if missing_requirements:
                raise ValueError(
                    f"{case_id}: expected snippet does not answer query requirement(s): "
                    f"{', '.join(missing_requirements)}"
                )
        if expected_evidence:
            for evidence in expected_evidence:
                if not isinstance(evidence, dict):
                    raise ValueError(f"{case_id}: expected evidence entry is not an object")
                evidence_chunk_id = str(evidence.get("chunk_id") or "").strip()
                evidence_chunk = chunks_by_id.get(evidence_chunk_id)
                if not evidence_chunk or evidence_chunk.get("is_active") is not True:
                    raise ValueError(f"{case_id}: expected evidence chunk is missing or inactive: {evidence_chunk_id}")
                evidence_document_id = str(evidence.get("source_document_id") or "").strip()
                if evidence_document_id != str(evidence_chunk.get("source_document_id") or ""):
                    raise ValueError(f"{case_id}: expected evidence document does not match persisted chunk")
                evidence_snippet = _normalized(evidence.get("snippet"))
                evidence_content = _normalized(evidence_chunk.get("content"))
                if not evidence_snippet or evidence_snippet not in evidence_content:
                    raise ValueError(f"{case_id}: expected evidence snippet is not present in persisted chunk")
                missing_evidence_terms = [
                    str(term)
                    for term in evidence.get("expected_terms") or []
                    if _normalized(term) not in evidence_snippet
                ]
                if missing_evidence_terms:
                    raise ValueError(
                        f"{case_id}: expected evidence terms absent from snippet: {missing_evidence_terms}"
                    )
                evidence_hashes[evidence_chunk_id] = hashlib.sha256(
                    str(evidence_chunk.get("content") or "").encode("utf-8")
                ).hexdigest()
        else:
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
                    "evidence_chunk_sha256": evidence_hashes,
                },
            }
        )
    return frozen


def partition_verified_cases(
    cases: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    tuning_document_ids: set[str],
    verified_at: str,
    reanchor_source_snippets: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze valid cases and retain an auditable record for every rejection."""

    frozen: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id") or "").strip()
        if not case_id or case_id in seen_case_ids:
            rejected.append(
                {
                    "case_id": case_id,
                    "query": str(case.get("query") or ""),
                    "reason": f"missing or duplicate case_id: {case_id!r}",
                }
            )
            continue
        seen_case_ids.add(case_id)
        try:
            frozen.extend(
                verify_and_freeze_cases(
                    [case],
                    chunks_by_id,
                    tuning_document_ids=tuning_document_ids,
                    verified_at=verified_at,
                    reanchor_source_snippets=reanchor_source_snippets,
                )
            )
        except ValueError as exc:
            rejected.append(
                {
                    "case_id": case_id,
                    "query": str(case.get("query") or ""),
                    "reason": str(exc),
                }
            )
    if not frozen:
        raise ValueError("cannot freeze a held-out bank with zero valid cases")
    return frozen, rejected


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
    parser.add_argument(
        "--rejections-output",
        type=Path,
        help="Write rejected cases and continue freezing valid cases instead of failing on the first defect.",
    )
    parser.add_argument(
        "--reanchor-source-snippets",
        action="store_true",
        help="Recompute each answer snippet from the current persisted source chunk before freezing.",
    )
    args = parser.parse_args()
    if args.output.exists() or args.manifest_output.exists() or (args.rejections_output and args.rejections_output.exists()):
        parser.error("refusing to overwrite a frozen dataset or manifest")

    cases = _load_jsonl(args.input)
    tuning_cases = [case for path in args.tuning_dataset for case in _load_jsonl(path)]
    verified_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    chunk_ids = {
        str(case.get("source_chunk_id") or "")
        for case in cases
    }
    chunk_ids.update(
        str(evidence.get("chunk_id") or "")
        for case in cases
        for evidence in case.get("expected_evidence") or []
        if isinstance(evidence, dict)
    )
    chunks_by_id = _fetch_chunks(sorted(chunk_id for chunk_id in chunk_ids if chunk_id))
    rejected: list[dict[str, Any]] = []
    if args.rejections_output:
        frozen, rejected = partition_verified_cases(
            cases,
            chunks_by_id,
            tuning_document_ids=referenced_document_ids(tuning_cases),
            verified_at=verified_at,
            reanchor_source_snippets=args.reanchor_source_snippets,
        )
        _write_jsonl(args.rejections_output, rejected)
    else:
        frozen = verify_and_freeze_cases(
            cases,
            chunks_by_id,
            tuning_document_ids=referenced_document_ids(tuning_cases),
            verified_at=verified_at,
            reanchor_source_snippets=args.reanchor_source_snippets,
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
        "input_case_count": len(cases),
        "case_count": len(frozen),
        "rejected_case_count": len(rejected),
        "rejections_path": str(args.rejections_output) if args.rejections_output else None,
        "rejections_sha256": _sha256(args.rejections_output) if args.rejections_output else None,
        "held_out_document_ids": sorted(referenced_document_ids(frozen)),
        "tuning_datasets": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.tuning_dataset
        ],
        "tuning_document_ids": sorted(referenced_document_ids(tuning_cases)),
        "document_disjoint": True,
        "adjudication_method": "assistant_persisted_chunk_exact_snippet_v1",
        "human_reviewed": False,
        "source_snippets_reanchored": args.reanchor_source_snippets,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_count": len(frozen),
                "rejected_case_count": len(rejected),
                "sha256": manifest["output_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
