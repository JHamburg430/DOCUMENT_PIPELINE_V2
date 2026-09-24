#!/usr/bin/env python3
"""Fail-closed visual audit for metadata claims and page-grounded QA.

This tool is deliberately read-only.  It uses rendered page images to audit a
planned metadata report (or currently persisted metadata), proposes realistic
page-specific questions, and asks a second vision model to answer those
questions without seeing the proposed answers.  The immutable output artifact
is a gate for metadata rollout; it never mutates PostgreSQL or Qdrant.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import fitz
import httpx

from manuals_rag_common.config import settings
from manuals_rag_common.db import fetch_all
from manuals_rag_common.storage import ObjectStore
from manuals_rag_parsers.metadata import METADATA_PIPELINE_VERSION


AUDIT_VERSION = "visual_document_understanding_v2"
CLAIMS_PER_VISUAL_CALL = 4
SCOPED_RELATIONS = {
    "primary_product", "applies_to", "compatible_with", "accessory_for",
    "document_revision",
}
UNIT_TOKENS = {
    "a", "bar", "db", "deg", "degree", "degrees", "fps", "hz", "inch",
    "inches", "kg", "khz", "kw", "m", "ma", "mbps", "mhz", "mm", "ms",
    "nm", "pa", "pixel", "pixels", "s", "v", "vac", "vdc", "w",
}
PROHIBITED_QUESTION_RE = re.compile(
    r"\b(?:address|contact|copyright|e-?mail|fax|phone|telephone|website)\b",
    re.IGNORECASE,
)


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\bver\.(?=\d)", "", text)
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text))


def _tokens(value: Any) -> list[str]:
    return _normalized(value).split()


def _numbers(value: Any) -> list[str]:
    raw = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for identifier in re.findall(r"\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+)+\b", raw):
        raw = raw.replace(identifier, " ")
    return re.findall(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?", raw)


def _identifiers(value: Any) -> set[str]:
    raw = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return {
        re.sub(r"[^a-z0-9]", "", token)
        for token in re.findall(r"\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+)+\b", raw)
    }


def _units(value: Any) -> set[str]:
    return {token for token in _tokens(value) if token in UNIT_TOKENS}


def answer_matches(expected: str, actual: str) -> tuple[bool, dict[str, Any]]:
    expected_tokens = _tokens(expected)
    actual_tokens = _tokens(actual)
    expected_counter = Counter(expected_tokens)
    actual_counter = Counter(actual_tokens)
    overlap = sum((expected_counter & actual_counter).values())
    precision = overlap / max(len(actual_tokens), 1)
    recall = overlap / max(len(expected_tokens), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    number_match = _numbers(expected) == _numbers(actual)
    identifier_match = _identifiers(expected) == _identifiers(actual)
    unit_match = _units(expected) == _units(actual)
    containment = bool(_normalized(expected)) and (
        _normalized(expected) in _normalized(actual) or _normalized(actual) in _normalized(expected)
    )
    token_subset = bool(expected_tokens) and not (expected_counter - actual_counter)
    passed = bool(expected_tokens and actual_tokens) and number_match and identifier_match and unit_match and (
        containment or token_subset or f1 >= 0.85
    )
    return passed, {
        "token_f1": round(f1, 4),
        "number_match": number_match,
        "identifier_match": identifier_match,
        "unit_match": unit_match,
        "containment": containment,
        "token_subset": token_subset,
    }


def answer_matches_in_context(expected: str, actual: str, question: str) -> tuple[bool, dict[str, Any]]:
    """Allow an answer to inherit units only when the question states them."""
    passed, checks = answer_matches(expected, actual)
    if passed:
        return passed, checks
    expected_identifiers = _identifiers(expected)
    actual_identifiers = _identifiers(actual)
    question_identifiers = _identifiers(question)
    contextual_identifier_match = (
        expected_identifiers.issubset(actual_identifiers)
        and (actual_identifiers - expected_identifiers).issubset(question_identifiers)
    )
    checks["identifier_match_in_context"] = contextual_identifier_match
    if not checks["number_match"] or not contextual_identifier_match:
        return False, checks
    token_match = checks["containment"] or checks["token_subset"] or checks["token_f1"] >= 0.85
    if checks["unit_match"] and token_match:
        return True, checks
    expected_units = _units(expected)
    if expected_units and expected_units.issubset(_units(question)):
        checks["units_inherited_from_question"] = True
        return token_match, checks
    checks["units_inherited_from_question"] = False
    return False, checks


def remap_batch_results(
    claim_batch: list[dict[str, Any]], raw_results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Map a model's batch-local c1..cN identifiers back to global IDs."""
    local_to_global = {
        f"c{index}": str(item["claim_id"])
        for index, item in enumerate(claim_batch, start=1)
    }
    remapped = [
        {**item, "claim_id": local_to_global[str(item.get("claim_id"))]}
        for item in raw_results
        if str(item.get("claim_id")) in local_to_global
    ]
    return remapped, local_to_global


def quote_is_anchored(quote: str, page_text: str) -> bool:
    return bool(_normalized(quote)) and _normalized(quote) in _normalized(page_text)


def reanchor_answer(answer: str, page_text: str) -> str | None:
    """Recompute a short answer-bearing source span from persisted page text."""
    expected_tokens = _tokens(answer)
    expected_numbers = set(_numbers(answer))
    expected_identifiers = _identifiers(answer)
    expected_units = _units(answer)
    candidates: list[tuple[float, str]] = []
    lines = [" ".join(raw.split()).strip() for raw in page_text.splitlines() if raw.strip()]
    for start in range(len(lines)):
        for width in range(1, min(8, len(lines) - start) + 1):
            span_lines = lines[start:start + width]
            line = "\n".join(span_lines)
            if not expected_numbers.issubset(set(_numbers(line))):
                continue
            if not expected_identifiers.issubset(_identifiers(line)):
                continue
            if expected_identifiers and not any(
                expected_identifiers.issubset(_identifiers(row))
                and expected_numbers.issubset(set(_numbers(row)))
                for row in span_lines
            ):
                # Do not fabricate an answer by joining an identifier from one
                # table row with a value from a neighbouring row.
                continue
            if not expected_units.issubset(_units(line)):
                continue
            overlap = sum((Counter(expected_tokens) & Counter(_tokens(line))).values())
            recall = overlap / max(len(expected_tokens), 1)
            has_answer_keys = bool(expected_numbers) and bool(expected_identifiers or expected_units)
            if _normalized(answer) in _normalized(line) or recall >= 0.6 or has_answer_keys:
                candidates.append((recall, line))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    return candidates[0][1][:1200]


def claim_support_passes(claim: dict[str, Any], result: dict[str, Any], page_text: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    claim_value = str(claim.get("value") or "")
    evidence = str(result.get("evidence_text") or "")
    subject = str(claim.get("subject") or "").strip()
    scoped = bool(subject and str(claim.get("relation") or "") in SCOPED_RELATIONS)
    exact_visual_binding = (
        bool(_normalized(claim_value))
        and _normalized(claim_value) in _normalized(evidence)
        and (not scoped or _normalized(subject) in _normalized(evidence))
    )
    explanation = _normalized(result.get("explanation"))
    verdict_text_supports = any(
        phrase in explanation
        for phrase in ("claim is supported", "claim supported", "supports the claim", "therefore the claim is supported")
    )
    reconciled_model_flags = exact_visual_binding and verdict_text_supports
    if result.get("supported") is not True and not reconciled_model_flags:
        failures.append("auditor_did_not_confirm")
    if result.get("conflicting") is True and not reconciled_model_flags:
        failures.append("auditor_reported_conflict")
    visible_value = str(result.get("visible_value") or "")
    value_ok, _ = answer_matches(claim_value, visible_value)
    value_ok = value_ok or _normalized(claim_value) in _normalized(visible_value)
    if not value_ok:
        failures.append("visible_value_mismatch")
    source_anchor = str(claim.get("source_quote") or "")
    if not quote_is_anchored(source_anchor, page_text):
        failures.append("claim_source_quote_not_contiguous_in_page_text")
    if _normalized(claim_value) not in _normalized(source_anchor):
        failures.append("claim_source_quote_missing_value")
    if scoped:
        subject_haystack = f"{result.get('visible_subject') or ''} {evidence}"
        subject_ok, _ = answer_matches(subject, subject_haystack)
        if not subject_ok and _normalized(subject) not in _normalized(subject_haystack):
            failures.append("visual_evidence_missing_subject_scope")
        if _normalized(subject) not in _normalized(source_anchor):
            failures.append("claim_source_quote_missing_subject_scope")
    return not failures, failures


def adjudication_can_override(primary_result: dict[str, Any], adjudication_passed: bool) -> bool:
    """Permit corroboration of omissions, never an explicit visual conflict."""
    return primary_result.get("conflicting") is not True and adjudication_passed


def anchored_unit_completion_passes(
    expected: str,
    actual: str,
    comparison: dict[str, Any],
    source_anchor: str | None,
    independent_anchor: str | None,
) -> bool:
    """Accept a verifier-supplied unit only when both answers re-anchor."""
    return bool(
        comparison.get("number_match")
        and comparison.get("identifier_match_in_context", comparison.get("identifier_match"))
        and not _units(expected)
        and _units(actual)
        and source_anchor
        and independent_anchor
    )


def dual_visual_consensus_passes(
    expected: str,
    actual: str,
    comparison: dict[str, Any],
    proposed_evidence: str,
    independent_evidence: str,
) -> bool:
    """Accept image-only facts only when two models agree on the answer.

    Diagram labels and dimensions are often absent from parsed page text. In
    that case, requiring a text anchor would turn the visual audit back into an
    OCR audit. Consensus remains strict about numbers and identifiers and
    requires both independently produced visual evidence descriptions. A unit
    may be completed by one model only when the other answer is otherwise a
    literal subset (for example ``145.7`` versus ``145.7 mm``).
    """
    if not proposed_evidence.strip() or not independent_evidence.strip():
        return False
    if not comparison.get("number_match"):
        return False
    if not comparison.get("identifier_match_in_context", comparison.get("identifier_match")):
        return False
    token_match = bool(
        comparison.get("containment")
        or comparison.get("token_subset")
        or float(comparison.get("token_f1") or 0.0) >= 0.85
    )
    if not token_match:
        return False
    if comparison.get("unit_match"):
        return True
    expected_units = _units(expected)
    actual_units = _units(actual)
    return bool(expected_units) != bool(actual_units)


def _source_state() -> dict[str, Any]:
    """Fingerprint both repository state and this exact audit implementation."""
    revision = _git_revision()
    supplied_branch = os.getenv("SOURCE_BRANCH", "").strip()
    supplied_dirty = os.getenv("SOURCE_DIRTY", "").strip().lower()
    supplied_status_sha256 = os.getenv("SOURCE_STATUS_SHA256", "").strip()
    if supplied_branch and supplied_dirty in {"true", "false"} and supplied_status_sha256:
        return {
            "source_revision": revision,
            "source_branch": supplied_branch,
            "source_dirty": supplied_dirty == "true",
            "source_status_sha256": supplied_status_sha256,
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
        branch = subprocess.run(
            ["git", "branch", "--show-current"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("source status unavailable") from exc
    return {
        "source_revision": revision,
        "source_branch": branch,
        "source_dirty": bool(status),
        "source_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def select_pages(
    *, page_count: int, claims: Iterable[dict[str, Any]], page_metrics: list[dict[str, Any]], max_question_pages: int
) -> tuple[list[int], list[int]]:
    claim_pages = sorted({
        int(item["page_from"])
        for item in claims
        if item.get("page_from") is not None and 1 <= int(item["page_from"]) <= page_count
    })
    ranked = sorted(
        page_metrics,
        key=lambda item: (int(item.get("table_count") or 0), int(item.get("text_chars") or 0), -int(item["page"])),
        reverse=True,
    )
    candidates = [1, page_count, *[int(item["page"]) for item in ranked]]
    question_pages: list[int] = []
    for page in [*claim_pages, *candidates]:
        if 1 <= page <= page_count and page not in question_pages:
            question_pages.append(page)
        if len(question_pages) >= max(max_question_pages, len(claim_pages)):
            break
    return claim_pages, sorted(question_pages)


def _git_revision() -> str:
    supplied = os.getenv("SOURCE_REVISION", "").strip()
    if supplied:
        return supplied
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("source revision unavailable; set SOURCE_REVISION") from exc
    return result.stdout.strip()


def _page_metrics(version_id: str) -> list[dict[str, Any]]:
    return fetch_all(
        """
        select page_from as page,
               count(*) filter (where table_json is not null) as table_count,
               sum(length(coalesce(text_raw, ''))) as text_chars
        from logical_nodes
        where document_version_id = %s and page_from is not null
        group by page_from order by page_from
        """,
        (version_id,),
    )


def _page_text(version_id: str, page: int) -> str:
    rows = fetch_all(
        """
        select text_raw from logical_nodes
        where document_version_id = %s and page_from <= %s and page_to >= %s
        order by ordinal
        """,
        (version_id, page, page),
    )
    return "\n".join(str(row.get("text_raw") or "") for row in rows if str(row.get("text_raw") or "").strip())


def _read_s3(store: ObjectStore, uri: str) -> bytes:
    parsed = urlparse(uri)
    response = store.client.get_object(parsed.netloc, parsed.path.lstrip("/"))
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _page_png(
    store: ObjectStore,
    document: dict[str, Any],
    page: int,
    *,
    terms: Iterable[str] = (),
    scale: float = 2.5,
) -> tuple[bytes, str, str, list[float] | None]:
    """Render a high-resolution full page or an evidence-targeted horizontal crop."""
    pdf_bytes = _read_s3(store, str(document["storage_uri"]))
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page < 1 or page > pdf.page_count:
            raise ValueError(f"page {page} outside PDF range 1-{pdf.page_count}")
        pdf_page = pdf.load_page(page - 1)
        rectangles = [
            rectangle
            for term in dict.fromkeys(str(value).strip() for value in terms if str(value).strip())
            for rectangle in pdf_page.search_for(term)
        ]
        clip: fitz.Rect | None = None
        if rectangles:
            y0 = max(0.0, min(rect.y0 for rect in rectangles) - 36.0)
            y1 = min(pdf_page.rect.height, max(rect.y1 for rect in rectangles) + 36.0)
            clip = fitz.Rect(0.0, y0, pdf_page.rect.width, y1)
        png = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, clip=clip).tobytes("png")
    finally:
        pdf.close()
    return (
        png,
        str(document["storage_uri"]),
        "targeted_original_pdf_crop" if clip is not None else "high_resolution_original_pdf_page",
        [round(clip.x0, 2), round(clip.y0, 2), round(clip.x1, 2), round(clip.y1, 2)] if clip is not None else None,
    )


def _vision_json(
    *, client: httpx.Client, model: str, prompt: str, image: bytes, schema: dict[str, Any],
    timeout_seconds: float, num_predict: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, 3):
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": schema,
            "messages": [{
                "role": "user", "content": prompt,
                "images": [base64.b64encode(image).decode("ascii")],
            }],
            "options": {
                "temperature": 0.0, "num_ctx": 8192,
                "num_predict": num_predict * attempt, "num_batch": 64,
            },
            "keep_alive": "10m",
            "think": False,
        }
        response = client.post("/api/chat", json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        body = response.json()
        content = str((body.get("message") or {}).get("content") or "").strip()
        try:
            if not content:
                raise ValueError(f"{model} returned empty visual audit content")
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            continue
        return parsed, {
            "response_model": body.get("model"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "total_duration_ns": body.get("total_duration"),
            "load_duration_ns": body.get("load_duration"),
            "response_attempt": attempt,
        }
    raise ValueError(f"{model} returned invalid visual audit content after retry: {last_error}")


def _auditor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "readable": {"type": "boolean"},
            "claim_results": {"type": "array", "items": {"type": "object", "properties": {
                "claim_id": {"type": "string"}, "supported": {"type": "boolean"},
                "conflicting": {"type": "boolean"}, "visible_value": {"type": "string"},
                "visible_subject": {"type": "string"}, "evidence_text": {"type": "string"},
                "explanation": {"type": "string"},
            }, "required": ["claim_id", "supported", "conflicting", "visible_value", "visible_subject", "evidence_text", "explanation"]}},
            "questions": {"type": "array", "items": {"type": "object", "properties": {
                "question_id": {"type": "string"}, "question": {"type": "string"},
                "expected_answer": {"type": "string"}, "answer_type": {"type": "string"},
                "qualifiers": {"type": "array", "items": {"type": "string"}},
                "evidence_text": {"type": "string"},
            }, "required": ["question_id", "question", "expected_answer", "answer_type", "qualifiers", "evidence_text"]}},
        },
        "required": ["readable", "claim_results", "questions"],
    }


def _answer_schema() -> dict[str, Any]:
    return {
        "type": "object", "properties": {
            "answers": {"type": "array", "items": {"type": "object", "properties": {
                "question_id": {"type": "string"}, "answerable": {"type": "boolean"},
                "answer": {"type": "string"}, "evidence_text": {"type": "string"},
                "qualifiers": {"type": "array", "items": {"type": "string"}},
            }, "required": ["question_id", "answerable", "answer", "evidence_text", "qualifiers"]}},
        }, "required": ["answers"],
    }


def _documents(filenames: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    params: list[Any] = [settings.default_corpus_id]
    filters = ["sd.corpus_id = %s"]
    if filenames:
        filters.append("sd.source_filename = any(%s)")
        params.append(filenames)
    suffix = " limit %s" if limit else ""
    if limit:
        params.append(limit)
    return fetch_all(
        f"""
        select sd.id as document_id, sd.tenant_id, sd.source_filename, sd.storage_uri,
               sd.title, dv.id as version_id, dv.page_count, dme.model as metadata_model,
               dme.metadata_json
        from source_documents sd
        join document_versions dv on dv.id = sd.current_version_id
        left join document_metadata_extractions dme on dme.source_document_id = sd.id
        where {' and '.join(filters)} order by sd.source_filename {suffix}
        """,
        tuple(params),
    )


def _load_report(path: Path) -> list[dict[str, Any]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(body, dict):
        body = body.get("results") or body.get("documents")
    if not isinstance(body, list) or not body:
        raise ValueError("metadata report must contain a non-empty result list")
    return body


def _merge_report(documents: list[dict[str, Any]], report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item["document_id"]): item for item in documents}
    merged: list[dict[str, Any]] = []
    for entry in report:
        item = by_id.get(str(entry.get("document_id")))
        if item is None:
            # A caller may select a strict filename subset from a wider
            # checkpoint report.  Unselected entries remain covered by the
            # report hash but are not silently treated as audited.
            continue
        if str(item["version_id"]) != str(entry.get("version_id")):
            raise ValueError(f"report version is stale for document {entry.get('document_id')}")
        if entry.get("status") not in {"planned", "ok", "applied", "skipped_current"}:
            raise ValueError(f"report entry is not auditable: {entry.get('status')}")
        merged.append({**item, "metadata_json": entry.get("metadata") or {}, "metadata_model": entry.get("model")})
    if len(merged) != len(documents):
        missing = sorted(set(by_id) - {str(item["document_id"]) for item in merged})
        raise ValueError(f"selected documents missing from metadata report: {missing}")
    return merged


def _claims(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int | None]] = set()
    for item in [*(metadata.get("metadata_claims") or []), *(metadata.get("metadata_evidence") or [])]:
        status = item.get("verification_status")
        if item.get("grounded") is not True or status in {"rejected", "conflicting", "unresolved"}:
            continue
        fingerprint = (
            str(item.get("kind") or ""), _normalized(item.get("value")),
            str(item.get("relation") or ""), _normalized(item.get("subject")), item.get("page_from"),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        audited.append(item)
    return audited


def audit_document(
    document: dict[str, Any], *, store: ObjectStore, client: httpx.Client, auditor_model: str,
    answer_model: str, max_question_pages: int, max_questions_per_page: int,
    question_generation_attempts: int,
    min_verified_questions: int, timeout_seconds: float,
) -> dict[str, Any]:
    metadata = document.get("metadata_json") or {}
    claims = _claims(metadata)
    claim_pages, selected_pages = select_pages(
        page_count=int(document["page_count"]), claims=claims,
        page_metrics=_page_metrics(str(document["version_id"])), max_question_pages=max_question_pages,
    )
    page_results: list[dict[str, Any]] = []
    for page in selected_pages:
        page_claims = [item for item in claims if int(item.get("page_from") or -1) == page]
        claim_inputs = [
            {"claim_id": f"c{index}", "kind": item.get("kind"), "value": item.get("value"),
             "relation": item.get("relation"), "subject": item.get("subject") or ""}
            for index, item in enumerate(page_claims, start=1)
        ]
        png, image_uri, image_source, full_page_clip = _page_png(store, document, page)
        page_text = _page_text(str(document["version_id"]), page)
        raw_questions: list[dict[str, Any]] = []
        filtered_questions: list[dict[str, Any]] = []
        questions: list[dict[str, Any]] = []
        question_usages: list[dict[str, Any]] = []
        question_readability: list[bool] = []
        seen_question_text: set[str] = set()
        for generation_attempt in range(1, question_generation_attempts + 1):
            prior_questions = [str(item.get("question") or "") for item in questions]
            exclusion = (
                f" Do not repeat these prior questions: {json.dumps(prior_questions, ensure_ascii=False)}."
                if prior_questions else ""
            )
            question_prompt = (
                "Inspect only the supplied manual page image. Propose at most "
                f"{max_questions_per_page} realistic standalone technical retrieval questions answerable from this page. "
                "Preserve every model, row/column, polarity, axis, unit, and version qualifier. expected_answer and "
                "evidence_text must contain the actual answer-bearing value. Include the unit in expected_answer whenever "
                "the page displays one. Do not ask about page layout, generic document identity, marketing copy, addresses, "
                "phone numbers, or contact details. Prefer specifications, settings, compatibility, procedures, warnings, "
                "and troubleshooting facts. Each question must target one atomic row/value or one short procedure; do not "
                "ask for a range, all compatible models, or multiple table rows. Return no claim results and use q1, q2 IDs."
                f"{exclusion}\n\nClaims: []"
            )
            question_audit, question_usage = _vision_json(
                client=client, model=auditor_model, prompt=question_prompt, image=png,
                schema=_auditor_schema(), timeout_seconds=timeout_seconds, num_predict=1200,
            )
            generated = list(question_audit.get("questions") or [])[:max_questions_per_page]
            raw_questions.extend(generated)
            question_readability.append(question_audit.get("readable") is True or bool(generated))
            question_usages.append({"kind": "question_generation", "generation_attempt": generation_attempt, **question_usage})
            for local_index, question in enumerate(generated, start=1):
                if PROHIBITED_QUESTION_RE.search(str(question.get("question") or "")):
                    filtered_questions.append(question)
                    continue
                fingerprint = _normalized(question.get("question"))
                if not fingerprint or fingerprint in seen_question_text:
                    continue
                seen_question_text.add(fingerprint)
                questions.append({**question, "question_id": f"a{generation_attempt}_q{local_index}"})
        claim_batches = [
            claim_inputs[offset: offset + CLAIMS_PER_VISUAL_CALL]
            for offset in range(0, len(claim_inputs), CLAIMS_PER_VISUAL_CALL)
        ]
        audited_claim_results: list[dict[str, Any]] = []
        audit_usage: list[dict[str, Any]] = list(question_usages)
        audit_regions: list[dict[str, Any]] = []
        question_effectively_readable = bool(question_readability) and all(question_readability)
        readable_results: list[bool] = [question_effectively_readable]
        for batch_index, claim_batch in enumerate(claim_batches):
            prompt_batch = [
                {**item, "claim_id": f"c{index}"}
                for index, item in enumerate(claim_batch, start=1)
            ]
            crop_terms = [
                str(item.get(key) or "")
                for item in claim_batch
                for key in ("subject", "value")
                if item.get(key)
            ]
            claim_png, _claim_uri, claim_image_source, claim_clip = _page_png(
                store, document, page, terms=crop_terms
            )
            prompt = (
                "Inspect only the supplied manual page image. Verify each proposed metadata claim against visible "
                "page content. A supported claim requires a short verbatim evidence_text containing the claimed value; "
                "for scoped relations it must also visibly bind the subject. Mark conflict when the page contradicts it. "
                "Return exactly one claim_result per input claim and return an empty questions array.\n\n"
                f"Claims:\n{json.dumps(prompt_batch, ensure_ascii=False)}"
            )
            audited, usage = _vision_json(
                client=client, model=auditor_model, prompt=prompt, image=claim_png,
                schema=_auditor_schema(), timeout_seconds=timeout_seconds, num_predict=1800,
            )
            raw_claim_results = list(audited.get("claim_results") or [])
            remapped_results, local_to_global = remap_batch_results(claim_batch, raw_claim_results)
            audited_claim_results.extend(remapped_results)
            unique_local_ids = {
                str(item.get("claim_id")) for item in raw_claim_results
                if str(item.get("claim_id")) in local_to_global
            }
            retry_records: list[dict[str, Any]] = []
            returned_global_ids = {str(item.get("claim_id")) for item in remapped_results}
            for local_id, global_id in local_to_global.items():
                if local_id in unique_local_ids:
                    continue
                retry_item = next(item for item in claim_batch if str(item["claim_id"]) == global_id)
                retry_prompt_item = {**retry_item, "claim_id": "c1"}
                retry_terms = [str(retry_item.get(key) or "") for key in ("subject", "value") if retry_item.get(key)]
                retry_png, _retry_uri, retry_source, retry_clip = _page_png(
                    store, document, page, terms=retry_terms
                )
                retry_prompt = (
                    "Inspect only the supplied manual page image. Verify this metadata claim against visible page "
                    "content. A supported claim requires a short verbatim evidence_text containing the claimed value; "
                    "for a scoped relation it must also visibly bind the subject. Mark conflict when contradicted. "
                    "Return exactly one claim_result with claim_id c1 and an empty questions array.\n\n"
                    f"Claims:\n{json.dumps([retry_prompt_item], ensure_ascii=False)}"
                )
                retry_audited, retry_usage = _vision_json(
                    client=client, model=auditor_model, prompt=retry_prompt, image=retry_png,
                    schema=_auditor_schema(), timeout_seconds=timeout_seconds, num_predict=700,
                )
                retry_raw = list(retry_audited.get("claim_results") or [])
                retry_remapped, _retry_mapping = remap_batch_results([retry_item], retry_raw)
                audited_claim_results.extend(retry_remapped)
                returned_global_ids.update(str(item.get("claim_id")) for item in retry_remapped)
                audit_usage.append({
                    "kind": "claim_audit_retry", "batch_index": batch_index,
                    "global_claim_id": global_id, **retry_usage,
                })
                retry_records.append({
                    "global_claim_id": global_id,
                    "raw_claim_results": retry_raw,
                    "image_source": retry_source,
                    "clip": retry_clip,
                    "image_sha256": hashlib.sha256(retry_png).hexdigest(),
                    "model_readable": retry_audited.get("readable") is True,
                })
            effectively_readable = (
                audited.get("readable") is True
                or returned_global_ids == {str(item["claim_id"]) for item in claim_batch}
            )
            audit_usage.append({"kind": "claim_audit", "batch_index": batch_index, **usage})
            audit_regions.append({
                "batch_index": batch_index,
                "claim_ids": [str(item["claim_id"]) for item in claim_batch],
                "local_to_global_claim_ids": local_to_global,
                "raw_claim_results": raw_claim_results,
                "single_claim_retries": retry_records,
                "image_source": claim_image_source,
                "clip": claim_clip,
                "image_sha256": hashlib.sha256(claim_png).hexdigest(),
                "model_readable": audited.get("readable") is True,
                "effective_readable": effectively_readable,
            })
            readable_results.append(effectively_readable)
        result_by_id = {str(item.get("claim_id")): item for item in audited_claim_results}
        claim_results = []
        claim_adjudications: list[dict[str, Any]] = []
        for claim_input, claim in zip(claim_inputs, page_claims, strict=True):
            visual = result_by_id.get(str(claim_input["claim_id"])) or {}
            passed, failures = claim_support_passes(claim, visual, page_text)
            adjudication_visual: dict[str, Any] | None = None
            adjudication_failures: list[str] = []
            accepted_by_adjudication = False
            if not passed:
                adjudication_terms = [
                    str(value) for value in (claim.get("subject"), claim.get("value")) if value
                ]
                adjudication_png, _adjudication_uri, adjudication_source, adjudication_clip = _page_png(
                    store, document, page, terms=adjudication_terms
                )
                adjudication_input = {
                    "claim_id": "c1", "kind": claim.get("kind"), "value": claim.get("value"),
                    "relation": claim.get("relation"), "subject": claim.get("subject") or "",
                }
                adjudication_prompt = (
                    "Independently adjudicate this disputed metadata claim using only the supplied manual page image. "
                    "Check the exact table row or nearby source context. Set supported true only if a short verbatim "
                    "evidence_text visibly binds the exact value to the subject; set conflicting true if contradicted. "
                    "Return exactly one claim_result with claim_id c1 and an empty questions array.\n\n"
                    f"Claims:\n{json.dumps([adjudication_input], ensure_ascii=False)}"
                )
                adjudicated, adjudication_usage = _vision_json(
                    client=client, model=answer_model, prompt=adjudication_prompt, image=adjudication_png,
                    schema=_auditor_schema(), timeout_seconds=timeout_seconds, num_predict=700,
                )
                adjudication_visual = next(iter(adjudicated.get("claim_results") or []), {})
                adjudication_passed, adjudication_failures = claim_support_passes(
                    claim, adjudication_visual, page_text
                )
                accepted_by_adjudication = adjudication_can_override(visual, adjudication_passed)
                audit_usage.append({
                    "kind": "claim_disagreement_adjudication", "claim_id": claim_input["claim_id"],
                    **adjudication_usage,
                })
                claim_adjudications.append({
                    "claim_id": claim_input["claim_id"], "image_source": adjudication_source,
                    "clip": adjudication_clip, "image_sha256": hashlib.sha256(adjudication_png).hexdigest(),
                    "visual_result": adjudication_visual, "passed": adjudication_passed,
                    "failures": adjudication_failures,
                })
            claim_results.append({
                "claim": claim, "visual_result": visual,
                "adjudication_visual_result": adjudication_visual,
                "accepted_by_adjudication": accepted_by_adjudication,
                "passed": passed or accepted_by_adjudication,
                "failures": [] if accepted_by_adjudication else failures,
                "adjudication_failures": adjudication_failures,
            })
        verifier_answers: dict[str, dict[str, Any]] = {}
        answer_usage: dict[str, Any] | None = None
        if questions:
            blind_questions = [{"question_id": q.get("question_id"), "question": q.get("question")} for q in questions]
            answer_prompt = (
                "Answer each question using only the supplied manual page image. Do not use external knowledge. "
                "Return a concise answer with every required number, unit, identifier, qualifier, and a short verbatim "
                "answer-bearing evidence_text. If the page does not unambiguously answer it, set answerable false.\n\n"
                f"Questions:\n{json.dumps(blind_questions, ensure_ascii=False)}"
            )
            verified, answer_usage = _vision_json(
                client=client, model=answer_model, prompt=answer_prompt, image=png,
                schema=_answer_schema(), timeout_seconds=timeout_seconds, num_predict=1200,
            )
            verifier_answers = {str(item.get("question_id")): item for item in verified.get("answers") or []}
        qa_results = []
        for question in questions:
            failures: list[str] = []
            expected = str(question.get("expected_answer") or "")
            source_anchor = reanchor_answer(expected, page_text)
            verified = verifier_answers.get(str(question.get("question_id"))) or {}
            if verified.get("answerable") is not True:
                failures.append("independent_model_not_answerable")
            actual_answer = str(verified.get("answer") or "")
            matched, comparison = answer_matches_in_context(
                expected,
                actual_answer,
                str(question.get("question") or ""),
            )
            independent_anchor = reanchor_answer(actual_answer, page_text)
            if not matched and anchored_unit_completion_passes(
                expected, actual_answer, comparison, source_anchor, independent_anchor
            ):
                matched = True
                comparison["unit_completed_from_independent_source_anchor"] = True
            text_anchored = bool(source_anchor and independent_anchor and matched)
            visual_consensus = dual_visual_consensus_passes(
                expected,
                actual_answer,
                comparison,
                str(question.get("evidence_text") or ""),
                str(verified.get("evidence_text") or ""),
            )
            evidence_mode = (
                "text_anchored_dual_model" if text_anchored
                else "visual_only_dual_model" if visual_consensus
                else "unverified"
            )
            if evidence_mode == "unverified":
                if source_anchor is None:
                    failures.append("proposed_answer_could_not_be_reanchored")
                if independent_anchor is None:
                    failures.append("independent_answer_could_not_be_reanchored")
                if not matched:
                    failures.append("independent_answer_mismatch")
                failures.append("no_text_anchor_or_dual_visual_consensus")
            normalized_expected = _normalized(expected)
            expected_qualifiers = {
                _normalized(value)
                for value in question.get("qualifiers") or []
                if _normalized(value) and _normalized(value) in normalized_expected
            }
            actual_qualifiers = _normalized(
                f"{verified.get('answer') or ''} {verified.get('evidence_text') or ''} {' '.join(verified.get('qualifiers') or [])}"
            )
            missing_qualifiers = sorted(value for value in expected_qualifiers if value not in actual_qualifiers)
            if missing_qualifiers:
                failures.append("independent_answer_missing_qualifiers")
            qa_results.append({
                "proposal": question, "independent_answer": verified, "comparison": comparison,
                "source_anchor": source_anchor, "independent_source_anchor": independent_anchor,
                "evidence_mode": evidence_mode,
                "missing_qualifiers": missing_qualifiers, "passed": not failures, "failures": failures,
            })
        page_results.append({
            "page": page, "image_uri": image_uri, "image_source": image_source,
            "image_sha256": hashlib.sha256(png).hexdigest(), "page_text_sha256": hashlib.sha256(page_text.encode()).hexdigest(),
            "page_text_chars": len(page_text), "readable": bool(readable_results) and all(readable_results),
            "raw_question_count": len(raw_questions),
            "filtered_question_count": len(filtered_questions),
            "filtered_questions": filtered_questions,
            "qa_required": bool(questions) or not filtered_questions,
            "auditor_usage": audit_usage, "answer_usage": answer_usage,
            "full_page_clip": full_page_clip, "audit_regions": audit_regions,
            "claim_adjudications": claim_adjudications,
            "claim_results": claim_results, "qa_results": qa_results,
        })
    all_claim_results = [item for page in page_results for item in page["claim_results"]]
    all_qa_results = [item for page in page_results for item in page["qa_results"]]
    missing_claim_pages = sorted(set(claim_pages) - {item["page"] for item in page_results})
    question_pages_without_verified_qa = [
        page["page"] for page in page_results
        if page["qa_required"] and not any(item["passed"] for item in page["qa_results"])
    ]
    failures: list[str] = []
    if metadata.get("metadata_pipeline_version") != METADATA_PIPELINE_VERSION:
        failures.append("metadata_pipeline_version_mismatch")
    if missing_claim_pages:
        failures.append("metadata_claim_pages_not_audited")
    if len(all_claim_results) != len(claims):
        failures.append("metadata_claim_coverage_mismatch")
    if any(not item["passed"] for item in all_claim_results):
        failures.append("metadata_claim_visual_failure")
    if sum(item["passed"] for item in all_qa_results) < min_verified_questions:
        failures.append("insufficient_independently_verified_questions")
    if question_pages_without_verified_qa:
        failures.append("selected_page_without_verified_question")
    if any(not page["readable"] for page in page_results):
        failures.append("unreadable_selected_page")
    return {
        "document_id": str(document["document_id"]), "version_id": str(document["version_id"]),
        "source_filename": document["source_filename"], "page_count": int(document["page_count"]),
        "metadata_model": document.get("metadata_model"), "metadata_pipeline_version": metadata.get("metadata_pipeline_version"),
        "audited_metadata_item_count": len(claims), "audited_claim_count": len(all_claim_results),
        "passed_claim_count": sum(item["passed"] for item in all_claim_results),
        "question_count": len(all_qa_results), "verified_question_count": sum(item["passed"] for item in all_qa_results),
        "selected_pages": selected_pages, "claim_pages": claim_pages, "missing_claim_pages": missing_claim_pages,
        "metadata_checks_passed": not any(value.startswith("metadata_") for value in failures),
        "qa_checks_passed": not any(value in {
            "insufficient_independently_verified_questions", "selected_page_without_verified_question"
        } for value in failures),
        "question_pages_without_verified_qa": question_pages_without_verified_qa,
        "checks_passed": not failures, "failures": failures, "pages": page_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-report", type=Path)
    parser.add_argument("--filename", action="append", dest="filenames")
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--max-question-pages", type=int, default=4)
    parser.add_argument("--max-questions-per-page", type=int, default=2)
    parser.add_argument("--question-generation-attempts", type=int, default=1)
    parser.add_argument("--min-verified-questions", type=int, default=2)
    parser.add_argument("--auditor-model", default="gemma4:latest")
    parser.add_argument("--answer-model", default="qwen3.5:9b")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite artifact: {args.output}")
    if args.auditor_model == args.answer_model:
        parser.error("auditor and independent answer models must differ")
    if (args.max_question_pages < 1 or args.max_questions_per_page < 1
            or args.question_generation_attempts < 1 or args.min_verified_questions < 1):
        parser.error("question limits must be positive")
    documents = _documents(args.filenames, args.document_limit)
    if args.metadata_report:
        documents = _merge_report(documents, _load_report(args.metadata_report))
    if not documents:
        parser.error("no documents selected")
    source_state = _source_state()
    started = datetime.now(UTC)
    results: list[dict[str, Any]] = []
    store = ObjectStore()
    with httpx.Client(base_url=settings.ollama_url, timeout=args.timeout_seconds) as client:
        for index, document in enumerate(documents, start=1):
            try:
                result = audit_document(
                    document, store=store, client=client, auditor_model=args.auditor_model,
                    answer_model=args.answer_model, max_question_pages=args.max_question_pages,
                    max_questions_per_page=args.max_questions_per_page,
                    question_generation_attempts=args.question_generation_attempts,
                    min_verified_questions=args.min_verified_questions, timeout_seconds=args.timeout_seconds,
                )
            except Exception as exc:
                result = {
                    "document_id": str(document["document_id"]), "version_id": str(document["version_id"]),
                    "source_filename": document["source_filename"], "checks_passed": False,
                    "failures": [f"runtime_error:{type(exc).__name__}:{exc}"], "pages": [],
                }
            results.append(result)
            print(json.dumps({
                "index": index, "total": len(documents), "source_filename": result["source_filename"],
                "checks_passed": result["checks_passed"], "failures": result["failures"],
                "verified_questions": result.get("verified_question_count", 0),
            }), flush=True)
    finished = datetime.now(UTC)
    artifact = {
        "audit_version": AUDIT_VERSION, "generated_at": finished.isoformat(),
        "started_at": started.isoformat(), **source_state,
        "metadata_report": str(args.metadata_report) if args.metadata_report else None,
        "metadata_report_sha256": hashlib.sha256(args.metadata_report.read_bytes()).hexdigest() if args.metadata_report else None,
        "auditor_model": args.auditor_model, "answer_model": args.answer_model,
        "configuration": {
            "max_question_pages": args.max_question_pages,
            "max_questions_per_page": args.max_questions_per_page,
            "question_generation_attempts": args.question_generation_attempts,
            "min_verified_questions": args.min_verified_questions,
            "timeout_seconds": args.timeout_seconds,
        },
        "complete": len(results) == len(documents), "checks_passed": bool(results) and all(item["checks_passed"] for item in results),
        "summary": {
            "documents": len(results), "documents_passed": sum(item["checks_passed"] for item in results),
            "claims_audited": sum(int(item.get("audited_claim_count") or 0) for item in results),
            "claims_passed": sum(int(item.get("passed_claim_count") or 0) for item in results),
            "questions": sum(int(item.get("question_count") or 0) for item in results),
            "questions_verified": sum(int(item.get("verified_question_count") or 0) for item in results),
            "elapsed_seconds": round((finished - started).total_seconds(), 3),
        },
        "documents": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks_passed": artifact["checks_passed"], **artifact["summary"]}, indent=2))
    raise SystemExit(0 if artifact["checks_passed"] else 1)


if __name__ == "__main__":
    main()
