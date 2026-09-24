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


def answer_relevant_expected_terms(query: str, terms: list[object]) -> list[str]:
    """Drop source-layout labels that the question does not ask the answer to repeat.

    Table serialization can introduce a generic ``Model:`` header even when the
    question already identifies a specific product and asks for a different
    value.  Requiring that header in the generated answer penalizes concise,
    correct answers.  Preserve it when ``model`` is itself part of the query.
    """

    normalized_query = _normalized(query)
    query_mentions_model = re.search(r"(?:^|\b)model(?:\b|$)", normalized_query) is not None
    asks_display_code_meaning = bool(
        re.search(r"\bdisplay\s+code\b", normalized_query)
        and re.search(r"\b(?:indicate|indicates|mean|means|meaning)\b", normalized_query)
    )
    single_count_answer = re.match(r"^how many\b", normalized_query) is not None
    kept_plain_count = False
    filtered: list[str] = []
    for term in terms:
        value = str(term).strip()
        if not value:
            continue
        if _normalized(value) == "model" and not query_mentions_model:
            continue
        if _normalized(value) == "display" and asks_display_code_meaning:
            continue
        if single_count_answer and re.fullmatch(r"\d+(?:\.\d+)?", _normalized(value)):
            if kept_plain_count:
                continue
            kept_plain_count = True
        if value not in filtered:
            filtered.append(value)
    return filtered


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
    deictic_subject = re.search(r"\b(?:this|these|those)\b", normalized_query) or re.search(
        r"\bthat(?:\s+[a-z0-9][a-z0-9-]*){0,4}\s+"
        r"(?:cameras?|sensors?|controllers?|devices?|products?|units?|models?|"
        r"systems?|manuals?|series|filters?|components?|accessories|cables?|connectors?)\b",
        normalized_query,
    )
    if deictic_subject:
        # Frozen evaluation questions are executed without conversational
        # context. Any demonstrative reference therefore cannot establish
        # which source model, component, or prior statement applies.
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
    # The LR-T manual exposes separate response-time settings for the laser
    # sensor and for an attached MU-N main/expansion controller.  A question
    # naming only the sensor cannot identify which table is authoritative.
    # Require the controller scope whenever the expected row is explicitly
    # inside the MU-N unit table.
    normalized_context = _normalized(source_context)
    if (
        re.search(r"\bresponse times?\b", normalized_query)
        and re.search(r"\bmu[- ]?n(?:11|12)?\b", normalized_context)
        and re.search(r"\bmain unit\b|\bexpansion unit\b", normalized_context)
        and not re.search(
            r"\bmu[- ]?n(?:11|12)?\b|\bcontroller\b|\bmain unit\b|\bexpansion unit\b",
            normalized_query,
        )
    ):
        missing.append("MU-N controller")
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


_ANSWER_VALUE_INTENTS = {
    "accuracy", "current", "distance", "frequency", "height", "length",
    "limit", "range", "rating", "resolution", "speed", "temperature",
    "time", "tolerance", "torque", "voltage", "weight", "width",
}

_SEMANTIC_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "can", "could", "did", "do", "does",
    "for", "how", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "which", "with", "would",
}


def _contract_tokens(text: object) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*", _normalized(text))


def _term_covers_token(terms: list[str], token: str) -> bool:
    normalized_token = _normalized(token)
    compact_token = re.sub(r"[^a-z0-9]+", "", normalized_token)
    for term in terms:
        normalized_term = _normalized(term)
        compact_term = re.sub(r"[^a-z0-9]+", "", normalized_term)
        if normalized_token.replace(".", "", 1).isdigit():
            if normalized_token in re.findall(r"\d+(?:\.\d+)?", normalized_term):
                return True
            continue
        if normalized_token in normalized_term or (compact_token and compact_token in compact_term):
            return True
    return False


_QUANTITY_UNIT = (
    r"%|vdc|vac|v|ma|a|kw|w|mm|cm|m|msec|ms|sec|s|hz|khz|mhz|ghz|fps|"
    r"kg|g|n|nm|mpa|deg|gb|tb|bits?|pixels?|\u00b0c|in(?:ch(?:es)?)?|\""
)


def _answer_quantity_values(query: str, snippet: str) -> list[str]:
    normalized_query = _normalized(query)
    values: list[str] = []

    how_many = re.match(
        r"^how many (.+?) (?:are|can|could|does|do|fit|may|should|will)\b",
        normalized_query,
    )
    if how_many:
        target_tokens = {
            token
            for token in _contract_tokens(how_many.group(1))
            if token not in _SEMANTIC_QUERY_STOPWORDS and len(token) >= 2
        }
        for match in re.finditer(r"(?<![*a-z0-9-])(\d+(?:\.\d+)?)\b", snippet, flags=re.IGNORECASE):
            window = set(_contract_tokens(snippet[max(0, match.start() - 30) : match.end() + 70]))
            if target_tokens.intersection(window):
                value = match.group(1)
                if value not in values:
                    values.append(value)
        # A single-target count question has one answer. Structured snippets
        # may serialize sibling model columns after the requested cell; those
        # values are context, not additional required answer values.
        return values[:1]

    if re.search(r"\brange\b", normalized_query):
        for match in re.finditer(
            r"(?<![*a-z0-9-])(\d+(?:\.\d+)?)\s*(?:to|[-\u2013])\s*(\d+(?:\.\d+)?)(?![a-z0-9])",
            snippet,
            flags=re.IGNORECASE,
        ):
            for value in match.groups():
                if value not in values:
                    values.append(value)

    for match in re.finditer(
        rf"(?<![*a-z0-9-])([+\-\u00b1]?\s*\d+(?:\.\d+)?)\s*(?:to|[-\u2013])\s*"
        rf"([+\-\u00b1]?\s*\d+(?:\.\d+)?)\s*(?:{_QUANTITY_UNIT})(?![a-z])",
        snippet,
        flags=re.IGNORECASE,
    ):
        for raw in match.groups():
            value_match = re.search(r"\d+(?:\.\d+)?", raw)
            if value_match and value_match.group(0) not in values:
                values.append(value_match.group(0))

    for match in re.finditer(
        rf"(?<![*a-z0-9-])(?:[+\-\u00b1]\s*)?(\d+(?:\.\d+)?)\s*(?:{_QUANTITY_UNIT})(?![a-z])",
        snippet,
        flags=re.IGNORECASE,
    ):
        value = match.group(1)
        if value not in values:
            values.append(value)
    return values


def enrich_expected_answer_terms(query: str, snippet: str, terms: list[object]) -> list[str]:
    """Add mechanically identifiable answer values to re-anchored terms."""

    enriched = [str(term).strip() for term in terms if str(term).strip()]
    normalized_query = _normalized(query)
    query_tokens = set(_contract_tokens(normalized_query))

    for value in _answer_quantity_values(query, snippet):
        if value not in query_tokens and not _term_covers_token(enriched, value):
            enriched.append(value)

    if re.search(r"\bconnector type\b|\bwhat (?:type of )?connector\b", normalized_query):
        for token in _contract_tokens(snippet):
            if (
                any(char.isalpha() for char in token)
                and any(char.isdigit() for char in token)
                and token not in query_tokens
                and not _term_covers_token(enriched, token)
            ):
                enriched.append(token)

    if re.search(r"\bwhich command\b|\bwhat command\b", normalized_query):
        for command in re.findall(r"\b[a-z][a-z0-9_-]*command\b", _normalized(snippet)):
            if command != "command" and not _term_covers_token(enriched, command):
                enriched.append(command)

    if re.search(r"\bwhich interfaces?\b|\bwhat interfaces?\b", normalized_query):
        interface_pattern = r"\b(?:usb|ethernet(?:/ip)?|profinet|udp|tcp(?:/ip)?|rs-?232c?|rs-?485|cc-link|profisafe|cip safety)\b"
        for interface in re.findall(interface_pattern, _normalized(snippet)):
            if not _term_covers_token(enriched, interface):
                enriched.append(interface)

    return enriched


def missing_expected_answer_contract(
    query: str,
    expected_snippet: str,
    expected_terms: list[object],
) -> list[str]:
    """Return semantic answer-contract defects not caught by literal anchoring.

    A literal source substring can still be the wrong row, omit the requested
    value, or carry expected terms copied entirely from the question. These
    checks cover only high-confidence question shapes for which the
    answer-bearing token is mechanically identifiable.
    """

    normalized_query = _normalized(query)
    normalized_snippet = _normalized(expected_snippet)
    terms = [str(term).strip() for term in expected_terms if str(term).strip()]
    missing: list[str] = []
    query_tokens = set(_contract_tokens(normalized_query))

    if query_tokens and terms and not any(
        not set(_contract_tokens(term)).issubset(query_tokens)
        for term in terms
        if _contract_tokens(term)
    ):
        missing.append("answer-specific expected term")

    asks_value = bool(
        re.match(r"^what(?!\s+(?:do|does|did)\b)", normalized_query)
        and not re.match(r"^what safety risks?\b", normalized_query)
        and any(
            re.search(rf"\b{re.escape(intent)}\b", normalized_query)
            for intent in _ANSWER_VALUE_INTENTS
        )
    ) or bool(re.search(r"^how (?:many|much|long|wide|high|fast|far)\b", normalized_query))
    quantities = [
        value
        for value in _answer_quantity_values(query, expected_snippet)
        if value not in query_tokens
    ]
    if asks_value:
        if not quantities:
            missing.append("quantified answer value")
        else:
            absent_values = [value for value in quantities if not _term_covers_token(terms, value)]
            if absent_values:
                missing.append("expected answer value term(s) " + ", ".join(absent_values))

    if re.search(r"\bconnector type\b|\bwhat (?:type of )?connector\b", normalized_query):
        identifiers = [
            token
            for token in _contract_tokens(expected_snippet)
            if any(char.isalpha() for char in token)
            and any(char.isdigit() for char in token)
            and token not in query_tokens
        ]
        if not identifiers:
            missing.append("connector identifier")
        else:
            absent = [token for token in identifiers if not _term_covers_token(terms, token)]
            if absent:
                missing.append("expected connector term(s) " + ", ".join(absent))

    if re.search(r"\bwhich command\b|\bwhat command\b", normalized_query):
        commands = re.findall(r"\b[a-z][a-z0-9_-]*command\b", normalized_snippet)
        commands = [command for command in commands if command != "command"]
        if not commands:
            missing.append("command identifier")
        else:
            absent = [command for command in commands if not _term_covers_token(terms, command)]
            if absent:
                missing.append("expected command term(s) " + ", ".join(absent))

    if re.search(r"\bwhich interfaces?\b|\bwhat interfaces?\b", normalized_query):
        interface_pattern = r"\b(?:usb|ethernet(?:/ip)?|profinet|udp|tcp(?:/ip)?|rs-?232c?|rs-?485|cc-link|profisafe|cip safety)\b"
        interfaces = re.findall(interface_pattern, normalized_snippet)
        if not interfaces:
            missing.append("interface identifier")
        elif not any(_term_covers_token(terms, interface) for interface in interfaces):
            missing.append("expected interface term")

    if re.match(r"^(?:does|do|did|can|could|is|are|will|would|should|has|have)\b", normalized_query):
        meaningful_query_terms = {
            token
            for token in _contract_tokens(normalized_query)
            if token not in _SEMANTIC_QUERY_STOPWORDS and len(token) >= 3
        }
        snippet_tokens = set(_contract_tokens(normalized_snippet))
        overlap = meaningful_query_terms.intersection(snippet_tokens)
        if len(overlap) < 2:
            missing.append("question subject/outcome alignment")

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
            terms = enrich_expected_answer_terms(
                str(case.get("query") or ""),
                snippet_text,
                extract_anchor_terms(snippet_text)[:4],
            )
            if not snippet_text or not terms:
                raise ValueError(f"{case_id}: source re-anchoring produced no usable evidence")
            case["expected_snippet"] = snippet_text
            case["expected_terms"] = terms
            case["anchor_terms"] = terms
        evidence_hashes: dict[str, str] = {}
        expected_evidence = case.get("expected_evidence") or []
        if not expected_evidence:
            original_expected_terms = list(case.get("expected_terms") or [])
            filtered_expected_terms = answer_relevant_expected_terms(
                str(case.get("query") or ""),
                original_expected_terms,
            )
            if original_expected_terms and not filtered_expected_terms:
                raise ValueError(f"{case_id}: no answer-relevant expected terms remain")
            case["expected_terms"] = filtered_expected_terms
            if "anchor_terms" in case:
                case["anchor_terms"] = answer_relevant_expected_terms(
                    str(case.get("query") or ""),
                    list(case.get("anchor_terms") or []),
                )
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
            missing_requirements.extend(
                missing_expected_answer_contract(
                    str(case.get("query") or ""),
                    str(case.get("expected_snippet") or ""),
                    list(case.get("expected_terms") or []),
                )
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
                    "method": "assistant_persisted_chunk_answer_contract_v2",
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
        "adjudication_method": "assistant_persisted_chunk_answer_contract_v2",
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
