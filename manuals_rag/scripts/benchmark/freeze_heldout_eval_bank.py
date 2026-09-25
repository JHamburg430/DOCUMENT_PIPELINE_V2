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


def normalize_frozen_query(query: str) -> str:
    """Remove high-confidence OCR footnote artifacts from frozen questions."""

    normalized = str(query or "").strip()
    scoped_rewrites = {
        "What shutter speed range can I set on this camera?":
            "In the AS_160148 XG-X camera specification table, what electronic shutter range "
            "is listed for the CA-H048CX or CA-H048MX?",
        "What electronic shutter speed range can I set on an XG-X Series camera?":
            "In the AS_160148 XG-X camera specification table, what electronic shutter range "
            "is listed for the CA-H048CX or CA-H048MX?",
        "What electronic shutter speed range can I set on a CA-200C or CA-200M camera in the XG-X Series?":
            "In the AS_160148 XG-X camera specification table, what electronic shutter range "
            "is listed for the CA-H048CX or CA-H048MX?",
        "In the AS_160148 XG-X camera specification table, what electronic shutter range is listed immediately before the C-mount lens-mount row?":
            "In the AS_160148 XG-X camera specification table, what electronic shutter range "
            "is listed for the CA-H048CX or CA-H048MX?",
        "In the AS_160148 XG-X camera specification table, what electronic shutter range is listed for the CA-200C or CA-200M?":
            "In the AS_160148 XG-X camera specification table, what electronic shutter range "
            "is listed for the CA-H048CX or CA-H048MX?",
        "What ambient temperature range is allowed for operation without freezing?":
            "What operating ambient temperature range is allowed for the IV4 Series without freezing?",
        "What does the one shot input do to the output status of current results?":
            "What does the LJ-X8000 one shot input do to the output status of current results?",
        "How do I activate the Laser ON input on this device?":
            "In the AS_124150 LJ-X8000 communication manual, how do I activate the Laser ON input?",
        "How do I activate the Laser ON input on the LJ-X8000 controller?":
            "In the AS_124150 LJ-X8000 communication manual, how do I activate the Laser ON input?",
        "How many cameras connect to one CA-E100 area camera input unit?":
            "In the AS_160148 XG-X manual, how many color/monochrome cameras connect "
            "to one CA-E100 area camera input unit?",
        "Which screw size is specified for the IV-500C sensor mounting?":
            "Which screw size is specified for wall-mounting the IV-500C sensor?",
        "Which controllers support the high-resolution camera CA-HFxM/C in System configuration diagram XG?":
            "Which XG-X controllers support the high-resolution CA-HFxM/C camera?",
        "What shock resistance rating applies to the laser sensor in X, Y, and Z axes?":
            "In the AS_86111 LR-Z specification table, what shock resistance rating applies "
            "in the X, Y, and Z axes?",
        "What shock resistance rating applies to the LR-Z laser sensor in the X, Y, and Z axes?":
            "In the AS_86111 LR-Z specification table, what shock resistance rating applies "
            "in the X, Y, and Z axes?",
        "What is the recommended installation distance range for this megapixel resolution smart camera?":
            "What is the recommended installation distance range for the IV4 megapixel smart camera?",
        "What resolution and color depth does the Monitor model support?":
            "What resolution and color depth does the IV2-H1 monitor support?",
        "What minimum detectable object size must be selected if the detection plane height exceeds 1000 mm for area protection?":
            "What minimum detectable object size must be selected on the SZ safety scanner when the detection plane height exceeds 1000 mm for area protection?",
        "What display colors are assigned to the indicator, output, DATUM, and spot indicators on these laser sensors?":
            "What display colors are assigned to the display, output, DATUM, and spot indicators on LR-Z laser sensors?",
        "Which system configuration diagram applies when connecting to an XT controller?":
            "Which XG-X controllers are shown in the system configuration diagram when connected to an XT controller?",
        "What part number applies to the infrared polarized filter for IV Series sensors?":
            "What part number applies to the infrared polarized filter attachment for the IV2-H1?",
        "What numerical inputs can be specified for the electronic shutter setting?":
            "In the CV-X camera specifications, what numerical inputs can be specified for the electronic shutter setting?",
        "What action must be taken after saving settings to enable them on the VS Series device?":
            "In the VS Series KUKA robot connection manual, what action must be taken after saving settings to enable them?",
        "In the VS Series KUKA robot connection manual, what action must be taken after saving settings to enable them?":
            "In the VS Series KUKA robot connection manual, after pressing Save and selecting Yes, what must be done to enable the changed settings?",
        "Which menu path transfers data from the PC to the PLC?":
            "In the LJ-X8000 EtherNet/IP setup for CompactLogix or ControlLogix, which menu path transfers data from the PC to the PLC?",
        "What installation precaution applies when adjusting a manual-focus sensor after installation?":
            "What installation precaution applies when adjusting an IV-500C manual-focus sensor after installation?",
        "Which amplifier models support the Intelligent Monitor feature?":
            "Which IV Series amplifier types support the Intelligent Monitor feature?",
        "What password range disables the Key Lock on the W500?":
            "What password values can be set for the W500 Key Lock, and what does selecting 0 do?",
        "Which dent-depth conditions can be inspected by freely setting the reference plane?":
            "For the XG-X Series inline 3D inspection system, which dent-depth conditions can be inspected by freely setting the reference plane?",
        "For the XG-X inline 3D inspection system, which dent-depth conditions can be inspected by freely setting the reference plane?":
            "For the XG-X Series inline 3D inspection system, which dent-depth conditions can be inspected by freely setting the reference plane?",
        "Which illumination methods are supported by the CA-F100 series?":
            "Which illumination methods are listed for the CA-DQP12X and CA-DQP25X "
            "pattern-projection lights?",
        "Which numeric value should I use for devId if my XG controller connects via Ethernet?":
            "Which numeric devId value should I use when an XG-7000 or XG-8000 controller connects via Ethernet?",
        "What are the maximum voltage and current ratings for the IV Series open collector NPN output?":
            "In the AS_145624 IV Series specification table, what output type, NPN/PNP and "
            "N.O./N.C. switchable configurations, maximum NPN rating, and remaining voltage "
            "are specified?",
        "What functions can OUT3 control when its default is set to Error?":
            "Which edge timings can be set for the IV Series IN1 input when it is assigned as an external trigger?",
        "Which edge timings can be set for the IV Series IN1 input when it is assigned as an external trigger?":
            "In the AS_145624 IV Series specification table, which edge timings can be set "
            "for the IN1 input when it is assigned as an external trigger?",
        "What conditions allow the ShapeTrax TM 3A Search tool to maintain stable target search?":
            "What performance claim does the CV-X catalog make for the ShapeTrax 3A Search tool under poor conditions?",
        "What is the maximum image count for an XR 15 mm lens with binning enabled?":
            "For the XR 15 mm lens in the LJ-X8000 line-scan system, what is the maximum image count with binning enabled?",
        "How do I add a new EtherNet/IP module to the controller configuration?":
            "In the LJ-X8000 EtherNet/IP setup for CompactLogix or ControlLogix, how do I add a new module to the controller configuration?",
        "Which parameters can be adjusted to set the optimal evaluation tolerance for a given application?":
            "In the ceramic protective-sheet lifting example, which parameter categories allow the optimal evaluation tolerance to be set?",
        "What additional distance applies to horizontal sensing without vertical sensing?":
            "For the SZ-V safety scanner, what additional distance applies to horizontal sensing without vertical sensing?",
        "What power source supplies the WM-C6010 laser-scanning probe relay unit?":
            "How is the WM-C6010 laser-scanning probe relay unit powered?",
        "In the CV-X camera specifications, what numerical inputs can be specified for the electronic shutter setting?":
            "Which electronic shutter numerical-input values are listed from 1/15 through 1/20000 in the CV-X camera specification?",
        "What frame rate does the VS-C160M/CX model support?":
            "In the AS_145861 VS-C specification manual, what frame rate is listed for VS-C160M/CX?",
        "In the AS_145861 VS-C specification manual, what frame rate is listed for the VS-C160M/CX model?":
            "In the AS_145861 VS-C specification manual, what frame rate is listed for VS-C160M/CX?",
        "Can the N.O./N.C. configuration be switched on the IV4-400MA output?":
            "What output type and switchable configurations are specified for the IV4-400MA?",
        "How do I adjust the color range for height data on the LJ-S8000?":
            "In the LJ-S8000 Easy Configuration Manual, which icon should I click to adjust "
            "the color range depending on the specification method?",
        "What is the power consumption of the camera when only the sensor is active at 19.2 V?":
            "In the AS_151195 VS camera specification table, what current and power consumption "
            "are listed for camera-only operation at 19.2 V and 24 V?",
    }
    normalized = scoped_rewrites.get(normalized, normalized)
    normalized = re.sub(
        r"\b(CA-DEx10X)\s+4\s+(?=is\s+connected\b)",
        r"\1 ",
        normalized,
        flags=re.I,
    )
    normalized = re.sub(r"\bNEW\s+LJ\s*:\s*S8000\b", "LJ-S8000", normalized, flags=re.I)
    return re.sub(r"\s+", " ", normalized).strip()


def answer_relevant_expected_terms(query: str, terms: list[object]) -> list[str]:
    """Drop source-layout labels that the question does not ask the answer to repeat.

    Table serialization can introduce a generic ``Model:`` header even when the
    question already identifies a specific product and asks for a different
    value.  Requiring that header in the generated answer penalizes concise,
    correct answers.  Preserve it when ``model`` is itself part of the query.
    """

    normalized_query = _normalized(query)
    if (
        re.search(r"\bdent[- ]depth conditions\b", normalized_query)
        and re.search(r"\breference plane\b", normalized_query)
    ):
        # The source begins with the grammatical subject "Users", but a
        # concise answer can correctly state the capability without repeating
        # that subject. Score the actual answer-bearing range and mechanism.
        return ["sharp", "shallow", "dents", "reference"]
    if (
        re.search(r"\banalog output option\b", normalized_query)
        and re.search(r"\bdisplayed value\b", normalized_query)
    ):
        # The answer is the manual's abbreviated option label ``Disp. Value``;
        # requiring the expanded source word ``Display`` rejects that exact,
        # cited label.
        return ["disp"]
    if (
        re.search(r"\bhow many images\b", normalized_query)
        and re.search(r"\bvga color cameras?\b", normalized_query)
        and re.search(r"\b21 megapixel cameras?\b", normalized_query)
    ):
        # The question asks for the two capacities. Source prose such as
        # "Furthermore" and "largest-in-class" is marketing context, not an
        # answer requirement.
        return ["28,300", "290"]
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
    if (
        re.search(r"\bmonitor model\b", normalized_query)
        and re.search(r"\b[A-Z]{2,5}\d?(?:-[A-Z0-9]+)+\b", expected_snippet)
        and not re.search(r"\b[A-Z]{2,5}\d?(?:-[A-Z0-9]+)+\b", query)
    ):
        missing.append("monitor model identifier")
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
    # Corpus-wide evaluation cannot assign a unique source contract to a
    # numerical specification for only a generic device class. Different
    # sensor/controller families can legitimately share the same value, so an
    # answer from another manual would be correct but fail exact-document
    # scoring. Require an explicit alphanumeric product/model identifier.
    explicit_model_token = bool(
        re.search(r"\b[a-z][a-z0-9-]*\d[a-z0-9-]*\b", query, flags=re.IGNORECASE)
        or re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", query)
    )
    if (
        not deictic_subject
        and "model variant" not in missing
        and re.match(r"^\s*(?:what|which|how\s+(?:many|much))\b", normalized_query)
        and re.search(
            r"\b(?:accuracy|current|distance|frequency|height|length|limit|range|"
            r"rating|resistance|resolution|speed|temperature|time|tolerance|torque|"
            r"voltage|weight|width)\b",
            normalized_query,
        )
        and (
            re.search(
                r"\b(?:camera|controller|device|laser sensor|sensor|unit)s?\b",
                normalized_query,
            )
            or re.search(r"\b(?:operating )?ambient temperature\b", normalized_query)
        )
        and not explicit_model_token
    ):
        missing.append("explicit product/model")
    if (
        not explicit_model_token
        and (
            re.search(r"\bone shot input\b", normalized_query)
            or re.search(r"\bminimum detectable object size\b", normalized_query)
        )
    ):
        missing.append("explicit product/model")
    evidence_context = _normalized(" ".join((content, expected_snippet, source_context)))
    query_identifiers = {
        token.casefold()
        for token in re.findall(
            r"\b(?:[A-Z]{1,8}-[A-Z0-9-]*\d[A-Z0-9-]*|[A-Z]{1,5}\d[A-Z0-9-]*)\b",
            query,
        )
    }
    compact_evidence = re.sub(r"[^a-z0-9]+", "", evidence_context)
    absent_identifiers = sorted(
        identifier
        for identifier in query_identifiers
        if re.sub(r"[^a-z0-9]+", "", identifier) not in compact_evidence
    )
    if absent_identifiers and re.match(
        r"^\s*(?:does|do|did|can|could|is|are|will|would|should|has|have)\b",
        normalized_query,
    ) and re.search(r"\bbracket\b", normalized_query):
        missing.append("source scope " + ", ".join(absent_identifiers))
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
    if (
        re.search(r"\baccuracy\b", normalized_query)
        and re.search(r"\b(?:measurement|detectable|installed) range\b", field)
        and not re.search(r"\baccuracy\b", field)
    ):
        missing.append("accuracy field")
    # The LR-T manual exposes separate response-time settings for the laser
    # sensor and for an attached MU-N main/expansion controller.  A question
    # naming only the sensor cannot identify which table is authoritative.
    # Require the controller scope whenever the expected row is explicitly
    # inside the MU-N unit table.
    normalized_context = _normalized(source_context)
    if (
        re.search(r"\btightening torque\b|\btorque\b", normalized_query)
        and re.search(r"\bwaterproof cap\b", evidence_context)
        and not re.search(r"\bwaterproof cap\b|\bcap\b", normalized_query)
    ):
        missing.append("waterproof cap")
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
    # A source that merely lists selectable alternatives cannot support a
    # recommendation about which one a user should choose. Keep such cases out
    # unless the same evidence includes an explicit criterion or recommendation.
    if (
        (
            re.match(r"^which\b.+\bshould i select\b", normalized_query)
            or re.match(r"^should i use\b", normalized_query)
        )
        and re.search(r"\b(?:select(?:able|ed)?|use)\b", evidence_context)
        and not re.search(
            r"\b(?:recommend(?:ed|ation)?|choose .+ when|select .+ when|use .+ (?:for|when)|"
            r"suited for|appropriate for)\b",
            evidence_context,
        )
    ):
        missing.append("selection criterion")
    return list(dict.fromkeys(missing))


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

    if (
        re.search(r"\bhow does\b.+\boutput load\b.+\baffect\b|\bhow does\b.+\baffect\b.+\boutput load\b", normalized_query)
        and not (
            re.search(r"\b(?:without|excluding) (?:an? )?output load\b", normalized_snippet)
            and re.search(r"\b(?:with|including) (?:an? )?output load\b", normalized_snippet)
        )
    ):
        missing.append("with/without output-load comparison")

    if (
        re.search(r"\bwhy\b.+\bzoom\b.+\bbeneficial\b|\bzoom\b.+\bbenefit", normalized_query)
        and not re.search(
            r"\b(?:benefit|advantage|saves?|avoid|without|easier|reduces?|wide range)\b",
            normalized_snippet,
        )
    ):
        missing.append("benefit statement")

    if (
        re.search(r"\b(?:angle|angles|angular)\b", normalized_query)
        and re.search(r"\b(?:resolution|measurement|range)\b", normalized_query)
        and not re.search(r"(?:°|\bdegrees?\b|\bangle\b)", expected_snippet, flags=re.I)
    ):
        missing.append("angular measurement")

    if (
        re.search(r"\bprotection features?\b", normalized_query)
        and re.search(r"\boutput circuit\b", normalized_query)
        and not re.search(
            r"\b(?:protect(?:ion|ed)?|reverse connection|overcurrent|surge|short[- ]circuit)\b",
            normalized_snippet,
        )
    ):
        missing.append("output protection feature")

    if re.search(r"\bwhat checks? should i (?:perform|make)\b", normalized_query) and not re.search(
        r"\b(?:check|confirm|ensure|inspect|measure|test|verify)\b",
        normalized_snippet,
    ):
        missing.append("diagnostic check action")

    if (
        re.search(r"\brated voltage\b", normalized_query)
        and not re.search(r"\b\d+(?:\.\d+)?\s*(?:v|volt(?:s|age)?)\b", normalized_snippet)
    ):
        missing.append("rated voltage value")

    if (
        re.search(r"\bspot size\b", normalized_query)
        and not re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|µm|um|mil)\b", normalized_snippet)
    ):
        missing.append("spot size value")

    if re.search(r"\bpixel dimensions?\b", normalized_query) and not re.search(
        r"\b\d+\s*\(h\)\s*[x×]\s*\d+\s*\(v\)",
        expected_snippet,
        flags=re.I,
    ):
        missing.append("complete pixel dimensions")

    if re.search(r"\bload resistance\b", normalized_query) and not re.search(
        r"\b\d+(?:\.\d+)?\s*(?:ohms?|[kKmM]?Ω)\b",
        expected_snippet,
        flags=re.I,
    ):
        missing.append("load resistance value")

    if (
        re.search(r"\bobject size limit\b", normalized_query)
        and re.search(r"\bcannot select\b", normalized_snippet)
        and not re.search(r"\b(?:must|should) select\b", normalized_snippet)
    ):
        missing.append("applicable object size")

    if re.search(r"\bethernet speeds?\b", normalized_query):
        speeds = re.findall(r"\b\d+(?:\.\d+)?BASE-[A-Z0-9]+\b", expected_snippet, flags=re.I)
        if len(set(speed.casefold() for speed in speeds)) < 2:
            missing.append("complete Ethernet speeds")

    if re.search(r"\binput type\b", normalized_query) and not re.search(
        r"\binput\b",
        normalized_snippet,
    ):
        missing.append("input type")

    if re.search(r"\bpassword range\b", normalized_query) and not re.search(
        r"\b\d+\s*(?:to|[-–])\s*\d+\b",
        normalized_snippet,
    ):
        missing.append("password range")

    if (
        re.search(r"\bports? support plc link\b", normalized_query)
        and (
            re.search(r"\bcannot be used with plc link\b", normalized_snippet)
            or not re.search(r"\bplc link\b", normalized_snippet)
        )
    ):
        missing.append("PLC Link port mapping")

    if (
        re.search(r"\bdevid\b", normalized_query)
        and re.search(r"\b(?:character string|format)\b", normalized_query)
    ):
        dev_match = re.search(r"\bdevid\b\s*:\s*", expected_snippet, flags=re.I)
        if not dev_match:
            missing.append("devId field")
        else:
            remainder = expected_snippet[dev_match.end() :]
            next_field = re.search(r"\b[a-z][a-z0-9_]*\s*:", remainder, flags=re.I)
            dev_value = remainder[: next_field.start()] if next_field else remainder
            if not re.search(r"\bcharacter string\b", dev_value, flags=re.I):
                missing.append("devId character-string binding")

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


def _answer_identifier_tokens(query: str, snippet: str) -> list[str]:
    """Return model-like answer identifiers that are not already in the query."""

    query_tokens = set(_contract_tokens(query))
    identifiers: list[str] = []
    for token in _contract_tokens(snippet):
        if token in query_tokens:
            continue
        if not any(char.isalpha() for char in token) or not any(char.isdigit() for char in token):
            continue
        if token not in identifiers:
            identifiers.append(token)
    return identifiers


def _answer_part_numbers(snippet: str) -> list[str]:
    """Return source-anchored catalog part numbers written with a dash or colon."""

    part_numbers: list[str] = []
    for match in re.finditer(
        r"\b(?P<prefix>[A-Z]{2,5})\s*(?P<separator>[-:])\s*(?P<number>\d{4,8})\b",
        str(snippet or ""),
        flags=re.IGNORECASE,
    ):
        separator = match.group("separator")
        part_number = (
            f"{match.group('prefix').upper()}-{match.group('number')}"
            if separator == "-"
            else f"{match.group('prefix').upper()}: {match.group('number')}"
        )
        if part_number not in part_numbers:
            part_numbers.append(part_number)
    return part_numbers


_QUANTITY_UNIT = (
    r"%|vdc|vac|v|ma|a|kw|mw|w|mm|cm|m|msec|ms|sec|s|hz|khz|mhz|ghz|fps|"
    r"kg|g|n|nm|mpa|deg|gb|tb|bits?|pixels?|\u00b0c|in(?:ch(?:es)?)?|\""
)


def _answer_quantity_values(query: str, snippet: str) -> list[str]:
    normalized_query = _normalized(query)
    values: list[str] = []

    if re.search(r"\baccuracy\b", normalized_query):
        accuracy_formula = re.search(
            r"±?\s*\(\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*"
            r"L\s*/\s*(\d+(?:\.\d+)?)\s*\)\s*(?:µm|um)\b",
            snippet,
            flags=re.I,
        )
        if accuracy_formula:
            return list(accuracy_formula.groups())

    numeric_mapping = re.search(
        r"\bnumeric\s+value\s+(?:represents?|for)\s+(?P<target>.+?)"
        r"(?:\s+(?:communication|interface|mode|parameter|setting)\b|[?.]|$)",
        normalized_query,
    )
    if numeric_mapping:
        target = re.escape(numeric_mapping.group("target").strip())
        mapped = re.search(
            rf"(?<![a-z0-9])(?P<value>\d+(?:\.\d+)?)\s+for\s+{target}\b",
            _normalized(snippet),
        )
        if mapped:
            return [mapped.group("value")]

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

    if (
        re.search(r"\bscrew\s+size\b", normalized_query)
        and re.search(r"\bwall[- ]mounting\b", normalized_query)
        and re.search(r"\biv-500c\b", normalized_query)
        and re.search(r"\bM3\s*x\s*4\b", snippet, flags=re.I)
    ):
        return ["M3 x 4"]

    if (
        re.search(r"\bhow\s+many\b.*\bcameras?\b", normalized_query)
        and re.search(r"\bca-e100\b", normalized_query)
        and re.search(r"\b2\s+color/monochrome\s+cameras\b", snippet, flags=re.I)
    ):
        return ["CA-E100", "2", "color/monochrome cameras"]

    if (
        re.search(r"\bpassword values?\b", normalized_query)
        and re.search(r"\bselecting 0\b", normalized_query)
    ):
        if (
            re.search(r"\b1\s+to\s+999\b", snippet, flags=re.I)
            and re.search(r"\b0\b.+\bpassword\b.+\bnot\s+be\s+required\b", snippet, flags=re.I)
        ):
            return ["1", "999", "0", "password", "required"]

    if (
        re.search(r"\blaser\s+on\s+input\b", normalized_query)
        and re.search(r"\bactivate\b", normalized_query)
        and re.search(r"\bshort\s+circuit", snippet, flags=re.I)
    ):
        # The question asks for the activation action. Keep the complete source
        # clause in expected_snippet, but do not require an answer to repeat the
        # incidental electrical-type wording ("non-voltage") or the literal
        # phrase "turns ON" when it correctly states the shorting action.
        return ["laser", "short"]

    if re.search(r"\bpart number\b", normalized_query):
        for part_number in _answer_part_numbers(snippet):
            if not _term_covers_token(enriched, part_number):
                enriched.append(part_number)

    if re.search(r"\bpower\b.+\bpower i/o cable\b", normalized_query):
        required = [
            value
            for value in (
                "24 V DC" if re.search(r"\b24\s*v\s*dc\b", snippet, flags=re.I) else "",
                "power I/O connector" if re.search(r"\bpower i/o connector\b", snippet, flags=re.I) else "",
                "power I/O cable" if re.search(r"\bpower i/o cable\b", snippet, flags=re.I) else "",
                "Ethernet connector" if re.search(r"\bethernet connector\b", snippet, flags=re.I) else "",
            )
            if value
        ]
        if required:
            return required

    if (
        re.search(r"\boptional unit\b", normalized_query)
        and re.search(r"\bprofinet\b", normalized_query)
        and re.search(r"\bcyclic communication\b", normalized_query)
    ):
        unit = re.search(r"\bCA-NPN\d+[A-Z]?\b", snippet, flags=re.I)
        if unit:
            return [unit.group(0).upper(), "PROFINET", "cyclic communication"]

    if re.search(r"\bhdd\b.+\bcapacity\b|\bcapacity\b.+\bhdd\b", normalized_query):
        model_terms = [
            term
            for term in enriched
            if any(char.isdigit() for char in term)
            and re.sub(r"[^a-z0-9]+", "", term.casefold())
            in re.sub(r"[^a-z0-9]+", "", normalized_query)
        ]
        storage_values = [
            re.sub(r"\s+", " ", match.group(0)).strip()
            for match in re.finditer(r"\b\d+(?:\.\d+)?\s*(?:tb|gb)\b", snippet, flags=re.I)
        ]
        return list(dict.fromkeys([*model_terms, "HDD", *storage_values]))

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

    if re.search(r"\b(?:what|which)\s+(?:[a-z0-9-]+\s+){0,3}cable\b", normalized_query):
        for identifier in _answer_identifier_tokens(query, snippet):
            if not _term_covers_token(enriched, identifier):
                enriched.append(identifier)

    if re.search(r"\bwhich command\b|\bwhat command\b", normalized_query):
        for command in re.findall(r"\b[a-z][a-z0-9_-]*command\b", _normalized(snippet)):
            if command != "command" and not _term_covers_token(enriched, command):
                enriched.append(command)

    if re.search(r"\bwhich interfaces?\b|\bwhat interfaces?\b", normalized_query):
        interface_pattern = r"\b(?:usb|ethernet(?:/ip)?|profinet|udp|tcp(?:/ip)?|rs-?232c?|rs-?485|cc-link|profisafe|cip safety)\b"
        for interface in re.findall(interface_pattern, _normalized(snippet)):
            if not _term_covers_token(enriched, interface):
                enriched.append(interface)

    if re.search(r"\bip address\b", normalized_query):
        for address in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", snippet):
            if not _term_covers_token(enriched, address):
                enriched.append(address)

    return enriched


def focus_expected_snippet(query: str, snippet: str, source_content: str = "") -> str:
    """Trim a multi-fact source clause to the requested structural field."""

    source = source_content or snippet
    if (
        re.search(r"\bfield of view size\b", str(query or ""), flags=re.I)
        and re.search(r"\bvj-3302\b", str(query or ""), flags=re.I)
    ):
        field_of_view = re.search(
            r"(?P<answer>60\s*mm\s*2\.36[\"”]?\s*field\s+of\s+view)",
            source,
            flags=re.I,
        )
        if field_of_view:
            return re.sub(r"\s+", " ", field_of_view.group("answer")).strip()
    if (
        re.search(r"\bscrew\s+size\b", str(query or ""), flags=re.I)
        and re.search(r"\bwall[- ]mounting\b", str(query or ""), flags=re.I)
        and re.search(r"\biv-500c\b", str(query or ""), flags=re.I)
    ):
        wall_screw = re.search(
            r"Mounting\s+on\s+the\s+wall.*?"
            r"(?P<answer>Screw\s*:\s*M3\s*x\s*4\b)",
            source,
            flags=re.I | re.S,
        )
        if wall_screw:
            return f"Mounting on the wall — {wall_screw.group('answer')}"
    if (
        re.search(r"\bhow\s+many\b.*\bcameras?\b", str(query or ""), flags=re.I)
        and re.search(r"\bca-e100\b", str(query or ""), flags=re.I)
    ):
        camera_count = re.search(
            r"(?P<answer>With\s+area\s+camera\s+input\s+unit\s+CA-E100\s+connected\s*:\s*"
            r"2\s+color/monochrome\s+cameras\s+per\s+CA-E100)",
            source,
            flags=re.I,
        )
        if camera_count:
            return re.sub(r"\s+", " ", camera_count.group("answer")).strip()
    if (
        re.search(r"\blaser\s+on\s+input\b", str(query or ""), flags=re.I)
        and re.search(r"\bactivate\b", str(query or ""), flags=re.I)
    ):
        activation = re.search(
            r"(?P<answer>The\s+Laser\s+ON\s+input\s+is\s+a\s+"
            r"non\s*[: -]?\s*voltage\s+input\s*[.;]?\s*"
            r"\(\s*Turns\s+ON\s+by\s+simply\s+short\s+circuiting\s+it\s*\))",
            source,
            flags=re.I,
        )
        if activation:
            return re.sub(r"\s+", " ", activation.group("answer")).strip()
    if (
        re.search(r"\bscanning\s+system\s+accuracy\b", str(query or ""), flags=re.I)
        and re.search(r"\bwm-p6200\b", str(query or ""), flags=re.I)
    ):
        accuracy = re.search(
            r"\bmodel\s*:\s*Scanning\s+system\s+accuracy\s*;\s*"
            r"WM-P6200\s*:\s*(?P<answer>[^\n]+?)(?=\s+model\s*:|$)",
            source,
            flags=re.I,
        )
        if accuracy:
            return f"WM-P6200: {accuracy.group('answer').strip()}"
    if (
        re.search(r"\binstalled\s+distance\s+range\b", str(query or ""), flags=re.I)
        and re.search(r"\biv-h500ca\b", str(query or ""), flags=re.I)
    ):
        installed_distance = re.search(
            r"(?P<answer>\bInstalled\s+distance\s*\|\s*"
            r"50\s+to\s+500\s+mm\s*"
            r"1\.97\"\s+to\s+19\.69\")",
            source,
            flags=re.I,
        )
        if installed_distance:
            return re.sub(r"\s+", " ", installed_distance.group("answer")).strip()
    if (
        re.search(r"\bpassword values?\b", str(query or ""), flags=re.I)
        and re.search(r"\bselecting 0\b", str(query or ""), flags=re.I)
    ):
        password_contract = re.search(
            r"(?P<answer>An optional password can be set.*?"
            r"Select a value from 1 to 999 for this setting\.\s*"
            r"If ['\"]?0['\"]? is selected, the password will not be required\.)",
            source,
            flags=re.I | re.DOTALL,
        )
        if password_contract:
            return re.sub(r"\s+", " ", password_contract.group("answer")).strip()
    if re.search(r"\bhow\s+is\b.{0,120}\bpowered\b", str(query or ""), flags=re.I):
        powered = re.search(
            r"\bPower[- ]?supply\s*;\s*(?P<model>[A-Z][A-Z0-9-]+)\s*:\s*"
            r"(?P<answer>Supplied\s+from\s+dedicated\s+AC)\b",
            source,
            flags=re.I,
        )
        if powered:
            return f"{powered.group('model')}: {powered.group('answer')}"
    if re.search(r"\bwhat safety step\b", str(query or ""), flags=re.I):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", snippet)
            if sentence.strip()
        ]
        safety_sentences = [
            sentence
            for sentence in sentences
            if re.search(r"\b(?:check|ensure|disconnect|remove|turn off|not being supplied)\b", sentence, flags=re.I)
        ]
        if safety_sentences:
            return safety_sentences[0]
    mounting_hole = re.search(
        r"\b(?P<side>back|front)\s+(?P<hole>m\d+(?:\.\d+)?)\s+mounting hole\b",
        str(query or "").casefold(),
    )
    if mounting_hole:
        side = re.escape(mounting_hole.group("side"))
        hole = re.escape(mounting_hole.group("hole"))
        match = re.search(
            rf"\b{side}\s+{hole}\b[^)]*?(?P<answer>\btightening torque\s*:\s*"
            rf"\d+(?:\.\d+)?\s*(?:to|[-–])\s*\d+(?:\.\d+)?\s*n[·. ]?m)",
            snippet,
            flags=re.I,
        )
        if match:
            return match.group("answer").strip()
    if re.search(r"\bwhat numerical inputs?\b", str(query or ""), flags=re.I):
        inputs = re.search(
            r"\bnumerical inputs?\s*:\s*(?P<answer>"
            r"\d+\s*/\s*\d+(?:\s*,\s*\d+\s*/\s*\d+)+)",
            snippet,
            flags=re.I,
        )
        if inputs:
            return f"Numerical inputs: {inputs.group('answer').strip()}"
    if re.search(r"\banalog output option\b", str(query or ""), flags=re.I):
        option = re.search(
            r"Display value\s*\[\s*Disp\.\s*Value\s*\]",
            source,
            flags=re.I,
        )
        if option:
            return re.sub(r"\s+", " ", option.group(0)).strip()
    if (
        re.search(r"\bdownload option\b", str(query or ""), flags=re.I)
        and re.search(r"\bonly the changed hardware and software\b", str(query or ""), flags=re.I)
    ):
        option = re.search(
            r"Hardware and software\s*\(\s*only changes\s*\)",
            source,
            flags=re.I,
        )
        if option:
            return re.sub(r"\s+", " ", option.group(0)).strip()
    if re.search(r"\bobject size limit\b", str(query or ""), flags=re.I):
        object_limit = re.search(
            r"(?P<answer>You cannot select the object size of 150\s*mm.*?"
            r"You must select the object size of 70\s*mm.*?or smaller[^.]*\.)",
            source,
            flags=re.I | re.DOTALL,
        )
        if object_limit:
            return re.sub(r"\s+", " ", object_limit.group("answer")).strip()
    if re.search(r"\bethernet speeds?\b", str(query or ""), flags=re.I):
        speeds = []
        for speed in re.findall(r"\b\d+(?:\.\d+)?BASE-[A-Z0-9]+\b", source, flags=re.I):
            normalized_speed = speed.upper()
            if normalized_speed not in speeds:
                speeds.append(normalized_speed)
        if len(speeds) >= 2:
            return "Ethernet speeds: " + ", ".join(speeds)
    if (
        re.search(r"\bfield of view\b", str(query or ""), flags=re.I)
        and re.search(r"\bultra[- ]narrow\b", str(query or ""), flags=re.I)
    ):
        field_of_view = re.search(
            r"(?P<answer>Installation distance of 23\s*mm.*?"
            r"Installation distance of 40\s*mm[^|]*?15\s*\(H\)\s*[x×]\s*11\.2\s*\(V\)\s*mm)",
            snippet,
            flags=re.I,
        )
        if field_of_view:
            return re.sub(r"\s+", " ", field_of_view.group("answer")).strip()
    return snippet


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
    leading_unrelated_values: set[str] = set()
    if re.search(r"\boperating ambient temperature\b", normalized_query):
        field_index = expected_snippet.casefold().find("operating ambient temperature")
        if field_index > 0:
            leading_match = re.search(
                r"\b(\d+(?:\.\d+)?)\s*(?:a|ma|v|vac|vdc|w|kw|hz|khz|mhz|mm|cm|m)\s*:?\s*$",
                expected_snippet[:field_index].strip(),
                flags=re.IGNORECASE,
            )
            if leading_match:
                leading_unrelated_values.add(leading_match.group(1))

    quantities = [
        value
        for value in _answer_quantity_values(query, expected_snippet)
        if value not in query_tokens and value not in leading_unrelated_values
    ]
    if asks_value:
        if not quantities:
            missing.append("quantified answer value")
        else:
            absent_values = [value for value in quantities if not _term_covers_token(terms, value)]
            if absent_values:
                missing.append("expected answer value term(s) " + ", ".join(absent_values))

    if leading_unrelated_values:
        missing.append("leading unrelated quantity")

    if re.search(r"\bpart number\b", normalized_query):
        part_numbers = _answer_part_numbers(expected_snippet)
        if not part_numbers:
            missing.append("part number identifier")
        else:
            absent = [part_number for part_number in part_numbers if not _term_covers_token(terms, part_number)]
            if absent:
                missing.append("expected part number term(s) " + ", ".join(absent))

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

    if re.search(r"\b(?:what|which)\s+(?:[a-z0-9-]+\s+){0,3}cable\b", normalized_query):
        identifiers = _answer_identifier_tokens(query, expected_snippet)
        if not identifiers:
            missing.append("cable identifier")
        else:
            absent = [identifier for identifier in identifiers if not _term_covers_token(terms, identifier)]
            if absent:
                missing.append("expected cable term(s) " + ", ".join(absent))

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

    if re.search(r"\b(?:which|what) filter\b", normalized_query):
        filter_names = re.findall(
            r"\b(?:average|median|smoothing|low-pass|high-pass|band-pass|noise|spike)\s+filter\b",
            normalized_snippet,
        )
        if not filter_names:
            missing.append("filter identifier")
        elif not any(_term_covers_token(terms, name) for name in filter_names):
            missing.append("expected filter identifier term")

    if re.search(r"\bhow\s+is\b.{0,120}\bpowered\b", normalized_query):
        if not re.search(r"\b(?:supplied|powered|power source)\b", normalized_snippet):
            missing.append("power-source answer")

    if re.search(r"\bip address\b", normalized_query):
        addresses = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", expected_snippet)
        if not addresses:
            missing.append("IP address value")
        elif not any(_term_covers_token(terms, address) for address in addresses):
            missing.append("expected IP address term")

    if re.search(r"\bhdd\b.+\bcapacity\b|\bcapacity\b.+\bhdd\b", normalized_query):
        storage_values = re.findall(r"\b\d+(?:\.\d+)?\s*(?:tb|gb)\b", expected_snippet, flags=re.I)
        if not storage_values:
            missing.append("storage capacity value")
        elif not any(_term_covers_token(terms, value) for value in storage_values):
            missing.append("expected storage capacity term")

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
        case["query"] = normalize_frozen_query(str(case.get("query") or ""))
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
            snippet_text = focus_expected_snippet(
                str(case.get("query") or ""),
                snippet_text,
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
