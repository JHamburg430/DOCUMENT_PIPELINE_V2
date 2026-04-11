from __future__ import annotations

import re
from typing import Any

from manuals_rag_retrieval.embeddings import tokenize


STRUCTURED_CHUNK_TYPES = {"spec_record", "datasheet_record", "table_record", "procedure_record", "warning_record"}
LOW_INFORMATION_PATTERNS = [
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:toll free|phone|fax)\b", re.IGNORECASE),
    re.compile(r"\.(?:gif|jpg|jpeg|png|svg|tif|tiff|bmp)\b", re.IGNORECASE),
]
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{6,}\d)")
VALUE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:v|a|ma|w|kw|mm|cm|m|ms|s|hz|khz|mhz|fps|kg|g|n|mpa|deg|c|°c|%)\b",
    re.IGNORECASE,
)


def content_quality_flags(*, content: str, title: str = "", section_path: list[str] | None = None, chunk_type: str = "") -> dict[str, Any]:
    text = " ".join(part for part in [title, " ".join(section_path or []), content] if part).strip()
    content_text = (content or "").strip()
    lowered = content_text.lower()
    tokens = tokenize(content_text)
    token_count = len(tokens)
    alpha_tokens = sum(1 for token in tokens if any(char.isalpha() for char in token))
    digit_tokens = sum(1 for token in tokens if any(char.isdigit() for char in token))
    url_like = any(pattern.search(content_text) for pattern in LOW_INFORMATION_PATTERNS[:1])
    page_marker = bool(LOW_INFORMATION_PATTERNS[1].search(content_text))
    contact_like = bool(LOW_INFORMATION_PATTERNS[2].search(content_text) or PHONE_PATTERN.search(content_text))
    asset_like = bool(LOW_INFORMATION_PATTERNS[3].search(content_text))
    has_value = bool(VALUE_PATTERN.search(content_text))
    has_sentence = any(mark in content_text for mark in ".:;") or token_count >= 8
    mostly_numeric = token_count > 0 and digit_tokens >= alpha_tokens and not has_value
    very_short = token_count <= 2
    low_alpha = alpha_tokens <= 1 and not has_value
    low_information = (
        bool(content_text)
        and (
            url_like
            or page_marker
            or contact_like
            or asset_like
            or (very_short and not has_value)
            or (low_alpha and not has_sentence)
            or mostly_numeric
        )
    )
    structured_low_information = chunk_type in STRUCTURED_CHUNK_TYPES and low_information
    return {
        "token_count": token_count,
        "alpha_tokens": alpha_tokens,
        "digit_tokens": digit_tokens,
        "has_value": has_value,
        "url_like": url_like,
        "page_marker": page_marker,
        "contact_like": contact_like,
        "asset_like": asset_like,
        "low_information": low_information,
        "structured_low_information": structured_low_information,
        "technical_signal_score": _technical_signal_score(content_text, chunk_type),
        "text_sample": text[:240],
    }


def summarize_result_quality(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", {}) or {}
    return content_quality_flags(
        content=str(getattr(result, "content", "") or ""),
        title=str(getattr(result, "title", "") or ""),
        section_path=list(getattr(result, "section_path", []) or []),
        chunk_type=str(metadata.get("chunk_type", "") or ""),
    )


def _technical_signal_score(content: str, chunk_type: str) -> float:
    tokens = tokenize(content)
    if not tokens:
        return 0.0
    score = 0.0
    if len(tokens) >= 5:
        score += 0.3
    if VALUE_PATTERN.search(content):
        score += 0.4
    if any(any(char.isdigit() for char in token) and any(char.isalpha() for char in token) for token in tokens):
        score += 0.2
    if chunk_type in STRUCTURED_CHUNK_TYPES:
        score += 0.15
    if PHONE_PATTERN.search(content) or LOW_INFORMATION_PATTERNS[0].search(content):
        score -= 0.4
    if LOW_INFORMATION_PATTERNS[1].search(content):
        score -= 0.3
    return round(score, 4)
