from __future__ import annotations

from typing import Any


DOCUMENT_METADATA_SIGNAL_KEYS = (
    "product_model",
    "product_models",
    "product_family",
    "product_families",
    "devices",
    "part_numbers",
    "settings",
    "parameters",
    "section_path",
    "keywords",
    "unit_tokens",
    "identifier_tokens",
    "table_column_headers",
    "table_row_headers",
    "document_topics",
    "menu_labels",
    "protocol_terms",
)


def enrich_document_metadata_with_chunk_signals(
    document: dict[str, Any],
    chunk_rows: list[dict[str, Any]],
    *,
    max_values_per_key: int = 200,
) -> dict[str, Any]:
    metadata_json = dict(document.get("metadata_json") or {})
    signals: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for row in chunk_rows:
        metadata = row.get("metadata_json") if isinstance(row, dict) else None
        if not isinstance(metadata, dict):
            continue
        for key in DOCUMENT_METADATA_SIGNAL_KEYS:
            values = signals.setdefault(key, [])
            seen_values = seen.setdefault(key, set())
            if len(values) >= max_values_per_key:
                continue
            for value in _flatten_metadata_values(metadata.get(key)):
                normalized = value.strip()
                if not normalized or normalized in seen_values:
                    continue
                values.append(normalized)
                seen_values.add(normalized)
                if len(values) >= max_values_per_key:
                    break
    metadata_json["chunk_metadata_signals"] = {key: values for key, values in signals.items() if values}
    return {**document, "metadata_json": metadata_json}


def _flatten_metadata_values(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_metadata_values(item))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_metadata_values(item))
        return flattened
    return [str(value)]
