"""Conservative subject-scoped numeric version checks over grounded metadata.

A mention of version 2 does not prove version 3 incompatible. Only explicit
bounds establish conflicts. Unknowns remain retrievable, with a trace.
"""
from __future__ import annotations

import re
from typing import Any

from manuals_rag_schemas.documents import SearchResult

_NUMBER = r'\d+(?:\.\d+)*'


def _compact(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.casefold())


def _version(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in value.split('.')]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def version_relation(requested: str, expression: str) -> str:
    """Return applicable/conflicting/unknown without guessing version semantics."""
    text = expression.strip().casefold()
    exact = re.fullmatch(rf'v?({_NUMBER})', text)
    if exact:
        return 'applicable' if _version(requested) == _version(exact[1]) else 'unknown'
    interval = re.fullmatch(rf'v?({_NUMBER})\s*(?:through|to|–|-)\s*v?({_NUMBER})', text)
    if interval:
        low, high = _version(interval[1]), _version(interval[2])
        if low > high:
            return 'unknown'
        return 'applicable' if low <= _version(requested) <= high else 'conflicting'
    bound = re.fullmatch(rf'(>=|<=|>|<|at least|minimum|before|prior to|after)\s*v?({_NUMBER})', text)
    suffix = re.fullmatch(rf'v?({_NUMBER})\s*(or later|or newer|and later|or earlier|and earlier|or higher)', text)
    if suffix:
        op, number = ('<=' if 'earlier' in suffix[2] else '>='), suffix[1]
    elif bound:
        op, number = bound[1], bound[2]
    else:
        return 'unknown'
    op = {'at least': '>=', 'minimum': '>=', 'before': '<', 'prior to': '<', 'after': '>'}.get(op, op)
    left, right = _version(requested), _version(number)
    matched = {'>=': left >= right, '<=': left <= right, '>': left > right, '<': left < right}[op]
    return 'applicable' if matched else 'conflicting'


def assess_metadata_applicability(query: str, metadata: dict[str, Any]) -> dict[str, Any]:
    constraints = list(re.finditer(rf'\b(firmware|software)(?:\s+(?:version|release))?\s+v?({_NUMBER})\b', query, re.I))
    if not constraints:
        return {'state': 'not_requested', 'checks': []}
    if len(constraints) > 1:
        return {'state': 'unknown', 'checks': [], 'reason': 'requires_per_entity_version_branches'}
    checks = []
    for constraint in constraints:
        kind, requested = constraint[1].lower(), constraint[2]
        records = []
        for item in metadata.get(f'{kind}_applicability') or []:
            if not isinstance(item, dict):
                continue
            subject = _compact(str(item.get('subject') or ''))
            # Subject must be explicitly named. Never borrow another controller's
            # firmware, or infer software identity from the manual's product.
            pattern = r'(?<![a-z0-9])' + r'[^a-z0-9]*'.join(map(re.escape, subject)) + r'(?![a-z0-9])'
            if not subject or not re.search(pattern, query, re.I):
                continue
            try:
                confident = float(item.get('confidence') or 0) >= .8
            except (TypeError, ValueError):
                confident = False
            if not (item.get('grounded') is True and confident and item.get('source_quote')
                    and item.get('page_from') is not None
                    and item.get('relation') in {'applies_to', 'compatible_with'}):
                continue
            if item.get('verification_status', 'confirmed') != 'confirmed':
                continue
            records.append({'subject': item['subject'], 'version': item.get('version'),
                            'state': version_relation(requested, str(item.get('version') or '')),
                            'page_from': item['page_from'], 'source_quote': item['source_quote']})
        states = {item['state'] for item in records}
        # Contradictory metadata is not permission to select the convenient fact.
        state = ('unknown' if not states or 'unknown' in states or len(states) > 1
                 else next(iter(states)))
        checks.append({'kind': kind, 'requested': requested, 'state': state, 'records': records})
    states = {check['state'] for check in checks}
    state = 'conflicting' if 'conflicting' in states else ('unknown' if 'unknown' in states else 'applicable')
    return {'state': state, 'checks': checks}


def rank_by_applicability(query: str, results: list[SearchResult]) -> list[SearchResult]:
    ranked = []
    for result in results:
        assessment = assess_metadata_applicability(query, result.metadata or {})
        if assessment['state'] == 'not_requested':
            ranked.append(result)
        else:
            ranked.append(result.model_copy(update={'metadata': {
                **(result.metadata or {}), 'query_applicability': assessment,
            }}))
    # Preserve unknowns and conflicts for diagnostics/corrective evidence. Only
    # explicit conflict is penalized; unknown evidence is not declared invalid.
    return sorted(ranked, key=lambda result: (result.metadata or {}).get('query_applicability', {}).get('state') == 'conflicting')
