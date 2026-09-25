from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

ROLE_ALIASES = {
    "acceleration": ("acceleration",),
    "angle": ("angle",),
    "capacity": ("capacity",),
    "count": ("count", "counts", "total"),
    "current": ("current",),
    "delay": ("delay",),
    "depth": ("depth",),
    "diameter": ("diameter",),
    "distance": ("distance",),
    "duration": ("duration",),
    "frequency": ("frequency",),
    "height": ("height",),
    "interval": ("interval",),
    "length": ("length",),
    "limit": ("limit", "limits"),
    "line": ("line", "lines"),
    "overlap_line": ("overlap line", "overlap lines", "overlapping line", "overlapping lines"),
    "power": ("power",),
    "precision": ("precision",),
    "pressure": ("pressure",),
    "quantity": ("quantity",),
    "range": ("range",),
    "resolution": ("resolution",),
    "speed": ("speed",),
    "temperature": ("temperature",),
    "tolerance": ("tolerance",),
    "torque": ("torque",),
    "voltage": ("voltage",),
    "wavelength": ("wavelength",),
    "weight": ("weight", "mass"),
    "width": ("width",),
}

_UNIT_ALIASES = {
    "volt": "v", "volts": "v", "vdc": "v", "amp": "a", "amps": "a",
    "ampere": "a", "amperes": "a", "line": "lines", "watt": "w", "watts": "w",
    "inch": "in", "inches": "in", "sec": "s", "secs": "s", "seconds": "s",
    "minute": "min", "minutes": "min", "grams": "g", "kilograms": "kg",
    "degrees": "deg", "degree": "deg",
    "millivolt": "mv", "millivolts": "mv", "kilovolt": "kv", "kilovolts": "kv",
    "microamp": "ua", "microamps": "ua", "µa": "ua", "milliamp": "ma", "milliamps": "ma",
    "milliwatt": "mw", "milliwatts": "mw", "kilowatt": "kw", "kilowatts": "kw",
    "micrometer": "um", "micrometers": "um", "µm": "um", "nanometer": "nm", "nanometers": "nm",
    "meter": "m", "meters": "m", "millisecond": "ms", "milliseconds": "ms",
    "second": "s", "kilohertz": "khz", "megahertz": "mhz",
    "newton-meter": "nm", "newton-meters": "nm", "n m": "nm",
    "gram": "g", "milligram": "mg", "milligrams": "mg", "kilogram": "kg",
    "foot": "ft", "feet": "ft", "byte": "bytes", "kilobyte": "kb", "kilobytes": "kb",
    "megabyte": "mb", "megabytes": "mb", "gigabyte": "gb", "gigabytes": "gb",
    '"': "in",
}
_NUMBER = r"(?:" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r"|[-+]?\d+(?:\.\d+)?)"
_UNIT = (
    r"(?:milliwatts?|kilowatts?|watts?|mw|kw|w|millivolts?|kilovolts?|mv|kv|vdc|volts?|v|"
    r"microamps?|milliamps?|amperes?|amps?|[µu]a|ma|a|lines?|micrometers?|nanometers?|"
    r"[µu]m|nm|mm|cm|meters?|m|milliseconds?|seconds?|minutes?|ms|secs?|min|s|%|"
    r"megahertz|kilohertz|mhz|khz|hz|rpm|newton[- ]?meters?|n\s*m|n|mpa|kpa|bar|psi|"
    r"kilograms?|grams?|milligrams?|kg|mg|g|inches?|inch|in|\"|feet|ft|degrees?|deg|°c|c|"
    r"gigabytes?|megabytes?|kilobytes?|bytes?|gb|mb|kb|bps|baud)?"
)
_VALUE_RE = re.compile(rf"(?<![\w.])(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT})(?!\w)", re.I)
_ROLE_RE = re.compile(
    r"\b(?P<role>" + "|".join(sorted({alias for aliases in ROLE_ALIASES.values() for alias in aliases}, key=len, reverse=True)) + r")\b",
    re.I,
)
ACTION_ALIASES = {
    "check": ("check", "ensure", "verify"),
    "configure": ("adjust", "change", "configure", "set"),
    "connect": ("connect",),
    "disable": ("disable", "turn off"),
    "enable": ("enable", "turn on"),
    "increase": ("increase",),
    "install": ("install",),
    "perform": ("perform",),
    "reduce": ("reduce",),
    "remove": ("remove",),
    "replace": ("replace",),
    "restart": ("restart",),
    "use": ("use",),
    "wait": ("wait",),
}
_ACTION_RE = re.compile(
    r"\b(?P<action>" + "|".join(
        sorted({alias for aliases in ACTION_ALIASES.values() for alias in aliases}, key=len, reverse=True)
    ) + r")\b",
    re.I,
)
_NEGATION_RE = re.compile(r"\b(?:do\s+not|don't|must\s+not|never|not|prohibited|forbidden|without)\b", re.I)
_TARGET_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "before", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "then", "this", "to", "with", "above", "below",
}


@dataclass(frozen=True)
class RelationProfile:
    role_values: dict[str, frozenset[str]]
    actions: frozenset[str]
    action_polarities: frozenset[str]
    role_action_polarities: dict[str, frozenset[str]]
    action_targets: dict[str, frozenset[str]]


def canonical_value(number: str, unit: str = "") -> str:
    number_value = NUMBER_WORDS.get(number.lower(), number.lower())
    unit_value = _UNIT_ALIASES.get(unit.lower(), unit.lower())
    return " ".join(part for part in (number_value, unit_value) if part)


def _canonical_role(value: str) -> str:
    lowered = value.lower()
    for role, aliases in ROLE_ALIASES.items():
        if lowered in aliases:
            return role
    return lowered


def _canonical_action(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.lower()).strip()
    for action, aliases in ACTION_ALIASES.items():
        if lowered in aliases:
            return action
    return lowered


def _action_target_tokens(clause: str, match: re.Match[str], next_start: int) -> frozenset[str]:
    target_text = clause[match.end():next_start]
    tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]*", target_text.lower())
        if token not in _TARGET_STOPWORDS
        and token not in NUMBER_WORDS
        and not token.isdigit()
        and token not in _UNIT_ALIASES
        and token not in _UNIT_ALIASES.values()
    }
    return frozenset(tokens)


def _clauses(text: str) -> list[str]:
    normalized = re.sub(r"[^\S\n]+", " ", text)
    return [
        clause.strip()
        for clause in re.split(
            r"[\n;|]+|(?<!\d)\.|\.(?!\d)|\b(?:while|whereas|but)\b",
            normalized,
            flags=re.I,
        )
        if clause.strip()
    ]


def _is_nominal_set_value(clause: str, match: re.Match[str]) -> bool:
    if match.group("action").lower() != "set":
        return False
    prefix = clause[: match.start()]
    suffix = clause[match.end() :]
    return bool(
        re.search(r"(?:=|\bthe)\s*$", prefix, flags=re.IGNORECASE)
        and re.match(r"\s+value\b", suffix, flags=re.IGNORECASE)
    )


def _action_polarity(clause: str, match: re.Match[str]) -> str:
    """Bind negation to the comma-delimited phrase containing the action."""
    left = clause.rfind(",", 0, match.start())
    right = clause.find(",", match.end())
    local_phrase = clause[left + 1 : right if right >= 0 else len(clause)]
    return "negative" if _NEGATION_RE.search(local_phrase) else "affirmative"


def relation_profile(text: str) -> RelationProfile:
    role_values: dict[str, set[str]] = {}
    actions: set[str] = set()
    action_polarities: set[str] = set()
    role_action_polarities: dict[str, set[str]] = {}
    action_targets: dict[str, set[str]] = {}
    for clause in _clauses(text):
        values = [
            match
            for match in _VALUE_RE.finditer(clause)
            if not (
                not match.group("unit")
                and re.match(r"\s+times?\b", clause[match.end() :], flags=re.IGNORECASE)
            )
        ]
        roles = [
            role_match
            for role_match in _ROLE_RE.finditer(clause)
            if not any(
                value_match.start() <= role_match.start()
                and role_match.end() <= value_match.end()
                for value_match in values
            )
        ]
        claimed_values: set[int] = set()
        action_matches = [
            match for match in _ACTION_RE.finditer(clause) if not _is_nominal_set_value(clause, match)
        ]
        clause_polarity: str | None = None
        if action_matches:
            action_match_polarities = [
                _action_polarity(clause, action_match) for action_match in action_matches
            ]
            action_polarities.update(action_match_polarities)
            if len(set(action_match_polarities)) == 1:
                clause_polarity = action_match_polarities[0]
            for index, action_match in enumerate(action_matches):
                action = _canonical_action(action_match.group("action"))
                actions.add(action)
                end = action_matches[index + 1].start() if index + 1 < len(action_matches) else len(clause)
                signature = f"{action_match_polarities[index]}:{action}"
                action_targets.setdefault(signature, set()).update(
                    _action_target_tokens(clause, action_match, end)
                )
        # Count-like units can overlap their role token ("10 lines"), while a
        # compound role can follow a bare number ("2 overlap lines"). Claim
        # these before scanning the ordinary role-following value segment so a
        # prior role cannot steal them.
        for role_match in roles:
            role = _canonical_role(role_match.group("role"))
            for value_index, value_match in enumerate(values):
                if value_match.start() <= role_match.start() < value_match.end():
                    role_values.setdefault(role, set()).add(
                        canonical_value(value_match.group("number"), value_match.group("unit"))
                    )
                    claimed_values.add(value_index)
            if role == "overlap_line":
                preceding = [
                    (value_index, value_match)
                    for value_index, value_match in enumerate(values)
                    if value_index not in claimed_values
                    and value_match.end() <= role_match.start()
                    and role_match.start() - value_match.end() <= 3
                ]
                if preceding:
                    value_index, value_match = preceding[-1]
                    role_values.setdefault(role, set()).add(
                        canonical_value(value_match.group("number"), "lines")
                    )
                    claimed_values.add(value_index)

        for index, role_match in enumerate(roles):
            role = _canonical_role(role_match.group("role"))
            end = roles[index + 1].start() if index + 1 < len(roles) else len(clause)
            following_values = {
                canonical_value(
                    value_match.group("number"),
                    value_match.group("unit") or ("lines" if role in {"line", "overlap_line"} else ""),
                )
                for value_index, value_match in enumerate(values)
                if value_index not in claimed_values
                and value_match.start() >= role_match.end()
                and value_match.start() < end
            }
            if following_values:
                role_values.setdefault(role, set()).update(following_values)
                claimed_values.update(
                    value_index
                    for value_index, value_match in enumerate(values)
                    if value_index not in claimed_values
                    and value_match.start() >= role_match.end()
                    and value_match.start() < end
                )
                if clause_polarity:
                    role_action_polarities.setdefault(role, set()).add(clause_polarity)
            elif role in role_values and clause_polarity:
                role_action_polarities.setdefault(role, set()).add(clause_polarity)

        # A count written directly as "10 lines" has no separate role token
        # after unit-token filtering. Preserve it as the ordinary line count.
        for value_index, value_match in enumerate(values):
            if value_index in claimed_values:
                continue
            if _UNIT_ALIASES.get(value_match.group("unit").lower(), value_match.group("unit").lower()) == "lines":
                role_values.setdefault("line", set()).add(
                    canonical_value(value_match.group("number"), value_match.group("unit"))
                )
    structured_cell = re.search(
        r"Column headers:\s*(?P<column>.*?);\s*Row headers:\s*(?P<row>.*?);\s*"
        r"Cell value:\s*(?P<value>.*?)(?:;\s*Row:\s*\d+;\s*Column:\s*\d+)?\s*$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if structured_cell:
        column_roles = {
            _canonical_role(match.group("role"))
            for match in _ROLE_RE.finditer(structured_cell.group("column"))
        }
        cell_values = {
            canonical_value(match.group("number"), match.group("unit"))
            for match in _VALUE_RE.finditer(structured_cell.group("value"))
        }
        if cell_values:
            for role in column_roles:
                role_values[role] = set(cell_values)

    return RelationProfile(
        role_values={role: frozenset(values) for role, values in role_values.items()},
        actions=frozenset(actions),
        action_polarities=frozenset(action_polarities),
        role_action_polarities={
            role: frozenset(polarities) for role, polarities in role_action_polarities.items()
        },
        action_targets={
            signature: frozenset(targets) for signature, targets in action_targets.items()
        },
    )


def _action_relations_preserved(expected: RelationProfile, actual: RelationProfile) -> tuple[bool, dict[str, object]]:
    missing_actions = sorted(actual.actions - expected.actions)
    target_mismatches: dict[str, dict[str, list[str]]] = {}
    for signature, targets in actual.action_targets.items():
        expected_targets = expected.action_targets.get(signature, frozenset())
        if not targets.issubset(expected_targets):
            target_mismatches[signature] = {
                "expected": sorted(expected_targets),
                "actual": sorted(targets),
            }
    polarity_mismatch = bool(
        actual.action_polarities
        and not actual.action_polarities.issubset(expected.action_polarities)
    )
    details: dict[str, object] = {
        "missing_actions": missing_actions,
        "action_target_mismatches": target_mismatches,
        "expected_actions": sorted(expected.actions),
        "actual_actions": sorted(actual.actions),
        "expected_action_targets": {
            key: sorted(values) for key, values in expected.action_targets.items()
        },
        "actual_action_targets": {
            key: sorted(values) for key, values in actual.action_targets.items()
        },
    }
    return not missing_actions and not target_mismatches and not polarity_mismatch, details


def profile_is_preserved(expected: RelationProfile, actual: RelationProfile) -> tuple[bool, dict[str, object]]:
    missing = {
        role: {"expected": sorted(values), "actual": sorted(actual.role_values.get(role, frozenset()))}
        for role, values in expected.role_values.items()
        if not values.issubset(actual.role_values.get(role, frozenset()))
    }
    actions_preserved, action_details = _action_relations_preserved(expected, actual)
    polarity_mismatch = bool(action_details.get("actual_actions") and not actions_preserved)
    role_polarity_mismatch = {
        role: {
            "expected": sorted(polarities),
            "actual": sorted(actual.role_action_polarities.get(role, frozenset())),
        }
        for role, polarities in actual.role_action_polarities.items()
        if expected.role_action_polarities.get(role)
        and not polarities.issubset(expected.role_action_polarities[role])
    }
    details: dict[str, object] = {
        "expected": {role: sorted(values) for role, values in expected.role_values.items()},
        "actual": {role: sorted(values) for role, values in actual.role_values.items()},
        "missing_or_mismatched": missing,
        "expected_action_polarities": sorted(expected.action_polarities),
        "actual_action_polarities": sorted(actual.action_polarities),
        "polarity_mismatch": polarity_mismatch,
        "role_polarity_mismatch": role_polarity_mismatch,
        **action_details,
    }
    return not missing and actions_preserved and not role_polarity_mismatch, details


def answer_relations_supported(answer: str, evidence_texts: Iterable[str]) -> tuple[bool, dict[str, object]]:
    """Require every quantitative/action relation emitted by an answer to exist in one evidence item."""
    answer_profile = relation_profile(answer)
    if not answer_profile.role_values and not answer_profile.actions:
        return True, {"checked": False}
    evidence_profiles = [relation_profile(text) for text in evidence_texts]
    missing: dict[str, list[str]] = {}
    for role, values in answer_profile.role_values.items():
        answer_role_polarity = answer_profile.role_action_polarities.get(role, frozenset())
        if not any(
            values.issubset(profile.role_values.get(role, frozenset()))
            and (
                not answer_role_polarity
                or answer_role_polarity.issubset(profile.role_action_polarities.get(role, frozenset()))
            )
            for profile in evidence_profiles
        ):
            missing[role] = sorted(values)
    action_supported = True
    action_details: dict[str, object] = {}
    if answer_profile.actions:
        matching_profiles = []
        for profile in evidence_profiles:
            preserved, details = _action_relations_preserved(profile, answer_profile)
            if preserved:
                matching_profiles.append(profile)
            else:
                action_details = details
        action_supported = bool(matching_profiles)

    # A multi-relation answer clause may not assemble its support from separate citations.
    # Require one evidence item to preserve the complete emitted relation profile.
    complete_profile_supported = any(
        all(
            values.issubset(profile.role_values.get(role, frozenset()))
            for role, values in answer_profile.role_values.items()
        )
        and _action_relations_preserved(profile, answer_profile)[0]
        for profile in evidence_profiles
    )
    details = {
        "checked": True,
        "answer": {role: sorted(values) for role, values in answer_profile.role_values.items()},
        "missing": missing,
        "answer_action_polarities": sorted(answer_profile.action_polarities),
        "answer_role_action_polarities": {
            role: sorted(polarities)
            for role, polarities in answer_profile.role_action_polarities.items()
        },
        "action_supported": action_supported,
        "polarity_supported": action_supported,
        "complete_profile_supported": complete_profile_supported,
        **action_details,
    }
    return not missing and action_supported and complete_profile_supported, details
