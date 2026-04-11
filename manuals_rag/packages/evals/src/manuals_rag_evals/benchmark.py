from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalCase:
    query: str
    expected_term: str


def score_case(answer_text: str, expected_term: str) -> bool:
    return expected_term.lower() in answer_text.lower()
