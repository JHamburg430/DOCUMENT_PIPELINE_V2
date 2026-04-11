from __future__ import annotations

from pathlib import Path


def fixture_pdf_path(name: str = "CA-EN100U_Datasheet.pdf") -> Path:
    return Path(__file__).resolve().parent / "fixtures" / name


def tmp_eval_pdf_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "tmp_eval_docs" / name


def tmp_eval_small_pdf_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "tmp_eval_docs_small" / name
