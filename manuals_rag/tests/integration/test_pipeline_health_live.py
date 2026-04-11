from __future__ import annotations

import httpx
import pytest

from manuals_rag_evals.pipeline_health import check_live_api_stage
from tests.helpers import fixture_pdf_path


API_BASE = "http://127.0.0.1:8600"
FIXTURE = fixture_pdf_path()


def _api_available() -> bool:
    try:
        response = httpx.get(f"{API_BASE}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _api_available(), reason="Live API stack is not running.")


def test_pipeline_health_live_api_stage():
    result = check_live_api_stage(FIXTURE, api_base=API_BASE)
    assert result.status == "pass"
    assert result.details["search_results"] > 0
    assert result.details["citations"] > 0
