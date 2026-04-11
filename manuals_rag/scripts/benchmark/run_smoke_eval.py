from __future__ import annotations

import json
from pathlib import Path

import httpx

from manuals_rag_evals.benchmark import score_case


API_BASE = "http://127.0.0.1:8600"
AUTH = {"Authorization": "Bearer user-token"}


def main() -> int:
    cases = json.loads((Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "smoke_eval_cases.json").read_text())
    passed = 0
    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        for case in cases:
            response = client.post(
                "/query",
                headers=AUTH,
                json={
                    "query": case["query"],
                    "corpus_ids": ["manuals_vendor_keyence"],
                    "filters": case.get("filters", {}),
                    "response_mode": "answer_with_citations",
                },
            )
            response.raise_for_status()
            payload = response.json()
            ok = score_case(payload["answer"], case["expected_term"])
            passed += int(ok)
            print(json.dumps({"query": case["query"], "passed": ok, "answer": payload["answer"][:240]}, indent=2))
    print(json.dumps({"passed": passed, "total": len(cases)}, indent=2))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
