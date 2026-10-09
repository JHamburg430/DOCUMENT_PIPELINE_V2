from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark" / "audit_onsite_agent_eval.py"
SPEC = importlib.util.spec_from_file_location("audit_onsite_agent_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path):
    rows = [{"case_id": "case-1"}, {"case_id": "case-2"}]
    dataset = tmp_path / "bank.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "bank.manifest.json"
    manifest.write_text(json.dumps({"sha256": MODULE._sha256(dataset)}))
    records = []
    for row in rows:
        records.append({
            "case_id": row["case_id"],
            "backends": {
                backend: {
                    "status": "pass",
                    "insufficient_evidence": False,
                    "evidence_gate": {"status": "accepted"},
                }
                for backend in ("langgraph_agent", "llamaindex_agent")
            },
        })
    result = {
        "run_id": "run",
        "dataset_sha256": MODULE._sha256(dataset),
        "selected_case_ids": [row["case_id"] for row in rows],
        "backends": ["langgraph_agent", "llamaindex_agent"],
        "records": records,
        "summary": {"passed": 4, "failed": 0, "errors": 0, "total": 4},
        "status": "completed",
        "exit_code": 0,
        "provenance": {"source": {"revision": "abc", "dirty": False}},
    }
    return dataset, manifest, result


def test_audit_accepts_exact_clean_all_pass_matrix(tmp_path: Path) -> None:
    dataset, manifest, result = _fixture(tmp_path)
    result["summary"] = {"passed": 4, "failed": 0, "errors": 0, "total": 4}
    report = MODULE.audit(result, dataset, manifest)
    assert report["accepted_for_production_enablement"] is True
    assert report["coverage"]["observed"] == 4
    assert report["rollout_recommendation"] == "GO"


def test_audit_rejects_gate_accepted_source_contract_failure(tmp_path: Path) -> None:
    dataset, manifest, result = _fixture(tmp_path)
    result["records"][0]["backends"]["langgraph_agent"]["status"] = "fail"
    result["summary"] = {"passed": 3, "failed": 1, "errors": 0, "total": 4}
    result["status"] = "failed"
    result["exit_code"] = 1
    report = MODULE.audit(result, dataset, manifest)
    assert report["accepted_for_production_enablement"] is False
    assert report["unsafe_accepted_count"] == 1
    assert report["rollout_recommendation"] == "NO-GO"
