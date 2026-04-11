from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TmpEvalDocSet:
    name: str
    directory: str
    documents: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_tmp_eval_doc_sets(manuals_root: Path) -> list[TmpEvalDocSet]:
    sets: list[TmpEvalDocSet] = []
    for path in sorted(manuals_root.glob("tmp_eval_docs*")):
        if not path.is_dir():
            continue
        documents = sorted(str(doc) for doc in path.glob("*.pdf"))
        if not documents:
            continue
        sets.append(TmpEvalDocSet(name=path.name, directory=str(path), documents=documents))
    return sets


def discover_keyence_inventory(repo_root: Path) -> list[str]:
    keyence_root = repo_root / "Technical_Documents" / "Keyence"
    return sorted(str(doc) for doc in keyence_root.glob("*.pdf"))


def aggregate_tmp_eval_results(
    *,
    keyence_inventory: list[str],
    doc_sets: list[TmpEvalDocSet],
    run_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_queries = sum(int(result["summary"].get("total_queries", 0)) for result in run_results)
    passed_queries = sum(int(result["summary"].get("passed_queries", 0)) for result in run_results)
    failed_queries = sum(int(result["summary"].get("failed_queries", 0)) for result in run_results)

    chunk_type_totals: dict[str, dict[str, int]] = {}
    by_document: dict[str, dict[str, int]] = {}
    for result in run_results:
        summary = result["summary"]
        for chunk_type, stats in summary.get("by_chunk_type", {}).items():
            bucket = chunk_type_totals.setdefault(chunk_type, {"total": 0, "passed": 0})
            bucket["total"] += int(stats.get("total", 0))
            bucket["passed"] += int(stats.get("passed", 0))
        for filename, stats in summary.get("by_document", {}).items():
            bucket = by_document.setdefault(filename, {"total": 0, "passed": 0})
            bucket["total"] += int(stats.get("total", 0))
            bucket["passed"] += int(stats.get("passed", 0))

    tmp_docs = [Path(doc).name for doc_set in doc_sets for doc in doc_set.documents]
    keyence_names = {Path(path).name for path in keyence_inventory}
    tmp_coverage = {
        "tmp_document_count": len(tmp_docs),
        "keyence_inventory_count": len(keyence_inventory),
        "tmp_documents_present_in_keyence": sum(1 for name in tmp_docs if name in keyence_names),
        "missing_from_keyence": sorted(name for name in tmp_docs if name not in keyence_names),
    }

    aggregate = {
        "tmp_doc_sets": [doc_set.to_dict() for doc_set in doc_sets],
        "tmp_coverage": tmp_coverage,
        "runs": run_results,
        "overall": {
            "total_queries": total_queries,
            "passed_queries": passed_queries,
            "failed_queries": failed_queries,
            "pass_rate": round(passed_queries / total_queries, 4) if total_queries else 0.0,
            "benchmark_validity_rate": round(
                sum(float(result["summary"].get("benchmark_validity_rate", 0.0)) * int(result["summary"].get("total_queries", 0)) for result in run_results) / total_queries,
                4,
            )
            if total_queries
            else 0.0,
            "candidate_recall_rate": round(
                sum(float(result["summary"].get("candidate_recall_rate", 0.0)) * int(result["summary"].get("total_queries", 0)) for result in run_results) / total_queries,
                4,
            )
            if total_queries
            else 0.0,
            "by_chunk_type": chunk_type_totals,
            "by_document": by_document,
        },
    }
    overall = aggregate["overall"]
    structured_scores = [
        stats["passed"] / stats["total"]
        for chunk_type, stats in chunk_type_totals.items()
        if chunk_type in {"spec_record", "table_record", "procedure_record"} and stats["total"]
    ]
    aggregate["production_readiness"] = {
        "ready": (
            overall["pass_rate"] >= 0.70
            and overall["benchmark_validity_rate"] >= 0.95
            and chunk_type_totals.get("atomic_text", {}).get("passed", 0) / max(chunk_type_totals.get("atomic_text", {}).get("total", 1), 1) >= 0.60
            and (min(structured_scores) >= 0.75 if structured_scores else False)
        ),
        "thresholds": {
            "overall_pass_rate": 0.70,
            "benchmark_validity_rate": 0.95,
            "atomic_text_pass_rate": 0.60,
            "structured_chunk_min_pass_rate": 0.75,
        },
    }
    return aggregate


def render_tmp_eval_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tmp Document Retrieval Eval",
        "",
        "## Coverage",
        f"- Keyence inventory PDFs: {report['tmp_coverage']['keyence_inventory_count']}",
        f"- Tmp eval PDFs: {report['tmp_coverage']['tmp_document_count']}",
        f"- Tmp PDFs present in Keyence inventory: {report['tmp_coverage']['tmp_documents_present_in_keyence']}",
    ]
    missing = report["tmp_coverage"]["missing_from_keyence"]
    if missing:
        lines.append(f"- Missing from Keyence inventory: {', '.join(missing)}")
    lines.extend(
        [
            "",
            "## Overall",
            f"- Total queries: {report['overall']['total_queries']}",
            f"- Passed queries: {report['overall']['passed_queries']}",
            f"- Failed queries: {report['overall']['failed_queries']}",
            f"- Pass rate: {report['overall']['pass_rate']:.2%}",
            f"- Benchmark validity: {report['overall'].get('benchmark_validity_rate', 0.0):.2%}",
            f"- Candidate recall: {report['overall'].get('candidate_recall_rate', 0.0):.2%}",
            f"- Production ready: {'yes' if report.get('production_readiness', {}).get('ready') else 'no'}",
            "",
            "## Per Run",
        ]
    )
    for run in report["runs"]:
        summary = run["summary"]
        lines.append(
            f"- {run['doc_set_name']}: {summary.get('passed_queries', 0)}/{summary.get('total_queries', 0)} "
            f"({summary.get('pass_rate', 0.0):.2%})"
        )
    return "\n".join(lines) + "\n"


def write_tmp_eval_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_tmp_eval_markdown(report), encoding="utf-8")
