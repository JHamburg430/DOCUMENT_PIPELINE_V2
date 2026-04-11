from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANUALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "evals" / "src"))

from manuals_rag_evals.tmp_eval_suite import (  # noqa: E402
    aggregate_tmp_eval_results,
    discover_keyence_inventory,
    discover_tmp_eval_doc_sets,
    write_tmp_eval_report,
)


OUTPUT_DIR = MANUALS_ROOT / "test_reports"
RUNNER = MANUALS_ROOT / "scripts" / "benchmark" / "run_large_retrieval_eval.py"


def _parse_runner_output(stdout: str) -> dict[str, object]:
    payloads: list[dict[str, object]] = []
    buffer: list[str] = []
    depth = 0
    for char in stdout:
        if char == "{":
            depth += 1
        if depth > 0:
            buffer.append(char)
        if char == "}":
            depth -= 1
            if depth == 0 and buffer:
                chunk = "".join(buffer)
                buffer = []
                try:
                    payloads.append(json.loads(chunk))
                except json.JSONDecodeError:
                    continue
    if not payloads:
        raise RuntimeError("Eval runner produced no JSON payloads.")
    summary = next(
        (payload for payload in reversed(payloads) if "total_queries" in payload and "passed_queries" in payload),
        None,
    )
    paths = next(
        (payload for payload in reversed(payloads) if "summary_path" in payload and "results_path" in payload),
        None,
    )
    if summary is None or paths is None:
        raise RuntimeError("Eval runner output did not contain summary and artifact paths.")
    return {"summary": summary, "paths": paths}


def _psql_copy(sql: str) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["docker", "exec", "-i", "compose-postgres-1", "psql", "-U", "manuals", "-d", "manuals_rag", "-c", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return list(csv.DictReader(io.StringIO(completed.stdout)))


def _find_existing_corpus(doc_set_name: str, document_names: list[str]) -> str | None:
    quoted_names = ",".join(f"'{name}'" for name in document_names)
    rows = _psql_copy(
        f"""
COPY (
    WITH matching AS (
        SELECT
            sd.corpus_id,
            regexp_replace(sd.source_filename, '^[0-9a-f-]{{36}}_', '') AS canonical_name,
            sd.ingest_status,
            dv.ingested_at
        FROM source_documents sd
        JOIN document_versions dv ON dv.id = sd.current_version_id
        WHERE regexp_replace(sd.source_filename, '^[0-9a-f-]{{36}}_', '') IN ({quoted_names})
          AND sd.ingest_status = 'indexed'
    )
    SELECT
        corpus_id,
        count(*) AS doc_count,
        count(distinct canonical_name) AS distinct_name_count,
        max(ingested_at) AS latest_ingested_at
    FROM matching
    GROUP BY corpus_id
    HAVING count(distinct canonical_name) = {len(document_names)}
       AND count(*) = {len(document_names)}
    ORDER BY max(ingested_at) DESC
) TO STDOUT WITH CSV HEADER
"""
    )
    if not rows:
        return None
    return rows[0]["corpus_id"]


def run_tmp_suite(*, max_queries: int) -> dict[str, object]:
    keyence_inventory = discover_keyence_inventory(REPO_ROOT)
    doc_sets = discover_tmp_eval_doc_sets(MANUALS_ROOT)
    run_results: list[dict[str, object]] = []
    for doc_set in doc_sets:
        document_names = [Path(path).name for path in doc_set.documents]
        existing_corpus_id = _find_existing_corpus(doc_set.name, document_names)
        runner_args = [sys.executable, str(RUNNER), "--max-queries", str(max_queries)]
        if existing_corpus_id:
            runner_args.extend(["--existing-corpus-id", existing_corpus_id])
        else:
            runner_args.extend(["--docs-dir", doc_set.directory, "--max-docs", str(len(doc_set.documents))])
        completed = subprocess.run(
            runner_args,
            cwd=str(MANUALS_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        parsed = _parse_runner_output(completed.stdout)
        run_results.append(
            {
                "doc_set_name": doc_set.name,
                "documents": document_names,
                "existing_corpus_id": existing_corpus_id,
                "summary": parsed["summary"],
                "artifacts": parsed["paths"],
            }
        )
    return aggregate_tmp_eval_results(
        keyence_inventory=keyence_inventory,
        doc_sets=doc_sets,
        run_results=run_results,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=240)
    args = parser.parse_args()

    report = run_tmp_suite(max_queries=args.max_queries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"tmp_eval_suite_{timestamp}.json"
    markdown_path = OUTPUT_DIR / f"tmp_eval_suite_{timestamp}.md"
    write_tmp_eval_report(report, json_path=json_path, markdown_path=markdown_path)
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(markdown_path)}, indent=2), flush=True)
    print(json.dumps(report["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
