from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "CA-EN100U_Datasheet.pdf"
OUTPUT_DIR = REPO_ROOT / "test_reports"
sys.path.insert(0, str(REPO_ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "schemas" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "parsers" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "normalizers" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "chunking" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "retrieval" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "answering" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "permissions" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "observability" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "evals" / "src"))

from manuals_rag_evals.pipeline_health import report_to_json, report_to_markdown, run_pipeline_health_checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8600")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    report = run_pipeline_health_checks(args.fixture, api_base=args.api_base, include_live=not args.skip_live)
    timestamp = report.generated_at.replace(":", "").replace("-", "").replace("T", "_")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"pipeline_health_{timestamp}.json"
    md_path = OUTPUT_DIR / f"pipeline_health_{timestamp}.md"
    json_path.write_text(report_to_json(report), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    print(report_to_json(report))
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
