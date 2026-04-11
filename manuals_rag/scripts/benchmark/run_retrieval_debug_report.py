from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MANUALS_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = MANUALS_ROOT / "test_reports"
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "schemas" / "src"))
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "retrieval" / "src"))
sys.path.insert(0, str(MANUALS_ROOT / "packages" / "evals" / "src"))

from manuals_rag_evals.retrieval_debug import debug_report_to_json, debug_report_to_markdown, debug_retrieval_report


DEFAULT_QUERIES = [
    "What product is described in the CA-EN100U datasheet?",
    "What does the datasheet say about the encoder relay unit?",
    "What revision date is listed for CA-EN100U?",
    "Which document section mentions KEYENCE AMERICA?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-id", action="append", dest="corpus_ids", required=True)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    report = debug_retrieval_report(corpus_ids=args.corpus_ids, queries=queries, top_k=args.top_k)
    timestamp = report["generated_at"].replace(":", "").replace("-", "").replace("T", "_")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"retrieval_debug_{timestamp}.json"
    md_path = OUTPUT_DIR / f"retrieval_debug_{timestamp}.md"
    json_path.write_text(debug_report_to_json(report), encoding="utf-8")
    md_path.write_text(debug_report_to_markdown(report), encoding="utf-8")
    print(debug_report_to_json(report))
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
