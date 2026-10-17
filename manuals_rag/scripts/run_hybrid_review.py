#!/usr/bin/env python3
"""Create a local fallback or injected-provider review artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manuals_rag_retrieval.hybrid_improvement import review_with_provider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    result = review_with_provider(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"provider": result["provider"], "review_status": result["review_status"], "ranked_changes": len(result["ranked_changes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
