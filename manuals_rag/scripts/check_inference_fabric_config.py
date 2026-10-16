#!/usr/bin/env python3
"""Run non-mutating production checks for the central inference gateway."""

from __future__ import annotations

import os
import sys

from manuals_rag_inference.preflight import validate_gateway_environment


def main() -> int:
    result = validate_gateway_environment(os.environ)
    print("inference_gateway_preflight=" + ("PASS" if result.ok else "FAIL"))
    for warning in result.warnings:
        print("warning=" + warning)
    for error in result.errors:
        print("error=" + error)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
