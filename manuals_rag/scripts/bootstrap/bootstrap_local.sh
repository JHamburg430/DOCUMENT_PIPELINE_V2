#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python"

cd "$ROOT"
"$PYTHON" -m pip install -r requirements.txt
docker compose -f infra/compose/docker-compose.yml up -d --build
