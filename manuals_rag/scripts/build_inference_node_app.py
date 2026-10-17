#!/usr/bin/env python3
"""Build the managed Linux desktop zipapp (system Python/Tk; optional GTK tray)."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipapp
import json
import os
from pathlib import Path
from build_inference_node_bundle import _validate_managed_gateway_url


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "manuals_rag/packages/inference_fabric/src/manuals_rag_inference"
APP_SOURCE = REPO_ROOT / "manuals_rag/scripts/run_inference_node_client.py"


def build(output: Path, *, gateway_url: str = "") -> Path:
    gateway_url = _validate_managed_gateway_url(gateway_url)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="inference-node-build-") as temporary:
        stage = Path(temporary)
        shutil.copytree(PACKAGE_ROOT, stage / "manuals_rag_inference", ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(APP_SOURCE, stage / "node_app.py")
        shutil.copy2(REPO_ROOT / 'manuals_rag/scripts/provis_agent_link.py', stage / 'provis_agent_link.py')
        (stage / "__main__.py").write_text("from node_app import main\n\nmain()\n", encoding='utf-8')
        (stage / 'provis-node-config.json').write_text(json.dumps({'gateway_url': gateway_url}), encoding='utf-8')
        zipapp.create_archive(stage, str(output), interpreter="/usr/bin/env python3")
    output.chmod(0o755)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "manuals_rag/dist/inference-node.pyz")
    parser.add_argument('--gateway-url', default=os.getenv('PROVIS_INFERENCE_GATEWAY_URL', ''), help='Verified managed HTTPS gateway URL')
    args = parser.parse_args()
    print(build(args.output, gateway_url=args.gateway_url))


if __name__ == "__main__":
    main()
