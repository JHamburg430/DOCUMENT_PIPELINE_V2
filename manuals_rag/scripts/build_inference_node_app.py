#!/usr/bin/env python3
"""Build the legacy managed Linux zipapp (system Python/Tk; optional GTK tray).

Use build_inference_node_linux.py for employee deployment; it produces a
self-contained one-file executable with Python and Tk embedded.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipapp
import json
import os
from pathlib import Path
from build_inference_node_bundle import _validate_managed_gateway_url, managed_config


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "manuals_rag/packages/inference_fabric/src/manuals_rag_inference"
APP_SOURCE = REPO_ROOT / "manuals_rag/scripts/run_inference_node_client.py"
INSTALLER_ROOT = REPO_ROOT / "manuals_rag/installers"


def build(output: Path, *, gateway_url: str = "", update_config=None) -> Path:
    gateway_url = _validate_managed_gateway_url(gateway_url)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="inference-node-build-") as temporary:
        stage = Path(temporary)
        shutil.copytree(PACKAGE_ROOT, stage / "manuals_rag_inference", ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(APP_SOURCE, stage / "node_app.py")
        shutil.copy2(REPO_ROOT / 'manuals_rag/scripts/provis_agent_link.py', stage / 'provis_agent_link.py')
        shutil.copy2(REPO_ROOT / 'manuals_rag/scripts/provis_agent_link_mcp.py', stage / 'provis_agent_link_mcp.py')
        shutil.copy2(REPO_ROOT / 'manuals_rag/scripts/run_agent_link_mcp.py', stage / 'run_agent_link_mcp.py')
        if INSTALLER_ROOT.is_dir():
            shutil.copytree(INSTALLER_ROOT, stage / "installers")
        (stage / "__main__.py").write_text("from node_app import main\n\nmain()\n", encoding='utf-8')
        (stage / 'provis-node-config.json').write_text(json.dumps(managed_config(gateway_url, update_config)), encoding='utf-8')
        zipapp.create_archive(stage, str(output), interpreter="/usr/bin/env python3")
    output.chmod(0o755)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "manuals_rag/dist/inference-node.pyz")
    parser.add_argument('--gateway-url', default=os.getenv('PROVIS_INFERENCE_GATEWAY_URL', ''), help='Verified managed HTTPS gateway URL')
    parser.add_argument('--update-config', type=Path)
    args = parser.parse_args()
    print(build(args.output, gateway_url=args.gateway_url, update_config=json.loads(args.update_config.read_text(encoding='utf-8')) if args.update_config else None))


if __name__ == "__main__":
    main()
