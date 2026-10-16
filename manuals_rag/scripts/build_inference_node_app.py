#!/usr/bin/env python3
"""Build a dependency-free zipapp containing the inference node client."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipapp
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "manuals_rag/packages/inference_fabric/src/manuals_rag_inference"
APP_SOURCE = REPO_ROOT / "manuals_rag/scripts/run_inference_node_app.py"


def build(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="inference-node-build-") as temporary:
        stage = Path(temporary)
        shutil.copytree(PACKAGE_ROOT, stage / "manuals_rag_inference")
        shutil.copy2(APP_SOURCE, stage / "node_app.py")
        (stage / "__main__.py").write_text("from node_app import main\n\nmain()\n")
        zipapp.create_archive(stage, str(output), interpreter="/usr/bin/env python3")
    output.chmod(0o755)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "manuals_rag/dist/inference-node.pyz")
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
