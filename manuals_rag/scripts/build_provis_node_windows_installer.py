#!/usr/bin/env python3
"""Build an unsigned, standalone Windows per-user installer candidate.

This script packages only the installer. It does not sign or publish output.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_ROOT = REPO_ROOT / "manuals_rag" / "windows_installer"
SOURCE_ROOT = REPO_ROOT / "manuals_rag" / "packages" / "inference_fabric" / "src"
ICON_PATH = REPO_ROOT / "manuals_rag" / "assets" / "brand" / "provis-lens.png"
ENTRYPOINT = INSTALLER_ROOT / "entrypoint.py"


def build(output: Path, *, python: str | None = None, runner=subprocess.run) -> Path:
    if os.name != "nt":
        raise RuntimeError("The standalone Windows installer can only be packaged on Windows.")
    if not ICON_PATH.is_file():
        raise FileNotFoundError(ICON_PATH)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="provis-node-installer-build-") as temp_name:
        temp = Path(temp_name)
        command = [
            python, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
            "--name", "PROVISInferenceNodeInstaller",
            "--distpath", str(temp / "dist"),
            "--workpath", str(temp / "work"),
            "--specpath", str(temp / "spec"),
            "--paths", str(SOURCE_ROOT),
            "--paths", str(INSTALLER_ROOT),
            "--hidden-import", "provis_node_windows_installer_gui",
            "--add-data", f"{ICON_PATH};.",
            "--collect-all", "cryptography",
            str(ENTRYPOINT),
        ]
        result = runner(command, cwd=REPO_ROOT, check=False)
        if result.returncode:
            raise RuntimeError(f"PyInstaller failed with exit code {result.returncode}.")
        built = temp / "dist" / "PROVISInferenceNodeInstaller.exe"
        if not built.is_file():
            raise RuntimeError("PyInstaller did not produce the expected installer executable.")
        shutil.copy2(built, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="local unsigned installer candidate path")
    args = parser.parse_args(argv)
    print(build(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
