#!/usr/bin/env python3
"""Build an unsigned, standalone Windows per-user installer candidate.

This script packages only the installer. It does not sign or publish output.
"""

from __future__ import annotations

import argparse
import ast
import json
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


def stage_release(target: Path, document: dict, keys: dict, gateway: str) -> tuple[Path, Path]:
    """Stage approved public bindings without modifying checked-out runtime code."""
    sys.path.insert(0,str(SOURCE_ROOT))
    from manuals_rag_inference.node_release import verify_release, canonical
    from provis_release_pipeline import public_pins
    public_pins(keys)
    verify_release(document,keys,platform='windows-x86_64')
    from urllib.parse import urlsplit
    parsed=urlsplit(gateway)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError('Approved HTTPS gateway required')
    shutil.copytree(INSTALLER_ROOT,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    module=target/'provis_node_windows_installer.py'
    text=module.read_text(encoding='utf-8');lines=text.splitlines(keepends=True)
    replacements={'PUBLISHER':document['publisher'],'PUBLIC_KEYS':keys,'RELEASE_DOCUMENT':document,'GATEWAY_BASE':gateway}
    assignments={node.targets[0].id:node for node in ast.parse(text).body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements}
    if set(assignments)!=set(replacements):raise ValueError('Installer binding source contract changed')
    for name,node in sorted(assignments.items(),key=lambda item:item[1].lineno,reverse=True):lines[node.lineno-1:node.end_lineno]=[name+' = '+repr(replacements[name])+'\n']
    updated=''.join(lines);ast.parse(updated);module.write_text(updated,encoding='utf-8')
    binding=target/'provis-installer-binding.json'
    binding.write_bytes(canonical({'schema':1,'app_release':document,'gateway_url':gateway}))
    return target/'entrypoint.py',binding


def build(output: Path, *, python: str | None = None, runner=subprocess.run, app_release=None, public_keys=None, gateway_url=None) -> Path:
    if os.name != "nt":
        raise RuntimeError("The standalone Windows installer can only be packaged on Windows.")
    if not ICON_PATH.is_file():
        raise FileNotFoundError(ICON_PATH)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    if (app_release is None)!=(public_keys is None):raise ValueError('App manifest and trusted public keys must be supplied together')
    if app_release is None:
        sys.path.insert(0,str(INSTALLER_ROOT))
        from provis_node_windows_installer import RELEASE_DOCUMENT, PUBLIC_KEYS, GATEWAY_BASE
        document,keys,gateway=RELEASE_DOCUMENT,PUBLIC_KEYS,gateway_url or GATEWAY_BASE
    else:
        document=json.loads(Path(app_release).read_text(encoding='utf-8'))
        keys=json.loads(Path(public_keys).read_text(encoding='utf-8'))
        if not gateway_url:raise ValueError('Approved gateway URL required for release build')
        gateway=gateway_url
    with tempfile.TemporaryDirectory(prefix="provis-node-installer-build-") as temp_name:
        temp = Path(temp_name)
        staged_entry,binding=stage_release(temp/'installer-source',document,keys,gateway)
        command = [
            python, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
            "--name", "PROVISInferenceNodeInstaller",
            "--distpath", str(temp / "dist"),
            "--workpath", str(temp / "work"),
            "--specpath", str(temp / "spec"),
            "--paths", str(SOURCE_ROOT),
            "--paths", str(staged_entry.parent),
            "--hidden-import", "provis_node_windows_installer_gui",
            "--add-data", f"{ICON_PATH};.",
            "--add-data", f"{binding};.",
            "--collect-all", "cryptography",
            str(staged_entry),
        ]
        result = runner(command, cwd=REPO_ROOT, check=False)
        if result.returncode:
            raise RuntimeError(f"PyInstaller failed with exit code {result.returncode}.")
        built = temp / "dist" / "PROVISInferenceNodeInstaller.exe"
        if not built.is_file():
            raise RuntimeError("PyInstaller did not produce the expected installer executable.")
        from provis_release_pipeline import frozen_json
        embedded=frozen_json(built.read_bytes(),'provis-installer-binding.json')
        if embedded!={'schema':1,'app_release':document,'gateway_url':gateway}:raise RuntimeError('Frozen installer binding mismatch')
        shutil.copy2(built, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="local unsigned installer candidate path")
    parser.add_argument('--app-release',type=Path,help='Approved signed Windows app manifest')
    parser.add_argument('--public-keys',type=Path,help='Externally trusted public verification pins')
    parser.add_argument('--gateway-url',help='Approved managed HTTPS gateway URL')
    args = parser.parse_args(argv)
    print(build(args.output,app_release=args.app_release,public_keys=args.public_keys,gateway_url=args.gateway_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
