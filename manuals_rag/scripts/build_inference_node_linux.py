#!/usr/bin/env python3
"""Build the self-contained Linux PROVIS Inference Node executable.

PyInstaller embeds the Python runtime, Tk/Tcl runtime, cryptography runtime,
application package, and managed nonsecret configuration. PyInstaller and its build dependencies
are build-machine-only requirements; the resulting executable does not need
Python or Tk installed on the employee's machine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from build_inference_node_bundle import _validate_managed_gateway_url, managed_config


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT / "manuals_rag/packages/inference_fabric/src"
BROKER_PARENT = REPO_ROOT / "provis_inference_node/src"
APP_SOURCE = REPO_ROOT / "manuals_rag/scripts/run_inference_node_client.py"
SOURCE_ROOT = REPO_ROOT / "manuals_rag/scripts"
INSTALLER_ROOT = REPO_ROOT / "manuals_rag/installers"
BRAND_ASSET_ROOT = REPO_ROOT / "manuals_rag/assets/brand"


def _optional_tray_imports() -> list[str]:
    """Include GTK Python modules when the build host has them.

    GTK/AppIndicator shared libraries remain owned by the desktop image.  The
    client catches unavailable tray support and keeps the main Tk panel open;
    this keeps the executable usable across GNOME, KDE and minimal desktops.
    """
    try:
        import gi  # noqa: F401
    except ImportError:
        return []
    return [
        "--hidden-import=gi",
        "--hidden-import=gi.repository",
        "--hidden-import=gi.repository.Gtk",
        "--hidden-import=gi.repository.GLib",
        "--hidden-import=gi.repository.Gio",
    ]


def _optional_onnx_imports() -> list[str]:
    """Embed ONNX Runtime only when the build host installed the optional stack."""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return []
    return [
        "--hidden-import=onnxruntime",
        "--collect-all",
        "onnxruntime",
        "--collect-all",
        "numpy",
    ]


def pyinstaller_args(output: Path, config_path: Path, work_root: Path) -> list[str]:
    """Return the reproducible one-file build command for tests and tooling."""
    separator = os.pathsep
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        output.name,
        "--distpath",
        str(output.parent),
        "--workpath",
        str(work_root / "build"),
        "--specpath",
        str(work_root / "spec"),
        "--paths",
        str(PACKAGE_PARENT),
        "--paths",
        str(SOURCE_ROOT),
        "--paths",
        str(BROKER_PARENT),
        "--add-data",
        f"{config_path}{separator}.",
        "--add-data",
        f"{BRAND_ASSET_ROOT / 'provis-lens.png'}{separator}assets/brand",
        "--collect-all",
        "cryptography",
        *_optional_onnx_imports(),
        *_optional_tray_imports(),
        str(APP_SOURCE),
    ]
    if INSTALLER_ROOT.is_dir():
        args[-1:-1] = ["--add-data", f"{INSTALLER_ROOT}{separator}installers"]
    return args


def build(output: Path, *, gateway_url: str = "", update_config=None) -> Path:
    gateway_url = _validate_managed_gateway_url(gateway_url)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="provis-inference-node-linux-") as temporary:
        root = Path(temporary)
        config_path = root / "provis-node-config.json"
        config_path.write_text(json.dumps(managed_config(gateway_url, update_config)), encoding="utf-8")
        if not (BRAND_ASSET_ROOT / "provis-lens.png").is_file():
            raise RuntimeError("PROVIS Lens icon asset is missing from the source tree")
        try:
            import PyInstaller  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PyInstaller is required only on the build machine; install the pinned build tool "
                "there, then rebuild the Linux artifact"
            ) from exc
        subprocess.run(pyinstaller_args(output, config_path, root), check=True)
    if not output.is_file():
        raise RuntimeError(f"PyInstaller completed without creating {output}")
    output.chmod(0o755)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "manuals_rag/dist/PROVISInferenceNode")
    parser.add_argument("--gateway-url", default=os.getenv("PROVIS_INFERENCE_GATEWAY_URL", ""))
    parser.add_argument("--update-config", type=Path)
    args = parser.parse_args()
    update_config = json.loads(args.update_config.read_text(encoding="utf-8")) if args.update_config else None
    print(build(args.output, gateway_url=args.gateway_url, update_config=update_config))


if __name__ == "__main__":
    main()
