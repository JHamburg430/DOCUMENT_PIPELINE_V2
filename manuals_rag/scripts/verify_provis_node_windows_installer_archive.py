#!/usr/bin/env python3
"""Assert that the standalone installer executable contains its runtime modules."""

from __future__ import annotations

import argparse
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader


REQUIRED_MODULES = {
    "provis_node_windows_installer",
    "provis_node_windows_installer_gui",
    "manuals_rag_inference.node_release",
}
REQUIRED_DATA = {"provis-lens.png", "provis-installer-binding.json"}


def verify(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = CArchiveReader(str(path))
    missing_data = REQUIRED_DATA.difference(archive.toc)
    if missing_data:
        raise RuntimeError(f"Installer archive is missing data files: {sorted(missing_data)}")

    embedded_name = next(
        (name for name, entry in archive.toc.items() if entry[-1] == "z"), None
    )
    if embedded_name is None:
        raise RuntimeError("Installer archive has no embedded Python module archive.")
    modules = CArchiveReader(str(path)).open_embedded_archive(embedded_name).toc
    missing_modules = REQUIRED_MODULES.difference(modules)
    if missing_modules:
        raise RuntimeError(f"Installer archive is missing Python modules: {sorted(missing_modules)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    args = parser.parse_args(argv)
    verify(args.executable)
    print(f"Verified installer archive contents: {args.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
