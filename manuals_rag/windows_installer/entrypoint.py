from pathlib import Path
import sys

from provis_node_windows_installer import main


def _icon_bytes() -> bytes:
    if getattr(sys, "frozen", False):
        return (Path(sys._MEIPASS) / "provis-lens.png").read_bytes()
    return (Path(__file__).resolve().parents[1] / "assets" / "brand" / "provis-lens.png").read_bytes()


if __name__ == "__main__":
    raise SystemExit(main(icon_png=_icon_bytes()))
