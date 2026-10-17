#!/usr/bin/env python3
"""Build the standalone PROVIS Inference Node distribution.

This builder deliberately lives outside ``manuals_rag``. Its output contains
only the node runtime, standard-library launchers, and a nonsecret managed
gateway URL. It never copies RAG packages, credentials, or the Docker socket.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


LINUX_SETUP = r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PROVIS_NODE_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys, tkinter; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3.10+ with Tkinter is required. Install the distribution Tk package, then run again." >&2
    exit 1
fi
CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME:?HOME is required}/.config}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export INFERENCE_NODE_CONFIG_PATH="$CONFIG_HOME/ProVIS/inference-node.json"
export PROVIS_NODE_BOOTSTRAP_CONFIG="$ROOT/provis-node-config.json"
exec "$PYTHON_BIN" "$ROOT/src/run_node_client.py" "$@"
'''


LINUX_AGENT = r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PROVIS_NODE_PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python 3.10+ is required." >&2
    exit 1
fi
CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME:?HOME is required}/.config}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export INFERENCE_NODE_CONFIG_PATH="$CONFIG_HOME/ProVIS/inference-node.json"
export PROVIS_NODE_BOOTSTRAP_CONFIG="$ROOT/provis-node-config.json"
exec "$PYTHON_BIN" "$ROOT/src/run_node_agent.py" "$@"
'''


LINUX_BUILD = r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG="$ROOT/provis-node-config.json"
PYTHON_BIN="${PROVIS_NODE_PYTHON:-python3}"
"$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys
from urllib.parse import urlparse
with open(sys.argv[1], encoding="utf-8") as stream:
    url = urlparse(str(json.load(stream).get("gateway_url", "")).strip())
if url.scheme.lower() != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
    raise SystemExit("Managed gateway_url must be a valid HTTPS URL without credentials, query, or fragment.")
PY
if ! "$PYTHON_BIN" -c 'import sys, tkinter; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
    echo "Python 3.10+ with Tkinter is required to build the Linux desktop executable." >&2
    exit 1
fi
if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
    echo "PyInstaller is not installed. Install the build-only tool with:" >&2
    echo "  $PYTHON_BIN -m pip install --user 'pyinstaller>=6.11,<7'" >&2
    exit 1
fi
"$PYTHON_BIN" -m PyInstaller --noconfirm --clean --onefile --windowed \
    --name PROVISInferenceNode --distpath "$ROOT/dist" --workpath "$ROOT/build" \
    --specpath "$ROOT" --paths "$ROOT/src" --add-data "$CONFIG:." \
    "$ROOT/src/run_node_client.py"
test -f "$ROOT/dist/PROVISInferenceNode"
chmod 0755 "$ROOT/dist/PROVISInferenceNode"
echo "Created $ROOT/dist/PROVISInferenceNode"
'''


WINDOWS_SETUP = r'''param()
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Get-Command pyw -ErrorAction SilentlyContinue
$Args = @("-3", (Join-Path $Root "src\run_node_client.py"))
if (-not $Python) { $Python = Get-Command pythonw -ErrorAction SilentlyContinue; $Args = @((Join-Path $Root "src\run_node_client.py")) }
if (-not $Python) { throw "Python 3.10+ with Tkinter is required." }
$env:PYTHONPATH = Join-Path $Root "src"
$env:INFERENCE_NODE_CONFIG_PATH = Join-Path $env:LOCALAPPDATA "ProVIS\inference-node.json"
$env:PROVIS_NODE_BOOTSTRAP_CONFIG = Join-Path $Root "provis-node-config.json"
& $Python.Source @Args
exit $LASTEXITCODE
'''


README = r'''# PROVIS Inference Node

This is a standalone internal inference-node application. It is intentionally
separate from Manuals RAG: its runtime imports only Python standard-library
modules and its own `provis_inference_node` package. It can serve RAG,
employee-agent, OpenClaw, and other approved applications through the central
gateway.

## Linux

Requirements: Tailscale membership, local Ollama, Python 3.10+, and the Linux
distribution's Tkinter package (`python3-tk` on Debian/Ubuntu). Extract the
managed archive and run:

```bash
./start-node-setup.sh
```

The launcher resolves its root from the script location, not the current
working directory. It discovers the managed HTTPS gateway from the bundled
`provis-node-config.json`, stores only enrolled node state under
`$XDG_CONFIG_HOME/ProVIS/inference-node.json` or `$HOME/.config/ProVIS`, and
opens the Microsoft 365 device sign-in flow. Employees do not enter gateway,
Entra, or pairing settings.

To create a Linux-only one-file executable on the target Linux machine, install
the build-only tool and run `./build-node-linux.sh`:

```bash
python3 -m pip install --user 'pyinstaller>=6.11,<7'
./build-node-linux.sh
```

The result is `dist/PROVISInferenceNode`. PyInstaller outputs are OS-specific;
build Linux artifacts on Linux and Windows artifacts on Windows.

## Windows

`start-node-setup.ps1` remains available for the already-proven Windows flow.
The same managed configuration contract is used, with per-user state under
`%LOCALAPPDATA%\ProVIS\inference-node.json`.

## Security and dependency boundary

The archive contains no access token, pairing token, node secret, model
credentials, or Docker socket. The gateway URL is the only managed value
embedded, and the builder rejects empty or non-HTTPS URLs. Runtime dependencies
are empty; `requirements-dev.txt` is only for local tests and packaging.
'''


def _validate_gateway_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("--gateway-url must be a non-empty HTTPS URL with a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("--gateway-url must not contain credentials, query parameters, or a fragment")
    return parsed.geturl().rstrip("/")


def _write(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(mode)


def _add(archive: zipfile.ZipFile, path: Path, name: str) -> None:
    archive.write(path, name)
    archive.getinfo(name).external_attr = (path.stat().st_mode & 0xFFFF) << 16


def build(output: Path, *, gateway_url: str) -> Path:
    gateway_url = _validate_gateway_url(gateway_url)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="provis-node-") as temporary:
        stage = Path(temporary) / "provis-inference-node"
        shutil.copytree(SOURCE, stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        _write(stage / "start-node-setup.sh", LINUX_SETUP, 0o755)
        _write(stage / "start-node-agent.sh", LINUX_AGENT, 0o755)
        _write(stage / "build-node-linux.sh", LINUX_BUILD, 0o755)
        _write(stage / "start-node-setup.ps1", WINDOWS_SETUP)
        _write(stage / "provis-node-config.json", json.dumps({"gateway_url": gateway_url}, indent=2) + "\n")
        _write(stage / "README.md", README)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(stage.rglob("*")):
                if item.is_file():
                    _add(archive, item, item.relative_to(stage.parent).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default=os.getenv("PROVIS_INFERENCE_GATEWAY_URL", ""))
    parser.add_argument("--output", type=Path, default=ROOT / "dist/provis-inference-node.zip")
    args = parser.parse_args()
    print(build(args.output, gateway_url=args.gateway_url))


if __name__ == "__main__":
    main()
