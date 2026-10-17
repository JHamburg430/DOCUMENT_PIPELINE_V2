#!/usr/bin/env python3
"""Build a portable PROVIS Inference Node bundle for Windows or Linux hosts.

The bundle contains only the standard-library node client and setup helpers.
It deliberately contains no pairing token, gateway credential, Docker access,
Tailscale key, or repository secrets.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "manuals_rag/packages/inference_fabric/src/manuals_rag_inference"
SOURCE_ROOT = REPO_ROOT / "manuals_rag/scripts"

LINUX_SETUP = '''#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT/src"
export PROVIS_NODE_BOOTSTRAP_CONFIG="$ROOT/provis-node-config.json"
exec python3 "$ROOT/src/run_inference_node_client.py"
'''


WINDOWS_SETUP = r'''param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Get-Command pyw -ErrorAction SilentlyContinue
$Args = @("-3", (Join-Path $Root "src\run_inference_node_client.py"))
if (-not $Python) {
    $Python = Get-Command pythonw -ErrorAction SilentlyContinue
    $Args = @((Join-Path $Root "src\run_inference_node_client.py"))
}
if (-not $Python) {
    $Python = Get-Command py -ErrorAction SilentlyContinue
    $Args = @("-3", (Join-Path $Root "src\run_inference_node_client.py"))
}
if (-not $Python) {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    $Args = @((Join-Path $Root "src\run_inference_node_client.py"))
}
if (-not $Python) { throw "Python 3.10 or newer is required. Install it from python.org, then run this file again." }
$env:PYTHONPATH = Join-Path $Root "src"
$env:INFERENCE_NODE_CONFIG_PATH = Join-Path $env:LOCALAPPDATA "ProVIS\inference-node.json"
$env:PROVIS_NODE_BOOTSTRAP_CONFIG = Join-Path $Root "provis-node-config.json"
Write-Host "Starting the PROVIS Inference Node desktop client"
Write-Host "Microsoft 365 sign-in is used for enrollment; tokens are not stored by this bundle."
& $Python.Source @Args
if ($LASTEXITCODE) { exit $LASTEXITCODE }
'''


WINDOWS_AGENT = r'''$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Get-Command py -ErrorAction SilentlyContinue
$Args = @("-3", (Join-Path $Root "src\run_inference_node_agent.py"))
if (-not $Python) {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    $Args = @((Join-Path $Root "src\run_inference_node_agent.py"))
}
if (-not $Python) { throw "Python 3.10 or newer is required." }
$env:PYTHONPATH = Join-Path $Root "src"
$env:INFERENCE_NODE_CONFIG_PATH = Join-Path $env:LOCALAPPDATA "ProVIS\inference-node.json"
& $Python.Source @Args
exit $LASTEXITCODE
'''


WINDOWS_SETUP_CMD = r'''@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-node-setup.ps1"
pause
'''


WINDOWS_AGENT_CMD = r'''@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-node-agent.ps1"
pause
'''


WINDOWS_EXE_BUILD = r'''$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ManagedConfig = Join-Path $Root "provis-node-config.json"
if (-not (Test-Path $ManagedConfig -PathType Leaf)) { throw "Managed configuration is missing: $ManagedConfig" }
$Managed = Get-Content -Raw -LiteralPath $ManagedConfig | ConvertFrom-Json
$GatewayUri = [Uri]([string]$Managed.gateway_url)
if ($GatewayUri.Scheme -ne "https" -or -not $GatewayUri.Host -or $GatewayUri.UserInfo -or $GatewayUri.Query -or $GatewayUri.Fragment) {
    throw "Managed gateway_url must be a valid HTTPS URL without credentials, query, or fragment."
}
$env:PYTHONHOME = $null
$env:PYTHONPATH = $null
$Candidates = @()
$PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($PyLauncher) { $Candidates += ,@($PyLauncher.Source, "-3") }
$PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if ($PythonCommand) { $Candidates += ,@($PythonCommand.Source) }
$Python3Command = Get-Command python3.exe -ErrorAction SilentlyContinue
if ($Python3Command) { $Candidates += ,@($Python3Command.Source) }
foreach ($RootPath in @(
    (Join-Path $env:LocalAppData "Programs\Python"),
    (Join-Path $env:ProgramFiles "Python"),
    (Join-Path ${env:ProgramFiles(x86)} "Python")
)) {
    if (Test-Path $RootPath) {
        Get-ChildItem -Path $RootPath -Filter python.exe -File -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object { $Candidates += ,@($_.FullName) }
    }
}
$Python = $null
$PythonArgs = @()
foreach ($Candidate in $Candidates) {
    $CandidateExe = $Candidate[0]
    $CandidateArgs = @($Candidate | Select-Object -Skip 1)
    $CandidateExit = 1
    $ProbeOutput = ""
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $ProbeOutput = (& $CandidateExe @CandidateArgs -c "import sys, pip, tkinter; assert sys.version_info >= (3,10); print(sys.executable)" 2>$null | Out-String).Trim()
        $CandidateExit = $LASTEXITCODE
    } catch {
        $CandidateExit = 1
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($CandidateExit -eq 0 -and $ProbeOutput -match "(?i)\\python(?:w)?\.exe$") {
        $Python = $ProbeOutput
        $PythonArgs = @()
        break
    }
}
if (-not $Python) {
    throw "No complete Python 3.10+ installation with pip and Tkinter was found. Install the full Python distribution from python.org (not the embeddable package), then rerun the builder."
}
Write-Host "Using Python: $Python"
& $Python @PythonArgs -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the pinned build-only PyInstaller tool from PyPI..."
    & $Python @PythonArgs -m pip install --user "pyinstaller>=6.11,<7"
}
& $Python @PythonArgs -m PyInstaller --noconfirm --clean --onefile --noconsole `
    --name PROVISInferenceNode `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $Root "build") `
    --specpath $Root `
    --paths (Join-Path $Root "src") `
    --add-data ((Join-Path $Root "provis-node-config.json") + ";.") `
    (Join-Path $Root "src\run_inference_node_client.py")
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$Output = Join-Path $Root "dist\PROVISInferenceNode.exe"
if (-not (Test-Path $Output -PathType Leaf)) {
    throw "PyInstaller completed without creating the expected executable: $Output"
}
Write-Host "Created: $Output"
'''


README = r'''# PROVIS Inference Node

This is a portable node client for the internal, Tailscale-only PROVIS
inference fabric. It serves all approved inference workloads, not just RAG.
It uses Python's standard library and does not install packages at runtime.

## Requirements

- The machine is already joined to the organization's Tailscale network.
- Ollama is installed and running locally at `http://127.0.0.1:11434`.
- Python 3.10 or newer is installed on Windows (the `py` launcher is preferred).
- The administrator has built this bundle with the managed PROVIS gateway URL.
- The PROVIS gateway has an Entra app registration configured for device-code
  sign-in.

Microsoft 365/Entra credentials are used only to approve enrollment. They are
not embedded in this bundle, stored in the node configuration, or used as the
node's permanent machine secret.

## Windows quick start

1. Extract this folder outside a temporary download directory.
2. Double-click `start-node-setup.cmd`, or run
   `start-node-setup.ps1` from PowerShell.
3. Click **Sign in with Microsoft 365**. The app discovers the tenant, public
   client, API scope, node identity, Tailscale endpoint, and approved
   application policy automatically, then opens Microsoft's device sign-in
   page.
4. Use **Resume**, **Pause**, and **Disconnect** in the tray panel.
   The panel shows events and GPU utilization only; no prompts, responses,
   model names, credentials, or raw error text are displayed.
5. Choose **Dark mode** or **Light mode** in the header. The choice is saved
   separately from enrollment credentials. Close hides the panel to the tray;
   relaunching opens the existing node. Use **Quit** to disconnect and exit.

## Linux desktop quick start

Use the same managed bundle and run `sh start-node-setup.sh`, or launch the
administrator-built `inference-node.pyz` with `python3 inference-node.pyz`.
Both use the same Microsoft sign-in, code Copy button, appearance controls,
privacy-safe events, GPU utilization and safe sharing controls as Windows.

On Ubuntu/Debian the desktop dependencies are `python3-tk`, `python3-gi`,
`gir1.2-gtk-3.0`, and `gir1.2-ayatanaappindicator3-0.1` (installed by your
administrator). The tray uses the desktop's StatusNotifier/AppIndicator
support. On GNOME this may require its AppIndicator extension. Use the icon's
**Open PROVIS** menu to open the panel. If the desktop cannot provide a tray,
the window stays visible and Close uses the safe Quit flow.

Build the managed Linux zipapp from the repository:

```bash
python3 manuals_rag/scripts/build_inference_node_app.py \
  --gateway-url https://<managed-provis-gateway> \
  --output dist/inference-node.pyz
```

No root service, browser credentials or node credentials are included. Run the
desktop app as the employee, not with sudo. The existing headless node service
is separate; do not run it concurrently on the same node port.

To create a standalone Windows executable, run `build-node-exe.ps1` once from
the extracted folder. It installs only the build-time PyInstaller tool and
creates `dist\PROVISInferenceNode.exe`; the resulting app does not need
Python at runtime.

The builder clears stale `PYTHONHOME`/`PYTHONPATH` values and skips broken or
embeddable Python installations. If it reports that no complete Python was
found, install the full Python distribution from python.org with pip and
Tkinter, then rerun the builder.

The client automatically proposes the machine's Tailscale IPv4 address as the
advertised node endpoint. Keep that value unless the gateway administrator
gave you a different tailnet address. Do not expose port 8765 publicly.

The administrator embeds the verified HTTPS gateway URL when building the
bundle. The builder rejects empty, non-HTTPS, credential-bearing, query-bearing,
or fragment-bearing URLs:

```bash
python3 manuals_rag/scripts/build_inference_node_bundle.py \
  --gateway-url https://<managed-provis-gateway> \
  --output /tmp/provis-inference-node.zip
```

The source launcher contract is an explicit `PROVIS_NODE_BOOTSTRAP_CONFIG`
path, followed by the bundle's root `provis-node-config.json`; it never uses
the current working directory. The installed executable can use an
administrator-controlled `provis-node-config.json` beside the executable, and
the one-file executable loads the same nonsecret JSON embedded with
PyInstaller. A managed executable therefore works without Python, developer
environment variables, or a special working directory.

Employees do not enter the gateway URL, tenant ID, client ID, scope, node
name, or application allowlist. If the bundle has no managed gateway URL, it
fails closed and asks for the managed installer.

## Safety boundaries

- No Microsoft token, pairing token, or node secret is included in this archive.
- The node exposes only a signed inference/command proxy and a read-only
  snapshot endpoint; it does not expose the Docker socket.
- The gateway controls model catalog admission, application allowlists,
  signed model pulls, and safe eviction.
- If the gateway is unavailable, setup stops rather than creating a partial
  production registration.
'''


def _write_text(root: Path, relative: str, content: str) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")


def _validate_managed_gateway_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("--gateway-url must be a non-empty HTTPS URL with a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("--gateway-url must not contain credentials, query parameters, or a fragment")
    return parsed.geturl().rstrip("/")


def build(output: Path, *, gateway_url: str = "") -> Path:
    gateway_url = _validate_managed_gateway_url(gateway_url)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="provis-inference-node-") as temporary:
        stage = Path(temporary) / "provis-inference-node"
        source = stage / "src"
        source.mkdir(parents=True)
        shutil.copytree(
            PACKAGE_ROOT,
            source / "manuals_rag_inference",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in ("run_inference_node_client.py", "run_inference_node_agent.py", "provis_agent_link.py"):
            shutil.copy2(SOURCE_ROOT / name, source / name)
        _write_text(stage, "start-node-setup.ps1", WINDOWS_SETUP)
        _write_text(stage, "start-node-agent.ps1", WINDOWS_AGENT)
        _write_text(stage, "start-node-setup.cmd", WINDOWS_SETUP_CMD)
        _write_text(stage, "start-node-agent.cmd", WINDOWS_AGENT_CMD)
        _write_text(stage, "build-node-exe.ps1", WINDOWS_EXE_BUILD)
        _write_text(stage, "start-node-setup.sh", LINUX_SETUP)
        (stage / 'start-node-setup.sh').chmod(0o755)
        _write_text(stage, "provis-node-config.json", json.dumps({"gateway_url": gateway_url}, indent=2) + "\n")
        _write_text(stage, "README.md", README)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(stage.rglob("*")):
                if item.is_file() and item.suffix != ".pyc" and "__pycache__" not in item.parts:
                    archive.write(item, item.relative_to(stage.parent).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "manuals_rag/dist/provis-inference-node.zip",
    )
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("PROVIS_INFERENCE_GATEWAY_URL", ""),
        help="Managed gateway URL embedded for employees; tenant/client/scope are discovered from the gateway",
    )
    args = parser.parse_args()
    print(build(args.output, gateway_url=args.gateway_url))


if __name__ == "__main__":
    main()
