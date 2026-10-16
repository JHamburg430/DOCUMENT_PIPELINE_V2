#!/usr/bin/env python3
"""Build a portable PROVIS Inference Node bundle for Windows or Linux hosts.

The bundle contains only the standard-library node client and setup helpers.
It deliberately contains no pairing token, gateway credential, Docker access,
Tailscale key, or repository secrets.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "manuals_rag/packages/inference_fabric/src/manuals_rag_inference"
SOURCE_ROOT = REPO_ROOT / "manuals_rag/scripts"


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
- The PROVIS gateway has an Entra app registration configured for device-code
  sign-in, including its tenant ID, public client ID, and API scope.

Microsoft 365/Entra credentials are used only to approve enrollment. They are
not embedded in this bundle, stored in the node configuration, or used as the
node's permanent machine secret.

## Windows quick start

1. Extract this folder outside a temporary download directory.
2. Double-click `start-node-setup.cmd`, or run
   `start-node-setup.ps1` from PowerShell.
3. Use **Sign in with Microsoft 365** to enter the gateway URL, tenant ID,
   public client ID, API scope, unique node name, and gateway-approved
   application list. The app opens Microsoft's device sign-in page.
4. Use **Enable inference**, **Disable inference**, and **Disconnect safely**
   from the same window. The activity table shows routing operations, model,
   status, duration, and sanitized error type.

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


def build(output: Path) -> Path:
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
        for name in ("run_inference_node_client.py", "run_inference_node_agent.py"):
            shutil.copy2(SOURCE_ROOT / name, source / name)
        _write_text(stage, "start-node-setup.ps1", WINDOWS_SETUP)
        _write_text(stage, "start-node-agent.ps1", WINDOWS_AGENT)
        _write_text(stage, "start-node-setup.cmd", WINDOWS_SETUP_CMD)
        _write_text(stage, "start-node-agent.cmd", WINDOWS_AGENT_CMD)
        _write_text(stage, "build-node-exe.ps1", WINDOWS_EXE_BUILD)
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
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
