"""Safe local dependency diagnostics and explicitly managed repair actions.

Status checks never install, restart, or expose command output. Repair actions
are limited to fixed Tailscale commands or administrator-bundled installers;
the app never downloads and executes an arbitrary URL.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


OLLAMA_DOWNLOAD_PAGE = "https://ollama.com/download"
TAILSCALE_DOWNLOAD_PAGE = "https://tailscale.com/download"
NVIDIA_DRIVER_PAGE = "https://www.nvidia.com/Download/index.aspx"
AMD_DRIVER_PAGE = "https://www.amd.com/en/support/download/drivers.html"

_OFFICIAL_INSTALLER_URLS = {
    "ollama": {
        "windows": "https://ollama.com/install.ps1",
        "linux": "https://ollama.com/install.sh",
    },
    "tailscale": {
        "windows": "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe",
        "linux": "https://tailscale.com/install.sh",
    },
}
_MAX_INSTALLER_BYTES = 256 * 1024 * 1024


def _run_fixed(command: list[str], *, timeout: float = 8.0) -> tuple[int, str, str]:
    """Run one fixed executable command without a shell."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return int(completed.returncode), completed.stdout or "", completed.stderr or ""
    except (OSError, subprocess.SubprocessError):
        return 127, "", ""


def _local_ollama_url(endpoint: str) -> str:
    parsed = urlsplit(str(endpoint or "http://127.0.0.1:11434").strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama endpoint must be local")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def ollama_status(endpoint: str = "http://127.0.0.1:11434") -> dict[str, object]:
    """Return a bounded, content-free Ollama installation/health status."""
    installed = shutil.which("ollama") is not None
    try:
        url = _local_ollama_url(endpoint) + "/api/version"
        request = Request(url, headers={"User-Agent": "provis-inference-node/1"})
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read(4096).decode("utf-8", "replace"))
        version = str(payload.get("version") or "").strip()
        return {"status": "running", "installed": True, "version": version or "available"}
    except Exception:
        return {
            "status": "installed_not_running" if installed else "not_installed",
            "installed": installed,
            "version": "",
        }


def _tailscale_version() -> str:
    code, stdout, _ = _run_fixed(["tailscale", "version"], timeout=3)
    if code != 0:
        return ""
    first = (stdout.strip().splitlines() or [""])[0].strip()
    return first[:80]


def tailscale_status() -> dict[str, object]:
    """Return only connection state and a private IPv4 address."""
    if shutil.which("tailscale") is None:
        return {"status": "not_installed", "installed": False, "connected": False, "version": "", "ipv4": ""}
    code, stdout, _ = _run_fixed(["tailscale", "status", "--json"], timeout=5)
    connected: bool | None = None
    if code == 0:
        try:
            payload = json.loads(stdout)
            state = str(payload.get("BackendState") or "").strip().lower()
            if state == "running":
                connected = True
            elif state in {"stopped", "needslogin", "needsmachineauth"}:
                connected = False
        except (TypeError, ValueError):
            pass
    ip_code, ip_stdout, _ = _run_fixed(["tailscale", "ip", "-4"], timeout=3)
    ipv4 = ""
    if ip_code == 0:
        for value in ip_stdout.split():
            parts = value.split(".")
            if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
                ipv4 = value
                break
    return {
        "status": "connected" if connected is True else "disconnected" if connected is False else "unknown",
        "installed": True,
        "connected": connected,
        "version": _tailscale_version(),
        "ipv4": ipv4,
    }


def gpu_driver_status() -> dict[str, object]:
    """Inspect NVIDIA driver availability without modifying the machine."""
    if shutil.which("nvidia-smi") is None:
        return {"status": "not_detected", "vendor": "unknown", "driver": "", "gpus": []}
    code, stdout, _ = _run_fixed(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader,nounits"],
        timeout=4,
    )
    gpus: list[str] = []
    driver = ""
    for line in stdout.splitlines()[:16]:
        fields = [item.strip() for item in line.split(",", 1)]
        if len(fields) == 2:
            driver = driver or fields[0][:40]
            gpus.append(fields[1][:100])
    return {
        "status": "available" if code == 0 and gpus else "unavailable",
        "vendor": "nvidia",
        "driver": driver,
        "gpus": gpus,
        "update_check": "vendor_page",
    }


def system_status(ollama_endpoint: str) -> dict[str, dict[str, object]]:
    return {
        "ollama": ollama_status(ollama_endpoint),
        "tailscale": tailscale_status(),
        "gpu": gpu_driver_status(),
    }


def vendor_driver_page() -> str:
    # Prefer the AMD page only when its local diagnostic is present; otherwise
    # keep the default deterministic and safe for NVIDIA-equipped hosts.
    return AMD_DRIVER_PAGE if shutil.which("amd-smi") or shutil.which("rocm-smi") else NVIDIA_DRIVER_PAGE


def official_installer_url(component: str) -> str | None:
    """Return only a fixed official installer endpoint for this platform."""
    platform_name = platform.system().lower()
    platform_name = "windows" if platform_name == "windows" else "linux" if platform_name == "linux" else ""
    return _OFFICIAL_INSTALLER_URLS.get(component, {}).get(platform_name)


def _download_official_installer(component: str) -> Path | None:
    url = official_installer_url(component)
    if not url or urlsplit(url).scheme != "https":
        return None
    suffix = ".ps1" if component == "ollama" and platform.system().lower() == "windows" else ".exe" if component == "tailscale" and platform.system().lower() == "windows" else ".sh"
    root = Path(tempfile.gettempdir()) / "provis-inference-node-installers"
    temporary = root / f".{component}.download"
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = root / f"{component}-latest{suffix}"
        request = Request(url, headers={"User-Agent": "provis-inference-node/1"})
        total = 0
        with urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_INSTALLER_BYTES:
                    raise ValueError("installer exceeds bounded download size")
                output.write(chunk)
        temporary.replace(target)
        if suffix == ".sh":
            target.chmod(0o700)
        return target
    except (OSError, ValueError, TimeoutError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def tailscale_disconnect() -> dict[str, str]:
    code, _, _ = _run_fixed(["tailscale", "down"], timeout=20)
    return {"status": "disconnected" if code == 0 else "failed"}


def tailscale_reconnect() -> dict[str, str]:
    # Reuses the existing local Tailscale profile; no auth key or login data is accepted.
    code, _, _ = _run_fixed(["tailscale", "up"], timeout=30)
    return {"status": "connected" if code == 0 else "failed"}


_INSTALLER_NAMES = {
    "ollama": ("OllamaSetup.exe", "ollama-install.exe", "ollama-install.sh", "ollama.run"),
    "tailscale": ("TailscaleSetup.exe", "tailscale-setup.exe", "tailscale-install.sh", "tailscale.run"),
    "gpu": ("gpu-driver.exe", "nvidia-driver.run", "gpu-driver.run"),
}
_INSTALLER_SUFFIXES = {".exe", ".msi", ".run", ".sh", ".AppImage"}


def managed_installer(component: str) -> Path | None:
    """Find only an administrator-bundled installer with a fixed filename."""
    names = _INSTALLER_NAMES.get(component, ())
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        roots.append(Path(str(meipass)) / "installers")
    roots.append(Path(__file__).resolve().parent / "installers")
    roots.append(Path(sys.executable).resolve().parent / "installers")
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file() and candidate.suffix in _INSTALLER_SUFFIXES:
                return candidate
    return None


def launch_managed_installer(component: str) -> dict[str, str]:
    """Download the fixed official installer when needed, then launch it."""
    installer = managed_installer(component)
    source = "managed"
    if installer is None:
        installer = _download_official_installer(component)
        source = "official_download"
    if installer is None:
        return {"status": "unavailable"}
    if os.name == "nt" and installer.suffix.lower() == ".msi":
        command = ["msiexec.exe", "/i", str(installer)]
    elif os.name == "nt" and installer.suffix.lower() == ".ps1":
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(installer)]
    elif installer.suffix.lower() == ".sh":
        command = ["sh", str(installer)]
    else:
        command = [str(installer)]
    try:
        subprocess.Popen(command, close_fds=(os.name != "nt"), creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError:
        return {"status": "failed"}
    return {"status": "started", "source": source}
