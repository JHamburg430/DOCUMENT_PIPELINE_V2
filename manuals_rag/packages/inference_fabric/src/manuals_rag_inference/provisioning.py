"""Pinned, staged Ollama provisioning primitives.

This module deliberately does not install packages, invoke a shell, or restart
services by itself. A separately installed root-owned wrapper may call
``stage_artifact`` and, after an explicit maintenance approval, provide a
service controller for activation. The node agent must not expose this API as
an arbitrary remote command.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_PLATFORM_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
_SAFE_MEMBER_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class ProvisioningManifest:
    version: str
    platform: str
    artifact_url: str
    sha256: str
    binary_member: str = "bin/ollama"
    rollback_version: str | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "ProvisioningManifest":
        required = ("version", "platform", "artifact_url", "sha256")
        missing = [key for key in required if not str(payload.get(key) or "").strip()]
        if missing:
            raise ValueError(f"manifest missing required fields: {', '.join(missing)}")
        manifest = cls(
            version=str(payload["version"]).strip(),
            platform=str(payload["platform"]).strip(),
            artifact_url=str(payload["artifact_url"]).strip(),
            sha256=str(payload["sha256"]).strip().lower(),
            binary_member=str(payload.get("binary_member") or "bin/ollama").strip(),
            rollback_version=(str(payload["rollback_version"]).strip() if payload.get("rollback_version") else None),
        )
        manifest.validate()
        return manifest

    def validate(self, *, allowed_hosts: Iterable[str] = ("ollama.com",)) -> None:
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError("manifest version is invalid")
        if not _PLATFORM_PATTERN.fullmatch(self.platform):
            raise ValueError("manifest platform is invalid")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("manifest sha256 must be a 64-character hexadecimal digest")
        parsed = urlparse(self.artifact_url)
        allowed = {str(host).lower().lstrip(".") for host in allowed_hosts if str(host).strip()}
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or not hostname or hostname not in allowed:
            raise ValueError("artifact_url must use HTTPS and an approved artifact host")
        if not self.binary_member or self.binary_member.startswith("/") or "\\" in self.binary_member:
            raise ValueError("binary_member must be a relative POSIX path")
        if not _SAFE_MEMBER_PATTERN.fullmatch(self.binary_member) or ".." in self.binary_member.split("/"):
            raise ValueError("binary_member contains unsafe path components")
        if self.rollback_version is not None and not _VERSION_PATTERN.fullmatch(self.rollback_version):
            raise ValueError("rollback_version is invalid")


class ServiceController(Protocol):
    def restart(self) -> None: ...

    def healthcheck(self) -> bool: ...


class SystemdOllamaController:
    """Fixed-argument controller for an explicitly approved local service."""

    _SERVICE_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")

    def __init__(self, service_name: str = "ollama.service", *, health_url: str = "http://127.0.0.1:11434/api/version") -> None:
        if not self._SERVICE_PATTERN.fullmatch(service_name):
            raise ValueError("service_name contains unsafe characters")
        parsed = urlparse(health_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("health_url must target the local Ollama API")
        self.service_name = service_name
        self.health_url = health_url

    def restart(self) -> None:
        subprocess.run(
            ["systemctl", "restart", self.service_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def healthcheck(self) -> bool:
        try:
            with urlopen(self.health_url, timeout=5) as response:
                return 200 <= int(response.status) < 300
        except Exception:
            return False


def _download(url: str, destination: Path, *, max_bytes: int, timeout_seconds: float) -> int:
    request = Request(url, headers={"User-Agent": "manuals-rag-inference-provisioner/1"})
    total = 0
    with urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as output:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("artifact exceeds configured size limit")
        while True:
            block = response.read(min(1024 * 1024, max_bytes - total + 1))
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError("artifact exceeds configured size limit")
            output.write(block)
    return total


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if not hmac.compare_digest(digest.hexdigest(), expected.lower()):
        raise ValueError("artifact SHA-256 does not match the approved manifest")


def _extract_binary(archive: Path, destination: Path, *, binary_member: str, max_binary_bytes: int) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, mode="r:*") as bundle:
        member = bundle.getmember(binary_member)
        if not member.isfile() or member.issym() or member.islnk():
            raise ValueError("approved binary member must be a regular file")
        if member.size > max_binary_bytes:
            raise ValueError("binary member exceeds configured size limit")
        extracted = bundle.extractfile(member)
        if extracted is None:
            raise ValueError("approved binary member could not be read")
        target = destination / "bin" / "ollama"
        target.parent.mkdir(parents=True, exist_ok=True)
        with extracted, target.open("wb") as output:
            shutil.copyfileobj(extracted, output, length=1024 * 1024)
        target.chmod(0o755)
        return target


def stage_artifact(
    manifest: ProvisioningManifest,
    *,
    install_root: str | Path,
    downloader: Callable[..., int] | None = None,
    max_artifact_bytes: int = 2 * 1024 * 1024 * 1024,
    max_binary_bytes: int = 512 * 1024 * 1024,
    timeout_seconds: float = 120.0,
) -> dict[str, str | int]:
    """Download, verify, and stage one version without activating it."""
    manifest.validate()
    if max_artifact_bytes <= 0 or max_binary_bytes <= 0:
        raise ValueError("artifact size limits must be positive")
    root = Path(install_root).expanduser()
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    final_dir = versions / manifest.version
    if final_dir.exists():
        raise FileExistsError(f"version is already staged: {manifest.version}")
    temporary_root = Path(tempfile.mkdtemp(prefix=f".ollama-{manifest.version}-", dir=str(versions)))
    archive = temporary_root / "artifact"
    temporary_version = temporary_root / "version"
    try:
        fetch = downloader or _download
        size = fetch(
            manifest.artifact_url,
            archive,
            max_bytes=max_artifact_bytes,
            timeout_seconds=timeout_seconds,
        )
        _verify_sha256(archive, manifest.sha256)
        binary = _extract_binary(
            archive,
            temporary_version,
            binary_member=manifest.binary_member,
            max_binary_bytes=max_binary_bytes,
        )
        metadata = temporary_version / "MANIFEST.json"
        metadata.write_text(
            json.dumps(
                {
                    "version": manifest.version,
                    "platform": manifest.platform,
                    "sha256": manifest.sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_version, final_dir)
        return {"version": manifest.version, "path": str(final_dir), "binary": str(final_dir / binary.relative_to(temporary_version)), "bytes": int(size)}
    except Exception:
        shutil.rmtree(temporary_version, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def activate_staged(
    manifest: ProvisioningManifest,
    *,
    install_root: str | Path,
    controller: ServiceController,
    maintenance_approved: bool,
    active_requests: int = 0,
) -> dict[str, str | None]:
    """Atomically activate a staged version and roll back if health fails.

    ``controller`` is deliberately injected so the node agent cannot gain
    arbitrary service-control capability. A production wrapper must construct
    a fixed controller only after its local maintenance approval gate passes.
    """
    manifest.validate()
    if not maintenance_approved:
        raise PermissionError("explicit maintenance approval is required")
    if int(active_requests) != 0:
        raise RuntimeError("cannot activate Ollama while inference requests are active")
    root = Path(install_root).expanduser()
    staged = root / "versions" / manifest.version
    binary = staged / "bin" / "ollama"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"staged Ollama version is incomplete: {manifest.version}")
    current = root / "current"
    previous_target: str | None = None
    if current.exists() or current.is_symlink():
        if not current.is_symlink():
            raise RuntimeError("current Ollama selection must be a symbolic link")
        previous_target = os.readlink(current)
    temporary_link = root / f".current-{os.getpid()}-{next(tempfile._get_candidate_names())}"
    try:
        temporary_link.symlink_to(Path("versions") / manifest.version)
        os.replace(temporary_link, current)
        controller.restart()
        if not controller.healthcheck():
            raise RuntimeError("Ollama health check failed after activation")
        return {"active_version": manifest.version, "previous_target": previous_target}
    except Exception as error:
        try:
            if previous_target is None:
                current.unlink(missing_ok=True)
            else:
                rollback_link = root / f".rollback-{os.getpid()}-{next(tempfile._get_candidate_names())}"
                rollback_link.symlink_to(previous_target)
                os.replace(rollback_link, current)
            controller.restart()
        except Exception as rollback_error:
            raise RuntimeError(f"activation failed and rollback failed: {rollback_error}") from error
        raise
    finally:
        temporary_link.unlink(missing_ok=True)
