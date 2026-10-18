"""Per-user bootstrapper for the pinned, signed PROVIS Inference Node release.

This installer is deliberately separate from the node's managed updater. It
installs only the release pinned below, never calls the owner update API, and
never creates update plans or changes enrollment/configuration data.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from manuals_rag_inference.node_release import (
    MAX_ARTIFACT,
    verify_artifact,
    verify_release,
)


PUBLISHER = "provis-release-20261017"
PUBLIC_KEYS = {
    PUBLISHER: "Ya5jNNCS7S8mfNJCjXZg80zjJVF9h3xT0nk7WQD2Pbw=",
}
RELEASE_DOCUMENT = {
    "publisher": PUBLISHER,
    "signature": "PrW99KJqa0q7jbnfKaUGXjFxFMioEyr3gIPAgtDAJe5rDARZiKWIvj3Fywe00QjcGctM0YhVULVQu27NMZB9DA==",
    "manifest": {
        "format": 1,
        "release_id": "39fd973c-c97a-4b54-97de-938a2f88bd6c",
        "version": "2026.10.17.10",
        "platform": "windows-x86_64",
        "filename": "PROVISInferenceNode.exe",
        "size": 20191457,
        "sha256": "e635207ca0b1d13869523f7b679b0100ba05234222af487c9fadebaa94b914f2",
    },
}
GATEWAY_BASE = "https://provis-openclaw-gateway-01.taile64607.ts.net:8445"
APP_DIR_NAME = "PROVIS"
APP_EXE_NAME = "PROVISInferenceNode.exe"
APP_SHORTCUT_NAME = "PROVIS Inference Node.lnk"
MAX_DOWNLOAD_SECONDS = 300

# Known-folder GUIDs. Never use LOCALAPPDATA or USERPROFILE for installation
# paths: those environment variables may be rewritten by a sandbox launcher.
FOLDER_IDS = {
    "local_app_data": "F1B32785-6FBA-4FCF-9D55-7B8E7F157091",
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "programs": "A77F5D77-2E2B-44C3-A6A2-ABA601054A51",
}


class InstallError(RuntimeError):
    """A safe installer failure that is suitable for displaying to a user."""


@dataclass(frozen=True)
class KnownFolders:
    local_app_data: Path
    desktop: Path
    programs: Path


class ShortcutAdapter(Protocol):
    def read(self, path: Path) -> dict[str, str]: ...

    def write(self, path: Path, *, target: Path, working_directory: Path,
              icon: Path, description: str) -> None: ...


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, value: str) -> "_GUID":
        raw = uuid.UUID(value).bytes_le
        result = cls()
        ctypes.memmove(ctypes.byref(result), raw, len(raw))
        return result


def _known_folder(folder_id: str) -> Path:
    if os.name != "nt":
        raise InstallError("The PROVIS Inference Node installer is Windows-only.")
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    get_path = shell32.SHGetKnownFolderPath
    get_path.argtypes = [ctypes.POINTER(_GUID), ctypes.c_uint32, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_wchar_p)]
    get_path.restype = ctypes.c_long
    free = ole32.CoTaskMemFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    pointer = ctypes.c_wchar_p()
    result = get_path(ctypes.byref(_GUID.parse(folder_id)), 0, None, ctypes.byref(pointer))
    if result != 0 or not pointer.value:
        raise OSError(result, "Windows could not resolve a required per-user folder")
    try:
        return Path(pointer.value)
    finally:
        free(pointer)


def resolve_known_folders(resolver: Callable[[str], Path] = _known_folder) -> KnownFolders:
    """Resolve paths from Windows Known Folders and reject Codex sandbox roots."""
    paths = KnownFolders(**{name: Path(os.path.abspath(resolver(folder_id)))
                            for name, folder_id in FOLDER_IDS.items()})
    for path in (paths.local_app_data, paths.desktop, paths.programs):
        folded = str(path).replace("/", "\\").casefold()
        if "\\appdata\\local\\packages\\sandbox." in folded or "\\appdata\\local\\packages\\openai.codex_" in folded:
            raise InstallError(
                "Run this installer from the normal Windows user desktop, not a Codex sandbox."
            )
    return paths


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise InstallError("The release catalog redirected the download; refusing it.")


def _validate_gateway(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise InstallError("A credential-free HTTPS release catalog URL is required.")
    return base_url.rstrip("/")


def _artifact_url(base_url: str, release_id: str) -> str:
    return f"{_validate_gateway(base_url)}/v1/node-releases/{release_id}"


def _download_artifact(url: str, destination: Path, expected_size: int,
                       *, opener=None, clock=time.monotonic) -> None:
    if not (1 <= expected_size <= MAX_ARTIFACT):
        raise InstallError("The signed release size is outside the supported limit.")
    if opener is None:
        opener = build_opener(_NoRedirect())
    request = Request(url, headers={"Accept": "application/octet-stream"})
    start = clock()
    try:
        with opener.open(request, timeout=30) as response, destination.open("xb") as output:
            if getattr(response, "status", 200) != 200:
                raise InstallError("The release catalog did not return the signed artifact.")
            total = 0
            while True:
                if clock() - start > MAX_DOWNLOAD_SECONDS:
                    raise InstallError("The release download exceeded its time limit.")
                chunk = response.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_size:
                    raise InstallError("The release artifact exceeded its signed size.")
                output.write(chunk)
            if total != expected_size:
                raise InstallError("The release artifact was incomplete.")
    except (HTTPError, URLError, OSError, TimeoutError) as error:
        destination.unlink(missing_ok=True)
        if isinstance(error, InstallError):
            raise
        raise InstallError("The signed release could not be downloaded.") from error
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _png_to_ico(png: bytes) -> bytes:
    """Wrap the packaged PNG in a Vista+ ICO directory entry."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise InstallError("The packaged PROVIS icon is not a valid PNG.")
    width, height = struct.unpack(">II", png[16:24])
    if not 1 <= width <= 256 or not 1 <= height <= 256:
        raise InstallError("The packaged icon dimensions are unsupported.")
    entry = struct.pack("<BBBBHHII", width & 0xFF, height & 0xFF, 0, 0,
                        1, 32, len(png), 22)
    return struct.pack("<HHH", 0, 1, 1) + entry + png


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _is_reparse_point(path: Path) -> bool:
    """Detect symlinks, junctions, and other Windows reparse points without following them."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _assert_no_reparse_components(path: Path, *, stop_at: Path) -> None:
    """Reject existing reparse components beneath a trusted known-folder root."""
    root = Path(os.path.abspath(stop_at))
    target = Path(os.path.abspath(path))
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise InstallError("The install path is outside the resolved per-user application folder.") from error
    current = root
    if _is_reparse_point(current):
        raise InstallError("The resolved per-user application folder is a reparse point; refusing it.")
    for part in relative.parts:
        current = current / part
        if _is_reparse_point(current):
            raise InstallError("An install destination contains a symlink or junction; refusing it.")


class PowerShellShortcutAdapter:
    """Use Windows' built-in WScript.Shell COM implementation for .lnk files."""

    def __init__(self, runner=subprocess.run):
        self.runner = runner

    @staticmethod
    def _ps_quote(value: Path | str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def _run(self, script: str) -> str:
        encoded = __import__("base64").b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            result = self.runner(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                capture_output=True, text=True, timeout=20, check=False,
            )
        except OSError as error:
            raise InstallError("Windows shortcut support is unavailable.") from error
        if result.returncode != 0:
            raise InstallError("Windows could not create or inspect a PROVIS shortcut.")
        return result.stdout.strip()

    def read(self, path: Path) -> dict[str, str]:
        script = "$w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut(" + self._ps_quote(path) + "); [pscustomobject]@{TargetPath=$s.TargetPath;WorkingDirectory=$s.WorkingDirectory;IconLocation=$s.IconLocation}|ConvertTo-Json -Compress"
        raw = self._run(script)
        if not raw:
            raise InstallError("Windows returned no shortcut metadata.")
        value = json.loads(raw)
        return {str(k): str(v or "") for k, v in value.items()}

    def write(self, path: Path, *, target: Path, working_directory: Path,
              icon: Path, description: str) -> None:
        script = (
            "$w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut(" + self._ps_quote(path) + "); "
            "$s.TargetPath=" + self._ps_quote(target) + "; "
            "$s.WorkingDirectory=" + self._ps_quote(working_directory) + "; "
            "$s.IconLocation=" + self._ps_quote(str(icon) + ",0") + "; "
            "$s.Description=" + self._ps_quote(description) + "; $s.Save()"
        )
        self._run(script)


class WindowsNodeInstaller:
    def __init__(self, *, folders: KnownFolders, icon_png: bytes,
                 release_document: dict = RELEASE_DOCUMENT,
                 public_keys: dict[str, str] = PUBLIC_KEYS,
                 gateway_base: str = GATEWAY_BASE,
                 platform: str = "windows-x86_64",
                 shortcut_adapter: ShortcutAdapter | None = None,
                 download_opener=None,
                 process_check: Callable[[Path], bool] | None = None):
        self.folders = folders
        self.icon_png = icon_png
        self.release_document = release_document
        self.public_keys = public_keys
        self.gateway_base = _validate_gateway(gateway_base)
        self.platform = platform
        self.shortcut_adapter = shortcut_adapter or PowerShellShortcutAdapter()
        self.download_opener = download_opener
        self.process_check = process_check or _is_image_running
        self.manifest = verify_release(self.release_document, self.public_keys, platform=self.platform)
        # Keep lexical paths so reparse points can be detected before any resolve/follow.
        self.program_root = Path(os.path.abspath(self.folders.local_app_data / "Programs" / APP_DIR_NAME))
        self.version_root = Path(os.path.abspath(self.program_root / self.manifest["version"]))
        self.exe_path = self.version_root / APP_EXE_NAME
        self.data_root = self.folders.local_app_data / APP_DIR_NAME

    def install_or_repair(self) -> Path:
        """Install the pinned release side-by-side, then atomically repair shortcuts.

        Existing version folders and per-user app data are never deleted or
        overwritten. A same-version hash conflict must be inspected manually.
        The app is not restarted or launched by this operation.
        """
        _assert_no_reparse_components(self.program_root, stop_at=self.folders.local_app_data)
        self.program_root.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_components(self.program_root, stop_at=self.folders.local_app_data)
        existing = self._existing_target()
        created = False
        backup: Path | None = None
        if not existing:
            if self.process_check(self.exe_path):
                raise InstallError("The target node is running; close it normally before reinstalling this version.")
            stage_dir = Path(tempfile.mkdtemp(prefix=".install-", dir=self.program_root))
            try:
                artifact = stage_dir / APP_EXE_NAME
                url = _artifact_url(self.gateway_base, self.manifest["release_id"])
                _download_artifact(url, artifact, self.manifest["size"], opener=self.download_opener)
                verify_artifact(artifact, self.manifest)
                if artifact.is_symlink():
                    raise InstallError("A symlinked release artifact is not allowed.")
                icon_path = stage_dir / "provis-lens.ico"
                icon_path.write_bytes(_png_to_ico(self.icon_png))
                receipt = {
                    "format": 1,
                    "release_id": self.manifest["release_id"],
                    "version": self.manifest["version"],
                    "platform": self.manifest["platform"],
                    "filename": APP_EXE_NAME,
                    "size": self.manifest["size"],
                    "sha256": self.manifest["sha256"],
                    "publisher": self.release_document["publisher"],
                }
                (stage_dir / "installer-release.json").write_text(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8"
                )
                if self.version_root.exists():
                    if self.version_root.is_symlink():
                        raise InstallError("The target version folder is a symlink; refusing it.")
                    backup = self.program_root / (self.manifest["version"] + ".preserved-" + uuid.uuid4().hex[:8])
                    os.replace(self.version_root, backup)
                os.replace(stage_dir, self.version_root)
                created = True
            except Exception:
                if stage_dir.exists() and _path_is_within(stage_dir, self.program_root):
                    shutil.rmtree(stage_dir, ignore_errors=True)
                if backup is not None and backup.exists() and not self.version_root.exists():
                    os.replace(backup, self.version_root)
                raise

        self._verify_installed()
        icon_path = self.version_root / "provis-lens.ico"
        if not icon_path.is_file():
            icon_path.write_bytes(_png_to_ico(self.icon_png))
        try:
            self._repair_shortcuts(icon_path)
        except Exception:
            if created and self.version_root.exists() and not self.process_check(self.exe_path):
                shutil.rmtree(self.version_root, ignore_errors=True)
            if backup is not None and backup.exists() and not self.version_root.exists():
                os.replace(backup, self.version_root)
            raise
        return self.exe_path

    def _existing_target(self) -> bool:
        if not self.version_root.exists():
            return False
        if self.version_root.is_symlink():
            raise InstallError("The target version folder is a symlink; refusing it.")
        if not self.exe_path.is_file() or self.exe_path.is_symlink():
            raise InstallError("The target version folder exists but has no regular app executable.")
        try:
            verify_artifact(self.exe_path, self.manifest)
        except PermissionError as error:
            raise InstallError("The existing target release differs from its signed artifact; preserving it for review.") from error
        return True

    def _verify_installed(self) -> None:
        verify_artifact(self.exe_path, self.manifest)

    def _repair_shortcuts(self, icon_path: Path) -> None:
        committed: list[tuple[Path, Path | None, list[Path]]] = []
        try:
            committed.append(self._repair_shortcut_folder(self.folders.desktop, icon_path))
            committed.append(self._repair_shortcut_folder(self.folders.programs, icon_path))
        except Exception:
            for destination, backup, rollback_links in reversed(committed):
                if backup and backup.exists():
                    os.replace(backup, destination)
                elif destination.exists():
                    try:
                        metadata = self.shortcut_adapter.read(destination)
                        if Path(metadata.get("TargetPath", "")).resolve() == self.exe_path.resolve():
                            destination.unlink()
                    except (OSError, InstallError, ValueError):
                        pass
                for link in rollback_links:
                    link.unlink(missing_ok=True)
            raise
        else:
            for _, backup, _ in committed:
                if backup:
                    backup.unlink(missing_ok=True)

    def _repair_shortcut_folder(self, folder: Path, icon_path: Path) -> tuple[Path, Path | None, list[Path]]:
        folder = Path(os.path.abspath(folder))
        _assert_no_reparse_components(folder, stop_at=folder.parent)
        folder.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_components(folder, stop_at=folder.parent)
        destination = folder / APP_SHORTCUT_NAME
        rollback_paths: list[Path] = []
        if destination.exists():
            metadata = self.shortcut_adapter.read(destination)
            target = Path(metadata.get("TargetPath", ""))
            if not self._owned_node_target(target):
                destination = folder / "PROVIS Inference Node (PROVIS).lnk"
                if destination.exists():
                    existing = self.shortcut_adapter.read(destination)
                    if not self._owned_node_target(Path(existing.get("TargetPath", ""))):
                        raise InstallError("A non-PROVIS shortcut already occupies the managed shortcut name.")
            else:
                old_version = self._version_from_target(target)
                if old_version and old_version != self.manifest["version"]:
                    rollback = folder / f"PROVIS Inference Node (Rollback {old_version}).lnk"
                    counter = 2
                    while rollback.exists():
                        rollback = folder / f"PROVIS Inference Node (Rollback {old_version} {counter}).lnk"
                        counter += 1
                    if not rollback.exists():
                        temp_rollback = folder / ("." + uuid.uuid4().hex + ".lnk")
                        self.shortcut_adapter.write(
                            temp_rollback, target=target,
                            working_directory=Path(metadata.get("WorkingDirectory") or target.parent),
                            icon=Path((metadata.get("IconLocation") or str(target)).split(",", 1)[0]),
                            description=f"PROVIS Inference Node rollback ({old_version})",
                        )
                        os.replace(temp_rollback, rollback)
                        rollback_paths.append(rollback)

        if destination.exists():
            current = self.shortcut_adapter.read(destination)
            if not self._owned_node_target(Path(current.get("TargetPath", ""))):
                raise InstallError("Refusing to replace a shortcut that does not target a PROVIS node install.")
            # Always rewrite the owned link so icon and working directory are repaired too.

        temp_link = folder / (".provis-node-" + uuid.uuid4().hex + ".lnk")
        backup_link: Path | None = None
        try:
            self.shortcut_adapter.write(
                temp_link, target=self.exe_path, working_directory=self.version_root,
                icon=icon_path, description="PROVIS Inference Node",
            )
            if destination.exists():
                backup_link = folder / (".provis-node-backup-" + uuid.uuid4().hex + ".lnk")
                shutil.copy2(destination, backup_link)
            os.replace(temp_link, destination)
        except Exception:
            temp_link.unlink(missing_ok=True)
            if backup_link and backup_link.exists():
                backup_link.unlink(missing_ok=True)
            for rollback in rollback_paths:
                rollback.unlink(missing_ok=True)
            raise
        return destination, backup_link, rollback_paths

    def _owned_node_target(self, target: Path) -> bool:
        if not target or target.name.casefold() != APP_EXE_NAME.casefold():
            return False
        try:
            lexical_target = Path(os.path.abspath(target))
            relative = lexical_target.relative_to(self.program_root)
            _assert_no_reparse_components(lexical_target, stop_at=self.program_root)
            return bool(relative.parts)
        except ValueError:
            return False
        except InstallError:
            return False

    def _version_from_target(self, target: Path) -> str | None:
        if not self._owned_node_target(target):
            return None
        try:
            relative = target.resolve().relative_to(self.program_root.resolve())
        except (OSError, ValueError):
            return None
        if len(relative.parts) >= 2 and relative.parts[-1].casefold() == APP_EXE_NAME.casefold():
            return relative.parts[0]
        return None

    def uninstall_owned(self, *, purge_user_data: bool = False) -> list[Path]:
        """Remove only installer-receipted binaries/owned links; preserve user data by default."""
        if purge_user_data:
            raise InstallError("Purging enrollment or user configuration is not supported by this installer.")
        if not self._is_receipted_install():
            return []
        if self.process_check(self.exe_path):
            raise InstallError("The node is still running; close it normally before uninstalling.")
        self._verify_installed()
        removed: list[Path] = []
        shortcut_backups: list[tuple[Path, Path]] = []
        version_backup = self.program_root / (self.manifest["version"] + ".uninstalling-" + uuid.uuid4().hex[:8])
        os.replace(self.version_root, version_backup)
        try:
            for folder in (self.folders.desktop, self.folders.programs):
                if not folder.is_dir():
                    continue
                for link in folder.glob("PROVIS Inference Node*.lnk"):
                    try:
                        metadata = self.shortcut_adapter.read(link)
                        target = Path(metadata.get("TargetPath", ""))
                    except (OSError, InstallError, ValueError):
                        continue
                    if self._owned_node_target(target) and target.resolve() == self.exe_path.resolve():
                        backup = folder / (".provis-uninstall-" + uuid.uuid4().hex + ".lnk")
                        os.replace(link, backup)
                        shortcut_backups.append((link, backup))
                        removed.append(link)
            shutil.rmtree(version_backup)
            for _, backup in shortcut_backups:
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    # Leave the hidden recoverable shortcut copy behind rather
                    # than turning a completed uninstall into a partial rollback.
                    pass
            removed.append(self.version_root)
        except Exception:
            if not self.version_root.exists() and version_backup.exists():
                os.replace(version_backup, self.version_root)
            for original, backup in reversed(shortcut_backups):
                if backup.exists() and not original.exists():
                    os.replace(backup, original)
            raise
        return removed

    def _is_receipted_install(self) -> bool:
        receipt_path = self.version_root / "installer-release.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            return False
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not (receipt.get("release_id") == self.manifest["release_id"]
                    and receipt.get("version") == self.manifest["version"]
                    and receipt.get("sha256") == self.manifest["sha256"]
                    and receipt.get("publisher") == self.release_document["publisher"]):
                return False
            verify_artifact(self.exe_path, self.manifest)
            return True
        except (OSError, ValueError, TypeError):
            return False


def _is_image_running(image_path: Path) -> bool:
    """Read-only process check; never terminates a process or inspects its arguments."""
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "@(Get-Process -Name PROVISInferenceNode -ErrorAction SilentlyContinue | ForEach-Object { try { $_.Path } catch { '' } }) | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            raise InstallError("Could not safely inspect whether the node is running.")
        value = json.loads(result.stdout or "[]")
        if isinstance(value, str):
            value = [value]
        wanted = str(image_path.resolve()).casefold()
        return any(str(item).casefold() == wanted for item in value if item)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise InstallError("Could not safely inspect whether the node is running.") from error


def main(argv: list[str] | None = None, *, icon_png: bytes | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install or repair the signed PROVIS Inference Node for this Windows user.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true", help="download and install the pinned signed release")
    mode.add_argument("--repair", action="store_true", help="verify the pinned release and repair its owned shortcuts")
    mode.add_argument("--uninstall", action="store_true", help="remove installer-owned files; keep enrollment/configuration")
    parser.add_argument("--purge-user-data", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.uninstall:
            confirmation = input("Remove installer-owned PROVIS node files and shortcuts? User configuration will remain. [y/N] ")
            if confirmation.strip().casefold() != "y":
                print("Uninstall cancelled.")
                return 0
        else:
            confirmation = input("Install or repair signed PROVIS Inference Node " + RELEASE_DOCUMENT["manifest"]["version"] + "? [y/N] ")
            if confirmation.strip().casefold() != "y":
                print("Installation cancelled.")
                return 0
        folders = resolve_known_folders()
        if icon_png is None:
            icon_path = Path(__file__).resolve().parents[1] / "assets" / "brand" / "provis-lens.png"
            if getattr(sys, "frozen", False):
                icon_path = Path(getattr(sys, "_MEIPASS", "")) / "provis-lens.png"
            icon_png = icon_path.read_bytes()
        installer = WindowsNodeInstaller(folders=folders, icon_png=icon_png)
        if args.uninstall:
            removed = installer.uninstall_owned(purge_user_data=args.purge_user_data)
            print(f"Removed {len(removed)} installer-owned item(s). User configuration was preserved.")
        else:
            exe = installer.install_or_repair()
            print(f"Verified signed PROVIS Inference Node {installer.manifest['version']}: {exe}")
            print("Use the Desktop or Start Menu shortcut to open it in your normal Windows session.")
        return 0
    except (InstallError, OSError, PermissionError, ValueError) as error:
        print(f"Installation stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
