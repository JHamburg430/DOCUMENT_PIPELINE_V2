"""Build, install into a temporary profile, and launch the real Windows node app.

This integration check never touches the current user's PROVIS data, shortcuts,
or installed application. It signs its test executable with a one-run fixture
key and closes only the process it starts.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from ctypes import wintypes

if os.name != "nt":
    raise SystemExit("This post-install launch check is Windows-only.")

ROOT = Path(__file__).resolve().parents[2]
MANUALS_RAG = ROOT / "manuals_rag"
sys.path.insert(0, str(MANUALS_RAG / "packages" / "inference_fabric" / "src"))
sys.path.insert(0, str(MANUALS_RAG / "scripts"))
sys.path.insert(0, str(MANUALS_RAG / "windows_installer"))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from manuals_rag_inference.node_release import canonical  # noqa: E402
from build_inference_node_bundle import build as build_bundle  # noqa: E402
from provis_node_windows_installer import (  # noqa: E402
    APP_EXE_NAME,
    GATEWAY_BASE,
    KnownFolders,
    WindowsNodeInstaller,
)


class _Response:
    def __init__(self, body: bytes):
        self.body = body
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, size: int) -> bytes:
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class _Opener:
    def __init__(self, body: bytes):
        self.body = body

    def open(self, _request, timeout):
        if timeout <= 0:
            raise AssertionError("Installer must use a positive timeout")
        return _Response(self.body)


class _Shortcuts:
    def read(self, path: Path) -> dict[str, str]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write(self, path: Path, *, target: Path, working_directory: Path,
              icon: Path, description: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "TargetPath": str(target),
            "WorkingDirectory": str(working_directory),
            "IconLocation": str(icon) + ",0",
            "Description": description,
        }), encoding="utf-8")


def _visible_app_window(pid: int) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_windows = user32.EnumWindows
    enum_windows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]
    enum_windows.restype = wintypes.BOOL
    get_pid = user32.GetWindowThreadProcessId
    get_pid.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    get_text_length = user32.GetWindowTextLengthW
    get_text_length.argtypes = [wintypes.HWND]
    get_text_length.restype = ctypes.c_int
    get_text = user32.GetWindowTextW
    get_text.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    is_visible = user32.IsWindowVisible
    is_visible.argtypes = [wintypes.HWND]
    found = False

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        nonlocal found
        window_pid = wintypes.DWORD()
        get_pid(hwnd, ctypes.byref(window_pid))
        if window_pid.value != pid or not is_visible(hwnd):
            return True
        length = get_text_length(hwnd)
        title = ctypes.create_unicode_buffer(max(length + 1, 2))
        get_text(hwnd, title, len(title))
        if title.value == "PROVIS Inference Node":
            found = True
            return False
        return True

    enum_windows(visit, 0)
    return found


def _launch_and_wait(executable: Path, config_path: Path, expected_config: bytes | None) -> None:
    if expected_config is None:
        assert not config_path.parent.exists(), "Fresh profile must start without its config parent"
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(expected_config)

    update_root = config_path.resolve().parent / "app-updates"
    assert not update_root.exists(), "Each scenario must begin without an updater directory"
    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(config_path.parents[1])
    env["INFERENCE_NODE_CONFIG_PATH"] = str(config_path)
    env.pop("PROVIS_INFERENCE_GATEWAY_URL", None)
    env.pop("INFERENCE_GATEWAY_URL", None)
    env.pop("INFERENCE_ENTRA_TENANT_ID", None)
    env.pop("INFERENCE_ENTRA_CLIENT_ID", None)
    env.pop("INFERENCE_ENTRA_SCOPE", None)
    process = subprocess.Popen(
        [str(executable)],
        cwd=str(executable.parent),
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise AssertionError(f"Installed app exited before its main window (exit {exit_code})")
            if update_root.is_dir() and _visible_app_window(process.pid):
                if expected_config is not None:
                    assert config_path.read_bytes() == expected_config, "App startup changed existing config"
                return
            time.sleep(0.25)
        raise AssertionError("Installed app did not create app-updates and show its main window within 25 seconds")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="provis-postinstall-startup-") as temporary:
        root = Path(temporary)
        bundle_path = root / "managed-node.zip"
        extract_root = root / "bundle"
        build_root = extract_root / "provis-inference-node"
        dist_root = root / "dist"

        key = Ed25519PrivateKey.generate()
        public = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        public_b64 = base64.b64encode(public).decode("ascii")
        test_version = "2026.10.18.2"
        build_bundle(
            bundle_path,
            gateway_url="https://gateway.invalid",
            update_config={"enabled": True, "version": test_version, "public_keys": {"ci-fixture": public_b64}},
        )
        extract_root.mkdir(parents=True)
        with zipfile.ZipFile(bundle_path) as archive:
            archive.extractall(extract_root)

        app_executable = dist_root / APP_EXE_NAME
        dist_root.mkdir()
        app_source = build_root / "src" / "run_inference_node_client.py"
        icon_asset = build_root / "src" / "assets" / "brand" / "provis-lens.png"
        config_asset = build_root / "provis-node-config.json"
        command = [
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--noconsole",
            "--name", Path(APP_EXE_NAME).stem,
            "--distpath", str(dist_root), "--workpath", str(root / "pyinstaller-work"),
            "--specpath", str(root), "--paths", str(build_root / "src"),
            "--add-data", str(config_asset) + ";.",
            "--add-data", str(icon_asset) + ";assets\\brand",
            "--collect-all", "cryptography", str(app_source),
        ]
        subprocess.run(command, cwd=build_root, check=True, timeout=600)
        payload = app_executable.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        manifest = {
            "format": 1,
            "release_id": str(uuid.uuid4()),
            "version": test_version,
            "platform": "windows-x86_64",
            "filename": APP_EXE_NAME,
            "size": len(payload),
            "sha256": digest,
        }
        document = {
            "publisher": "provis-postinstall-ci",
            "manifest": manifest,
            "signature": base64.b64encode(key.sign(canonical(manifest))).decode("ascii"),
        }

        profile = root / "profile"
        local_app_data = profile / "Local"
        folders = KnownFolders(local_app_data, root / "Desktop", root / "StartMenu")
        icon_png = icon_asset.read_bytes()
        installer = WindowsNodeInstaller(
            folders=folders,
            icon_png=icon_png,
            release_document=document,
            public_keys={"provis-postinstall-ci": public_b64},
            gateway_base=GATEWAY_BASE,
            platform="windows-x86_64",
            shortcut_adapter=_Shortcuts(),
            download_opener=_Opener(payload),
            process_check=lambda _path: False,
        )
        installed = installer.install_or_repair()
        assert installed.is_file() and hashlib.sha256(installed.read_bytes()).hexdigest() == digest
        assert (installed.parent / "installer-release.json").is_file()

        first_run = profile / "Fresh" / "Local" / "ProVIS" / "inference-node.json"
        _launch_and_wait(installed, first_run, None)
        assert not first_run.exists(), "First-run startup must not fabricate enrollment/config data"

        existing_root = profile / "Existing" / "Local" / "ProVIS"
        existing_config = existing_root / "inference-node.json"
        existing_bytes = b'{"gateway_url":"https://gateway.invalid","node_id":"ci-fixture"}'
        _launch_and_wait(installed, existing_config, existing_bytes)
        assert existing_config.read_bytes() == existing_bytes

        print(json.dumps({
            "status": "passed",
            "postinstall_path": str(installed),
            "sha256": digest,
            "first_run_missing_parent": "passed",
            "existing_config_preserved": "passed",
            "actual_app_window_seen": True,
            "production_profile_touched": False,
            "production_release_signed": False,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
