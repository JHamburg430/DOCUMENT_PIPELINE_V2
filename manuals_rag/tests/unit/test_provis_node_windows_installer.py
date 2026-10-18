import base64
import hashlib
import json
import struct
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC = ROOT / "packages" / "inference_fabric" / "src"
WINDOWS_INSTALLER = ROOT / "windows_installer"
sys.path.insert(0, str(PACKAGE_SRC))
sys.path.insert(0, str(WINDOWS_INSTALLER))

from manuals_rag_inference.node_release import canonical  # noqa: E402
from provis_node_windows_installer import (  # noqa: E402
    APP_EXE_NAME,
    APP_SHORTCUT_NAME,
    GATEWAY_BASE,
    InstallError,
    KnownFolders,
    PUBLIC_KEYS,
    RELEASE_DOCUMENT,
    WindowsNodeInstaller,
    _NoRedirect,
    _artifact_url,
    _png_to_ico,
    _assert_no_reparse_components,
    resolve_known_folders,
)


def _release(data=b"signed-node-payload"):
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    manifest = {
        "format": 1,
        "release_id": str(uuid.uuid4()),
        "version": "2026.10.17.10",
        "platform": "windows-x86_64",
        "filename": APP_EXE_NAME,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return ({"publisher": "unit-test", "manifest": manifest,
             "signature": base64.b64encode(key.sign(canonical(manifest))).decode()},
            {"unit-test": base64.b64encode(public).decode()}, data)


def _png(width=16, height=16):
    # The icon builder only reads the PNG signature and IHDR dimensions.
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00" + b"fixture"


class Response:
    status = 200

    def __init__(self, body):
        self.body = body
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size):
        part = self.body[self.offset:self.offset + size]
        self.offset += len(part)
        return part


class Opener:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.requests = body, error, []

    def open(self, request, timeout):
        self.requests.append((request.full_url, timeout))
        if self.error:
            raise self.error
        return Response(self.body)


class FakeShortcuts:
    def read(self, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value

    def write(self, path, *, target, working_directory, icon, description):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "TargetPath": str(target),
            "WorkingDirectory": str(working_directory),
            "IconLocation": str(icon) + ",0",
            "Description": description,
        }), encoding="utf-8")


@pytest.fixture
def folders(tmp_path):
    return KnownFolders(tmp_path / "Local", tmp_path / "Desktop", tmp_path / "StartMenu")


def _installer(folders, *, data=b"signed-node-payload", opener=None, shortcuts=None, process_check=None):
    document, keys, payload = _release(data)
    return WindowsNodeInstaller(
        folders=folders, icon_png=_png(), release_document=document, public_keys=keys,
        gateway_base=GATEWAY_BASE, platform="windows-x86_64",
        shortcut_adapter=shortcuts or FakeShortcuts(),
        download_opener=opener or Opener(payload),
        process_check=process_check or (lambda _: False),
    ), payload


def test_known_folders_are_injected_and_codex_sandbox_roots_are_rejected(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Jane\AppData\Local\Packages\sandbox.{guid}\AC")
    known = {
        "F1B32785-6FBA-4FCF-9D55-7B8E7F157091": Path("Users/Jane/AppData/Local"),
        "B4BFCC3A-DB2C-424C-B029-7FE99A87C641": Path("Users/Jane/Desktop"),
        "A77F5D77-2E2B-44C3-A6A2-ABA601054A51": Path("Users/Jane/AppData/Roaming/Programs"),
    }
    result = resolve_known_folders(lambda value: known[value])
    assert result.local_app_data == known["F1B32785-6FBA-4FCF-9D55-7B8E7F157091"].absolute()
    with pytest.raises(InstallError, match="normal Windows user desktop"):
        resolve_known_folders(lambda value: Path("Users/Jane/AppData/Local/Packages/sandbox.guid/AC"))


def test_reparse_components_are_rejected_before_install_destination_is_used(tmp_path, monkeypatch):
    import provis_node_windows_installer as installer_module

    local = tmp_path / "Local"
    local.mkdir()
    program_root = local / "Programs" / "PROVIS"
    program_root.mkdir(parents=True)
    real_check = installer_module._is_reparse_point
    monkeypatch.setattr(installer_module, "_is_reparse_point",
                        lambda path: Path(path) == program_root or real_check(path))
    with pytest.raises(InstallError, match="symlink or junction"):
        _assert_no_reparse_components(program_root / "2026.10.17.10", stop_at=local)


def test_installer_pins_https_catalog_and_rejects_non_https_or_credentials():
    assert _artifact_url(GATEWAY_BASE, "39fd973c-c97a-4b54-97de-938a2f88bd6c").endswith(
        "/v1/node-releases/39fd973c-c97a-4b54-97de-938a2f88bd6c")
    with pytest.raises(InstallError):
        _artifact_url("http://example.test", str(uuid.uuid4()))
    with pytest.raises(InstallError):
        _artifact_url("https://user:pass@example.test", str(uuid.uuid4()))


def test_checked_in_installer_release_pin_verifies_with_existing_publisher_key():
    from manuals_rag_inference.node_release import verify_release

    manifest = verify_release(RELEASE_DOCUMENT, PUBLIC_KEYS, platform="windows-x86_64")
    assert manifest["version"] == "2026.10.17.10"
    assert manifest["release_id"] == "39fd973c-c97a-4b54-97de-938a2f88bd6c"
    assert manifest["sha256"] == "e635207ca0b1d13869523f7b679b0100ba05234222af487c9fadebaa94b914f2"


def test_signature_and_hash_are_verified_before_files_or_shortcuts_change(folders):
    installer, payload = _installer(folders)
    user_data = folders.local_app_data / "ProVIS"
    user_data.mkdir(parents=True)
    config = user_data / "inference-node.json"
    prefs = user_data / "inference-node-ui.json"
    config.write_bytes(b"existing enrollment config")
    prefs.write_bytes(b"existing UI preferences")
    old_root = folders.local_app_data / "Programs" / "PROVIS" / "2026.10.17.9"
    old_root.mkdir(parents=True)
    old_binary = old_root / APP_EXE_NAME
    old_binary.write_bytes(b"old rollback release")
    old_hash = hashlib.sha256(old_binary.read_bytes()).hexdigest()
    older_root = folders.local_app_data / "Programs" / "PROVIS" / "2026.10.17.8"
    older_root.mkdir(parents=True)
    older_binary = older_root / APP_EXE_NAME
    older_binary.write_bytes(b"older rollback release")
    shortcuts = installer.shortcut_adapter
    for folder in (folders.desktop, folders.programs):
        shortcuts.write(folder / APP_SHORTCUT_NAME, target=old_binary,
                        working_directory=old_root, icon=old_binary, description="old PROVIS node")
        shortcuts.write(folder / "PROVIS Inference Node (2026.10.17.8).lnk",
                        target=older_binary, working_directory=older_root,
                        icon=older_binary, description="old PROVIS node")

    installed = installer.install_or_repair()

    assert installed.read_bytes() == payload
    assert config.read_bytes() == b"existing enrollment config"
    assert prefs.read_bytes() == b"existing UI preferences"
    assert hashlib.sha256(old_binary.read_bytes()).hexdigest() == old_hash
    assert (installed.parent / "installer-release.json").is_file()
    assert (installed.parent / "provis-lens.ico").read_bytes().startswith(b"\x00\x00\x01\x00")
    for folder in (folders.desktop, folders.programs):
        shortcut = json.loads((folder / APP_SHORTCUT_NAME).read_text(encoding="utf-8"))
        assert Path(shortcut["TargetPath"]) == installed
        assert Path(shortcut["WorkingDirectory"]) == installed.parent
        assert Path(shortcut["IconLocation"].rsplit(",", 1)[0]).is_file()
        rollback = folder / "PROVIS Inference Node (Rollback 2026.10.17.9).lnk"
        assert json.loads(rollback.read_text(encoding="utf-8"))["TargetPath"] == str(old_binary)
        assert (folder / "PROVIS Inference Node (2026.10.17.8).lnk").is_file()
        assert not (folder / "PROVIS Inference Node (Rollback 2026.10.17.8).lnk").exists()


def test_tampered_artifact_leaves_no_install_or_shortcuts(folders):
    installer, _ = _installer(folders, opener=Opener(b"x" * len(b"signed-node-payload")))
    with pytest.raises(PermissionError, match="hash mismatch"):
        installer.install_or_repair()
    assert not installer.version_root.exists()
    assert not (folders.desktop / APP_SHORTCUT_NAME).exists()
    assert not (folders.programs / APP_SHORTCUT_NAME).exists()


def test_bad_signature_stops_before_catalog_request(folders):
    valid, keys, payload = _release()
    invalid = dict(valid)
    invalid["signature"] = base64.b64encode(b"x" * 64).decode()
    opener = Opener(payload)
    with pytest.raises(PermissionError, match="Release signature invalid"):
        WindowsNodeInstaller(
            folders=folders, icon_png=_png(), release_document=invalid, public_keys=keys,
            gateway_base=GATEWAY_BASE, platform="windows-x86_64", shortcut_adapter=FakeShortcuts(),
            download_opener=opener, process_check=lambda _: False,
        )
    assert not opener.requests


def test_redirect_is_rejected():
    with pytest.raises(InstallError, match="redirected"):
        _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://example.test")


def test_non_provis_same_name_shortcut_is_preserved_and_managed_link_is_separate(folders):
    adapter = FakeShortcuts()
    foreign = folders.desktop / APP_SHORTCUT_NAME
    adapter.write(foreign, target=Path(r"C:\OtherVendor\App.exe"), working_directory=Path(r"C:\OtherVendor"), icon=Path(r"C:\OtherVendor\App.exe"), description="foreign")
    installer, _ = _installer(folders, shortcuts=adapter)
    installer.install_or_repair()
    assert Path(adapter.read(foreign)["TargetPath"]) == Path(r"C:\OtherVendor\App.exe")
    assert (folders.desktop / "PROVIS Inference Node (PROVIS).lnk").exists()


def test_active_target_process_aborts_without_downloading(folders):
    opener = Opener(b"signed-node-payload")
    installer, _ = _installer(folders, opener=opener, process_check=lambda _: True)
    with pytest.raises(InstallError, match="target node is running"):
        installer.install_or_repair()
    assert not opener.requests
    assert not installer.version_root.exists()


def test_shortcut_phase_failure_rolls_back_program_files_and_first_folder(folders):
    class FailOnStartMenu(FakeShortcuts):
        def write(self, path, **kwargs):
            if Path(path).parent == folders.programs:
                raise InstallError("simulated Start Menu failure")
            return super().write(path, **kwargs)

    installer, _ = _installer(folders, shortcuts=FailOnStartMenu())
    with pytest.raises(InstallError, match="simulated"):
        installer.install_or_repair()
    assert not installer.version_root.exists()
    assert not (folders.desktop / APP_SHORTCUT_NAME).exists()
    assert not (folders.programs / APP_SHORTCUT_NAME).exists()


def test_shortcut_phase_failure_restores_existing_owned_shortcut(folders):
    adapter = FakeShortcuts()
    old_root = folders.local_app_data / "Programs" / "PROVIS" / "2026.10.17.9"
    old_root.mkdir(parents=True)
    old_binary = old_root / APP_EXE_NAME
    old_binary.write_bytes(b"old rollback release")
    original = folders.desktop / APP_SHORTCUT_NAME
    adapter.write(original, target=old_binary, working_directory=old_root,
                  icon=old_binary, description="existing shortcut")

    class FailOnStartMenu(FakeShortcuts):
        def write(self, path, **kwargs):
            if Path(path).parent == folders.programs:
                raise InstallError("simulated Start Menu failure")
            return super().write(path, **kwargs)

    installer, _ = _installer(folders, shortcuts=FailOnStartMenu())
    with pytest.raises(InstallError, match="simulated"):
        installer.install_or_repair()
    assert json.loads(original.read_text(encoding="utf-8"))["TargetPath"] == str(old_binary)
    assert not installer.version_root.exists()
    assert not list(folders.desktop.glob(".provis-node-backup-*.lnk"))


def test_uninstall_removes_only_receipted_release_and_preserves_user_data(folders):
    installer, _ = _installer(folders)
    installed = installer.install_or_repair()
    data = folders.local_app_data / "ProVIS"
    data.mkdir(parents=True, exist_ok=True)
    config = data / "inference-node.json"
    config.write_bytes(b"keep me")
    old = folders.local_app_data / "Programs" / "PROVIS" / "2026.10.17.9"
    old.mkdir(parents=True)
    old_binary = old / APP_EXE_NAME
    old_binary.write_bytes(b"old release")

    removed = installer.uninstall_owned()

    assert installed.parent in removed
    assert not installed.parent.exists()
    assert config.read_bytes() == b"keep me"
    assert old_binary.read_bytes() == b"old release"
    assert not (folders.desktop / APP_SHORTCUT_NAME).exists()
    assert not (folders.programs / APP_SHORTCUT_NAME).exists()


def test_uninstall_does_not_remove_unreceipted_preexisting_release(folders):
    installer, _ = _installer(folders)
    installer.version_root.mkdir(parents=True)
    installer.exe_path.write_bytes(b"preexisting")
    assert installer.uninstall_owned() == []
    assert installer.exe_path.read_bytes() == b"preexisting"


def test_uninstall_refuses_running_target_before_removing_shortcuts(folders):
    installer, _ = _installer(folders)
    installer.install_or_repair()
    installer.process_check = lambda _: True
    with pytest.raises(InstallError, match="still running"):
        installer.uninstall_owned()
    assert installer.version_root.exists()
    assert (folders.desktop / APP_SHORTCUT_NAME).exists()


def test_ico_wraps_png_dimensions_and_rejects_bad_assets():
    ico = _png_to_ico(_png(32, 24))
    assert ico[:6] == b"\x00\x00\x01\x00\x01\x00"
    assert struct.unpack("<BBBBHHII", ico[6:22])[:2] == (32, 24)
    with pytest.raises(InstallError):
        _png_to_ico(b"not a png")
