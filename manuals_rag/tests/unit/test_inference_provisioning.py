import hashlib
import io
import tarfile

import pytest

from manuals_rag_inference.provisioning import ProvisioningManifest, activate_staged, stage_artifact


def _manifest(**overrides):
    values = {
        "version": "0.1.42",
        "platform": "linux-amd64",
        "artifact_url": "https://ollama.com/download/ollama-linux-amd64.tgz",
        "sha256": "0" * 64,
    }
    values.update(overrides)
    return values


def _tar_bytes(binary=b"ollama-binary"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("bin/ollama")
        info.size = len(binary)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(binary))
    return buffer.getvalue()


def test_manifest_rejects_non_official_or_unsafe_artifact():
    with pytest.raises(ValueError, match="HTTPS"):
        ProvisioningManifest.from_dict(_manifest(artifact_url="http://example.com/ollama.tgz"))
    with pytest.raises(ValueError, match="unsafe"):
        ProvisioningManifest.from_dict(_manifest(binary_member="../ollama"))


def test_stage_verifies_digest_and_preserves_versioned_install(tmp_path):
    artifact = _tar_bytes()
    digest = hashlib.sha256(artifact).hexdigest()
    manifest = ProvisioningManifest.from_dict(_manifest(sha256=digest))

    def downloader(url, destination, **kwargs):
        destination.write_bytes(artifact)
        return len(artifact)

    result = stage_artifact(manifest, install_root=tmp_path / "ollama", downloader=downloader)
    binary = tmp_path / "ollama" / "versions" / "0.1.42" / "bin" / "ollama"
    assert result["version"] == "0.1.42"
    assert binary.read_bytes() == b"ollama-binary"
    assert (binary.parent.parent / "MANIFEST.json").exists()


def test_stage_rejects_tampered_artifact_without_leaving_version(tmp_path):
    artifact = _tar_bytes()
    manifest = ProvisioningManifest.from_dict(_manifest(sha256="1" * 64))

    def downloader(url, destination, **kwargs):
        destination.write_bytes(artifact)
        return len(artifact)

    with pytest.raises(ValueError, match="SHA-256"):
        stage_artifact(manifest, install_root=tmp_path / "ollama", downloader=downloader)
    assert not (tmp_path / "ollama" / "versions" / "0.1.42").exists()


def test_stage_rejects_archive_symlink(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("bin/ollama")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        bundle.addfile(info)
    artifact = buffer.getvalue()
    manifest = ProvisioningManifest.from_dict(_manifest(sha256=hashlib.sha256(artifact).hexdigest()))

    def downloader(url, destination, **kwargs):
        destination.write_bytes(artifact)
        return len(artifact)

    with pytest.raises(ValueError, match="regular file"):
        stage_artifact(manifest, install_root=tmp_path / "ollama", downloader=downloader)


class FakeController:
    def __init__(self, healthy=True):
        self.healthy = healthy
        self.restarts = 0

    def restart(self):
        self.restarts += 1

    def healthcheck(self):
        return self.healthy


def _stage_version(tmp_path, version="0.1.42"):
    artifact = _tar_bytes(binary=version.encode())
    manifest = ProvisioningManifest.from_dict(_manifest(version=version, sha256=hashlib.sha256(artifact).hexdigest()))

    def downloader(url, destination, **kwargs):
        destination.write_bytes(artifact)
        return len(artifact)

    stage_artifact(manifest, install_root=tmp_path / "ollama", downloader=downloader)
    return manifest


def test_activation_requires_approval_and_idle_node(tmp_path):
    manifest = _stage_version(tmp_path)
    controller = FakeController()
    with pytest.raises(PermissionError, match="approval"):
        activate_staged(manifest, install_root=tmp_path / "ollama", controller=controller, maintenance_approved=False)
    with pytest.raises(RuntimeError, match="active"):
        activate_staged(manifest, install_root=tmp_path / "ollama", controller=controller, maintenance_approved=True, active_requests=1)
    assert controller.restarts == 0


def test_activation_switches_atomically_and_health_checks(tmp_path):
    first = _stage_version(tmp_path, "0.1.41")
    root = tmp_path / "ollama"
    current = root / "current"
    current.symlink_to("versions/0.1.41")
    second = _stage_version(tmp_path, "0.1.42")
    controller = FakeController(healthy=True)
    result = activate_staged(second, install_root=root, controller=controller, maintenance_approved=True)
    assert result["previous_target"] == "versions/0.1.41"
    assert current.is_symlink()
    assert current.resolve() == (root / "versions/0.1.42").resolve()
    assert controller.restarts == 1


def test_activation_rolls_back_when_health_check_fails(tmp_path):
    first = _stage_version(tmp_path, "0.1.41")
    root = tmp_path / "ollama"
    current = root / "current"
    current.symlink_to("versions/0.1.41")
    second = _stage_version(tmp_path, "0.1.42")
    controller = FakeController(healthy=False)
    with pytest.raises(RuntimeError, match="health check"):
        activate_staged(second, install_root=root, controller=controller, maintenance_approved=True)
    assert current.is_symlink()
    assert current.resolve() == (root / "versions/0.1.41").resolve()
    assert controller.restarts == 2
