from pathlib import Path

import manuals_rag_inference.node_system as node_system


def test_ollama_status_rejects_non_local_endpoint(monkeypatch):
    monkeypatch.setattr(node_system.shutil, "which", lambda _name: None)

    result = node_system.ollama_status("https://gateway.example.test")

    assert result == {"status": "not_installed", "installed": False, "version": ""}


def test_tailscale_status_is_sanitized_and_reports_connection(monkeypatch):
    monkeypatch.setattr(node_system.shutil, "which", lambda name: "/usr/bin/tailscale" if name == "tailscale" else None)

    def fake_run(command, *, timeout):
        assert timeout > 0
        if command == ["tailscale", "status", "--json"]:
            return 0, '{"BackendState":"Running","Self":{"HostName":"private-host"}}', ""
        if command == ["tailscale", "ip", "-4"]:
            return 0, "100.64.0.7\n", ""
        if command == ["tailscale", "version"]:
            return 0, "1.80.0\nother internal detail\n", ""
        raise AssertionError(command)

    monkeypatch.setattr(node_system, "_run_fixed", fake_run)
    result = node_system.tailscale_status()

    assert result == {
        "status": "connected",
        "installed": True,
        "connected": True,
        "version": "1.80.0",
        "ipv4": "100.64.0.7",
    }
    assert "private-host" not in str(result)


def test_gpu_driver_status_returns_bounded_inventory(monkeypatch):
    monkeypatch.setattr(node_system.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(
        node_system,
        "_run_fixed",
        lambda command, *, timeout: (0, "555.42.02, NVIDIA RTX\n", ""),
    )

    result = node_system.gpu_driver_status()

    assert result["status"] == "available"
    assert result["vendor"] == "nvidia"
    assert result["driver"] == "555.42.02"
    assert result["gpus"] == ["NVIDIA RTX"]


def test_managed_installer_uses_only_fixed_bundled_names(monkeypatch, tmp_path: Path):
    installer_root = tmp_path / "installers"
    installer_root.mkdir()
    approved = installer_root / "OllamaSetup.exe"
    approved.write_bytes(b"placeholder")
    (installer_root / "arbitrary.exe").write_bytes(b"should-not-be-selected")
    monkeypatch.setattr(node_system.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert node_system.managed_installer("ollama") == approved
    assert node_system.managed_installer("unknown") is None


def test_official_installer_urls_are_fixed_to_vendor_hosts(monkeypatch):
    monkeypatch.setattr(node_system.platform, "system", lambda: "Linux")
    assert node_system.official_installer_url("ollama") == "https://ollama.com/install.sh"
    assert node_system.official_installer_url("tailscale") == "https://tailscale.com/install.sh"
    monkeypatch.setattr(node_system.platform, "system", lambda: "Windows")
    assert node_system.official_installer_url("ollama") == "https://ollama.com/install.ps1"
    assert node_system.official_installer_url("tailscale") == "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe"
    assert node_system.official_installer_url("gpu") is None


def test_official_download_is_bounded_and_uses_temp_storage(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(node_system.platform, "system", lambda: "Linux")
    monkeypatch.setattr(node_system.tempfile, "gettempdir", lambda: str(tmp_path))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if getattr(self, "done", False):
                return b""
            self.done = True
            return b"#!/bin/sh\necho official\n"

    monkeypatch.setattr(node_system, "urlopen", lambda request, timeout: Response())
    result = node_system._download_official_installer("ollama")

    assert result is not None
    assert result.read_bytes().startswith(b"#!/bin/sh")
    assert result.parent == tmp_path / "provis-inference-node-installers"


def test_launch_installer_downloads_when_no_managed_copy_exists(monkeypatch, tmp_path: Path):
    installer = tmp_path / "ollama-latest.sh"
    installer.write_text("#!/bin/sh\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(node_system, "managed_installer", lambda _component: None)
    monkeypatch.setattr(node_system, "official_installer_url", lambda _component: "https://ollama.com/install.sh")
    monkeypatch.setattr(node_system, "_download_official_installer", lambda _component: installer)
    monkeypatch.setattr(node_system.subprocess, "Popen", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(node_system.os, "name", "posix")

    result = node_system.launch_managed_installer("ollama")

    assert result == {"status": "started", "source": "official_download"}
    assert commands == [["sh", str(installer)]]


def test_tailscale_actions_use_fixed_local_commands(monkeypatch):
    commands = []

    def fake_run(command, *, timeout):
        commands.append((command, timeout))
        return 0, "", ""

    monkeypatch.setattr(node_system, "_run_fixed", fake_run)

    assert node_system.tailscale_disconnect() == {"status": "disconnected"}
    assert node_system.tailscale_reconnect() == {"status": "connected"}
    assert commands == [(["tailscale", "down"], 20), (["tailscale", "up"], 30)]
