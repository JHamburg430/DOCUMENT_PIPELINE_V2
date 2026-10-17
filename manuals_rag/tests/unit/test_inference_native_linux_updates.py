"""Native Linux update contract, including compatible legacy zipapps."""
import base64
import hashlib
import sys
import uuid
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from manuals_rag_inference import node_release, node_updater


def test_linux_default_version_matches_current_windows_release():
    assert node_release.DEFAULT_VERSION == '2026.10.17.7'


@pytest.mark.parametrize('platform', ['linux-x86_64', 'linux-aarch64'])
@pytest.mark.parametrize('filename', ['PROVISInferenceNode', 'inference-node.pyz'])
def test_both_signed_linux_formats_have_fixed_filenames(platform, filename):
    key = Ed25519PrivateKey.generate()
    manifest = dict(format=1, release_id=str(uuid.uuid4()), version='2026.10.17.3',
                    platform=platform, filename=filename, size=4, sha256=hashlib.sha256(b'test').hexdigest())
    document = dict(manifest=manifest, publisher='fixture',
                    signature=base64.b64encode(key.sign(node_release.canonical(manifest))).decode())
    keys = {'fixture': base64.b64encode(key.public_key().public_bytes_raw()).decode()}
    assert node_release.verify_release(document, keys, platform=platform)['filename'] == filename
    manifest['filename'] = '../PROVISInferenceNode'
    document['signature'] = base64.b64encode(key.sign(node_release.canonical(manifest))).decode()
    with pytest.raises(ValueError):
        node_release.verify_release(document, keys, platform=platform)

def test_native_linux_command_does_not_launch_python_on_an_elf(tmp_path, monkeypatch):
    app = tmp_path / 'PROVISInferenceNode'
    app.write_bytes(b'\x7fELF' + b'synthetic-header')
    # Swap only the module's OS view; pathlib must keep the actual host flavor.
    from types import SimpleNamespace
    monkeypatch.setattr(node_updater, 'os', SimpleNamespace(name='posix'))
    assert node_updater.command(app) == [str(app.resolve())]
    app.write_bytes(b'#!/usr/bin/env python3\n')
    assert node_updater.command(app) == [sys.executable, str(app.resolve())]
