import json
import os
from pathlib import Path
import socket
import threading
import zipfile

import pytest

from manuals_rag_inference.node_theme import PALETTES, load_theme, save_theme
import build_inference_node_app as builder
import run_inference_node_client as client


def test_theme_preference_survives_restart_and_does_not_touch_enrollment(tmp_path):
    enrollment = tmp_path / 'inference-node.json'
    enrollment.write_text('{"shared_secret":"fixture-never-copy"}')
    appearance = enrollment.with_name('inference-node-ui.json')
    assert load_theme(appearance) == 'light'
    save_theme(appearance, 'dark')
    assert load_theme(appearance) == 'dark'
    assert json.loads(appearance.read_text()) == {'theme': 'dark'}
    assert json.loads(enrollment.read_text())['shared_secret'] == 'fixture-never-copy'
    save_theme(appearance, 'light')
    assert load_theme(appearance) == 'light'
    appearance.write_text('broken')
    assert load_theme(appearance) == 'light'
    with pytest.raises(ValueError): save_theme(appearance, 'arbitrary')
    assert PALETTES['dark']['background'] != PALETTES['light']['background']


def test_managed_linux_zipapp_uses_shared_desktop_and_embedded_url(tmp_path, monkeypatch):
    app = builder.build(tmp_path / 'node.pyz', gateway_url='https://gateway.example.test:8445')
    with zipfile.ZipFile(app) as archive:
        source = archive.read('node_app.py').decode()
        assert 'InferenceNodeClient' in source and 'public_events(events)' in source
        assert '_toggle_theme' in source and 'Copy' in source
        assert 'manuals_rag_inference/node_linux_desktop.py' in archive.namelist()
        assert json.loads(archive.read('provis-node-config.json')) == {'gateway_url': 'https://gateway.example.test:8445'}
        assert not any('__pycache__' in name for name in archive.namelist())
    monkeypatch.setattr(client.sys, 'argv', [str(app)])
    assert client._load_managed_gateway_url() == 'https://gateway.example.test:8445'


@pytest.mark.skipif(os.name != 'posix', reason='Requires real Linux peer credentials and flock')
def test_linux_relaunch_opens_existing_node_and_releases_instance(tmp_path, monkeypatch):
    from manuals_rag_inference.node_linux_desktop import LinuxInstance
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    first = LinuxInstance()
    received = threading.Event()
    first.watch(lambda event, value: received.set() if event == 'show_existing' else None)
    second = LinuxInstance()
    try:
        assert second.existing and received.wait(2)
        assert first.path.stat().st_mode & 0o777 == 0o600
    finally:
        second.close();first.close()
    third = LinuxInstance()
    try: assert not third.existing
    finally: third.close()
