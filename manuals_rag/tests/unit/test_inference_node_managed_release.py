import json
import zipfile

import pytest

from build_inference_node_bundle import build


def test_managed_bundle_requires_verified_https_gateway(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path / "missing.zip")
    with pytest.raises(ValueError):
        build(tmp_path / "http.zip", gateway_url="http://gateway.example.test")
    with pytest.raises(ValueError):
        build(tmp_path / "credential.zip", gateway_url="https://user:pass@gateway.example.test")
    with pytest.raises(ValueError):
        build(tmp_path / "query.zip", gateway_url="https://gateway.example.test/?token=bad")


def test_managed_bundle_contains_nonsecret_config_and_pyinstaller_data_contract(tmp_path):
    output = build(tmp_path / "managed.zip", gateway_url="https://inference.example.test")
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "provis-inference-node/provis-node-config.json" in names
        assert "provis-inference-node/src/run_inference_node_client.py" in names
        config = json.loads(archive.read("provis-inference-node/provis-node-config.json"))
        assert config == {"gateway_url": "https://inference.example.test"}
        build_script = archive.read("provis-inference-node/build-node-exe.ps1").decode()
        assert "--add-data" in build_script
        assert "provis-node-config.json" in build_script
        assert "INFERENCE_GATEWAY_TOKEN" not in archive.read("provis-inference-node/provis-node-config.json").decode()

