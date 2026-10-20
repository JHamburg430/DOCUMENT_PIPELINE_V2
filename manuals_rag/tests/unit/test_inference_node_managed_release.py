import json
import os
import zipfile

import pytest

from build_inference_node_bundle import build
import build_inference_node_linux as linux_builder


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
        assert "provis-inference-node/src/provis_agent_link_mcp.py" in names
        assert "provis-inference-node/src/run_agent_link_mcp.py" in names
        assert "provis-inference-node/provis-node-config.json" in names
        assert "provis-inference-node/src/run_inference_node_client.py" in names
        assert "provis-inference-node/src/provis_inference_node/openai_broker.py" in names
        assert "provis-inference-node/src/assets/brand/provis-lens.png" in names
        config = json.loads(archive.read("provis-inference-node/provis-node-config.json"))
        assert config == {"gateway_url": "https://inference.example.test"}
        build_script = archive.read("provis-inference-node/build-node-exe.ps1").decode()
        assert "--add-data" in build_script
        assert "--collect-all cryptography" in build_script
        assert "provis-node-config.json" in build_script
        assert "assets\\brand" in build_script
        assert "INFERENCE_GATEWAY_TOKEN" not in archive.read("provis-inference-node/provis-node-config.json").decode()


def test_linux_release_is_one_file_and_embeds_managed_config(tmp_path):
    config = tmp_path / "provis-node-config.json"
    output = tmp_path / "PROVISInferenceNode"
    args = linux_builder.pyinstaller_args(output, config, tmp_path / "work")
    assert args[:3] == [linux_builder.sys.executable, "-m", "PyInstaller"]
    assert "--onefile" in args
    assert "--windowed" in args
    assert "--add-data" in args
    assert args[args.index("--collect-all") + 1] == "cryptography"
    assert f"{config}{os.pathsep}." in args
    assert str(linux_builder.BRAND_ASSET_ROOT / "provis-lens.png") + os.pathsep + "assets/brand" in args
    assert str(linux_builder.PACKAGE_PARENT) in args
    assert str(linux_builder.SOURCE_ROOT) in args
    mcp_args = linux_builder.mcp_pyinstaller_args(output.with_name("PROVISAgentLinkMCP"), tmp_path / "work")
    assert "--console" in mcp_args
    assert str(linux_builder.MCP_SOURCE) in mcp_args
    assert str(linux_builder.BROKER_PARENT) in args
