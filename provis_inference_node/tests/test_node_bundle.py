from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.build_node_bundle import build


class NodeBundleTests(unittest.TestCase):
    def test_gateway_url_is_required_and_nonsecret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "node.zip"
            with self.assertRaises(ValueError):
                build(output, gateway_url="")
            with self.assertRaises(ValueError):
                build(output, gateway_url="http://gateway.internal")

    def test_linux_bundle_is_location_independent_and_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = build(Path(temporary) / "node.zip", gateway_url="https://inference.example.test")
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertIn("provis-inference-node/start-node-setup.sh", names)
                self.assertIn("provis-inference-node/build-node-linux.sh", names)
                self.assertIn("provis-inference-node/src/provis_inference_node/node_agent.py", names)
                config = json.loads(archive.read("provis-inference-node/provis-node-config.json"))
                self.assertEqual(config, {"gateway_url": "https://inference.example.test"})
                for name in ("start-node-setup.sh", "start-node-agent.sh", "build-node-linux.sh"):
                    info = archive.getinfo(f"provis-inference-node/{name}")
                    self.assertTrue(info.external_attr >> 16 & 0o111)
                    script = archive.read(f"provis-inference-node/{name}").decode()
                    self.assertIn('$(dirname -- "${BASH_SOURCE[0]}")', script)
                    self.assertNotIn("$(pwd)", script)
                    result = subprocess.run(
                        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn(b"INFERENCE_GATEWAY_TOKEN", archive.read("provis-inference-node/provis-node-config.json"))

    def test_linux_config_uses_xdg_without_importing_rag(self) -> None:
        source = Path(__file__).parents[1] / "src" / "run_node_client.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("from manuals_rag", text)
        self.assertNotIn("import manuals_rag", text)
        with patch.dict("os.environ", {"XDG_CONFIG_HOME": "/tmp/provis-xdg"}, clear=False):
            self.assertIn("XDG_CONFIG_HOME", text)

    def test_device_code_has_copy_and_sign_in_controls(self) -> None:
        source = Path(__file__).parents[1] / "src" / "run_node_client.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn('text="Copy code"', text)
        self.assertIn("clipboard_append(code)", text)
        self.assertIn('"user_code"', text)
        self.assertIn('"verification_url"', text)

    def test_node_diagnostics_are_sanitized(self) -> None:
        source = Path(__file__).parents[1] / "src" / "provis_inference_node" / "node_agent.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("HTTPError", text)
        self.assertIn('"heartbeat"', text)
        self.assertIn('"gpu"', text)
        self.assertIn('"ollama"', text)
        self.assertNotIn("sensitive rejection detail", text)


if __name__ == "__main__":
    unittest.main()
