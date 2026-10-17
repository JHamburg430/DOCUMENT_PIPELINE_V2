from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tools.build_node_bundle import build
from provis_inference_node.ui_helpers import copy_endpoint_to_clipboard, local_broker_endpoints


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

    def test_local_broker_urls_follow_validated_configured_port(self) -> None:
        self.assertEqual(local_broker_endpoints(9001), (
            "http://127.0.0.1:9001/v1",
            "http://host.docker.internal:9001/v1",
        ))
        self.assertEqual(local_broker_endpoints("bad"), (
            "http://127.0.0.1:8767/v1",
            "http://host.docker.internal:8767/v1",
        ))
        self.assertEqual(local_broker_endpoints(70000), local_broker_endpoints(8767))

    def test_endpoint_copy_is_limited_to_displayed_urls(self) -> None:
        copied: list[str] = []
        actions: list[str] = []
        endpoints = local_broker_endpoints(8767)
        allowed = copy_endpoint_to_clipboard(
            endpoints[0], endpoints,
            clipboard_clear=lambda: actions.append("clear"),
            clipboard_append=copied.append,
            update=lambda: actions.append("update"),
        )
        self.assertTrue(allowed)
        self.assertEqual(copied, [endpoints[0]])
        self.assertEqual(actions, ["clear", "update"])
        self.assertFalse(copy_endpoint_to_clipboard(
            "https://gateway.example.test/?token=secret", endpoints,
            clipboard_clear=lambda: self.fail("unlisted URL must not clear clipboard"),
            clipboard_append=lambda _: self.fail("unlisted URL must not be copied"),
            update=lambda: self.fail("unlisted URL must not update clipboard"),
        ))

    def test_app_explains_shared_compute_endpoint_and_copy_controls(self) -> None:
        source = Path(__file__).parents[1] / "src" / "run_node_client.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("PROVIS shared compute — OpenAI-compatible endpoint", text)
        self.assertIn('text="Copy URL"', text)
        self.assertIn("employee-built workflows and apps can use it too", text)
        self.assertIn("This endpoint is local to this PC", text)
        self.assertIn("local_broker_endpoints(broker_port)", text)

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
