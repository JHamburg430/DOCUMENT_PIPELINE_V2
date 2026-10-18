from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from provis_inference_node.connect_diagnostics import (
    CONNECT_PHASES,
    ERROR_CATEGORIES,
    append_connect_diagnostic,
)
from provis_inference_node.entra_client import EntraHttpError


class ConnectDiagnosticTests(unittest.TestCase):
    def test_http_failure_records_only_allowlisted_fields_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inference-node-diagnostics.jsonl"
            error = EntraHttpError(403)
            error.args = ("https://gateway.invalid/?token=secret user=john@example.test",)
            append_connect_diagnostic(path, "entra_enrollment", error=error)

            line = path.read_text(encoding="utf-8").strip()
            record = json.loads(line)
            self.assertEqual(record["phase"], "entra_enrollment")
            self.assertEqual(record["http_status"], 403)
            self.assertEqual(record["error_category"], "http_error")
            self.assertTrue(record["time_utc"].endswith("Z"))
            self.assertEqual(set(record), {"time_utc", "phase", "http_status", "error_category"})
            for sensitive in ("gateway.invalid", "token=secret", "john@example.test"):
                self.assertNotIn(sensitive, line)

    def test_unrecognized_phase_and_error_text_are_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.jsonl"
            secret_message = "https://host.invalid/path?token=abc user=alice@example.test node=HOST-123"
            append_connect_diagnostic(path, secret_message, error=RuntimeError(secret_message))

            line = path.read_text(encoding="utf-8").strip()
            record = json.loads(line)
            self.assertEqual(record["phase"], "unknown")
            self.assertIn(record["error_category"], ERROR_CATEGORIES)
            self.assertNotIn("http_status", record)
            self.assertEqual(set(record), {"time_utc", "phase", "error_category"})
            for sensitive in ("host.invalid", "token=abc", "alice@example.test", "HOST-123"):
                self.assertNotIn(sensitive, line)

    def test_http_error_uses_numeric_status_and_never_serializes_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.jsonl"
            error = HTTPError(
                "https://host.invalid/enroll?access_token=secret",
                401,
                "sensitive rejection detail",
                {},
                None,
            )
            append_connect_diagnostic(path, "entra_enrollment", error=error)

            line = path.read_text(encoding="utf-8").strip()
            record = json.loads(line)
            self.assertEqual(record["http_status"], 401)
            self.assertEqual(record["error_category"], "http_error")
            self.assertNotIn("host.invalid", line)
            self.assertNotIn("access_token", line)
            self.assertNotIn("sensitive rejection detail", line)

    def test_device_code_auth_failure_keeps_status_and_safe_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.jsonl"
            error = PermissionError("access_token=secret user=alice@example.test")
            error.http_status = 400
            append_connect_diagnostic(path, "device_code_poll", error=error)

            line = path.read_text(encoding="utf-8").strip()
            record = json.loads(line)
            self.assertEqual(record["http_status"], 400)
            self.assertEqual(record["error_category"], "authorization_denied")
            self.assertNotIn("access_token", line)
            self.assertNotIn("alice@example.test", line)

    def test_success_record_has_only_timestamp_and_allowlisted_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.jsonl"
            for phase in CONNECT_PHASES:
                append_connect_diagnostic(path, phase)

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({record["phase"] for record in records}, CONNECT_PHASES)
            self.assertTrue(all(set(record) == {"time_utc", "phase"} for record in records))


if __name__ == "__main__":
    unittest.main()
