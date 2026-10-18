import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

from manuals_rag_inference.connect_diagnostics import append_connect_diagnostic


def test_diagnostic_records_only_phase_status_category_and_timestamp():
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "inference-node-diagnostics.jsonl"
        error = HTTPError(
            "https://gateway.example/enroll?access_token=secret",
            403,
            "owner subject john@example.test was rejected",
            {},
            None,
        )
        append_connect_diagnostic(path, "entra_registration_check", error=error)
        line = path.read_text(encoding="utf-8").strip()
        record = json.loads(line)

        assert record["phase"] == "entra_registration_check"
        assert record["http_status"] == 403
        assert record["error_category"] == "http_error"
        assert record["time_utc"].endswith("Z")
        assert set(record) == {"time_utc", "phase", "http_status", "error_category"}
        for secret in ("gateway.example", "access_token", "secret", "john@example.test", "owner subject"):
            assert secret not in line


def test_unknown_phase_and_exception_message_are_never_logged():
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "diagnostics.jsonl"
        message = "https://host.example/?token=abc user=alice@example.test"
        append_connect_diagnostic(path, message, error=RuntimeError(message))
        line = path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["phase"] == "unknown"
        assert record["error_category"] == "unknown_error"
        assert set(record) == {"time_utc", "phase", "error_category"}
        assert "host.example" not in line
        assert "token=abc" not in line
        assert "alice@example.test" not in line


def test_success_record_contains_no_registration_or_identity_data():
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "diagnostics.jsonl"
        append_connect_diagnostic(path, "connect_complete")
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["phase"] == "connect_complete"
        assert set(record) == {"time_utc", "phase"}
