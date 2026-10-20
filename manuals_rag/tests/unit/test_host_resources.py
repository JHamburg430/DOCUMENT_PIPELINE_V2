import math
from pathlib import Path

import pytest

from manuals_rag_inference.host_resources import HostResourceSampler, validate_host_resources
from manuals_rag_inference.gateway import _snapshot_from_payload


def test_linux_capacity_and_cpu_interval(monkeypatch):
    import manuals_rag_inference.host_resources as module
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    ticks = iter(["cpu 100 0 100 800 0 0 0 0 0 0", "cpu 150 0 150 900 0 0 0 0 0 0"])
    def read(path):
        if path.as_posix() == "/proc/meminfo":
            return "MemTotal: 1000 kB\nMemAvailable: 400 kB\n"
        return next(ticks)
    monkeypatch.setattr(Path, "read_text", read)
    sampler = HostResourceSampler()
    first = sampler.sample()
    assert "cpu_percent" not in first  # A first sample is not measured zero usage.
    second = sampler.sample()
    assert second["cpu_percent"] == 50
    assert second["memory_total_bytes"] == 1024000
    assert second["memory_available_bytes"] == 409600
    assert second["disk_total_bytes"] > 0
    assert not any("path" in key for key in second)


def test_unavailable_probe_does_not_break_snapshot(monkeypatch):
    import manuals_rag_inference.host_resources as module
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    def unavailable(*args):
        raise OSError("unavailable")
    monkeypatch.setattr(Path, "read_text", unavailable)
    monkeypatch.setattr(module.shutil, "disk_usage", unavailable)
    result = HostResourceSampler().sample()
    assert "memory_total_bytes" not in result
    assert "disk_total_bytes" not in result
    assert result["observed_at_epoch"] > 0


def test_gateway_preserves_host_metrics_and_accepts_legacy_nodes():
    base = {"node_id": "test", "endpoint": "http://127.0.0.1:11434"}
    assert _snapshot_from_payload(base).host_resources is None
    metrics = {"memory_total_bytes": 1024, "memory_available_bytes": 512, "cpu_percent": 21.5}
    snapshot = _snapshot_from_payload({**base, "host_resources": metrics})
    assert snapshot.to_dict()["host_resources"] == metrics


@pytest.mark.parametrize("metrics", [
    {"cpu_percent": 101}, {"cpu_percent": math.nan}, {"cpu_count": True},
    {"memory_total_bytes": -1}, {"disk_total_bytes": 1.5},
    {"memory_total_bytes": 10, "memory_available_bytes": 11}, [],
])
def test_gateway_rejects_invalid_metrics(metrics):
    with pytest.raises(ValueError):
        validate_host_resources(metrics)


def test_gateway_drops_unrecognized_content():
    assert validate_host_resources({"cpu_count": 8, "processes": "private", "prompt": "private"}) == {"cpu_count": 8}
