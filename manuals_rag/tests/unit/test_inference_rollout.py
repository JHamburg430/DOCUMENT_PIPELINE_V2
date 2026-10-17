from scripts import check_inference_fabric_rollout as rollout


def test_rollout_verifier_accepts_two_fresh_gpu_nodes(monkeypatch):
    responses = {
        "/health": {"status": "ok"},
        "/v1/nodes": {
            "nodes": [
                {
                    "node_id": "gpu-a",
                    "healthy": True,
                    "observed_at_epoch": rollout.time.time(),
                    "gpus": [{"gpu_id": "0"}],
                    "request_accounting_available": True,
                    "allowed_applications": ["manuals_rag"],
                    "tailnet_name": "tailnet",
                },
                {
                    "node_id": "gpu-b",
                    "healthy": True,
                    "observed_at_epoch": rollout.time.time(),
                    "gpus": [{"gpu_id": "0"}],
                    "request_accounting_available": True,
                    "allowed_applications": ["manuals_rag"],
                    "tailnet_name": "tailnet",
                },
            ]
        },
    }
    monkeypatch.setattr(rollout, "_get_json", lambda _base, path, _token: responses[path])
    errors, warnings = rollout.verify_rollout(base_url="http://gateway", token="t" * 40)
    assert errors == ()
    assert warnings == ()


def test_rollout_verifier_reports_stale_and_unaccounted_node(monkeypatch):
    monkeypatch.setattr(
        rollout,
        "_get_json",
        lambda _base, path, _token: {"status": "ok"} if path == "/health" else {
            "nodes": [{
                "node_id": "gpu-a",
                "healthy": True,
                "observed_at_epoch": 1,
                "gpus": [],
                "request_accounting_available": False,
                "allowed_applications": [],
            }]
        },
    )
    errors, _warnings = rollout.verify_rollout(base_url="http://gateway", token="t" * 40, expected_nodes=1)
    assert any("stale heartbeat" in error for error in errors)
    assert any("no GPUs" in error for error in errors)
    assert any("proxy request accounting" in error for error in errors)
    assert any("application allowlist" in error for error in errors)
