from manuals_rag_inference.node_presentation import public_events, utilization
from manuals_rag_inference.models import GpuSnapshot


def test_untrusted_fields_cannot_reach_panel():
    marker = "PRIVATE_PROMPT_MODEL_SECRET"
    rows = public_events([{"timestamp": 1, "operation": marker, "model": marker, "status": marker,
                           "error": marker, "prompt": marker, "response": marker}])
    assert marker not in str(rows)
    assert rows[0][1:] == ("Node activity", "Updated")


def test_heartbeat_noise_does_not_hide_work_events():
    events = [{"timestamp": 1, "operation": "heartbeat", "status": "accepted"}] * 80
    events += [{"timestamp": 2, "operation": "model_warm_started", "status": "started"}]
    events += [{"timestamp": 3, "operation": "heartbeat", "status": "accepted"}] * 80
    assert len(public_events(events)) == 2
    assert public_events(events)[1][1] == "Preparing compute"


def test_missing_and_bounded_utilization():
    assert utilization(()) == (0, 0, 0)
    assert utilization((GpuSnapshot("0", "private gpu", 100, 200, 120),)) == (100, 100, 100)
