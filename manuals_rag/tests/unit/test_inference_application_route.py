import json
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_inference_node_broker_gateway import broker, envelope, serving
from manuals_rag_inference.gateway import create_gateway_handler


def app_call(url, body, token="application-test-token", path="/v1/applications/inference"):
    return urlopen(Request(url + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "X-Inference-Application-Token": token}), timeout=4)


def test_scoped_application_inference_without_operator_access(broker):
    with serving(create_gateway_handler(broker)) as url:
        with app_call(url, envelope()) as response:
            assert json.load(response)["routing"]["node_id"] == "worker"
        for path, body in (("/v1/nodes/pairing-token", {}), ("/v1/nodes/revoke", {"node_id": "worker"}),
                           ("/v1/inference", {"request": {**envelope()["request"], "workload": "chat", "model_size_bytes": 1, "min_vram_mb": 1}})):
            with pytest.raises(HTTPError) as rejected:
                app_call(url, body, path=path)
            assert rejected.value.code == 401
    assert not broker.scheduler._reservations


@pytest.mark.parametrize("token,application", [("", "manuals_rag"), ("wrong", "manuals_rag"), ("application-test-token", "other")])
def test_application_identity_cannot_be_substituted(broker, token, application):
    body = envelope()
    body["request"]["application"] = application
    with serving(create_gateway_handler(broker)) as url:
        with pytest.raises(HTTPError) as rejected:
            app_call(url, body, token)
        assert rejected.value.code == 401
    assert not broker.scheduler._reservations


def test_server_worker_policy_cannot_be_widened_by_caller(broker):
    broker.application_worker_allowlist = {"manuals_rag": ("ingress",)}
    body = envelope()
    body["request"]["allowed_node_ids"] = ["worker"]
    with serving(create_gateway_handler(broker)) as url:
        with pytest.raises(HTTPError) as rejected:
            app_call(url, body)
        assert rejected.value.code == 503
    assert not broker.scheduler._reservations
