import json
import os
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from manuals_rag_inference.enrollment import EnrollmentManager
from manuals_rag_inference.gateway import InferenceGateway, create_gateway_handler, node_request_signature
from manuals_rag_inference.entra_client import returning_node_registration


def _config():
    return {
        "node_id": "johns-dell",
        "shared_secret": "existing-node-secret",
        "gateway_url": "https://gateway.example",
        "entra_tenant_id": "tenant-1",
        "entra_client_id": "client-1",
        "entra_scope": "api://gateway/Node.Enroll",
        "endpoint": "http://127.0.0.1:11434",
        "agent_url": "http://100.64.0.5:8765",
    }


def _args(config, verify, enroll):
    return {
        "config": config,
        "gateway_url": "https://gateway.example",
        "tenant_id": "tenant-1",
        "client_id": "client-1",
        "scope": "api://gateway/Node.Enroll",
        "access_token": "opaque-test-token",
        "node_id": "johns-dell",
        "verify": verify,
        "enroll": enroll,
    }


def test_new_empty_config_enrolls_once():
    calls = []
    created = {"node_id": "johns-dell", "shared_secret": "new-secret"}
    result = returning_node_registration(
        **_args({}, lambda **_: pytest.fail("new node must not be verified"), lambda: calls.append("enroll") or created)
    )
    assert result == created
    assert calls == ["enroll"]


def test_returning_node_verifies_and_reuses_unchanged_registration():
    saved = _config()
    calls = []
    result = returning_node_registration(
        **_args(saved, lambda **kwargs: calls.append(kwargs) or True, lambda: pytest.fail("must not enroll"))
    )
    assert result == saved
    assert result["shared_secret"] == saved["shared_secret"]
    assert len(calls) == 1


def test_repeated_sign_in_reverifies_without_enrolling_or_rotating_secret():
    saved = _config()
    verifies = []
    enrolls = []
    for _ in range(2):
        reused = returning_node_registration(
            **_args(saved, lambda **kwargs: verifies.append(kwargs) or True, lambda: enrolls.append(True))
        )
        assert reused["shared_secret"] == saved["shared_secret"]
    assert len(verifies) == 2
    assert enrolls == []


@pytest.mark.parametrize("field,value", [
    ("gateway_url", "https://other.example"),
    ("entra_tenant_id", "other-tenant"),
    ("entra_client_id", "other-client"),
    ("entra_scope", "api://other/Scope"),
    ("node_id", "other-node"),
    ("shared_secret", ""),
])
def test_mismatched_or_incomplete_saved_registration_fails_closed(field, value):
    saved = _config()
    saved[field] = value
    enrolls = []
    with pytest.raises(PermissionError):
        returning_node_registration(
            **_args(saved, lambda **_: pytest.fail("invalid registration must not be verified"), lambda: enrolls.append(True))
        )
    assert enrolls == []


def test_owner_verification_denial_never_falls_back_to_enrollment():
    enrolls = []
    with pytest.raises(PermissionError):
        returning_node_registration(
            **_args(_config(), lambda **_: False, lambda: enrolls.append(True))
        )
    assert enrolls == []


def test_gateway_verifies_current_entra_owner_tenant_and_saved_secret_only():
    class FakeValidator:
        def validate(self, token):
            identities = {
                "owner": {"subject": "subject-1", "tenant_id": "tenant-1"},
                "other-owner": {"subject": "subject-2", "tenant_id": "tenant-1"},
                "other-tenant": {"subject": "subject-1", "tenant_id": "tenant-2"},
            }
            if token not in identities:
                raise PermissionError("invalid token")
            return identities[token]

    enrollment = EnrollmentManager()
    gateway = InferenceGateway(enrollment=enrollment, entra_validator=FakeValidator(), require_entra_enrollment=True)
    created = gateway.enroll_node_with_entra(
        access_token="owner",
        node_id="johns-dell",
        endpoint="http://127.0.0.1:11434",
        agent_url="http://127.0.0.1:8765",
        tailnet_name="provis-tailnet",
    )
    challenge = "random-single-use-test-challenge"
    proof = {"node_id": created["node_id"], "challenge": challenge}
    signature = node_request_signature(proof, created["shared_secret"].encode())
    arguments = {"node_id": created["node_id"], "challenge": challenge, "signature": signature}

    assert gateway.verify_node_registration_with_entra(access_token="owner", **arguments) is True
    assert gateway.verify_node_registration_with_entra(access_token="other-owner", **arguments) is False
    assert gateway.verify_node_registration_with_entra(access_token="other-tenant", **arguments) is False
    assert gateway.verify_node_registration_with_entra(access_token="invalid", **arguments) is False
    assert gateway.verify_node_registration_with_entra(
        access_token="owner", node_id="johns-dell", challenge="other", signature=signature
    ) is False
    assert set(created) == {"node_id", "shared_secret", "endpoint", "agent_url", "tailnet_name", "enrollment_method"}


def test_revoked_or_non_entra_registration_is_not_reusable():
    class FakeValidator:
        def validate(self, token):
            return {"subject": "subject-1", "tenant_id": "tenant-1"}

    enrollment = EnrollmentManager()
    gateway = InferenceGateway(enrollment=enrollment, entra_validator=FakeValidator())
    pairing = enrollment.create_pairing_token()
    node = enrollment.enroll(
        pairing_token=pairing["token"],
        node_id="pairing-node",
        endpoint="http://127.0.0.1:11434",
        agent_url="http://127.0.0.1:8765",
        tailnet_name="provis-tailnet",
    )
    assert not gateway.verify_node_registration_with_entra(
        access_token="owner", node_id=node.node_id, challenge="challenge", signature="signature"
    )


@pytest.mark.skipif(os.name == "nt", reason="Windows sandbox may prohibit loopback sockets")
def test_http_registration_verification_returns_no_node_or_credential_data():
    class FakeValidator:
        def validate(self, token):
            if token == "owner":
                return {"subject": "subject-1", "tenant_id": "tenant-1"}
            if token == "other-owner":
                return {"subject": "subject-2", "tenant_id": "tenant-1"}
            raise PermissionError("invalid token")

    enrollment = EnrollmentManager()
    gateway = InferenceGateway(enrollment=enrollment, entra_validator=FakeValidator(), require_entra_enrollment=True)
    created = gateway.enroll_node_with_entra(
        access_token="owner",
        node_id="johns-dell",
        endpoint="http://127.0.0.1:11434",
        agent_url="http://127.0.0.1:8765",
        tailnet_name="provis-tailnet",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_gateway_handler(gateway))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def request(token):
            challenge = "unique-test-challenge-" + token
            signature = node_request_signature(
                {"node_id": created["node_id"], "challenge": challenge},
                created["shared_secret"].encode(),
            )
            req = Request(
                f"http://127.0.0.1:{server.server_port}/v1/nodes/verify-registration/entra",
                data=json.dumps({"node_id": created["node_id"], "challenge": challenge, "signature": signature}).encode(),
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            )
            return req

        with urlopen(request("owner"), timeout=5) as response:
            body = json.loads(response.read().decode())
            assert response.status == 200
            assert body == {"status": "accepted"}
            assert "shared_secret" not in json.dumps(body)
            assert "node_id" not in body
        try:
            urlopen(request("other-owner"), timeout=5)
            pytest.fail("mismatched owner must be denied")
        except HTTPError as error:
            body = json.loads(error.read().decode())
            assert error.code == 403
            assert body == {"status": "denied"}
            assert "shared_secret" not in json.dumps(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    enrollment.revoke(created["node_id"])
    assert not gateway.verify_node_registration_with_entra(
        access_token="owner", node_id=created["node_id"], challenge="challenge", signature="signature"
    )
