import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from manuals_rag_inference.enrollment import EnrollmentManager
from manuals_rag_inference.entra import EntraTokenValidator, EntraValidatorConfig
from manuals_rag_inference.gateway import InferenceGateway, create_gateway_handler
from manuals_rag_inference.preflight import validate_gateway_environment


def _environment(**overrides):
    values = {
        "INFERENCE_GATEWAY_TOKEN": "t" * 40,
        "INFERENCE_GATEWAY_BIND": "100.64.0.10",
        "INFERENCE_GATEWAY_STATE_PATH": "/var/lib/manuals-rag-inference/gateway-state.json",
        "INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS": "true",
        "INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST": "true",
        "INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY": "true",
        "INFERENCE_APPLICATION_TOKENS_JSON": json.dumps({"manuals_rag": "a" * 40}),
        "INFERENCE_NODE_SECRETS_JSON": json.dumps({"gpu-a": "s" * 40}),
        "INFERENCE_NODE_AGENT_URLS_JSON": json.dumps({"gpu-a": "http://100.64.0.20:8765"}),
        "INFERENCE_MODEL_CATALOG_JSON": json.dumps({
            "qwen3.5:9b": {"size_bytes": 10_000_000_000, "min_vram_mb": 8000, "capabilities": ["chat"]},
        }),
    }
    values.update(overrides)
    return values


def _start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_entra_enrollment_issues_node_secret_and_records_identity(tmp_path):
    class FakeEntraValidator:
        def validate(self, access_token):
            assert access_token == "access-token"
            return {"subject": "entra-subject", "tenant_id": "tenant-id"}

    manager = EnrollmentManager(tmp_path / "gateway-state.json")
    gateway = InferenceGateway(
        enrollment=manager,
        entra_validator=FakeEntraValidator(),
        require_entra_enrollment=True,
    )
    enrolled = gateway.enroll_node_with_entra(
        access_token="access-token",
        node_id="gpu-a",
        endpoint="http://127.0.0.1:11434",
        agent_url="http://127.0.0.1:8765",
        tailnet_name="provis-tailnet",
    )
    record = manager.nodes()[0]
    assert enrolled["enrollment_method"] == "entra"
    assert enrolled["shared_secret"] == record.secret
    assert record.identity_subject == "entra-subject"
    assert record.identity_tenant == "tenant-id"
    with pytest.raises(PermissionError, match="Entra enrollment"):
        gateway.enroll_node(
            pairing_token="unused",
            node_id="gpu-b",
            endpoint="http://127.0.0.1:11434",
            agent_url="http://127.0.0.1:8766",
            tailnet_name="provis-tailnet",
        )


def test_http_entra_enrollment_uses_bearer_token():
    class FakeEntraValidator:
        def validate(self, access_token):
            assert access_token == "entra-access-token"
            return {"subject": "subject-1", "tenant_id": "tenant-1"}

    enrollment = EnrollmentManager()
    gateway = InferenceGateway(enrollment=enrollment, entra_validator=FakeEntraValidator(), require_entra_enrollment=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_gateway_handler(gateway))
    thread = _start(server)
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/nodes/enroll/entra",
            data=json.dumps({
                "node_id": "gpu-a",
                "endpoint": "http://127.0.0.1:11434",
                "agent_url": "http://127.0.0.1:8765",
                "tailnet_name": "provis-tailnet",
            }).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer entra-access-token"},
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode())
        assert result["enrollment_method"] == "entra"
        assert enrollment.nodes()[0].identity_subject == "subject-1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_public_enrollment_config_contains_only_bootstrap_values():
    validator = EntraTokenValidator(EntraValidatorConfig(
        tenant_id="tenant-1",
        audience="api://provis-gateway",
        issuer="https://login.microsoftonline.com/tenant-1/v2.0",
        jwks_url="https://login.microsoftonline.com/tenant-1/discovery/v2.0/keys",
    ))
    gateway = InferenceGateway(
        entra_validator=validator,
        entra_client_id="public-client-id",
        entra_scope="api://provis-gateway/.default",
        node_default_tailnet="provis-tailnet",
        node_default_applications=("provis_employee_agent",),
    )
    config = gateway.enrollment_config()
    assert config == {
        "tenant_id": "tenant-1",
        "client_id": "public-client-id",
        "scope": "api://provis-gateway/.default",
        "tailnet_name": "provis-tailnet",
        "allowed_applications": ["provis_employee_agent"],
        "port": 8765,
    }
    assert "secret" not in json.dumps(config).lower()


def test_operator_rollout_auth_does_not_require_application_identity():
    gateway = InferenceGateway(
        gateway_token="operator-token",
        application_tokens={"manuals_rag": "application-token"},
        require_application_identity=True,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_gateway_handler(gateway))
    thread = _start(server)
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/nodes",
            headers={"X-Gateway-Token": "operator-token"},
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert json.loads(response.read().decode()) == {"nodes": []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_preflight_requires_restricted_entra_configuration():
    result = validate_gateway_environment(_environment(
        INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT="true",
        INFERENCE_ENTRA_TENANT_ID="tenant-id",
        INFERENCE_ENTRA_AUDIENCE="api://provis-gateway",
    ))
    assert not result.ok
    assert any("ALLOWED_ROLES" in error for error in result.errors)

    accepted = validate_gateway_environment(_environment(
        INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT="true",
        INFERENCE_ENTRA_TENANT_ID="tenant-id",
        INFERENCE_ENTRA_AUDIENCE="api://provis-gateway",
        INFERENCE_ENTRA_CLIENT_ID="public-client-id",
        INFERENCE_ENTRA_SCOPE="api://provis-gateway/Node.Enroll",
        INFERENCE_ENTRA_ALLOWED_ROLES="Provis.InferenceNode.Enroll",
    ))
    assert accepted.ok


def test_preflight_does_not_require_entra_dependency_when_disabled():
    result = validate_gateway_environment(_environment(INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT="false"))
    assert result.ok
    assert not any("PyJWT" in error for error in result.errors)
