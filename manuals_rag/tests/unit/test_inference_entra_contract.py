import io
import json
import sys
import types

import pytest

from manuals_rag_inference import entra_client
from manuals_rag_inference.entra import EntraTokenValidator, EntraValidatorConfig
from manuals_rag_inference.gateway import AuthenticationError, InferenceGateway


def test_http_400_oauth_pending_and_slow_down_are_returned_to_poller(monkeypatch):
    from urllib.error import HTTPError
    from urllib.request import Request

    responses = iter([
        {"error": "authorization_pending"},
        {"error": "slow_down"},
        {"access_token": "access-token", "token_type": "Bearer"},
    ])
    waits = []
    real_post_form = entra_client._post_form

    def fake_post(*_args, **_kwargs):
        return next(responses)

    monkeypatch.setattr(entra_client, "_post_form", fake_post)
    monkeypatch.setattr(entra_client.time, "sleep", waits.append)
    result = entra_client.poll_device_token(
        tenant_id="tenant",
        client_id="client",
        device_code="device-code",
        interval=2,
        expires_in=30,
    )
    assert result["access_token"] == "access-token"
    assert waits == [2, 7]

    def fake_urlopen(_request: Request, timeout: float):
        assert timeout == 10.0
        return (_ for _ in ()).throw(HTTPError(
            "https://login.microsoftonline.com/token",
            400,
            "Bad Request",
            {},
            io.BytesIO(json.dumps({"error": "authorization_pending"}).encode()),
        ))

    monkeypatch.setattr(entra_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(entra_client, "_post_form", real_post_form)
    assert entra_client._post_form("https://login.microsoftonline.com/token", {}) == {"error": "authorization_pending"}


@pytest.mark.parametrize("oauth_error,expected", [
    ("authorization_declined", PermissionError),
    ("access_denied", PermissionError),
    ("expired_token", TimeoutError),
    ("invalid_grant", TimeoutError),
])
def test_device_code_denial_and_expiry_are_cleanly_reported(monkeypatch, oauth_error, expected):
    monkeypatch.setattr(entra_client, "_post_form", lambda *_args, **_kwargs: {"error": oauth_error})
    with pytest.raises(expected):
        entra_client.poll_device_token(
            tenant_id="tenant", client_id="client", device_code="device", expires_in=30,
        )


def _validator(monkeypatch, claims):
    class FakeJWKClient:
        def __init__(self, _url):
            pass

        def get_signing_key_from_jwt(self, _token):
            return types.SimpleNamespace(key="key")

    fake_jwt = types.SimpleNamespace(
        PyJWKClient=FakeJWKClient,
        decode=lambda *_args, **_kwargs: claims,
    )
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)
    return EntraTokenValidator(EntraValidatorConfig(
        tenant_id="tenant",
        audience="gateway-audience",
        issuer="https://issuer",
        jwks_url="https://jwks",
        client_id="desktop-client",
        required_scope="api://gateway/Node.Enroll",
        allowed_roles=("InferenceNode.Enroll",),
    ))


def test_entra_validator_requires_public_client_and_delegated_scope(monkeypatch):
    claims = {
        "exp": 2,
        "iat": 1,
        "iss": "https://issuer",
        "aud": "gateway-audience",
        "tid": "tenant",
        "sub": "employee",
        "azp": "desktop-client",
        "scp": "Node.Enroll",
        "roles": ["InferenceNode.Enroll"],
    }
    validator = _validator(monkeypatch, claims)
    assert validator.validate("signed-token")["subject"] == "employee"

    claims["scp"] = "User.Read"
    with pytest.raises(PermissionError, match="delegated enrollment scope"):
        validator.validate("signed-token")

    claims["scp"] = "Node.Enroll"
    claims["azp"] = "another-client"
    with pytest.raises(PermissionError, match="client mismatch"):
        validator.validate("signed-token")


def test_app_only_role_token_is_rejected_without_scope(monkeypatch):
    claims = {
        "exp": 2,
        "iat": 1,
        "iss": "https://issuer",
        "aud": "gateway-audience",
        "tid": "tenant",
        "sub": "service-principal",
        "azp": "desktop-client",
        "roles": ["InferenceNode.Enroll"],
    }
    validator = _validator(monkeypatch, claims)
    with pytest.raises(PermissionError, match="delegated enrollment scope"):
        validator.validate("signed-token")


def test_operator_auth_is_separate_from_application_inference_auth():
    gateway = InferenceGateway(
        gateway_token="operator-token",
        application_tokens={"manuals_rag": "application-token"},
        require_application_identity=True,
    )
    gateway.authorize_operator("operator-token")
    with pytest.raises(AuthenticationError, match="application identity"):
        gateway.authorize_inference("operator-token", application="manuals_rag")
    gateway.authorize_inference(
        "operator-token", application="manuals_rag", application_token="application-token",
    )

