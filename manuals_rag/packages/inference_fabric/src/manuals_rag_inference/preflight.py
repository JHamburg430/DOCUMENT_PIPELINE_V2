"""Non-mutating production configuration checks for the inference gateway."""

from __future__ import annotations

import json
import ipaddress
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

from .enrollment import NODE_ID_PATTERN, validate_node_endpoint
from .gateway import validate_gateway_runtime_security


@dataclass(frozen=True)
class PreflightResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _json_object(raw: str, field: str, errors: list[str]) -> dict:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        errors.append(f"{field} is not valid JSON")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{field} must be a JSON object")
        return {}
    return value


def validate_gateway_environment(env: Mapping[str, str]) -> PreflightResult:
    errors: list[str] = []
    warnings: list[str] = []
    token = str(env.get("INFERENCE_GATEWAY_TOKEN") or "")
    bind = str(env.get("INFERENCE_GATEWAY_BIND") or "127.0.0.1")
    allow_public = str(env.get("INFERENCE_GATEWAY_ALLOW_PUBLIC_BIND") or "false").lower() == "true"
    try:
        validate_gateway_runtime_security(
            gateway_token=token,
            bind=bind,
            allow_public_bind=allow_public,
        )
    except RuntimeError as error:
        errors.append(str(error))
    if allow_public and bind in {"0.0.0.0", "::"}:
        warnings.append("wildcard gateway bind is explicitly approved; restrict exposure with Tailscale ACLs/firewall")
    if not str(env.get("INFERENCE_GATEWAY_STATE_PATH") or "").strip():
        errors.append("INFERENCE_GATEWAY_STATE_PATH is required for persistent enrollment state")
    if str(env.get("INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS") or "false").lower() != "true":
        errors.append("INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS must be true for production")
    if str(env.get("INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST") or "false").lower() != "true":
        errors.append("INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST must be true for production")
    require_app_identity = str(env.get("INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY") or "false").lower() == "true"
    if not require_app_identity:
        errors.append("INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY must be true for production")
    application_tokens = _json_object(str(env.get("INFERENCE_APPLICATION_TOKENS_JSON") or ""), "INFERENCE_APPLICATION_TOKENS_JSON", errors)
    if require_app_identity and not application_tokens:
        errors.append("INFERENCE_APPLICATION_TOKENS_JSON must contain application credentials when application identity is required")
    for application, application_token in application_tokens.items():
        if not str(application).strip() or not isinstance(application_token, str) or len(application_token) < 32:
            errors.append(f"application token for {application!r} is missing or too short")
    try:
        port = int(env.get("INFERENCE_GATEWAY_PORT") or "8780")
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        errors.append("INFERENCE_GATEWAY_PORT must be between 1 and 65535")

    secrets = _json_object(str(env.get("INFERENCE_NODE_SECRETS_JSON") or ""), "INFERENCE_NODE_SECRETS_JSON", errors)
    for node_id, secret in secrets.items():
        if not isinstance(secret, str) or len(secret) < 32:
            errors.append(f"node secret for {node_id!r} is missing or too short")
    agent_urls = _json_object(str(env.get("INFERENCE_NODE_AGENT_URLS_JSON") or ""), "INFERENCE_NODE_AGENT_URLS_JSON", errors)
    suffixes = tuple(item.strip() for item in str(env.get("INFERENCE_NODE_ALLOWED_HOST_SUFFIXES") or ".ts.net,.tailnet").split(",") if item.strip())
    for node_id, url in agent_urls.items():
        try:
            validate_node_endpoint(str(url), allowed_host_suffixes=suffixes, allow_local_endpoints=False)
        except ValueError:
            errors.append(f"node agent URL for {node_id!r} is invalid or outside the approved tailnet boundary")
    if secrets and agent_urls and set(secrets) != set(agent_urls):
        errors.append("node secret and node-agent URL node IDs do not match")
    if not secrets and not agent_urls:
        warnings.append("no statically configured nodes; enrollment state must provide nodes before inference")

    require_entra = str(env.get("INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT") or "false").lower() == "true"
    entra_tenant = str(env.get("INFERENCE_ENTRA_TENANT_ID") or "").strip()
    entra_audience = str(env.get("INFERENCE_ENTRA_AUDIENCE") or "").strip()
    entra_client_id = str(env.get("INFERENCE_ENTRA_CLIENT_ID") or "").strip()
    entra_scope = str(env.get("INFERENCE_ENTRA_SCOPE") or "").strip()
    if require_entra and (not entra_tenant or not entra_audience):
        errors.append("INFERENCE_ENTRA_TENANT_ID and INFERENCE_ENTRA_AUDIENCE are required when Entra enrollment is required")
    if bool(entra_tenant) != bool(entra_audience):
        errors.append("INFERENCE_ENTRA_TENANT_ID and INFERENCE_ENTRA_AUDIENCE must be configured together")
    if require_entra and not str(env.get("INFERENCE_ENTRA_ALLOWED_ROLES") or "").strip() and not str(env.get("INFERENCE_ENTRA_ALLOWED_GROUPS") or "").strip():
        errors.append("INFERENCE_ENTRA_ALLOWED_ROLES or INFERENCE_ENTRA_ALLOWED_GROUPS must restrict Entra enrollment")
    if (require_entra or entra_tenant or entra_audience) and not entra_client_id:
        errors.append("INFERENCE_ENTRA_CLIENT_ID is required when Microsoft Entra enrollment is configured")
    if require_entra and not entra_scope:
        errors.append("INFERENCE_ENTRA_SCOPE is required when Microsoft Entra enrollment is required")
    if require_entra or entra_tenant or entra_audience:
        try:
            import jwt  # noqa: F401
            from jwt import PyJWKClient  # noqa: F401
        except ImportError:
            errors.append("PyJWT[crypto] is required when Microsoft Entra enrollment is configured")

    catalog = _json_object(str(env.get("INFERENCE_MODEL_CATALOG_JSON") or ""), "INFERENCE_MODEL_CATALOG_JSON", errors)
    if not catalog:
        errors.append("INFERENCE_MODEL_CATALOG_JSON must contain at least one model")
    for model, item in catalog.items():
        if not isinstance(item, dict):
            errors.append(f"model catalog entry {model!r} must be an object")
            continue
        try:
            if int(item.get("size_bytes") or 0) <= 0 or int(item.get("min_vram_mb") or 0) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"model catalog entry {model!r} has invalid size_bytes/min_vram_mb")
        if not isinstance(item.get("capabilities") or [], list) or not item.get("capabilities"):
            errors.append(f"model catalog entry {model!r} must declare capabilities")
    return PreflightResult(tuple(errors), tuple(warnings))


def validate_node_environment(env: Mapping[str, str]) -> PreflightResult:
    """Validate a production node-agent environment without reading secrets."""
    errors: list[str] = []
    warnings: list[str] = []
    suffixes = tuple(
        item.strip()
        for item in str(env.get("INFERENCE_NODE_ALLOWED_HOST_SUFFIXES") or ".ts.net,.tailnet").split(",")
        if item.strip()
    )

    node_id = str(env.get("INFERENCE_NODE_ID") or "")
    if not NODE_ID_PATTERN.fullmatch(node_id):
        errors.append("INFERENCE_NODE_ID is required and must contain only approved node-id characters")

    gateway_url = str(env.get("INFERENCE_GATEWAY_URL") or "").strip()
    if not gateway_url:
        errors.append("INFERENCE_GATEWAY_URL is required")
    else:
        try:
            validate_node_endpoint(gateway_url, allowed_host_suffixes=suffixes, allow_local_endpoints=False)
        except ValueError:
            errors.append("INFERENCE_GATEWAY_URL is outside the approved tailnet/private boundary")

    advertise_url = str(env.get("INFERENCE_NODE_ADVERTISE_URL") or "").strip()
    if not advertise_url:
        errors.append("INFERENCE_NODE_ADVERTISE_URL is required")
    else:
        try:
            validate_node_endpoint(advertise_url, allowed_host_suffixes=suffixes, allow_local_endpoints=False)
        except ValueError:
            errors.append("INFERENCE_NODE_ADVERTISE_URL is outside the approved tailnet/private boundary")

    ollama_url = str(env.get("OLLAMA_NODE_URL") or "http://127.0.0.1:11434").strip()
    try:
        validate_node_endpoint(ollama_url, allowed_host_suffixes=suffixes, allow_local_endpoints=True)
    except ValueError:
        errors.append("OLLAMA_NODE_URL must be a private/local Ollama endpoint")

    bind = str(env.get("INFERENCE_NODE_BIND") or "").strip()
    if not bind or bind in {"0.0.0.0", "::", "localhost", "127.0.0.1", "::1"}:
        errors.append("INFERENCE_NODE_BIND must be an explicit non-loopback tailnet/private address")
    else:
        try:
            address = ipaddress.ip_address(bind.strip("[]"))
            if address.is_loopback or address.is_unspecified or not (address.is_private or address in ipaddress.ip_network("100.64.0.0/10")):
                errors.append("INFERENCE_NODE_BIND must be an explicit non-loopback tailnet/private address")
        except ValueError:
            if not any(bind.lower().endswith(suffix.lower()) for suffix in suffixes):
                errors.append("INFERENCE_NODE_BIND hostname is outside the approved tailnet boundary")

    try:
        port = int(env.get("INFERENCE_NODE_PORT") or "8765")
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        errors.append("INFERENCE_NODE_PORT must be between 1 and 65535")

    if str(env.get("INFERENCE_NODE_PROXY_ONLY") or "false").lower() != "true":
        errors.append("INFERENCE_NODE_PROXY_ONLY must be true for production")
    applications = tuple(item.strip() for item in str(env.get("INFERENCE_NODE_ALLOWED_APPLICATIONS") or "").split(",") if item.strip())
    if not applications:
        errors.append("INFERENCE_NODE_ALLOWED_APPLICATIONS must contain at least one application")
    if str(env.get("INFERENCE_GATEWAY_ENROLLMENT_TOKEN") or "").strip():
        errors.append("INFERENCE_GATEWAY_ENROLLMENT_TOKEN must not be persisted in the production service environment")
    if not str(env.get("INFERENCE_NODE_CONFIG_PATH") or "").strip():
        warnings.append("INFERENCE_NODE_CONFIG_PATH is not explicit; the user-home default will be used")
    return PreflightResult(tuple(errors), tuple(warnings))
