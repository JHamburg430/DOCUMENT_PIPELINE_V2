import json

from manuals_rag_inference.preflight import validate_gateway_environment, validate_node_environment


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


def test_production_preflight_accepts_private_node_agent_configuration():
    result = validate_gateway_environment(_environment())
    assert result.ok
    assert not result.errors


def test_preflight_rejects_weak_token_and_direct_fallback():
    result = validate_gateway_environment(_environment(
        INFERENCE_GATEWAY_TOKEN="short",
        INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS="false",
        INFERENCE_APPLICATION_TOKENS_JSON="{}",
    ))
    assert not result.ok
    assert any("32 characters" in error for error in result.errors)
    assert any("REQUIRE_NODE_AGENTS" in error for error in result.errors)
    assert all("short" not in error for error in result.errors)


def test_preflight_rejects_invalid_catalog_and_mismatched_nodes():
    result = validate_gateway_environment(_environment(
        INFERENCE_NODE_AGENT_URLS_JSON=json.dumps({"gpu-b": "not-a-url"}),
        INFERENCE_MODEL_CATALOG_JSON=json.dumps({"qwen3.5:9b": {"size_bytes": 0}}),
    ))
    assert not result.ok
    assert any("node IDs" in error for error in result.errors)
    assert any("invalid" in error for error in result.errors)


def _node_environment(**overrides):
    values = {
        "INFERENCE_NODE_ID": "gpu-a",
        "INFERENCE_GATEWAY_URL": "http://gateway.tailnet:8780",
        "INFERENCE_NODE_ADVERTISE_URL": "http://100.64.0.20:8765",
        "INFERENCE_NODE_BIND": "100.64.0.20",
        "INFERENCE_NODE_PORT": "8765",
        "OLLAMA_NODE_URL": "http://127.0.0.1:11434",
        "INFERENCE_NODE_PROXY_ONLY": "true",
        "INFERENCE_NODE_ALLOWED_APPLICATIONS": "manuals_rag,provis_employee_agent",
        "INFERENCE_NODE_CONFIG_PATH": "/var/lib/manuals-rag-inference-node/node.json",
    }
    values.update(overrides)
    return values


def test_node_preflight_accepts_hardened_tailnet_configuration():
    result = validate_node_environment(_node_environment())
    assert result.ok
    assert not result.errors


def test_node_preflight_rejects_loopback_binding_direct_mode_and_bootstrap_token():
    result = validate_node_environment(_node_environment(
        INFERENCE_NODE_BIND="127.0.0.1",
        INFERENCE_NODE_PROXY_ONLY="false",
        INFERENCE_GATEWAY_ENROLLMENT_TOKEN="one-time-token",
    ))
    assert not result.ok
    assert any("BIND" in error for error in result.errors)
    assert any("PROXY_ONLY" in error for error in result.errors)
    assert any("must not be persisted" in error for error in result.errors)
