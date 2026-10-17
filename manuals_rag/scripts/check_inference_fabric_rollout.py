#!/usr/bin/env python3
"""Non-mutating post-deployment checks for the inference fabric."""

from __future__ import annotations

import json
import os
import sys
import time
from urllib.request import Request, urlopen


def _get_json(base_url: str, path: str, token: str) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Accept": "application/json", "X-Gateway-Token": token},
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def verify_rollout(*, base_url: str, token: str, expected_nodes: int = 2, heartbeat_ttl_seconds: float = 15.0) -> tuple[tuple[str, ...], tuple[str, ...]]:
    errors: list[str] = []
    warnings: list[str] = []
    if len(token) < 32:
        errors.append("INFERENCE_GATEWAY_TOKEN must be at least 32 characters")
        return tuple(errors), tuple(warnings)
    try:
        health = _get_json(base_url, "/health", token)
        if health.get("status") != "ok":
            errors.append("gateway health did not return status=ok")
        payload = _get_json(base_url, "/v1/nodes", token)
    except Exception as error:
        errors.append(f"gateway probe failed: {error.__class__.__name__}")
        return tuple(errors), tuple(warnings)

    nodes = payload.get("nodes") or []
    if len(nodes) < expected_nodes:
        errors.append(f"expected at least {expected_nodes} enrolled nodes; found {len(nodes)}")
    now = time.time()
    for node in nodes:
        node_id = str(node.get("node_id") or "<unnamed>")
        if not node.get("healthy"):
            errors.append(f"node {node_id} is unhealthy")
        observed = float(node.get("observed_at_epoch") or 0)
        if observed <= 0 or now - observed > heartbeat_ttl_seconds:
            errors.append(f"node {node_id} has a stale heartbeat")
        if not node.get("gpus"):
            errors.append(f"node {node_id} reports no GPUs")
        if not node.get("request_accounting_available"):
            errors.append(f"node {node_id} does not provide proxy request accounting")
        if not node.get("allowed_applications"):
            errors.append(f"node {node_id} has no application allowlist")
    if nodes and len({str(node.get("tailnet_name") or "") for node in nodes}) != 1:
        warnings.append("enrolled nodes report different tailnet names; verify this is intentional")
    return tuple(errors), tuple(warnings)


def main() -> int:
    base_url = os.getenv("INFERENCE_GATEWAY_URL", "").strip()
    token = os.getenv("INFERENCE_GATEWAY_TOKEN", "")
    if not base_url:
        print("inference_fabric_rollout=FAIL")
        print("error=INFERENCE_GATEWAY_URL is required")
        return 1
    expected_nodes = int(os.getenv("INFERENCE_EXPECTED_NODE_COUNT", "2"))
    errors, warnings = verify_rollout(
        base_url=base_url,
        token=token,
        expected_nodes=max(1, expected_nodes),
        heartbeat_ttl_seconds=float(os.getenv("INFERENCE_HEARTBEAT_TTL_SECONDS", "15")),
    )
    print("inference_fabric_rollout=" + ("PASS" if not errors else "FAIL"))
    for warning in warnings:
        print("warning=" + warning)
    for error in errors:
        print("error=" + error)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
