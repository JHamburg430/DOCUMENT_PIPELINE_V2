# PROVIS Inference Gateway — droplet handoff

This handoff is for the agent with access to the OpenClaw droplet. The gateway
belongs on the always-on droplet beside OpenClaw; it does **not** belong in the
local RAG Docker compose stack and it does not require Ollama on the droplet.
GPU workstations run the PROVIS Inference Node and connect over Tailscale.

## Scope and safety

- Deploy only the dedicated `inference-fabric-gateway.service`.
- Do not restart OpenClaw, RAG API/UI/worker containers, databases, Qdrant,
  Redis, or node services.
- Do not copy Tailscale keys, browser credentials, model caches, Docker socket,
  or production volumes.
- Do not print or commit values from the gateway environment file.
- Do not enable production callers until preflight and node rollout checks pass.

## 1. Checkout the code

Use the pushed commit supplied by the requesting agent. Preserve any unrelated
droplet work. The expected deployment directory is:

```text
/opt/manuals-rag-inference
```

The gateway source is under:

```text
/opt/manuals-rag-inference/manuals_rag/packages/inference_fabric/src
```

## 2. Prepare an isolated Python runtime

Use a dedicated virtual environment for the gateway. Do not install into the
system Python and do not install the full RAG requirements on the droplet.

```bash
cd /opt/manuals-rag-inference
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r manuals_rag/requirements-inference-gateway-auth.txt
```

If the droplet cannot use the pinned/auth dependency file as written, stop and
report the exact package or Python error; do not bypass JWT validation.

## 3. Configure the service

Create `/etc/manuals-rag/inference-gateway.env` as root-owned mode `0600`.
Start from `manuals_rag/infra/systemd/inference-gateway.env.example`, but fill
the values from the approved secret manager/configuration source. Required
values include:

```text
INFERENCE_GATEWAY_TOKEN=<strong operator token>
INFERENCE_GATEWAY_STATE_PATH=/var/lib/manuals-rag-inference/gateway-state.json
INFERENCE_GATEWAY_BIND=127.0.0.1
INFERENCE_GATEWAY_PORT=8780
INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS=true
INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST=true
INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY=true
INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT=true
INFERENCE_GATEWAY_ALLOW_PUBLIC_BIND=false
INFERENCE_APPLICATION_TOKENS_JSON=<approved secret-manager map>
INFERENCE_MODEL_CATALOG_JSON=<approved model catalog>
INFERENCE_ENTRA_TENANT_ID=b97c06d3-1cd7-47cf-a95a-380d4dd2eb72
INFERENCE_ENTRA_AUDIENCE=fab62a38-0489-4d21-b8d1-63645712d271
INFERENCE_ENTRA_ALLOWED_ROLES=InferenceNode.Enroll
INFERENCE_ENTRA_ALLOWED_GROUPS=<approved enrollment group IDs if used>
INFERENCE_ENTRA_CLIENT_ID=68cfe37b-6fbe-4a73-8df0-db29dd8cd690
INFERENCE_ENTRA_SCOPE=api://fab62a38-0489-4d21-b8d1-63645712d271/Node.Enroll
INFERENCE_NODE_DEFAULT_TAILNET=<tailnet name>
INFERENCE_NODE_DEFAULT_APPLICATIONS=manuals_rag,provis_employee_agent,openclaw
INFERENCE_NODE_DEFAULT_PORT=8765
```

Use at least one of the approved Entra role/group restrictions. Do not set the
Entra-required flag true until the tenant, audience, app role/group, and public
client registration have been verified.

## 4. Install and validate the systemd unit

The unit must use the dedicated virtualenv. Update its Python paths from
`/usr/bin/python3` to `/opt/manuals-rag-inference/.venv/bin/python` before
starting it, then install the unit:

```bash
sudo install -d -o manuals-rag -g manuals-rag /var/lib/manuals-rag-inference
sudo install -m 0644 manuals_rag/infra/systemd/inference-fabric-gateway.service \
  /etc/systemd/system/inference-fabric-gateway.service
sudo systemctl daemon-reload
cd /opt/manuals-rag-inference/manuals_rag
PYTHONPATH=packages/inference_fabric/src \
  /opt/manuals-rag-inference/.venv/bin/python scripts/check_inference_fabric_config.py
```

Expected result: `inference_gateway_preflight=PASS`. A warning for zero
statically configured nodes is acceptable before node enrollment. Any Entra,
model-catalog, application-token, or dependency error is a blocker.

## 5. Start and verify only this service

```bash
sudo systemctl enable --now inference-fabric-gateway.service
curl --fail http://127.0.0.1:8780/health
sudo systemctl --no-pager --full status inference-fabric-gateway.service
```

Expose the loopback service to approved Tailscale clients using the existing
Tailscale Serve/Service policy. Do not bind `0.0.0.0` or open a public port.

## 6. Verify node enrollment

Use the managed PROVIS Inference Node bundle. Employees should see only the
Microsoft 365 sign-in action; the app discovers the public client/tenant/scope
from `GET /v1/node/enrollment-config` and auto-configures node settings.
Confirm the gateway
returns a node secret and stores only the enrollment audit subject/tenant; the
access token must not be written to the node config.

Then run the non-mutating rollout verifier with the operator token supplied
through the protected environment, not in shell history:

```bash
INFERENCE_GATEWAY_URL=http://gateway.tailnet:8780 \
INFERENCE_GATEWAY_TOKEN='<operator-token>' \
INFERENCE_EXPECTED_NODE_COUNT=<approved-count> \
PYTHONPATH=packages/inference_fabric/src \
  /opt/manuals-rag-inference/.venv/bin/python scripts/check_inference_fabric_rollout.py
```

Do not route RAG or OpenClaw traffic through the gateway until this verifier
passes, node heartbeats are fresh, GPUs are reported, application allowlists
are correct, and at least one node-agent inference smoke test has passed.

## Report back

Report separately:

1. Commit checked out and deployment path.
2. Preflight result and any warnings.
3. Service status and `/health` result.
4. Enrolled node count and heartbeat/VRAM evidence without secrets.
5. Exact remaining blocker before caller cutover.
