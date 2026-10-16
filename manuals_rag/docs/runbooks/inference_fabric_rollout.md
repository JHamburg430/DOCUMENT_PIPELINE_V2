# PROVIS Inference Fabric rollout runbook

This runbook is for the always-on droplet that hosts the OpenClaw gateway. It
does not authorize a rollout on a workstation and does not require the GPU
workstation to remain online for the control plane to operate.

## 0. Preconditions

- The droplet is already a member of the organization's Tailscale network.
- Tailscale ACLs or grants allow the gateway identity to reach only the node
  agent port on approved GPU nodes.
- The droplet has a dedicated service account and a persistent path such as
  `/var/lib/manuals-rag-inference`.
- Each GPU node has Ollama installed locally and can reach the droplet over the
  tailnet.
- No node exposes its Ollama port or agent port on a public interface.
- Entra enrollment is enabled only after the gateway has a supported JWT
  signature/JWKS validator, tenant policy, and a registered public client for
  device-code sign-in. Keep the required flag false until those checks pass.

## 1. Install the gateway without starting it

Copy the reviewed repository content to the droplet's deployment directory,
then install the supplied systemd unit. Do not copy browser credentials,
Tailscale auth keys, Docker sockets, production volumes, or local model caches.

```bash
sudo install -d -o manuals-rag -g manuals-rag /opt/manuals-rag-inference
sudo install -d -o manuals-rag -g manuals-rag /var/lib/manuals-rag-inference
sudo install -m 0644 infra/systemd/inference-fabric-gateway.service \
  /etc/systemd/system/inference-fabric-gateway.service
sudo systemctl daemon-reload
```

Create the deployment environment in the systemd environment-file location,
using a secret manager or a mode-`0600` root-owned file. The file must include:

```text
INFERENCE_GATEWAY_TOKEN=<at least 32 random characters>
INFERENCE_GATEWAY_STATE_PATH=/var/lib/manuals-rag-inference/gateway-state.json
INFERENCE_GATEWAY_BIND=127.0.0.1
INFERENCE_GATEWAY_PORT=8780
INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS=true
INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST=true
INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY=true
INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT=true
INFERENCE_GATEWAY_ALLOW_PUBLIC_BIND=false
INFERENCE_MODEL_CATALOG_JSON=<approved JSON catalog>
INFERENCE_APPLICATION_TOKENS_JSON=<secret-manager supplied application-token map>
INFERENCE_ENTRA_TENANT_ID=<directory-tenant-id>
INFERENCE_ENTRA_AUDIENCE=<gateway-api-application-id-or-audience>
INFERENCE_ENTRA_ALLOWED_ROLES=<approved-app-role>
INFERENCE_ENTRA_ALLOWED_GROUPS=<approved-group-id>
```

The gateway may be exposed to approved tailnet clients through Tailscale
Serve/Service; do not bind it publicly just to avoid configuring Serve.

## 2. Run the preflight before starting

```bash
cd /opt/manuals-rag-inference/manuals_rag
PYTHONPATH=packages/inference_fabric/src \
  python3 scripts/check_inference_fabric_config.py
```

The result must be `inference_gateway_preflight=PASS`. The command does not
print token or node-secret values. A warning about no statically configured
nodes is acceptable before the first enrollment; it is not acceptable before
the first inference request.

When Entra enrollment is enabled, this same preflight also verifies that the
gateway runtime can import `PyJWT[crypto]`; it does not install anything.

After enrolling the nodes and enabling their agents, run the non-mutating
rollout verifier. It checks gateway health, node count, heartbeat freshness,
GPU reporting, proxy accounting, and application allowlists without issuing
inference, pull, eviction, or enrollment commands:

```bash
INFERENCE_GATEWAY_URL=http://gateway.tailnet:8780 \
INFERENCE_GATEWAY_TOKEN='<operator-token>' \
INFERENCE_EXPECTED_NODE_COUNT=2 \
PYTHONPATH=packages/inference_fabric/src \
  python3 scripts/check_inference_fabric_rollout.py
```

The verifier must pass before routing production callers through the gateway.

Each application deployment also receives only its own
`INFERENCE_APPLICATION_TOKEN`; do not reuse the gateway operator token or
another application's token.

## 3. Start in observe-only mode

```bash
sudo systemctl enable --now inference-fabric-gateway.service
curl --fail http://127.0.0.1:8780/health
```

Distribute a managed **PROVIS Inference Node** bundle containing the gateway
URL. The employee completes only the Microsoft device sign-in; the app
discovers the tenant, public client, API scope, node identity, and approved
application policy. The short-lived access token is used only for the
enrollment request and is never saved. Configure each node with:

```text
INFERENCE_NODE_ALLOWED_APPLICATIONS=manuals_rag,provis_employee_agent,openclaw
INFERENCE_NODE_PROXY_ONLY=true
```

The node agent must advertise its Tailscale address, not `127.0.0.1` or a
public address. Confirm its heartbeat reports GPUs, loaded models,
`request_accounting_available=true`, and the expected application allowlist.

Before enabling the node service, run the non-mutating node preflight. It must
pass with an explicit Tailscale/private bind, `INFERENCE_NODE_PROXY_ONLY=true`,
and no enrollment token left in the service environment:

```bash
cd /opt/manuals-rag-inference/manuals_rag
PYTHONPATH=packages/inference_fabric/src \
  python3 scripts/check_inference_node_config.py
```

The gateway's model catalog is authoritative at runtime; node callers cannot
lower the cataloged model-size or minimum-VRAM requirement in an inference
request. Unknown models and unsupported capability requests are rejected.

For reboot persistence, install the supplied
`infra/systemd/inference-node-agent.service` on the node after the local
Connect step. Create `/etc/manuals-rag/inference-node.env` as a root-owned
mode-`0600` file containing the gateway URL, node ID, Tailscale-advertised
agent URL, Ollama URL, `INFERENCE_NODE_PROXY_ONLY=true`, and the approved
application allowlist. Create `/var/lib/manuals-rag-inference-node` owned by
the service user. The service is intentionally non-root and has no Docker
socket, shell-control, or remote-install capability.

```bash
sudo install -d -o manuals-rag-node -g manuals-rag-node \
  /var/lib/manuals-rag-inference-node
sudo install -m 0644 infra/systemd/inference-node-agent.service \
  /etc/systemd/system/inference-node-agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now inference-node-agent.service
```

## 4. Verify admission and failover

Before routing production callers, verify all of the following from the
gateway's authenticated node view:

1. Fresh nodes are visible and stale nodes disappear after the heartbeat TTL.
2. An application is admitted only to an allowlisted node.
3. A request with insufficient VRAM is rejected unless model-pull/eviction
   policy explicitly permits the operation.
4. A cold model pull is issued only through the signed bounded command path.
5. Idle eviction is approved explicitly and never targets an active or
   scheduler-reserved model.
6. Taking one node offline leaves the other node routable and does not make the
   droplet gateway unavailable.

Record the node inventory, model catalog, test request IDs, selected node IDs,
and timestamps as the rollout evidence. Do not record secrets.

## 5. Cut over callers gradually

Set the gateway URL, token, application, and model catalog in the API and
worker deployment environment. Keep direct Ollama fallback disabled by policy
only after the gateway path has passed the failover gate. The RAG compatibility
client routes chat, generation, and embeddings through the gateway when
`INFERENCE_GATEWAY_URL` is set.

Roll out in this order:

1. Evaluation workloads.
2. RAG API and workers.
3. OpenClaw/ProVIS employee-agent callers.

Keep the previous direct endpoint available only as a time-bounded rollback
configuration, not as an untracked alternate production path.

## 6. Rollback

If admission, latency, or correctness regresses:

1. Stop routing new callers to the gateway.
2. Preserve gateway, node, and evaluation logs.
3. Revert only the caller environment variables to the prior endpoint.
4. Leave node agents running in observe-only mode unless they are the source of
   the fault.
5. Re-run the preflight and record the failure before retrying cutover.

Do not delete enrollment state, model caches, or prior Ollama versions during a
rollback. Revoke a compromised node credential through the gateway instead.

## Current evidence and blockers

The local implementation has **265 passing** focused scheduler, gateway,
node-agent, HTTP failover, RAG-client, provisioning, preflight, rollout-gate,
and API-auth tests. The current PC has no running `tailscaled`, the gateway is
not deployed on the droplet, and no real two-node failover evidence exists yet.
OpenClaw currently reports connected workstation nodes, but they are not
enrolled inference-fabric agents and do not provide signed inference-fabric
heartbeats. These are deployment gates, not reasons to weaken admission or
evidence checks.
