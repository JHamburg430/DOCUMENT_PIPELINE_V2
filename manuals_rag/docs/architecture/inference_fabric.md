# PROVIS Inference Fabric

## Purpose

The PROVIS inference fabric provides one application-facing scheduling contract over
multiple Ollama servers. It is local-first: documents, prompts, responses, and
production traffic remain on the private network. Tailscale supplies the
private transport and node identity; the fabric still performs its own
application authorization and command signing.

## Deployment topology

The gateway is a control-plane service and must run on the always-on droplet
that hosts the OpenClaw gateway, not on any workstation that also hosts an
Ollama server. The droplet is the only application-facing inference endpoint:

```text
RAG / OpenClaw applications
             |
      Tailscale service identity
             |
      inference gateway (droplet)
        /          |          \
       /           |           \
 Ollama node A  Ollama node B  Ollama node C
  + node agent   + node agent   + node agent
```

Each node agent independently maintains a Tailscale connection to the
droplet and publishes signed heartbeats. A workstation outage therefore
removes only that workstation's node from the eligible pool; it must not make
the gateway or the other nodes unavailable. Applications must never use a
workstation Ollama URL as their primary endpoint after cutover.

The droplet deployment should bind the gateway to loopback behind a Tailscale
Service/Serve identity, or to the droplet's Tailscale interface only. It must
not be exposed on the public interface. The gateway has no local Ollama
dependency and should remain useful when every node is temporarily absent by
returning an explicit admission failure and node-health state.

## Components

### Central scheduler

The scheduler receives an `InferenceRequest` with the application, workload,
model, model-size/VRAM requirement, capabilities, priority, and whether model
pull or eviction is allowed. It admits only healthy nodes with a fresh
heartbeat, permitted application access, required capabilities, queue headroom,
and sufficient VRAM. Multi-GPU placement is the default: known CUDA devices are
selected by stable GPU UUID and VRAM is reserved per device, while a request can
explicitly opt into `single_gpu`. A warm model receives a strong placement preference, but
VRAM headroom and queue pressure still affect the final score.

The gateway's model catalog is authoritative in production. It rejects models
that are not cataloged, rejects capability requests the catalog does not
declare, and replaces caller-supplied model-size/minimum-VRAM values with the
catalog values before admission. This prevents a caller from understating
resource requirements to influence placement. Production startup fails if the
catalog is empty. The forwarded Ollama payload must use the same model identity
as the admitted request, and direct model-pull commands are also restricted to
catalog entries and their cataloged size.

The gateway prototype exposes:

- `GET /health`
- `GET /v1/nodes` (gateway client token required)
- `POST /v1/nodes/heartbeat` (per-node HMAC required)
- `POST /v1/nodes/pairing-token` (gateway client token required)
- `POST /v1/nodes/enroll` (short-lived one-time pairing token required)
- `POST /v1/nodes/enroll/entra` (Microsoft Entra API access token required)
- `POST /v1/nodes/revoke` (gateway client token required)
- `POST /v1/inference` (gateway client token required)
- `POST /v1/node-command` (gateway client token required; forwards only the
  bounded signed node commands)
- `POST /v1/maintenance/evict-idle` (gateway client token and explicit
  maintenance approval required)

### Node agent

`manuals_rag/scripts/run_inference_node_agent.py` exposes:

- `GET /health`
- `GET /v1/node/snapshot`
- `POST /v1/node/inference` (signed gateway forwarding path)
- `POST /v1/node/command`

The agent reports GPU state from `nvidia-smi` and loaded models from Ollama's
local process endpoint. It also reports the optional ONNX Runtime providers and
Intel OpenVINO devices when configured. It exposes only two control actions:
`pull_model` and `evict_model`. It does not expose Docker, SSH, or arbitrary
shell execution.

### ONNX and Intel integrated GPUs

ONNX is an optional, manifest-driven inference backend. Install the pinned
requirements in `manuals_rag/requirements-inference-onnx.txt` on nodes that
need it, set `INFERENCE_ONNX_MODELS_JSON` to an allowlist of absolute `.onnx`
paths, and use an authoritative model-catalog entry with `backend: "onnx"` and
the `onnx` capability. The ONNX request contract is tensor-based:
`{"model": "name", "inputs": {"input_name": [[...]]}}`; it is not an
automatic conversion of arbitrary chat models.

The portable source bundle expects those optional wheels in the node's Python
environment. The self-contained Linux executable embeds them when the build
machine has them installed; otherwise it remains an Ollama-only executable.

When `OpenVINOExecutionProvider` is present, the node defaults to OpenVINO's
`AUTO` device selection, with `GPU.0` as the reported integrated-GPU device
when the driver exposes it. `INFERENCE_ONNX_DEVICE` can select `GPU`, `GPU.0`,
`MULTI`, or `HETERO`; `INFERENCE_ONNX_OPENVINO_DEVICES` controls the inventory
reported to the scheduler. The provider is optional, so an Ollama-only node
continues to run without NumPy or ONNX Runtime.

Commands require an HMAC signature, a recent timestamp, an allowed model
prefix configured on the node, and a model-size limit. Tailscale grants/tags must restrict which
scheduler identity can reach the node-agent port; the HMAC remains an
application-layer defense if network policy is misconfigured.

Forwarded inference envelopes also carry the gateway request ID and a signed
issue timestamp. The node agent rejects missing/stale timestamps and duplicate
request IDs, preventing replay of a captured inference request within the
allowed clock-skew window.

Heartbeat ingress likewise rejects future, stale, and non-monotonic signed
observations before they enter the scheduler registry. A node that stops
heartbeating therefore ages out without accepting an old captured heartbeat as
fresh state.

The central gateway additionally requires `INFERENCE_GATEWAY_TOKEN` for client
inference and control requests. Heartbeats use a separate per-node secret. Do
not put either value in a repository, compose file, or browser client.
Production startup rejects gateway tokens shorter than 32 characters and
wildcard binds unless `INFERENCE_GATEWAY_ALLOW_PUBLIC_BIND=true` is explicitly
approved. Set `INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS=true` for production so
the gateway cannot fall back to direct Ollama forwarding when a node-agent URL
is missing.
Set `INFERENCE_GATEWAY_REQUIRE_APPLICATION_ALLOWLIST=true` as well, and set
`INFERENCE_NODE_ALLOWED_APPLICATIONS` on each node. In that mode an empty node
allowlist is rejected rather than interpreted as access for every application.
For production caller identity, set
`INFERENCE_GATEWAY_REQUIRE_APPLICATION_IDENTITY=true` and provide an
`INFERENCE_APPLICATION_TOKENS_JSON` mapping of application names to distinct
long-lived secret values through a secret manager. Each caller sends only its
own `INFERENCE_APPLICATION_TOKEN`; the gateway checks that token against the
application in the request rather than trusting a self-asserted name.
The supplied node systemd unit also runs the node preflight before launch; it
requires an explicit tailnet/private bind, proxy-only request accounting, a
non-empty application allowlist, and no persisted enrollment token.
The runner repeats the same checks when `INFERENCE_NODE_REQUIRE_PRODUCTION=true`
so a manual launch cannot bypass the service-unit gate.

### Node connection UX

The node application does not require a user to enter a Tailscale auth key.
The machine must already be joined to the organization's tailnet by its normal
device-management process. The node runner and local Connect page automatically
use the machine's `tailscale ip -4` result when it is a valid
`100.64.0.0/10` address; otherwise they require an explicit production
advertised URL and never accept a public address. The application then uses the
private tailnet path to pair with the droplet:

1. An administrator builds the managed PROVIS Inference Node bundle with the
   gateway URL. The bundle contains no tenant secret, client secret, or node
   credential.
2. The employee clicks **Sign in with Microsoft 365**. The app fetches the
   public tenant/client/scope and node policy from the gateway, auto-detects
   the hostname, Tailscale endpoint, local Ollama endpoint, and node ID, then
   completes Microsoft's device-code flow.
3. The short-lived access token is sent only to the Entra enrollment endpoint.
   The gateway returns a unique node secret, which the app stores in a local
   mode-`0600` configuration file and uses it for signed heartbeats, inference,
   and control commands on later starts.
4. An administrator can revoke a node. Its secret is removed from the gateway
   registry and all subsequent heartbeats/commands fail.

This provides application-level pairing without distributing Tailscale keys.
The droplet gateway remains Tailscale-only; the pairing token is not a
replacement for tailnet membership.
Enrollment validates both the Ollama endpoint and node-agent endpoint against
approved tailnet host suffixes or private/Tailscale address ranges. Production
enrollment rejects public endpoints; loopback is reserved for local test mode.

### Microsoft Entra ID enrollment

Microsoft 365 credentials replace the manual pairing-token handoff for
human-approved enrollment when `INFERENCE_GATEWAY_REQUIRE_ENTRA_ENROLLMENT=true`.
They do not become the node's permanent machine credential. The flow is:

1. The PROVIS Inference Node app starts Microsoft's device-code sign-in flow
   with a public client ID and opens the verification page.
2. The droplet gateway validates the resulting access token for its own API:
   signature and signing-key issuer, expiration/not-before, tenant, audience,
   client/actor, and an allowed administrator group or application role.
3. After the policy check, the gateway issues the node's unique local secret
   and records the Entra subject and tenant as enrollment audit metadata.
4. Subsequent heartbeats, inference forwarding, and node commands use the
   node secret and the existing HMAC protocol. Entra credentials are not sent
   with inference requests and are not stored on the node.

This removes copying a pairing token while retaining per-node revocation and
least privilege. The machine still needs to be a member of the organization's
Tailscale network; Microsoft 365 identity does not provide the private network
path. The gateway uses a supported JWT/JWKS validator, explicit
tenant/audience and role/group configuration, and key-rotation-aware key
lookup. The one-time pairing route remains available only while the
Entra-required flag is false; production should enable the flag after the
tenant app registration and end-to-end validation are complete. Do not accept
an arbitrary ID token, validate only its email claim, or use a shared Microsoft
365 password on nodes.

The reference desktop client is the **PROVIS Inference Node** at
`manuals_rag/scripts/run_inference_node_client.py`. It provides the Microsoft
365 sign-in action, explicit **Enable inference**, **Disable inference**, and
**Disconnect safely** controls, plus a bounded activity monitor that records
operation, model, status, duration, and sanitized error type without prompts or
credentials. Disabling inference is carried in the signed heartbeat and is
enforced by scheduler admission; safe disconnect removes the node from routing
while preserving enrollment for a later reconnect. The agent should advertise
and bind on the node's Tailscale address; it must not expose its port on a
public interface.

The client sources can be distributed as a dependency-free bundle:

```bash
python3 manuals_rag/scripts/build_inference_node_bundle.py \
  --output /tmp/provis-inference-node.zip
```

Managed gateway configuration is loaded by contract, not from the current
working directory: a source launcher may name an administrator-controlled
`PROVIS_NODE_BOOTSTRAP_CONFIG`; the portable bundle checks its own root beside
`src/`; an installed executable checks a sidecar beside the executable; and a
PyInstaller one-file executable reads the same nonsecret JSON from its bundled
data directory. A managed build rejects an empty or non-HTTPS gateway URL.
Only the gateway URL is packaged. Operator/application tokens, node secrets,
and Microsoft access tokens are never packaged.

On Windows, `build-node-exe.ps1` in that bundle creates a no-console
`PROVISInferenceNode.exe` using a build-only PyInstaller install. The
bundle contains only the standard-library node client and does not include the
repository, application credentials, Docker access, or Tailscale keys.

### Compatibility client

The existing common Ollama client can opt into the gateway without changing
call sites. Set `INFERENCE_GATEWAY_URL` and a JSON model catalog in the
process environment. The catalog is intentionally explicit so placement does
not guess VRAM from a model name:

```json
{
  "qwen3.5:9b": {"size_bytes": 10000000000, "min_vram_mb": 10000, "context_vram_mb_per_1k": 120, "capabilities": ["chat"]},
  "qwen3-embedding:0.6b": {"size_bytes": 700000000, "min_vram_mb": 1500, "capabilities": ["embeddings"]}
}
```

`INFERENCE_APPLICATION` and `INFERENCE_WORKLOAD` are included in every
request. Chat and text-generation requests also include `num_ctx` and the
catalog's `context_vram_mb_per_1k`; the scheduler admits against base model
VRAM plus the estimated context working set. Chat, text-generation, and
embedding calls use the same gateway; embeddings are mapped to Ollama's
`/api/embed` endpoint and generation requests to `/api/generate`. The
evaluation question generator and operator model-prompt endpoint therefore
participate in the same admission, VRAM, model-pull, and node-failover policy
when the gateway is enabled. If the gateway URL is unset, the current direct-
Ollama behavior remains unchanged.
When `INFERENCE_GATEWAY_FALLBACK_ENABLED=true`, only transport failures and
gateway 502/503/504 responses fall back to the configured local Ollama URL;
401/403, catalog, validation, and model errors remain fail-closed. This keeps
the gateway as the normal policy boundary without masking configuration or
authorization defects.

## Model lifecycle

1. Scheduler checks residency and capacity.
2. If a model is absent, the request must explicitly allow a pull.
3. The scheduler may propose idle eviction only for models with no active
   requests and only when the request explicitly allows eviction.
4. The node agent performs the bounded Ollama pull or unload operation.
5. The node reports a fresh snapshot before the request is admitted.

The gateway also exposes an explicit idle-eviction maintenance operation. It
only plans models older than `INFERENCE_IDLE_MODEL_TTL_SECONDS` on fresh,
proxy-only nodes with request accounting, excludes active or scheduler-reserved
models, and requires an approval flag before issuing signed eviction commands.
It is not an autonomous background deletion loop.

No stale heartbeat is eligible for inference. No telemetry-only decision is
allowed to evict an active model.

### Controlled Ollama provisioning

Ollama installation is a separate maintenance capability and must not be
implemented as `curl | sh`, arbitrary package execution, or an installer
running with the node agent's privileges. The node must use a pinned artifact
manifest rather than silently installing whatever is current. The repository
now contains the pinned staging primitive
`manuals_rag_inference.provisioning.stage_artifact` and an explicit
`activate_staged` gate. The latter requires maintenance approval, zero active
requests, an injected fixed service controller, and a successful local health
check before activation; neither operation is exposed as a remote arbitrary
install command.

The intended design is:

1. The droplet holds an administrator-approved manifest containing the exact
   Ollama version, official artifact URL, SHA-256 digest, platform, and rollback
   version.
2. A root-owned, locally installed provisioning helper on each node accepts
   only that manifest shape and fixed arguments. The unprivileged node agent
   cannot execute arbitrary commands or choose a URL.
3. A new signed `install_ollama` command is accepted only when provisioning is
   enabled for that node, no active inference requests are running, and an
   explicit maintenance approval is present.
4. The helper downloads to a temporary directory, verifies the digest, keeps
   the prior binary/service definition, installs atomically, health-checks the
   local Ollama API, and rolls back on failure.
5. The node reports the installed Ollama version and provisioning result in its
   next heartbeat. Model pulls remain a separate command.

Production activation remains intentionally disabled by the current node
agent. The primitives have been unit-tested for digest mismatch cleanup,
archive symlink rejection, version preservation, explicit approval, active
request blocking, atomic switching, and rollback after a failed health check.
Deployment still requires an OS/platform decision, a root-helper installation
process, an approved artifact manifest source, and a maintenance/rollback
procedure.

Eviction is likewise disabled unless a node runs in proxy-only mode
(`INFERENCE_NODE_PROXY_ONLY=true`). In that mode all inference passes through
the node agent, which can account for active requests. A node that still
permits direct Ollama callers reports that eviction accounting is unavailable
and will not be selected for eviction.

## Migration plan

1. Deploy the gateway on the always-on droplet with Tailscale-only reachability;
   do not run the control plane on a GPU workstation.
2. Run node agents in observe-only mode and compare snapshots with the current
   single endpoint at `host.docker.internal:11437`.
3. Add scheduler health and placement metrics without changing callers.
4. Route evaluation workloads first, with a fixed fallback to the current
   endpoint.
5. Verify workstation loss: one node disappears after the heartbeat TTL while
   other nodes remain routable through the droplet.
6. Verify failover, stale-node rejection, VRAM admission, model-pull approval,
   and no-active-request eviction.
7. Add and test the approved, pinned Ollama provisioning helper separately from
   inference routing.
8. Migrate RAG API/worker model clients through a compatibility adapter.
9. Migrate OpenClaw only after application identity, rate limits, and rollback
   are verified.

The scheduler and node-agent package is currently isolated from the running
compose stack. Deployment requires explicit service ownership, a Tailscale
grant policy, per-node secrets, and a rollback decision.

Before enabling the gateway on the droplet, run the non-mutating production
preflight with the deployment environment loaded:

```bash
PYTHONPATH=manuals_rag/packages/inference_fabric/src \
  python3 manuals_rag/scripts/check_inference_fabric_config.py
```

The preflight never prints token or secret values. It fails until persistent
enrollment state, a strong gateway token, a model catalog, and
`INFERENCE_GATEWAY_REQUIRE_NODE_AGENTS=true` are present.

On each GPU node, run the corresponding non-mutating node preflight before
installing or enabling the agent service:

```bash
PYTHONPATH=manuals_rag/packages/inference_fabric/src \
  python3 manuals_rag/scripts/check_inference_node_config.py
```
