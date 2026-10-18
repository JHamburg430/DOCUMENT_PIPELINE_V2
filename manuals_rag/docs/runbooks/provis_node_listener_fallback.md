# PROVIS inference-node listener fallback

Status: bounded listener fallback is enabled by default in app builds containing
the automatic-fallback change. Existing signed releases retain their original
behavior until they are updated. A host can explicitly opt out with
`INFERENCE_NODE_ALLOW_PORT_FALLBACK=false`.

## Behavior

- The node tries its configured listener port first, then automatically tries
  the next seven adjacent TCP ports when needed. With the current default of
  8765, the bounded range is 8765-8772. Set
  `INFERENCE_NODE_ALLOW_PORT_FALLBACK=false` to disable this behavior on a host.
- The local-only OpenAI broker port is excluded from node-listener selection so
  a saved port conflict cannot prevent the signed gateway listener from starting.
- It does not inspect, stop, or signal a process that owns a port. If every
  candidate is unavailable, startup fails without publishing a healthy
  heartbeat.
- After a successful alternate bind, the node sends its actual advertised URL
  to `/v1/nodes/agent-endpoint`, signed with its existing per-node HMAC secret,
  including a fresh timestamp and nonce.
- The gateway accepts that route only when
  `INFERENCE_NODE_ALLOW_ENDPOINT_REFRESH=true`. It validates the node HMAC,
  timestamp, one-use persisted nonce, enrolled hostname, scheme, and original
  eight-port window. It persists the URL before changing its in-memory routing
  map. The enrollment's initial hostname and base port stay fixed.
- If the gateway does not accept the refresh, the node closes its newly-bound
  listener and startup fails. Existing Ollama endpoint and model routing are
  unchanged.

## Required deployment preflight

1. Review the isolated branch and build a signed release using the existing
   publisher pipeline. Never use a signing private key from the repository.
2. Confirm on each approved PC that ports 8765–8772 are reserved for this node
   listener, and that no existing companion/service process needs to be stopped.
   Add only the narrowly scoped Tailscale-interface firewall permissions needed
   for that PC's actual selected port; do not expose the range publicly.
3. Back up the gateway enrollment-state file using the operator's protected
   procedure. Do not display or export its contents; it contains node secrets.
4. Deploy the reviewed code to DigitalOcean's `inference-fabric-gateway.service`
   with endpoint refresh still disabled. Preflight, then restart only this
   gateway service. OpenClaw services do not need a restart.
5. After verifying gateway health and rollback readiness, enable
   `INFERENCE_NODE_ALLOW_ENDPOINT_REFRESH=true` on the gateway and restart only
   `inference-fabric-gateway.service` to load the flag.
6. For one explicitly approved pilot node only, install the signed node release
   containing automatic fallback and restart that node app during an agreed
   maintenance window. Confirm its
   signed update receipt, actual listener port, gateway persisted URL, fresh
   heartbeat, authenticated inference, and monitor telemetry.
7. Repeat per approved PC after the pilot passes. Do not expand the existing
   application worker allowlists as part of this listener change.

## Affected systems and rollback

Potentially affected systems are the DigitalOcean inference gateway and each
approved node host (Johns_Dell_V2, ProVisWork, john-System-Product-Name/Ubuntu,
or MSI only after their current node-app installation is verified). The exact
employee/user assignment and installed app version remain to be checked at
action time. A node-app restart briefly interrupts inference on that node; if
the selected fallback port cannot be reached through the approved tailnet
firewall, that node will remain unavailable until its prior listener is restored.

Rollback: set `INFERENCE_NODE_ALLOW_PORT_FALLBACK=false` on the affected PC and
return the node to its previous signed release; restore its prior enrolled URL
using the protected gateway-state backup/operator procedure; then restart only
the node app and verify a fresh heartbeat. If gateway rollback is needed,
disable endpoint refresh and restore the previous reviewed gateway release and
configuration, then restart only `inference-fabric-gateway.service`. Do not
touch OpenClaw services or revoke/reissue node credentials as a shortcut.

This runbook is a proposed rollout sequence. All firewall, gateway deployment,
feature activation, and employee-PC restart actions remain subject to explicit
action-time approval.
